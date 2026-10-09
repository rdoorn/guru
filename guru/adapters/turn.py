"""Shared, provider-agnostic tool-calling turn loop.

Every adapter's turn is the same skeleton — ask the model, and while it keeps
requesting tools, run them and ask again — differing only in how a provider is
called and how tool results are threaded back into its native history. This
module owns the shared skeleton so all adapters get the same behaviour:

* cancel checks (between rounds, and mid-stream where the adapter supports it),
* a reply without a tool call ends the turn: its text is the answer (the
  lead's reply to the user, a worker's report to the lead). Nothing is
  forced, so a thinking model keeps its reasoning on every round,
* the stall monitor instead of a round cap: a round makes progress when a
  file changed or a tool returned something not seen before in the turn
  (timings, clock times and addresses ignored). ``STALL_ROUNDS`` rounds
  without progress earn one warning on the next tool result;
  ``STALL_GRACE`` more end the turn — a worker's on a handoff built in
  code (files read and changed, its last text) with ``session.stalled``
  set, so the lead decides what happens next,
* duplicate-call suppression (same name and arguments as an earlier call
  in the turn; a call that failed may be retried),
* a reply cut at the output limit: its tool calls are not run (their
  arguments may be truncated),
* verify before reporting: an answer from an agent that changed files and
  ran no tests (or, when enabled, no lint) since is sent back once,
* the answer check (:mod:`guru.domain.claims`) on the lead's answer to a
  request it worked on, and
* final-answer rendering: the answer lands in ``session.messages`` as the
  assistant's text.

Each adapter supplies three closures over its per-turn state:

``step()``
    Perform one provider round. Return ``(text, tool_calls)`` where
    ``tool_calls`` is a list of ``(name, args, ref)`` (``ref`` is an opaque
    provider handle passed back to ``run_tools``). Append the assistant message
    to both ``session.messages`` and any provider-native history, and update
    token accounting. Return ``None`` to stop the loop — set
    ``session.cancel_requested`` first for a cancel, or print an error and
    leave it False for a failure.

``run_tools(pending)``
    Execute a round's tools. ``pending`` is an ordered list of
    ``(name, args, ref, duplicate)``; run the non-duplicates, emit a reused-
    result notice for duplicates, and thread every result into both
    ``session.messages`` and the provider-native history.

``add_user(text)``
    Append a user turn (a send-back) to both histories.
"""
import hashlib
import json
import re
import time

from rich.markdown import Markdown

from guru import config, session, ui
from guru.domain import (claims, conversation, decisions, ledger, tools,
                         usage)

# The stall monitor (no round cap): STALL_ROUNDS rounds without progress
# earn one warning; STALL_GRACE more end the turn.
STALL_ROUNDS = 20
STALL_GRACE = 5
STALL_TEXT = (
    '[guru] {n} rounds without progress: nothing changed and every result'
    ' repeated what you already saw. Change approach: make the change and'
    ' run the tests, or answer now with what you found and what blocks'
    ' you. The turn ends in {grace} rounds without progress.')
_STALLED_TEXT = '(guru stopped the turn after {n} rounds without progress.)'
# The handoff of a stalled worker, built in code so the lead gets what it
# learned without another paid call.
_HANDOFF_PATHS = 12

# Verify before reporting: VERIFY_TOOLS count as testing; the project's
# ``lint`` tool (when enabled) is required too.
VERIFY_TOOLS = frozenset(('run_tests', 'sandbox_run', 'sandbox_python'))
LINT_TOOL = 'lint'
VERIFY_REFUSAL = conversation.VERIFY_REFUSAL


def verify_refusal(missing: frozenset) -> str:
    """The send-back for ``missing`` checks (``tests``, ``lint``); always
    starts with ``VERIFY_REFUSAL``."""
    what = ' or '.join(n for n in ('tests', 'lint') if n in missing)
    steps = []
    if 'tests' in missing:
        steps.append('run_tests on what you changed (the targeted tests'
                     ' first)')
    if 'lint' in missing:
        steps.append('lint on the files you changed')
    return (f"{VERIFY_REFUSAL}{what} since. Run {' and '.join(steps)},"
            ' fix what they report, then answer again.')


# A round whose reply hit the provider's output limit: its tool calls'
# arguments may be cut off, so they are not run.
OUTPUT_CUT_REFUSAL = (
    'Not run: your reply hit the output limit and this call\'s arguments'
    ' were cut off. Write large content in parts: write_file a first part,'
    ' then add the rest with edit_file or apply_patch, one call per round.')
# Results that mean the call did not do its job: an identical retry is
# allowed (not answered as a duplicate), and they are never progress.
_FAILED_PREFIXES = ('Invalid arguments', 'Tool error:', 'Refused',
                    'Not run:', 'Unknown tool:', 'Already called')

# The tool result an adapter returns for a call the loop marked duplicate
# (``run_tools`` gets ``duplicate=True``); the call itself does not run.
DUPLICATE_RESULT = ('Already called {name} with these arguments. Use the'
                    ' previous result.')


def tool_result(name: str, args: dict, duplicate: bool,
                execute=None) -> str:
    """One tool call's result, as every adapter threads it: the refusal of
    a call cut at the output limit, the duplicate notice (neither runs) or
    the tool's output, plus the round's note on the first result of the
    round (``session.round_note``, consumed here). ``execute`` defaults to
    ``tools.execute_tool`` (tests pass a stub)."""
    if session.output_cut:
        content = OUTPUT_CUT_REFUSAL
    elif duplicate:
        ui.console.print(
            f"[yellow]\\[SKIP][/yellow] duplicate: {name}({args})")
        content = DUPLICATE_RESULT.format(name=name)
    else:
        content = (execute or tools.execute_tool)(name, args)
    note = session.round_note
    if note:
        session.round_note = ''
        content = f'{content}\n\n{note}'
    return content


# Historical: the act and delegation nudges' texts. The loop no longer
# sends them; the names stay for the eval runner's counter and for reading
# old histories.
_NUDGE_TEXT = conversation.NUDGE_TEXT

# The user's request in a history: the most recent user message that is
# neither a nudge nor a mailbox delivery, capped
# (:func:`guru.domain.conversation.request_in`).
request_in = conversation.request_in


def turn_request() -> str:
    """The user's request for this turn (:func:`request_in` over the bound
    session's messages): on a mailbox turn the human request the
    delivery answers, not the delivery. The sandbox gate hands it to the
    reviewer."""
    return request_in(session.messages)


def _is_worker() -> bool:
    """A sub-agent executing a delegated task (its stall ends on a
    handoff; the lead's turn is the user's conversation)."""
    return bool(session.task_id)


def reasoning_effort() -> str:
    """The thinking effort for the bound session's rounds: the lead's
    (``config.THINKING_LEAD``) or a worker's (``config.THINKING_WORKER``);
    ``''`` turns thinking off. Adapters translate it to their provider."""
    return config.THINKING_WORKER if _is_worker() else config.THINKING_LEAD


def _close_turn(start: float, in0: int, out0: int, cost0: float,
                unpriced0: int, struggle0: dict, tools_used: list) -> None:
    """Write the TurnRecord for any agent not executing a task.

    A sub-agent running a spawned task is accounted for by its task row.
    Tokens, cost and struggle counters are this turn's deltas over the
    session values snapshotted at turn start; cost is None only when a call
    made during *this* turn could not be priced.
    """
    if session.task_id:
        return
    exact = session.unpriced_calls == unpriced0
    cost = session.cost_usd - cost0 if exact else None
    ledger.record_turn(ledger.TurnRecord(
        turn_id=session.turn_id, request=turn_request(),
        model=session.model, seconds=time.monotonic() - start,
        tasks_spawned=tools_used.count('spawn'),
        tools_used=tools_used,
        tokens_in=session.session_in - in0,
        tokens_out=session.session_out - out0,
        cost_usd=cost,
        controller_executed=False,
        agent=session.agent_id,
        adapter=getattr(session.adapter, 'name', ''),
        struggle=ledger.struggle_delta(struggle0, session.struggle)))


def _render_answer(content: str) -> None:
    ui.console.print("\n[bold green]answer>[/bold green]")
    ui.console.print(Markdown(content))
    ui.console.print()


def run_loop(*, step, run_tools, add_user, nudge: bool = True) -> None:
    """Drive one user turn to a final answer using the adapter's closures.

    Owns the shared control flow; the adapter owns the provider calls and
    history threading. See the module docstring for the closure contracts.
    Exactly one TurnRecord is written per call, on every exit path.
    ``nudge`` is accepted for the adapters' signature and unused (the
    delegation nudge is gone: the lead decides when to delegate).
    """
    session.cancel_requested = False
    session.last_error = ''          # this turn's provider failure, if any
    session.turn_waiting = False     # set by join/check (orchestrator)
    session.check_polls = 0
    session.stalled = False
    session.round_note = ''
    if not session.task_id:
        # A sub-agent executing a task keeps the turn_id it inherited.
        session.turn_id = ledger.new_turn_id()
        if not conversation.mailbox_turn(session.messages):
            # A new user request: its topic (a mailbox delivery continues
            # the request it answers, and keeps that topic).
            usage.begin_topic(turn_request())
    start = time.monotonic()
    in0, out0 = session.session_in, session.session_out
    cost0, unpriced0 = session.cost_usd, session.unpriced_calls
    struggle0 = dict(session.struggle)
    tools_used: list = []
    try:
        _drive(step, run_tools, add_user, tools_used)
    finally:
        _close_turn(start, in0, out0, cost0, unpriced0, struggle0,
                    tools_used)


def _drive(step, run_tools, add_user, tools_used: list) -> str:
    """The round loop proper; every requested tool lands in
    ``tools_used``. Returns the final answer text (``''`` on cancel,
    error or a turn that waits for sub-agents)."""
    called: set = set()
    last_text = ''
    monitor = Monitor()
    panel_asked = False
    while True:
        if session.cancel_requested:
            ui.console.print("[yellow]* cancelled[/yellow]")
            return ''
        ui.note_thinking()
        session.output_cut = False      # the adapter sets it for this reply
        result = step()
        if result is None:
            # None = stop: a cancel (flagged) or an error (step printed it).
            if session.cancel_requested:
                ui.console.print("[yellow]* cancelled[/yellow]")
                return ''
            if _is_worker() and (monitor.changed or monitor.read):
                # A provider error (the context full, an outage) after
                # real work: hand back what was done, as a stall does,
                # so the lead can carry on from it.
                session.stalled = True
                content = monitor.handoff(last_text, stalled=False)
                _settle(content)
                return content
            return ''
        ui.status_draw()
        text, calls = (result[0] or '').strip(), result[1]
        if text:
            last_text = text

        if calls:
            pending = []
            for name, args, ref in calls:
                tools_used.append(name)
                key = (name, _args_key(args))
                duplicate = key in called
                called.add(key)
                pending.append((name, args, ref, duplicate))
            session.round_note = monitor.note()
            before = len(session.messages)
            run_tools(pending)
            session.round_note = ''
            _allow_retries(called, session.messages[before:])
            if monitor.saw(session.messages[before:]):
                # Files changed: earlier results (a test run, a read) may
                # be stale, so the same call runs again.
                called.clear()
            if session.turn_waiting:
                # A join opened a barrier (or check kept polling running
                # sub-agents): the mailbox resumes this agent with the
                # results.
                ui.console.print("[dim]\\[waiting for sub-agents][/dim]")
                return ''
            if not monitor.stalled():
                continue
            ledger.bump('stalls')
            ui.console.print(
                f"[dim yellow]\\[STALL][/dim yellow] {monitor.quiet} rounds"
                " without progress — ending the turn")
            session.stalled = True
            if _is_worker():
                content = monitor.handoff(last_text)
            else:
                content = '\n\n'.join(filter(None, (
                    last_text, _STALLED_TEXT.format(n=monitor.quiet))))
            _settle(content)
            _render_answer(content)
            return content

        if not text:
            # An empty reply: nothing to deliver and nothing to run.
            ledger.bump('protocol_violation')
            content = last_text or '(no answer produced)'
        else:
            sent_back = _send_back(monitor, text)
            if sent_back:
                add_user(sent_back)
                continue
            content = text
        if (session.can_spawn and not panel_asked
                and not conversation.mailbox_turn(session.messages)):
            # Shadowed for the judges' labels; changes nothing. A mailbox
            # delivery is the sub-agents' results, not a task to staff.
            panel_asked = True
            decisions.shadow('panel', decisions.panel_questions(
                turn_request()))
        _settle(content)
        _render_answer(content)
        return content


def _send_back(monitor: 'Monitor', answer: str) -> str:
    """The text that sends ``answer`` back for one more try, or ``''`` to
    deliver it: the verify check (once per turn) for an agent that changed
    files and did not test or lint them since, then the answer check for
    the lead (once per request)."""
    missing = monitor.verify_missing()
    if missing:
        monitor.verify_asked = True
        ledger.bump('verify_sendbacks')
        ui.console.print("[dim yellow]\\[VERIFY][/dim yellow] changed files"
                         " untested — sent back")
        return verify_refusal(missing)
    if _is_worker():
        return ''
    sent_back = claims.check_answer(session.messages, answer)
    if sent_back is None:
        return ''
    ui.console.print("[dim yellow]\\[CHECK][/dim yellow] answer vs work:"
                     " sent back")
    return sent_back


# The paths an ``apply_patch`` / ``sandbox_submit`` result reports as
# written ("<path>: N hunk(s) applied"), and the ``gate.stat_text`` rows
# ("<path> | +3 -1") an ``apply_work`` result lists, one per line.
_APPLIED_RE = re.compile(r'^(.+?): \d+ hunk', re.MULTILINE)
_STAT_RE = re.compile(r'^(\S.*?) \| \+\d+', re.MULTILINE)


def changed_paths(name: str, args: object, content: str) -> list[str]:
    """The files a finished tool call changed, read off its result: only
    a call that succeeded counts (a refused, invalid or failed write
    changed nothing). Empty for every other tool."""
    path = args.get('path') if isinstance(args, dict) else None
    if name in ('write_file', 'edit_file'):
        return [str(path)] if path and '(sha:' in content else []
    if name == 'delete_file':
        return [str(path)] if path and content.startswith('Deleted ') else []
    if name in ('apply_patch', 'sandbox_submit'):
        if content.startswith('Applied patch'):
            return _APPLIED_RE.findall(content) or [name]
    if name == 'apply_work' and content.startswith('Applied patch'):
        return [p.strip() for p in _STAT_RE.findall(content)] or [name]
    return []


# Sandbox scripts are how an agent edits its copy: a new script is new
# work even when its output (often just "exit 0") repeats.
_SCRIPT_TOOLS = frozenset(('sandbox_python',))
# What changes on every call without being news: a timing or other
# decimal ("in 3.21s"), a clock time, a hex address.
_NOISE_RE = re.compile(r'\d+\.\d+|\d{1,2}:\d{2}(?::\d{2})?|0x[0-9a-fA-F]+')


def _fingerprint(name: str, content: str) -> str:
    """A tool result's identity for the stall monitor: its tool name and
    content with the noise (``_NOISE_RE``) blanked."""
    text = _NOISE_RE.sub('#', content)
    return hashlib.sha1(f'{name}\0{text}'.encode()).hexdigest()


class Monitor:
    """What an agent's rounds produced: the results it has seen (by
    fingerprint), the rounds since the last progress, the paths it read
    and changed, and whether its changes are tested and linted."""

    def __init__(self) -> None:
        self.seen: set = set()
        self.quiet = 0                  # rounds since the last progress
        self.warned = False
        self.last_note = ''             # stripped before fingerprinting
        self.read: list[str] = []
        self.changed: list[str] = []
        self.untested = False           # changed files since the last test
        self.unlinted = False           # ... since the last lint
        self.verify_asked = False

    def saw(self, results: list) -> bool:
        """Take in a round's tool messages (after they ran); True when
        they changed files (or ran a sandbox script, which may have)."""
        progress = False
        changed = False
        for m in results:
            if not isinstance(m, dict) or m.get('role') != 'tool':
                continue
            name = str(m.get('tool_name', ''))
            args = m.get('tool_args')
            content = str(m.get('content', ''))
            failed = content.startswith(_FAILED_PREFIXES)
            for p in changed_paths(name, args, content):
                progress = changed = True
                self.untested = self.unlinted = True
                if p not in self.changed:
                    self.changed.append(p)
            if name in _SCRIPT_TOOLS and not failed:
                changed = True
            if name in VERIFY_TOOLS and not failed:
                self.untested = False
            if name == LINT_TOOL and not failed:
                self.unlinted = False
            path = args.get('path') if isinstance(args, dict) else None
            if (name in config.DELEGATION_READ_TOOLS and path
                    and str(path) not in self.read):
                self.read.append(str(path))
            if failed:
                continue
            if self.last_note:
                content = content.replace(f'\n\n{self.last_note}', '')
            if name in _SCRIPT_TOOLS:
                content = f'{_args_key(args)}\0{content}'
            key = _fingerprint(name, content)
            if key not in self.seen:
                self.seen.add(key)
                progress = True
        if progress:
            self.quiet = 0
            self.warned = False
        else:
            self.quiet += 1
        return changed

    def note(self) -> str:
        """The warning for the round about to run, once per quiet streak
        (bumps ``stall_warnings``); ``''`` otherwise."""
        if self.warned or self.quiet < STALL_ROUNDS:
            return ''
        self.warned = True
        ledger.bump('stall_warnings')
        ui.console.print("[dim yellow]\\[STALL][/dim yellow] no progress —"
                         " warning sent")
        self.last_note = STALL_TEXT.format(n=self.quiet, grace=STALL_GRACE)
        return self.last_note

    def stalled(self) -> bool:
        """Whether the turn ends: warned, and the grace rounds went by
        without progress too."""
        return self.warned and self.quiet >= STALL_ROUNDS + STALL_GRACE

    def verify_missing(self) -> frozenset:
        """The checks an answer is sent back for: tests and (when the
        ``lint`` tool is enabled) lint not run since files changed — once
        per turn."""
        if self.verify_asked:
            return frozenset()
        missing = set()
        if self.untested:
            missing.add('tests')
        if self.unlinted and tools.is_enabled(LINT_TOOL):
            missing.add('lint')
        return frozenset(missing)

    def handoff(self, last_text: str, stalled: bool = True) -> str:
        """A stopped worker's report: what it read and changed, then its
        last text — the lead decides from here. ``stalled`` False: the
        provider failed mid-task (the context full, an outage)."""
        def paths(items: list[str]) -> str:
            shown = ', '.join(items[:_HANDOFF_PATHS])
            more = len(items) - _HANDOFF_PATHS
            return shown + (f' (+{more} more)' if more > 0 else '')
        why = (f'stalled: guru ended this task after {self.quiet} rounds'
               ' without progress' if stalled else
               f'stopped: the provider failed ({session.last_error or "?"})')
        lines = [f'({why}; the handoff below is what it got to.)',
                 f'Files changed: {paths(self.changed) or "none"}',
                 f'Files read: {paths(self.read) or "none"}']
        if last_text:
            lines.append(f'Last note: {last_text}')
        return '\n'.join(lines)


def _allow_retries(called: set, results: list) -> None:
    """Forget the duplicate key of every call in ``results`` that failed
    (``_FAILED_PREFIXES``), so an identical retry runs instead of being
    answered "already called"."""
    for m in results:
        if (isinstance(m, dict) and m.get('role') == 'tool'
                and str(m.get('content', '')).startswith(_FAILED_PREFIXES)
                and not str(m.get('content', '')).startswith(
                    'Already called')):
            called.discard((m.get('tool_name'), _args_key(m.get('tool_args'))))


def _args_key(args: object) -> str:
    """A hashable identity for a call's arguments."""
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(args)


def _settle(answer: str) -> None:
    """Land ``answer`` in ``session.messages`` as the assistant's text:
    the last message's text becomes the answer when it is the assistant's
    own text reply, else the answer is appended after the round's tool
    results."""
    msgs = session.messages
    if msgs and conversation.msg_role(msgs[-1]) == 'assistant' \
            and not conversation._tool_calls_of(msgs[-1]):
        last = msgs[-1]
        if isinstance(last, dict):
            last['content'] = answer
        else:
            last.content = answer
        return
    msgs.append({'role': 'assistant', 'content': answer})
