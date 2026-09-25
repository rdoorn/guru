"""Shared, provider-agnostic tool-calling turn loop.

Every adapter's turn is the same skeleton — ask the model, and while it keeps
requesting tools, run them and ask again — differing only in how a provider is
called and how tool results are threaded back into its native history. This
module owns the shared skeleton so all adapters get the same behaviour:

* cancel checks (between rounds, and mid-stream where the adapter supports it),
* the act-nudge that pokes a model which announced an action but ran no tool,
* the delegation nudges (end of turn after a broad read-heavy answer, and
  the mid-turn over-read guard after ``config.OVER_READ_LIMIT`` distinct
  files without a spawn),
* duplicate-call suppression, and
* final-answer rendering.

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
    Append a user turn (the nudge) to both histories.
"""
import os
import re
import time

from rich.markdown import Markdown

from guru import config, session, ui
from guru.domain import decisions, ledger, tools

# A controller answer longer than this with no spawn in the turn counts as
# the controller doing the work itself (design doc §5: measured, not
# punished).
_CONTROLLER_ANSWER_CHARS = 600

# A weak model sometimes ends a turn by announcing an action ("Let me read the
# files…") without calling a tool; without a nudge that would be taken as the
# final answer. looks_like_preamble catches that stall so the loop can poke it.
_NUDGE_CAP = 2
_PREAMBLE_RE = re.compile(
    r"\b(let me|i'?ll|i will|let'?s|i'?m going to|i am going to|going to|"
    r"start by|next[,]? i|first[,]? i)\b", re.IGNORECASE)

_NUDGE_TEXT = (
    "Do not describe what you will do — do it now. Call the tool you need in"
    " this reply (use search_tools first if it is not active). If you are"
    " genuinely finished, give the final answer."
)

_DELEGATION_TEXT = (
    "You inspected several files yourself. This task spans multiple concerns —"
    " decompose it now instead of answering directly: spawn parallel"
    " sub-agents, one per domain, then join and synthesise. For a review,"
    " spawn(task='review the code for correctness, readability, tests',"
    " role='developer', skill='code-review') AND spawn(task='review the code"
    " for injection, authz, secrets, path traversal, vulnerable deps',"
    " role='security-engineer', skill='code-review'), then join both and give"
    " one consolidated report. Add architect/SRE sub-agents if design or"
    " reliability matter."
)


# A request shaped like a single edit: an edit verb and at most one
# file-like token. Such a task is never a review panel (triage
# 1f4f8262a80a: a one-file fix was nudged into a two-agent panel).
_EDIT_VERB_RE = re.compile(
    r"\b(fix|edit|rename|change|update|patch)\b", re.IGNORECASE)
_FILE_TOKEN_RE = re.compile(
    r"\S+\.(?:py|md|toml|txt|js|ts|json|yaml|yml)\b", re.IGNORECASE)


def _single_target_request(request: str) -> bool:
    """True for an edit-shaped request naming at most one file."""
    if not _EDIT_VERB_RE.search(request):
        return False
    return len(set(_FILE_TOKEN_RE.findall(request))) <= 1


def _is_nudge(text: str) -> bool:
    """True for a user message the loop itself injected (act, delegation
    or over-read nudge)."""
    return (text in (_NUDGE_TEXT, _DELEGATION_TEXT)
            or text.endswith(_DELEGATION_TEXT))


def _over_read_text(n: int) -> str:
    """The over-read nudge: how many files were read, then the delegation
    text (``_is_nudge`` recognises it by its suffix)."""
    return f'You have read {n} files without delegating. ' + _DELEGATION_TEXT


def _turn_start() -> int:
    """Index in ``session.messages`` of this turn's request (the last user
    message that is not a nudge); 0 when there is none."""
    for i in range(len(session.messages) - 1, -1, -1):
        m = session.messages[i]
        if not isinstance(m, dict) or m.get('role') != 'user':
            continue
        text = (m.get('content') or '').strip()
        if text and not _is_nudge(text):
            return i
    return 0


def _distinct_reads(start: int = 0) -> int:
    """Distinct paths read with the read tools (by the ``path`` argument
    recorded on tool messages) from ``session.messages[start:]`` on — the
    whole conversation by default, this turn with ``_turn_start()``; -1
    once a spawn ran in that span."""
    paths: set = set()
    for m in session.messages[start:]:
        if not isinstance(m, dict) or m.get('role') != 'tool':
            continue
        name = m.get('tool_name', '')
        if name == 'spawn':
            return -1                    # already delegated — leave it alone
        if name not in config.DELEGATION_READ_TOOLS:
            continue
        args = m.get('tool_args')
        path = args.get('path') if isinstance(args, dict) else None
        if path:
            paths.add(os.path.normpath(str(path)))
    return len(paths)


def _should_delegate() -> bool:
    """True when a delegation-capable main agent has read enough distinct
    files to make a domain panel worthwhile, the request is not a
    single-target edit, and it has not spawned a single sub-agent — the cue
    for the one-time delegation nudge. Never for a controller (it has no
    read tools and must delegate anyway) and disabled when
    DELEGATION_NUDGE_MIN_READS is 0."""
    if (session.controller or not session.can_spawn
            or config.DELEGATION_NUDGE_MIN_READS <= 0):
        return False
    reads = _distinct_reads()
    if reads < config.DELEGATION_NUDGE_MIN_READS:
        return False
    return not _single_target_request(_turn_request())


def _over_read() -> int:
    """How many distinct files a delegation-capable main agent has read
    this turn once that reaches ``config.OVER_READ_LIMIT`` without a spawn
    (the cue for the mid-turn over-read nudge); 0 otherwise. Never for a
    controller or a sub-agent; 0 disables the guard."""
    if (session.controller or not session.can_spawn
            or config.OVER_READ_LIMIT <= 0):
        return 0
    reads = _distinct_reads(_turn_start())
    return reads if reads >= config.OVER_READ_LIMIT else 0


def looks_like_preamble(content: str) -> bool:
    """True if text announces an action instead of answering — a short
    'Let me… / I'll…' preamble, or one trailing off into a promised list.
    Long substantive answers (the real result) do not match."""
    if len(content) > 600:
        return False
    if content.rstrip().endswith((':', '…', '...')):
        return True
    return bool(_PREAMBLE_RE.search(content))


def _turn_request() -> str:
    """The user's request for this turn: the most recent user message that
    is not one of the loop's own nudges."""
    for m in reversed(session.messages):
        if not isinstance(m, dict) or m.get('role') != 'user':
            continue
        text = (m.get('content') or '').strip()
        if text and not _is_nudge(text):
            return text
    return ''


# Public name: the sandbox gate hands the turn's request to the reviewer.
turn_request = _turn_request

_MAILBOX_PREFIXES = ('[joined results]', '[result from')


def _mailbox_turn() -> bool:
    """True when this turn was started by a mailbox delivery (a joined or
    single sub-agent result): the request text is the sub-agents' output,
    not a task from the user."""
    return _turn_request().startswith(_MAILBOX_PREFIXES)


def controller_executed(tools_used: list, answer: str) -> bool:
    """Did a controller do the work itself this turn?

    True only in controller mode, when a tool outside spawn/check/join/
    use_skill was attempted or the final answer exceeds
    ``_CONTROLLER_ANSWER_CHARS`` with no spawn in the turn. Turns driven by
    a mailbox delivery (a joined or single sub-agent result) are synthesis
    turns and never count.
    """
    if not session.controller:
        return False
    if _mailbox_turn():
        return False
    if any(name not in tools.CONTROLLER_TOOLS for name in tools_used):
        return True
    return (len(answer) > _CONTROLLER_ANSWER_CHARS
            and tools_used.count('spawn') == 0)


def _close_turn(start: float, in0: int, out0: int, cost0: float,
                unpriced0: int, struggle0: dict, tools_used: list,
                answer: str = '') -> None:
    """Write the TurnRecord for any agent not executing a task.

    A sub-agent running a spawned task is accounted for by its task row.
    Tokens, cost and struggle counters are this turn's deltas over the
    session values snapshotted at turn start; cost is None only when a call
    made during *this* turn could not be priced. ``answer`` is the final
    answer text (empty on cancel/error), for ``controller_executed``.
    """
    if session.task_id:
        return
    exact = session.unpriced_calls == unpriced0
    cost = session.cost_usd - cost0 if exact else None
    ledger.record_turn(ledger.TurnRecord(
        turn_id=session.turn_id, request=_turn_request(),
        model=session.model, seconds=time.monotonic() - start,
        tasks_spawned=tools_used.count('spawn'), tools_used=tools_used,
        tokens_in=session.session_in - in0,
        tokens_out=session.session_out - out0,
        cost_usd=cost,
        controller_executed=controller_executed(tools_used, answer),
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
    """
    session.cancel_requested = False
    session.last_error = ''          # this turn's provider failure, if any
    session.turn_waiting = False     # set by join/check (orchestrator)
    session.check_polls = 0
    if not session.task_id:
        # A sub-agent executing a task keeps the turn_id it inherited.
        session.turn_id = ledger.new_turn_id()
    start = time.monotonic()
    in0, out0 = session.session_in, session.session_out
    cost0, unpriced0 = session.cost_usd, session.unpriced_calls
    struggle0 = dict(session.struggle)
    tools_used: list = []
    answer = ''
    try:
        answer = _drive(step, run_tools, add_user, nudge, tools_used)
    finally:
        _close_turn(start, in0, out0, cost0, unpriced0, struggle0,
                    tools_used, answer)


def _drive(step, run_tools, add_user, nudge: bool, tools_used: list) -> str:
    """The round loop proper; every requested tool lands in ``tools_used``.
    Returns the final answer text (``''`` on cancel or error)."""
    called: set = set()
    nudged = 0
    delegation_nudged = False
    over_read_nudged = False
    panel_asked = False
    while True:
        if session.cancel_requested:
            ui.console.print("[yellow]* cancelled[/yellow]")
            return ''
        ui.note_thinking()
        result = step()
        if result is None:
            # None = stop: a cancel (flagged) or an error (step printed it).
            if session.cancel_requested:
                ui.console.print("[yellow]* cancelled[/yellow]")
            return ''
        ui.status_draw()
        text, tool_calls = result

        if not tool_calls:
            content = (text or '').strip()
            stalled = not content or looks_like_preamble(content)
            if content:
                # The stall judge sees the same candidate answer the
                # heuristic scored. Shadow: its verdict only lands in the
                # ledger. Active ([decisions.active] stall = true): its
                # verdict decides, the heuristic is the timeout fallback.
                stalled = bool(decisions.decide(
                    'stall', decisions.stall_question(content),
                    heuristic=stalled))
                if (session.can_spawn and not panel_asked
                        and not _mailbox_turn()):
                    # One boolean cannot stand in for three specialist
                    # questions, so the panel batch carries no heuristic.
                    # A mailbox delivery is the sub-agents' results, not a
                    # task to staff, so it is never judged (f1929d55c41a).
                    panel_asked = True
                    decisions.shadow(
                        'panel', decisions.panel_questions(_turn_request()))
            if nudge and stalled and nudged < _NUDGE_CAP:
                nudged += 1
                ledger.bump('stall_nudges')
                reason = ("empty response" if not content
                          else "announced an action but called no tool")
                ui.console.print(
                    f"[dim yellow]\\[NUDGE][/dim yellow] {reason}"
                    " — asking it to act"
                )
                add_user(_NUDGE_TEXT)
                continue
            if (nudge and not delegation_nudged and content
                    and _should_delegate()):
                delegation_nudged = True
                ledger.bump('delegation_nudges')
                ui.console.print(
                    "[dim yellow]\\[DELEGATE][/dim yellow] broad task, no"
                    " sub-agents — asking it to spawn a domain panel")
                add_user(_DELEGATION_TEXT)
                continue
            _render_answer(content)
            return content

        pending = []
        for name, args, ref in tool_calls:
            tools_used.append(name)
            key = (name, tuple(sorted(args.items())))
            duplicate = key in called
            if not duplicate:
                called.add(key)
            pending.append((name, args, ref, duplicate))
        run_tools(pending)
        if session.turn_waiting:
            # A join opened a barrier (or check kept polling running
            # sub-agents): stop here instead of another model round. No
            # answer is rendered; the mailbox resumes this agent with the
            # results.
            ui.console.print("[dim]\\[waiting for sub-agents][/dim]")
            return ''
        if nudge and not over_read_nudged:
            reads = _over_read()
            if reads:
                # Over-read guard (triage 2026-09-24): the model is reading
                # the codebase itself instead of delegating; tell it now,
                # once, rather than after it has read everything.
                over_read_nudged = True
                ledger.bump('over_read')
                ui.console.print(
                    f"[dim yellow]\\[DELEGATE][/dim yellow] {reads} files"
                    " read, no sub-agents — asking it to spawn a domain"
                    " panel")
                add_user(_over_read_text(reads))
                continue
