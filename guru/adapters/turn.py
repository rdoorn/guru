"""Shared, provider-agnostic tool-calling turn loop.

Every adapter's turn is the same skeleton — ask the model, and while it keeps
requesting tools, run them and ask again — differing only in how a provider is
called and how tool results are threaded back into its native history. This
module owns the shared skeleton so all adapters get the same behaviour:

* cancel checks (between rounds, and mid-stream where the adapter supports it),
* the **turn contract**: a round ends with a tool call. A worker's answer
  is a ``final_answer(text)`` call; a controller's every reply is one
  ``plan`` call (:mod:`guru.domain.plan`). Adapters that can force a tool
  call do (:func:`forced_tool` tells them what this round needs, their
  ``forces`` says whether they will); on such an adapter a text-only reply
  is a protocol violation: one deterministic re-prompt, then the text is
  the answer and the ``protocol_violation`` struggle counter is bumped. On
  an adapter that cannot force (Ollama) the text is the answer, and a
  controller's text is parsed for the plan object first.
* the controller plan: a rejected plan (the handler's re-ask) gets one more
  round, a second rejection ends the turn on a plain-text fallback with
  ``protocol_violation``; an ``answer`` plan is the reply (its text, else
  the round's own text; never rejected); a ``delegate`` plan ends the
  turn like a ``join`` (the mailbox resumes the agent); a second ``plan``
  in one round is refused unrun,
* the round cap (``_MAX_ROUNDS`` plan rounds for a controller,
  ``_MAX_TOOL_ROUNDS`` for everyone else): the turn ends on the last text
  with ``protocol_violation`` instead of paying for rounds without end,
* the delegation nudge (end of turn after a broad read-heavy answer) for a
  hands-on delegation-capable agent,
* duplicate-call suppression (same name and arguments as an earlier call
  in the turn; never for ``plan``/``final_answer``), and
* final-answer rendering: the answer lands in ``session.messages`` as the
  assistant's text (a lone ``final_answer``/``plan`` round is collapsed
  into it, so the next turn's history carries the text once).

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
    Append a user turn (a nudge or re-prompt) to both histories.
"""
import json
import os
import re
import time
from typing import Optional

from rich.markdown import Markdown

from guru import config, session, ui
from guru.adapters.base import FORCE_ANY, FORCE_PLAN
from guru.domain import conversation, decisions, ledger, plan, tools

# A controller answer longer than this with no delegation in the turn counts
# as the controller doing the work itself (design doc §5: measured, not
# punished).
_CONTROLLER_ANSWER_CHARS = 600

# One deterministic re-prompt per turn for a text-only reply where a tool
# call was forced (or an empty reply anywhere); one re-ask per turn for a
# rejected plan. Then the text is the answer.
_REPROMPT_CAP = 1
_REASK_CAP = 1

# Per-turn round caps: every provider round is a paid call and the loop
# below has no other exit while the model keeps calling tools. A
# controller's round is one ``plan`` call, so a controller that has not
# produced an accepted plan in _MAX_ROUNDS rounds is stuck (a refusal or
# an unavailable handler answered every time); a worker legitimately
# chains many tool rounds (read, edit, test, ...) so its cap is wider.
# At the cap the turn ends with ``protocol_violation`` on the last text the
# model wrote — for a controller ``plan.fallback_text`` over its last plan,
# so the user sees what was attempted.
_MAX_ROUNDS = 12
_MAX_TOOL_ROUNDS = 40
_CAPPED_TEXT = '(guru ended the turn after {n} rounds without a final answer.)'

# The tool result an adapter returns for a call the loop marked duplicate
# (``run_tools`` gets ``duplicate=True``); the call itself does not run.
DUPLICATE_RESULT = ('Already called {name} with these arguments. Use the'
                    ' previous result.')

# Calls that are never suppressed as duplicates: ``plan`` is the turn's
# protocol (the same plan again after a coverage re-ask is the accepted
# second try, and a re-issued plan must reach the handler to be judged)
# and ``final_answer`` ends the turn. Every other call with the same name
# and arguments as an earlier one in the turn is answered from the
# earlier result without running. A second ``plan`` in one round is
# refused the same way (unrun) so the round has one plan verdict.
_NEVER_DUPLICATE = frozenset(('plan', 'final_answer'))

_DELEGATION_TEXT = conversation.DELEGATION_TEXT
# Historical: the act nudge's text. The loop no longer sends it (the turn
# contract replaced the preamble heuristic); the name stays for the eval
# runner's counter and for reading old histories.
_NUDGE_TEXT = conversation.NUDGE_TEXT


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


# True for a user message the loop itself injected (the delegation nudge,
# a re-prompt or a plan re-ask).
_is_nudge = conversation.is_nudge


def _turn_start() -> int:
    """Index in ``session.messages`` of this turn's request (the last user
    message that is not a nudge); 0 when there is none."""
    return conversation.turn_start(session.messages)


def _distinct_reads(start: int = 0) -> int:
    """Distinct paths read with the read tools (by the ``path`` argument
    recorded on tool messages) from ``session.messages[start:]`` on (the
    whole conversation by default); -1 once a spawn ran in that span."""
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
    read tools and delegates through its plan) and disabled when
    DELEGATION_NUDGE_MIN_READS is 0."""
    if (session.controller or not session.can_spawn
            or config.DELEGATION_NUDGE_MIN_READS <= 0):
        return False
    reads = _distinct_reads()
    if reads < config.DELEGATION_NUDGE_MIN_READS:
        return False
    return not _single_target_request(_turn_request())


# The user's request in a history: the most recent user message that is
# neither a nudge nor a mailbox delivery, capped
# (:func:`guru.domain.conversation.request_in`).
request_in = conversation.request_in


def _turn_request() -> str:
    """The user's request for this turn (:func:`request_in` over the bound
    session's messages): on a mailbox turn the human request the
    delivery answers, not the delivery."""
    return request_in(session.messages)


# Public name: the sandbox gate hands the turn's request to the reviewer.
turn_request = _turn_request

_MAILBOX_PREFIXES = conversation.MAILBOX_PREFIXES


def _mailbox_turn() -> bool:
    """True when this turn was started by a mailbox delivery (a joined or
    single sub-agent result): the message that opened the turn (the last
    user message that is not a nudge) is the sub-agents' output, not a
    task from the user."""
    return conversation.mailbox_turn(session.messages)


# --- the turn contract -------------------------------------------------------

def forced_tool() -> str:
    """What this round must answer with: ``FORCE_PLAN`` (the ``plan``
    tool) for a controller, ``FORCE_ANY`` (any tool, ``final_answer`` to
    finish) for every other agent. Adapters read it per round and force
    when they can (``Adapter.forces``)."""
    return FORCE_PLAN if session.controller else FORCE_ANY


def _forcing(forced: str) -> bool:
    """Whether the bound session's adapter forces ``forced`` this round."""
    fn = getattr(session.adapter, 'forces', None)
    return callable(fn) and bool(fn(forced))


def controller_executed(tools_used: list, answer: str,
                        delegated: int = 0) -> bool:
    """Did a controller do the work itself this turn?

    True only in controller mode, when a tool outside ``plan`` was
    attempted or the final answer exceeds ``_CONTROLLER_ANSWER_CHARS``
    with nothing delegated in the turn (``delegated`` plan tasks, plus
    any ``spawn`` call). Turns driven by a mailbox delivery (a joined or
    single sub-agent result) are synthesis turns and never count.
    """
    if not session.controller:
        return False
    if _mailbox_turn():
        return False
    if any(name not in tools.CONTROLLER_TOOLS for name in tools_used):
        return True
    return (len(answer) > _CONTROLLER_ANSWER_CHARS
            and tools_used.count('spawn') + delegated == 0)


def _close_turn(start: float, in0: int, out0: int, cost0: float,
                unpriced0: int, struggle0: dict, tools_used: list,
                answer: str = '', delegated: int = 0) -> None:
    """Write the TurnRecord for any agent not executing a task.

    A sub-agent running a spawned task is accounted for by its task row.
    Tokens, cost and struggle counters are this turn's deltas over the
    session values snapshotted at turn start; cost is None only when a call
    made during *this* turn could not be priced. ``answer`` is the final
    answer text (empty on cancel/error), for ``controller_executed``;
    ``delegated`` the plan tasks guru spawned this turn.
    """
    if session.task_id:
        return
    exact = session.unpriced_calls == unpriced0
    cost = session.cost_usd - cost0 if exact else None
    ledger.record_turn(ledger.TurnRecord(
        turn_id=session.turn_id, request=_turn_request(),
        model=session.model, seconds=time.monotonic() - start,
        tasks_spawned=tools_used.count('spawn') + delegated,
        tools_used=tools_used,
        tokens_in=session.session_in - in0,
        tokens_out=session.session_out - out0,
        cost_usd=cost,
        controller_executed=controller_executed(tools_used, answer,
                                                delegated),
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
    ``nudge`` gates the delegation nudges only; the turn contract's
    re-prompts are not nudges.
    """
    session.cancel_requested = False
    session.last_error = ''          # this turn's provider failure, if any
    session.turn_waiting = False     # set by join/check/plan (orchestrator)
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
    delegated = 0
    try:
        answer, delegated = _drive(step, run_tools, add_user, nudge,
                                   tools_used)
    finally:
        _close_turn(start, in0, out0, cost0, unpriced0, struggle0,
                    tools_used, answer, delegated)


class _Round:
    """What one provider round produced and what the loop decided."""

    def __init__(self, text: str, tool_calls: list) -> None:
        self.text = (text or '').strip()
        self.calls = tool_calls
        self.answer: Optional[str] = None     # set when the turn ends here
        self.collapse = False                 # lone final_answer/plan round
        self.delegated = 0                    # plan tasks guru spawned
        # A contract re-prompt is due: ``reason`` says why, ``fallback``
        # is the answer once the re-prompt cap is spent.
        self.reprompt = False
        self.reason = ''
        self.fallback = ''

    def call(self, name: str) -> Optional[dict]:
        """Arguments of the first call named ``name`` in this round."""
        return next((args for n, args, _ in self.calls if n == name), None)

    def ask_again(self, reason: str, fallback: str) -> None:
        self.reprompt, self.reason, self.fallback = True, reason, fallback


def _drive(step, run_tools, add_user, nudge: bool, tools_used: list
           ) -> tuple[str, int]:
    """The round loop proper; every requested tool lands in ``tools_used``.
    Returns ``(answer, delegated)``: the final answer text (``''`` on
    cancel, error or a delegating turn) and the plan tasks spawned."""
    called: set = set()
    reprompts = 0
    rounds = 0
    last_text = ''
    last_plan: Optional[dict] = None
    delegation_nudged = False
    panel_asked = False
    delegated = 0
    turn0 = _turn_start()
    while True:
        if session.cancel_requested:
            ui.console.print("[yellow]* cancelled[/yellow]")
            return '', delegated
        cap = _MAX_ROUNDS if session.controller else _MAX_TOOL_ROUNDS
        if rounds >= cap:
            # The cap is the only exit while the model keeps calling
            # tools (or a controller keeps planning without an accepted
            # plan): end on what it last wrote.
            ledger.bump('protocol_violation')
            ui.console.print(
                f"[dim yellow]\\[CONTRACT][/dim yellow] {rounds} rounds"
                " without a final answer — ending the turn")
            content = _capped_text(last_text, last_plan, rounds)
            _settle(content, False)
            _render_answer(content)
            return content, delegated
        rounds += 1
        ui.note_thinking()
        result = step()
        if result is None:
            # None = stop: a cancel (flagged) or an error (step printed it).
            if session.cancel_requested:
                ui.console.print("[yellow]* cancelled[/yellow]")
            return '', delegated
        ui.status_draw()
        rnd = _Round(*result)
        forced = forced_tool()
        if rnd.text:
            last_text = rnd.text

        if not rnd.calls:
            # A text-only reply. A controller's text may carry the plan as
            # JSON (an adapter that cannot force); otherwise the contract
            # decides whether the text is the answer.
            if forced == FORCE_PLAN and _plan_from_text(
                    rnd, add_user, tools_used, turn0):
                last_plan = plan.from_text(rnd.text) or last_plan
            elif not rnd.text or _forcing(forced):
                rnd.ask_again('empty reply' if not rnd.text
                              else 'text where a tool call was forced',
                              rnd.text)
            else:
                rnd.answer = rnd.text
        else:
            pending = []
            seen_plan = False
            for name, args, ref in rnd.calls:
                tools_used.append(name)
                if name == 'plan':
                    duplicate = seen_plan       # one plan verdict per round
                    seen_plan = True
                    if not duplicate:
                        last_plan = args
                elif name in _NEVER_DUPLICATE:
                    duplicate = False
                else:
                    key = (name, _args_key(args))
                    duplicate = key in called
                    if not duplicate:
                        called.add(key)
                pending.append((name, args, ref, duplicate))
            before = len(session.messages)
            run_tools(pending)
            _after_tools(rnd, turn0, before)

        if rnd.reprompt:
            if reprompts < _REPROMPT_CAP:
                reprompts += 1
                ui.console.print(
                    f"[dim yellow]\\[CONTRACT][/dim yellow] {rnd.reason}"
                    " — asking for a tool call")
                add_user(plan.PLAN_REPROMPT_TEXT
                         if forced == FORCE_PLAN else plan.REPROMPT_TEXT)
                continue
            ledger.bump('protocol_violation')
            rnd.answer = rnd.fallback

        delegated += rnd.delegated
        if session.turn_waiting:
            # A plan delegated, a join opened a barrier (or check kept
            # polling running sub-agents): stop here instead of another
            # model round. No answer is rendered; the mailbox resumes this
            # agent with the results.
            ui.console.print("[dim]\\[waiting for sub-agents][/dim]")
            return '', delegated
        if rnd.answer is None:
            continue

        # The turn ends on rnd.answer.
        content = rnd.answer
        if (content and session.can_spawn and not panel_asked
                and not _mailbox_turn()):
            # One boolean cannot stand in for three specialist questions,
            # so the panel batch carries no heuristic. A mailbox delivery
            # is the sub-agents' results, not a task to staff, so it is
            # never judged (f1929d55c41a).
            panel_asked = True
            decisions.shadow(
                'panel', decisions.panel_questions(_turn_request()))
        if (nudge and not delegation_nudged and content
                and _should_delegate()):
            delegation_nudged = True
            ledger.bump('delegation_nudges')
            ui.console.print(
                "[dim yellow]\\[DELEGATE][/dim yellow] broad task, no"
                " sub-agents — asking it to spawn a domain panel")
            add_user(_DELEGATION_TEXT)
            continue
        _settle(content, rnd.collapse)
        _render_answer(content)
        return content, delegated


def _capped_text(last_text: str, last_plan: Optional[dict], rounds: int
                 ) -> str:
    """The answer when the round cap ends a turn: a controller's last
    plan rendered through ``plan.fallback_text`` (its own text first),
    else the last text the model wrote, else a fixed line."""
    if session.controller and last_plan is not None:
        return plan.fallback_text(
            last_text, last_plan,
            [f'{rounds} plan rounds without an accepted plan'])
    return last_text or _CAPPED_TEXT.format(n=rounds)


def _args_key(args: object) -> str:
    """A hashable identity for a call's arguments (nested lists — the
    plan's tasks — included)."""
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(args)


def _answer_text(args: dict, text: str) -> str:
    """The reply of an accepted ``answer`` plan: its ``answer`` field, else
    ``text`` (the round's own assistant text). ``answer`` is never
    rejected for lacking text; only when both are empty does the loop
    fall to the empty-reply re-prompt."""
    parsed, _ = plan.parse(args)
    answer = parsed.answer if parsed is not None else ''
    return answer or text.strip()


def _plan_from_text(rnd: _Round, add_user, tools_used: list,
                    turn0: int) -> bool:
    """A controller's text reply parsed as a plan (the text path of an
    adapter that cannot force a tool call). False when the text holds no
    plan object. Otherwise the plan runs through the ``plan`` tool: an
    accepted ``answer`` sets ``rnd.answer`` (its text, else the prose
    around the plan object, else the empty-reply re-prompt), a
    ``delegate`` ends the turn (``session.turn_waiting``), a re-ask or
    refusal goes back to the model as a user message — a second re-ask
    ends the turn on the fallback text with ``protocol_violation``."""
    args = plan.from_text(rnd.text)
    if args is None:
        return False
    tools_used.append('plan')
    result = tools.execute_tool('plan', args)
    kind = _plan_kind(result)
    if kind == 'answer':
        answer = _answer_text(args, plan.prose_around(rnd.text))
        if answer:
            rnd.answer = answer
        else:
            rnd.ask_again('answer plan without text',
                          plan.fallback_text('', args,
                                             ['outcome answer without text']))
    elif kind == 'delegated':
        rnd.delegated = _task_count(args)
    elif kind == 'reask' and _reasks_exceeded(turn0, result):
        ledger.bump('protocol_violation')
        rnd.answer = plan.fallback_text(
            '', args, [plan.reask_problems(result)])
    else:
        add_user(result)
    return True


def _after_tools(rnd: _Round, turn0: int, start: int) -> None:
    """Read a round's ``plan`` / ``final_answer`` outcome after its tools
    ran (the adapter threaded each result into ``session.messages`` from
    index ``start`` on; the first ``plan`` result there is the round's
    verdict — a second plan in the round was refused unrun)."""
    args = rnd.call('plan')
    if args is not None:
        result = _round_tool_result('plan', start)
        kind = _plan_kind(result)
        if kind == 'answer':
            answer = _answer_text(args, rnd.text)
            if answer:
                rnd.answer = answer
                rnd.collapse = len(rnd.calls) == 1
            else:
                rnd.ask_again('answer plan without text',
                              plan.fallback_text(
                                  '', args, ['outcome answer without text']))
        elif kind == 'delegated':
            rnd.delegated = _task_count(args)
        elif kind == 'reask' and _reasks_exceeded(turn0):
            ledger.bump('protocol_violation')
            rnd.answer = plan.fallback_text(
                rnd.text, args, [plan.reask_problems(result)])
        return
    args = rnd.call('final_answer')
    if args is not None:
        rnd.answer = plan.final_text(args)
        rnd.collapse = len(rnd.calls) == 1


def _plan_kind(result: str) -> str:
    """The handler's decision as its result text encodes it: ``answer``,
    ``delegated``, ``reask`` or ``other`` (refused, unavailable)."""
    if result == plan.ANSWER_ACK:
        return 'answer'
    if result.startswith(plan.DELEGATED_PREFIX):
        return 'delegated'
    if plan.is_reask(result):
        return 'reask'
    return 'other'


def _reasks_exceeded(turn0: int, pending: str = '') -> bool:
    """Whether this turn has had more than ``_REASK_CAP`` plan re-asks,
    counting the ones in ``session.messages`` since ``turn0`` plus a
    ``pending`` re-ask not yet appended (the text path)."""
    n = plan.reasks_in(session.messages[turn0:]) + (1 if pending else 0)
    return n > _REASK_CAP


def _task_count(args: dict) -> int:
    parsed, _ = plan.parse(args)
    return len(parsed.tasks) if parsed is not None else 0


def _round_tool_result(name: str, start: int) -> str:
    """Content of the first tool message named ``name`` appended at or
    after index ``start`` of ``session.messages`` (this round's results);
    ``''`` when there is none."""
    for m in session.messages[start:]:
        if isinstance(m, dict) and m.get('role') == 'tool' \
                and m.get('tool_name') == name:
            return m.get('content') or ''
    return ''


def _settle(answer: str, collapse: bool) -> None:
    """Land ``answer`` in ``session.messages`` as the assistant's text.

    A lone ``final_answer``/``plan`` round (``collapse``) — the assistant's
    tool-call message and its tool result — is replaced by one assistant
    text message, so the next turn's history (and ``final_answer`` readers:
    the orchestrator, the bench) carry the text once. Otherwise the last
    message's text becomes the answer when it is the assistant's (the text
    path: the model's own text or its JSON plan), or the answer is
    appended after the round's tool results.
    """
    msgs = session.messages
    if (collapse and len(msgs) >= 2
            and conversation.msg_role(msgs[-1]) == 'tool'
            and conversation.msg_role(msgs[-2]) == 'assistant'):
        del msgs[-2:]
        msgs.append({'role': 'assistant', 'content': answer})
        return
    if msgs and conversation.msg_role(msgs[-1]) == 'assistant' \
            and not conversation._tool_calls_of(msgs[-1]):
        last = msgs[-1]
        if isinstance(last, dict):
            last['content'] = answer
        else:
            last.content = answer
        return
    msgs.append({'role': 'assistant', 'content': answer})
