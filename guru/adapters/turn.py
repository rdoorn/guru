"""Shared, provider-agnostic tool-calling turn loop.

Every adapter's turn is the same skeleton — ask the model, and while it keeps
requesting tools, run them and ask again — differing only in how a provider is
called and how tool results are threaded back into its native history. This
module owns the shared skeleton so all adapters get the same behaviour:

* cancel checks (between rounds, and mid-stream where the adapter supports it),
* the act-nudge that pokes a model which announced an action but ran no tool,
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
import re

from rich.markdown import Markdown

from guru import config, session, ui

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


def _should_delegate() -> bool:
    """True when a delegation-capable main agent has read enough files to make
    a domain panel worthwhile but has not spawned a single sub-agent — the cue
    for the one-time delegation nudge. Gated on file reads so trivial Q&A never
    triggers it. Disabled when DELEGATION_NUDGE_MIN_READS is 0."""
    if not session.can_spawn or config.DELEGATION_NUDGE_MIN_READS <= 0:
        return False
    reads = 0
    for m in session.messages:
        if not isinstance(m, dict) or m.get('role') != 'tool':
            continue
        name = m.get('tool_name', '')
        if name == 'spawn':
            return False                 # already delegated — leave it alone
        if name in config.DELEGATION_READ_TOOLS:
            reads += 1
    return reads >= config.DELEGATION_NUDGE_MIN_READS


def looks_like_preamble(content: str) -> bool:
    """True if text announces an action instead of answering — a short
    'Let me… / I'll…' preamble, or one trailing off into a promised list.
    Long substantive answers (the real result) do not match."""
    if len(content) > 600:
        return False
    if content.rstrip().endswith((':', '…', '...')):
        return True
    return bool(_PREAMBLE_RE.search(content))


def _render_answer(content: str) -> None:
    ui.console.print("\n[bold green]answer>[/bold green]")
    ui.console.print(Markdown(content))
    ui.console.print()


def run_loop(*, step, run_tools, add_user, nudge: bool = True) -> None:
    """Drive one user turn to a final answer using the adapter's closures.

    Owns the shared control flow; the adapter owns the provider calls and
    history threading. See the module docstring for the closure contracts.
    """
    session.cancel_requested = False
    called: set = set()
    nudged = 0
    delegation_nudged = False
    while True:
        if session.cancel_requested:
            ui.console.print("[yellow]* cancelled[/yellow]")
            return
        ui.note_thinking()
        result = step()
        if result is None:
            # None = stop: a cancel (flagged) or an error (step printed it).
            if session.cancel_requested:
                ui.console.print("[yellow]* cancelled[/yellow]")
            return
        ui.status_draw()
        text, tool_calls = result

        if not tool_calls:
            content = (text or '').strip()
            stalled = not content or looks_like_preamble(content)
            if nudge and stalled and nudged < _NUDGE_CAP:
                nudged += 1
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
                ui.console.print(
                    "[dim yellow]\\[DELEGATE][/dim yellow] broad task, no"
                    " sub-agents — asking it to spawn a domain panel")
                add_user(_DELEGATION_TEXT)
                continue
            _render_answer(content)
            return

        pending = []
        for name, args, ref in tool_calls:
            key = (name, tuple(sorted(args.items())))
            duplicate = key in called
            if not duplicate:
                called.add(key)
            pending.append((name, args, ref, duplicate))
        run_tools(pending)
