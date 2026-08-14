"""Neutral conversation handling: save/resume and compaction (hybrid D).

The neutral message format is the normalized dict
``{role, content, tool_calls?, tool_name?}``. Adapters translate to/from it,
so these operations are provider-independent.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

from guru import config, session, ui
from guru.domain import tools


def message_to_dict(msg: object) -> dict:
    """Normalise a message (dict or provider object) to a clean JSON dict."""
    if not isinstance(msg, dict):
        msg = msg.model_dump() if hasattr(msg, 'model_dump') else dict(msg)
    out: dict = {
        'role': msg.get('role', 'user'),
        'content': msg.get('content') or '',
    }
    if msg.get('tool_name'):
        out['tool_name'] = msg['tool_name']
    if msg.get('tool_calls'):
        out['tool_calls'] = msg['tool_calls']
    return out


def msg_role(msg: object) -> str:
    if isinstance(msg, dict):
        return msg.get('role', '')
    return getattr(msg, 'role', '')


def msg_content(msg: object) -> str:
    if isinstance(msg, dict):
        return msg.get('content') or ''
    return getattr(msg, 'content', '') or ''


def group_messages(msgs: list) -> list:
    """Split messages into turn-groups starting at each user message.

    Keeping an assistant message and its tool results in one group means
    compaction never separates a tool call from its result.
    """
    groups: list = []
    current: list = []
    for m in msgs:
        if msg_role(m) == 'user' and current:
            groups.append(current)
            current = []
        current.append(m)
    if current:
        groups.append(current)
    return groups


def estimate_tokens(msgs: list) -> int:
    """Rough token estimate (~4 characters per token)."""
    return sum(len(msg_content(m)) for m in msgs) // 4


def _first_user_message(path: Path) -> str:
    """Return a short title from the first user message in a memory file."""
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return '(unreadable)'
    for m in data:
        if m.get('role') == 'user' and m.get('content'):
            text = ' '.join(m['content'].split())
            return text[:60] + ('…' if len(text) > 60 else '')
    return '(no user message)'


def save_conversation() -> None:
    """Write the current conversation (minus the system prompt) to disk."""
    if len(session.messages) <= 1:
        ui.console.print("[yellow]Nothing to save yet.[/yellow]")
        return
    directory = config.PROJECT_MEMORY_DIR
    directory.mkdir(parents=True, exist_ok=True)
    payload = [message_to_dict(m) for m in session.messages[1:]]
    path = directory / f"{uuid.uuid4()}.memory"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    ui.console.print(f"[green]Saved[/green] conversation to {path}")


def reactivate_tools(msgs: list) -> None:
    """Re-activate tools referenced by a restored conversation.

    Prevents the empty-response bug where the model tries to call a tool it
    used earlier that is no longer in the active tool set.
    """
    for m in msgs:
        candidates = []
        if m.get('tool_name'):
            candidates.append(m['tool_name'])
        for call in (m.get('tool_calls') or []):
            fn = call.get('function') if isinstance(call, dict) else None
            if fn and fn.get('name'):
                candidates.append(fn['name'])
        for name in candidates:
            tools.activate(name)


def resume_command() -> None:
    """Interactive selector to restore a saved conversation."""
    directory = config.PROJECT_MEMORY_DIR
    files = (
        sorted(
            directory.glob('*.memory'),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if directory.exists()
        else []
    )
    if not files:
        ui.console.print(
            "[yellow]No saved conversations for this project.[/yellow]"
        )
        return

    labels = []
    for p in files:
        stamp = datetime.fromtimestamp(p.stat().st_mtime).strftime(
            '%Y-%m-%d %H:%M'
        )
        labels.append(f"{stamp}  {_first_user_message(p)}")

    idx = ui.pick('Resume  ↑/↓ navigate · Enter select · Esc cancel', labels)
    if idx is None:
        return

    path = files[idx]
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as e:
        ui.console.print(f"[red]Could not load {path.name}: {e}[/red]")
        return

    session.messages = (
        [{"role": "system", "content": config.build_system_prompt()}] + data)
    tools.reset_active_tools()
    reactivate_tools(data)
    ui.console.print(
        f"[green]Resumed[/green] {path.name} ({len(data)} messages)."
    )


def _summarise_groups(groups: list) -> str:
    """Summarise old turn-groups into a compact note using the adapter."""
    lines = []
    for group in groups:
        for m in group:
            content = msg_content(m).strip()
            if content:
                lines.append(f"{msg_role(m)}: {content}")
    transcript = "\n".join(lines)[:12000]
    try:
        return session.adapter.summarise(transcript)
    except Exception as e:
        return f'(summary failed: {e})'


def prune_tool_exchanges(messages: list) -> None:
    """Drop tool results and text-less tool-call steps from history in place.

    Tool output (web pages, file dumps, directory listings) is the largest and
    least reusable part of the context, and it is otherwise re-sent on every
    subsequent turn. The assistant's text answer already captures the
    conclusion; if the raw data is needed again the model can simply re-run
    the tool. Called after each turn so the next one starts lean. The system
    prompt, user messages, and every assistant message with text are kept.
    """
    kept = []
    for m in messages:
        role = msg_role(m)
        if role == 'tool':
            continue
        if role == 'assistant' and not msg_content(m).strip():
            continue
        kept.append(m)
    messages[:] = kept


def context_breakdown(messages: list, active_tool_names=None,
                      can_spawn: bool = False) -> dict:
    """Estimate resident context tokens by category (rough, ~4 chars/token).

    Accounts for everything we actually send: message content by role, plus
    the active tool schemas (re-sent every turn but not stored in messages).
    The returned dict includes 'total' so callers can render percentages that
    sum to 100% of active usage regardless of how full the window is.
    """
    def toks(text: str) -> int:
        return len(text) // 4

    buckets = {'sys': 0, 'in': 0, 'out': 0, 'toolout': 0}
    role_bucket = {'system': 'sys', 'user': 'in',
                   'assistant': 'out', 'tool': 'toolout'}
    for m in messages:
        bucket = role_bucket.get(msg_role(m))
        if bucket:
            buckets[bucket] += toks(msg_content(m))

    schema = 0
    for spec in tools.specs_for(active_tool_names or set(), can_spawn):
        schema += toks(spec.get('name', '') + spec.get('description', ''))
        for key, val in spec.get('parameters', {}).items():
            schema += toks(key + str(val))
    buckets['tools'] = schema
    buckets['total'] = sum(buckets.values())
    return buckets


def after_turn() -> None:
    """Post-turn context maintenance shared by the TUI and the REPL.

    Prune tool output, refresh the token estimate (including tool schemas),
    then summarise-compact if the lean history still exceeds the threshold.
    Using the post-prune estimate to decide avoids compacting just because a
    turn fetched a large (now-discarded) tool result.
    """
    prune_tool_exchanges(session.messages)
    session.ctx_used = context_breakdown(
        session.messages, session.active_tool_names,
        session.can_spawn)['total']
    if session.num_ctx and session.ctx_used > config.COMPACT_AT \
            * session.num_ctx:
        compact_messages()
        session.ctx_used = context_breakdown(
            session.messages, session.active_tool_names,
            session.can_spawn)['total']


def compact_messages(force: bool = False) -> None:
    """Compact the conversation: trim thinking, evict tool output, summarise.

    Runs the cheap steps first and only asks the model for a summary if the
    conversation is still too large. Preserves the system prompt and the
    most recent turn-groups verbatim.
    """
    if len(session.messages) <= 1:
        return
    system = session.messages[0]
    # Normalising drops per-message 'thinking' blocks — a free saving.
    history = [message_to_dict(m) for m in session.messages[1:]]
    limit = int(session.num_ctx * config.COMPACT_AT)

    if not force and estimate_tokens(history) <= limit:
        session.messages = [system] + history
        return

    groups = group_messages(history)
    if len(groups) <= config.KEEP_RECENT_GROUPS:
        old, recent = [], groups
    else:
        old = groups[:-config.KEEP_RECENT_GROUPS]
        recent = groups[-config.KEEP_RECENT_GROUPS:]

    # Evict tool outputs from old groups (biggest, least-needed context).
    for group in old:
        for m in group:
            if m.get('role') == 'tool' and m.get('content'):
                m['content'] = '[old tool output evicted to save context]'

    flat = [m for g in (old + recent) for m in g]
    if not force and estimate_tokens(flat) <= limit:
        ui.console.print("[dim]\\[COMPACT] evicted old tool outputs[/dim]")
        session.messages = [system] + flat
        return

    if old:
        summary = _summarise_groups(old)
        note = {
            'role': 'system',
            'content': 'Summary of earlier conversation:\n' + summary,
        }
        recent_flat = [m for g in recent for m in g]
        session.messages = [system, note] + recent_flat
        ui.console.print(
            "[dim]\\[COMPACT] folded older turns into a summary[/dim]"
        )
    else:
        session.messages = [system] + flat
