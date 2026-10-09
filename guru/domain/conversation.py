"""Neutral conversation handling: save/resume and compaction (hybrid D).

The neutral message format is the normalized dict
``{role, content, tool_calls?, tool_name?, tool_args?, tool_call_id?}``
(``tool_args``, the call's arguments, is kept in memory only for the
delegation nudge and is not persisted; ``tool_call_id`` and the ``id`` on a
``tool_calls`` entry are the provider's ids, kept so the remote adapters
rebuild a past tool round in its native shape and the prompt cache reads
the prefix back). Adapters translate to/from it, so these operations are
provider-independent.
"""
import ast
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from guru import config, log, session, skills, ui
from guru.domain import claims, ledger, tools


def message_to_dict(msg: object) -> dict:
    """Normalise a message (dict or provider object) to a clean JSON dict."""
    if isinstance(msg, dict):
        data: dict = msg
    elif hasattr(msg, 'model_dump'):
        data = msg.model_dump()
    else:
        data = dict(msg)  # type: ignore[call-overload]
    out: dict = {
        'role': data.get('role', 'user'),
        'content': data.get('content') or '',
    }
    if data.get('tool_name'):
        out['tool_name'] = data['tool_name']
    if data.get('tool_call_id'):
        out['tool_call_id'] = data['tool_call_id']
    if data.get('tool_calls'):
        out['tool_calls'] = data['tool_calls']
    return out


TRANSCRIPT_ARG_CHARS = 500    # per-argument cap in saved transcripts


def _call_parts(call: object) -> tuple:
    """``(name, arguments)`` from any tool-call shape we store or receive:
    provider dicts/objects with ``function.name``/``function.arguments``,
    already-serialised ``{'name', 'args'}`` records, or bare names."""
    if isinstance(call, str):
        return call, {}
    if isinstance(call, dict):
        if 'function' not in call:            # transcript record shape
            return call.get('name', ''), call.get('args', {})
        fn = call.get('function')
    else:
        fn = getattr(call, 'function', None)
    if fn is None:
        return '', {}
    if isinstance(fn, dict):
        return fn.get('name', ''), fn.get('arguments')
    return getattr(fn, 'name', ''), getattr(fn, 'arguments', None)


def tool_call_records(tool_calls: Optional[Iterable],
                      limit: int = TRANSCRIPT_ARG_CHARS) -> list:
    """Normalise a message's tool calls for a saved transcript.

    Each call becomes ``{'name': str, 'args': dict}``. Argument values longer
    than ``limit`` characters (non-strings measured as JSON) are cut to
    ``limit`` and the record gains a ``note`` naming them with their original
    sizes. Accepts provider dicts, provider objects, JSON-string arguments
    (LiteLLM), earlier transcript records and bare names, so re-reading an
    old transcript through it is safe.
    """
    out: list = []
    for call in tool_calls or []:
        name, raw = _call_parts(call)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw or '{}')
            except ValueError:
                raw = {'_raw': raw}
        if not isinstance(raw, dict):
            raw = {'_raw': raw} if raw is not None else {}
        args: dict = {}
        clipped: list = []
        for key, value in raw.items():
            text = value if isinstance(value, str) else \
                json.dumps(value, ensure_ascii=False, default=str)
            if len(text) > limit:
                clipped.append(f'{key} ({len(text)} chars)')
                value = text[:limit]
            args[key] = value
        rec: dict = {'name': name, 'args': args}
        if clipped:
            rec['note'] = (f'truncated to {limit} chars: '
                           + ', '.join(clipped))
        out.append(rec)
    return out


def transcript_record(msg: object) -> dict:
    """:func:`message_to_dict` with ``tool_calls`` reduced to
    :func:`tool_call_records` — the shape saved transcripts use. Not for
    save/resume, which needs the raw provider ``tool_calls``."""
    rec = message_to_dict(msg)
    if rec.get('tool_calls'):
        rec['tool_calls'] = tool_call_records(rec['tool_calls'])
    return rec


def msg_role(msg: object) -> str:
    if isinstance(msg, dict):
        return msg.get('role', '')
    return getattr(msg, 'role', '')


def msg_content(msg: object) -> str:
    if isinstance(msg, dict):
        return msg.get('content') or ''
    return getattr(msg, 'content', '') or ''


# --- the turn's request ------------------------------------------------------

# User messages the turn loop itself injects; they are never the user's
# request. NUDGE_TEXT and DELEGATION_TEXT are historical (the loop no
# longer nudges) and are kept so saved histories that carry them are still
# read correctly, as are the retired turn contract's re-prompt and plan
# re-ask prefixes (_RETIRED_PREFIXES).
NUDGE_TEXT = (
    "Do not describe what you will do — do it now. Call the tool you need in"
    " this reply (use search_tools first if it is not active). If you are"
    " genuinely finished, give the final answer."
)
DELEGATION_TEXT = (
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
# The verify send-back (guru.adapters.turn): an answer from an agent that
# changed files and ran no tests or lint since goes back once.
VERIFY_REFUSAL = 'Not delivered: you changed files and ran no '
_RETIRED_PREFIXES = ('Plan not accepted: ', 'Reply with a tool call:',
                     'Reply with one plan tool call:')
# First characters of a mailbox delivery (the orchestrator's joined or
# single sub-agent result): a turn it starts is a synthesis turn, and its
# text is the sub-agents' output, not a request from the user.
MAILBOX_PREFIXES = ('[joined results]', '[result from')
# Cap on what :func:`request_in` returns: the request is quoted into judge
# packets (the panel judge, the gate reviewer) and ledger rows.
REQUEST_CHARS = 1000


def is_nudge(text: str) -> bool:
    """True for a user message the turn loop itself injected: the verify
    send-back, the answer check's problems (:mod:`guru.domain.claims`),
    or — in old histories — a delegation or act nudge, a turn-contract
    re-prompt or a plan re-ask."""
    return (text in (NUDGE_TEXT, DELEGATION_TEXT)
            or text.endswith(DELEGATION_TEXT)
            or text.startswith((VERIFY_REFUSAL, claims.PREFIX)
                               + _RETIRED_PREFIXES))


def is_mailbox(text: str) -> bool:
    """True for a user message that is a mailbox delivery."""
    return text.startswith(MAILBOX_PREFIXES)


def turn_start(messages: list) -> int:
    """Index in ``messages`` of the current turn's opening message: the
    last user message that is not a loop nudge; 0 when there is none."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if msg_role(m) != 'user':
            continue
        text = msg_content(m).strip()
        if text and not is_nudge(text):
            return i
    return 0


def request_start(messages: list) -> int:
    """Index in ``messages`` of the user's current request: the last user
    message that is neither a loop nudge nor a mailbox delivery (a
    delivery opens a *turn*, not a request); 0 when there is none. The
    span from here is one request's whole delegate/resume history."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if msg_role(m) != 'user':
            continue
        text = msg_content(m).strip()
        if text and not is_nudge(text) and not is_mailbox(text):
            return i
    return 0


def mailbox_turn(messages: list) -> bool:
    """True when the current turn was opened by a mailbox delivery (a
    joined or single sub-agent result): its text is the sub-agents'
    output, not a request from the user."""
    if not messages:
        return False
    return is_mailbox(msg_content(messages[turn_start(messages)]).strip())


def request_in(messages: list, cap: Optional[int] = REQUEST_CHARS) -> str:
    """The user's request in ``messages``: the most recent user message
    that is neither a loop nudge nor a mailbox delivery, capped at
    ``cap`` characters (``REQUEST_CHARS``; ``None`` for the full text —
    the answer check reads the whole request); when
    only deliveries are there (a history that starts with one), the most
    recent of those, capped the same way; ``''`` when there is no user
    message at all. Pure: for any agent's history (the orchestrator reads
    a parent's for the panel judge, the gate reviewer the worker's), not
    only the bound session's."""
    delivery = ''
    for m in reversed(messages):
        if msg_role(m) != 'user':
            continue
        text = msg_content(m).strip()
        if not text or is_nudge(text):
            continue
        if is_mailbox(text):
            delivery = delivery or text
            continue
        return text[:cap]
    return delivery[:cap]


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


# Boundary between the base system prompt and the per-turn dynamic context
# (catalog, active role/skill overlays, open-files ledger) re-rendered on it.
_DYN_SEP = "\n\n--- active context ---\n"


def project_block() -> str:
    """The '[project]' block: the working directory's name, absolute path
    and git branch, plus the rule that requests refer to it; in a
    sandbox-enabled project also ``config.SANDBOX_RULE`` (a worker's:
    ``SANDBOX_WORKER_RULE``, it does not submit; the direct write
    tools are disabled there, edits go through the sandbox verbs and the
    quality gate).

    Rendered for every agent so none of them has to guess which codebase a
    request means (triage 2026-09-23-claude-tiers: an agent asked "which
    repository?" instead of looking).
    """
    cwd = Path.cwd().resolve()
    lines = ["[project]", f"- name: {cwd.name}", f"- path: {cwd}"]
    branch = session.git_branch
    if branch:
        lines.append(f"- git branch: {branch}")
    lines.append(
        "This working directory is the current project; user requests refer"
        " to it unless they say otherwise.")
    if tools._sandbox_available():
        lines.append(config.SANDBOX_WORKER_RULE if session.task_id
                     else config.SANDBOX_RULE)
    return "\n".join(lines)


def _ledger_block() -> str:
    """The '[open files]' sha list, or '' when nothing is tracked."""
    ledger = session.file_shas
    if not ledger:
        return ''
    cwd = Path.cwd()
    lines = ["[open files]"]
    for key, sha in ledger.items():
        p = Path(key)
        try:
            shown = p.relative_to(cwd)
        except ValueError:
            shown = p
        lines.append(f"- {shown} (sha:{sha})")
    lines.append(
        "Reuse a sha for edit_file instead of re-reading; if edit_file"
        " reports a mismatch, that file changed — re-read it.")
    return "\n".join(lines)


def refresh_system_context() -> None:
    """Rebuild the dynamic tail of the system prompt (messages[0]) in place.

    Renders, in order, the project block (cwd + branch), the roles/skills
    catalog, the active role overlay, the active skill overlay, and the
    open-files sha ledger -- each a single copy that survives
    pruning/compaction (message 0 is always kept) and is counted in the
    'sys' context bucket. Idempotent: the previous tail is stripped first,
    so calling it every turn never doubles it.
    """
    msgs = session.messages
    if not msgs or not isinstance(msgs[0], dict) \
            or msgs[0].get('role') != 'system':
        return
    base = (msgs[0].get('content') or '').split(_DYN_SEP)[0]

    sections = [project_block()]
    catalog = skills.catalog_block()
    if catalog:
        sections.append(catalog)
    role = skills.get(session.active_role) if session.active_role else None
    if role is not None and role.kind == skills.ROLE:
        sections.append(f"[role: {role.name}]\n{role.body}")
    skill = skills.get(session.active_skill) if session.active_skill else None
    if skill is not None and skill.kind == skills.SKILL:
        sections.append(f"[skill: {skill.name}]\n{skill.body}")
    ledger = _ledger_block()
    if ledger:
        sections.append(ledger)

    if sections:
        msgs[0]['content'] = base + _DYN_SEP + "\n\n".join(sections)
    else:
        msgs[0]['content'] = base


def _recent_question(messages: list, index: int) -> str:
    """The most recent user message at or before ``index`` (the question the
    tool output was helping answer)."""
    for j in range(min(index, len(messages) - 1), -1, -1):
        if msg_role(messages[j]) == 'user':
            return msg_content(messages[j]).strip()
    return ''


def _summarize_relevant(question: str, content: str, tool_name: str) -> str:
    """Query-focused compaction: ask the model to keep only the parts of a
    bulky tool result relevant to the question. Falls back to truncation."""
    prompt = (
        "Extract only the parts of the following content that are relevant to"
        " answering this question. Keep concrete facts, values, and"
        " identifiers; be concise; output only the extract.\n\n"
        f"Question: {question}\n\nContent:\n{content}")
    try:
        summary = (session.adapter.summarise(prompt) or '').strip()
    except Exception:                                # noqa: BLE001
        log.exc('tool-result summarise failed')
        summary = ''
    if not summary:
        summary = content[:config.WEB_SUMMARIZE_OVER_CHARS] + "\n…(truncated)"
    q = question[:60]
    return f"[{tool_name} summary · query: {q}]\n{summary}"


_OUTLINE_RE = re.compile(r'^\s*(async def |def |class |@|import |from )')
_NUM_PREFIX = re.compile(r'^\s*\d+\t')


def _sig(node) -> str:
    """A def/class signature line (no body)."""
    if isinstance(node, ast.ClassDef):
        bases = ', '.join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases}):" if bases \
            else f"class {node.name}:"
    prefix = 'async def ' if isinstance(node, ast.AsyncFunctionDef) else 'def '
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ''
    return f"{prefix}{node.name}({ast.unparse(node.args)}){ret}:"


def _outline_code(read_output: str) -> str:
    """Compact a large read_file result to a navigable skeleton: header,
    imports, and def/class signatures + one-line docstrings (bodies dropped).
    AST for a full .py read; regex/truncate fallback otherwise."""
    parts = read_output.split('\n')
    header = parts[0] if parts else ''
    body = parts[1:]
    path = header.split(' (lines', 1)[0].strip()
    source = "\n".join(_NUM_PREFIX.sub('', ln) for ln in body)

    if path.endswith('.py'):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        if tree is not None:
            out = [header, '[outline]']
            doc = ast.get_docstring(tree)
            if doc:
                out.append(f'"""{doc.splitlines()[0]}"""')
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    out.append(ast.unparse(node))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef)):
                    out.append(_sig(node))
                    d = ast.get_docstring(node)
                    if d:
                        out.append(f'    """{d.splitlines()[0]}"""')
                    if isinstance(node, ast.ClassDef):
                        for sub in node.body:
                            if isinstance(sub, (ast.FunctionDef,
                                                ast.AsyncFunctionDef)):
                                out.append('    ' + _sig(sub))
                                sd = ast.get_docstring(sub)
                                if sd:
                                    out.append(
                                        f'        """{sd.splitlines()[0]}"""')
            return "\n".join(out)

    kept = [ln for ln in body if _OUTLINE_RE.match(_NUM_PREFIX.sub('', ln))]
    if len(kept) >= 3:
        return "\n".join([header, '[outline]'] + kept)
    return "\n".join([header, '[truncated]'] + body[:40])


def _tool_calls_of(msg: object) -> list:
    """A message's tool calls (dict key or attribute), ``[]`` when none."""
    if isinstance(msg, dict):
        return list(msg.get('tool_calls') or [])
    return list(getattr(msg, 'tool_calls', None) or [])


def apply_retention(messages: list) -> None:
    """Post-turn retention: drop empty assistant steps, then compact each
    tool result per its tool's retain policy (keep / summarize / outline),
    only when it exceeds the size threshold. Replaces the blanket prune so
    follow-up questions keep the useful, relevant context.

    A text-less assistant step that carries tool calls is kept: it is a
    few tokens, and dropping it changed the shape of every past tool round
    at the next turn (the results lost their call), which broke the prompt
    cache prefix on every turn (improve-loop iteration 3)."""
    kept = []
    for i, m in enumerate(messages):
        role = msg_role(m)
        if role == 'assistant' and not msg_content(m).strip() \
                and not _tool_calls_of(m):
            continue
        if role == 'tool' and isinstance(m, dict):
            name = m.get('tool_name', '')
            policy = tools.retain_policy(name)
            content = m.get('content') or ''
            if (policy == 'summarize'
                    and len(content) > config.WEB_SUMMARIZE_OVER_CHARS):
                q = _recent_question(messages, i)
                m['content'] = _summarize_relevant(q, content, name)
            elif (policy == 'outline'
                    and len(content) > config.OUTLINE_FILE_OVER_CHARS):
                m['content'] = _outline_code(content)
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

    Apply per-tool retention, refresh the token estimate (including tool
    schemas), then summarise-compact if the lean history still exceeds the
    threshold. Using the post-prune estimate to decide avoids compacting just
    because a turn fetched a large (now-discarded) tool result.
    """
    apply_retention(session.messages)
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
        ledger.bump('compactions')
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
        ledger.bump('compactions')
    else:
        session.messages = [system] + flat
