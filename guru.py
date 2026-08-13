import argparse
import atexit
import json
import os
import shutil
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import ollama
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme

_parser = argparse.ArgumentParser(description="guru — local Ollama agent")
_parser.add_argument("--model", default="qwen3-abliterated-32k:latest")
_parser.add_argument(
    "--num-ctx", type=int, default=0,
    help="Override the context window size (0 = auto-detect from the model)",
)
# parse_known_args (not parse_args) so importing the module under a test
# runner — which passes its own argv — does not trigger a SystemExit.
_args, _ = _parser.parse_known_args()

# Teach prompt_toolkit to recognise Shift+Enter.
# iTerm2 (and other modern terminals) already send these sequences natively —
# the same reason Shift+Enter works in Claude Code without any extra setup.
# ANSI_SEQUENCES is read at runtime on every keypress, so additions made here
# take effect immediately. We map to Keys.F13 because it is a valid Keys enum
# member that is never sent by ordinary terminal use.
ANSI_SEQUENCES['\x1b[13;2u'] = Keys.F13    # CSI u (iTerm2, Kitty, WezTerm)
ANSI_SEQUENCES['\x1b[27;2;13~'] = Keys.F13  # xterm modifyOtherKeys format
# Map Ctrl+C in modifyOtherKeys mode so prompt_toolkit raises KeyboardInterrupt
# instead of inserting the escape sequence literally into the buffer.
ANSI_SEQUENCES['\x1b[99;5u'] = Keys.ControlC    # CSI u Ctrl+C
ANSI_SEQUENCES['\x1b[27;5;99~'] = Keys.ControlC  # xterm modifyOtherKeys Ctrl+C
# Map keyboard Enter in modifyOtherKeys mode to F14 so it can be bound to
# submit independently of raw \r (which arrives from pasted newlines and must
# NOT submit). Pasted \r is handled by the 'enter' binding as insert-newline.
ANSI_SEQUENCES['\x1b[27;1;13~'] = Keys.F14  # xterm fmt, modifier=1 (no mod)
ANSI_SEQUENCES['\x1b[27;0;13~'] = Keys.F14  # xterm fmt, modifier=0 (alt)
ANSI_SEQUENCES['\x1b[13;1u'] = Keys.F14      # CSI u, modifier=1 (no mod)
ANSI_SEQUENCES['\x1b[13u'] = Keys.F14        # CSI u short form

# Override Rich's markdown link styles: bright blue, no underline. Defaults
# are markdown.link=bright_blue and markdown.link_url=blue+underline.
console = Console(
    highlight=False,
    theme=Theme({
        'markdown.link': 'bright_blue',
        'markdown.link_url': 'bright_blue',
    }),
)

_ctrl_c_times: list = []


def _sigint_handler(signum: int, frame: object) -> None:
    """Exit on double Ctrl+C within 1 s; otherwise cancel current operation."""
    now = time.monotonic()
    _ctrl_c_times[:] = [t for t in _ctrl_c_times if now - t <= 1.0]
    _ctrl_c_times.append(now)
    if len(_ctrl_c_times) >= 2:
        console.print("\n[bold red]Exiting.[/bold red]")
        sys.exit(0)
    raise KeyboardInterrupt


MODEL = _args.model

# --- Config, setup, and domain safeguards ------------------------------------

# Global config lives in ~/.guru; project-specific state lives in a .guru/
# folder inside the current project so it travels with the project.
GURU_HOME = Path(os.path.expanduser('~/.guru'))
GURU_MD_PATH = GURU_HOME / 'GURU.md'                 # global base persona

PROJECT_GURU_DIR = Path.cwd() / '.guru'
PROJECT_GURU_MD = PROJECT_GURU_DIR / 'GURU.md'       # appended to the global
DOMAINS_ALLOW_PATH = PROJECT_GURU_DIR / 'domains_allow.txt'  # per-project
PROJECT_MEMORY_DIR = PROJECT_GURU_DIR / 'memory'     # saved conversations

# Search-engine backend host. web_search gates on this so "allow internet
# access at least once" maps to approving the engine. Structured as a
# constant so additional engines can each declare their own backend host.
SEARCH_BACKEND_DOMAIN = 'duckduckgo.com'

# ollama's default context window when a model's modelfile does not set one.
DEFAULT_NUM_CTX = 4096
# Compact the conversation when occupancy crosses this fraction of num_ctx.
COMPACT_AT = 0.85
# Number of most-recent turn-groups kept verbatim during compaction.
KEEP_RECENT_GROUPS = 4

DEFAULT_GURU_MD = """# GURU.md

Instructions for the guru assistant. Edit this file to change guru's
behaviour globally. Add a `.guru/GURU.md` inside a project to append
project-specific instructions.

## Persona

- Be concise and direct.
- Cite sources when you use a tool result.

## Rules

- Do not invent facts. If a tool did not return something, say so.
"""


def _ensure_setup() -> None:
    """Create the global ~/.guru dir and a default GURU.md if missing.

    Project state (.guru/ in the current directory) is created lazily on
    first write so read-only sessions do not litter arbitrary directories.
    """
    GURU_HOME.mkdir(parents=True, exist_ok=True)
    if not GURU_MD_PATH.exists():
        GURU_MD_PATH.write_text(DEFAULT_GURU_MD, encoding='utf-8')


def _load_allowed_domains() -> set:
    """Read the allow-list file into a set of lowercased domains."""
    try:
        lines = DOMAINS_ALLOW_PATH.read_text(encoding='utf-8').splitlines()
    except OSError:
        return set()
    return {ln.strip().lower() for ln in lines if ln.strip()}


def _domain_of(url: str) -> str:
    """Return the lowercased hostname of a URL, port stripped."""
    host = urlparse(url).hostname
    if not host:
        # Bare host without a scheme (e.g. "example.com/path").
        host = urlparse('//' + url).hostname
    return (host or url).lower()


def _persist_domain(domain: str) -> None:
    """Append a newly approved domain to the project allow-list file."""
    DOMAINS_ALLOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DOMAINS_ALLOW_PATH.open('a', encoding='utf-8') as fh:
        fh.write(domain + '\n')


def _ensure_domain_allowed(domain: str) -> bool:
    """Return True if the domain is allowed, prompting the user if unknown.

    On approval the domain is added to the in-memory set and persisted.
    Denial (no / empty / Ctrl+C) returns False and asks again next time.
    """
    domain = domain.lower()
    if domain in _ALLOWED_DOMAINS:
        return True
    console.print(
        f"\n[yellow]\\[ACCESS][/yellow] Request to access"
        f" [bold]{domain}[/bold]."
    )
    try:
        answer = input(f"Allow access to '{domain}'? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        answer = ''
    if answer in ('y', 'yes'):
        _ALLOWED_DOMAINS.add(domain)
        _persist_domain(domain)
        console.print(
            f"[green]Allowed[/green] {domain} (saved to allow-list)."
        )
        return True
    console.print(f"[red]Denied[/red] {domain}.")
    return False


def _resolve_context_window(model: str) -> tuple:
    """Return (effective num_ctx, architecture ceiling) for a model.

    The effective window is what ollama actually allocates at runtime:
    the --num-ctx override if given, else the modelfile's num_ctx, else
    ollama's default. It is capped at the architecture ceiling reported in
    modelinfo (the model's trained maximum). Returns 0 for ceiling if the
    model info cannot be read.
    """
    try:
        info = ollama.show(model)
    except Exception:
        return (_args.num_ctx or DEFAULT_NUM_CTX, 0)

    modelinfo = info.modelinfo or {}
    arch = modelinfo.get('general.architecture', '')
    ceiling = int(modelinfo.get(f'{arch}.context_length', 0) or 0)

    modelfile_num_ctx = 0
    for line in (info.parameters or '').splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == 'num_ctx':
            try:
                modelfile_num_ctx = int(parts[1])
            except ValueError:
                pass

    num_ctx = _args.num_ctx or modelfile_num_ctx or DEFAULT_NUM_CTX
    if ceiling:
        num_ctx = min(num_ctx, ceiling)
    return (num_ctx, ceiling)


_ensure_setup()
_ALLOWED_DOMAINS: set = _load_allowed_domains()

_STOP_WORDS = {
    'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'for',
    'of', 'and', 'or', 'but', 'what', 'how', 'when', 'where', 'who',
    'which', 'that', 'this', 'are', 'was', 'were', 'be', 'been',
    'being', 'do', 'does', 'did', 'me', 'my', 'you', 'your', 'its',
}


def _relevance_score(query: str, result: dict) -> int:
    """Score a search result by keyword overlap with the query."""
    keywords = {
        w.lower() for w in query.split()
        if w.lower() not in _STOP_WORDS and len(w) > 2
    }
    text = (
        (result.get('title') or '') + ' ' + (result.get('body') or '')
    ).lower()
    return sum(1 for kw in keywords if kw in text)


def web_search(query: str) -> str:
    """
    Search the internet for current information.
    ONLY use this for: current events, recent news, live prices/weather,
    information that changes over time, or topics likely after your training
    cutoff. Do NOT use for math, logic, coding, or stable well-known facts.
    """
    if not _ensure_domain_allowed(SEARCH_BACKEND_DOMAIN):
        return (
            f"Access to the search engine '{SEARCH_BACKEND_DOMAIN}' was"
            " denied by the user. The search was not performed."
        )
    # Tool output shown on screen is debug info for the user (the same text
    # is sent to the model as normal input), so render it in the debug style.
    console.print(f"\n[SEARCH] {query}", style="dim yellow", markup=False)

    results = list(DDGS().text(query, max_results=10))

    scored = sorted(
        results,
        key=lambda r: _relevance_score(query, r),
        reverse=True,
    )
    relevant = [r for r in scored if _relevance_score(query, r) > 0][:5]
    if not relevant:
        relevant = scored[:3]

    output = []
    for i, result in enumerate(relevant, 1):
        entry = (
            f"{i}. {result.get('title')}\n"
            f"URL: {result.get('href')}\n"
            f"Summary: {result.get('body')}\n"
        )
        output.append(entry)
        console.print(entry, style="dim yellow", markup=False)

    return "\n".join(output)


def web_fetch(url: str) -> str:
    """
    Fetch and read the text content of a webpage.
    Use this after web_search when you need more information from a result.
    """
    domain = _domain_of(url)
    if not _ensure_domain_allowed(domain):
        return f"Access to domain '{domain}' was denied by the user."
    console.print(f"\n[cyan]\\[FETCH][/cyan] {url}")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove junk
    for element in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside"
    ]):
        element.decompose()

    text = soup.get_text(
        separator="\n",
        strip=True
    )

    # Don't dump enormous pages into the model
    return text[:15000]


def fetch_github_releases(repo: str) -> str:
    """
    Fetch the latest release for a GitHub repository.
    Use this for software version questions when the project is on
    GitHub. Pass the repo as 'owner/repo', e.g. 'kubernetes/kubernetes'.
    Do NOT use web_search for version questions about GitHub projects —
    use this tool directly.
    """
    console.print(f"\n[cyan]\\[GITHUB][/cyan] {repo}")
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Mozilla/5.0",
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    notes = (data.get('body') or '')[:500]
    return (
        f"Repository: {repo}\n"
        f"Latest release: {data.get('tag_name')}\n"
        f"Name: {data.get('name')}\n"
        f"Published: {data.get('published_at')}\n"
        f"URL: {data.get('html_url')}\n"
        f"Release notes: {notes}\n"
    )


# Each entry: fn (callable), description, tags, parameters.
# search_tools matches against all of these fields.
TOOL_REGISTRY: dict = {
    "web_search": {
        "fn": web_search,
        "description": (
            "Search the web and return ranked results with titles, URLs, and "
            "text snippets. Use for current events, live data, news, prices, "
            "weather, or any information that changes over time."
        ),
        "tags": [
            "search", "web", "internet", "find", "lookup", "query",
            "google", "news", "current", "live", "weather", "prices",
            "events", "information", "recent", "today", "latest",
        ],
        "parameters": {
            "query": "Search terms to look up on the web",
        },
    },
    "web_fetch": {
        "fn": web_fetch,
        "description": (
            "Fetch and read the full text content of any webpage given its"
            " URL. Use to read a specific page, follow a link, retrieve an"
            " endpoint response, or read content from a known URL."
        ),
        "tags": [
            "fetch", "read", "url", "webpage", "page", "html", "content",
            "download", "retrieve", "open", "link", "site", "http", "https",
            "get", "load", "visit", "endpoint", "curl", "request",
        ],
        "parameters": {
            "url": "Full URL to fetch (e.g. https://example.com/page)",
        },
    },
    "fetch_github_releases": {
        "fn": fetch_github_releases,
        "description": (
            "Get the latest release version, tag, and release notes for any"
            " GitHub repository. Use for software version questions on"
            " projects hosted on GitHub."
        ),
        "tags": [
            "github", "release", "version", "latest", "software", "package",
            "tag", "changelog", "update", "repo", "repository", "library",
            "binary", "open source", "project",
        ],
        "parameters": {
            "repo": (
                "Repository in 'owner/repo' format"
                " (e.g. 'kubernetes/kubernetes')"
            ),
        },
    },
}


def _match_tools(query: str) -> list:
    """Rank TOOL_REGISTRY entries by weighted match across metadata fields."""
    keywords = [
        w.lower() for w in query.replace('-', ' ').split()
        if len(w) > 2 and w.lower() not in _STOP_WORDS
    ]
    if not keywords:
        return list(TOOL_REGISTRY.keys())
    scores: dict = {}
    for name, info in TOOL_REGISTRY.items():
        score = 0
        tags_text = ' '.join(info['tags']).lower()
        desc_text = info['description'].lower()
        params_text = ' '.join(info['parameters'].values()).lower()
        for kw in keywords:
            if kw in name.lower():
                score += 5   # exact name match
            if kw in tags_text:
                score += 3   # tag hit
            if kw in desc_text:
                score += 2   # description hit
            if kw in params_text:
                score += 1   # parameter description hit
        if score > 0:
            scores[name] = score
    if not scores:
        return list(TOOL_REGISTRY.keys())
    return sorted(scores, key=scores.__getitem__, reverse=True)


def search_tools(query: str) -> str:
    """
    Search the tool directory for tools matching an action you want to perform.

    Call this with a short phrase describing WHAT YOU WANT TO DO — not the
    user's question. Use action-oriented terms that match what a tool does:
      search_tools("search the web")
      search_tools("fetch webpage url")
      search_tools("get latest github release version")

    Matched tools are added to your active tool set and can be called directly.
    """
    console.print(f"\n[cyan]\\[TOOL_SEARCH][/cyan] {query}")
    matched = _match_tools(query)
    lines: list = [f"Tools matching '{query}':\n"]
    for name in matched:
        info = TOOL_REGISTRY[name]
        param_lines = "\n".join(
            f"      {k}: {v}" for k, v in info['parameters'].items()
        )
        lines.append(
            f"  {name}\n"
            f"    {info['description']}\n"
            f"    Parameters:\n{param_lines}\n"
        )
    lines.append("These tools are now active — call them directly by name.")
    return "\n".join(lines)


SYSTEM_PROMPT = """
You are a helpful assistant with access to a tool directory.

You start each turn with one tool: search_tools.
When you need a capability, call search_tools with a short phrase describing
the ACTION you want to perform — not the user's question.

Correct usage:
  User asks "what is the weather in Amsterdam?"
    → search_tools("search the web current data")

  User asks "what is the latest kubernetes version?"
    → search_tools("get latest github release version")

  User asks "read this URL for me: https://..."
    → search_tools("fetch webpage url")

  User asks "query the /metrics endpoint on localhost"
    → search_tools("fetch url endpoint http")

After search_tools returns matching tools, call them directly by name.
Do not try to call a tool that has not been returned by search_tools first.

When the user asks you to DO something (fetch a URL, search for something,
query an endpoint), use the tool — do not describe how the user could do it
themselves.

Guidelines:
- Never use tools for math, logic, coding, or stable facts from training data.
- If the first search returns poor results, refine the query and search again.
- Always cite sources. Report only what tool results explicitly state.
- If a question requires a location or name not provided, ask first.
"""


def _build_system_prompt() -> str:
    """Assemble the system prompt: built-in + global GURU.md + local .GURU.md.

    The built-in prompt is always first so the search_tools mechanism is
    never lost. The project .guru/GURU.md extends (appends to) the global one.
    """
    parts = [SYSTEM_PROMPT.strip()]
    for path in (GURU_MD_PATH, PROJECT_GURU_MD):
        try:
            text = path.read_text(encoding='utf-8').strip()
        except OSError:
            continue
        if text:
            parts.append(text)
    return "\n\n".join(parts)


messages = [
    {
        "role": "system",
        "content": _build_system_prompt(),
    }
]

# Effective context window and architecture ceiling, resolved at startup and
# on every model switch. _ctx_used tracks the latest measured occupancy.
_NUM_CTX, _CTX_CEILING = _resolve_context_window(MODEL)
_ctx_used: int = 0

# Active tools persist for the lifetime of the conversation — once a tool is
# discovered via search_tools it stays available without re-searching.
active_tool_names: set = set()
active_tools: list = [search_tools]


# Key bindings.
#
# The terminal sends no bracketed-paste markers and encodes pasted newlines
# as \r — byte-identical to keyboard Enter — so the two cannot be told apart
# by value alone. They differ by CONTEXT: a paste feeds every byte into
# prompt_toolkit's input queue at once, so when the handler for a paste's
# first \r fires, the rest of the pasted keys are still queued. A deliberate
# keyboard Enter is the last key, leaving the queue empty. The _enter handler
# uses that: queued keys behind this \r → insert newline; empty queue → submit.
#
# F14 (keyboard Enter re-encoded by modifyOtherKeys, if the terminal does so)
# and Escape+Enter always submit. Shift+Enter and c-j (\n) always insert.
_kb = KeyBindings()


@_kb.add('f13')  # Shift+Enter — mapped via ANSI_SEQUENCES above
def _shift_enter(event: object) -> None:
    event.current_buffer.insert_text('\n')


@_kb.add('f14')  # keyboard Enter re-encoded via modifyOtherKeys
def _mk_enter(event: object) -> None:
    event.current_buffer.validate_and_handle()


@_kb.add('escape', 'enter')  # universal submit fallback
def _escape_enter(event: object) -> None:
    event.current_buffer.validate_and_handle()


@_kb.add('enter')  # \r — submit if deliberate, insert if part of a paste
def _enter(event: object) -> None:
    if event.app.key_processor.input_queue:
        # More keys are queued behind this \r: it is a paste-internal
        # newline, not a deliberate Enter. Insert instead of submitting.
        event.current_buffer.insert_text('\n')
    else:
        event.current_buffer.validate_and_handle()


@_kb.add('c-j')  # \n — some terminals paste newlines as \n → insert
def _linefeed(event: object) -> None:
    event.current_buffer.insert_text('\n')


_session = PromptSession(
    history=FileHistory(os.path.expanduser('~/.ollama_wrapper_history')),
    multiline=True,
    key_bindings=_kb,
)


def _enable_terminal_modes() -> None:
    """Enable modifyOtherKeys mode 2 and bracketed paste before each prompt.

    modifyOtherKeys (\x1b[>4;2m): makes Shift+Enter send a distinct sequence
    so it can be bound to newline instead of submit.

    Bracketed paste (\x1b[?2004h): wraps pasted text in \x1b[200~...\x1b[201~
    so prompt_toolkit delivers it as a single BracketedPaste event — newlines
    inside the paste are inserted literally instead of triggering submit.

    prompt_toolkit also sends \x1b[?2004h on first render, but sending it here
    too ensures it is active before the terminal processes any input.
    """
    sys.stdout.write('\x1b[>4;2m\x1b[?2004h')
    sys.stdout.flush()


_RESET_TERMINAL = '\x1b[>4;0m\x1b[?2004l'

_SELECT_STYLE = Style.from_dict({
    'cursor': 'bold ansicyan',
    'cursor-active': 'bold ansigreen',
    'active': 'ansigreen',
})


def _pick(title: str, options: list, active_idx: int = -1):
    """Arrow-key selector. Returns the chosen index, or None if cancelled.

    The option at active_idx is marked with a check and highlighted.
    """
    if not options:
        return None
    state = {'idx': active_idx if active_idx >= 0 else 0}

    def _text() -> FormattedText:
        lines: list = [('bold', f' {title}\n\n')]
        for i, opt in enumerate(options):
            cursor = i == state['idx']
            is_active = i == active_idx
            prefix = '▶ ' if cursor else '  '
            suffix = ' ✓' if is_active else ''
            if cursor and is_active:
                style = 'class:cursor-active'
            elif cursor:
                style = 'class:cursor'
            elif is_active:
                style = 'class:active'
            else:
                style = ''
            lines.append((style, f'  {prefix}{opt}{suffix}\n'))
        return FormattedText(lines)

    kb = KeyBindings()

    @kb.add('up')
    def _up(event: object) -> None:
        state['idx'] = (state['idx'] - 1) % len(options)

    @kb.add('down')
    def _down(event: object) -> None:
        state['idx'] = (state['idx'] + 1) % len(options)

    @kb.add('enter')
    def _select(event: object) -> None:
        event.app.exit(result=state['idx'])

    @kb.add('escape')
    @kb.add('c-c')
    def _cancel(event: object) -> None:
        event.app.exit(result=None)

    app = Application(
        layout=Layout(
            Window(
                FormattedTextControl(_text, focusable=True),
                height=len(options) + 2,
            )
        ),
        key_bindings=kb,
        style=_SELECT_STYLE,
        full_screen=False,
        mouse_support=False,
    )
    return app.run()


def _models_command() -> None:
    """Interactive model selector — ↑/↓ to navigate, Enter to select."""
    global MODEL, _NUM_CTX, _CTX_CEILING

    try:
        model_list = ollama.list().models
    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        return

    names = sorted(m.model for m in model_list)
    if not names:
        console.print("[yellow]No Ollama models found.[/yellow]")
        return

    active_idx = next((i for i, n in enumerate(names) if n == MODEL), -1)
    idx = _pick(
        'Models  ↑/↓ navigate · Enter select · Esc cancel',
        names,
        active_idx,
    )
    if idx is None:
        return
    MODEL = names[idx]
    _NUM_CTX, _CTX_CEILING = _resolve_context_window(MODEL)
    console.print(
        f"\n[green]Model:[/green] [bold]{MODEL}[/bold]"
        f" [dim](context {_NUM_CTX:,})[/dim]"
    )


# --- Conversation memory: /save and /resume ---------------------------------

def _project_memory_dir() -> Path:
    """Return the project's .guru/memory directory."""
    return PROJECT_MEMORY_DIR


def _message_to_dict(msg: object) -> dict:
    """Normalise a message (dict or ollama Message) to a clean JSON dict."""
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


def _save_conversation() -> None:
    """Write the current conversation (minus the system prompt) to disk."""
    if len(messages) <= 1:
        console.print("[yellow]Nothing to save yet.[/yellow]")
        return
    directory = _project_memory_dir()
    directory.mkdir(parents=True, exist_ok=True)
    payload = [_message_to_dict(m) for m in messages[1:]]
    path = directory / f"{uuid.uuid4()}.memory"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    console.print(f"[green]Saved[/green] conversation to {path}")


def _reactivate_tools(msgs: list) -> None:
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
            if name in TOOL_REGISTRY and name not in active_tool_names:
                active_tool_names.add(name)
                active_tools.append(TOOL_REGISTRY[name]['fn'])


def _resume_command() -> None:
    """Interactive selector to restore a saved conversation."""
    global messages
    directory = _project_memory_dir()
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
        console.print(
            "[yellow]No saved conversations for this project.[/yellow]"
        )
        return

    labels = []
    for p in files:
        stamp = datetime.fromtimestamp(p.stat().st_mtime).strftime(
            '%Y-%m-%d %H:%M'
        )
        labels.append(f"{stamp}  {_first_user_message(p)}")

    idx = _pick(
        'Resume  ↑/↓ navigate · Enter select · Esc cancel',
        labels,
    )
    if idx is None:
        return

    path = files[idx]
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as e:
        console.print(f"[red]Could not load {path.name}: {e}[/red]")
        return

    messages = [{"role": "system", "content": _build_system_prompt()}] + data
    active_tool_names.clear()
    active_tools[:] = [search_tools]
    _reactivate_tools(data)
    console.print(
        f"[green]Resumed[/green] {path.name} ({len(data)} messages)."
    )


# --- Context status bar ------------------------------------------------------
#
# A status line pinned to the bottom of the screen during model generation,
# using a DECSTBM scroll region so output scrolls above it. During the input
# prompt the same information is shown via prompt_toolkit's bottom_toolbar.

_status_active: bool = False
_SGR = {'green': '32', 'yellow': '33', 'red': '31'}
_TB_TAG = {'green': 'ansigreen', 'yellow': 'ansiyellow', 'red': 'ansired'}


def _term_size() -> tuple:
    """Return (rows, columns) of the terminal."""
    size = shutil.get_terminal_size(fallback=(80, 24))
    return size.lines, size.columns


def _status_fields() -> tuple:
    """Return (text, colour) for the current context occupancy."""
    total = _NUM_CTX or 1
    pct = min(1.0, _ctx_used / total)
    filled = round(pct * 10)
    bar = '█' * filled + '░' * (10 - filled)
    if pct >= COMPACT_AT:
        colour = 'red'
    elif pct >= 0.70:
        colour = 'yellow'
    else:
        colour = 'green'
    text = (
        f" {MODEL.split(':')[0]} · ctx {bar} {int(pct * 100)}%"
        f"  {_ctx_used / 1000:.1f}k/{_NUM_CTX / 1000:.1f}k"
        f" · compact at {int(COMPACT_AT * 100)}%"
    )
    return text, colour


def _status_enable() -> None:
    """Reserve the bottom line via a scroll region and draw the bar."""
    global _status_active
    if not sys.stdout.isatty():
        return
    rows, _ = _term_size()
    sys.stdout.write(
        '\n'                     # ensure a spare line before reserving
        f'\x1b[1;{rows - 1}r'    # scroll region = rows 1..rows-1
        f'\x1b[{rows - 1};1H'    # cursor to the last line of the region
    )
    sys.stdout.flush()
    _status_active = True
    _status_draw()


def _status_draw() -> None:
    """Paint the status line at the bottom row without moving the cursor."""
    if not _status_active:
        return
    rows, cols = _term_size()
    text, colour = _status_fields()
    text = text[:cols].ljust(cols)
    sys.stdout.write(
        '\x1b7'                       # save cursor
        f'\x1b[{rows};1H\x1b[2K'      # go to status row, clear it
        f'\x1b[2;{_SGR[colour]}m{text}\x1b[0m'
        '\x1b8'                       # restore cursor
    )
    sys.stdout.flush()


def _status_disable() -> None:
    """Release the scroll region and clear the status line."""
    global _status_active
    if not _status_active:
        return
    rows, _ = _term_size()
    sys.stdout.write(
        '\x1b7'
        '\x1b[r'                      # reset scroll region to full screen
        f'\x1b[{rows};1H\x1b[2K'      # clear the status line
        '\x1b8'
    )
    sys.stdout.flush()
    _status_active = False


def _sigwinch_handler(signum: int, frame: object) -> None:
    """Re-apply the scroll region and redraw the bar after a resize."""
    if _status_active:
        rows, _ = _term_size()
        sys.stdout.write(f'\x1b[1;{rows - 1}r')
        sys.stdout.flush()
        _status_draw()


def _bottom_toolbar() -> HTML:
    """Return the context status as a prompt_toolkit bottom toolbar."""
    text, colour = _status_fields()
    safe = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return HTML(f'<{_TB_TAG[colour]}>{safe}</{_TB_TAG[colour]}>')


# --- Conversation compaction (hybrid: trim → evict → summarise) --------------

def _msg_role(msg: object) -> str:
    """Return a message's role whether it is a dict or an ollama Message."""
    if isinstance(msg, dict):
        return msg.get('role', '')
    return getattr(msg, 'role', '')


def _msg_content(msg: object) -> str:
    """Return a message's content whether it is a dict or ollama Message."""
    if isinstance(msg, dict):
        return msg.get('content') or ''
    return getattr(msg, 'content', '') or ''


def _group_messages(msgs: list) -> list:
    """Split messages into turn-groups starting at each user message.

    Keeping an assistant message and its tool results in one group means
    compaction never separates a tool call from its result.
    """
    groups: list = []
    current: list = []
    for m in msgs:
        if _msg_role(m) == 'user' and current:
            groups.append(current)
            current = []
        current.append(m)
    if current:
        groups.append(current)
    return groups


def _estimate_tokens(msgs: list) -> int:
    """Rough token estimate (~4 characters per token)."""
    return sum(len(_msg_content(m)) for m in msgs) // 4


def _summarise_groups(groups: list) -> str:
    """Summarise old turn-groups into a compact note using the model."""
    lines = []
    for group in groups:
        for m in group:
            content = _msg_content(m).strip()
            if content:
                lines.append(f"{_msg_role(m)}: {content}")
    transcript = "\n".join(lines)[:12000]
    try:
        resp = ollama.chat(
            model=MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'Summarise the following conversation concisely.'
                        ' Keep facts, decisions, and any URLs or identifiers'
                        ' the user may refer to later. Output only the'
                        ' summary.'
                    ),
                },
                {'role': 'user', 'content': transcript},
            ],
            think=False,
            options={'num_ctx': _NUM_CTX},
        )
        return (resp.message.content or '').strip() or '(summary unavailable)'
    except Exception as e:
        return f'(summary failed: {e})'


def _compact_messages(force: bool = False) -> None:
    """Compact the conversation: trim thinking, evict tool output, summarise.

    Runs the cheap steps first and only calls the model for a summary if the
    conversation is still too large. Preserves the system prompt and the
    most recent turn-groups verbatim.
    """
    global messages
    if len(messages) <= 1:
        return
    system = messages[0]
    # Normalising drops per-message 'thinking' blocks — a free saving.
    history = [_message_to_dict(m) for m in messages[1:]]
    limit = int(_NUM_CTX * COMPACT_AT)

    if not force and _estimate_tokens(history) <= limit:
        messages = [system] + history
        return

    groups = _group_messages(history)
    if len(groups) <= KEEP_RECENT_GROUPS:
        old, recent = [], groups
    else:
        old = groups[:-KEEP_RECENT_GROUPS]
        recent = groups[-KEEP_RECENT_GROUPS:]

    # Evict tool outputs from old groups (biggest, least-needed context).
    for group in old:
        for m in group:
            if m.get('role') == 'tool' and m.get('content'):
                m['content'] = '[old tool output evicted to save context]'

    flat = [m for g in (old + recent) for m in g]
    if not force and _estimate_tokens(flat) <= limit:
        console.print("[dim]\\[COMPACT] evicted old tool outputs[/dim]")
        messages = [system] + flat
        return

    if old:
        summary = _summarise_groups(old)
        note = {
            'role': 'system',
            'content': 'Summary of earlier conversation:\n' + summary,
        }
        recent_flat = [m for g in recent for m in g]
        messages = [system, note] + recent_flat
        console.print(
            "[dim]\\[COMPACT] folded older turns into a summary[/dim]"
        )
    else:
        messages = [system] + flat


def _handle_slash_search(query: str) -> None:
    """Directly invoke web_search and optionally web_fetch for testing."""
    if not _ensure_domain_allowed(SEARCH_BACKEND_DOMAIN):
        console.print(
            f"[red]Denied[/red] access to '{SEARCH_BACKEND_DOMAIN}';"
            " cannot search."
        )
        return
    web_search(query)

    # Collect URLs from the raw DDGS results for the fetch menu
    raw = list(DDGS().text(query, max_results=10))
    scored = sorted(
        raw, key=lambda r: _relevance_score(query, r), reverse=True
    )
    relevant = (
        [r for r in scored if _relevance_score(query, r) > 0][:5]
        or scored[:3]
    )
    urls = [r.get('href') for r in relevant if r.get('href')]

    if not urls:
        return

    console.print("\n[bold]Fetch one of these URLs?[/bold]")
    for i, url in enumerate(urls, 1):
        console.print(f"  [cyan]{i}[/cyan] {url}")
    console.print(
        "  [dim]Enter a number to fetch, or press Enter to skip[/dim]"
    )

    choice = input("> ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(urls):
            content = web_fetch(urls[idx])
            console.print(
                "\n[bold green]--- Page content (first 2000 chars) ---"
                "[/bold green]"
            )
            console.print(content[:2000])
            console.print("[bold green]--- End ---[/bold green]")


def _print_banner() -> None:
    """Print the startup banner and command hints."""
    console.print(f"Using model: [bold]{MODEL}[/bold]")
    console.print("[bold]Qwen Web Agent[/bold]")
    console.print("Type [italic]'exit'[/italic] to quit.")
    console.print(
        "Enter to submit · Shift+Enter for newline"
        " · pasted newlines are safe.\n"
    )
    console.print(
        "[italic]/search <query>[/italic] search · [italic]/models[/italic]"
        " model · [italic]/save[/italic] save · [italic]/resume[/italic]"
        " restore · [italic]/compact[/italic] shrink context.\n"
    )


def _main() -> None:
    """Run the interactive prompt loop."""
    global _ctx_used
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGWINCH, _sigwinch_handler)
    atexit.register(
        lambda: (sys.stdout.write(_RESET_TERMINAL), sys.stdout.flush())
    )
    _print_banner()
    while True:
        try:
            question = _session.prompt(
                '\nYou> ',
                pre_run=_enable_terminal_modes,
                bottom_toolbar=_bottom_toolbar,
            ).strip()
        except KeyboardInterrupt:
            # modifyOtherKeys Ctrl+C reaches here via prompt_toolkit (not
            # SIGINT), so track the press manually for double-Ctrl+C exit.
            now = time.monotonic()
            _ctrl_c_times[:] = [t for t in _ctrl_c_times if now - t <= 1.0]
            _ctrl_c_times.append(now)
            if len(_ctrl_c_times) >= 2:
                console.print("\n[bold red]Exiting.[/bold red]")
                sys.exit(0)
            continue
        except EOFError:
            break      # Ctrl+D exits

        # Restore normal mode so Ctrl+C delivers SIGINT during the agent
        # loop. pre_run=_enable_terminal_modes re-enables it next prompt.
        sys.stdout.write(_RESET_TERMINAL)
        sys.stdout.flush()

        if question.lower() in ["exit", "quit"]:
            break

        if not question:
            continue

        if question in ('/models', '/model'):
            _models_command()
            continue

        if question == '/save':
            _save_conversation()
            continue

        if question == '/resume':
            _resume_command()
            continue

        if question == '/compact':
            _compact_messages(force=True)
            _ctx_used = _estimate_tokens(messages)
            console.print(
                f"[green]Compacted[/green] · ~{_ctx_used:,} tokens"
                f" of {_NUM_CTX:,}."
            )
            continue

        if question.startswith("/search "):
            _handle_slash_search(question[8:].strip())
            continue

        # Display the user question with a styled background.
        console.print()
        console.print(f" {question} ", style="bold white on grey23")
        console.print()

        msg_checkpoint = len(messages)
        messages.append({
            "role": "user",
            "content": question,
        })

        called: set = set()
        nudged: int = 0
        total_prompt_tokens: int = 0
        total_eval_tokens: int = 0

        _status_enable()
        try:
            while True:
                # Non-streaming for tool-call rounds: more reliable tool-call
                # detection; streaming is reserved for the final text answer.
                response = ollama.chat(
                    model=MODEL,
                    messages=messages,
                    think=True,
                    tools=active_tools,
                    options={'num_ctx': _NUM_CTX},
                )

                total_prompt_tokens += (
                    getattr(response, 'prompt_eval_count', 0) or 0)
                total_eval_tokens += (
                    getattr(response, 'eval_count', 0) or 0)
                # Latest prompt_eval_count is the true window occupancy.
                _ctx_used = (
                    getattr(response, 'prompt_eval_count', 0) or _ctx_used)
                _status_draw()

                msg = response.message

                if msg.thinking:
                    console.print("\n[dim]\\[THINKING][/dim]")
                    console.print(f"[dim italic]{msg.thinking}[/dim italic]")

                # markup=False so brackets in the content/tool_calls repr are
                # printed literally; style colours the whole line like the tag.
                console.print(
                    f"[DEBUG] content={msg.content!r}"
                    f" tool_calls={msg.tool_calls}",
                    style="dim yellow",
                    markup=False,
                )

                messages.append(msg)

                if not msg.tool_calls:
                    content = (msg.content or '').strip()
                    if not content and nudged < 1:
                        nudged += 1
                        console.print(
                            "[dim yellow]\\[NUDGE][/dim yellow]"
                            " empty response — retrying"
                        )
                        messages.append({
                            "role": "user",
                            "content": (
                                "Please continue — use search_tools"
                                " to find what you need, then call it."
                            ),
                        })
                        continue
                    console.print("\n[bold green]Qwen>[/bold green]")
                    console.print(Markdown(content))
                    console.print()
                    ctx_str = (
                        f"{_ctx_used:,} / {_NUM_CTX:,}"
                        if _NUM_CTX else
                        f"{_ctx_used:,}"
                    )
                    console.rule(
                        f"[dim]{MODEL}  ·  ctx {ctx_str}  ·  "
                        f"in {total_prompt_tokens:,}"
                        f"  ·  out {total_eval_tokens:,}[/dim]",
                        style="dim",
                    )
                    break

                # Execute requested tools
                for call in msg.tool_calls:

                    name = call.function.name
                    arguments = call.function.arguments

                    # Skip exact duplicates — do not retry identical calls.
                    call_key = (name, tuple(sorted(arguments.items())))
                    if call_key in called:
                        console.print(
                            f"[yellow]\\[SKIP][/yellow]"
                            f" duplicate: {name}({arguments})"
                        )
                        messages.append({
                            "role": "tool",
                            "tool_name": name,
                            "content": (
                                f"Already called {name} with these"
                                " arguments. Use the previous result."
                            ),
                        })
                        continue
                    called.add(call_key)

                    console.print(
                        f"[cyan]\\[TOOL][/cyan] [bold]{name}[/bold]:"
                        f" {arguments}"
                    )

                    if name == "search_tools":
                        result = search_tools(**arguments)
                        for tn in _match_tools(arguments.get("query", "")):
                            if tn not in active_tool_names:
                                active_tool_names.add(tn)
                                active_tools.append(TOOL_REGISTRY[tn]["fn"])
                                console.print(
                                    f"[green]\\[ACTIVATED][/green] {tn}"
                                )
                    elif name in TOOL_REGISTRY:
                        try:
                            result = TOOL_REGISTRY[name]["fn"](**arguments)
                        except Exception as e:
                            result = f"Tool error: {e}"
                    else:
                        result = f"Unknown tool: {name}"

                    messages.append({
                        "role": "tool",
                        "tool_name": name,
                        "content": result,
                    })
        except KeyboardInterrupt:
            console.print(
                "\n[yellow]\\[CANCELLED][/yellow] Response cancelled."
            )
            del messages[msg_checkpoint:]
            sys.stdout.write(_RESET_TERMINAL)
            sys.stdout.flush()
        finally:
            _status_disable()

        # Proactive compaction once the turn's tool calls are all resolved,
        # so we never split a tool call from its result.
        if _NUM_CTX and _ctx_used > COMPACT_AT * _NUM_CTX:
            console.print(
                f"[dim]\\[COMPACT] context {_ctx_used:,}/{_NUM_CTX:,}"
                f" over {int(COMPACT_AT * 100)}% — compacting[/dim]"
            )
            _compact_messages()
            _ctx_used = _estimate_tokens(messages)


if __name__ == '__main__':
    _main()
