import argparse
import atexit
import os
import signal
import sys
import time
import ollama
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
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

_parser = argparse.ArgumentParser(description="guru — local Ollama agent")
_parser.add_argument("--model", default="qwen3-abliterated-32k:latest")
_args = _parser.parse_args()

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

console = Console(highlight=False)

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


signal.signal(signal.SIGINT, _sigint_handler)

MODEL = _args.model

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
    console.print(f"\n[cyan]\\[SEARCH][/cyan] {query}")

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
        console.print(f"  [bold]{i}. {result.get('title')}[/bold]")
        console.print(f"  [dim]URL: {result.get('href')}[/dim]")
        console.print(f"  {result.get('body')}\n")

    return "\n".join(output)


def web_fetch(url: str) -> str:
    """
    Fetch and read the text content of a webpage.
    Use this after web_search when you need more information from a result.
    """
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


messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]

# Fetch context window size once at startup for the stats footer.
try:
    _model_info = ollama.show(MODEL)
    _ctx_size: int = getattr(_model_info.details, 'context_length', 0) or 0
except Exception:
    _ctx_size = 0

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

atexit.register(
    lambda: (sys.stdout.write(_RESET_TERMINAL), sys.stdout.flush())
)

console.print(f"Using model: [bold]{MODEL}[/bold]")
console.print("[bold]Qwen Web Agent[/bold]")
console.print("Type [italic]'exit'[/italic] to quit.")
console.print(
    "Enter to submit · Shift+Enter for newline"
    " · pasted newlines are safe.\n"
)
console.print(
    "Type [italic]'/search <query>'[/italic] to search"
    " · [italic]'/models'[/italic] to switch model.\n"
)


def _models_command() -> None:
    """Interactive model selector — ↑/↓ to navigate, Enter to select."""
    global MODEL, _ctx_size

    try:
        model_list = ollama.list().models
    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        return

    names = sorted(m.model for m in model_list)
    if not names:
        console.print("[yellow]No Ollama models found.[/yellow]")
        return

    state = {
        'idx': next(
            (i for i, n in enumerate(names) if n == MODEL), 0
        ),
    }

    def _get_text() -> FormattedText:
        lines: list = [
            ('bold', ' Models  ↑/↓ navigate · Enter select · Esc cancel\n\n'),
        ]
        for i, name in enumerate(names):
            cursor = i == state['idx']
            active = name == MODEL
            prefix = '▶ ' if cursor else '  '
            suffix = ' ✓' if active else ''
            if cursor and active:
                style = 'class:cursor-active'
            elif cursor:
                style = 'class:cursor'
            elif active:
                style = 'class:active'
            else:
                style = ''
            lines.append((style, f'  {prefix}{name}{suffix}\n'))
        return FormattedText(lines)

    kb = KeyBindings()

    @kb.add('up')
    def _up(event: object) -> None:
        state['idx'] = (state['idx'] - 1) % len(names)

    @kb.add('down')
    def _down(event: object) -> None:
        state['idx'] = (state['idx'] + 1) % len(names)

    @kb.add('enter')
    def _select(event: object) -> None:
        event.app.exit(result=names[state['idx']])

    @kb.add('escape')
    @kb.add('c-c')
    def _cancel(event: object) -> None:
        event.app.exit(result=None)

    model_style = Style.from_dict({
        'cursor': 'bold ansicyan',
        'cursor-active': 'bold ansigreen',
        'active': 'ansigreen',
    })

    app = Application(
        layout=Layout(
            Window(
                FormattedTextControl(_get_text, focusable=True),
                height=len(names) + 2,
            )
        ),
        key_bindings=kb,
        style=model_style,
        full_screen=False,
        mouse_support=False,
    )

    selected: str | None = app.run()

    if selected is None:
        return
    MODEL = selected
    try:
        info = ollama.show(MODEL)
        _ctx_size = getattr(info.details, 'context_length', 0) or 0
    except Exception:
        _ctx_size = 0
    console.print(f"\n[green]Model:[/green] [bold]{MODEL}[/bold]")


def _handle_slash_search(query: str) -> None:
    """Directly invoke web_search and optionally web_fetch for testing."""
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


while True:
    try:
        question = _session.prompt(
            '\nYou> ', pre_run=_enable_terminal_modes
        ).strip()
    except KeyboardInterrupt:
        # modifyOtherKeys Ctrl+C reaches here via prompt_toolkit (not SIGINT),
        # so track the press manually for double-Ctrl+C exit.
        now = time.monotonic()
        _ctrl_c_times[:] = [t for t in _ctrl_c_times if now - t <= 1.0]
        _ctrl_c_times.append(now)
        if len(_ctrl_c_times) >= 2:
            console.print("\n[bold red]Exiting.[/bold red]")
            sys.exit(0)
        continue
    except EOFError:
        break      # Ctrl+D exits

    # Restore normal mode so Ctrl+C delivers SIGINT during the agent loop.
    # pre_run=_enable_terminal_modes re-enables both before the next prompt.
    sys.stdout.write(_RESET_TERMINAL)
    sys.stdout.flush()

    if question.lower() in ["exit", "quit"]:
        break

    if not question:
        continue

    if question in ('/models', '/model'):
        _models_command()
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

    try:
        while True:
            # Non-streaming for tool-call rounds: more reliable tool-call
            # detection; streaming is reserved for the final text answer.
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                think=True,
                tools=active_tools,
            )

            total_prompt_tokens += (
                getattr(response, 'prompt_eval_count', 0) or 0)
            total_eval_tokens += (
                getattr(response, 'eval_count', 0) or 0)

            msg = response.message

            if msg.thinking:
                console.print("\n[dim]\\[THINKING][/dim]")
                console.print(f"[dim italic]{msg.thinking}[/dim italic]")

            console.print(
                f"[dim yellow]\\[DEBUG][/dim yellow]"
                f" content={repr(msg.content)}"
                f" tool_calls={msg.tool_calls}",
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
                    f"{total_prompt_tokens:,} / {_ctx_size:,}"
                    if _ctx_size else
                    f"{total_prompt_tokens:,}"
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
                            f"Already called {name} with these arguments. "
                            "Use the result from the previous call."
                        ),
                    })
                    continue
                called.add(call_key)

                console.print(
                    f"[cyan]\\[TOOL][/cyan] [bold]{name}[/bold]: {arguments}"
                )

                if name == "search_tools":
                    result = search_tools(**arguments)
                    for tool_name in _match_tools(arguments.get("query", "")):
                        if tool_name not in active_tool_names:
                            active_tool_names.add(tool_name)
                            active_tools.append(TOOL_REGISTRY[tool_name]["fn"])
                            console.print(
                                f"[green]\\[ACTIVATED][/green]"
                                f" {tool_name}"
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
        console.print("\n[yellow]\\[CANCELLED][/yellow] Response cancelled.")
        del messages[msg_checkpoint:]
        sys.stdout.write(_RESET_TERMINAL)
        sys.stdout.flush()
