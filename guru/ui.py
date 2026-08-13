"""Terminal UI: console, key bindings, the model picker, and the status bar.

Rendering and terminal control live here so the domain and adapters stay
free of prompt_toolkit / rich / escape-sequence concerns.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

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
from rich.theme import Theme

from guru import config, session

# Teach prompt_toolkit to recognise Shift+Enter and keyboard Enter under
# modifyOtherKeys. See the key-binding notes below for why F13/F14 are used.
ANSI_SEQUENCES['\x1b[13;2u'] = Keys.F13      # CSI u Shift+Enter
ANSI_SEQUENCES['\x1b[27;2;13~'] = Keys.F13   # xterm modifyOtherKeys
ANSI_SEQUENCES['\x1b[99;5u'] = Keys.ControlC     # CSI u Ctrl+C
ANSI_SEQUENCES['\x1b[27;5;99~'] = Keys.ControlC  # xterm modifyOtherKeys Ctrl+C
ANSI_SEQUENCES['\x1b[27;1;13~'] = Keys.F14   # keyboard Enter, modifier=1
ANSI_SEQUENCES['\x1b[27;0;13~'] = Keys.F14   # keyboard Enter, modifier=0
ANSI_SEQUENCES['\x1b[13;1u'] = Keys.F14      # CSI u keyboard Enter
ANSI_SEQUENCES['\x1b[13u'] = Keys.F14        # CSI u short form

# Override Rich's markdown link styles: bright blue, no underline.
console = Console(
    highlight=False,
    theme=Theme({
        'markdown.link': 'bright_blue',
        'markdown.link_url': 'bright_blue',
    }),
)

_RESET_TERMINAL = '\x1b[>4;0m\x1b[?2004l'
_ctrl_c_times: list = []

_SGR = {'green': '32', 'yellow': '33', 'red': '31'}
_TB_TAG = {'green': 'ansigreen', 'yellow': 'ansiyellow', 'red': 'ansired'}
_GRAY_SGR = '90'
_GRAY_TAG = 'ansibrightblack'
_status_active = False

_SELECT_STYLE = Style.from_dict({
    'cursor': 'bold ansicyan',
    'cursor-active': 'bold ansigreen',
    'active': 'ansigreen',
    'warn': 'ansired',
})
# prompt_toolkit's default bottom-toolbar style is reverse-video; clear it.
_TOOLBAR_STYLE = Style.from_dict({'bottom-toolbar': 'noreverse bg:default'})


# --- Ctrl+C handling ---------------------------------------------------------

def sigint_handler(signum: int, frame: object) -> None:
    """Exit on double Ctrl+C within 1 s; otherwise cancel current operation."""
    now = time.monotonic()
    _ctrl_c_times[:] = [t for t in _ctrl_c_times if now - t <= 1.0]
    _ctrl_c_times.append(now)
    if len(_ctrl_c_times) >= 2:
        console.print("\n[bold red]Exiting.[/bold red]")
        sys.exit(0)
    raise KeyboardInterrupt


def note_ctrl_c() -> bool:
    """Record a Ctrl+C from the prompt; return True if it should exit now."""
    now = time.monotonic()
    _ctrl_c_times[:] = [t for t in _ctrl_c_times if now - t <= 1.0]
    _ctrl_c_times.append(now)
    return len(_ctrl_c_times) >= 2


# --- Key bindings ------------------------------------------------------------
#
# The terminal encodes pasted newlines as \r — byte-identical to keyboard
# Enter — so the two differ only by CONTEXT: a paste feeds every byte into
# prompt_toolkit's input queue at once, so when the handler for the paste's
# first \r fires, later pasted keys are still queued. A deliberate Enter is
# the last key, leaving the queue empty. F14 (Enter re-encoded by
# modifyOtherKeys) and Escape+Enter always submit; Shift+Enter and c-j insert.

_kb = KeyBindings()


@_kb.add('f13')
def _shift_enter(event: object) -> None:
    event.current_buffer.insert_text('\n')


@_kb.add('f14')
def _mk_enter(event: object) -> None:
    event.current_buffer.validate_and_handle()


@_kb.add('escape', 'enter')
def _escape_enter(event: object) -> None:
    event.current_buffer.validate_and_handle()


@_kb.add('enter')
def _enter(event: object) -> None:
    if event.app.key_processor.input_queue:
        event.current_buffer.insert_text('\n')
    else:
        event.current_buffer.validate_and_handle()


@_kb.add('c-j')
def _linefeed(event: object) -> None:
    event.current_buffer.insert_text('\n')


prompt_session = PromptSession(
    history=FileHistory(str(config.GURU_HOME / 'history')),
    multiline=True,
    key_bindings=_kb,
)


def enable_terminal_modes() -> None:
    """Enable modifyOtherKeys mode 2 and bracketed paste before each prompt."""
    sys.stdout.write('\x1b[>4;2m\x1b[?2004h')
    sys.stdout.flush()


def reset_terminal() -> None:
    """Restore normal terminal modes."""
    sys.stdout.write(_RESET_TERMINAL)
    sys.stdout.flush()


def read_line(prompt: str = '> ') -> str:
    """Read one line, showing the status bar as the prompt toolbar."""
    return prompt_session.prompt(
        prompt,
        pre_run=enable_terminal_modes,
        bottom_toolbar=_bottom_toolbar,
        style=_TOOLBAR_STYLE,
    ).strip()


# --- Model / list picker -----------------------------------------------------

def pick(title: str, options: list, active_idx: int = -1,
         selectable=None, row_styles=None):
    """Arrow-key selector. Returns the chosen index, or None if cancelled.

    ``options`` are display strings. ``selectable`` is an optional list of
    booleans (same length) marking which rows can be chosen — non-selectable
    rows (e.g. group headers) are skipped by the cursor and not highlightable.
    ``row_styles`` optionally gives a per-row style class (e.g. 'class:warn')
    applied when a selectable row is neither the cursor nor active.
    """
    if not options:
        return None
    if selectable is None:
        selectable = [True] * len(options)
    if row_styles is None:
        row_styles = [''] * len(options)

    def _first_selectable(start: int, step: int) -> int:
        i = start
        for _ in range(len(options)):
            if selectable[i]:
                return i
            i = (i + step) % len(options)
        return start

    start = active_idx if active_idx >= 0 else 0
    if not selectable[start]:
        start = _first_selectable(start, 1)
    state = {'idx': start}

    def _text() -> FormattedText:
        lines: list = [('bold', f' {title}\n\n')]
        for i, opt in enumerate(options):
            if not selectable[i]:
                lines.append(('bold', f' {opt}\n'))
                continue
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
                style = row_styles[i]
            lines.append((style, f'  {prefix}{opt}{suffix}\n'))
        return FormattedText(lines)

    kb = KeyBindings()

    @kb.add('up')
    def _up(event: object) -> None:
        state['idx'] = _first_selectable(
            (state['idx'] - 1) % len(options), -1)

    @kb.add('down')
    def _down(event: object) -> None:
        state['idx'] = _first_selectable(
            (state['idx'] + 1) % len(options), 1)

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
                height=len(options) + 3,
            )
        ),
        key_bindings=kb,
        style=_SELECT_STYLE,
        full_screen=False,
        mouse_support=False,
    )
    return app.run()


def pick_multi(title: str, options: list, states: list):
    """Checkbox multi-select. Space toggles, Enter confirms, Esc cancels.

    Returns the updated list of booleans, or None if cancelled.
    """
    if not options:
        return None
    states = list(states)
    state = {'idx': 0}

    def _text() -> FormattedText:
        lines: list = [
            ('bold', f' {title}\n'),
            ('', ' Space toggles · Enter confirms · Esc cancels\n\n'),
        ]
        for i, opt in enumerate(options):
            box = '[x]' if states[i] else '[ ]'
            cursor = i == state['idx']
            prefix = '▶ ' if cursor else '  '
            if cursor:
                style = 'class:cursor'
            elif states[i]:
                style = 'class:active'
            else:
                style = ''
            lines.append((style, f'  {prefix}{box} {opt}\n'))
        return FormattedText(lines)

    kb = KeyBindings()

    @kb.add('up')
    def _up(event: object) -> None:
        state['idx'] = (state['idx'] - 1) % len(options)

    @kb.add('down')
    def _down(event: object) -> None:
        state['idx'] = (state['idx'] + 1) % len(options)

    @kb.add('space')
    def _toggle(event: object) -> None:
        states[state['idx']] = not states[state['idx']]

    @kb.add('enter')
    def _confirm(event: object) -> None:
        event.app.exit(result=states)

    @kb.add('escape')
    @kb.add('c-c')
    def _cancel(event: object) -> None:
        event.app.exit(result=None)

    app = Application(
        layout=Layout(
            Window(
                FormattedTextControl(_text, focusable=True),
                height=len(options) + 4,
            )
        ),
        key_bindings=kb,
        style=_SELECT_STYLE,
        full_screen=False,
        mouse_support=False,
    )
    return app.run()


# --- Status bar --------------------------------------------------------------

def _term_size() -> tuple:
    size = shutil.get_terminal_size(fallback=(80, 24))
    return size.lines, size.columns


def total_memory_bytes() -> int:
    """Return total physical memory in bytes, or 0 if it can't be read."""
    try:
        return (os.sysconf('SC_PAGE_SIZE')
                * os.sysconf('SC_PHYS_PAGES'))
    except (ValueError, OSError, AttributeError):
        pass
    try:                                   # macOS fallback
        out = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True, text=True, timeout=1)
        return int(out.stdout.strip())
    except Exception:
        return 0


def format_bytes(num: int) -> str:
    """Human-readable size, e.g. '8.2 GB'."""
    value = float(num)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if value < 1024 or unit == 'TB':
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def refresh_git_branch() -> None:
    """Cache the current git branch (or None if not a repo)."""
    try:
        proc = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=1, cwd=str(Path.cwd()),
        )
        branch = proc.stdout.strip()
        session.git_branch = (
            branch if proc.returncode == 0 and branch else None)
    except Exception:
        session.git_branch = None


def _status_parts() -> tuple:
    """Return (left, ctx_segment, right, colour) for the status line."""
    total = session.num_ctx or 1
    pct = min(1.0, session.ctx_used / total)
    filled = round(pct * 10)
    bar = '█' * filled + '░' * (10 - filled)
    if pct >= config.COMPACT_AT:
        colour = 'red'
    elif pct >= 0.70:
        colour = 'yellow'
    else:
        colour = 'green'
    model = (session.model or '?').split(':')[0]
    left = f"🤖 {model} | 💪 {session.model_size} | "
    ctx_segment = f"🧠 {int(pct * 100)}% {bar}"
    right = (
        f" | ↓ {session.session_in} | ↑ {session.session_out}"
        f" | 📁 {Path.cwd().name} | 🌿 {session.git_branch or 'none'}"
    )
    return left, ctx_segment, right, colour


# The status bar is two pinned lines: a full-width rule, then the status.
# The rule stays fixed above the status (and, at the prompt, just below the
# input line), while the per-turn console.rule() above the prompt scrolls away.
_STATUS_HEIGHT = 2


def status_enable() -> None:
    """Reserve the bottom two lines via a scroll region and draw the bar."""
    global _status_active
    if not sys.stdout.isatty():
        return
    rows, _ = _term_size()
    top = rows - _STATUS_HEIGHT
    sys.stdout.write(
        '\n'
        f'\x1b[1;{top}r'      # scroll region = rows 1..top
        f'\x1b[{top};1H'      # cursor to the last line of the region
    )
    sys.stdout.flush()
    _status_active = True
    status_draw()


def _status_body(cols: int) -> str:
    """ANSI-coloured status text truncated to the terminal width."""
    left, ctx_segment, right, colour = _status_parts()
    plain = left + ctx_segment + right
    if len(plain) > cols:
        return plain[:cols]
    return (
        f'\x1b[{_GRAY_SGR}m{left}\x1b[0m'
        f'\x1b[{_SGR[colour]}m{ctx_segment}\x1b[0m'
        f'\x1b[{_GRAY_SGR}m{right}\x1b[0m'
    )


def status_draw() -> None:
    """Paint the rule + status on the bottom two rows, keeping the cursor."""
    if not _status_active:
        return
    rows, cols = _term_size()
    rule = '─' * cols
    body = _status_body(cols)
    sys.stdout.write(
        '\x1b7'                                       # save cursor
        f'\x1b[{rows - 1};1H\x1b[2K'                  # rule row
        f'\x1b[{_GRAY_SGR}m{rule}\x1b[0m'
        f'\x1b[{rows};1H\x1b[2K{body}'                # status row
        '\x1b8'                                        # restore cursor
    )
    sys.stdout.flush()


def status_disable() -> None:
    """Release the scroll region and clear the status rows."""
    global _status_active
    if not _status_active:
        return
    rows, _ = _term_size()
    sys.stdout.write(
        '\x1b7'
        '\x1b[r'
        f'\x1b[{rows - 1};1H\x1b[2K'
        f'\x1b[{rows};1H\x1b[2K'
        '\x1b8'
    )
    sys.stdout.flush()
    _status_active = False


def sigwinch_handler(signum: int, frame: object) -> None:
    """Re-apply the scroll region and redraw the bar after a resize."""
    if _status_active:
        rows, _ = _term_size()
        sys.stdout.write(f'\x1b[1;{rows - _STATUS_HEIGHT}r')
        sys.stdout.flush()
        status_draw()


def note_thinking() -> None:
    """Show that the agent is working (before a blocking model call)."""
    console.print("[dim]* thinking…[/dim]")


def note_tool(name: str, arg: str) -> None:
    """Show a tool invocation, e.g. '* web_search kubernetes version'."""
    text = f"* {name} {arg}".rstrip()
    console.print(text, style="cyan", markup=False)


def debug(text: str) -> None:
    """Print a debug line only when GURU_DEBUG is set in the environment."""
    if os.environ.get('GURU_DEBUG'):
        console.print(f"[DEBUG] {text}", style="dim yellow", markup=False)


def _tb_escape(text: str) -> str:
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _bottom_toolbar() -> HTML:
    """Two-line prompt toolbar: a fixed rule, then the status line."""
    _, cols = _term_size()
    left, ctx_segment, right, colour = _status_parts()
    rule = '─' * cols
    return HTML(
        f'<{_GRAY_TAG}>{rule}</{_GRAY_TAG}>\n'
        f'<{_GRAY_TAG}>{_tb_escape(left)}</{_GRAY_TAG}>'
        f'<{_TB_TAG[colour]}>{_tb_escape(ctx_segment)}</{_TB_TAG[colour]}>'
        f'<{_GRAY_TAG}>{_tb_escape(right)}</{_GRAY_TAG}>'
    )
