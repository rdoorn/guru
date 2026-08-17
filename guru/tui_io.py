"""TUI output plumbing and status formatting, split out of ``guru.tui``.

These pieces carry no closure state from the interactive ``run()`` loop, so
they live here to keep ``tui.py`` focused on the coordinator itself:

* ``_BufferWriter`` / ``_MainWriter`` — file-like sinks a rich Console writes
  into, routing a sub-agent's output to its viewport buffer and the main
  agent's output to the normal terminal buffer.
* ``_app_cols`` — the true render width under prompt_toolkit.
* ``_status_from`` — the status-bar tuple (left, ctx bar, right, colour).
"""
import shutil
import sys
import threading
from pathlib import Path

from prompt_toolkit.application import get_app

from guru import config, log
from guru.domain import conversation


class _BufferWriter:
    """File-like sink that routes rich output into a sub-agent's buffer."""

    def __init__(self) -> None:
        self.target = None
        self.refresh = lambda: None
        self._partial = ''

    def write(self, text: str) -> None:
        if self.target is None:
            return
        self._partial += text
        while '\n' in self._partial:
            line, self._partial = self._partial.split('\n', 1)
            self.target.append(line)
        self.refresh()

    def flush(self) -> None:
        if self.target is not None and self._partial:
            self.target.append(self._partial)
            self._partial = ''
            self.refresh()


class _MainWriter:
    """Routes the main agent's output to the normal terminal buffer.

    Lines stream to stdout (above the live prompt via patch_stdout) while the
    [main] view is on screen; while the sub-agent TUI is showing they are held
    and flushed on return. Every line is also mirrored into the agent buffer.
    """

    def __init__(self, agent, state) -> None:
        self.agent = agent
        self.state = state
        self._partial = ''
        self._pending: list = []
        self._lock = threading.Lock()

    def _emit(self, line: str) -> None:
        self.agent.append(line)
        # While [main] is on screen, write to the terminal — patch_stdout
        # (active during the prompt) lifts it above the live input so main's
        # turn streams in sync. While the viewer (alt screen) is up, hold the
        # line and flush it on return.
        if self.state.get('view') == 'main':
            try:
                sys.stdout.write(line + '\n')
                sys.stdout.flush()
                return
            except Exception:                            # noqa: BLE001
                log.exc('stdout emit failed')
                pass
        self._pending.append(line)

    def write(self, text: str) -> None:
        with self._lock:
            self._partial += text
            while '\n' in self._partial:
                line, self._partial = self._partial.split('\n', 1)
                self._emit(line)

    def flush(self) -> None:
        with self._lock:
            if self._partial:
                self._emit(self._partial)
                self._partial = ''

    def drain(self) -> None:
        """Flush lines held while the TUI was showing (call in [main] view)."""
        with self._lock:
            for line in self._pending:
                try:
                    sys.stdout.write(line + '\n')
                except Exception:                        # noqa: BLE001
                    log.exc('drain write failed')
                    pass
            sys.stdout.flush()
            self._pending.clear()


def _app_cols() -> int:
    """Full render width from the active app (accurate under prompt_toolkit;
    shutil returns the fallback size while a prompt owns the terminal)."""
    try:
        return get_app().output.get_size().columns
    except Exception:                                    # noqa: BLE001
        log.exc('app cols read failed')
        return shutil.get_terminal_size((100, 30)).columns


def _status_from(st) -> tuple:
    """Build the status tuple (left, ctx, right, colour) from state."""
    total = st.num_ctx or 1
    used = st.ctx_used or 0
    pct = min(1.0, used / total)
    filled = round(pct * 10)
    bar = '█' * filled + '░' * (10 - filled)
    if pct >= config.COMPACT_AT:
        colour = 'red'
    elif pct >= 0.70:
        colour = 'yellow'
    else:
        colour = 'green'
    model = (st.model or '?').split(':')[0]
    mode = {config.MODE_READ_ONLY: 'read-only',
            config.MODE_ASK: 'ask', config.MODE_AUTO: 'auto'}.get(
        config.MODE, config.MODE)
    rs = st.active_role or 'general'
    if st.active_skill:
        rs += f"/{st.active_skill}"
    left = (f"🤖 {model} | 💪 {st.model_size or '?'} | 🔐 {mode}"
            f" | 🎭 {rs} | ")
    ctx = f"🧠 {int(pct * 100)}% {bar}"
    # Composition of the resident context in tokens (rough, ~4 chars/token).
    bd = conversation.context_breakdown(
        st.messages, st.active_tool_names, st.can_spawn)

    def _kt(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    comp = (f"📊 sys:{_kt(bd['sys'])} tl:{_kt(bd['tools'])}"
            f" in:{_kt(bd['in'])} out:{_kt(bd['out'])}")
    if bd['toolout']:
        comp += f" res:{_kt(bd['toolout'])}"
    right = (
        f" · {comp}"
        f" | ↓ {st.session_in} | ↑ {st.session_out}"
        f" | 📁 {Path.cwd().name} | 🌿 {st.git_branch or 'none'}"
    )
    return left, ctx, right, colour
