"""Full-screen multi-viewport TUI (Phase B increment 1).

Each agent viewport owns its own conversation, tools, model, and counters.
Submitting runs that agent's turn in a background thread; the prompt stays
live and you can queue more or switch viewports (Ctrl+Left/Right). Ctrl+N
spawns a new agent (inherits the current model). A single Ctrl+C cancels the
running turn (cooperatively); a double Ctrl+C exits.

Turns are serialized by a lock: the adapters share the global ``session``
state, so only one agent's turn runs at a time (its state is swapped in for
the duration). True parallel execution needs async adapters — that's the next
increment. Conversations are already fully independent.

Layout, top to bottom: output pane · rule · prompt · rule · status · tabs.
"""
import asyncio
import shutil
import sys
import threading
from pathlib import Path

from prompt_toolkit import Application
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import HorizontalLine, TextArea
from rich.console import Console

from guru import config, session, ui
from guru.agents import AgentManager
from guru.domain import tools

_CTX_COLOUR = {'green': 'ansigreen', 'yellow': 'ansiyellow', 'red': 'ansired'}
_CHROME_ROWS = 5   # 2 rules + prompt + status + tabs

# Per-agent state fields mirrored between the Agent and the global session.
_STATE_FIELDS = (
    'messages', 'active_tools', 'active_tool_names', 'model', 'adapter',
    'num_ctx', 'ctx_ceiling', 'model_size', 'ctx_used',
    'session_in', 'session_out',
)


class _BufferWriter:
    """File-like sink that routes rich output into an agent's buffer."""

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


def _sync_out(agent) -> None:
    """Copy the global session state into the agent."""
    for f in _STATE_FIELDS:
        setattr(agent, f, getattr(session, f))


def _sync_in(agent) -> None:
    """Load the agent's state into the global session."""
    for f in _STATE_FIELDS:
        setattr(session, f, getattr(agent, f))


def _status_from(agent) -> tuple:
    total = getattr(agent, 'num_ctx', 0) or 1
    used = getattr(agent, 'ctx_used', 0) or 0
    pct = min(1.0, used / total)
    filled = round(pct * 10)
    bar = '█' * filled + '░' * (10 - filled)
    if pct >= config.COMPACT_AT:
        colour = 'red'
    elif pct >= 0.70:
        colour = 'yellow'
    else:
        colour = 'green'
    model = (getattr(agent, 'model', '') or '?').split(':')[0]
    left = f"🤖 {model} | 💪 {getattr(agent, 'model_size', '?')} | "
    ctx = f"🧠 {int(pct * 100)}% {bar}"
    right = (
        f" | ↓ {getattr(agent, 'session_in', 0)}"
        f" | ↑ {getattr(agent, 'session_out', 0)}"
        f" | 📁 {Path.cwd().name} | 🌿 {session.git_branch or 'none'}"
    )
    return left, ctx, right, colour


def run() -> None:
    ui.refresh_git_branch()
    manager = AgentManager()
    manager.active.append("guru — Ctrl+N new agent · Ctrl+Left/Right switch"
                          " · Ctrl+C cancel · double Ctrl+C exit")
    _sync_out(manager.active)   # seed the main agent from the startup session

    state = {'loop': None, 'running': None}
    lock = threading.Lock()
    cols = shutil.get_terminal_size((100, 30)).columns
    writer = _BufferWriter()
    ui.console = Console(
        file=writer, force_terminal=True,
        color_system='standard', width=cols)

    def _ask_terminal(domain: str) -> bool:
        sys.stdout.write(f"\nAllow access to '{domain}'? [y/N] ")
        sys.stdout.flush()
        try:
            return input().strip().lower() in ('y', 'yes')
        except (EOFError, KeyboardInterrupt):
            return False

    def _domain_asker(domain: str) -> bool:
        fut = asyncio.run_coroutine_threadsafe(
            run_in_terminal(lambda: _ask_terminal(domain)), state['loop'])
        try:
            return bool(fut.result())
        except Exception:
            return False

    tools.set_domain_asker(_domain_asker)

    # --- background execution ------------------------------------------------

    def _work(agent) -> None:
        with lock:
            state['running'] = agent
            _sync_in(agent)
            writer.target = agent
            agent.status = 'thinking'
            try:
                while agent.queue:
                    message = agent.queue.pop(0)
                    session.messages.append(
                        {'role': 'user', 'content': message})
                    session.adapter.run_turn()
            except Exception as e:                           # noqa: BLE001
                agent.append(f"[error] {e}")
            finally:
                _sync_out(agent)
                state['running'] = None
                writer.flush()
                agent.status = 'idle'
                agent.busy = False
                _invalidate()

    def _submit(text: str) -> None:
        agent = manager.active
        agent.append(f"> {text}")
        agent.queue.append(text)
        if not agent.busy:
            agent.busy = True
            state['loop'].run_in_executor(None, _work, agent)

    def _new_agent() -> None:
        base = manager.active
        agent = manager.add(f"agent{len(manager.agents)}")
        agent.messages = [
            {'role': 'system', 'content': config.build_system_prompt()}]
        agent.active_tools = [tools.search_tools]
        agent.active_tool_names = set()
        agent.model = base.model
        agent.adapter = base.adapter
        agent.num_ctx = base.num_ctx
        agent.ctx_ceiling = base.ctx_ceiling
        agent.model_size = base.model_size
        agent.ctx_used = 0
        agent.session_in = 0
        agent.session_out = 0
        agent.append(f"[{agent.title}] new agent · model {agent.model}")
        manager.active_index = len(manager.agents) - 1

    # --- views ---------------------------------------------------------------

    def _output():
        rows = shutil.get_terminal_size((100, 30)).lines
        visible = max(1, rows - _CHROME_ROWS)
        return ANSI('\n'.join(manager.active.lines[-visible:]))

    output = Window(
        FormattedTextControl(_output),
        wrap_lines=True, height=Dimension(weight=1))

    def _tabs() -> FormattedText:
        parts: list = []
        for active, title in manager.tabs():
            parts.append(
                ('reverse' if active else 'ansibrightblack', f'[{title}]'))
            parts.append(('', ' '))
        return FormattedText(parts)

    tabline = Window(FormattedTextControl(_tabs), height=1)

    def _status() -> FormattedText:
        # The running agent's live state lives in `session`; idle agents keep
        # their own snapshot.
        active = manager.active
        src = session if state.get('running') is active else active
        left, ctx, right, colour = _status_from(src)
        return FormattedText([
            ('ansibrightblack', left),
            (_CTX_COLOUR[colour], ctx),
            ('ansibrightblack', right),
        ])

    statusline = Window(FormattedTextControl(_status), height=1)

    input_area = TextArea(height=1, prompt='> ', multiline=False)

    def _accept(buff) -> bool:
        text = buff.text.strip()
        if text:
            _submit(text)
        return False

    input_area.accept_handler = _accept

    # --- keys ----------------------------------------------------------------

    kb = KeyBindings()

    @kb.add('c-d')
    def _quit(event) -> None:
        event.app.exit()

    @kb.add('c-c')
    def _ctrl_c(event) -> None:
        if ui.note_ctrl_c():
            event.app.exit()
        elif manager.active.busy:
            session.cancel_requested = True
        else:
            event.current_buffer.text = ''

    @kb.add('c-n')
    def _spawn(event) -> None:
        _new_agent()

    @kb.add('c-right')
    def _next(event) -> None:
        manager.switch(1)

    @kb.add('c-left')
    def _prev(event) -> None:
        manager.switch(-1)

    # --- run -----------------------------------------------------------------

    root = HSplit([
        output,
        HorizontalLine(),
        input_area,
        HorizontalLine(),
        statusline,
        tabline,
    ])
    app = Application(
        layout=Layout(root, focused_element=input_area),
        key_bindings=kb,
        full_screen=True,
        mouse_support=False,
    )

    def _invalidate() -> None:
        app.invalidate()   # thread-safe in prompt_toolkit

    writer.refresh = _invalidate

    async def _amain() -> None:
        state['loop'] = asyncio.get_running_loop()
        await app.run_async()

    asyncio.run(_amain())
