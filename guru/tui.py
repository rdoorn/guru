"""Full-screen multi-viewport TUI (Phase B increment 2a — parallel turns).

Each agent viewport owns its own conversation, tools, model, counters, and
output console. Submitting runs that agent's turn in a background thread; the
prompt stays live and you can queue more or switch viewports
(Shift+Left/Right). Ctrl+N spawns a new agent (inherits the current model). A
single Ctrl+C cancels the active agent's running turn (cooperatively); a
double Ctrl+C exits.

Turns run in parallel: each agent binds its own ``session.state`` and console
via contextvars for the duration of its turn, so the adapters — which read and
write ``session`` — operate on that agent's state without a shared lock. Local
Ollama turns against the same model still serialize inside the daemon; remote
adapters (Anthropic/LiteLLM) and mixed models parallelise fully.

Layout, top to bottom: output pane · rule · prompt · rule · status · tabs.
"""
import asyncio
import shutil
import sys
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
from guru.agents import Agent, AgentManager
from guru.domain import tools

_CTX_COLOUR = {'green': 'ansigreen', 'yellow': 'ansiyellow', 'red': 'ansired'}
_CHROME_ROWS = 5   # 2 rules + prompt + status + tabs


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
    left = f"🤖 {model} | 💪 {st.model_size or '?'} | "
    ctx = f"🧠 {int(pct * 100)}% {bar}"
    right = (
        f" | ↓ {st.session_in} | ↑ {st.session_out}"
        f" | 📁 {Path.cwd().name} | 🌿 {st.git_branch or 'none'}"
    )
    return left, ctx, right, colour


def run() -> None:
    ui.refresh_git_branch()
    manager = AgentManager()
    # The main agent adopts the startup session cli already populated (model,
    # adapter, system prompt, context accounting) and may delegate via spawn.
    main = manager.active
    main.state = session.current()
    main.state.can_spawn = True
    if tools.spawn not in main.state.active_tools:
        main.state.active_tools.append(tools.spawn)
    main.append("guru — Ctrl+N new agent · Shift+Left/Right switch"
                " · Ctrl+C cancel · double Ctrl+C exit")

    state = {'loop': None}
    cols = shutil.get_terminal_size((100, 30)).columns

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

    # --- per-agent output ----------------------------------------------------

    def _attach_console(agent) -> None:
        writer = _BufferWriter()
        writer.target = agent
        writer.refresh = _invalidate
        agent.console = Console(
            file=writer, force_terminal=True,
            color_system='standard', width=cols)

    # --- background execution ------------------------------------------------

    def _work(agent) -> None:
        # Bind this agent's state + console to the worker thread's context, so
        # the adapters (which touch ``session``) and tool output target this
        # agent only. No lock: other agents run concurrently on their own.
        token = session.use(agent.state)
        ctoken = ui.use_console(agent.console)
        agent.status = 'thinking'
        try:
            while agent.queue:
                message = agent.queue.pop(0)
                agent.state.messages.append(
                    {'role': 'user', 'content': message})
                session.adapter.run_turn()
        except Exception as e:                           # noqa: BLE001
            agent.append(f"[error] {e}")
        finally:
            agent.console.file.flush()
            ui.reset_console(ctoken)
            session.reset(token)
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

    def _configure(agent, base, can_spawn: bool) -> None:
        """Seed a child agent's state from ``base`` and attach its console."""
        st = agent.state
        st.messages = [
            {'role': 'system', 'content': config.build_system_prompt()}]
        st.active_tools = [tools.search_tools]
        if can_spawn:
            st.active_tools.append(tools.spawn)
        st.active_tool_names = set()
        st.model = base.model
        st.adapter = base.adapter
        st.num_ctx = base.num_ctx
        st.ctx_ceiling = base.ctx_ceiling
        st.model_size = base.model_size
        st.git_branch = base.git_branch
        st.can_spawn = can_spawn
        _attach_console(agent)

    def _new_agent() -> None:
        base = manager.active.state
        agent = manager.add(f"agent{len(manager.agents)}")
        _configure(agent, base, can_spawn=True)
        agent.append(f"[{agent.title}] new agent · model {agent.state.model}")
        manager.active_index = len(manager.agents) - 1

    def _spawn_agent(task: str) -> str:
        """Spawn handler (worker thread): delegate a task in parallel.

        Sub-agents can't spawn further agents. The manager mutation and turn
        launch are scheduled on the loop thread to avoid racing the renderer.
        """
        base = session.current()   # the delegating agent's state
        title = f"agent{len(manager.agents)}"
        agent = Agent(id=title, title=title)
        _configure(agent, base, can_spawn=False)
        agent.append(f"[{title}] spawned · task: {task}")
        agent.append(f"> {task}")
        agent.queue.append(task)
        agent.busy = True

        def _add_and_start() -> None:
            manager.agents.append(agent)
            state['loop'].run_in_executor(None, _work, agent)
            _invalidate()

        state['loop'].call_soon_threadsafe(_add_and_start)
        return (
            f"Spawned {title} to work on this task in parallel. Its progress"
            f" and result appear in the {title} tab, not in your reply."
        )

    tools.set_spawn_handler(_spawn_agent)

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
        # Each agent's live state is mutated in place by its worker thread, so
        # reading the active agent's state here always reflects current values.
        left, ctx, right, colour = _status_from(manager.active.state)
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
            manager.active.state.cancel_requested = True
        else:
            event.current_buffer.text = ''

    # eager=True so these win over the input buffer's default bindings.
    @kb.add('c-n', eager=True)
    def _spawn(event) -> None:
        _new_agent()

    # Shift+Left/Right cycle viewports. Plain Ctrl+Left/Right can't be used:
    # macOS grabs Ctrl+Arrow for Mission Control, so they never reach the
    # terminal (Shift+Arrow sends a distinct CSI 1;2 sequence that does).
    @kb.add('s-right', eager=True)
    def _next(event) -> None:
        manager.switch(1)

    @kb.add('s-left', eager=True)
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

    _attach_console(manager.active)

    async def _amain() -> None:
        state['loop'] = asyncio.get_running_loop()
        await app.run_async()

    asyncio.run(_amain())
