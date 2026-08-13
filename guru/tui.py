"""Full-screen multi-viewport TUI (Phase A.2).

Submitting a message runs the agent's turn in a background thread (off the UI
loop) so the prompt stays live. Adapter output (rich) is captured into the
active agent's buffer and shown live. A single Ctrl+C cancels the running turn
(cooperatively, between rounds); a double Ctrl+C exits. The domain-approval
prompt runs via ``run_in_terminal`` so it works inside the full-screen app.

Layout, top to bottom: output pane · rule · prompt · rule · status · tabs.

Phase B (multiple agents fully wired with per-agent conversation state + a
spawn/delegate tool) builds on this.
"""
import asyncio
import shutil
import sys

from prompt_toolkit import Application
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import HorizontalLine, TextArea
from rich.console import Console

from guru import session, ui
from guru.agents import AgentManager
from guru.domain import tools

_CTX_COLOUR = {'green': 'ansigreen', 'yellow': 'ansiyellow', 'red': 'ansired'}
# Non-output rows: 2 rules + prompt + status + tabs.
_CHROME_ROWS = 5


class _BufferWriter:
    """File-like sink that routes rich output into an agent's buffer."""

    def __init__(self) -> None:
        self.target = None          # the Agent currently receiving output
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


def run() -> None:
    manager = AgentManager()
    manager.active.append("guru — type below · Ctrl+Left/Right switch"
                          " viewports · Ctrl+C cancel · double Ctrl+C exit")

    state = {'loop': None}
    cols = shutil.get_terminal_size((100, 30)).columns
    writer = _BufferWriter()
    # Redirect all adapter/tool output into the active agent's buffer.
    ui.console = Console(
        file=writer, force_terminal=True,
        color_system='standard', width=cols)

    # Domain approval inside the full-screen app (temporarily leave it).
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
        try:
            while agent.queue:
                message = agent.queue.pop(0)
                writer.target = agent
                agent.status = 'thinking'
                session.messages.append(
                    {'role': 'user', 'content': message})
                try:
                    session.adapter.run_turn()
                except Exception as e:                       # noqa: BLE001
                    agent.append(f"[error] {e}")
        finally:
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
        left, ctx, right, colour = ui._status_parts()
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
        return False   # clear the input

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
            session.cancel_requested = True   # cooperative cancel
        else:
            event.current_buffer.text = ''

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
