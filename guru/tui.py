"""Full-screen multi-viewport TUI shell (Phase A, increment 1).

A runnable skeleton: a scrollable output pane bound to the active agent's
buffer, a tab line of agents, the context status line, and a live input box.
Enter appends to the active agent's buffer; Ctrl+Left/Right switch viewports;
Ctrl+D exits.

Agent execution (running a turn in the background and streaming its output
into the buffer) is wired in the next increment — the goal here is to validate
that the shell layout, key routing, and rendering work on the real terminal
before the concurrency lands.
"""
from prompt_toolkit import Application
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import HorizontalLine, TextArea

from guru import ui
from guru.agents import AgentManager

_CTX_COLOUR = {'green': 'ansigreen', 'yellow': 'ansiyellow', 'red': 'ansired'}


def run() -> None:
    manager = AgentManager()
    manager.active.append("[guru TUI — increment 1: shell only]")
    manager.active.append("Type below · Ctrl+Left/Right switch · Ctrl+D exit")

    def _output():
        return ANSI(manager.active.text)

    output = Window(
        FormattedTextControl(_output),
        wrap_lines=True,
        height=Dimension(weight=1),
    )

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
            manager.active.append(f"> {text}")
            manager.active.append("(agent execution wired in the next step)")
        return False   # clear the input

    input_area.accept_handler = _accept

    kb = KeyBindings()

    @kb.add('c-d')
    def _exit(event) -> None:
        event.app.exit()

    @kb.add('c-right')
    def _next(event) -> None:
        manager.switch(1)

    @kb.add('c-left')
    def _prev(event) -> None:
        manager.switch(-1)

    # Layout, top to bottom: output · rule · prompt · rule · status · tabs.
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
    app.run()
