"""Terminal reporters for the startup steps (the ``Progress`` seam of
:mod:`guru.domain.startup`).

``RichProgress`` is bound by ``cli.main`` for the startup phases: a spinner
while a step runs, then one ``✓ title  detail  0.4s`` line (``✗`` and the
error when it raises). Outside startup the default ``PlainProgress`` prints
a detail as a dim line, so adapter code calling ``current().detail(...)``
behaves the same when a model is switched from ``/models``.
"""
from __future__ import annotations

import contextlib
import time
from typing import Any, Iterator, Optional

from rich.live import Live
from rich.markup import escape
from rich.spinner import Spinner
from rich.text import Text

from guru import ui
from guru.domain.startup import Progress


def _spinner(markup: str) -> Spinner:
    return Spinner('dots', text=Text.from_markup(markup))


class _Step:
    def __init__(self, owner: 'RichProgress', title: str) -> None:
        self._owner = owner
        self.title = title
        self.text = ''

    def detail(self, text: str) -> None:
        self.text = text
        self._owner._spin(self)


class RichProgress:
    """Spinner-and-tick step list on a Rich console. Steps do not nest.

    The spinner is a ``Live`` with stdout/stderr redirection off: rich's
    ``console.status`` swaps ``sys.stdout``/``sys.stderr`` for its own
    proxy while it runs, and a ``logging.StreamHandler`` created during a
    step (``log.setup`` under GURU_DEBUG, transformers' handler at import)
    would keep that proxy for good. ``ui.console`` prints still render
    above the spinner (same console).
    """

    def __init__(self, console: Any) -> None:
        self._console = console
        self._status: Any = None
        self._step: Optional[_Step] = None

    def header(self) -> None:
        self._console.print('[bold]guru[/bold] [dim]· starting[/dim]')

    def _spin(self, step: _Step) -> None:
        if self._status is None:
            return
        text = f'[bold]{escape(step.title)}[/bold]'
        if step.text:
            text += f'  [dim]{escape(step.text.splitlines()[0])}[/dim]'
        self._status.update(_spinner(text))

    @contextlib.contextmanager
    def step(self, title: str) -> Iterator[_Step]:
        step = _Step(self, title)
        prev, self._step = self._step, step
        t0 = time.monotonic()
        self._status = Live(_spinner(f'[bold]{escape(title)}[/bold]'),
                            console=self._console, transient=True,
                            refresh_per_second=12.5, redirect_stdout=False,
                            redirect_stderr=False)
        self._status.start()
        try:
            yield step
        except BaseException as e:
            self._stop()
            self._console.print(
                f'  [red]✗[/red] [bold]{escape(title)}[/bold]  '
                f'[red]{escape(str(e) or type(e).__name__)}[/red]')
            raise
        else:
            self._stop()
            self._finish(step, time.monotonic() - t0)
        finally:
            self._step = prev

    def _stop(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _finish(self, step: _Step, secs: float) -> None:
        first, *rest = (step.text or '').splitlines() or ['']
        line = f'  [green]✓[/green] [bold]{escape(step.title)}[/bold]'
        if first:
            line += f'  {escape(first)}'
        line += f'  [dim]{secs:.1f}s[/dim]'
        self._console.print(line, highlight=False)
        # Continuation lines start under the first detail: '  ✓ title  '.
        indent = ' ' * (len(step.title) + 6)
        for extra in rest:
            self._console.print(f'{indent}{escape(extra)}', highlight=False)

    def detail(self, text: str) -> None:
        if self._step is not None:
            self._step.detail(text)
        else:
            self._console.print(f'[dim]{escape(text)}[/dim]',
                                highlight=False)

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        status = self._status
        if status is not None:
            status.stop()
        try:
            yield
        finally:
            if status is not None and self._status is status:
                status.start()


class _PlainStep:
    def __init__(self, owner: 'PlainProgress') -> None:
        self._owner = owner

    def detail(self, text: str) -> None:
        self._owner.detail(text)


class PlainProgress:
    """Outside startup: no step lines, each detail a dim line."""

    def __init__(self, console: Any = None) -> None:
        self._console = console

    def _out(self) -> Any:
        return self._console if self._console is not None else ui.console

    @contextlib.contextmanager
    def step(self, title: str) -> Iterator[_PlainStep]:
        yield _PlainStep(self)

    def detail(self, text: str) -> None:
        self._out().print(f'[dim]{escape(text)}[/dim]', highlight=False)

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        yield


_default = PlainProgress()
_current: Progress = _default


def current() -> Progress:
    """The bound reporter (``PlainProgress`` unless startup bound one)."""
    return _current


@contextlib.contextmanager
def use(progress: Progress) -> Iterator[Progress]:
    """Bind ``progress`` as :func:`current` for the block."""
    global _current
    prev, _current = _current, progress
    try:
        yield progress
    finally:
        _current = prev
