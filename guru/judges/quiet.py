"""Keep the judge libraries' console output out of the TUI.

The warm-up thread loads transformers / gliner2 models, which print a
config banner (gliner2), log ``Device set to use mps`` (transformers
pipelines, a StreamHandler on ``sys.stderr``) and warn about the attention
implementation. While the TUI runs that lands in the chat box. ``install``
wraps ``sys.stdout`` and ``sys.stderr`` in a :class:`ThreadRouter` that
sends whatever the warm-up thread writes to the guru log (debug) and passes
every other thread's output through unchanged — a process-wide
``redirect_stdout`` would also swallow the prompt and other threads.

Call ``install`` before transformers is imported: its handler keeps the
``sys.stderr`` object it saw at import time.
"""
from __future__ import annotations

import sys
import threading
from typing import Any, Callable, TextIO

from guru import log

WARM_THREAD = 'guru-judge-warm-up'


class ThreadRouter:
    """A text stream that diverts one thread's lines to ``sink``."""

    def __init__(self, target: TextIO, thread_name: str,
                 sink: Callable[[str], None]) -> None:
        self.target = target
        self._thread_name = thread_name
        self._sink = sink
        self._partial = ''
        self._local = threading.local()

    def _diverted(self) -> bool:
        return (threading.current_thread().name == self._thread_name
                and not getattr(self._local, 'sinking', False))

    def write(self, text: str) -> int:
        if not self._diverted():
            return self.target.write(text)
        *lines, self._partial = (self._partial + text).split('\n')
        # A progress bar redraws with '\r': keep only its last state.
        lines = [line.rstrip('\r').rsplit('\r', 1)[-1] for line in lines]
        self._partial = self._partial.rsplit('\r', 1)[-1]
        self._local.sinking = True    # a sink that writes here passes through
        try:
            for line in lines:
                if line.strip():
                    self._sink(line.rstrip())
        finally:
            self._local.sinking = False
        return len(text)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self.target.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)


def _to_log(line: str) -> None:
    log.log.debug('judge warm-up output: %s', line)


def install() -> None:
    """Route the warm-up thread's stdout/stderr to the log (idempotent)."""
    for name in ('stdout', 'stderr'):
        stream = getattr(sys, name)
        if stream is not None and not isinstance(stream, ThreadRouter):
            setattr(sys, name, ThreadRouter(stream, WARM_THREAD, _to_log))
