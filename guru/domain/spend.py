"""Spend confirmation: the once-per-run question before remote model spend.

``[routing] spend_confirm`` is ``ask`` (default), ``auto`` or ``never``.
In ``ask`` mode the first remote pick asks the user once through a
pluggable asker (the TUI installs an interactive prompt; the benchmark and
the eval runner install a denier) and the answer is remembered for the rest
of the run (design doc §4). The values returned here are the
``confirmation`` vocabulary of :func:`guru.domain.routing.resolve`:
``pending`` (not yet asked), ``granted``, ``declined`` and ``never``.

Thread-safe: spawns run on worker threads, so two concurrent remote picks
still produce exactly one question.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Callable, Optional

from guru import log

MODES = ('ask', 'auto', 'never')
QUESTION = 'Allow remote model spend for this run?'

_asker: Optional[Callable[[str], bool]] = None
_state: str = 'pending'
_lock = threading.Lock()


def on_event_loop() -> bool:
    """True when called on a thread that is running an asyncio loop.

    An interactive asker must never block there (the prompt itself needs
    the loop), so callers deny or defer instead of asking.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def set_spend_asker(fn: Optional[Callable[[str], bool]]) -> None:
    """Install the spend prompt ``fn(question) -> bool`` (None restores the
    default, which denies)."""
    global _asker
    _asker = fn


def reset() -> None:
    """Forget the remembered answer (a new run)."""
    global _state
    with _lock:
        _state = 'pending'


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f'unknown spend_confirm {mode!r}; expected one of '
                         + ', '.join(MODES))


def status(mode: str) -> str:
    """The confirmation value for ``mode`` without asking: ``never``,
    ``granted`` (auto), or the remembered answer (``pending`` until asked)."""
    _check_mode(mode)
    if mode == 'never':
        return 'never'
    if mode == 'auto':
        return 'granted'
    return _state


def confirmation(mode: str) -> str:
    """The confirmation value for ``mode``, asking once in ``ask`` mode.

    A denied, dismissed or failing prompt is a decline (design doc §5); the
    default asker denies. The answer is remembered until :func:`reset`.
    """
    global _state
    current = status(mode)
    if current != 'pending':
        return current
    with _lock:
        if _state != 'pending':
            return _state
        asker = _asker
        granted = False
        if asker is not None:
            try:
                granted = bool(asker(QUESTION))
            except Exception:                            # noqa: BLE001
                log.exc('spend asker failed; treating as a decline')
        _state = 'granted' if granted else 'declined'
        log.info('spend confirmation: %s', _state)
        return _state
