"""The usage dashboard's rules: one server among many gurus, and how a
guru recognises another guru's dashboard on the port.

Every guru tries to bind ``127.0.0.1:<port>`` at startup; the operating
system makes the bind exclusive, so exactly one serves. The others ask
``/api/health`` who holds the port: a guru dashboard answers with
:data:`APP` and its pid, anything else is "another program". Gurus that do
not serve retry the bind every :func:`retry_delay` seconds, so one of them
takes over when the server goes away. The server itself is
``guru.dashboard.server``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

APP = 'guru-dashboard'
RETRY_S = 30.0
RETRY_JITTER = 0.3

SERVING = 'serving'
OTHER_GURU = 'other-guru'
PORT_BUSY = 'port-busy'
OFF = 'off'


@dataclass(frozen=True)
class Holder:
    """Who serves the dashboard when this guru does not."""
    pid: int
    started: str = ''


def parse_health(data: object) -> Optional[Holder]:
    """The guru holding the port, from its ``/api/health`` JSON; None when
    the answer is not a guru dashboard's."""
    if not isinstance(data, dict) or data.get('app') != APP:
        return None
    pid = data.get('pid')
    if not isinstance(pid, int) or isinstance(pid, bool):
        return None
    return Holder(pid=pid, started=str(data.get('started') or ''))


def retry_delay(rand: Optional[random.Random] = None) -> float:
    """Seconds until the next bind attempt: ``RETRY_S`` +/- ``RETRY_JITTER``
    so waiting gurus do not all retry in the same instant."""
    r = (rand or random).uniform(-RETRY_JITTER, RETRY_JITTER)
    return RETRY_S * (1 + r)


def status_text(state: str, url: str, holder: Optional[Holder]) -> str:
    """The startup step / ``/dashboard`` line for a state."""
    if state == SERVING:
        return f'serving at {url}'
    if state == OTHER_GURU and holder is not None:
        return f'served by guru pid {holder.pid} at {url}'
    if state == PORT_BUSY:
        return f'{url} is used by another program; retrying'
    return 'off'
