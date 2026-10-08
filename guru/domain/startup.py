"""Startup reporting rules: the progress seam, where a model runs, and the
judge warm-up status shown in the TUI statusline.

The endpoints live elsewhere: :mod:`guru.startup` renders steps on the
terminal, :mod:`guru.judges` publishes the warm-up status. This module
holds only the Protocols and the pure rules both sides agree on.
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import ContextManager, Optional, Protocol
from urllib.parse import urlsplit

READY_SHOWN_S = 10.0    # how long 'judges ready' stays in the statusline


class StepHandle(Protocol):
    """A running startup step."""

    def detail(self, text: str) -> None:
        """Set the text shown after the step title."""


class Progress(Protocol):
    """Reports startup steps (titles, details, timing)."""

    def step(self, title: str) -> ContextManager[StepHandle]:
        """Run a step: shown while the block runs, timed when it ends."""

    def detail(self, text: str) -> None:
        """Set the detail of the running step (or print it outside one)."""

    def paused(self) -> ContextManager[None]:
        """Stop any live rendering while a subprocess owns the terminal."""


@dataclass(frozen=True)
class Location:
    """Where a model runs: ``local`` or ``remote``, plus the host."""
    kind: str
    host: str = ''

    def __str__(self) -> str:
        return f'{self.kind}, {self.host}' if self.host else self.kind


def _is_loopback(hostname: str) -> bool:
    if hostname == 'localhost':
        return True
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_unspecified


def host_location(url: Optional[str], remote_default: bool) -> Location:
    """Classify ``url``: loopback hosts are local, any other host remote.

    Without a url the adapter's own ``remote`` flag decides and the host is
    left empty.
    """
    if not url:
        return Location('remote' if remote_default else 'local')
    parts = urlsplit(url if '//' in url else f'//{url}')
    hostname = parts.hostname or ''
    host = parts.netloc.rsplit('@', 1)[-1]
    kind = 'local' if _is_loopback(hostname) else 'remote'
    return Location(kind, host)


@dataclass(frozen=True)
class WarmStatus:
    """Background judge warm-up: ``idle``, ``loading``, ``ready`` or
    ``failed``. ``at`` is the monotonic time the state was entered."""
    state: str = 'idle'
    name: str = ''
    seconds: float = 0.0
    failed: tuple = ()
    at: float = 0.0


def status_text(status: WarmStatus, now: float) -> str:
    """The statusline segment for ``status`` at monotonic time ``now``;
    empty when idle or once a ready notice is older than READY_SHOWN_S."""
    if status.state == 'loading':
        return f'judges: loading {status.name}…'
    if status.state == 'ready':
        if now - status.at > READY_SHOWN_S:
            return ''
        return f'judges ready {status.seconds:.1f}s'
    if status.state == 'failed':
        return f"judges: {', '.join(status.failed)} failed (see log)"
    return ''
