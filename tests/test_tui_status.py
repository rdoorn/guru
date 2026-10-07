"""The judge warm-up segment of the TUI statusline."""
from __future__ import annotations

import time
from types import SimpleNamespace as NS

import guru.judges as judges
from guru import tui_io
from guru.domain.startup import WarmStatus


def _state() -> NS:
    return NS(num_ctx=8192, ctx_used=0, model='m:1', model_size='8B',
              active_role=None, active_skill=None, messages=[],
              active_tool_names=set(), can_spawn=False, session_in=0,
              session_out=0, git_branch='main')


def test_no_segment_when_idle(monkeypatch) -> None:
    monkeypatch.setattr(judges, 'warm_status', lambda: WarmStatus())
    _, _, right, _ = tui_io._status_from(_state())
    assert 'judges' not in right


def test_loading_segment(monkeypatch) -> None:
    monkeypatch.setattr(judges, 'warm_status',
                        lambda: WarmStatus('loading', name='decide'))
    _, _, right, _ = tui_io._status_from(_state())
    assert right.endswith(' | judges: loading decide…')


def test_ready_segment_then_gone(monkeypatch) -> None:
    now = time.monotonic()
    monkeypatch.setattr(judges, 'warm_status',
                        lambda: WarmStatus('ready', seconds=8.1, at=now))
    assert tui_io._judges_segment() == ' | judges ready 8.1s'
    monkeypatch.setattr(judges, 'warm_status',
                        lambda: WarmStatus('ready', seconds=8.1,
                                           at=now - 60))
    assert tui_io._judges_segment() == ''
