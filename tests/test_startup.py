"""The terminal startup reporters (guru.startup)."""
from __future__ import annotations

import io

import pytest
from rich.console import Console

from guru import startup


def _console() -> tuple:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=100), buf


class TestRichProgress:
    def test_step_prints_tick_title_detail_and_time(self) -> None:
        con, buf = _console()
        p = startup.RichProgress(con)
        with p.step('adapters') as s:
            s.detail('Ollama (local, localhost:11434)')
        line = buf.getvalue()
        assert '✓ adapters' in line
        assert 'Ollama (local, localhost:11434)' in line
        assert line.rstrip().endswith('s')

    def test_detail_on_progress_targets_the_running_step(self) -> None:
        con, buf = _console()
        p = startup.RichProgress(con)
        with p.step('main model'):
            p.detail('loading…')
            p.detail('gpt-oss · local · ctx 131,072')
        out = buf.getvalue()
        assert 'gpt-oss · local · ctx 131,072' in out
        assert 'loading…' not in out

    def test_multi_line_detail_is_indented(self) -> None:
        con, buf = _console()
        p = startup.RichProgress(con)
        with p.step('adapters') as s:
            s.detail('A (local)\nB (remote)')
        lines = buf.getvalue().splitlines()
        assert 'A (local)' in lines[0]
        assert lines[1].strip().startswith('B (remote)')
        # Aligned under the first detail: '  ✓ adapters  '.
        assert lines[1].index('B (remote)') == lines[0].index('A (local)')

    def test_failure_prints_cross_and_reraises(self) -> None:
        con, buf = _console()
        p = startup.RichProgress(con)
        with pytest.raises(RuntimeError):
            with p.step('main model'):
                raise RuntimeError('daemon down')
        out = buf.getvalue()
        assert '✗ main model' in out and 'daemon down' in out

    def test_header(self) -> None:
        con, buf = _console()
        startup.RichProgress(con).header()
        assert 'guru · starting' in buf.getvalue()

    def test_paused_outside_a_step_is_harmless(self) -> None:
        con, _ = _console()
        with startup.RichProgress(con).paused():
            pass


class TestPlainProgress:
    def test_detail_prints_a_line(self) -> None:
        con, buf = _console()
        startup.PlainProgress(con).detail('gpt-oss ready.')
        assert buf.getvalue().strip() == 'gpt-oss ready.'

    def test_step_is_silent(self) -> None:
        con, buf = _console()
        with startup.PlainProgress(con).step('x') as s:
            s.detail('y')
        assert buf.getvalue().strip() == 'y'


class TestCurrent:
    def test_default_is_plain_and_use_binds(self) -> None:
        assert isinstance(startup.current(), startup.PlainProgress)
        con, _ = _console()
        rich_p = startup.RichProgress(con)
        with startup.use(rich_p):
            assert startup.current() is rich_p
        assert isinstance(startup.current(), startup.PlainProgress)


class TestCliDetails:
    def test_model_detail(self, monkeypatch) -> None:
        from guru import cli, session
        from guru.adapters.ollama import OllamaAdapter
        a = OllamaAdapter()
        monkeypatch.setattr(session, 'model', 'gpt-oss:20b')
        monkeypatch.setattr(session, 'num_ctx', 131072)
        monkeypatch.setattr(a, 'placement', lambda: 'GPU')
        assert cli._model_detail(a) == (
            'gpt-oss:20b · Ollama (local, localhost:11434) · ctx 131,072'
            ' · GPU')

    def test_model_detail_without_placement(self, monkeypatch) -> None:
        from guru import cli, session
        from guru.adapters.anthropic import AnthropicAdapter
        monkeypatch.setattr(session, 'model', 'claude-sonnet-5')
        monkeypatch.setattr(session, 'num_ctx', 0)
        assert cli._model_detail(AnthropicAdapter()) == (
            'claude-sonnet-5 · Anthropic (remote, api.anthropic.com)')

    def test_home(self) -> None:
        from pathlib import Path

        from guru import cli
        assert cli._home(Path.home() / '.guru' / 'settings.toml') == (
            '~/.guru/settings.toml')
        assert cli._home(Path('/etc/x')) == '/etc/x'


def test_a_step_leaves_sys_streams_alone() -> None:
    """A StreamHandler made during a step (log.setup, transformers' import)
    must bind the real stream, not a rich redirect proxy."""
    import sys
    con = Console(file=io.StringIO(), force_terminal=True, width=80)
    before = (sys.stdout, sys.stderr)
    with startup.RichProgress(con).step('x'):
        assert (sys.stdout, sys.stderr) == before
