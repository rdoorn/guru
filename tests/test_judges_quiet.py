"""Library output from the judge warm-up thread goes to the log."""
from __future__ import annotations

import io
import sys
import threading

from guru.judges import quiet


def _run_in(name: str, fn) -> None:
    t = threading.Thread(target=fn, name=name)
    t.start()
    t.join()


class TestThreadRouter:
    def test_warm_up_thread_lines_go_to_the_sink(self) -> None:
        target, lines = io.StringIO(), []
        r = quiet.ThreadRouter(target, 'w', lines.append)
        _run_in('w', lambda: r.write('banner\nmore'))
        _run_in('w', lambda: r.write(' text\n'))
        assert lines == ['banner', 'more text']
        assert target.getvalue() == ''

    def test_other_threads_pass_through(self) -> None:
        target, lines = io.StringIO(), []
        r = quiet.ThreadRouter(target, 'w', lines.append)
        r.write('prompt> ')
        _run_in('other', lambda: r.write('x\n'))
        assert target.getvalue() == 'prompt> x\n'
        assert lines == []

    def test_blank_lines_are_dropped_and_attributes_delegate(self) -> None:
        target, lines = io.StringIO(), []
        r = quiet.ThreadRouter(target, 'w', lines.append)
        _run_in('w', lambda: r.write('\n\n  \n'))
        assert lines == []
        assert r.getvalue() == ''           # delegated to the target
        r.flush()


class TestInstall:
    def test_install_wraps_stdout_and_stderr_once(self, monkeypatch) -> None:
        out, err = io.StringIO(), io.StringIO()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        quiet.install()
        quiet.install()
        assert isinstance(sys.stdout, quiet.ThreadRouter)
        assert isinstance(sys.stderr, quiet.ThreadRouter)
        assert sys.stdout.target is out and sys.stderr.target is err

    def test_routed_lines_are_logged(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(sys, 'stderr', io.StringIO())
        monkeypatch.setattr(sys, 'stdout', io.StringIO())
        quiet.install()
        with caplog.at_level('DEBUG', logger='guru'):
            _run_in(quiet.WARM_THREAD,
                    lambda: print('Device set to use mps', file=sys.stderr))
        assert any('Device set to use mps' in r.getMessage()
                   for r in caplog.records)


def test_progress_bar_redraws_keep_the_last_state() -> None:
    target, lines = io.StringIO(), []
    r = quiet.ThreadRouter(target, 'w', lines.append)
    _run_in('w', lambda: r.write('10%\r50%'))
    _run_in('w', lambda: r.write('\r100%\n'))
    assert lines == ['100%']


def test_writelines_is_diverted_too() -> None:
    target, lines = io.StringIO(), []
    r = quiet.ThreadRouter(target, 'w', lines.append)
    _run_in('w', lambda: r.writelines(['a\n', 'b\n']))
    assert lines == ['a', 'b'] and target.getvalue() == ''


def test_crlf_line_endings_are_kept() -> None:
    target, lines = io.StringIO(), []
    r = quiet.ThreadRouter(target, 'w', lines.append)
    _run_in('w', lambda: r.write('banner\r\n'))
    assert lines == ['banner']
