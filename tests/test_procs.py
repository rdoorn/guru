"""Tests for guru.domain.procs: the fixed-argv subprocess runner behind the
audited tools (env scrubbing, limits, output caps, allow-list gate)."""
import os
import sys
from pathlib import Path

import pytest

from guru import config
from guru.domain import files, procs


@pytest.fixture
def allowed(tmp_path, monkeypatch):
    """A cwd under the read allow-list, with the prompt denying escalation."""
    monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
    monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
    files.set_path_asker(lambda question: False)
    try:
        yield tmp_path
    finally:
        files.set_path_asker(None)


def _py(code: str) -> list:
    return [sys.executable, '-c', code]


class TestLimits:
    def test_defaults_follow_config(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'PROC_TIMEOUT_S', 7)
        monkeypatch.setattr(config, 'PROC_OUT_KB', 3)
        lim = procs.Limits()
        assert (lim.timeout_s, lim.cpu_s, lim.mem_mb, lim.fsize_mb,
                lim.out_kb) == (7, 120, 2048, 64, 3)

    def test_from_dict_overrides_and_ignores_unknown(self) -> None:
        lim = procs.Limits.from_dict({'timeout_s': 5, 'bogus': 1})
        assert lim.timeout_s == 5 and lim.cpu_s == config.PROC_CPU_S


class TestRun:
    def test_runs_fixed_argv_and_captures_output(self, allowed) -> None:
        res = procs.run(_py('import sys; print("out"); '
                            'print("err", file=sys.stderr)'), allowed)
        assert res.returncode == 0
        assert res.stdout == 'out\n' and res.stderr == 'err\n'
        assert res.argv[0] == sys.executable
        assert res.timed_out is False and res.truncated is False
        assert res.denied == '' and res.seconds >= 0

    def test_argv_must_be_a_list_never_a_shell_string(self, allowed) -> None:
        with pytest.raises(TypeError):
            procs.run('echo hi', allowed)          # type: ignore[arg-type]
        with pytest.raises(TypeError):
            procs.run([sys.executable, 1], allowed)   # type: ignore[list-item]

    def test_env_is_built_from_scratch(self, allowed, monkeypatch) -> None:
        monkeypatch.setenv('GURU_TEST_SECRET', 'hunter2')
        res = procs.run(_py('import os, json; print(json.dumps(dict('
                            'os.environ)))'), allowed,
                        env_extra={'GURU_EXTRA': 'yes'})
        import json
        env = json.loads(res.stdout)
        assert 'GURU_TEST_SECRET' not in env
        assert env['GURU_EXTRA'] == 'yes'
        assert env['PYTHONDONTWRITEBYTECODE'] == '1'
        assert env['PYTHONPATH'] == str(allowed)
        assert 'PATH' in env and 'LANG' in env
        assert env['HOME'] != os.path.expanduser('~')
        assert not Path(env['HOME']).exists()     # temp HOME removed after

    def test_timeout_kills_a_sleeping_child(self, allowed) -> None:
        res = procs.run(_py('import time; time.sleep(30)'), allowed,
                        limits=procs.Limits(timeout_s=1))
        assert res.timed_out is True
        assert res.returncode == -1
        assert res.seconds < 10

    def test_output_is_capped(self, allowed) -> None:
        res = procs.run(_py('import sys; sys.stdout.write("x" * 5000)'),
                        allowed, limits=procs.Limits(out_kb=1))
        assert res.truncated is True
        assert len(res.stdout) == 1024
        assert res.returncode == 0

    def test_bad_utf8_is_replaced_not_raised(self, allowed) -> None:
        res = procs.run(_py('import sys; sys.stdout.buffer.write(b"a\\xffb")'),
                        allowed)
        assert res.stdout == 'a�b'

    @pytest.mark.skipif(sys.platform == 'win32', reason='no rlimits')
    def test_rlimits_are_applied_to_the_child(self, allowed) -> None:
        code = ('import resource as r; '
                'print(r.getrlimit(r.RLIMIT_CPU)[0]); '
                'print(r.getrlimit(r.RLIMIT_FSIZE)[0])')
        res = procs.run(_py(code), allowed,
                        limits=procs.Limits(cpu_s=9, fsize_mb=1))
        cpu, fsize = res.stdout.split()
        assert int(cpu) == 9
        assert int(fsize) == 1024 * 1024

    def test_cwd_outside_allow_list_is_refused(self, allowed, tmp_path_factory,
                                               monkeypatch) -> None:
        outside = tmp_path_factory.mktemp('outside')
        calls: list = []
        import subprocess
        monkeypatch.setattr(subprocess, 'run',
                            lambda *a, **k: calls.append(a))
        res = procs.run(_py('print(1)'), outside)
        assert res.returncode == -1
        assert 'denied' in res.denied.lower() or str(outside) in res.denied
        assert res.stdout == '' and calls == []

    def test_nonexistent_cwd_is_denied_not_raised(self, allowed) -> None:
        res = procs.run(_py('print(1)'), allowed / 'missing')
        assert res.returncode == -1 and res.denied

    def test_missing_executable_is_an_error_result(self, allowed) -> None:
        res = procs.run(['/nonexistent/binary-xyz'], allowed)
        assert res.returncode == -1
        assert 'nonexistent' in res.stderr or 'No such file' in res.stderr
