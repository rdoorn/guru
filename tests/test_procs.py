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

    def test_from_dict_keeps_default_on_bad_value(self, caplog) -> None:
        with caplog.at_level('WARNING', logger='guru'):
            lim = procs.Limits.from_dict({'timeout_s': 'soon', 'cpu_s': 3})
        assert lim.timeout_s == config.PROC_TIMEOUT_S and lim.cpu_s == 3
        assert any('timeout_s' in r.getMessage() for r in caplog.records)


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
        monkeypatch.setattr(subprocess, 'Popen',
                            lambda *a, **k: calls.append(a))
        res = procs.run(_py('print(1)'), outside)
        assert res.returncode == -1
        assert res.denied.startswith(procs.DENIED_PREFIX)
        assert str(outside) in res.denied
        assert res.stdout == '' and calls == []

    def test_nonexistent_cwd_is_denied_not_raised(self, allowed) -> None:
        res = procs.run(_py('print(1)'), allowed / 'missing')
        assert res.returncode == -1 and res.denied

    def test_missing_executable_is_an_error_result(self, allowed) -> None:
        res = procs.run(['/nonexistent/binary-xyz'], allowed)
        assert res.returncode == 127            # the shim's not-found exit
        assert 'nonexistent' in res.stderr


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_dead(pid: int, seconds: float = 3.0) -> bool:
    import time
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


_SPAWN_SLEEPER = ('import subprocess, sys; '
                  'p = subprocess.Popen(["sleep", "20"]); '
                  'print(p.pid); sys.stdout.flush(); ')


class TestProcessGroup:
    """Children run in their own session; the whole group dies with them."""

    def test_timeout_kills_the_grandchild_too(self, allowed) -> None:
        res = procs.run(_py(_SPAWN_SLEEPER + 'import time; time.sleep(30)'),
                        allowed, limits=procs.Limits(timeout_s=1))
        assert res.timed_out is True and res.returncode == -1
        pid = int(res.stdout.split()[0])
        assert _wait_dead(pid), f'grandchild {pid} survived the timeout'

    def test_normal_exit_is_not_timed_out_and_strays_are_reaped(
            self, allowed) -> None:
        res = procs.run(_py(_SPAWN_SLEEPER), allowed,
                        limits=procs.Limits(timeout_s=10))
        assert res.timed_out is False and res.returncode == 0
        pid = int(res.stdout.split()[0])
        assert _wait_dead(pid), f'stray grandchild {pid} survived'


class TestBoundedCapture:
    """Output goes to files under the temp HOME so RLIMIT_FSIZE bounds it;
    guru reads only the head."""

    def test_reads_only_the_head_and_fsize_bounds_the_file(
            self, allowed, monkeypatch) -> None:
        import shutil
        sizes: dict = {}
        real_rmtree = shutil.rmtree

        def spy(path, *a, **k):
            for f in Path(path).iterdir():
                sizes[f.name] = f.stat().st_size
            return real_rmtree(path, *a, **k)
        monkeypatch.setattr(procs.shutil, 'rmtree', spy)
        res = procs.run(
            _py('import sys\n'
                'for _ in range(40): sys.stdout.write("y" * 65536)\n'
                'sys.stdout.flush()'),
            allowed, limits=procs.Limits(out_kb=4, fsize_mb=1))
        assert res.truncated is True
        assert len(res.stdout) == 4096
        assert sizes, 'capture files were not inspected before removal'
        assert max(sizes.values()) <= 1024 * 1024
        assert res.returncode != 0             # SIGXFSZ stopped the writer

    def test_capture_files_are_removed(self, allowed) -> None:
        res = procs.run(_py('import os; print(os.environ["HOME"])'), allowed)
        assert not Path(res.stdout.strip()).exists()


class TestArgvDefence:
    @pytest.mark.parametrize('shell', sorted(procs.SHELLS))
    def test_shell_binaries_are_refused(self, allowed, shell,
                                        monkeypatch) -> None:
        import subprocess
        calls: list = []
        monkeypatch.setattr(subprocess, 'Popen',
                            lambda *a, **k: calls.append(a))
        res = procs.run([f'/bin/{shell}', '-c', 'echo hi'], allowed)
        assert res.returncode == -1 and calls == []
        assert res.denied.startswith(procs.DENIED_PREFIX)
        assert shell in res.denied

    def test_env_extra_cannot_override_protected_keys(
            self, allowed, caplog) -> None:
        import json
        with caplog.at_level('WARNING', logger='guru'):
            res = procs.run(
                _py('import os, json; print(json.dumps(dict(os.environ)))'),
                allowed, env_extra={'PATH': '/evil', 'HOME': '/evil',
                                    'PYTHONPATH': '/evil',
                                    'PYTHONDONTWRITEBYTECODE': '0',
                                    'OK_KEY': 'fine'})
        env = json.loads(res.stdout)
        assert env['PATH'] != '/evil' and env['HOME'] != '/evil'
        assert env['PYTHONPATH'] == str(allowed)
        assert env['PYTHONDONTWRITEBYTECODE'] == '1'
        assert env['OK_KEY'] == 'fine'
        assert any('PATH' in r.getMessage() for r in caplog.records)

    def test_argv0_is_resolved_on_the_scrubbed_path(self, allowed) -> None:
        res = procs.run(['true'], allowed)
        assert res.returncode == 0 and res.denied == ''

    def test_unresolvable_argv0_is_an_error_result(self, allowed) -> None:
        res = procs.run(['no-such-binary-xyz'], allowed)
        assert res.returncode == 127
        assert 'no-such-binary-xyz' in res.stderr
