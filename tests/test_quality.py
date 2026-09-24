"""Tests for guru.domain.quality: run_tests, check_syntax, lint (plan B2).

The subprocess tests run the real pytest/flake8 of this .venv on tiny
projects under tmp_path through procs.run (fixed argv, scrubbed env)."""
import sys
from pathlib import Path

import pytest

from guru import config
from guru.domain import files, procs, quality, tools

PASSING = 'def test_ok():\n    assert 1 == 1\n'
FAILING = ('def test_ok():\n    assert 1 == 1\n\n\n'
           'def test_bad():\n    x = 2\n    assert x + 1 == 4\n')


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
    monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
    monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
    monkeypatch.chdir(tmp_path)
    files.set_path_asker(lambda q: False)
    tools.set_policy(None)
    try:
        yield tmp_path
    finally:
        files.set_path_asker(None)
        tools.set_policy(None)


def _argv_spy(monkeypatch):
    """Record every argv procs.run receives (still running the child)."""
    seen: list = []
    real = procs.run

    def spy(argv, cwd, limits=None, env_extra=None):
        seen.append((list(argv), Path(cwd)))
        return real(argv, cwd, limits, env_extra)
    monkeypatch.setattr(procs, 'run', spy)
    return seen


class TestRunTests:
    def test_passing_digest_and_fixed_argv(self, project, monkeypatch):
        seen = _argv_spy(monkeypatch)
        (project / 'test_a.py').write_text(PASSING)
        out = quality.run_tests()
        assert out.startswith('1 passed in')
        argv, cwd = seen[0]
        assert argv == [sys.executable, '-m', 'pytest', '-q', '-p',
                        'no:cacheprovider', '--maxfail=1', '-rfE']
        assert cwd == project
        assert not (project / '.pytest_cache').exists()
        assert not list(project.glob('**/__pycache__'))

    def test_failing_digest_has_id_and_assertion(self, project) -> None:
        (project / 'test_a.py').write_text(FAILING)
        out = quality.run_tests('test_a.py', maxfail='3')
        lines = out.splitlines()
        assert lines[0] == '1 failed, 1 passed in ' + lines[0].split()[-1]
        assert lines[1] == '  test_a.py::test_bad — assert (2 + 1) == 4'
        assert 'detail=<test id>' in lines[2]
        assert len(out) < 600

    def test_detail_returns_the_failure_block(self, project) -> None:
        (project / 'test_a.py').write_text(FAILING)
        out = quality.run_tests('test_a.py', detail='test_a.py::test_bad')
        assert out.startswith('test_a.py::test_bad:\n')
        assert '>       assert x + 1 == 4' in out
        assert 'E       assert (2 + 1) == 4' in out
        assert 'test_a.py:7: AssertionError' in out
        assert 'test_ok' not in out
        out = quality.run_tests('test_a.py', detail='test_a.py::test_nope')
        assert out.startswith("No failure block for 'test_a.py::test_nope'")
        assert 'test_a.py::test_bad' in out

    def test_detail_block_is_capped(self, project, monkeypatch) -> None:
        monkeypatch.setattr(quality, '_DETAIL_BYTES', 80)
        (project / 'test_a.py').write_text(FAILING)
        out = quality.run_tests('test_a.py', detail='test_a.py::test_bad')
        assert len(out) < 140 and 'more chars' in out

    def test_target_node_id_and_k(self, project, monkeypatch) -> None:
        seen = _argv_spy(monkeypatch)
        (project / 'test_a.py').write_text(FAILING)
        out = quality.run_tests('test_a.py::test_ok')
        assert out.startswith('1 passed in')
        assert seen[-1][0][-1] == 'test_a.py::test_ok'
        out = quality.run_tests('test_a.py', k='ok')
        assert out.startswith('1 passed, 1 deselected in')
        assert seen[-1][0][-3:] == ['test_a.py', '-k', 'ok']

    def test_timeout_message(self, project, monkeypatch) -> None:
        tools.set_policy(tools.ToolsPolicy(limits={'timeout_s': 1}))
        (project / 'test_a.py').write_text(
            'import time\n\n\ndef test_ok():\n    pass\n\n\n'
            'def test_slow():\n    time.sleep(30)\n')
        out = quality.run_tests('test_a.py')
        assert out.startswith('timed out after 1s;')
        assert 'tests ran' in out

    def test_target_outside_allow_list_is_refused(self, project,
                                                  tmp_path) -> None:
        outside = tmp_path.parent / 'nowhere' / 'test_x.py'
        out = quality.run_tests(str(outside))
        assert 'denied' in out
        assert quality.run_tests('missing_dir').startswith('No such path')

    def test_unittest_runner_argv(self, project, monkeypatch) -> None:
        seen = _argv_spy(monkeypatch)
        tools.set_policy(tools.ToolsPolicy(test_runner='unittest'))
        (project / 'test_u.py').write_text(
            'import unittest\n\n\nclass T(unittest.TestCase):\n'
            '    def test_ok(self):\n        self.assertEqual(1, 1)\n\n'
            '    def test_bad(self):\n        self.assertEqual(1, 2)\n')
        out = quality.run_tests('test_u.py')
        assert seen[0][0] == [sys.executable, '-m', 'unittest', '-q',
                              'test_u.py']
        assert 'Ran 2 tests' in out and 'FAILED (failures=1)' in out
        assert 'test_u.T.test_bad' in out and 'AssertionError: 1 != 2' in out
        out = quality.run_tests('test_u.py', detail='T.test_bad')
        assert 'self.assertEqual(1, 2)' in out

    def test_pytest_digest_without_summary(self) -> None:
        res = procs.ProcResult(['x'], 2, '', 'boom: no such option', 0.1)
        out = quality._pytest_digest(res, '')
        assert out.startswith('pytest exited 2 without a summary line')
        assert 'boom' in out

    def test_many_failures_are_capped(self) -> None:
        summary = "\n".join(f'FAILED t.py::test_{i} - assert False'
                            for i in range(12))
        res = procs.ProcResult(['x'], 1, summary + '\n12 failed in 0.10s\n',
                               '', 0.1)
        out = quality._pytest_digest(res, '').splitlines()
        assert out[0] == '12 failed in 0.10s'
        assert out[1] == '  t.py::test_0 — assert False'
        assert '… 2 more failures' in out[11]


class TestCheckSyntax:
    def test_ok(self, project) -> None:
        (project / 'good.py').write_text('x = 1\n')
        assert quality.check_syntax('good.py') == (
            f"ok: {project / 'good.py'} compiles.")
        assert not list(project.glob('**/*.pyc'))

    def test_error_names_line_and_text(self, project) -> None:
        (project / 'bad.py').write_text('x = 1\ndef f(:\n    pass\n')
        out = quality.check_syntax('bad.py')
        assert out.startswith(f"SyntaxError at {project / 'bad.py'}:2:")
        assert 'invalid syntax' in out and '\n  def f(:' in out

    def test_gates_and_non_python(self, project, tmp_path) -> None:
        (project / 'n.txt').write_text('x')
        assert 'not a Python file' in quality.check_syntax('n.txt')
        assert quality.check_syntax('gone.py').startswith('No such file')
        assert 'denied' in quality.check_syntax(str(tmp_path.parent / 'z.py'))


class TestLint:
    @pytest.fixture(autouse=True)
    def _fresh_cache(self):
        quality._reset_cache()
        yield
        quality._reset_cache()

    def test_flake8_without_config_and_mypy_skipped(self, project,
                                                    monkeypatch) -> None:
        seen = _argv_spy(monkeypatch)
        (project / 'a.py').write_text('import os\nx=1\n')
        out = quality.lint('a.py')
        if 'flake8: not installed' in out:
            pytest.skip('flake8 not installed in this venv')
        assert out.splitlines()[0] == 'flake8: 2 issue(s)'
        assert "a.py:1:1: F401 'os' imported but unused" in out
        assert 'a.py:2:2: E225' in out
        assert 'mypy: not configured for this project; skipped' in out
        argv = seen[-1][0]
        assert argv[:3] == [sys.executable, '-m', 'flake8']
        assert argv[3].startswith('--extend-exclude=') and '.venv' in argv[3]
        assert argv[4] == 'a.py'
        assert not any('mypy' in a for a, _ in seen)

    def test_flake8_config_is_honoured_and_clean(self, project) -> None:
        (project / 'setup.cfg').write_text('[flake8]\nignore = E225,F401\n')
        (project / 'a.py').write_text('import os\nx=1\n')
        out = quality.lint()
        if 'flake8: not installed' in out:
            pytest.skip('flake8 not installed in this venv')
        assert out.splitlines()[0] == 'flake8: clean'
        assert quality._flake8_configured(project) is True

    def test_detail_expands_flake8(self, project) -> None:
        (project / 'a.py').write_text('import os\n')
        out = quality.lint('a.py', detail='flake8')
        if 'flake8: not run' in out:
            pytest.skip('flake8 not installed in this venv')
        assert out.startswith('flake8 (a.py):\n')
        assert 'F401' in out
        assert "detail must be" in quality.lint('a.py', detail='ruff')

    def test_mypy_runs_only_when_configured(self, project,
                                            monkeypatch) -> None:
        seen = _argv_spy(monkeypatch)
        (project / 'pyproject.toml').write_text(
            '[tool.mypy]\nignore_missing_imports = true\n')
        (project / 'a.py').write_text('def f(x: int) -> str:\n    return x\n')
        monkeypatch.setitem(quality._available, (sys.executable, 'flake8'),
                            False)
        out = quality.lint('a.py')
        if 'mypy: configured but not installed' in out:
            pytest.skip('mypy not installed in this venv')
        assert 'flake8: not installed; skipped' in out
        assert 'mypy: 1 issue(s)' in out
        assert 'a.py:2: error: Incompatible return value type' in out
        mypy_argv = [a for a, _ in seen if a[2] == 'mypy' and a[-1] == 'a.py']
        assert mypy_argv == [[sys.executable, '-m', 'mypy', 'a.py']]

    def test_issues_are_capped(self) -> None:
        rows = "\n".join(f'a.py:{i}:1: E999 x' for i in range(1, 15))
        res = procs.ProcResult(['x'], 1, rows + '\n', '', 0.1)
        out = quality._linter_digest('flake8', res, quality._FLAKE8_ROW)
        lines = out.splitlines()
        assert lines[0] == 'flake8: 14 issue(s)' and len(lines) == 12
        assert '4 more' in lines[-1]

    def test_denied_and_missing_path(self, project, tmp_path) -> None:
        assert 'denied' in quality.lint(str(tmp_path.parent / 'zz'))
        assert quality.lint('nope').startswith('No such path')
