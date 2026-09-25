"""Tests for the working-copy half of ``guru.sandbox.colima`` (iteration
2 gate fixes): the read-only ``.git`` mount in the ``docker run`` argv, the
baseline sha ``prepare_copy`` records and the ``BaselineChanged`` refusal
of ``diff``/``show_baseline`` when the copy's ``HEAD`` moved, and
``show_baseline``'s path and size rules on a real temporary repository."""
from pathlib import Path

import pytest

from guru import config
from guru.domain import files, procs
from guru.sandbox import colima
from tests.test_sandbox import FakeRun, _project, _spec


@pytest.fixture
def fake_run(monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(procs, 'run', fake)
    colima.reset_cache()
    yield fake
    colima.reset_cache()


@pytest.fixture
def sbhome(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'SANDBOX_HOME', tmp_path / 'sbhome')
    return tmp_path / 'sbhome'


@pytest.fixture
def allowed(tmp_path, monkeypatch):
    """The temp dir under the read allow-list so git runs through
    procs.run; the prompt denies any escalation."""
    monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
    monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
    files.set_path_asker(lambda question: False)
    try:
        yield tmp_path
    finally:
        files.set_path_asker(None)


def _git(root: Path, *args: str) -> str:
    res = colima._git(list(args), root, root)
    assert res.returncode == 0, res.stderr
    return res.stdout


# --- the read-only .git mount ------------------------------------------------

class TestGitReadOnlyMount:
    def test_argv_mounts_git_read_only_after_the_copy(self, tmp_path):
        spec = _spec(tmp_path)
        copy = tmp_path / 'copy'
        argv = colima.docker_run_argv(spec, ['pytest'], copy, 'n',
                                      git_ro=True)
        i = argv.index(f'{copy}:/work')
        assert argv[i - 1] == '-v'
        assert argv[i + 1:i + 3] == ['-v', f'{copy}/.git:/work/.git:ro']
        assert argv[i + 3:i + 5] == ['-w', '/work']
        plain = colima.docker_run_argv(spec, ['pytest'], copy, 'n')
        assert f'{copy}/.git:/work/.git:ro' not in plain
        assert plain == [a for a in argv
                         if a != f'{copy}/.git:/work/.git:ro'][:i + 1] \
            + argv[i + 3:]

    def test_run_mounts_git_when_the_copy_has_one(self, fake_run, sbhome,
                                                  tmp_path) -> None:
        spec = _spec(tmp_path)
        copy = tmp_path / 'copy'
        (copy / '.git').mkdir(parents=True)
        res = colima.run(spec, ['python', '-c', 'x'], copy)
        assert res.returncode == 0
        argv = fake_run.calls[0]['argv']
        assert argv[argv.index(f'{copy}:/work') + 1:][:2] == [
            '-v', f'{copy}/.git:/work/.git:ro']
        assert res.docker_argv == argv

    def test_run_without_a_git_dir_mounts_only_the_copy(self, fake_run,
                                                        sbhome, tmp_path):
        spec = _spec(tmp_path)
        copy = tmp_path / 'copy'
        copy.mkdir()
        (copy / '.git').write_text('gitdir: elsewhere\n')   # a file, not a dir
        colima.run(spec, ['python', '-c', 'x'], copy)
        argv = fake_run.calls[0]['argv']
        assert ':ro' not in ' '.join(argv)
        assert argv.count('-v') == 1


class TestContainerEnv:
    def test_home_and_cache_point_at_the_tmpfs(self, tmp_path) -> None:
        assert colima.CONTAINER_ENV == (('HOME', '/tmp'),
                                        ('XDG_CACHE_HOME', '/tmp/.cache'))
        spec = _spec(tmp_path)
        argv = colima.docker_run_argv(spec, ['pytest'], tmp_path / 'c', 'n')
        i = argv.index('-w') + 2
        assert argv[i:i + 4] == ['-e', 'HOME=/tmp',
                                 '-e', 'XDG_CACHE_HOME=/tmp/.cache']
        assert argv[i + 4] == spec.image_tag
        # /tmp is the tmpfs: HOME does not outlive the container
        assert argv[argv.index('--tmpfs') + 1] == '/tmp'
        # the caller's env follows the fixed pairs (provisioning's proxy)
        argv = colima.docker_run_argv(spec, ['uv'], tmp_path / 'c', 'n',
                                      env={'HTTPS_PROXY': 'http://p:1'})
        assert argv[i:i + 6] == ['-e', 'HOME=/tmp',
                                 '-e', 'XDG_CACHE_HOME=/tmp/.cache',
                                 '-e', 'HTTPS_PROXY=http://p:1']

    def test_run_passes_the_fixed_env(self, fake_run, sbhome, tmp_path):
        colima.run(_spec(tmp_path), ['python', '-c', 'x'], tmp_path)
        argv = fake_run.calls[0]['argv']
        assert argv.count('-e') == 2 and 'HOME=/tmp' in argv


# --- the recorded baseline ---------------------------------------------------

class TestBaseline:
    def test_prepare_copy_records_the_baseline_sha(self, allowed) -> None:
        root = _project(allowed)
        copy = colima.prepare_copy(root, allowed / 'copy', [])
        sha = _git(copy, 'rev-parse', 'HEAD').strip()
        assert len(sha) == 40
        marker = (copy / colima.COPY_MARKER).read_text()
        assert marker.splitlines() == [
            'working copy made by guru; safe to delete', f'baseline: {sha}']
        assert colima.baseline_sha(copy) == sha
        assert colima.check_baseline(copy, root) == sha
        assert colima.diff(copy, project=root) == ''
        assert colima.show_baseline(copy, 'pkg/__init__.py', root) == \
            'X = 1\n'
        # the marker stays out of the diff
        (copy / 'pkg' / '__init__.py').write_text('X = 2\n')
        text = colima.diff(copy, project=root)
        assert '+X = 2' in text and colima.COPY_MARKER not in text

    def test_rewritten_history_is_refused_by_both_readers(self, allowed):
        root = _project(allowed)
        copy = colima.prepare_copy(root, allowed / 'copy', [])
        recorded = colima.baseline_sha(copy)
        # what a worker with a writable .git could do: commit its edit so
        # the baseline "already contains" it and the diff is empty
        (copy / 'pkg' / '__init__.py').write_text('import os\nX = 2\n')
        _git(copy, 'add', '-A')
        _git(copy, 'commit', '-q', '-m', 'hide')
        assert colima.baseline_sha(copy) == recorded       # marker untouched
        with pytest.raises(colima.BaselineChanged, match='baseline changed'):
            colima.diff(copy, project=root)
        with pytest.raises(colima.BaselineChanged):
            colima.show_baseline(copy, 'pkg/__init__.py', root)
        with pytest.raises(colima.BaselineChanged) as info:
            colima.check_baseline(copy, root)
        assert recorded[:12] in str(info.value)
        assert issubclass(colima.BaselineChanged, RuntimeError)
        # an amend (same tree, new sha) is a rewrite too
        colima.remove_copy(copy)
        copy = colima.prepare_copy(root, allowed / 'copy2', [])
        _git(copy, 'commit', '-q', '--amend', '--allow-empty', '-m', 'x')
        with pytest.raises(colima.BaselineChanged):
            colima.diff(copy, project=root)

    def test_marker_without_a_sha_is_not_read(self, allowed) -> None:
        root = _project(allowed)
        copy = colima.prepare_copy(root, allowed / 'copy', [])
        (copy / colima.COPY_MARKER).write_text('copy\n')
        assert colima.baseline_sha(copy) == ''
        with pytest.raises(RuntimeError, match='records no baseline sha'):
            colima.diff(copy, project=root)
        with pytest.raises(RuntimeError, match='records no baseline sha'):
            colima.show_baseline(copy, 'pkg/__init__.py', root)
        (copy / colima.COPY_MARKER).write_text('baseline: nothex\n')
        assert colima.baseline_sha(copy) == ''
        (copy / colima.COPY_MARKER).unlink()
        assert colima.baseline_sha(copy) == ''
        with pytest.raises(RuntimeError):
            colima.check_baseline(copy, root)
        # not a repository at all: git fails, a RuntimeError (not a crash)
        plain = allowed / 'plain'
        plain.mkdir()
        (plain / colima.COPY_MARKER).write_text(
            'x\nbaseline: ' + 'a' * 40 + '\n')
        with pytest.raises(RuntimeError, match='rev-parse HEAD failed'):
            colima.check_baseline(plain, root)


# --- show_baseline -----------------------------------------------------------

class TestShowBaseline:
    @pytest.fixture
    def copy(self, allowed):
        root = _project(allowed)
        (root / 'pkg' / 'big.py').write_text('# ' + 'x' * 3000 + '\n')
        copy = colima.prepare_copy(root, allowed / 'copy', [])
        return root, copy

    def test_normal_file_is_its_baseline_content(self, copy) -> None:
        root, dest = copy
        (dest / 'pkg' / '__init__.py').write_text('X = 2\n')   # changed
        assert colima.show_baseline(dest, 'pkg/__init__.py', root) == \
            'X = 1\n'
        assert colima.show_baseline(dest, './pkg/__init__.py', root) == \
            'X = 1\n'

    def test_missing_file_is_none(self, copy) -> None:
        root, dest = copy
        assert colima.show_baseline(dest, 'pkg/nope.py', root) is None
        (dest / 'pkg' / 'new.py').write_text('Y = 1\n')  # created after
        assert colima.show_baseline(dest, 'pkg/new.py', root) is None
        assert colima.show_baseline(dest, '', root) is None

    def test_absolute_and_parent_paths_are_refused(self, copy) -> None:
        root, dest = copy
        assert colima.show_baseline(dest, str(root / 'pkg' / '__init__.py'),
                                    root) is None
        assert colima.show_baseline(dest, '/etc/passwd', root) is None
        assert colima.show_baseline(dest, '../proj/pkg/__init__.py',
                                    root) is None
        assert colima.show_baseline(dest, 'pkg/../../x', root) is None

    def test_truncated_content_is_none(self, copy, monkeypatch) -> None:
        root, dest = copy
        assert colima.show_baseline(dest, 'pkg/big.py', root) is not None
        monkeypatch.setattr(colima, 'DIFF_OUT_KB', 1)
        assert colima.show_baseline(dest, 'pkg/big.py', root) is None
        assert colima.show_baseline(dest, 'pkg/__init__.py', root) == \
            'X = 1\n'
