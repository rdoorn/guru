"""Tests for guru.domain.gitread: git_status and git_diff (plan B3) on a
temporary repository, through procs.run with fixed argv."""
import shutil
import subprocess

import pytest

from guru import config
from guru.domain import files, gitread, procs, tools

pytestmark = pytest.mark.skipif(shutil.which('git') is None,
                                reason='git not installed')


def _git(repo, *args) -> None:
    subprocess.run(['git', '-c', 'user.email=t@x', '-c', 'user.name=t',
                    *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A committed temp repo that is the only allow-listed dir and cwd."""
    monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
    monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
    monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
    monkeypatch.chdir(tmp_path)
    files.set_path_asker(lambda q: False)
    tools.set_policy(None)
    _git(tmp_path, 'init', '-q')
    (tmp_path / 'a.py').write_text('x = 1\ny = 2\n')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'b.txt').write_text('b\n')
    _git(tmp_path, 'add', '.')
    _git(tmp_path, 'commit', '-qm', 'init')
    try:
        yield tmp_path
    finally:
        files.set_path_asker(None)


def _spy(monkeypatch) -> list:
    seen: list = []
    real = procs.run

    def spy(argv, cwd, limits=None, env_extra=None):
        seen.append(list(argv))
        return real(argv, cwd, limits, env_extra)
    monkeypatch.setattr(procs, 'run', spy)
    return seen


class TestGitStatus:
    def test_clean(self, repo, monkeypatch) -> None:
        seen = _spy(monkeypatch)
        out = gitread.git_status()
        assert out == f"{repo}: working tree clean."
        assert seen == [
            ['git', '-C', str(repo), 'rev-parse', '--show-toplevel'],
            ['git', '-C', str(repo), 'status', '--porcelain=v1', '-uall']]

    def test_edit_and_untracked(self, repo) -> None:
        (repo / 'a.py').write_text('x = 1\ny = 3\n')
        (repo / 'sub' / 'new.txt').write_text('n\n')
        out = gitread.git_status().splitlines()
        assert out[0] == f"{repo}: 2 path(s) — 1 modified, 1 untracked"
        assert ' M a.py' in out and '?? sub/new.txt' in out

    def test_rows_are_capped(self, repo, monkeypatch) -> None:
        monkeypatch.setattr(gitread, '_MAX_STATUS_ROWS', 2)
        for i in range(4):
            (repo / f'u{i}.txt').write_text('u\n')
        out = gitread.git_status().splitlines()
        assert len(out) == 4 and out[-1] == '… 2 more'

    def test_not_a_repo(self, tmp_path, monkeypatch) -> None:
        other = tmp_path / 'plain'
        other.mkdir()
        monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(other)})
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
        monkeypatch.chdir(other)
        files.set_path_asker(lambda q: False)
        try:
            assert 'not inside a git repository' in gitread.git_status()
        finally:
            files.set_path_asker(None)

    def test_toplevel_outside_allow_list_is_denied(self, repo,
                                                   monkeypatch) -> None:
        sub = repo / 'sub'
        monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(sub)})
        monkeypatch.chdir(sub)
        out = gitread.git_status()
        assert 'repository root' in out and 'denied' in out


class TestGitDiff:
    def test_stat_and_detail(self, repo, monkeypatch) -> None:
        seen = _spy(monkeypatch)
        (repo / 'a.py').write_text('x = 1\ny = 3\nz = 4\n')
        out = gitread.git_diff().splitlines()
        assert out[0] == f"{repo}: 1 file changed, 2 insertions(+), " \
            "1 deletion(-)"
        assert out[1].startswith('a.py |') and '+-' in out[1]
        assert 'detail=true' in out[-1]
        assert seen[-1] == ['git', '-C', str(repo), 'diff', '--stat', '--']
        full = gitread.git_diff(detail='true')
        assert '-y = 2\n+y = 3\n+z = 4\n' in full
        assert seen[-1] == ['git', '-C', str(repo), 'diff', '--']
        assert gitread.git_diff(detail=True) == full

    def test_path_limits_the_diff(self, repo, monkeypatch) -> None:
        seen = _spy(monkeypatch)
        (repo / 'a.py').write_text('x = 0\n')
        (repo / 'sub' / 'b.txt').write_text('bb\n')
        out = gitread.git_diff('sub')
        assert seen[-1] == ['git', '-C', str(repo), 'diff', '--stat', '--',
                            'sub']
        assert 'b.txt' in out and 'a.py' not in out
        assert f'({repo / "sub"})' not in out and '(sub)' in out

    def test_no_changes(self, repo) -> None:
        assert gitread.git_diff() == f"{repo}: no unstaged changes."
        assert gitread.git_diff(detail=True) == (
            f"{repo}: no unstaged changes.")

    def test_detail_is_capped(self, repo, monkeypatch) -> None:
        monkeypatch.setattr(gitread, '_DETAIL_BYTES', 200)
        (repo / 'a.py').write_text(''.join(f'v{i} = {i}\n'
                                           for i in range(100)))
        out = gitread.git_diff(detail=True)
        assert len(out) < 320 and 'diff truncated at 0 KB' in out

    def test_path_outside_is_denied(self, repo, tmp_path) -> None:
        out = gitread.git_diff(str(tmp_path.parent / 'zzz'))
        assert 'denied' in out

    def test_never_writes(self, repo, monkeypatch) -> None:
        seen = _spy(monkeypatch)
        (repo / 'a.py').write_text('x = 0\n')
        gitread.git_status()
        gitread.git_diff()
        gitread.git_diff('a.py', detail=True)
        for argv in seen:
            assert argv[0] == 'git' and argv[3] in (
                'rev-parse', 'status', 'diff')
        status = subprocess.run(['git', 'status', '--porcelain'], cwd=repo,
                                capture_output=True, text=True).stdout
        assert status.strip() == 'M a.py'          # still unstaged


class TestRepoRoot:
    def test_repo_root(self, repo) -> None:
        assert gitread.repo_root() == repo
        assert gitread.repo_root('sub/b.txt') == repo
