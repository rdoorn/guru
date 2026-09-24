"""Tests for guru.domain.patch: unified-diff parsing/application and the
apply_patch verb (plan B4)."""
import pytest

from guru import config, session
from guru.domain import files, patch

A_OLD = ''.join(f'a{i}\n' for i in range(1, 11))
B_OLD = 'one\ntwo\nthree\n'

TWO_FILES = '''\
diff --git a/a.txt b/a.txt
index 1111111..2222222 100644
--- a/a.txt
+++ b/a.txt
@@ -1,4 +1,5 @@
 a1
+inserted
 a2
 a3
 a4
@@ -8,3 +9,3 @@
 a8
-a9
+nine
 a10
--- b.txt
+++ b.txt
@@ -1,3 +1,3 @@
 one
-two
+TWO
 three
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A writable temp project (read+write allow-listed), cwd inside it,
    prompts denying, fresh sha ledger, console output silenced."""
    monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
    monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', {str(tmp_path)})
    monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
    monkeypatch.setattr(config, 'persist_write_dir', lambda d: None)
    monkeypatch.setattr(session, 'file_shas', {})
    monkeypatch.setattr(files, '_show_change', lambda block: None)
    monkeypatch.chdir(tmp_path)
    files.set_path_asker(lambda q: False)
    (tmp_path / 'a.txt').write_text(A_OLD)
    (tmp_path / 'b.txt').write_text(B_OLD)
    try:
        yield tmp_path
    finally:
        files.set_path_asker(None)


class TestParse:
    def test_two_files_three_hunks(self) -> None:
        fps = patch.parse(TWO_FILES)
        assert [fp.path for fp in fps] == ['a.txt', 'b.txt']
        assert [len(fp.hunks) for fp in fps] == [2, 1]
        h = fps[0].hunks[1]
        assert (h.old_start, h.new_start) == (8, 9)
        assert h.old_lines == ['a8', 'a9', 'a10']
        assert h.new_lines == ['a8', 'nine', 'a10']
        assert fps[0].new_file is False

    def test_new_file_and_no_newline_marker(self) -> None:
        [fp] = patch.parse('--- /dev/null\n+++ b/n.py\n@@ -0,0 +1 @@\n'
                           '+x = 1\n\\ No newline at end of file\n')
        assert fp.new_file and fp.path == 'n.py'
        assert fp.hunks[0].new_no_newline is True
        assert patch.apply_hunks('', fp.hunks) == 'x = 1'

    @pytest.mark.parametrize('diff, why', [
        ('diff --git a/x b/y\nrename from x\nrename to y\n', 'renames'),
        ('--- a/x\n+++ b/y\n@@ -1 +1 @@\n-a\n+b\n', 'rename x -> y'),
        ('Binary files a/x and b/x differ\n', 'binary'),
        ('--- a/x\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n', 'delete_file'),
        ('@@ -1 +1 @@\n-a\n+b\n', 'before any'),
        ('just some prose\n', 'no ---'),
        ('--- a/x\n+++ b/x\n', 'no hunks'),
    ])
    def test_refusals(self, diff, why) -> None:
        with pytest.raises(patch.PatchError, match=why):
            patch.parse(diff)

    def test_targets_best_effort(self) -> None:
        assert patch.targets(TWO_FILES) == ['a.txt', 'b.txt']
        assert patch.targets('--- a/x\n+++ b/x\n') == ['x']
        assert patch.targets('') == []


class TestApplyHunks:
    def test_offset_allowed_context_mismatch_not(self) -> None:
        [fp] = patch.parse('--- a/a.txt\n+++ b/a.txt\n@@ -6,3 +6,3 @@\n'
                           ' a8\n-a9\n+nine\n a10\n')       # header off by 2
        out = patch.apply_hunks(A_OLD, fp.hunks, 'a.txt')
        assert out == A_OLD.replace('a9\n', 'nine\n')
        [fp] = patch.parse('--- a/a.txt\n+++ b/a.txt\n@@ -8,3 +8,3 @@\n'
                           ' a8\n-a9x\n+nine\n a10\n')
        with pytest.raises(patch.PatchError, match='hunk 1 .*context does'):
            patch.apply_hunks(A_OLD, fp.hunks, 'a.txt')

    def test_ambiguous_context_refused(self) -> None:
        [fp] = patch.parse('--- a/r\n+++ b/r\n@@ -9,1 +9,1 @@\n-x\n+y\n')
        with pytest.raises(patch.PatchError, match='2 places'):
            patch.apply_hunks('x\nq\nx\n', fp.hunks)

    def test_preserves_missing_trailing_newline(self) -> None:
        [fp] = patch.parse('--- a/r\n+++ b/r\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n'
                           '\\ No newline at end of file\n')
        assert patch.apply_hunks('a\nb', fp.hunks) == 'a\nc'
        [fp] = patch.parse('--- a/r\n+++ b/r\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n')
        assert patch.apply_hunks('a\nb\n', fp.hunks) == 'a\nc\n'


class TestApplyPatch:
    def test_two_files_three_hunks(self, project) -> None:
        out = patch.apply_patch(TWO_FILES).splitlines()
        assert out[0] == 'Applied patch:'
        a_new = (project / 'a.txt').read_text()
        b_new = (project / 'b.txt').read_text()
        assert a_new == ('a1\ninserted\n' + ''.join(
            f'a{i}\n' for i in range(2, 9)) + 'nine\na10\n')
        assert b_new == 'one\nTWO\nthree\n'
        sha_a, sha_b = files._sha(a_new), files._sha(b_new)
        assert out[1] == (
            f"{project / 'a.txt'}: 2 hunk(s) applied (sha:{sha_a})")
        assert out[2] == (
            f"{project / 'b.txt'}: 1 hunk(s) applied (sha:{sha_b})")
        assert session.file_shas == {str(project / 'a.txt'): sha_a,
                                     str(project / 'b.txt'): sha_b}

    def test_context_mismatch_writes_nothing(self, project) -> None:
        bad = TWO_FILES.replace(' one\n-two\n', ' uno\n-two\n')
        out = patch.apply_patch(bad)
        assert out.startswith('Patch rejected: b.txt hunk 1 (@@ -1): context')
        assert out.endswith('Nothing was written.')
        assert (project / 'a.txt').read_text() == A_OLD     # first file intact
        assert (project / 'b.txt').read_text() == B_OLD
        assert session.file_shas == {}

    def test_new_file_inside_project(self, project) -> None:
        out = patch.apply_patch('--- /dev/null\n+++ b/pkg/n.py\n'
                                '@@ -0,0 +1,2 @@\n+def n():\n+    return 1\n')
        assert out.startswith('Applied patch:')
        assert (project / 'pkg' / 'n.py').read_text() == (
            'def n():\n    return 1\n')
        out = patch.apply_patch('--- /dev/null\n+++ b/a.txt\n@@ -0,0 +1 @@\n'
                                '+x\n')
        assert 'already exists' in out

    def test_new_file_outside_project_refused(self, project,
                                              tmp_path) -> None:
        outside = tmp_path.parent / 'elsewhere_new.txt'
        out = patch.apply_patch(f'--- /dev/null\n+++ {outside}\n'
                                '@@ -0,0 +1 @@\n+x\n')
        assert out.startswith('Patch rejected') and 'outside' in out
        assert not outside.exists()

    def test_rename_and_delete_refused(self, project) -> None:
        out = patch.apply_patch('--- a/a.txt\n+++ b/c.txt\n@@ -1 +1 @@\n'
                                '-a1\n+A\n')
        assert out.startswith('Patch rejected') and 'rename' in out
        out = patch.apply_patch('--- a/a.txt\n+++ /dev/null\n@@ -1,10 +0,0 @@'
                                + ''.join(f'\n-a{i}' for i in range(1, 11))
                                + '\n')
        assert 'delete_file' in out
        assert (project / 'a.txt').read_text() == A_OLD

    def test_read_only_refused(self, project, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_READ_ONLY)
        out = patch.apply_patch(TWO_FILES)
        assert out.startswith('Refused: read-only mode')
        assert (project / 'a.txt').read_text() == A_OLD

    def test_write_denied_for_one_file_writes_none(self, project,
                                                   monkeypatch) -> None:
        # a.txt (cwd) is granted, a sibling directory is denied: nothing
        # may change, and the grant for cwd does not leak to the sibling.
        other = project.parent / f'{project.name}_other'
        other.mkdir()
        (other / 'b.txt').write_text(B_OLD)
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
        asked: list = []

        def ask(question):
            asked.append(question)
            return 'Update(a.txt)' in question
        files.set_path_asker(ask)
        diff = TWO_FILES.replace(
            '--- b.txt\n+++ b.txt', f'--- {other}/b.txt\n+++ {other}/b.txt')
        out = patch.apply_patch(diff)
        assert out.startswith("Write access to")
        assert 'Nothing was written' in out
        assert (project / 'a.txt').read_text() == A_OLD
        assert (other / 'b.txt').read_text() == B_OLD
        assert len(asked) == 2 and 'Update(a.txt)' in asked[0]
        assert 'Update(b.txt)' in asked[1]

    def test_missing_file_and_bad_diff(self, project) -> None:
        assert 'no such file' in patch.apply_patch(
            '--- a/zz.txt\n+++ b/zz.txt\n@@ -1 +1 @@\n-a\n+b\n')
        assert patch.apply_patch('nonsense').startswith('Patch rejected:')
        assert patch.apply_patch('').startswith('Patch rejected:')
