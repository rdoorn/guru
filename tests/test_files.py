"""Tests for the filesystem tools and the directory allow-list gate."""
from pathlib import Path

from guru import config
from guru.domain import files, tools


class TestFileTools:
    """Tests for the filesystem tools and the directory allow-list gate."""

    def _only(self, monkeypatch, *dirs) -> None:
        """Make ALLOWED_READ_DIRS contain exactly ``dirs`` for this test."""
        monkeypatch.setattr(
            config, 'ALLOWED_READ_DIRS',
            {str(Path(d).resolve()) for d in dirs})

    def test_list_dir_shows_perms_and_size(self, monkeypatch) -> None:
        self._only(monkeypatch, Path.cwd())
        out = files.list_dir('guru')
        assert 'session.py' in out and '644' in out

    def test_read_file_range_is_line_numbered(self, monkeypatch) -> None:
        self._only(monkeypatch, Path.cwd())
        out = files.read_file('guru/session.py', '1-3')
        assert 'lines 1-3 of' in out
        assert '\n     1\t' in out

    def test_read_file_bad_range(self, monkeypatch) -> None:
        self._only(monkeypatch, Path.cwd())
        assert 'Invalid line range' in files.read_file('guru/session.py', 'x')

    def test_read_file_caps_large_files(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        big = tmp_path / 'big.txt'
        big.write_text('\n'.join(str(i) for i in range(1, 501)) + '\n')
        out = files.read_file(str(big))
        assert 'lines 1-400 of 500' in out
        assert 'showing first 400 of 500' in out

    def test_read_file_refuses_binary(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        b = tmp_path / 'b.bin'
        b.write_bytes(b'\x00\x01\x02data')
        assert 'binary' in files.read_file(str(b))

    def test_list_tree_skips_noise_dirs(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / '.git').mkdir()
        (tmp_path / '.git' / 'INSIDE_GIT').write_text('y')
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'a.py').write_text('z')
        out = files.list_tree(str(tmp_path), '3')
        assert '*skip' in out
        assert 'INSIDE_GIT' not in out   # noise dir not descended
        assert 'src/a.py' in out         # normal dir descended, flat relpath

    def test_cwd_is_not_auto_allowed(self, monkeypatch) -> None:
        # A fresh project: nothing pre-allowed, so even cwd must be approved.
        self._only(monkeypatch)                       # empty allow-list
        files.set_path_asker(lambda d: False)
        try:
            assert 'denied' in files.list_dir('.').lower()
        finally:
            files.set_path_asker(None)

    def test_gate_denies_outside(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch)
        files.set_path_asker(lambda d: False)
        try:
            assert 'denied' in files.list_dir(str(tmp_path)).lower()
        finally:
            files.set_path_asker(None)

    def test_gate_approves_and_persists(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch)
        saved: list = []
        monkeypatch.setattr(config, 'persist_read_dir', saved.append)
        files.set_path_asker(lambda d: True)
        (tmp_path / 'f.txt').write_text('hello\n')
        try:
            out = files.list_dir(str(tmp_path))
            resolved = str(tmp_path.resolve())
            assert 'f.txt' in out
            assert resolved in config.ALLOWED_READ_DIRS
            assert saved == [resolved]
        finally:
            files.set_path_asker(None)

    def test_parse_range(self) -> None:
        assert files._parse_range('', 500) == (1, 400)
        assert files._parse_range('10-20', 500) == (10, 20)
        assert files._parse_range('bad', 10) == (None, None)
        assert files._parse_range('5-3', 10) == (None, None)
        assert files._parse_range('1-9999', 50) == (1, 50)

    def test_search_code_finds_matches(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / 'a.py').write_text('def foo():\n    return 1\n')
        (tmp_path / 'b.py').write_text('x = foo()\n')
        out = files.search_code('foo', str(tmp_path))
        assert 'a.py:1:' in out and 'b.py:1:' in out

    def test_search_code_regex(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / 'a.py').write_text('def   foo():\n')
        assert 'a.py:1:' in files.search_code(r'def\s+foo', str(tmp_path))

    def test_search_code_invalid_regex_falls_back(
            self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / 'c.py').write_text('foo(bar)\n')
        # '(' is not a valid regex -> literal search
        assert 'c.py:1:' in files.search_code('(', str(tmp_path))

    def test_search_code_no_match(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / 'd.py').write_text('hello\n')
        assert 'No matches' in files.search_code('zzz', str(tmp_path))

    def test_search_code_skips_noise_dirs(
            self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / '.git').mkdir()
        (tmp_path / '.git' / 'x.py').write_text('TOKEN\n')
        (tmp_path / 'src.py').write_text('TOKEN\n')
        out = files.search_code('TOKEN', str(tmp_path))
        assert 'src.py:1:' in out and '.git' not in out

    def test_search_code_gate_denies(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch)              # empty allow-list
        files.set_path_asker(lambda d: False)
        try:
            assert 'denied' in files.search_code('x', str(tmp_path)).lower()
        finally:
            files.set_path_asker(None)

    def test_search_code_skips_escaping_symlink(
            self, tmp_path, monkeypatch) -> None:
        proj = tmp_path / 'proj'
        proj.mkdir()
        secret = tmp_path / 'secret.txt'           # outside the searched tree
        secret.write_text('NEEDLE_SECRET\n')
        (proj / 'real.py').write_text('NEEDLE_OK\n')
        try:
            (proj / 'link.txt').symlink_to(secret)
        except (OSError, NotImplementedError):
            import pytest
            pytest.skip('symlinks not supported here')
        self._only(monkeypatch, proj)
        out = files.search_code('NEEDLE', str(proj))
        assert 'real.py' in out and 'SECRET' not in out

    def test_search_code_smart_case(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / 'a.py').write_text('class Widget:\n    Thing = 1\n')
        # all-lowercase pattern -> case-insensitive (matches 'Widget')
        assert 'a.py:1:' in files.search_code('widget', str(tmp_path))
        # uppercase in pattern -> case-sensitive ('WIDGET' != 'Widget')
        assert 'No matches' in files.search_code('WIDGET', str(tmp_path))
        assert 'a.py:2:' in files.search_code('Thing', str(tmp_path))

    def test_search_code_glob_filter(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / 'a.py').write_text('needle\n')
        (tmp_path / 'b.txt').write_text('needle\n')
        out = files.search_code('needle', str(tmp_path), glob='*.py')
        assert 'a.py:1:' in out and 'b.txt' not in out

    def test_search_code_per_file_cap(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        from guru.domain.files import _MAX_PER_FILE
        big = "\n".join('hit' for _ in range(_MAX_PER_FILE + 10))
        (tmp_path / 'big.py').write_text(big + '\n')
        (tmp_path / 'small.py').write_text('hit\n')
        out = files.search_code('hit', str(tmp_path))
        # capped at _MAX_PER_FILE lines from big.py, with a 'more' note...
        assert out.count('big.py:') == _MAX_PER_FILE + 1   # +1 = more note
        assert 'more matches' in out
        # ...and small.py still gets searched (not crowded out)
        assert 'small.py:1:' in out


class TestWriteTools:
    """Write gate + write_file/edit_file + access modes."""

    def _write_allowed(self, monkeypatch, *dirs):
        monkeypatch.setattr(
            config, 'ALLOWED_WRITE_DIRS',
            {str(Path(d).resolve()) for d in dirs})
        monkeypatch.setattr(config, 'persist_write_dir', lambda d: None)

    def test_write_file_creates(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'x.txt'
        out = files.write_file(str(p), 'hello')
        assert p.read_text() == 'hello' and 'Wrote' in out

    def test_write_refused_in_read_only(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_READ_ONLY)
        self._write_allowed(monkeypatch, tmp_path)
        out = files.write_file(str(tmp_path / 'x.txt'), 'hi')
        assert 'read-only' in out and not (tmp_path / 'x.txt').exists()

    def test_read_allow_does_not_grant_write(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(
            config, 'ALLOWED_READ_DIRS', {str(tmp_path.resolve())})
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
        files.set_path_asker(lambda q: False)     # deny the write prompt
        try:
            out = files.write_file(str(tmp_path / 'z.txt'), 'hi')
            assert 'denied' in out.lower()
            assert not (tmp_path / 'z.txt').exists()
        finally:
            files.set_path_asker(None)

    def test_write_gate_uses_write_list_and_persists(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
        saved: list = []
        monkeypatch.setattr(config, 'persist_write_dir', saved.append)
        files.set_path_asker(lambda q: True)
        try:
            files.write_file(str(tmp_path / 'y.txt'), 'hi')
            resolved = str(tmp_path.resolve())
            assert resolved in config.ALLOWED_WRITE_DIRS
            assert saved == [resolved]
        finally:
            files.set_path_asker(None)

    def test_auto_mode_writes_without_prompt(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_AUTO)
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
        monkeypatch.setattr(config, 'persist_write_dir', lambda d: None)

        def boom(q):
            raise AssertionError('should not prompt in auto mode')
        files.set_path_asker(boom)
        try:
            files.write_file(str(tmp_path / 'a.txt'), 'hi')
            assert (tmp_path / 'a.txt').read_text() == 'hi'
        finally:
            files.set_path_asker(None)

    def test_edit_file_unique_replace(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'c.py'
        p.write_text('a = 1\nb = 2\n')
        sha = files._sha(p.read_text())
        out = files.edit_file(str(p), 'b = 2', 'b = 3', sha)
        assert p.read_text() == 'a = 1\nb = 3\n' and 'Edited' in out

    def test_edit_file_not_found_and_ambiguous(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'c.py'
        p.write_text('x\nx\n')
        sha = files._sha(p.read_text())
        assert 'not found' in files.edit_file(str(p), 'zzz', 'q', sha)
        assert 'appears 2 times' in files.edit_file(str(p), 'x', 'y', sha)

    def test_edit_file_sha_mismatch_refuses(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'c.py'
        p.write_text('a = 1\n')
        out = files.edit_file(str(p), 'a = 1', 'a = 2', 'deadbeef1234')
        assert 'sha mismatch' in out and p.read_text() == 'a = 1\n'

    def test_read_and_write_report_sha(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        monkeypatch.setattr(
            config, 'ALLOWED_READ_DIRS', {str(tmp_path.resolve())})
        p = tmp_path / 'c.py'
        assert 'sha:' in files.write_file(str(p), 'hello\n')
        out = files.read_file(str(p))
        assert 'sha:' in out
        # the sha from read_file is accepted by edit_file
        sha = out.split('sha:')[1].split(')')[0].split(':')[0].strip()
        edited = files.edit_file(str(p), 'hello', 'bye', sha)
        assert 'Edited' in edited and p.read_text() == 'bye\n'

    def test_delete_file_removes(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'd.txt'
        p.write_text('bye')
        out = files.delete_file(str(p))
        assert not p.exists() and 'Deleted' in out

    def test_delete_refused_in_read_only(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_READ_ONLY)
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'd.txt'
        p.write_text('bye')
        out = files.delete_file(str(p))
        assert 'read-only' in out and p.exists()

    def test_delete_refuses_directory_and_missing(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        assert 'directory' in files.delete_file(str(tmp_path))
        assert 'No such file' in files.delete_file(str(tmp_path / 'nope'))

    def test_delete_uses_write_gate(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
        p = tmp_path / 'd.txt'
        p.write_text('bye')
        files.set_path_asker(lambda q: False)         # deny
        try:
            out = files.delete_file(str(p))
            assert 'denied' in out.lower() and p.exists()
        finally:
            files.set_path_asker(None)


class TestAccessPromptDefaults:
    """Access prompts allow only on an explicit yes (Enter/y/yes); any other
    input, junk, escape-sequence text, or error denies."""

    def test_path_enter_allows(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: '')
        assert files._ask_path('/x') is True

    def test_path_y_and_yes_allow(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: 'y')
        assert files._ask_path('/x') is True
        monkeypatch.setattr('builtins.input', lambda *a: 'YES')
        assert files._ask_path('/x') is True

    def test_path_junk_denies(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: 'x')
        assert files._ask_path('/x') is False

    def test_path_escape_sequence_denies(self, monkeypatch) -> None:
        # The CSI-u text a modifyOtherKeys terminal emits for Ctrl+C must NOT
        # be read as approval (the reported bug).
        monkeypatch.setattr('builtins.input', lambda *a: '\x1b[27;5;99~')
        assert files._ask_path('/x') is False

    def test_domain_junk_denies(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: 'maybe')
        assert tools._ask_domain('x.com') is False

    def test_path_explicit_no_denies(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: 'no')
        assert files._ask_path('/x') is False

    def test_path_error_denies(self, monkeypatch) -> None:
        def boom(*a):
            raise EOFError
        monkeypatch.setattr('builtins.input', boom)
        assert files._ask_path('/x') is False

    def test_domain_enter_allows(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: '  ')
        assert tools._ask_domain('x.com') is True

    def test_domain_no_denies(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: 'n')
        assert tools._ask_domain('x.com') is False

    def test_domain_error_denies(self, monkeypatch) -> None:
        def boom(*a):
            raise KeyboardInterrupt
        monkeypatch.setattr('builtins.input', boom)
        assert tools._ask_domain('x.com') is False
