"""Tests for guru.domain.code: outline and find_symbol (plan B1)."""
import shutil
from pathlib import Path

import pytest

from guru import config, session
from guru.domain import code, files

HERE = Path(__file__).resolve().parent.parent


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A temp project that is the only allow-listed dir, cwd inside it,
    every approval prompt denying."""
    monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
    monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
    monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
    monkeypatch.setattr(session, 'file_shas', {})
    monkeypatch.chdir(tmp_path)
    files.set_path_asker(lambda q: False)
    try:
        yield tmp_path
    finally:
        files.set_path_asker(None)


class TestProjectRoot:
    def test_nearest_allowed_dir_wins(self, tmp_path, monkeypatch) -> None:
        inner = tmp_path / 'a' / 'b'
        inner.mkdir(parents=True)
        monkeypatch.setattr(config, 'ALLOWED_READ_DIRS',
                            {str(tmp_path), str(tmp_path / 'a')})
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
        assert files.project_root(inner / 'x.py') == tmp_path / 'a'
        assert files.project_root(tmp_path / 'y.py') == tmp_path

    def test_falls_back_to_cwd_or_none(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', set())
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
        monkeypatch.chdir(tmp_path)
        assert files.project_root(Path('/nowhere/x')) == tmp_path.resolve()
        assert files.project_root(Path('/nowhere/x'), fallback=False) is None


class TestOutline:
    def test_outline_of_a_real_module(self, project) -> None:
        shutil.copy(HERE / 'guru' / 'domain' / 'pricing.py',
                    project / 'pricing.py')
        out = code.outline('pricing.py')
        assert out.startswith(str(project / 'pricing.py'))
        assert 'sha:' in out and 'outline:' in out
        assert '"""Price table and per-call cost.' in out
        assert '\nL43-53 class Usage\n' in out
        assert 'def cost_usd(model: str, usage: Usage' in out
        assert '-> Optional[float]' in out
        # bodies are not included
        assert 'return' not in out
        assert session.file_shas[str(project / 'pricing.py')] in out

    def test_nested_defs_are_indented_with_ranges(self, project) -> None:
        (project / 'm.py').write_text(
            'class A(Base):\n'
            '    def f(self, x: int) -> str:\n'
            '        def inner():\n'
            '            pass\n'
            '        return ""\n'
            '\n'
            'async def g():\n'
            '    pass\n')
        out = code.outline('m.py').splitlines()
        assert out[1] == 'L1-5 class A(Base)'
        assert out[2] == '  L2-5 def f(self, x: int) -> str'
        assert out[3] == '    L3-4 def inner()'
        assert out[4] == 'L7-8 async def g()'

    def test_non_python_file_shows_numbered_head(self, project) -> None:
        (project / 'notes.md').write_text(
            '\n'.join(f'row {i}' for i in range(1, 61)) + '\n')
        out = code.outline('notes.md')
        assert '(60 lines, sha:' in out
        assert '\n     1\trow 1\n' in out
        assert '    40\trow 40' in out
        assert 'row 41' not in out
        assert '20 more lines' in out

    def test_syntax_error_falls_back_to_head(self, project) -> None:
        (project / 'bad.py').write_text('def f(:\n    pass\n')
        out = code.outline('bad.py')
        assert 'SyntaxError line 1' in out
        assert '\n     1\tdef f(:' in out

    def test_entries_are_capped(self, project, monkeypatch) -> None:
        monkeypatch.setattr(code, '_MAX_ENTRIES', 3)
        (project / 'many.py').write_text(
            ''.join(f'def f{i}():\n    pass\n\n' for i in range(6)))
        out = code.outline('many.py')
        assert out.count('def f') == 3
        assert '3 more entries' in out

    def test_gates_and_errors(self, project, tmp_path) -> None:
        outside = tmp_path.parent / 'elsewhere.py'
        assert 'denied' in code.outline(str(outside))
        assert code.outline('missing.py').startswith('No such file')
        assert 'is a directory' in code.outline('.')
        (project / 'b.bin').write_bytes(b'\x00\x01')
        assert 'binary' in code.outline('b.bin')


class TestFindSymbol:
    @pytest.fixture
    def tree(self, project):
        pkg = project / 'pkg'
        pkg.mkdir()
        (pkg / 'a.py').write_text(
            'LIMIT = 3\n'
            'class Widget:\n'
            '    def render(self):\n'
            '        return LIMIT\n'
            'def render():\n'
            '    return Widget()\n')
        (pkg / 'b.py').write_text(
            'from pkg.a import Widget, render\n'
            'w = Widget()\n'
            'x = render()\n'
            'y = "Widgets"   # not a whole-word match\n')
        (project / '.venv').mkdir()
        (project / '.venv' / 'noise.py').write_text('Widget = 1\n')
        (project / 'README.md').write_text('Widget docs\n')
        return project

    def test_defs_then_refs(self, tree) -> None:
        out = code.find_symbol('Widget').splitlines()
        assert out[0].endswith("'Widget': 1 definition(s), 3 reference(s):")
        assert out[1] == 'def: pkg/a.py:2 (class)'
        refs = [ln for ln in out if ln.startswith('ref:')]
        assert 'ref: pkg/a.py:6: return Widget()' in refs
        assert 'ref: pkg/b.py:1: from pkg.a import Widget, render' in refs
        assert 'ref: pkg/b.py:2: w = Widget()' in refs
        assert not any('Widgets' in r for r in refs)      # word boundary
        assert not any('noise' in ln or 'README' in ln for ln in out)

    def test_methods_and_assignments_count_as_defs(self, tree) -> None:
        out = code.find_symbol('render', 'def').splitlines()
        assert out[1:] == ['def: pkg/a.py:3 (def)', 'def: pkg/a.py:5 (def)']
        out = code.find_symbol('LIMIT', kind='def').splitlines()
        assert out[1:] == ['def: pkg/a.py:1 (assign)']

    def test_kind_ref_only(self, tree) -> None:
        out = code.find_symbol('Widget', 'ref')
        assert 'def:' not in out
        assert out.count('ref:') == 3

    def test_kind_and_name_validation(self, tree) -> None:
        assert "kind must be" in code.find_symbol('Widget', 'both')
        assert 'not a symbol name' in code.find_symbol('a.b')
        assert 'No match' in code.find_symbol('nothing_here')
        assert 'No definition' in code.find_symbol('nothing_here', 'def')

    def test_cap(self, tree, monkeypatch) -> None:
        monkeypatch.setattr(code, '_MAX_ROWS', 2)
        out = code.find_symbol('Widget').splitlines()
        assert len([ln for ln in out if ln[:4] in ('def:', 'ref:')]) == 2
        assert out[-1].startswith('… 2 more')

    def test_denied_root(self, tree, monkeypatch) -> None:
        monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', set())
        assert 'denied' in code.find_symbol('Widget')
