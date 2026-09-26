"""Structural round, Package C item 4: the project brief (domain), its
store (repository) and the ``/brief`` endpoint body."""
import json
from pathlib import Path

import pytest

from guru import config
from guru.domain import brief
from guru.repositories import briefs

PROJECT_PY = '''"""A module."""


def alpha(x: int) -> int:
    return x


class Widget:
    def render(self) -> str:
        return 'w'

    def _hidden(self) -> None:
        def inner() -> None:
            pass
'''


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A small project: a package with two modules, tests, Makefile with a
    test target, pyproject with tool sections, and a noise dir."""
    monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
    pkg = tmp_path / 'app'
    pkg.mkdir()
    (pkg / '__init__.py').write_text('')
    (pkg / 'core.py').write_text(PROJECT_PY)
    (pkg / 'util.py').write_text('def helper():\n    return 1\n')
    (pkg / 'broken.py').write_text('def (:\n')
    (tmp_path / 'script.py').write_text('X = 1\n')
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests' / 'test_core.py').write_text(
        'import pytest\n\ndef test_a():\n    assert True\n')
    (tmp_path / 'Makefile').write_text(
        '.PHONY: test lint\n\nVAR := x\n\ntest:\n\tpytest -q\n\nlint:\n'
        '\tflake8\n')
    (tmp_path / 'pyproject.toml').write_text(
        '[project]\nname = "app"\nrequires-python = ">=3.12"\n\n'
        '[tool.mypy]\nfiles = ["app"]\nstrict = true\n\n'
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n\n'
        '[tool.other]\nnested = { a = 1 }\n')
    (tmp_path / '.flake8').write_text('[flake8]\nmax-line-length = 79\n')
    (tmp_path / '__pycache__').mkdir()
    (tmp_path / '__pycache__' / 'junk.py').write_text('def junk(): pass\n')
    return tmp_path


class TestBuild:
    def test_map_modules_and_counts(self, project) -> None:
        b = brief.build(project, 'abc')
        assert b.head_sha == 'abc' and b.root == str(project.resolve())
        assert b.python_files == 6 and b.files == 9
        assert b.dirs['app'] == 4 and b.dirs['tests'] == 1 and b.dirs['.'] == 4
        assert 'junk' not in json.dumps(b.to_dict())   # noise dir skipped
        assert b.modules == ['app', 'script']
        assert not b.truncated and b.build_seconds >= 0

    def test_outlines_and_symbols(self, project) -> None:
        b = brief.build(project, 'abc')
        rows = b.outlines['app/core.py']
        assert rows[0] == 'L4-5 def alpha(x: int) -> int'
        assert rows[1] == 'L8-14 class Widget'
        assert rows[2] == '  L9-10 def render(self) -> str'
        assert 'app/broken.py' not in b.outlines      # does not parse
        assert 'script.py' not in b.outlines          # no def/class
        assert b.symbols['alpha'] == ['app/core.py:4']
        assert b.symbols['render'] == ['app/core.py:9']
        assert 'inner' not in b.symbols                # depth 2: not indexed
        assert b.symbols['helper'] == ['app/util.py:1']

    def test_outline_rows_are_capped(self, project, monkeypatch) -> None:
        monkeypatch.setattr(brief, 'OUTLINE_ROWS_PER_MODULE', 2)
        b = brief.build(project, 'abc')
        rows = b.outlines['app/core.py']
        assert len(rows) == 3 and rows[-1].startswith('… ')

    def test_test_command_and_make_targets(self, project) -> None:
        b = brief.build(project, 'abc')
        assert b.make_targets == ['test', 'lint']       # .PHONY / VAR skipped
        assert b.test_command == 'make test'
        (project / 'Makefile').unlink()
        assert brief.build(project, 'abc').test_command == \
            'python -m pytest -q'
        (project / 'pyproject.toml').unlink()
        (project / 'tests' / 'test_core.py').write_text(
            'import unittest\n\nclass T(unittest.TestCase):\n    pass\n')
        assert brief.build(project, 'abc').test_command == \
            'python -m unittest discover -s tests'
        assert brief.build(project / 'app', 'abc').test_command == ''

    def test_conventions(self, project) -> None:
        conv = brief.build(project, 'abc').conventions
        assert conv['project'] == {'requires-python': '>=3.12'}
        assert conv['mypy'] == {'files': ['app'], 'strict': True}
        assert conv['pytest'] == {}                    # nested table dropped
        assert conv['other'] == {}
        assert conv['flake8'] == {'max-line-length': '79'}

    def test_budget_truncates_outlining_not_counting(self, project,
                                                     monkeypatch) -> None:
        clock = iter([0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                      10.0, 10.0, 10.0, 10.0])
        monkeypatch.setattr(brief.time, 'monotonic', lambda: next(clock))
        b = brief.build(project, 'abc', budget_s=1.0)
        assert b.truncated and b.files == 9
        assert len(b.outlines) < 2

    def test_module_cap(self, project, monkeypatch) -> None:
        monkeypatch.setattr(brief, 'MAX_MODULES', 1)
        b = brief.build(project, 'abc')
        assert b.truncated and len(b.outlines) == 1

    def test_head_sha_via_procs(self, project, monkeypatch) -> None:
        # Not a checkout -> NO_HEAD; the read allow-list gate applies.
        assert brief.head_sha(project) == brief.NO_HEAD
        from guru.domain import procs
        monkeypatch.setattr(procs, 'run', lambda argv, cwd, limits=None:
                            procs.ProcResult(argv, 0, 'a' * 40 + '\n', '',
                                             0.1))
        assert brief.head_sha(project) == 'a' * 40

    def test_head_sha_on_a_real_checkout(self) -> None:
        root = Path(__file__).resolve().parents[1]
        if not (root / '.git').exists():
            pytest.skip('not a git checkout')
        config.ALLOWED_READ_DIRS.add(str(root))
        sha = brief.head_sha(root)
        assert len(sha) == 40 and sha != brief.NO_HEAD

    def test_round_trip(self, project) -> None:
        b = brief.build(project, 'abc')
        again = brief.Brief.from_dict(json.loads(json.dumps(b.to_dict())))
        assert again == b
        assert brief.Brief.from_dict({**b.to_dict(), 'future': 1}) == b


class TestSlice:
    @pytest.fixture
    def built(self, project) -> brief.Brief:
        return brief.build(project, 'abcdef0123456789')

    def test_map_first_then_test_command(self, built) -> None:
        text = brief.slice(built, 'nothing relevant here')
        head = brief.render_map(built)
        assert text.startswith(head)
        assert '[project brief]' in head and '@ abcdef0' in head
        assert 'tests: make test' in head
        assert 'dirs: ./ 4, app/ 4, tests/ 1' in head
        assert 'modules: app, script' in head
        assert 'app/core.py:' not in text

    def test_files_named_by_path_basename_or_stem(self, built) -> None:
        text = brief.slice(built, 'Look at app/core.py')
        assert 'app/core.py:\n  L4-5 def alpha' in text
        assert 'app/util.py' not in text
        text = brief.slice(built, 'fix util (core.py too)')
        assert text.index('app/core.py:') < text.index('app/util.py:')
        assert '__init__' not in text

    def test_symbols_named_in_the_task(self, built) -> None:
        text = brief.slice(built, 'make helper return 2 and rename Widget')
        assert 'symbols:\n' in text
        assert '  Widget: app/core.py:8' in text
        assert '  helper: app/util.py:1' in text
        # A file already shown is not repeated under symbols.
        text = brief.slice(built, 'in core.py rename Widget')
        assert 'symbols:' not in text

    def test_prose_words_are_not_symbols(self, built) -> None:
        # 'alpha' is a five-letter prose word; only code-like words count
        # (an underscore, CamelCase or six letters and up).
        text = brief.slice(built, 'the alpha version')
        assert 'symbols:' not in text
        assert '  render: app/core.py:9' in brief.slice(built, 'fix render')
        assert '  Widget: app/core.py:8' in brief.slice(built, 'a Widget')

    def test_budget_cuts_at_the_estimate(self, built) -> None:
        text = brief.slice(built, 'core.py util.py', max_tokens=90)
        assert brief.estimate_tokens(text) <= 90 + 10   # the cut row itself
        assert text.endswith('… (brief cut at the token budget)')
        assert 'app/util.py:' not in text
        full = brief.slice(built, 'core.py util.py', max_tokens=1500)
        assert 'app/util.py:' in full and 'brief cut' not in full

    def test_estimate_tokens(self) -> None:
        assert brief.estimate_tokens('') == 0
        assert brief.estimate_tokens('abcd') == 1
        assert brief.estimate_tokens('abcde') == 2
        assert brief.CHARS_PER_TOKEN == 4 and brief.SLICE_MAX_TOKENS == 1500


class FakeStore:
    def __init__(self) -> None:
        self.saved: list = []
        self.stored: dict = {}

    def load(self, root, head_sha):
        return self.stored.get(head_sha)

    def save(self, b):
        self.saved.append(b)
        self.stored[b.head_sha] = b
        return Path('/fake') / f'{b.head_sha}.json'


class TestCurrent:
    def test_builds_then_loads_by_head(self, project, monkeypatch) -> None:
        monkeypatch.setattr(brief, 'head_sha', lambda root: 'h1')
        store = FakeStore()
        first = brief.current(project, store=store)
        assert store.saved == [first]
        again = brief.current(project, store=store)
        assert again is first and len(store.saved) == 1
        monkeypatch.setattr(brief, 'head_sha', lambda root: 'h2')
        third = brief.current(project, store=store)
        assert third.head_sha == 'h2' and len(store.saved) == 2

    def test_refresh_rebuilds(self, project, monkeypatch) -> None:
        monkeypatch.setattr(brief, 'head_sha', lambda root: 'h1')
        store = FakeStore()
        first = brief.current(project, store=store)
        second = brief.current(project, refresh=True, store=store)
        assert second is not first and store.saved == [first, second]

    def test_no_head_is_never_stored(self, project) -> None:
        store = FakeStore()
        b = brief.current(project, store=store)
        assert b.head_sha == brief.NO_HEAD and store.saved == []

    def test_default_store_is_the_repository(self, project, tmp_path,
                                             monkeypatch) -> None:
        monkeypatch.setattr(brief, 'head_sha', lambda root: 'h1')
        monkeypatch.setattr(briefs, 'BRIEFS_DIR', tmp_path / 'store')
        b = brief.current(project)
        assert briefs.path_for(project, 'h1').is_file()
        assert briefs.load(project, 'h1') == b


class TestBriefCommand:
    def test_show_refresh_slice_usage(self, project, monkeypatch) -> None:
        monkeypatch.setattr(brief, 'head_sha', lambda root: 'h1')
        store = FakeStore()
        out = brief.brief_command('', root=project, store=store)
        assert out.startswith('[project brief] ')
        assert 'built in ' in out and 'modules outlined' in out
        assert len(store.saved) == 1
        brief.brief_command('refresh', root=project, store=store)
        assert len(store.saved) == 2
        out = brief.brief_command('slice fix core.py', root=project,
                                  store=store)
        assert 'app/core.py:' in out and len(store.saved) == 2
        assert brief.brief_command('bogus', root=project) == brief.BRIEF_USAGE
        assert '/brief refresh' in brief.BRIEF_USAGE

    def test_no_git_note(self, project) -> None:
        out = brief.brief_command('', root=project, store=FakeStore())
        assert 'not a git checkout, so not stored' in out


class TestRepository:
    def test_key_and_path(self, tmp_path) -> None:
        key = briefs.project_key(tmp_path)
        assert key.startswith(tmp_path.name + '-')
        assert len(key.split('-')[-1]) == 8
        assert briefs.project_key(tmp_path) == key
        assert briefs.project_key(tmp_path / 'x') != key
        path = briefs.path_for(tmp_path, 'abc/../def', base=tmp_path / 's')
        assert path == tmp_path / 's' / key / 'abcdef.json'
        assert briefs.path_for(tmp_path, '', base=tmp_path).name == \
            'nogit.json'

    def test_save_load_and_prune(self, project, tmp_path) -> None:
        base = tmp_path / 'store'
        b = brief.build(project, 'h1')
        path = briefs.save(b, base=base)
        assert path == briefs.path_for(project, 'h1', base=base)
        assert briefs.load(project, 'h1', base=base) == b
        assert briefs.load(project, 'h2', base=base) is None
        for i in range(2, 2 + briefs.KEEP + 1):
            briefs.save(brief.build(project, f'h{i}'), base=base)
        kept = sorted(p.name for p in path.parent.glob('*.json'))
        assert len(kept) == briefs.KEEP and 'h1.json' not in kept

    def test_unreadable_is_none(self, project, tmp_path) -> None:
        base = tmp_path / 'store'
        path = briefs.path_for(project, 'h1', base=base)
        path.parent.mkdir(parents=True)
        path.write_text('{not json')
        assert briefs.load(project, 'h1', base=base) is None
        path.write_text('[]')
        assert briefs.load(project, 'h1', base=base) is None
        path.write_text('{"root": "x"}')
        assert briefs.load(project, 'h1', base=base) is None

    def test_default_dir_under_guru_home(self) -> None:
        assert briefs.BRIEFS_DIR == config.GURU_HOME / 'briefs'
