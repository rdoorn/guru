"""Tests for the eval case format (guru.evals.cases)."""
from pathlib import Path

import pytest

from guru.evals import cases

FULL = '''
name = "review-multi-file"
fixture = "flaskish"
prompt = "Review this repository for correctness and security issues."
mode = "ask-for-changes"
model = "default"
timeout_s = 240
tags = ["review", "slow"]

[expect.behaviour]
tools_used_any = ["read_file", "search_code"]
tools_used_all = ["read_file"]
tools_used_none = ["delete_file"]
spawned_min = 2
spawned_max = 4
roles_include = ["security-engineer"]
stall_nudges_max = 0
max_seconds = 200

[expect.content]
answer_contains = ["path traversal"]
answer_not_contains = ["I'll start by"]
answer_regex = ["swapp?ed|operands"]
files_changed = []
files_unchanged = ["README.md"]
fixture_tests_pass = true

[expect.rubric]
text = "Names the unchecked user path in upload.py."
'''

MINIMAL = '''
name = "greet"
fixture = "docs-only"
prompt = "hi"
'''


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    d = tmp_path / 'fixtures'
    for name in ('flaskish', 'docs-only'):
        (d / name).mkdir(parents=True)
    return d


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / f'{name}.toml'
    p.write_text(text, encoding='utf-8')
    return p


class TestLoadCase:
    """One TOML file -> Case."""

    def test_full_case_parses_every_field(self, tmp_path,
                                          fixtures_dir) -> None:
        c = cases.load_case(_write(tmp_path, 'review-multi-file', FULL),
                            fixtures_dir=fixtures_dir)
        assert c.name == 'review-multi-file'
        assert c.fixture == 'flaskish'
        assert c.prompt.startswith('Review this')
        assert c.mode == 'ask-for-changes'
        assert c.model == 'default'
        assert c.timeout_s == 240
        assert c.tags == ['review', 'slow']
        e = c.expect
        assert e.tools_used_any == ['read_file', 'search_code']
        assert e.tools_used_all == ['read_file']
        assert e.tools_used_none == ['delete_file']
        assert (e.spawned_min, e.spawned_max) == (2, 4)
        assert e.roles_include == ['security-engineer']
        assert e.stall_nudges_max == 0
        assert e.max_seconds == 200
        assert e.answer_contains == ['path traversal']
        assert e.answer_not_contains == ["I'll start by"]
        assert e.answer_regex == ['swapp?ed|operands']
        assert e.files_changed == []
        assert e.files_unchanged == ['README.md']
        assert e.fixture_tests_pass is True
        assert e.rubric.startswith('Names the unchecked')

    def test_minimal_case_gets_defaults(self, tmp_path, fixtures_dir) -> None:
        c = cases.load_case(_write(tmp_path, 'greet', MINIMAL),
                            fixtures_dir=fixtures_dir)
        assert (c.mode, c.model, c.timeout_s, c.tags) == \
            ('ask-for-changes', 'default', 300, [])
        assert c.expect == cases.Expect()
        assert c.expect.files_changed is None
        assert c.expect.spawned_max is None

    def test_default_fixtures_dir_is_the_repo_checkout(self, tmp_path) -> None:
        c = cases.load_case(_write(tmp_path, 'greet', MINIMAL))
        assert c.fixture == 'docs-only'
        assert (cases.FIXTURES_DIR / 'docs-only' / 'FIXTURE.md').exists()

    def test_missing_fixture_names_the_path(self, tmp_path,
                                            fixtures_dir) -> None:
        text = MINIMAL.replace('docs-only', 'nope')
        with pytest.raises(ValueError) as ei:
            cases.load_case(_write(tmp_path, 'greet', text),
                            fixtures_dir=fixtures_dir)
        assert 'nope' in str(ei.value)
        assert str(fixtures_dir / 'nope') in str(ei.value)

    @pytest.mark.parametrize('missing', ['name', 'fixture', 'prompt'])
    def test_required_keys(self, tmp_path, fixtures_dir, missing) -> None:
        lines = [ln for ln in MINIMAL.splitlines()
                 if not ln.startswith(missing)]
        with pytest.raises(ValueError, match=missing):
            cases.load_case(_write(tmp_path, 'x', '\n'.join(lines)),
                            fixtures_dir=fixtures_dir)

    def test_unknown_expect_key_raises_naming_it(self, tmp_path,
                                                 fixtures_dir) -> None:
        text = MINIMAL + '\n[expect.behaviour]\ntools_used_ani = ["x"]\n'
        with pytest.raises(ValueError, match='tools_used_ani'):
            cases.load_case(_write(tmp_path, 'x', text),
                            fixtures_dir=fixtures_dir)

    def test_unknown_expect_section_raises(self, tmp_path,
                                           fixtures_dir) -> None:
        text = MINIMAL + '\n[expect.behavior]\nspawned_min = 1\n'
        with pytest.raises(ValueError, match='behavior'):
            cases.load_case(_write(tmp_path, 'x', text),
                            fixtures_dir=fixtures_dir)

    def test_unknown_top_level_key_raises(self, tmp_path,
                                          fixtures_dir) -> None:
        text = MINIMAL + 'timeout = 10\n'
        with pytest.raises(ValueError, match='timeout'):
            cases.load_case(_write(tmp_path, 'x', text),
                            fixtures_dir=fixtures_dir)

    def test_wrong_type_raises(self, tmp_path, fixtures_dir) -> None:
        text = MINIMAL + '\n[expect.behaviour]\nspawned_min = "two"\n'
        with pytest.raises(ValueError, match='spawned_min'):
            cases.load_case(_write(tmp_path, 'x', text),
                            fixtures_dir=fixtures_dir)
        text = MINIMAL + '\n[expect.content]\nanswer_contains = "yes"\n'
        with pytest.raises(ValueError, match='answer_contains'):
            cases.load_case(_write(tmp_path, 'x', text),
                            fixtures_dir=fixtures_dir)

    def test_unknown_mode_raises(self, tmp_path, fixtures_dir) -> None:
        text = MINIMAL + 'mode = "yolo"\n'
        with pytest.raises(ValueError, match='yolo'):
            cases.load_case(_write(tmp_path, 'x', text),
                            fixtures_dir=fixtures_dir)

    def test_rubric_must_be_text(self, tmp_path, fixtures_dir) -> None:
        text = MINIMAL + '\n[expect.rubric]\nscore = 2\n'
        with pytest.raises(ValueError, match='score'):
            cases.load_case(_write(tmp_path, 'x', text),
                            fixtures_dir=fixtures_dir)

    def test_invalid_regex_raises_naming_file_and_pattern(
            self, tmp_path, fixtures_dir) -> None:
        text = MINIMAL + '\n[expect.content]\nanswer_regex = ["ok", "(oops"]\n'
        with pytest.raises(ValueError) as ei:
            cases.load_case(_write(tmp_path, 'rx', text),
                            fixtures_dir=fixtures_dir)
        assert 'rx.toml' in str(ei.value) and '(oops' in str(ei.value)

    def test_max_seconds_int_is_coerced_to_float(
            self, tmp_path, fixtures_dir) -> None:
        text = MINIMAL + '\n[expect.behaviour]\nmax_seconds = 60\n'
        c = cases.load_case(_write(tmp_path, 'x', text),
                            fixtures_dir=fixtures_dir)
        assert c.expect.max_seconds == 60.0
        assert isinstance(c.expect.max_seconds, float)

    def test_bad_toml_raises_value_error_with_path(self, tmp_path,
                                                   fixtures_dir) -> None:
        p = _write(tmp_path, 'broken', 'name = \n')
        with pytest.raises(ValueError, match='broken.toml'):
            cases.load_case(p, fixtures_dir=fixtures_dir)


class TestLoadCases:
    """A directory of cases, optionally filtered by name."""

    def test_loads_sorted_and_filters(self, tmp_path, fixtures_dir) -> None:
        d = tmp_path / 'cases'
        d.mkdir()
        _write(d, 'b-case', MINIMAL.replace('greet', 'b-case'))
        _write(d, 'a-case', MINIMAL.replace('greet', 'a-case'))
        (d / 'notes.md').write_text('ignored')
        got = cases.load_cases(d, fixtures_dir=fixtures_dir)
        assert [c.name for c in got] == ['a-case', 'b-case']
        only = cases.load_cases(d, names=['b-case'], fixtures_dir=fixtures_dir)
        assert [c.name for c in only] == ['b-case']

    def test_unknown_name_raises(self, tmp_path, fixtures_dir) -> None:
        d = tmp_path / 'cases'
        d.mkdir()
        _write(d, 'greet', MINIMAL)
        with pytest.raises(ValueError, match='nope'):
            cases.load_cases(d, names=['nope'], fixtures_dir=fixtures_dir)

    def test_duplicate_names_raise(self, tmp_path, fixtures_dir) -> None:
        d = tmp_path / 'cases'
        d.mkdir()
        _write(d, 'one', MINIMAL)
        _write(d, 'two', MINIMAL)
        with pytest.raises(ValueError, match='greet'):
            cases.load_cases(d, fixtures_dir=fixtures_dir)

    def test_filter_by_tags_selects_any_match(self, tmp_path,
                                              fixtures_dir) -> None:
        d = tmp_path / 'cases'
        d.mkdir()
        _write(d, 'a', MINIMAL.replace('greet', 'a') + 'tags = ["fast"]\n')
        _write(d, 'b', MINIMAL.replace('greet', 'b')
               + 'tags = ["edit", "slow"]\n')
        _write(d, 'c', MINIMAL.replace('greet', 'c'))
        got = cases.load_cases(d, tags=['fast', 'edit'],
                               fixtures_dir=fixtures_dir)
        assert [c.name for c in got] == ['a', 'b']
        only = cases.load_cases(d, tags=['slow'], fixtures_dir=fixtures_dir)
        assert [c.name for c in only] == ['b']
        # combinable with names: both filters apply
        both = cases.load_cases(d, names=['a', 'c'], tags=['fast', 'edit'],
                                fixtures_dir=fixtures_dir)
        assert [c.name for c in both] == ['a']

    def test_unknown_tag_raises_listing_available(self, tmp_path,
                                                  fixtures_dir) -> None:
        d = tmp_path / 'cases'
        d.mkdir()
        _write(d, 'a', MINIMAL + 'tags = ["fast"]\n')
        with pytest.raises(ValueError) as ei:
            cases.load_cases(d, tags=['nope'], fixtures_dir=fixtures_dir)
        assert 'nope' in str(ei.value) and 'fast' in str(ei.value)


class TestShippedCases:
    """The committed cases under evals/cases."""

    FAST = {'greet', 'trivial-fact', 'explain-readme', 'find-symbol',
            'security-only', 'logic-bug'}

    def test_fast_gate_is_tagged_and_short(self) -> None:
        fast = cases.load_cases(cases.CASES_DIR, tags=['fast'])
        assert {c.name for c in fast} == self.FAST
        assert all(c.timeout_s == 120 for c in fast)

    def test_fast_cases_keep_their_other_tags(self) -> None:
        by_name = {c.name: c for c in cases.load_cases(cases.CASES_DIR)}
        assert 'read' in by_name['explain-readme'].tags
        assert 'search' in by_name['find-symbol'].tags
        assert {'review', 'security'} <= set(by_name['security-only'].tags)
        assert 'review' in by_name['logic-bug'].tags
        assert 'no-tools' in by_name['greet'].tags
