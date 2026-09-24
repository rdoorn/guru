"""Tests for the pure check evaluation (guru.evals.checks)."""
import pytest

from guru.evals.cases import Expect
from guru.evals.checks import CheckResult, Observed, evaluate, passed


def obs(**kw) -> Observed:
    base = dict(answer='Yes: a path traversal via os.path.join in upload.py',
                tools_used=['read_file', 'search_code'], spawned=2,
                roles=['security-engineer', 'reviewer'], stall_nudges=0,
                seconds=12.5, files_changed=[], fixture_tests_pass=True,
                timed_out=False)
    base.update(kw)
    return Observed(**base)


def one(expect: Expect, o: Observed) -> CheckResult:
    results = evaluate(expect, o)
    assert len(results) == 1, results
    return results[0]


class TestConfiguredOnly:
    """Only expectations that were set produce a CheckResult."""

    def test_empty_expect_yields_nothing_and_passes(self) -> None:
        assert evaluate(Expect(), obs()) == []
        assert passed([]) is True

    def test_rubric_is_not_a_check(self) -> None:
        assert evaluate(Expect(rubric='graded by hand'), obs()) == []

    def test_one_result_per_configured_expectation(self) -> None:
        e = Expect(tools_used_any=['read_file'], spawned_min=1,
                   answer_contains=['yes'], files_changed=[])
        names = [r.name for r in evaluate(e, obs())]
        assert names == ['tools_used_any', 'spawned_min',
                         'answer_contains', 'files_changed']


class TestTools:
    def test_any(self) -> None:
        e = Expect(tools_used_any=['read_file', 'web_search'])
        assert one(e, obs()).passed
        r = one(e, obs(tools_used=['spawn']))
        assert not r.passed and 'read_file' in r.detail

    def test_all(self) -> None:
        e = Expect(tools_used_all=['read_file', 'search_code'])
        assert one(e, obs()).passed
        r = one(e, obs(tools_used=['read_file']))
        assert not r.passed and 'search_code' in r.detail

    def test_none(self) -> None:
        e = Expect(tools_used_none=['delete_file'])
        assert one(e, obs()).passed
        r = one(e, obs(tools_used=['delete_file']))
        assert not r.passed and 'delete_file' in r.detail


class TestDelegation:
    def test_spawned_min_max(self) -> None:
        assert one(Expect(spawned_min=2), obs(spawned=2)).passed
        assert not one(Expect(spawned_min=2), obs(spawned=1)).passed
        assert one(Expect(spawned_max=0), obs(spawned=0)).passed
        r = one(Expect(spawned_max=0), obs(spawned=1))
        assert not r.passed and '1' in r.detail

    def test_roles_include(self) -> None:
        e = Expect(roles_include=['security-engineer'])
        assert one(e, obs()).passed
        r = one(e, obs(roles=['reviewer']))
        assert not r.passed and 'security-engineer' in r.detail

    def test_stall_nudges_max(self) -> None:
        assert one(Expect(stall_nudges_max=1), obs(stall_nudges=1)).passed
        assert not one(Expect(stall_nudges_max=0), obs(stall_nudges=1)).passed

    def test_max_seconds(self) -> None:
        assert one(Expect(max_seconds=60), obs(seconds=60.0)).passed
        r = one(Expect(max_seconds=10), obs(seconds=12.5))
        assert not r.passed and '12.5' in r.detail


class TestAnswer:
    def test_contains_is_case_insensitive(self) -> None:
        e = Expect(answer_contains=['PATH TRAVERSAL', 'upload.py'])
        assert one(e, obs()).passed
        r = one(e, obs(answer='looks fine'))
        assert not r.passed and 'path traversal' in r.detail.lower()

    def test_not_contains(self) -> None:
        e = Expect(answer_not_contains=["I'll start by"])
        assert one(e, obs()).passed
        r = one(e, obs(answer="Sure! I'LL START BY reading"))
        assert not r.passed and "i'll start by" in r.detail.lower()

    def test_regex_search_ignorecase_multiline(self) -> None:
        e = Expect(answer_regex=[r'^- swapp?ed'])
        assert one(e, obs(answer='Findings:\n- SWAPPED operands\n')).passed
        r = one(e, obs(answer='nothing'))
        assert not r.passed and 'swapp?ed' in r.detail

    def test_regex_all_patterns_must_match(self) -> None:
        e = Expect(answer_regex=['yes', 'never-there'])
        r = one(e, obs())
        assert not r.passed and 'never-there' in r.detail


class TestFiles:
    def test_files_changed_exact_set(self) -> None:
        e = Expect(files_changed=['wordcount.py'])
        assert one(e, obs(files_changed=['wordcount.py'])).passed
        r = one(e, obs(files_changed=['wordcount.py', 'tests/t.py']))
        assert not r.passed and 'tests/t.py' in r.detail

    def test_files_changed_empty_fails_when_anything_changed(self) -> None:
        e = Expect(files_changed=[])
        assert one(e, obs(files_changed=[])).passed
        r = one(e, obs(files_changed=['app/session.py']))
        assert not r.passed and 'app/session.py' in r.detail

    def test_files_unchanged(self) -> None:
        e = Expect(files_unchanged=['tests/test_wordcount.py'])
        assert one(e, obs(files_changed=['wordcount.py'])).passed
        r = one(e, obs(files_changed=['tests/test_wordcount.py']))
        assert not r.passed and 'test_wordcount' in r.detail

    def test_fixture_tests_pass(self) -> None:
        e = Expect(fixture_tests_pass=True)
        assert one(e, obs(fixture_tests_pass=True)).passed
        assert not one(e, obs(fixture_tests_pass=False)).passed
        r = one(e, obs(fixture_tests_pass=None))
        assert not r.passed and 'not run' in r.detail
        assert one(Expect(fixture_tests_pass=False),
                   obs(fixture_tests_pass=False)).passed


class TestRunFailures:
    def test_timeout_fails_every_check_with_detail_timeout(self) -> None:
        e = Expect(tools_used_any=['read_file'], answer_contains=['yes'],
                   files_changed=[])
        results = evaluate(e, obs(timed_out=True))
        assert [r.name for r in results] == ['tools_used_any',
                                             'answer_contains',
                                             'files_changed']
        assert all(not r.passed and r.detail == 'timeout' for r in results)
        assert passed(results) is False

    def test_timeout_with_no_checks_still_fails(self) -> None:
        results = evaluate(Expect(), obs(timed_out=True))
        assert [(r.name, r.passed, r.detail) for r in results] == \
            [('timeout', False, 'timeout')]

    def test_error_fails_every_check_with_the_message(self) -> None:
        results = evaluate(Expect(spawned_min=1), obs(error='boom'))
        assert [(r.name, r.passed) for r in results] == \
            [('spawned_min', False)]
        assert results[0].detail == 'error: boom'
        [only] = evaluate(Expect(), obs(error='boom'))
        assert (only.name, only.detail) == ('error', 'error: boom')


def test_passed_requires_all() -> None:
    ok = CheckResult('a', True, '')
    assert passed([ok, ok]) is True
    assert passed([ok, CheckResult('b', False, 'x')]) is False


@pytest.mark.parametrize('field', [
    'tools_used_any', 'tools_used_all', 'tools_used_none', 'roles_include',
    'answer_contains', 'answer_not_contains', 'answer_regex',
    'files_unchanged'])
def test_list_fields_default_to_unconfigured(field: str) -> None:
    assert getattr(Expect(), field) == []
