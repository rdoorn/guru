"""Pure assertions: an :class:`Expect` against what a run :class:`Observed`.

No I/O here; the runner builds ``Observed`` and the run file stores the
``CheckResult`` rows. Only expectations that were configured produce a
result, so a case with no deterministic checks passes trivially (its rubric
is graded by hand).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from guru.evals.cases import Expect


@dataclass
class Observed:
    """What one case run produced."""
    answer: str
    tools_used: list[str]
    spawned: int
    roles: list[str]
    stall_nudges: int
    seconds: float
    files_changed: list[str]
    fixture_tests_pass: Optional[bool]
    timed_out: bool
    error: str = ''


@dataclass
class CheckResult:
    """Outcome of one expectation; ``detail`` explains a failure."""
    name: str
    passed: bool
    detail: str


def _fmt(items: list[str]) -> str:
    return '[' + ', '.join(repr(i) for i in items) + ']'


def _tools_used_any(e: Expect, o: Observed) -> CheckResult:
    hit = [t for t in e.tools_used_any if t in o.tools_used]
    return CheckResult('tools_used_any', bool(hit),
                       '' if hit else f'none of {_fmt(e.tools_used_any)} '
                       f'in {_fmt(o.tools_used)}')


def _tools_used_all(e: Expect, o: Observed) -> CheckResult:
    missing = [t for t in e.tools_used_all if t not in o.tools_used]
    return CheckResult('tools_used_all', not missing,
                       '' if not missing else f'missing {_fmt(missing)} '
                       f'from {_fmt(o.tools_used)}')


def _tools_used_none(e: Expect, o: Observed) -> CheckResult:
    bad = [t for t in e.tools_used_none if t in o.tools_used]
    return CheckResult('tools_used_none', not bad,
                       '' if not bad else f'forbidden tool(s) used: '
                       f'{_fmt(bad)}')


def _spawned_min(e: Expect, o: Observed) -> CheckResult:
    ok = o.spawned >= e.spawned_min
    return CheckResult('spawned_min', ok, '' if ok else
                       f'spawned {o.spawned} < {e.spawned_min}')


def _spawned_max(e: Expect, o: Observed) -> CheckResult:
    ok = e.spawned_max is not None and o.spawned <= e.spawned_max
    return CheckResult('spawned_max', ok, '' if ok else
                       f'spawned {o.spawned} > {e.spawned_max}')


def _roles_include(e: Expect, o: Observed) -> CheckResult:
    missing = [r for r in e.roles_include if r not in o.roles]
    return CheckResult('roles_include', not missing,
                       '' if not missing else f'missing role(s) '
                       f'{_fmt(missing)} from {_fmt(o.roles)}')


def _stall_nudges_max(e: Expect, o: Observed) -> CheckResult:
    ok = e.stall_nudges_max is not None and \
        o.stall_nudges <= e.stall_nudges_max
    return CheckResult('stall_nudges_max', ok, '' if ok else
                       f'{o.stall_nudges} stall nudge(s) > '
                       f'{e.stall_nudges_max}')


def _max_seconds(e: Expect, o: Observed) -> CheckResult:
    ok = e.max_seconds is not None and o.seconds <= e.max_seconds
    return CheckResult('max_seconds', ok, '' if ok else
                       f'took {o.seconds:g}s > {e.max_seconds:g}s')


def _answer_contains(e: Expect, o: Observed) -> CheckResult:
    low = o.answer.lower()
    missing = [s for s in e.answer_contains if s.lower() not in low]
    return CheckResult('answer_contains', not missing,
                       '' if not missing else f'answer lacks {_fmt(missing)}')


def _answer_not_contains(e: Expect, o: Observed) -> CheckResult:
    low = o.answer.lower()
    found = [s for s in e.answer_not_contains if s.lower() in low]
    return CheckResult('answer_not_contains', not found,
                       '' if not found else f'answer contains {_fmt(found)}')


def _answer_regex(e: Expect, o: Observed) -> CheckResult:
    flags = re.IGNORECASE | re.MULTILINE
    missing = [p for p in e.answer_regex if not re.search(p, o.answer, flags)]
    return CheckResult('answer_regex', not missing,
                       '' if not missing else f'no match for {_fmt(missing)}')


def _files_changed(e: Expect, o: Observed) -> CheckResult:
    want = sorted(set(e.files_changed or []))
    got = sorted(set(o.files_changed))
    if want == got:
        return CheckResult('files_changed', True, '')
    extra = [f for f in got if f not in want]
    missing = [f for f in want if f not in got]
    parts: list[str] = []
    if extra:
        parts.append(f'unexpected {_fmt(extra)}')
    if missing:
        parts.append(f'not changed {_fmt(missing)}')
    return CheckResult('files_changed', False, '; '.join(parts))


def _files_unchanged(e: Expect, o: Observed) -> CheckResult:
    bad = [f for f in e.files_unchanged if f in o.files_changed]
    return CheckResult('files_unchanged', not bad,
                       '' if not bad else f'changed {_fmt(bad)}')


def _fixture_tests_pass(e: Expect, o: Observed) -> CheckResult:
    if o.fixture_tests_pass is None:
        return CheckResult('fixture_tests_pass', False,
                           'fixture tests not run')
    ok = o.fixture_tests_pass is e.fixture_tests_pass
    got = 'passed' if o.fixture_tests_pass else 'failed'
    want = 'pass' if e.fixture_tests_pass else 'fail'
    return CheckResult('fixture_tests_pass', ok, '' if ok else
                       f'fixture tests {got}, expected {want}')


# (name, is-configured predicate, check) in the order results are reported.
_Check = Callable[[Expect, Observed], CheckResult]
_CHECKS: list[tuple[str, Callable[[Expect], bool], _Check]] = [
    ('tools_used_any', lambda e: bool(e.tools_used_any), _tools_used_any),
    ('tools_used_all', lambda e: bool(e.tools_used_all), _tools_used_all),
    ('tools_used_none', lambda e: bool(e.tools_used_none), _tools_used_none),
    ('spawned_min', lambda e: e.spawned_min > 0, _spawned_min),
    ('spawned_max', lambda e: e.spawned_max is not None, _spawned_max),
    ('roles_include', lambda e: bool(e.roles_include), _roles_include),
    ('stall_nudges_max', lambda e: e.stall_nudges_max is not None,
     _stall_nudges_max),
    ('max_seconds', lambda e: e.max_seconds is not None, _max_seconds),
    ('answer_contains', lambda e: bool(e.answer_contains), _answer_contains),
    ('answer_not_contains', lambda e: bool(e.answer_not_contains),
     _answer_not_contains),
    ('answer_regex', lambda e: bool(e.answer_regex), _answer_regex),
    ('files_changed', lambda e: e.files_changed is not None, _files_changed),
    ('files_unchanged', lambda e: bool(e.files_unchanged), _files_unchanged),
    ('fixture_tests_pass', lambda e: e.fixture_tests_pass is not None,
     _fixture_tests_pass),
]


def configured(expect: Expect) -> list[str]:
    """Names of the expectations that ``expect`` actually sets."""
    return [name for name, is_set, _ in _CHECKS if is_set(expect)]


def evaluate(expect: Expect, obs: Observed) -> list[CheckResult]:
    """One :class:`CheckResult` per configured expectation.

    A timed-out run fails every configured check with detail ``'timeout'``;
    a run that errored fails them with ``'error: <message>'``. When nothing
    is configured, such a run still yields one failing result so it cannot
    pass by having no checks.
    """
    failure_name, failure = '', ''
    if obs.timed_out:
        failure_name, failure = 'timeout', 'timeout'
    elif obs.error:
        failure_name, failure = 'error', f'error: {obs.error}'
    active = [(name, check) for name, is_set, check in _CHECKS
              if is_set(expect)]
    if failure:
        if not active:
            return [CheckResult(failure_name, False, failure)]
        return [CheckResult(name, False, failure) for name, _ in active]
    return [check(expect, obs) for _, check in active]


def passed(results: list[CheckResult]) -> bool:
    """True when every result passed (vacuously true for no results)."""
    return all(r.passed for r in results)
