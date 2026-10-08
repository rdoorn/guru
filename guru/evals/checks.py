"""Pure assertions: an :class:`Expect` against what a run :class:`Observed`.

No I/O here; the runner builds ``Observed`` and the run file stores the
``CheckResult`` rows. Only expectations that were configured produce a
result, so a case with no deterministic checks passes trivially (its rubric
is graded by hand).
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
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
    # Last lines of the fixture's pytest output when it FAILED (failing
    # test ids and the summary line); '' otherwise.
    fixture_tests_tail: str = ''
    # ``{'path', 'ref'}`` of a ``[fixture_git]`` case (traceability);
    # None for a directory fixture.
    fixture_git: Optional[dict] = None
    # Sandbox cases: the gate's verdicts (one per ``sandbox_submit``, in
    # order; the checks look at the last), ``{'image', 'digest',
    # 'build_seconds'}`` of the provisioned image, and ``skipped`` when the
    # case could not run at all (no Colima) — it then fails its checks
    # with ``error``.
    gate_verdicts: list[str] = field(default_factory=list)
    sandbox: Optional[dict] = None
    skipped: bool = False
    # The case's spend (sum of its ledger call rows; None when any call
    # could not be priced), its sub-tasks' closing statuses, and the patch
    # of what it changed in the copy ('' when nothing changed).
    cost_usd: Optional[float] = None
    task_statuses: list[str] = field(default_factory=list)
    diff_path: str = ''
    # Files the fixture's tests wrote into their private HOME.
    fixture_home_files: list[str] = field(default_factory=list)


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


def _files_changed_any(e: Expect, o: Observed) -> CheckResult:
    """Passes when at least one of the listed paths changed (the mirror of
    ``tools_used_any``: a task whose other edits are legitimately open, a
    new test file with a name the model chooses, is pinned on the file
    that must change)."""
    hit = [f for f in e.files_changed_any if f in o.files_changed]
    return CheckResult('files_changed_any', bool(hit),
                       '' if hit else f'none of {_fmt(e.files_changed_any)} '
                       f'in {_fmt(o.files_changed)}')


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
    detail = f'fixture tests {got}, expected {want}'
    if not ok and o.fixture_tests_tail:
        failing = [ln for ln in o.fixture_tests_tail.splitlines()
                   if ln.startswith(('FAILED', 'ERROR')) or ' failed' in ln]
        detail += ': ' + ' | '.join(failing[-6:] or
                                    o.fixture_tests_tail.splitlines()[-3:])
    return CheckResult('fixture_tests_pass', ok, '' if ok else detail)


def _last_verdict(o: Observed) -> str:
    return o.gate_verdicts[-1] if o.gate_verdicts else ''


def _gate_verdict(e: Expect, o: Observed) -> CheckResult:
    last = _last_verdict(o)
    if not last:
        return CheckResult('gate_verdict', False, 'no sandbox_submit '
                                                  'verdict recorded')
    ok = last == e.gate_verdict
    return CheckResult('gate_verdict', ok, '' if ok else
                       f'last gate verdict {last!r}, expected '
                       f'{e.gate_verdict!r}')


def _gate_verdict_any(e: Expect, o: Observed) -> CheckResult:
    """Passes when at least one ``sandbox_submit`` of the case ended in one
    of the listed verdicts (a controller may split the work over several
    workers, each submitting on its own)."""
    if not o.gate_verdicts:
        return CheckResult('gate_verdict_any', False, 'no sandbox_submit '
                                                      'verdict recorded')
    ok = any(v in e.gate_verdict_any for v in o.gate_verdicts)
    return CheckResult('gate_verdict_any', ok, '' if ok else
                       f'gate verdicts {_fmt(o.gate_verdicts)}, none in '
                       f'{_fmt(e.gate_verdict_any)}')


def _max_cost_usd(e: Expect, o: Observed) -> CheckResult:
    assert e.max_cost_usd is not None
    if o.cost_usd is None:
        return CheckResult('max_cost_usd', False, 'cost unknown')
    ok = o.cost_usd <= e.max_cost_usd
    return CheckResult('max_cost_usd', ok, '' if ok else
                       f'cost ${o.cost_usd:.2f} > ${e.max_cost_usd:.2f}')


def _task_status_none(e: Expect, o: Observed) -> CheckResult:
    hit = [s for s in o.task_statuses if s in e.task_status_none]
    return CheckResult('task_status_none', not hit, '' if not hit else
                       f'task statuses {_fmt(hit)} hit '
                       f'{_fmt(e.task_status_none)}')


def _fixture_home_clean(e: Expect, o: Observed) -> CheckResult:
    new = [f for f in o.fixture_home_files
           if not any(fnmatch.fnmatch(f, g) for g in e.fixture_home_allow)]
    return CheckResult('fixture_home_clean', not new, '' if not new else
                       'tests wrote to HOME: ' + _fmt(new[:5]))


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
    ('files_changed_any', lambda e: bool(e.files_changed_any),
     _files_changed_any),
    ('files_unchanged', lambda e: bool(e.files_unchanged), _files_unchanged),
    ('fixture_tests_pass', lambda e: e.fixture_tests_pass is not None,
     _fixture_tests_pass),
    ('gate_verdict', lambda e: bool(e.gate_verdict), _gate_verdict),
    ('gate_verdict_any', lambda e: bool(e.gate_verdict_any),
     _gate_verdict_any),
    ('max_cost_usd', lambda e: e.max_cost_usd is not None, _max_cost_usd),
    ('task_status_none', lambda e: bool(e.task_status_none),
     _task_status_none),
    ('fixture_home_clean', lambda e: e.fixture_home_clean,
     _fixture_home_clean),
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
