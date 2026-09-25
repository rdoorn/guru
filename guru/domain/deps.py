"""Dependency requests and lockfile diffs (design plan
``2026-09-24-sandbox-design-and-plan.md`` §1 domain, chunk S2).

Rule: an install is only a lockfile change. The model never installs; it
records a :class:`DependencyRequest` (a PEP 508 project name plus an
optional version-specifier string) which the user approves, after which
guru runs ``uv add`` in a provisioning container, brings the resulting
``pyproject.toml``/``uv.lock`` back through ``apply_patch`` and rebuilds
the image. :func:`lock_diff_summary` says what changed between two
``uv.lock`` texts (packages added, removed, version-changed) so the
approval shows the consequence, not just the request.

Validation is strict on purpose: the request text lands in a fixed argv
(``uv add <name><constraint>``), so a name is the PEP 508 name grammar
only (no extras, no markers, no URLs) and a constraint is a comma list of
``<operator><version>`` with no whitespace. Stdlib only.
"""
from __future__ import annotations

import difflib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

MAX_LEN = 200
# PEP 508 / PEP 503 project name grammar.
_NAME_RX = re.compile(r'[A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9]')
_NORMALISE_RX = re.compile(r'[-_.]+')
# One PEP 440 clause: operator then a version (letters, digits, . * + ! -).
_CLAUSE = r'(?:===|==|!=|<=|>=|~=|<|>)[A-Za-z0-9.*+!-]+'
_CONSTRAINT_RX = re.compile(rf'{_CLAUSE}(?:,{_CLAUSE})*')


@dataclass(frozen=True)
class DependencyRequest:
    """A requested package: ``name`` as typed, ``constraint`` such as
    ``>=1.2,<2`` or ``''``, and when it was requested (ISO, set by the
    store)."""
    name: str
    constraint: str = ''
    requested_at: str = ''

    @property
    def spec(self) -> str:
        """The requirement string ``uv add`` receives: ``name+constraint``."""
        return f'{self.name}{self.constraint}'

    @property
    def key(self) -> str:
        """The PEP 503 normalised name (lower-case, runs of ``-_.`` as one
        ``-``), for de-duplication and matching."""
        return normalise(self.name)


def normalise(name: str) -> str:
    """PEP 503 name normalisation."""
    return _NORMALISE_RX.sub('-', str(name)).lower()


def check_name(name: object) -> str:
    """``''`` when ``name`` is a valid PEP 508 project name, else why not."""
    if not isinstance(name, str) or not name:
        return 'dependency name is empty'
    if len(name) > MAX_LEN:
        return f'dependency name longer than {MAX_LEN} characters'
    if not _NAME_RX.fullmatch(name):
        return (f'dependency name {name!r} is not a PEP 508 project name '
                '(letters, digits, . _ -; no extras, markers or URLs)')
    return ''


def check_constraint(constraint: object) -> str:
    """``''`` when ``constraint`` is empty or a comma-separated list of
    ``<operator><version>`` clauses without whitespace, else why not."""
    if constraint is None or constraint == '':
        return ''
    if not isinstance(constraint, str):
        return 'dependency constraint must be a string'
    if len(constraint) > MAX_LEN:
        return f'dependency constraint longer than {MAX_LEN} characters'
    if not _CONSTRAINT_RX.fullmatch(constraint):
        return (f'dependency constraint {constraint!r} is not a version '
                'specifier (e.g. ">=1.2,<2", "==1.0"; operators == != <= '
                '>= ~= < > ===; no spaces, markers or URLs)')
    return ''


def request_from(name: str, constraint: str = '') -> DependencyRequest:
    """A validated :class:`DependencyRequest`; ``ValueError`` naming the
    offending part otherwise."""
    problem = check_name(name) or check_constraint(constraint)
    if problem:
        raise ValueError(problem)
    return DependencyRequest(name=name, constraint=constraint or '')


def lock_packages(text: str) -> dict[str, str]:
    """``{name: version}`` for every ``[[package]]`` of a ``uv.lock`` text
    (packages without a version are recorded as ``''``). ``ValueError``
    on invalid TOML."""
    try:
        data = tomllib.loads(text or '')
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f'uv.lock is not valid TOML: {e}') from e
    out: dict[str, str] = {}
    for pkg in data.get('package', []) or []:
        if isinstance(pkg, dict) and isinstance(pkg.get('name'), str):
            out[pkg['name']] = str(pkg.get('version', '') or '')
    return out


def lock_diff_summary(old: str, new: str) -> dict[str, list[str]]:
    """What changed between two ``uv.lock`` texts: ``added``
    (``name==version``), ``removed`` (same) and ``changed``
    (``name: old -> new``), each sorted."""
    before, after = lock_packages(old), lock_packages(new)
    added = sorted(f'{n}=={after[n]}' for n in set(after) - set(before))
    removed = sorted(f'{n}=={before[n]}' for n in set(before) - set(after))
    changed = sorted(f'{n}: {before[n]} -> {after[n]}'
                     for n in set(before) & set(after)
                     if before[n] != after[n])
    return {'added': added, 'removed': removed, 'changed': changed}


def summary_text(summary: dict[str, list[str]]) -> str:
    """One line per non-empty part of :func:`lock_diff_summary`;
    ``no package changes`` when all are empty."""
    parts = [f'{kind} ' + ', '.join(items)
             for kind in ('added', 'removed', 'changed')
             for items in [summary.get(kind) or []] if items]
    return '; '.join(parts) if parts else 'no package changes'


def unified_diff(path: Path, old: str, new: str) -> str:
    """A unified diff turning ``old`` into ``new`` for the file ``path``
    (absolute path in both headers, so ``apply_patch`` resolves it from any
    working directory); ``''`` when the texts are equal. Both texts are
    treated as newline-terminated."""
    if old == new:
        return ''
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(), fromfile=str(path),
        tofile=str(path), lineterm='', n=3))
    return '\n'.join(lines) + '\n'
