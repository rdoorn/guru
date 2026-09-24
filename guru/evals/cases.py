"""Eval case format: ``evals/cases/<name>.toml`` -> :class:`Case`.

A case is one prompt against one fixture plus the expectations under
``[expect.behaviour]`` (guru's choices), ``[expect.content]`` (the answer and
the repo) and ``[expect.rubric]`` (text graded by hand during triage).
Unknown keys anywhere raise ``ValueError`` naming the key, so typos cannot
silently disable a check.

The fixture is either ``fixture = "<name>"`` (a frozen directory under
``evals/fixtures``) or a ``[fixture_git]`` table (``path`` to a local git
repository, absolute or relative to this checkout, and ``ref``, a commit sha
or tag): the runner archives the repo at that ref, so a real project can be
a fixture without being copied into the suite. Exactly one of the two.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from guru import config

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / 'evals' / 'fixtures'
CASES_DIR = REPO_ROOT / 'evals' / 'cases'

DEFAULT_MODE = config.MODE_ASK
DEFAULT_MODEL = 'default'
DEFAULT_TIMEOUT_S = 300


@dataclass
class Expect:
    """Deterministic expectations; a default value means "do not check"."""
    tools_used_any: list[str] = field(default_factory=list)
    tools_used_all: list[str] = field(default_factory=list)
    tools_used_none: list[str] = field(default_factory=list)
    spawned_min: int = 0
    spawned_max: Optional[int] = None
    roles_include: list[str] = field(default_factory=list)
    stall_nudges_max: Optional[int] = None
    max_seconds: Optional[float] = None
    answer_contains: list[str] = field(default_factory=list)
    answer_not_contains: list[str] = field(default_factory=list)
    answer_regex: list[str] = field(default_factory=list)
    files_changed: Optional[list[str]] = None  # exact set; None = no check
    files_unchanged: list[str] = field(default_factory=list)
    fixture_tests_pass: Optional[bool] = None
    rubric: str = ''


@dataclass(frozen=True)
class GitFixture:
    """A local git repository pinned to ``ref`` (commit sha or tag)."""
    path: Path
    ref: str

    @property
    def label(self) -> str:
        """``git:<dir>@<ref>`` with a full sha cut to 7 characters."""
        ref = self.ref
        if len(ref) == 40 and all(c in '0123456789abcdef' for c in ref):
            ref = ref[:7]
        return f'git:{self.path.name}@{ref}'


@dataclass
class Case:
    """One eval case: prompt, fixture, run settings and expectations.

    ``fixture`` names a directory fixture; ``fixture_git`` pins a git
    repository instead (then ``fixture`` is ``''``).
    """
    name: str
    fixture: str
    prompt: str
    mode: str = DEFAULT_MODE
    model: str = DEFAULT_MODEL
    timeout_s: int = DEFAULT_TIMEOUT_S
    tags: list[str] = field(default_factory=list)
    expect: Expect = field(default_factory=Expect)
    fixture_git: Optional[GitFixture] = None

    @property
    def fixture_label(self) -> str:
        """What ``list`` shows: the fixture name or the git pin."""
        return (self.fixture_git.label if self.fixture_git is not None
                else self.fixture)


# Accepted TOML value types per key. ``int`` is also accepted where a float
# is expected; ``bool`` is never accepted as an int.
_TOP_TYPES: dict[str, type] = {
    'name': str, 'fixture': str, 'prompt': str, 'mode': str, 'model': str,
    'timeout_s': int, 'tags': list, 'fixture_git': dict,
}
_REQUIRED = ('name', 'prompt')
_GIT_KEYS = ('path', 'ref')
_EXPECT_TYPES: dict[str, type] = {
    'tools_used_any': list, 'tools_used_all': list, 'tools_used_none': list,
    'spawned_min': int, 'spawned_max': int, 'roles_include': list,
    'stall_nudges_max': int, 'max_seconds': float,
    'answer_contains': list, 'answer_not_contains': list,
    'answer_regex': list, 'files_changed': list, 'files_unchanged': list,
    'fixture_tests_pass': bool,
}
_EXPECT_SECTIONS = ('behaviour', 'content', 'rubric')
assert set(_EXPECT_TYPES) | {'rubric'} == {f.name for f in fields(Expect)}


def _typed(where: str, key: str, value: Any, want: type) -> Any:
    """Return ``value`` checked against ``want`` (list -> list of str).

    An int is coerced to float where a float is wanted.
    """
    is_bool = isinstance(value, bool)
    ok = isinstance(value, want) and not (want is int and is_bool)
    if want is float and isinstance(value, int) and not is_bool:
        value, ok = float(value), True
    if want is list and ok:
        ok = all(isinstance(v, str) for v in value)
    if not ok:
        raise ValueError(f'{where}: {key} must be {want.__name__}'
                         f'{" of str" if want is list else ""}, '
                         f'got {value!r}')
    return value


def _compiled_ok(where: str, patterns: list[str]) -> list[str]:
    """Return ``patterns`` after compiling each with the runtime flags."""
    for pat in patterns:
        try:
            re.compile(pat, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            raise ValueError(f'{where}: invalid answer_regex {pat!r}: {e}') \
                from e
    return patterns


def _build_expect(where: str, raw: Any) -> Expect:
    if not isinstance(raw, dict):
        raise ValueError(f'{where}: [expect] must be a table')
    kw: dict[str, Any] = {}
    for section, body in raw.items():
        if section not in _EXPECT_SECTIONS:
            raise ValueError(f'{where}: unknown section [expect.{section}] '
                             f'(known: {", ".join(_EXPECT_SECTIONS)}; keys '
                             'belong under [expect.behaviour] or '
                             '[expect.content])')
        if not isinstance(body, dict):
            raise ValueError(f'{where}: [expect.{section}] must be a table')
        for key, value in body.items():
            if section == 'rubric':
                if key != 'text':
                    raise ValueError(f'{where}: unknown key '
                                     f'[expect.rubric].{key} (only "text")')
                kw['rubric'] = _typed(where, 'text', value, str)
                continue
            if key not in _EXPECT_TYPES:
                raise ValueError(
                    f'{where}: unknown key [expect.{section}].{key} '
                    f'(known: {", ".join(sorted(_EXPECT_TYPES))})')
            if key in kw:
                raise ValueError(f'{where}: {key} given twice')
            kw[key] = _typed(where, key, value, _EXPECT_TYPES[key])
    if 'answer_regex' in kw:
        _compiled_ok(where, kw['answer_regex'])
    return Expect(**kw)


def _build_git_fixture(where: str, raw: dict) -> GitFixture:
    """``[fixture_git]`` -> :class:`GitFixture`: both keys required, no
    others; a relative ``path`` is taken from the checkout root; the path
    must hold a git repository (a ``.git`` entry). The ref itself is
    checked when the fixture is prepared (``git archive`` fails)."""
    for key in raw:
        if key not in _GIT_KEYS:
            raise ValueError(f'{where}: unknown key [fixture_git].{key} '
                             f'(known: {", ".join(_GIT_KEYS)})')
    for key in _GIT_KEYS:
        if key not in raw:
            raise ValueError(f'{where}: [fixture_git] needs {key}')
        if not _typed(where, f'[fixture_git].{key}', raw[key], str).strip():
            raise ValueError(f'{where}: [fixture_git].{key} is empty')
    repo = Path(raw['path']).expanduser()
    if not repo.is_absolute():
        repo = REPO_ROOT / repo
    repo = repo.resolve()
    if not repo.is_dir() or not (repo / '.git').exists():
        raise ValueError(f'{where}: fixture_git path {raw["path"]!r} is '
                         f'not a git repository ({repo})')
    return GitFixture(repo, raw['ref'].strip())


def load_case(path: Path, fixtures_dir: Optional[Path] = None) -> Case:
    """Parse one case file; raise ``ValueError`` on any problem.

    Exactly one of ``fixture`` and ``[fixture_git]`` must be given. A
    fixture directory must exist under ``fixtures_dir`` (default:
    ``evals/fixtures`` in this checkout).
    """
    path = Path(path)
    where = str(path)
    try:
        raw = tomllib.loads(path.read_text(encoding='utf-8'))
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f'{where}: invalid TOML: {e}') from e
    for key in _REQUIRED:
        if key not in raw:
            raise ValueError(f'{where}: missing required key {key!r}')
    kw: dict[str, Any] = {}
    for key, value in raw.items():
        if key == 'expect':
            continue
        if key not in _TOP_TYPES:
            raise ValueError(f'{where}: unknown key {key!r} '
                             f'(known: {", ".join(_TOP_TYPES)}, expect)')
        kw[key] = _typed(where, key, value, _TOP_TYPES[key])
    if kw.get('mode', DEFAULT_MODE) not in config.MODES:
        raise ValueError(f'{where}: mode {kw["mode"]!r} not one of '
                         f'{", ".join(config.MODES)}')
    if ('fixture' in kw) == ('fixture_git' in kw):
        raise ValueError(f'{where}: give exactly one of fixture (a name '
                         'under evals/fixtures) or [fixture_git] (path, '
                         'ref)')
    if 'fixture_git' in kw:
        kw['fixture_git'] = _build_git_fixture(where, kw['fixture_git'])
        kw['fixture'] = ''
    else:
        base = (Path(fixtures_dir) if fixtures_dir is not None
                else FIXTURES_DIR)
        fixture_path = base / kw['fixture']
        if not fixture_path.is_dir():
            raise ValueError(f'{where}: fixture {kw["fixture"]!r} not found '
                             f'at {fixture_path}')
    kw['expect'] = _build_expect(where, raw.get('expect', {}))
    return Case(**kw)


def load_cases(directory: Path, names: Optional[list[str]] = None,
               fixtures_dir: Optional[Path] = None,
               tags: Optional[list[str]] = None) -> list[Case]:
    """Load every ``*.toml`` in ``directory`` (sorted by file name).

    ``names`` restricts the result to those case names; ``tags`` to cases
    carrying ANY of the tags. Both filters apply when both are given. A
    name or tag that matches no case, or two files sharing a case name,
    raise ``ValueError``.
    """
    directory = Path(directory)
    loaded = [load_case(p, fixtures_dir=fixtures_dir)
              for p in sorted(directory.glob('*.toml'))]
    seen: dict[str, Case] = {}
    for c in loaded:
        if c.name in seen:
            raise ValueError(f'duplicate case name {c.name!r} in {directory}')
        seen[c.name] = c
    if names is not None:
        unknown = [n for n in names if n not in seen]
        if unknown:
            raise ValueError(f'unknown case(s): {", ".join(unknown)} '
                             f'(available: {", ".join(seen)})')
        loaded = [c for c in loaded if c.name in set(names)]
    if tags is not None:
        known = sorted({t for c in seen.values() for t in c.tags})
        unknown = [t for t in tags if t not in known]
        if unknown:
            raise ValueError(f'unknown tag(s): {", ".join(unknown)} '
                             f'(available: {", ".join(known)})')
        loaded = [c for c in loaded if set(c.tags) & set(tags)]
    return loaded
