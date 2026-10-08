"""Project brief: what a worker needs to know about a project before its
first tool call (structural round, Package C item 4).

``build(root, head_sha)`` walks the project once and returns a :class:`Brief`:
the file map (directories with file counts, top-level modules), a capped
outline per Python module, a symbol index (def/class name -> ``path:line``),
how to run the tests (Makefile target, pytest or unittest) and the
conventions declared in ``pyproject.toml`` ``[tool.*]`` sections.
``slice(brief, task_text)`` cuts one down to what a task mentions -- the map,
the test command and the modules whose names or symbols appear in the task
text -- under a token budget, so a worker's system context carries it and
the controller's sees the map (``render_map``). ``current(root, store)``
builds or loads the brief for the checked-out HEAD through the store the
caller passes (a :class:`BriefStore`; ``guru.repositories.briefs`` is the
one the endpoints use -- this module imports no repository). The ``/brief``
endpoint body lives in ``guru.briefcmd``.

Pure: the only I/O is reading the project tree (through
``files.walk_files``) and one fixed-argv ``git rev-parse`` through
``procs.run``. Stdlib only. Outline signatures carry no default values
(``code.tree_rows(defaults=False)``): the brief lands in a system prompt,
so no project string literal travels there verbatim.
"""
from __future__ import annotations

import ast
import configparser
import re
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

from guru.domain import code, files, procs

# Budget arithmetic for ``slice``: a 4-chars-per-token estimate.
CHARS_PER_TOKEN = 4
SLICE_MAX_TOKENS = 1500
# Build caps: rows kept per module outline, modules outlined, symbol index
# entries, directory depth in the map, and the wall-clock budget after
# which the walk stops parsing (it keeps counting files).
OUTLINE_ROWS_PER_MODULE = 40
MAX_MODULES = 600
MAX_SYMBOLS = 6000
MAP_DEPTH = 2
BUILD_BUDGET_S = 3.0
# Rows of one module shown in a slice, and the head sha used when the
# project is not a git checkout (such a brief is never stored).
SLICE_ROWS_PER_FILE = 25
NO_HEAD = 'nogit'
# ``make <target>`` lines: a plain target name at column 0.
_MAKE_TARGET = re.compile(r'^([A-Za-z][\w.-]*)\s*:(?!=)', re.M)
_WORD = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_PATHISH = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]+')
# A prose word counts as a symbol mention only when it looks like code: an
# underscore, CamelCase, or at least this long ("check" is prose; "run_tests",
# "Orchestrator" and "compaction" are names).
_MIN_SYMBOL_LEN = 6


@dataclass
class Brief:
    """One project's brief at one HEAD; JSON-able through ``to_dict``."""
    root: str
    head_sha: str
    built_at: str
    build_seconds: float
    files: int
    python_files: int
    dirs: dict = field(default_factory=dict)        # 'guru/domain' -> 22
    modules: list = field(default_factory=list)     # top-level packages
    outlines: dict = field(default_factory=dict)    # relpath -> [rows]
    symbols: dict = field(default_factory=dict)     # name -> ['p.py:12']
    test_command: str = ''
    make_targets: list = field(default_factory=list)
    conventions: dict = field(default_factory=dict)  # tool -> {key: value}
    truncated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Brief':
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class BriefStore(Protocol):
    """Where briefs persist (``guru.repositories.briefs`` implements it)."""

    def load(self, root: Path, head_sha: str) -> Optional[Brief]: ...

    def save(self, brief: Brief) -> Path: ...


# --- build -------------------------------------------------------------------

def head_sha(root: Path) -> str:
    """The checked-out commit of ``root`` through a fixed-argv ``git
    rev-parse HEAD`` (``procs.run``: no shell, scrubbed environment,
    limits); ``NO_HEAD`` when ``root`` is not a git checkout or the read
    allow-list refuses it."""
    res = procs.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], root,
                    procs.Limits(timeout_s=10))
    sha = res.stdout.strip()
    if res.returncode != 0 or not re.fullmatch(r'[0-9a-f]{40}', sha):
        return NO_HEAD
    return sha


def _dir_key(rel: Path) -> str:
    parts = rel.parent.parts[:MAP_DEPTH]
    return '/'.join(parts) if parts else '.'


def _scalar(value: object) -> bool:
    return isinstance(value, (str, int, float, bool))


def _conventions(root: Path) -> dict:
    """``{tool: {key: value}}`` from ``pyproject.toml`` ``[tool.*]`` (scalar
    keys and short scalar lists), ``requires-python`` under ``project``,
    plus ``[flake8]`` from ``.flake8``/``setup.cfg`` when present."""
    out: dict = {}
    pyproject = root / 'pyproject.toml'
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            data = {}
        project = data.get('project', {})
        if isinstance(project, dict) and project.get('requires-python'):
            out['project'] = {
                'requires-python': str(project['requires-python'])}
        tools = data.get('tool', {})
        if isinstance(tools, dict):
            for name, table in list(tools.items())[:12]:
                if not isinstance(table, dict):
                    continue
                keep = {}
                for key, value in table.items():
                    if _scalar(value):
                        keep[key] = value
                    elif (isinstance(value, list) and len(value) <= 8
                          and all(_scalar(v) for v in value)):
                        keep[key] = value
                out[str(name)] = keep
    for ini in ('.flake8', 'setup.cfg', 'tox.ini'):
        path = root / ini
        if not path.is_file() or 'flake8' in out:
            continue
        # No interpolation: flake8's own ``format = %(path)s:%(row)d``
        # would raise; ``items`` is inside the try because it, not
        # ``read``, is where a malformed value surfaces.
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(path, encoding='utf-8')
            if parser.has_section('flake8'):
                out['flake8'] = dict(parser.items('flake8'))
        except (OSError, configparser.Error, ValueError):
            continue
    return out


def _make_targets(root: Path) -> list:
    makefile = root / 'Makefile'
    if not makefile.is_file():
        return []
    try:
        text = makefile.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []
    seen: list = []
    for name in _MAKE_TARGET.findall(text):
        if name not in seen and not name.startswith('.'):
            seen.append(name)
    return seen


def _pytest_configured(root: Path, conventions: dict) -> bool:
    if 'pytest' in conventions:
        return True
    pyproject = root / 'pyproject.toml'
    if pyproject.is_file():
        try:
            if 'pytest' in pyproject.read_text(encoding='utf-8'):
                return True
        except OSError:
            pass
    return any((root / f).is_file()
               for f in ('pytest.ini', 'conftest.py'))


def _unittest_only(root: Path) -> bool:
    tests = root / 'tests'
    if not tests.is_dir():
        return False
    for f in sorted(tests.glob('test*.py'))[:20]:
        try:
            text = f.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        if 'import pytest' in text:
            return False
        if 'import unittest' in text or 'unittest.TestCase' in text:
            return True
    return False


def test_command(root: Path, make_targets: list, conventions: dict) -> str:
    """How to run the project's tests: ``make test`` when the Makefile has
    that target, else pytest when configured, else unittest when the tests
    use it, else pytest when a ``tests`` directory exists, else ''."""
    if 'test' in make_targets:
        return 'make test'
    if _pytest_configured(root, conventions):
        return 'python -m pytest -q'
    if _unittest_only(root):
        return 'python -m unittest discover -s tests'
    if (root / 'tests').is_dir():
        return 'python -m pytest -q'
    return ''


def _outline_module(rel: str, text: str, outlines: dict,
                    symbols: dict) -> bool:
    """Parse one module into ``outlines``/``symbols``; False when it does
    not parse (it is then counted but not outlined)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    rows = code.tree_rows(tree, OUTLINE_ROWS_PER_MODULE, defaults=False)
    if rows:
        outlines[rel] = rows
    for name, line, depth in code.definitions(tree):
        if depth > 1 or len(symbols) >= MAX_SYMBOLS and name not in symbols:
            continue
        symbols.setdefault(name, []).append(f'{rel}:{line}')
    return True


def build(root: Path, head_sha: str, budget_s: float = BUILD_BUDGET_S,
          ) -> Brief:
    """Walk ``root`` once (noise dirs skipped, symlink escapes ignored) and
    build its brief. Outlining stops once ``budget_s`` seconds have passed
    or ``MAX_MODULES`` modules are done (``truncated`` says so); counting
    always completes."""
    root = Path(root).resolve()
    started = time.monotonic()
    dirs: dict = {}
    outlines: dict = {}
    symbols: dict = {}
    modules: set = set()
    total = python = 0
    truncated = False
    for f in sorted(files.walk_files(root)):
        rel = f.relative_to(root)
        total += 1
        dirs[_dir_key(rel)] = dirs.get(_dir_key(rel), 0) + 1
        if f.suffix != '.py':
            continue
        python += 1
        if len(rel.parts) == 1:
            modules.add(rel.stem)
        elif (root / rel.parts[0] / '__init__.py').is_file():
            modules.add(rel.parts[0])
        if truncated:
            continue
        if (len(outlines) >= MAX_MODULES
                or time.monotonic() - started > budget_s):
            truncated = True
            continue
        try:
            if f.stat().st_size > files.MAX_FILE_BYTES:
                continue
            text = f.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        _outline_module(rel.as_posix(), text, outlines, symbols)
    conventions = _conventions(root)
    targets = _make_targets(root)
    return Brief(
        root=str(root), head_sha=head_sha,
        built_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        build_seconds=round(time.monotonic() - started, 3),
        files=total, python_files=python, dirs=dirs,
        modules=sorted(modules), outlines=outlines, symbols=symbols,
        test_command=test_command(root, targets, conventions),
        make_targets=targets, conventions=conventions, truncated=truncated)


# --- project rules -----------------------------------------------------------

# The project's own conventions file, the first that exists at the root:
# the instructions a human contributor reads (layering, defaults, test
# isolation, ...). Read fresh each time (not cached in the brief, which is
# keyed on HEAD) and cut at MAX_RULES_CHARS. Eval 06075f330988: a worker
# broke four of the project's conventions it had no way to know.
RULES_FILES = ('AGENTS.md', 'CLAUDE.md', '.guru/rules.md')
MAX_RULES_CHARS = 6000


def rules(root: Path) -> str:
    """The project rules text under ``root`` (``''`` when none), cut at
    ``MAX_RULES_CHARS`` with a closing note naming the file."""
    for name in RULES_FILES:
        path = Path(root) / name
        try:
            if not path.is_file():
                continue
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        text = text.strip()
        if len(text) > MAX_RULES_CHARS:
            text = (text[:MAX_RULES_CHARS]
                    + f'\n… (cut; read {name} for the rest)')
        return f'({name})\n{text}' if text else ''
    return ''


# --- render / slice ----------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """``CHARS_PER_TOKEN`` chars per token, rounded up."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _short_sha(sha: str) -> str:
    return sha[:7] if sha != NO_HEAD else sha


def render_map(brief: Brief) -> str:
    """The map block: name, HEAD, counts, directories, modules, test
    command and convention names -- what the controller sees to plan."""
    name = Path(brief.root).name
    dirs = sorted(brief.dirs.items(), key=lambda kv: (-kv[1], kv[0]))
    dir_text = ', '.join(f'{d}/ {n}' if d != '.' else f'./ {n}'
                         for d, n in dirs[:24])
    if len(dirs) > 24:
        dir_text += f', … {len(dirs) - 24} more'
    lines = [
        f"[project brief] {name} @ {_short_sha(brief.head_sha)}:"
        f" {brief.files} files ({brief.python_files} python) in"
        f" {len(brief.dirs)} dirs",
        f"dirs: {dir_text}",
        f"modules: {', '.join(brief.modules) or '(none)'}",
    ]
    if brief.test_command:
        lines.append(f"tests: {brief.test_command}")
    if brief.make_targets:
        lines.append(f"make targets: {', '.join(brief.make_targets[:16])}")
    if brief.conventions:
        lines.append("conventions: " + ', '.join(sorted(brief.conventions)))
    return "\n".join(lines)


def _mentioned_files(brief: Brief, task_text: str) -> list:
    """Outlined modules the task names: by relative path or basename
    first, then by stem as a whole word (``files`` for ``files.py``), each
    group in brief order."""
    words = set(_WORD.findall(task_text))
    paths = set(_PATHISH.findall(task_text))
    exact, by_stem = [], []
    for rel in brief.outlines:
        p = Path(rel)
        if (rel in paths or p.name in paths
                or any(path.endswith('/' + rel) for path in paths)):
            exact.append(rel)
        elif p.stem in words and p.stem != '__init__':
            by_stem.append(rel)
    return exact + by_stem


def _codelike(word: str) -> bool:
    return ('_' in word or len(word) >= _MIN_SYMBOL_LEN
            or any(c.isupper() for c in word[1:]))


def _mentioned_symbols(brief: Brief, task_text: str) -> list:
    words = {w for w in _WORD.findall(task_text) if _codelike(w)}
    return [name for name in brief.symbols if name in words]


def slice(brief: Brief, task_text: str,             # noqa: A001
          max_tokens: int = SLICE_MAX_TOKENS) -> str:
    """The brief cut to ``task_text``: the map, the test command, then the
    outline of every module the task names (path, basename or stem) and
    the location of every symbol it names, in that order, until the
    ``max_tokens`` budget (``CHARS_PER_TOKEN`` chars each) is spent; a
    final ``…`` row says when something was left out."""
    budget = max_tokens * CHARS_PER_TOKEN
    head = render_map(brief)
    lines: list = [head]
    used = len(head)
    mentioned = _mentioned_files(brief, task_text)

    def add(line: str) -> bool:
        nonlocal used
        if used + len(line) + 1 > budget:
            return False
        lines.append(line)
        used += len(line) + 1
        return True

    cut = False
    for rel in mentioned:
        rows = brief.outlines[rel]
        shown = rows[:SLICE_ROWS_PER_FILE]
        if not add(f"{rel}:"):
            cut = True
            break
        for row in shown:
            if not add(f"  {row}"):
                cut = True
                break
        if cut:
            break
        if len(rows) > len(shown):
            add(f"  … {len(rows) - len(shown)} more")
    if not cut:
        symbol_rows = []
        for name in _mentioned_symbols(brief, task_text):
            where = [loc for loc in brief.symbols[name]
                     if loc.rsplit(':', 1)[0] not in mentioned]
            if where:
                symbol_rows.append(f"  {name}: {', '.join(where[:4])}")
        if symbol_rows and add("symbols:"):
            for row in symbol_rows:
                if not add(row):
                    cut = True
                    break
    if cut:
        lines.append("… (brief cut at the token budget)")
    return "\n".join(lines)


# --- current ---------------------------------------------------------------

def current(root: Path, store: BriefStore, refresh: bool = False,
            head: Optional[str] = None) -> Brief:
    """The brief for ``root`` at its checked-out HEAD: loaded from
    ``store`` when one was built for that HEAD (and ``refresh`` is off),
    else built now and saved. ``head`` skips the ``git rev-parse`` when
    the caller already knows it. A project without a HEAD is built every
    time and never stored."""
    root = Path(root).resolve()
    head = head if head is not None else head_sha(root)
    if head != NO_HEAD and not refresh:
        found = store.load(root, head)
        if found is not None:
            return found
    built = build(root, head)
    if head != NO_HEAD:
        store.save(built)
    return built
