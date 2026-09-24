"""Verification verbs: ``run_tests``, ``check_syntax`` and ``lint`` (design
plan B2).

Each one is a fixed procedure: guru builds the argv list itself (the model
picks a target and a few switches, never a command), the child runs under
``procs.run`` (scrubbed environment, rlimits, timeout, output cap) and the
model sees a *digest* — a summary line and the first few failures — with a
``detail`` argument to expand one item. The full child output is logged to
guru's log file, not returned.

Reads only: tests and linters read the repository (``pytest`` runs without
its cache plugin and with ``PYTHONDONTWRITEBYTECODE``), so these verbs are
gated by the read allow-list, not the write gates. Stdlib only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from guru import log
from guru.domain import files, procs

_MAX_FAILURES = 10          # failing ids listed in the run_tests digest
_MAX_ISSUES = 10            # lint issues listed per linter in the digest
_DETAIL_BYTES = 4096        # cap on a detail block
_LOG_HEAD = 20_000          # chars of raw child output kept in the log

_FLAKE8_CONFIGS = ('setup.cfg', 'tox.ini', '.flake8')

_SUMMARY_RE = re.compile(
    r'^=*\s*((?:\d+ [a-z]+(?:, )?)+ in [\d.]+s(?: \(.*\))?|no tests ran'
    r'(?: in [\d.]+s)?)\s*=*$')
_FAILED_RE = re.compile(r'^(FAILED|ERROR) (\S+)(?: - (.*))?$')
_BLOCK_RE = re.compile(r'^_{3,} (.+?) _{3,}$')
_PROGRESS_RE = re.compile(r'^[.FEsxX]+(?:\s+\[\s*\d+%\])?$')
_UNITTEST_SUMMARY_RE = re.compile(r'^(OK|FAILED)(?: \(.*\))?$')
_UNITTEST_RAN_RE = re.compile(r'^Ran (\d+) tests? in')
_UNITTEST_BLOCK_RE = re.compile(r'^(FAIL|ERROR): (\S+) \((\S+)\)')
_FLAKE8_ROW = re.compile(r'^(.+?):(\d+):(\d+): ([A-Z]+\d+) (.*)$')
_MYPY_ROW = re.compile(r'^(.+?):(\d+)(?::\d+)?: (error|warning|note): (.*)$')

_available: dict = {}


def _policy():
    """The installed tool policy (lazy import: ``tools`` registers these
    verbs, so a top-level import would be circular)."""
    from guru.domain import tools
    return tools.active_policy()


def _limits() -> procs.Limits:
    return procs.Limits.from_dict(_policy().limits)


def _log_output(verb: str, res: procs.ProcResult) -> None:
    """Keep the child's full output (head) in guru's log, not the model."""
    log.info('%s: %s -> rc=%s %.1fs%s%s\n--- stdout ---\n%s\n--- stderr ---'
             '\n%s', verb, ' '.join(res.argv), res.returncode, res.seconds,
             ' timed out' if res.timed_out else '',
             ' truncated' if res.truncated else '',
             res.stdout[:_LOG_HEAD], res.stderr[:_LOG_HEAD])


def _rel(target: Path, root: Path) -> str:
    try:
        return str(target.relative_to(root)) or '.'
    except ValueError:
        return str(target)


def _resolve_target(path: str) -> tuple:
    """``(target, root, error)``: the resolved (gated) target path and the
    project root it belongs to; ``target`` is None for an empty path."""
    if not (path or '').strip():
        root = files.project_root(Path.cwd())
        if not files.ensure_path_allowed(root):
            return None, None, f"Access to '{root}' was denied by the user."
        return None, root, ''
    target = files._resolve(path)
    if not files.ensure_path_allowed(target):
        return None, None, f"Access to '{target}' was denied by the user."
    if not target.exists():
        return None, None, f"No such path: {target}"
    return target, files.project_root(target), ''


def _clip(text: str, limit: Optional[int] = None) -> str:
    limit = limit or _DETAIL_BYTES
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n… ({len(text) - limit} more chars)"


# --- run_tests ---------------------------------------------------------------

def _pytest_blocks(stdout: str) -> dict:
    """``{title: block}`` from the FAILURES/ERRORS sections."""
    blocks: dict = {}
    title: Optional[str] = None
    current: list = []
    in_failures = False
    for line in stdout.splitlines():
        if re.match(r'^=+ (FAILURES|ERRORS) =+$', line):
            in_failures = True
            continue
        if in_failures and re.match(r'^=+ .* =+$', line):
            if title is not None:
                blocks[title] = "\n".join(current).strip()
            title, current, in_failures = None, [], False
            continue
        if not in_failures:
            continue
        m = _BLOCK_RE.match(line)
        if m:
            if title is not None:
                blocks[title] = "\n".join(current).strip()
            title, current = m.group(1).strip(), []
            continue
        current.append(line)
    if title is not None:
        blocks[title] = "\n".join(current).strip()
    return blocks


def _block_for(test_id: str, blocks: dict) -> str:
    """The failure block whose title matches ``test_id`` (pytest titles are
    'test_x', 'TestC.test_x' or 'ERROR at setup of test_x')."""
    tail = test_id.split('::', 1)[1] if '::' in test_id else test_id
    tail = re.sub(r'\[.*\]$', '', tail).replace('::', '.')
    for title, block in blocks.items():
        clean = re.sub(r'\[.*\]$', '', title)
        clean = re.sub(r'^ERROR at (setup|teardown) of ', '', clean)
        if clean == tail or clean.endswith('.' + tail.rsplit('.', 1)[-1]):
            return block
    return ''


def _assertion_line(block: str, fallback: str) -> str:
    for line in block.splitlines():
        if line.startswith('E '):
            return line[1:].strip()
    return fallback


def _pytest_digest(res: procs.ProcResult, detail: str) -> str:
    stdout = res.stdout
    summary = ''
    for line in reversed(stdout.splitlines()):
        m = _SUMMARY_RE.match(line.strip())
        if m:
            summary = m.group(1)
            break
    failed: list = []
    for line in stdout.splitlines():
        m = _FAILED_RE.match(line.strip())
        if m:
            failed.append((m.group(2), m.group(3) or ''))
    blocks = _pytest_blocks(stdout)
    if detail:
        block = _block_for(detail, blocks)
        if not block:
            ids = ', '.join(i for i, _ in failed) or 'none'
            return f"No failure block for '{detail}'. Failing tests: {ids}."
        return _clip(f"{detail}:\n{block}")
    if not summary:
        err = (res.stderr.strip() or stdout.strip()).splitlines()
        head = "\n".join(err[-8:]) if err else '(no output)'
        return (f"pytest exited {res.returncode} without a summary line:\n"
                f"{head}")
    out = [summary]
    for test_id, reason in failed[:_MAX_FAILURES]:
        why = _assertion_line(_block_for(test_id, blocks), reason)
        out.append(f"  {test_id}" + (f" — {why[:200]}" if why else ''))
    if len(failed) > _MAX_FAILURES:
        out.append(f"  … {len(failed) - _MAX_FAILURES} more failures")
    if failed:
        out.append("(detail=<test id> shows one failure in full)")
    return "\n".join(out)


def _unittest_digest(res: procs.ProcResult, detail: str) -> str:
    text = res.stderr + "\n" + res.stdout
    lines = text.splitlines()
    summary = ''
    ran = ''
    for line in reversed(lines):
        s = line.strip()
        if not summary and _UNITTEST_SUMMARY_RE.match(s):
            summary = s
        elif not ran and _UNITTEST_RAN_RE.match(s):
            ran = s
        if summary and ran:
            break
    blocks: dict = {}
    title: Optional[str] = None
    current: list = []
    for line in lines:
        m = _UNITTEST_BLOCK_RE.match(line)
        if m:
            if title:
                blocks[title] = "\n".join(current).strip()
            qualname = m.group(3)          # 3.11+: the full dotted id
            if not qualname.endswith('.' + m.group(2)):
                qualname = f"{qualname}.{m.group(2)}"
            title, current = qualname, []
            continue
        if title is not None:
            if line.startswith('Ran ') or line.startswith('-' * 20):
                if line.startswith('Ran '):
                    blocks[title] = "\n".join(current).strip()
                    title = None
                continue
            current.append(line)
    if title:
        blocks[title] = "\n".join(current).strip()
    if detail:
        for key, block in blocks.items():
            if key == detail or key.endswith('.' + detail):
                return _clip(f"{key}:\n{block}")
        return (f"No failure block for '{detail}'. Failing tests: "
                + (', '.join(blocks) or 'none') + '.')
    if not summary:
        head = "\n".join(lines[-8:]) if lines else '(no output)'
        return (f"unittest exited {res.returncode} without a summary:\n"
                f"{head}")
    out = [f"{ran} — {summary}" if ran else summary]
    for key, block in list(blocks.items())[:_MAX_FAILURES]:
        why = ''
        for ln in reversed(block.splitlines()):
            if ln.strip() and not ln.startswith(' '):
                why = ln.strip()
                break
        out.append(f"  {key}" + (f" — {why[:200]}" if why else ''))
    if len(blocks) > _MAX_FAILURES:
        out.append(f"  … {len(blocks) - _MAX_FAILURES} more failures")
    return "\n".join(out)


def _tests_ran(stdout: str) -> int:
    n = 0
    for line in stdout.splitlines():
        s = line.strip()
        if _PROGRESS_RE.match(s):
            n += len(s.split()[0])
    return n


def run_tests(target: str = '', k: str = '', maxfail: int = 1,
              detail: str = '') -> str:
    """
    Run the project's tests (pytest, or unittest per the project policy) and
    return a short digest: the summary line and the failing test ids with
    their assertion line. ``target`` is a test file, directory or
    'file::test' node id (default: the whole project); ``k`` a pytest -k
    expression; ``maxfail`` stops after that many failures (default 1).
    Pass ``detail`` = a failing test id to get that failure's full block.
    Use this to verify an edit before reporting it done.
    """
    try:
        stop = max(1, int(maxfail))
    except (TypeError, ValueError):
        stop = 1
    node = ''
    path = (target or '').strip()
    if '::' in path:
        path, node = path.split('::', 1)
    resolved, root, err = _resolve_target(path)
    if err:
        return err
    runner = _policy().test_runner
    argv = [sys.executable, '-m', runner]
    if runner == 'pytest':
        argv += ['-q', '-p', 'no:cacheprovider', f'--maxfail={stop}', '-rfE']
        if resolved is not None:
            argv.append(_rel(resolved, root) + (f'::{node}' if node else ''))
        if k:
            argv += ['-k', str(k)]
    else:
        argv.append('-q')
        if resolved is not None:
            argv.append(_rel(resolved, root))
        if k:
            argv += ['-k', str(k)]
    res = procs.run(argv, root, _limits())
    if res.denied:
        return f"Refused: {res.denied}"
    _log_output('run_tests', res)
    if res.returncode == -1 and not res.timed_out:
        return f"Cannot run {runner}: {res.stderr.strip()[:300]}"
    if res.timed_out:
        return (f"timed out after {_limits().timeout_s}s;"
                f" {_tests_ran(res.stdout)} tests ran")
    if 'No module named' in res.stderr and not res.stdout.strip():
        return f"Cannot run {runner}: {res.stderr.strip()[:300]}"
    if runner == 'pytest':
        return _pytest_digest(res, (detail or '').strip())
    return _unittest_digest(res, (detail or '').strip())


# --- check_syntax ------------------------------------------------------------

def check_syntax(path: str) -> str:
    """
    Compile one Python file in-process (no bytecode written) and report 'ok'
    or the SyntaxError with its line, column and offending text. Cheap:
    call it after every edit to a .py file, before running tests.
    """
    target = files._resolve(path)
    if not files.ensure_path_allowed(target):
        return f"Access to '{target}' was denied by the user."
    if not target.exists():
        return f"No such file: {target}"
    if target.is_dir():
        return f"{target} is a directory."
    if target.suffix not in ('.py', '.pyi'):
        return f"{target} is not a Python file (.py); nothing checked."
    try:
        source = target.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as e:
        return f"Cannot read {target}: {e}"
    try:
        compile(source, str(target), 'exec', dont_inherit=True)
    except SyntaxError as e:
        text = (e.text or '').rstrip()
        where = f"{target}:{e.lineno}" + (f":{e.offset}" if e.offset else '')
        return (f"SyntaxError at {where}: {e.msg}"
                + (f"\n  {text}" if text else ''))
    except ValueError as e:                 # null bytes in source
        return f"Cannot compile {target}: {e}"
    return f"ok: {target} compiles."


# --- lint --------------------------------------------------------------------

def _has_section(path: Path, section: str) -> bool:
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return False
    return re.search(rf'^\[{re.escape(section)}\]\s*$', text,
                     re.MULTILINE) is not None


def _flake8_configured(root: Path) -> bool:
    return any(_has_section(root / name, 'flake8') for name in _FLAKE8_CONFIGS)


def _mypy_configured(root: Path) -> bool:
    if _has_section(root / 'pyproject.toml', 'tool.mypy'):
        return True
    return any(_has_section(root / name, 'mypy')
               for name in ('mypy.ini', 'setup.cfg'))


def _installed(module: str, root: Path) -> bool:
    """Whether ``python -m <module> --version`` works (checked once)."""
    key = (sys.executable, module)
    if key not in _available:
        res = procs.run([sys.executable, '-m', module, '--version'], root,
                        procs.Limits(timeout_s=60))
        _available[key] = res.returncode == 0
    return bool(_available[key])


def _reset_cache() -> None:
    """Forget the linter availability checks (tests)."""
    _available.clear()


def _issues(res: procs.ProcResult, row_re: re.Pattern) -> list:
    rows: list = []
    for line in res.stdout.splitlines():
        if row_re.match(line.strip()):
            rows.append(line.strip())
    return rows


def _linter_digest(name: str, res: procs.ProcResult, row_re: re.Pattern,
                   ) -> str:
    if res.timed_out:
        return f"{name}: timed out after {_limits().timeout_s}s"
    if res.returncode == -1:
        return f"{name}: cannot run: {res.stderr.strip()[:200]}"
    rows = _issues(res, row_re)
    if not rows:
        if res.returncode == 0:
            return f"{name}: clean"
        tail = (res.stderr.strip() or res.stdout.strip()).splitlines()[-3:]
        return f"{name}: exited {res.returncode}: " + ' | '.join(tail)
    out = [f"{name}: {len(rows)} issue(s)"]
    out.extend(f"  {r}" for r in rows[:_MAX_ISSUES])
    if len(rows) > _MAX_ISSUES:
        out.append(f"  … {len(rows) - _MAX_ISSUES} more"
                   f" (detail='{name}' expands)")
    return "\n".join(out)


def lint(path: str = '', detail: str = '') -> str:
    """
    Run the project's configured linters on a file or directory and return
    counts plus the first 10 issues per linter. flake8 runs when it is
    installed (its config, if any, is honoured); mypy runs only when the
    project configures it ([tool.mypy] in pyproject.toml or mypy.ini) and it
    is installed. ``detail`` = 'flake8' or 'mypy' returns that linter's
    output (up to 4 KB). Read-only.
    """
    resolved, root, err = _resolve_target(path)
    if err:
        return err
    detail = (detail or '').strip().lower()
    if detail and detail not in ('flake8', 'mypy'):
        return "detail must be 'flake8', 'mypy' or empty."
    rel = _rel(resolved, root) if resolved is not None else '.'
    limits = _limits()
    parts: list = []
    if _installed('flake8', root):
        argv = [sys.executable, '-m', 'flake8',
                '--extend-exclude=' + ','.join(sorted(files._NOISE_DIRS)),
                rel]
        res = procs.run(argv, root, limits)
        _log_output('lint/flake8', res)
        if detail == 'flake8':
            return _clip(f"flake8 ({rel}):\n"
                         + (res.stdout.strip() or res.stderr.strip()
                            or 'clean'))
        parts.append(_linter_digest('flake8', res, _FLAKE8_ROW))
    else:
        parts.append('flake8: not installed; skipped')
    if not _mypy_configured(root):
        parts.append('mypy: not configured for this project; skipped')
    elif not _installed('mypy', root):
        parts.append('mypy: configured but not installed; skipped')
    else:
        argv = [sys.executable, '-m', 'mypy']
        if resolved is not None:
            argv.append(rel)
        res = procs.run(argv, root, limits)
        _log_output('lint/mypy', res)
        if detail == 'mypy':
            return _clip(f"mypy ({rel}):\n"
                         + (res.stdout.strip() or res.stderr.strip()
                            or 'clean'))
        parts.append(_linter_digest('mypy', res, _MYPY_ROW))
    if detail:
        return f"{detail}: not run; " + '; '.join(parts)
    return "\n".join(parts)
