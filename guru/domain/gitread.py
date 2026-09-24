"""Read-only git verbs: ``git_status`` and ``git_diff`` (design plan B3).

Fixed argv only (``git -C <root> status --porcelain=v1 -uall``,
``git -C <root> diff --stat -- <path>``, ``git -C <root> diff -- <path>``,
each with ``-c core.fsmonitor=false -c core.hooksPath=/dev/null`` and the
diffs with ``--no-ext-diff --no-textconv`` so a hostile ``.git/config``
cannot make a read execute anything) through ``procs.run``; never ``add``,
``commit``, ``checkout`` or anything else that changes the tree or the
index. The root is the git toplevel of the
allow-listed project directory, and it must itself be under the read
allow-list (a project nested inside a larger repo does not expose the parent
repo's state). Digests are capped. Stdlib only.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from guru import log
from guru.domain import files, procs, toolpolicy

_MAX_STATUS_ROWS = 40       # porcelain rows shown by git_status
_MAX_STAT_ROWS = 40         # --stat rows shown by git_diff
_DETAIL_BYTES = 8 * 1024    # cap on a unified diff (git_diff detail=True)
_TRUTHY = ('1', 'true', 'yes', 'y', 'on')

_STATUS_WORDS = {
    'M': 'modified', 'A': 'added', 'D': 'deleted', 'R': 'renamed',
    'C': 'copied', 'U': 'unmerged', '?': 'untracked', '!': 'ignored',
}


# Hardening against a hostile checkout: no fsmonitor daemon or hook can be
# started by a read (``core.fsmonitor``, ``core.hooksPath``), and a diff
# never runs an external diff driver or a textconv filter configured in
# ``.git/config`` (``--no-ext-diff --no-textconv``).
_SAFE_CONFIG = ['-c', 'core.fsmonitor=false', '-c',
                'core.hooksPath=/dev/null']
_SAFE_DIFF = ['--no-ext-diff', '--no-textconv']


def _limits() -> procs.Limits:
    return procs.Limits.from_dict(toolpolicy.active_policy().limits)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUTHY


def _git(args: list, root: Path) -> procs.ProcResult:
    res = procs.run(['git', '-C', str(root)] + _SAFE_CONFIG + args, root,
                    _limits())
    log.info('gitread: %s -> rc=%s%s', ' '.join(res.argv), res.returncode,
             f' denied={res.denied}' if res.denied else '')
    return res


def _toplevel(project: Path) -> tuple[Optional[Path], str]:
    """``(toplevel, error)``: the repository root of ``project``."""
    res = _git(['rev-parse', '--show-toplevel'], project)
    if res.denied:
        return None, f"Refused: {res.denied}"
    if res.returncode != 0:
        why = res.stderr.strip().splitlines()
        return None, (f"{project} is not inside a git repository"
                      + (f" ({why[0][:120]})" if why else '') + '.')
    top = Path(res.stdout.strip()).resolve()
    if not files.ensure_path_allowed(top):
        return None, (f"Access to the repository root '{top}' was denied by"
                      " the user.")
    return top, ''


def _root_for(path: str) -> tuple[Optional[Path], str, str]:
    """``(root, rel, error)``: the repo root for ``path`` (default: the
    project of the working directory) and ``path`` relative to it."""
    if (path or '').strip():
        target = files.resolve_path(path)
        if not files.ensure_path_allowed(target):
            return None, '', f"Access to '{target}' was denied by the user."
        project = files.project_root(target)
    else:
        target = None
        project = files.project_root(Path.cwd())
        if not files.ensure_path_allowed(project):
            return None, '', f"Access to '{project}' was denied by the user."
    top, err = _toplevel(project)
    if err or top is None:
        return None, '', err
    rel = ''
    if target is not None:
        try:
            rel = str(target.relative_to(top))
        except ValueError:
            return None, '', f"{target} is not inside the repository {top}."
    return top, rel, ''


def _status_digest(porcelain: str, root: Path) -> str:
    rows = [ln for ln in porcelain.splitlines() if ln.strip()]
    if not rows:
        return f"{root}: working tree clean."
    counts: dict = {}
    for row in rows:
        code = row[:2]
        word = _STATUS_WORDS.get(code.strip()[:1], 'changed')
        if code == '??':
            word = 'untracked'
        counts[word] = counts.get(word, 0) + 1
    summary = ', '.join(f"{n} {w}" for w, n in sorted(counts.items()))
    out = [f"{root}: {len(rows)} path(s) — {summary}"]
    out.extend(rows[:_MAX_STATUS_ROWS])
    if len(rows) > _MAX_STATUS_ROWS:
        out.append(f"… {len(rows) - _MAX_STATUS_ROWS} more")
    return "\n".join(out)


def git_status() -> str:
    """
    Show the working tree status of the project's git repository (porcelain
    rows: 'XY path', untracked files listed one by one) with a one-line
    count. Read-only; never stages or commits.
    """
    root, _, err = _root_for('')
    if err or root is None:
        return err
    res = _git(['status', '--porcelain=v1', '-uall'], root)
    if res.denied:
        return f"Refused: {res.denied}"
    if res.timed_out:
        return f"git status timed out after {_limits().timeout_s}s."
    if res.returncode != 0:
        return f"git status failed: {res.stderr.strip()[:300]}"
    return _status_digest(res.stdout, root)


_STAT_TOTAL = re.compile(r'^\s*\d+ files? changed')


def _stat_digest(stat: str, root: Path, rel: str) -> str:
    rows = [ln.rstrip() for ln in stat.splitlines() if ln.strip()]
    where = f"{root}" + (f" ({rel})" if rel else '')
    if not rows:
        return f"{where}: no unstaged changes."
    total = next((r.strip() for r in rows if _STAT_TOTAL.match(r)), '')
    body = [r.strip() for r in rows if not _STAT_TOTAL.match(r)]
    out = [f"{where}: {total or f'{len(body)} file(s) changed'}"]
    out.extend(body[:_MAX_STAT_ROWS])
    if len(body) > _MAX_STAT_ROWS:
        out.append(f"… {len(body) - _MAX_STAT_ROWS} more files")
    out.append("(detail=true returns the unified diff, up to 8 KB)")
    return "\n".join(out)


def git_diff(path: str = '', detail: object = False) -> str:
    """
    Show unstaged changes in the project's git repository: by default a
    per-file +/- stat, optionally limited to ``path``; ``detail=true``
    returns the unified diff itself (capped at 8 KB). Read-only.
    """
    root, rel, err = _root_for(path)
    if err or root is None:
        return err
    if _truthy(detail):
        res = _git(['diff'] + _SAFE_DIFF + ['--'] + ([rel] if rel else []),
                   root)
    else:
        res = _git(['diff', '--stat'] + _SAFE_DIFF + ['--']
                   + ([rel] if rel else []), root)
    if res.denied:
        return f"Refused: {res.denied}"
    if res.timed_out:
        return f"git diff timed out after {_limits().timeout_s}s."
    if res.returncode != 0:
        return f"git diff failed: {res.stderr.strip()[:300]}"
    if not _truthy(detail):
        return _stat_digest(res.stdout, root, rel)
    text = res.stdout
    if not text.strip():
        where = f"{root}" + (f" ({rel})" if rel else '')
        return f"{where}: no unstaged changes."
    if len(text) > _DETAIL_BYTES:
        text = (text[:_DETAIL_BYTES].rstrip()
                + f"\n… diff truncated at {_DETAIL_BYTES // 1024} KB;"
                " pass a path to narrow it.")
    return text


def repo_root(path: str = '') -> Optional[Path]:
    """The repository toplevel for ``path`` (or the working directory's
    project), None when there is none or access is refused."""
    root, _, err = _root_for(path)
    return None if err else root
