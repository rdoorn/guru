"""The answer checker endpoint (``guru.domain.claims.AnswerChecker``): the
evidence is the working tree's changes (``git diff HEAD`` plus the start
of every new file), the reviewer the routing's ``standard`` review rung
(``llm.default_reviewer``: the secret scan, local-only mode and the spend
confirmation decide as for the sandbox gate), one JSON-only completion.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from guru import log, session
from guru.domain import claims, policy
from guru.judges import llm

MAX_TOKENS = 500
NEW_FILE_CHARS = 4000
_GIT_TIMEOUT_S = 20


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(['git', *args], cwd=root, capture_output=True,
                          text=True, timeout=_GIT_TIMEOUT_S, check=False)
    return proc.stdout if proc.returncode == 0 else ''


def _head_of(path: Path) -> Optional[str]:
    """The first ``NEW_FILE_CHARS`` of a text file (read bounded); None
    for a binary file or one that cannot be read."""
    try:
        with path.open('rb') as fh:
            data = fh.read(NEW_FILE_CHARS)
    except OSError:
        return None
    if b'\0' in data:
        return None
    return data.decode('utf-8', errors='replace')


def evidence(root: Optional[Path] = None) -> str:
    """What changed under ``root`` (default: the working directory): the
    start of each new untracked file first (the work a task adds is most
    often new files), then the diff of tracked files against HEAD — each
    with half the evidence budget, so a large diff cannot push the new
    files out. '' outside a git repository or when nothing changed."""
    root = Path(root or Path.cwd())
    try:
        if not _git(root, 'rev-parse', 'HEAD').strip():
            return ''                 # not a repository, or no commit yet
        diff = _git(root, 'diff', 'HEAD')
        new = _git(root, 'ls-files', '--others', '--exclude-standard')
    except (OSError, subprocess.SubprocessError):
        log.exc('answer check: git unavailable')
        return ''
    half = claims.MAX_EVIDENCE_CHARS // 2
    files: list = []
    used = 0
    for rel in new.splitlines():
        if used >= half:
            files.append('… (more new files cut)')
            break
        text = _head_of(root / rel)
        if text is None:
            continue
        block = f'--- new file: {rel}\n{text}'
        files.append(block)
        used += len(block)
    if len(diff) > half:
        diff = diff[:half] + '\n… (diff cut)'
    return '\n'.join(files + ([diff] if diff.strip() else []))


def _remote(adapter: object) -> bool:
    return bool(getattr(adapter, 'remote', True))


class ClaimsReviewer:
    """Checks a final answer with one completion on the review rung."""

    def check(self, request: str, answer: str) -> list[str]:
        ev = evidence()
        reviewer = llm.default_reviewer(ev, session.adapter, session.model)
        if reviewer is None:
            return []
        if policy.scan(ev) and _remote(reviewer.adapter):
            # default_reviewer falls back to the session model when no
            # local rung exists; a flagged diff never goes remote here.
            log.info('answer check skipped: secret findings in the'
                     ' changes and no local reviewer')
            return []
        text = reviewer.adapter.complete(       # type: ignore[attr-defined]
            claims.prompt(request, answer, ev), max_tokens=MAX_TOKENS,
            model=reviewer.model)
        problems = claims.parse(str(text))
        if problems is None:
            log.info('answer check: unparsable reply %r', str(text)[:200])
            return []
        return problems
