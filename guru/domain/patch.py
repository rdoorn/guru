"""``apply_patch``: apply a unified diff to project files (design plan B4).

Pure diff algebra (``parse`` and ``apply_hunks``) plus one file-writing verb.
The diff may touch several files; every hunk of every file is validated
against the current content before anything is written (context and removed
lines must match exactly — a hunk may sit at a different line number than
the header says, as with ``git apply``, but no context line may differ), and
the write is all-or-nothing across files (a write that fails midway rolls
the earlier files back). Refused: renames, binary patches, deletions (use
``delete_file``), a path listed twice, new files outside the project, and
anything under a noise dir (``.git``, ``.venv``, …). Each target passes
``files.ensure_path_allowed`` before it is read (a refused file is never
read, so the patch is no content oracle) and
``files.ensure_write_path_allowed`` (so read-only mode refuses)
and lands in the sha ledger like ``edit_file`` does, so a follow-up
``edit_file`` needs no re-read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from guru import config, log
from guru.domain import files

_HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
_DEV_NULL = '/dev/null'
_SKIP_PREFIXES = ('diff --git ', 'index ', 'similarity index ',
                  'dissimilarity index ', 'new file mode ',
                  'deleted file mode ', 'old mode ', 'new mode ')
_NO_NEWLINE = '\\ No newline at end of file'


class PatchError(ValueError):
    """A diff that cannot be parsed or applied; the message says why."""


@dataclass
class Hunk:
    """One ``@@`` block: the old-side start line and its body lines, each
    ``(tag, text)`` with tag ' ', '-' or '+'."""
    old_start: int
    new_start: int
    lines: list = field(default_factory=list)
    # Whether the last old-side / new-side line lacks a trailing newline.
    old_no_newline: bool = False
    new_no_newline: bool = False

    @property
    def old_lines(self) -> list:
        return [t for tag, t in self.lines if tag in (' ', '-')]

    @property
    def new_lines(self) -> list:
        return [t for tag, t in self.lines if tag in (' ', '+')]


@dataclass
class FilePatch:
    """All hunks for one target path. ``new_file`` when the old side is
    /dev/null."""
    path: str
    hunks: list = field(default_factory=list)
    new_file: bool = False


def _strip_prefix(raw: str) -> str:
    """The path of a ``---``/``+++`` line without a/ b/ prefixes or the
    trailing timestamp git-less tools add."""
    text = raw.strip()
    text = text.split('\t', 1)[0]
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text == _DEV_NULL:
        return text
    for prefix in ('a/', 'b/'):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def parse(diff: str) -> list[FilePatch]:
    """Parse a unified diff into ``FilePatch`` objects.

    Raises ``PatchError`` for renames, binary patches, deletions, a hunk
    without a ``---``/``+++`` header, a malformed hunk header, or a body
    whose line counts disagree with the header.
    """
    text = diff.replace('\r\n', '\n')
    lines = text.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    patches: list = []
    current: Optional[FilePatch] = None
    hunk: Optional[Hunk] = None
    old_left = new_left = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('rename from ') or line.startswith('rename to '):
            raise PatchError('renames are not supported; edit the file in'
                             ' place or use write_file + delete_file')
        if line.startswith('Binary files ') or line.startswith(
                'GIT binary patch'):
            raise PatchError('binary patches are not supported')
        hunk_open = hunk is not None and (old_left > 0 or new_left > 0)
        if (not hunk_open and line.startswith('--- ')
                and i + 1 < len(lines) and lines[i + 1].startswith('+++ ')):
            old = _strip_prefix(line[4:])
            new = _strip_prefix(lines[i + 1][4:])
            if new == _DEV_NULL:
                raise PatchError(f'deleting {old} via a patch is not'
                                 ' supported; use delete_file')
            if old != _DEV_NULL and old != new:
                raise PatchError(f'rename {old} -> {new} is not supported')
            if any(fp.path == new for fp in patches):
                raise PatchError(f'{new} appears twice in the diff; give'
                                 ' each file one ---/+++ section with all'
                                 ' its hunks')
            current = FilePatch(new, new_file=(old == _DEV_NULL))
            patches.append(current)
            hunk = None
            i += 2
            continue
        if line.startswith(_SKIP_PREFIXES) or (
                hunk is None and not line.strip()):
            i += 1
            continue
        m = _HUNK_RE.match(line)
        if m:
            if current is None:
                raise PatchError('hunk before any ---/+++ file header')
            old_left = int(m.group(2)) if m.group(2) is not None else 1
            new_left = int(m.group(4)) if m.group(4) is not None else 1
            hunk = Hunk(int(m.group(1)), int(m.group(3)))
            current.hunks.append(hunk)
            i += 1
            continue
        if hunk is None:
            i += 1                       # prose between files: ignored
            continue
        if line.startswith(_NO_NEWLINE):
            last = hunk.lines[-1][0] if hunk.lines else ' '
            if last in (' ', '-'):
                hunk.old_no_newline = True
            if last in (' ', '+'):
                hunk.new_no_newline = True
            i += 1
            continue
        tag, body = (line[0], line[1:]) if line else (' ', '')
        if tag not in (' ', '-', '+'):
            if old_left <= 0 and new_left <= 0:
                hunk = None              # hunk complete; trailing prose
                continue
            where = current.path if current is not None else '?'
            raise PatchError(
                f"{where}: unexpected line in hunk @@ -"
                f"{hunk.old_start}: {line[:60]!r}")
        if old_left <= 0 and new_left <= 0:
            hunk = None
            continue
        hunk.lines.append((tag, body))
        if tag in (' ', '-'):
            old_left -= 1
        if tag in (' ', '+'):
            new_left -= 1
        i += 1
    for fp in patches:
        if not fp.hunks:
            raise PatchError(f'{fp.path}: no hunks')
    if not patches:
        raise PatchError('no ---/+++ file headers found in the diff')
    return patches


def _find(old: list, needle: list, hint: int) -> int:
    """Index in ``old`` where ``needle`` matches: the header's position
    first, else the single other position that matches exactly."""
    if not needle:
        return max(0, min(hint, len(old)))
    n = len(needle)
    if 0 <= hint <= len(old) - n and old[hint:hint + n] == needle:
        return hint
    hits = [i for i in range(len(old) - n + 1) if old[i:i + n] == needle]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise PatchError('context does not match the file')
    raise PatchError(f'context matches at {len(hits)} places; add context')


def apply_hunks(text: str, hunks: list, where: str = '') -> str:
    """Return ``text`` with ``hunks`` applied, or raise ``PatchError``
    naming the first hunk whose context does not match."""
    old = text.split('\n')
    trailing = text.endswith('\n') or text == ''
    if trailing and old and old[-1] == '':
        old.pop()
    out: list = []
    cursor = 0
    offset = 0
    for k, h in enumerate(hunks, 1):
        hint = h.old_start - 1 + offset
        try:
            at = _find(old, h.old_lines, hint)
        except PatchError as e:
            raise PatchError(
                f"{where} hunk {k} (@@ -{h.old_start}): {e}") from None
        if at < cursor:
            raise PatchError(f"{where} hunk {k}: overlaps the previous hunk")
        out.extend(old[cursor:at])
        out.extend(h.new_lines)
        cursor = at + len(h.old_lines)
        offset = at - (h.old_start - 1)
        if h.old_no_newline or h.new_no_newline:
            trailing = not h.new_no_newline
    out.extend(old[cursor:])
    return "\n".join(out) + ("\n" if trailing and out else '')


def targets(diff: str) -> list[str]:
    """The target paths a diff names (best effort; for the audit row)."""
    try:
        return [fp.path for fp in parse(diff)]
    except PatchError:
        names = [_strip_prefix(ln[4:]) for ln in diff.splitlines()
                 if ln.startswith('+++ ')]
        return [n for n in names if n != _DEV_NULL]


def _inside_project(target: Path) -> bool:
    """Whether a NEW file would land inside the project (an allow-listed
    dir or the working directory)."""
    if files.project_root(target, fallback=False) is not None:
        return True
    cwd = Path.cwd().resolve()
    return target == cwd or cwd in target.parents


def apply_patch(diff: str) -> str:
    """
    Apply a unified diff (one or more files; '--- a/x' / '+++ b/x' or plain
    paths, relative to the working directory) to the project. Every hunk's
    context must match the current file exactly (line numbers may be off;
    context may not); the whole patch is validated first and applied
    all-or-nothing. New files are allowed inside the project; renames,
    deletions and binary patches are refused. Write-gated like edit_file
    (refused in read-only mode). Returns per file the hunks applied and the
    new sha (reusable with edit_file).
    """
    if config.MODE == config.MODE_READ_ONLY:
        return "Refused: read-only mode. Change mode to apply patches."
    try:
        patches = parse(diff or '')
    except PatchError as e:
        return f"Patch rejected: {e}"
    plan: list = []                  # (target, old_text, new_text, hunks)
    for fp in patches:
        target = files.resolve_path(fp.path)
        # Read gate first: a refused path is never read, so the patch cannot
        # be used as an oracle for the content of unapproved files.
        if not files.ensure_path_allowed(target):
            return f"Access to '{target}' was denied by the user."
        refusal = files.refuse_noise_write(target)
        if refusal:
            return refusal
        if target.is_dir():
            return f"Patch rejected: {target} is a directory."
        if fp.new_file:
            if target.exists():
                return (f"Patch rejected: {fp.path} already exists but the"
                        " diff creates it (--- /dev/null).")
            if not _inside_project(target):
                return (f"Patch rejected: new file {target} is outside the"
                        " project.")
            old_text = ''
        else:
            if not target.exists():
                return f"Patch rejected: no such file: {target}"
            try:
                old_text = target.read_text(encoding='utf-8')
            except (OSError, UnicodeDecodeError) as e:
                return f"Patch rejected: cannot read {target}: {e}"
        try:
            new_text = apply_hunks(old_text, fp.hunks, fp.path)
        except PatchError as e:
            return f"Patch rejected: {e}. Nothing was written."
        plan.append((target, old_text, new_text, len(fp.hunks)))
    # Gate every write before the first one so a denial leaves the tree as
    # it was.
    blocks: list = []
    for target, old_text, new_text, _ in plan:
        verb = 'Create' if not target.exists() else 'Update'
        block = files.write_detail(target, old_text, new_text, verb)
        silent = not files.will_prompt_write(target)
        if not files.ensure_write_path_allowed(target, block):
            return (f"Write access to '{target}' was denied."
                    " Nothing was written.")
        blocks.append((block, silent))
    written: list = []               # (target, existed, old_text)
    out: list = []
    for (target, old_text, new_text, n), (block, silent) in zip(plan, blocks):
        existed = target.exists()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_text, encoding='utf-8')
        except OSError as e:
            _rollback(written)
            return (f"FAILED: nothing applied (cannot write {target}: {e};"
                    f" {len(written)} earlier file(s) restored).")
        written.append((target, existed, old_text))
        if silent:
            files.show_change(block)
        sha = files.sha_of(new_text)
        files.remember_sha(target, sha)
        out.append(f"{target}: {n} hunk(s) applied (sha:{sha})")
    return "Applied patch:\n" + "\n".join(out)


def _rollback(written: list) -> None:
    """Restore files an aborted apply_patch already wrote: previous content
    for files that existed, removal for files it created. Best effort;
    failures are logged, never raised."""
    for target, existed, old_text in reversed(written):
        try:
            if existed:
                target.write_text(old_text, encoding='utf-8')
            else:
                target.unlink()
        except OSError:
            log.exc(f'apply_patch rollback failed for {target}')
        files.forget_sha(target)
