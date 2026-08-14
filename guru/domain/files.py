"""Filesystem tools: list, read, grep, and write/edit, gated by per-project
directory allow-lists with SEPARATE read and write lists.

Nothing is allowed by default. Reads are approved against ``read_dirs_allow``,
writes against ``write_dirs_allow`` (under ``.guru/``) — approving a directory
for reading never grants writing. The access mode decides how a not-yet-allowed
directory is handled: read-only refuses writes, ask prompts (writes show the
exact operation), auto approves silently. Paths are resolved to real absolute
paths before the check, so ``..``/symlink escapes cannot leave an allowed tree.
"""
import difflib
import hashlib
import re
import shutil
import stat
from pathlib import Path

from guru import config, ui

# Directories listed but not descended into by list_tree (pass a noise dir's
# full path to look inside it).
_NOISE_DIRS = {
    '.git', '.venv', 'venv', 'node_modules', '__pycache__',
    '.mypy_cache', '.pytest_cache', '.tox', '.idea', '.ruff_cache',
}
_MAX_ENTRIES = 400          # cap on entries emitted by list_tree
_DEFAULT_TREE_DEPTH = 3     # levels list_tree recurses when depth is unset
_MAX_READ_LINES = 400       # cap when read_file is given no range
_MAX_RANGE_SPAN = 2000      # cap on an explicit read_file range
_MAX_MATCHES = 100          # cap on rows returned by search_code
_MAX_FILE_BYTES = 1_000_000  # skip files larger than this in search_code


# --- directory allow-list gate ----------------------------------------------
#
# Reads and writes have SEPARATE allow-lists (config.ALLOWED_READ_DIRS /
# ALLOWED_WRITE_DIRS): approving a directory for reading never grants writing.
# The access MODE decides how a not-yet-allowed directory is handled —
# read-only refuses writes, ask prompts, auto approves silently. Every mode
# resolves paths first, so ``..``/symlink escapes are always blocked.

_path_asker = None


def set_path_asker(fn) -> None:
    """Install a custom approval prompt (used by the TUI).

    Signature: ``(question: str) -> bool``.
    """
    global _path_asker
    _path_asker = fn


def _ask_path(question: str) -> bool:
    """Default terminal approval prompt. Enter approves; errors deny."""
    try:
        answer = input(f"{question}\n[Y/n] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return not answer.startswith('n')


def _within_allowed(resolved: Path, allowed=None) -> bool:
    allowed = config.ALLOWED_READ_DIRS if allowed is None else allowed
    for d in allowed:
        base = Path(d)
        if resolved == base or base in resolved.parents:
            return True
    return False


def _approve(ask_dir: Path, allowed: set, persist, question: str) -> bool:
    """Grant access to ``ask_dir`` per the current mode; on grant add it to
    ``allowed`` and persist. auto approves silently, ask prompts."""
    key = str(ask_dir)
    if config.MODE == config.MODE_AUTO:
        allowed.add(key)
        persist(key)
        return True
    asker = _path_asker or _ask_path
    if asker(question):
        allowed.add(key)
        persist(key)
        ui.console.print(f"[green]Allowed[/green] {ask_dir}.")
        return True
    ui.console.print(f"[red]Denied[/red] {ask_dir}.")
    return False


def ensure_path_allowed(path: Path) -> bool:
    """Gate a READ of ``path`` against the read allow-list (mode-aware)."""
    if _within_allowed(path):
        return True
    ask_dir = (path if path.is_dir() else path.parent).resolve()
    question = f"Allow READ access to '{ask_dir}'?"
    return _approve(ask_dir, config.ALLOWED_READ_DIRS,
                    config.persist_read_dir, question)


def ensure_write_path_allowed(path: Path, detail: str) -> bool:
    """Gate a WRITE of ``path`` against the write allow-list. Refuses in
    read-only mode; the prompt states the exact write (``detail``)."""
    if config.MODE == config.MODE_READ_ONLY:
        return False
    if _within_allowed(path, config.ALLOWED_WRITE_DIRS):
        return True
    ask_dir = (path if path.is_dir() else path.parent).resolve()
    question = f"{detail}\nAllow WRITE access to '{ask_dir}'?"
    return _approve(ask_dir, config.ALLOWED_WRITE_DIRS,
                    config.persist_write_dir, question)


# --- formatting helpers ------------------------------------------------------

def _octal(mode: int) -> str:
    return format(stat.S_IMODE(mode), 'o')


def _human(size: int) -> str:
    value = float(size)
    for unit in ('B', 'K', 'M', 'G', 'T'):
        if value < 1024 or unit == 'T':
            if unit == 'B':
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def _row(st, name: str) -> str:
    """Compact 'perm size name' row (no padding); size is '-' for dirs."""
    is_dir = stat.S_ISDIR(st.st_mode)
    size = '-' if is_dir else _human(st.st_size)
    return f"{_octal(st.st_mode)} {size} {name}"


def _sha(text: str) -> str:
    """Short content hash used for optimistic-concurrency on writes."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]


def _resolve(path: str) -> Path:
    return Path(path or '.').expanduser().resolve()


def _sorted_children(d: Path) -> list:
    return sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))


# --- tools -------------------------------------------------------------------

def list_dir(path: str = '.') -> str:
    """
    List the immediate contents of a directory (non-recursive) as compact
    'perm size name' rows (octal perms, dirs end with '/', size '-' for dirs).
    Use list_tree for a recursive view, or read_file to read a file.
    """
    target = _resolve(path)
    if not ensure_path_allowed(target):
        return f"Access to '{target}' was denied by the user."
    if not target.exists():
        return f"No such path: {target}"
    if not target.is_dir():
        return f"Not a directory: {target} (use read_file to read it)."
    try:
        children = _sorted_children(target)
    except OSError as e:
        return f"Cannot list {target}: {e}"
    lines = [f"{target}:"]
    for p in children:
        try:
            st = p.lstat()
        except OSError:
            continue
        name = p.name + ('/' if stat.S_ISDIR(st.st_mode) else '')
        lines.append(_row(st, name))
    if len(lines) == 1:
        lines.append("(empty)")
    return "\n".join(lines)


def list_tree(path: str = '.', depth: str = '') -> str:
    """
    List a directory tree recursively as compact 'perm size relpath' rows.
    Paths are relative to the root and flat (not indented), so the structure
    is unambiguous and cheap. Noise directories (.git, node_modules,
    __pycache__, .venv, …) are shown with '*skip' but not expanded — to look
    inside one, call this with its full path. ``depth`` limits how many levels
    to recurse (default 3).
    """
    target = _resolve(path)
    if not ensure_path_allowed(target):
        return f"Access to '{target}' was denied by the user."
    if not target.exists():
        return f"No such path: {target}"
    if not target.is_dir():
        return f"Not a directory: {target} (use read_file to read it)."
    try:
        max_depth = int(depth)
    except (TypeError, ValueError):
        max_depth = _DEFAULT_TREE_DEPTH

    lines = [f"{target} (tree):"]
    state = {'count': 0, 'truncated': False}

    def walk(d: Path, prefix: str, level: int) -> None:
        try:
            children = _sorted_children(d)
        except OSError:
            return
        for p in children:
            if state['count'] >= _MAX_ENTRIES:
                state['truncated'] = True
                return
            try:
                st = p.lstat()
            except OSError:
                continue
            is_dir = stat.S_ISDIR(st.st_mode)
            rel = prefix + p.name + ('/' if is_dir else '')
            skip = is_dir and p.name in _NOISE_DIRS
            lines.append(_row(st, rel) + (' *skip' if skip else ''))
            state['count'] += 1
            if is_dir and not skip and level + 1 < max_depth:
                walk(p, rel, level + 1)

    walk(target, '', 0)
    if state['truncated']:
        lines.append(
            f"... truncated at {_MAX_ENTRIES} entries — narrow with a subpath"
            f" or a smaller depth.")
    if len(lines) == 1:
        lines.append("(empty)")
    return "\n".join(lines)


def _parse_range(spec: str, total: int) -> tuple:
    """Return (start, end) 1-based inclusive, capped; (None, None) if bad."""
    spec = (spec or '').strip()
    if not spec:
        return (1, min(total, _MAX_READ_LINES))
    a, _, b = spec.partition('-') if '-' in spec else (spec, '', spec)
    try:
        start, end = int(a), int(b)
    except ValueError:
        return (None, None)
    start = max(1, start)
    if end < start:
        return (None, None)
    end = min(end, total, start + _MAX_RANGE_SPAN - 1)
    return (start, end)


def read_file(path: str, lines: str = '') -> str:
    """
    Read the text content of a file. For large files, pass ``lines`` as a
    1-based inclusive range like '10-20' to read just that span. Output is
    line-numbered and includes the file's sha — pass that sha to edit_file so
    the edit is confirmed to apply to the file as it actually is.
    """
    target = _resolve(path)
    if not ensure_path_allowed(target):
        return f"Access to '{target}' was denied by the user."
    if not target.exists():
        return f"No such file: {target}"
    if target.is_dir():
        return f"{target} is a directory (use list_dir)."
    try:
        with target.open('rb') as fh:
            if b'\x00' in fh.read(4096):
                return f"{target} appears to be a binary file; not shown."
        full = target.read_text(encoding='utf-8', errors='replace')
    except OSError as e:
        return f"Cannot read {target}: {e}"

    sha = _sha(full)
    text_lines = full.splitlines()
    total = len(text_lines)
    if total == 0:
        return f"{target} is empty. (sha:{sha})"
    start, end = _parse_range(lines, total)
    if start is None:
        return (f"Invalid line range '{lines}'. Use 'start-end', e.g."
                f" '10-20'. {target} has {total} lines.")
    selected = text_lines[start - 1:end]
    body = "\n".join(
        f"{i:>6}\t{ln}" for i, ln in enumerate(selected, start))
    header = f"{target} (lines {start}-{end} of {total}, sha:{sha}):"
    note = ''
    if not lines.strip() and total > _MAX_READ_LINES:
        nxt = min(total, _MAX_READ_LINES * 2)
        note = (f"\n… showing first {_MAX_READ_LINES} of {total} lines;"
                f" request a range like '{_MAX_READ_LINES + 1}-{nxt}'"
                f" for more.")
    return f"{header}\n{body}{note}"


def _walk_files(root: Path):
    """Yield files under root (breadth-ish), skipping noise directories."""
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            children = _sorted_children(d)
        except OSError:
            continue
        for p in children:
            if p.is_dir():
                if p.name not in _NOISE_DIRS:
                    stack.append(p)
            else:
                yield p


def search_code(pattern: str, path: str = '.') -> str:
    """
    Search file contents for a string or regular expression under a directory
    (like grep), returning matching 'relpath:line: text' rows. Use this to
    find where something is defined or used — e.g. 'def compact_messages' or
    'web_fetch' — before concluding code or a feature is missing. Noise dirs
    (.git, node_modules, …) and binary/oversized files are skipped.
    """
    root = _resolve(path)
    if not ensure_path_allowed(root):
        return f"Access to '{root}' was denied by the user."
    if not root.exists():
        return f"No such path: {root}"
    try:
        rx = re.compile(pattern)
    except re.error:
        rx = re.compile(re.escape(pattern))

    files = [root] if root.is_file() else _walk_files(root)
    rows: list = []
    truncated = False
    for f in files:
        if len(rows) >= _MAX_MATCHES:
            truncated = True
            break
        try:
            if f.stat().st_size > _MAX_FILE_BYTES:
                continue
            with f.open('rb') as fh:
                if b'\x00' in fh.read(2048):
                    continue
            text = f.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        rel = f.name if root.is_file() else f.relative_to(root)
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                rows.append(f"{rel}:{i}: {line.strip()[:200]}")
                if len(rows) >= _MAX_MATCHES:
                    truncated = True
                    break

    if not rows:
        return f"No matches for {pattern!r} under {root}."
    out = [f"{root} — matches for {pattern!r}:"] + rows
    if truncated:
        out.append(
            f"... stopped at {_MAX_MATCHES} matches — narrow the pattern"
            f" or path.")
    return "\n".join(out)


# --- write tools (gated by the WRITE allow-list + access mode) ---------------

_MAX_DIFF_LINES = 200       # cap diff lines shown in the write prompt
# Diff row colours: very dark green/red background with light text, so the
# highlighted block reads clearly (the earlier 256-colour bg was too bright).
_BG_G = '\x1b[48;2;0;28;0m\x1b[38;5;150m'
_BG_R = '\x1b[48;2;38;0;0m\x1b[38;5;210m'
_D = '\x1b[2m'              # dim — context lines
_Z = '\x1b[0m'
_HUNK = re.compile(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@')


def _plural(n: int) -> str:
    return f"{n} line{'' if n == 1 else 's'}"


def _diff_row(lineno: int, sign: str, content: str, bg: str) -> str:
    """One diff row: '<n> <sign> <content>' as a full-width coloured block
    (one column short of the terminal to avoid cursor wrap)."""
    width = max(20, shutil.get_terminal_size(fallback=(100, 30)).columns - 1)
    text = f" {lineno:>4} {sign} {content}"[:width].ljust(width)
    return f"{bg or _D}{text}{_Z}"


def _write_detail(target: Path, old: str, new: str, verb: str) -> str:
    """A Claude-style diff of the pending change:

        ⏺ Update(name.py) added 2 lines, removed 1 line
           41 - old line
           41 + new line

    line-numbered, '+'/'-' as full-width green/red background blocks, dim
    context, no hunk headers, and zero counts omitted from the summary.
    """
    added = removed = 0
    rows: list = []
    truncated = False
    old_ln = new_ln = 0
    for ln in difflib.unified_diff(
            old.splitlines(), new.splitlines(), lineterm='', n=3):
        m = _HUNK.match(ln)
        if m:
            old_ln, new_ln = int(m.group(1)), int(m.group(2))
            continue
        if ln.startswith(('+++', '---')):
            continue
        if ln.startswith('+'):
            added += 1
            row = _diff_row(new_ln, '+', ln[1:], _BG_G)
            new_ln += 1
        elif ln.startswith('-'):
            removed += 1
            row = _diff_row(old_ln, '-', ln[1:], _BG_R)
            old_ln += 1
        else:
            row = _diff_row(new_ln, ' ', ln[1:], '')
            old_ln += 1
            new_ln += 1
        if len(rows) < _MAX_DIFF_LINES:
            rows.append(row)
        else:
            truncated = True
    if truncated:
        rows.append(f"{_D} … more lines{_Z}")
    summary = ", ".join(
        p for p in (
            f"added {_plural(added)}" if added else '',
            f"removed {_plural(removed)}" if removed else '') if p
    ) or "no changes"
    header = f"⏺ {verb}({target.name}) {summary}"
    return header + ("\n" + "\n".join(rows) if rows else "")


def _will_prompt_write(target: Path) -> bool:
    """Whether an approval prompt (with the diff) will be shown for a write."""
    return (config.MODE != config.MODE_AUTO
            and not _within_allowed(target, config.ALLOWED_WRITE_DIRS))


def _show_change(diff: str) -> None:
    """Print the applied diff to the console — the persistent record of a
    write that did not prompt (whitelisted dir / auto mode), so the change is
    always visible, not only when access is being asked for."""
    try:
        from rich.text import Text
        ui.console.print(Text.from_ansi(diff))
    except Exception:                                    # noqa: BLE001
        ui.console.print(diff)


def write_file(path: str, content: str) -> str:
    """
    Create or overwrite a file with the given content. Needs write access to
    the target directory (asked once, shown with the exact write); refused in
    read-only mode. Prefer edit_file for changing part of an existing file.
    """
    target = _resolve(path)
    if config.MODE == config.MODE_READ_ONLY:
        return "Refused: read-only mode. Change mode to write files."
    if target.is_dir():
        return f"{target} is a directory."
    old_content = ''
    if target.exists():
        try:
            old_content = target.read_text(encoding='utf-8', errors='replace')
        except OSError:
            old_content = ''
    verb = 'Update' if target.exists() else 'Create'
    block = _write_detail(target, old_content, content, verb)
    silent = not _will_prompt_write(target)
    if not ensure_write_path_allowed(target, block):
        return f"Write access to '{target}' was denied."
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
    except OSError as e:
        return f"Cannot write {target}: {e}"
    if silent:
        _show_change(block)
    return f"Wrote {len(content)} bytes to {target}. (sha:{_sha(content)})"


def edit_file(path: str, old: str, new: str, sha: str) -> str:
    """
    Replace a single unique occurrence of ``old`` with ``new`` in a file.
    ``old`` must appear exactly once (include enough surrounding context to be
    unique). You MUST pass ``sha`` — the sha your most recent read_file,
    write_file, or edit_file of this file returned (reuse it; no need to
    re-read). This call returns the new sha, so consecutive edits can chain.
    If the sha no longer matches (the file changed underneath you) the edit is
    refused — read the file again to refresh the sha, then retry. Needs write
    access; refused in read-only mode.
    """
    target = _resolve(path)
    if config.MODE == config.MODE_READ_ONLY:
        return "Refused: read-only mode. Change mode to write files."
    if not target.exists():
        return f"No such file: {target}"
    if target.is_dir():
        return f"{target} is a directory."
    try:
        text = target.read_text(encoding='utf-8')
    except OSError as e:
        return f"Cannot read {target}: {e}"
    current = _sha(text)
    if (sha or '').strip() != current:
        return (f"sha mismatch: you passed sha:{sha or '(none)'} but"
                f" {target} is now sha:{current}. Read the file again with"
                " read_file to get its current content and sha, then retry"
                " edit_file with that sha.")
    count = text.count(old)
    if count == 0:
        return f"'old' text not found in {target}; nothing changed."
    if count > 1:
        return (f"'old' text appears {count} times in {target}; add context"
                " to make it unique.")
    new_text = text.replace(old, new, 1)
    block = _write_detail(target, text, new_text, 'Update')
    silent = not _will_prompt_write(target)
    if not ensure_write_path_allowed(target, block):
        return f"Write access to '{target}' was denied."
    try:
        target.write_text(new_text, encoding='utf-8')
    except OSError as e:
        return f"Cannot write {target}: {e}"
    if silent:
        _show_change(block)
    return (f"Edited {target} (replaced 1 occurrence)."
            f" (sha:{_sha(new_text)})")


def delete_file(path: str) -> str:
    """
    Delete a single file. Destructive and write-gated: refused in read-only
    mode, asked once per directory (showing which file), auto-approved in auto
    mode. Refuses to delete directories.
    """
    target = _resolve(path)
    if config.MODE == config.MODE_READ_ONLY:
        return "Refused: read-only mode. Change mode to delete files."
    if not target.exists():
        return f"No such file: {target}"
    if target.is_dir():
        return f"{target} is a directory; refusing to delete directories."
    block = f"⏺ Delete({target.name})  {target}"
    silent = not _will_prompt_write(target)
    if not ensure_write_path_allowed(target, block):
        return f"Write access to '{target}' was denied."
    try:
        target.unlink()
    except OSError as e:
        return f"Cannot delete {target}: {e}"
    if silent:
        ui.console.print(f"[red]{block}[/red]")
    return f"Deleted {target}."
