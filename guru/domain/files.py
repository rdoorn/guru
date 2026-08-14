"""Filesystem tools: directory listing and file reading, gated by a
per-project directory allow-list.

Access is restricted to allow-listed directory subtrees. Nothing is allowed by
default: the first access to a directory (including the working directory) is
approved once, and the approval is persisted to ``.guru/dirs_allow.txt`` so it
is not asked again. Paths are resolved to real absolute paths before the check,
so ``..`` and symlink escapes cannot leave an allowed subtree.
"""
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


# --- directory allow-list gate ----------------------------------------------

_path_asker = None


def set_path_asker(fn) -> None:
    """Install a custom directory-approval prompt (used by the TUI)."""
    global _path_asker
    _path_asker = fn


def _ask_path(directory: str) -> bool:
    """Default terminal approval prompt (REPL). Enter approves; errors deny."""
    ui.console.print(
        f"\n[yellow]\\[ACCESS][/yellow] Request to read files under"
        f" [bold]{directory}[/bold]."
    )
    try:
        answer = input(
            f"Allow file access to '{directory}'? [Y/n] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return not answer.startswith('n')


def _within_allowed(resolved: Path) -> bool:
    for d in config.ALLOWED_DIRS:
        allowed = Path(d)
        if resolved == allowed or allowed in resolved.parents:
            return True
    return False


def ensure_path_allowed(path: Path) -> bool:
    """Return True if ``path`` sits in an allowed subtree, else prompt.

    On approval the containing directory is added to the allow-list and
    persisted so it is not asked about again.
    """
    if _within_allowed(path):
        return True
    ask_dir = (path if path.is_dir() else path.parent).resolve()
    asker = _path_asker or _ask_path
    if asker(str(ask_dir)):
        config.ALLOWED_DIRS.add(str(ask_dir))
        config.persist_dir(str(ask_dir))
        ui.console.print(
            f"[green]Allowed[/green] {ask_dir} (saved to allow-list).")
        return True
    ui.console.print(f"[red]Denied[/red] {ask_dir}.")
    return False


# --- formatting helpers ------------------------------------------------------

def _octal(mode: int) -> str:
    return format(stat.S_IMODE(mode), '04o')


def _human(size: int) -> str:
    value = float(size)
    for unit in ('B', 'K', 'M', 'G', 'T'):
        if value < 1024 or unit == 'T':
            if unit == 'B':
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def _entry_line(p: Path, indent: int, st) -> str:
    is_dir = stat.S_ISDIR(st.st_mode)
    name = p.name + ('/' if is_dir else '')
    size = '-' if is_dir else _human(st.st_size)
    return f"{_octal(st.st_mode)} {size:>7} {'  ' * indent}{name}"


def _resolve(path: str) -> Path:
    return Path(path or '.').expanduser().resolve()


def _sorted_children(d: Path) -> list:
    return sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))


# --- tools -------------------------------------------------------------------

def list_dir(path: str = '.') -> str:
    """
    List the immediate contents of a directory (non-recursive) with octal
    permissions and sizes. Directories are shown with a trailing slash. Use
    list_tree for a recursive view, or read_file to read a file.
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
            lines.append(_entry_line(p, 0, p.lstat()))
        except OSError:
            continue
    if len(lines) == 1:
        lines.append("(empty)")
    return "\n".join(lines)


def list_tree(path: str = '.', depth: str = '') -> str:
    """
    List a directory tree recursively with octal permissions and sizes.
    Noise directories (.git, node_modules, __pycache__, .venv, …) are shown
    but not expanded — to look inside one, call this with its full path.
    ``depth`` limits how many levels to recurse (default 3).
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

    lines = [f"{target}:"]
    state = {'count': 0, 'truncated': False}

    def walk(d: Path, level: int) -> None:
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
            skip = is_dir and p.name in _NOISE_DIRS
            lines.append(
                _entry_line(p, level, st) + ('  (skipped)' if skip else ''))
            state['count'] += 1
            if is_dir and not skip and level + 1 < max_depth:
                walk(p, level + 1)

    walk(target, 0)
    if state['truncated']:
        lines.append(
            f"… truncated at {_MAX_ENTRIES} entries — narrow with a subpath"
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
    line-numbered. Binary files are not shown.
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
        text_lines = target.read_text(
            encoding='utf-8', errors='replace').splitlines()
    except OSError as e:
        return f"Cannot read {target}: {e}"

    total = len(text_lines)
    if total == 0:
        return f"{target} is empty."
    start, end = _parse_range(lines, total)
    if start is None:
        return (f"Invalid line range '{lines}'. Use 'start-end', e.g."
                f" '10-20'. {target} has {total} lines.")
    selected = text_lines[start - 1:end]
    body = "\n".join(
        f"{i:>6}\t{ln}" for i, ln in enumerate(selected, start))
    header = f"{target} (lines {start}-{end} of {total}):"
    note = ''
    if not lines.strip() and total > _MAX_READ_LINES:
        nxt = min(total, _MAX_READ_LINES * 2)
        note = (f"\n… showing first {_MAX_READ_LINES} of {total} lines;"
                f" request a range like '{_MAX_READ_LINES + 1}-{nxt}'"
                f" for more.")
    return f"{header}\n{body}{note}"
