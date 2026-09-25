"""Code navigation without reading whole files: ``outline`` and
``find_symbol`` (design plan B1).

``outline`` turns a Python module into one line per def/class with its line
range and signature (``ast``), so a model can decide which span to read;
``find_symbol`` answers "where is X defined and who uses it" across the
project's ``.py`` files in one call. Both are reads gated by
``files.ensure_path_allowed``; both stay small (caps below), the point being
to replace whole-file reads, not to add to them.

Stdlib only.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator, Optional

from guru.domain import files

_MAX_ENTRIES = 80           # outline rows before "… N more"
_MAX_PLAIN_LINES = 40       # non-Python fallback: numbered head of the file
_MAX_ROWS = 30              # find_symbol rows (defs first, then refs)
_LINE_HEAD = 160            # chars of a reference line shown
_KINDS = ('', 'def', 'ref')
_DIGEST_BYTES = 4096        # safety net on any digest this module returns

_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


# --- outline -----------------------------------------------------------------

def _signature(node: ast.AST) -> str:
    """``def name(args) -> ret`` / ``class Name(Bases)`` from an AST node."""
    if isinstance(node, ast.ClassDef):
        bases = ', '.join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    prefix = 'async def' if isinstance(node, ast.AsyncFunctionDef) else 'def'
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ''
    return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"


def _entries(body: list, depth: int) -> Iterator[str]:
    """Outline rows for ``body`` (a module or block body), nested defs
    indented two spaces per level."""
    for node in body:
        if isinstance(node, _DEF_NODES):
            end = getattr(node, 'end_lineno', None) or node.lineno
            yield (f"{'  ' * depth}L{node.lineno}-{end} "
                   f"{_signature(node)}")
            yield from _entries(node.body, depth + 1)


def _clip(text: str, limit: int = _DIGEST_BYTES) -> str:
    """Hard cap on a digest (the row caps keep it far smaller normally)."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n… ({len(text) - limit} more chars)"


def _numbered_head(text: str, path: Path, sha: str, note: str = '') -> str:
    lines = text.splitlines()
    shown = lines[:_MAX_PLAIN_LINES]
    head = f"{path} ({len(lines)} lines, sha:{sha})"
    if note:
        head += f"\n{note}"
    body = "\n".join(f"{i:>6}\t{ln}" for i, ln in enumerate(shown, 1))
    tail = ''
    if len(lines) > len(shown):
        tail = (f"\n… {len(lines) - len(shown)} more lines; use read_file"
                " with a range for the rest.")
    return f"{head}\n{body}{tail}"


def outline(path: str) -> str:
    """
    Outline a source file instead of reading it: for a Python module, one
    row per def/class (nested ones indented) as 'L<start>-<end> <signature>'
    plus the module docstring's first line, so you can pick the exact line
    range to read_file next. Other text files show their first 40 numbered
    lines. Returns the file's sha (usable with edit_file). Restricted to
    allowed directories.
    """
    target = files.resolve_path(path)
    if not files.ensure_path_allowed(target):
        return f"Access to '{target}' was denied by the user."
    if not target.exists():
        return f"No such file: {target}"
    if target.is_dir():
        return f"{target} is a directory (use list_dir)."
    try:
        with target.open('rb') as fh:
            if b'\x00' in fh.read(4096):
                return f"{target} appears to be a binary file; not shown."
        text = target.read_text(encoding='utf-8', errors='replace')
    except OSError as e:
        return f"Cannot read {target}: {e}"
    sha = files.sha_of(text)
    files.remember_sha(target, sha)
    if target.suffix != '.py':
        return _clip(_numbered_head(text, target, sha))
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return _clip(_numbered_head(
            text, target, sha,
            f"(not parseable: SyntaxError line {e.lineno}: {e.msg})"))
    total = len(text.splitlines())
    out = [f"{target} ({total} lines, sha:{sha}) outline:"]
    doc = ast.get_docstring(tree)
    if doc:
        out.append(f'"""{doc.splitlines()[0]}"""')
    rows = list(_entries(tree.body, 0))
    if not rows:
        out.append("(no def/class at any level)")
    out.extend(rows[:_MAX_ENTRIES])
    if len(rows) > _MAX_ENTRIES:
        out.append(f"… {len(rows) - _MAX_ENTRIES} more entries; outline a"
                   " narrower file or read_file a range.")
    return _clip("\n".join(out))


# --- find_symbol -------------------------------------------------------------

def _def_kind(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        return 'class'
    if isinstance(node, ast.AsyncFunctionDef):
        return 'async def'
    return 'def'


def _definitions(tree: ast.Module, name: str) -> list[tuple[int, str]]:
    """``(line, kind)`` for every def/class named ``name`` at any depth and
    every module-level assignment to it."""
    out: list = []
    for node in ast.walk(tree):
        if isinstance(node, _DEF_NODES) and node.name == name:
            out.append((node.lineno, _def_kind(node)))
    for node in tree.body:
        targets: list = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name:
                out.append((node.lineno, 'assign'))
            elif isinstance(t, (ast.Tuple, ast.List)):
                for elt in t.elts:
                    if isinstance(elt, ast.Name) and elt.id == name:
                        out.append((node.lineno, 'assign'))
    return sorted(out)


def _python_files(root: Path) -> Iterator[Path]:
    for f in files.walk_files(root):
        if f.suffix != '.py':
            continue
        try:
            if f.stat().st_size > files.MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield f


def _read(f: Path) -> Optional[str]:
    try:
        return f.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None


def find_symbol(name: str, kind: str = '') -> str:
    """
    Find where a symbol (function, class, method or module-level name) is
    defined and where it is referenced across the project's Python files,
    in one call: 'def: relpath:line (class|def|assign)' rows first, then
    'ref: relpath:line: text' rows, capped at 30. ``kind`` narrows to 'def'
    or 'ref' (default both). The project is the allowed directory containing
    the working directory. Prefer this over search_code + read_file for
    "where is X / who calls X" questions.
    """
    name = (name or '').strip()
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
        return f"'{name}' is not a symbol name (identifiers only)."
    kind = (kind or '').strip().lower()
    if kind not in _KINDS:
        return f"kind must be 'def', 'ref' or empty, not {kind!r}."
    root = files.project_root(Path.cwd())
    if not files.ensure_path_allowed(root):
        return f"Access to '{root}' was denied by the user."
    word = re.compile(rf'\b{re.escape(name)}\b')
    defs: list = []
    refs: list = []
    total_refs = 0
    for f in _python_files(root):
        text = _read(f)
        if text is None or name not in text:
            continue
        rel = f.relative_to(root)
        def_lines: set = set()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:          # defs are never listed as refs
            for line, what in _definitions(tree, name):
                def_lines.add(line)
                if kind != 'ref':
                    defs.append(f"def: {rel}:{line} ({what})")
        if kind != 'def':
            for i, row in enumerate(text.splitlines(), 1):
                if i in def_lines or not word.search(row):
                    continue
                total_refs += 1
                if len(refs) < _MAX_ROWS:
                    refs.append(
                        f"ref: {rel}:{i}: {row.strip()[:_LINE_HEAD]}")
    rows = (defs + refs)[:_MAX_ROWS]
    more = len(defs) + total_refs - len(rows)
    if not rows:
        what = {'def': 'definition', 'ref': 'reference'}.get(kind, 'match')
        return f"No {what} of '{name}' in Python files under {root}."
    out = [f"{root} — '{name}': {len(defs)} definition(s),"
           f" {total_refs} reference(s):"] + rows
    if more:
        out.append(f"… {more} more; narrow with kind='def' or search_code.")
    return _clip("\n".join(out))
