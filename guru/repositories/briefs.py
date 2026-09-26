"""Where project briefs persist: ``<root>/<project-key>/<head>.json``.

One JSON file per (project, HEAD); ``guru.domain.brief.current`` loads the
one for the checked-out HEAD and saves a fresh build. The project key is the
directory name plus a short hash of its absolute path, so two checkouts with
the same name do not share briefs. ``KEEP`` newest briefs are kept per
project and ``KEEP_PROJECTS`` newest project directories under the root;
older ones are removed on save. The root is ``~/.guru/briefs`` unless the
environment variable ``GURU_BRIEFS_DIR`` names another directory -- the
eval runner points it inside each case's workdir, so fixture copies (a new
absolute path, hence a new project key, per case) never pile up under the
developer's home. This module satisfies ``brief.BriefStore``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from guru import config, log
from guru.domain.brief import Brief

BRIEFS_DIR = config.GURU_HOME / 'briefs'
BRIEFS_DIR_ENV = 'GURU_BRIEFS_DIR'
KEEP = 5
KEEP_PROJECTS = 20


def root() -> Path:
    """The store root: ``$GURU_BRIEFS_DIR`` when set, else ``BRIEFS_DIR``."""
    override = os.environ.get(BRIEFS_DIR_ENV, '').strip()
    return Path(override) if override else BRIEFS_DIR


def project_key(project: Path) -> str:
    """``<name>-<8 hex of the absolute path>`` for ``project``."""
    resolved = Path(project).resolve()
    digest = hashlib.sha1(str(resolved).encode('utf-8')).hexdigest()[:8]
    return f'{resolved.name or "root"}-{digest}'


def path_for(project: Path, head_sha: str,
             base: Optional[Path] = None) -> Path:
    """The brief file for ``project`` at ``head_sha`` under ``base``
    (default :func:`root`)."""
    safe = ''.join(c for c in head_sha if c.isalnum()) or 'nogit'
    base = Path(base) if base is not None else root()
    return base / project_key(project) / f'{safe}.json'


def load(project: Path, head_sha: str,
         base: Optional[Path] = None) -> Optional[Brief]:
    """The stored brief, or None when absent or unreadable (logged)."""
    path = path_for(project, head_sha, base)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        log.exc(f'brief unreadable: {path}')
        return None
    if not isinstance(data, dict):
        return None
    try:
        return Brief.from_dict(data)
    except TypeError:
        log.exc(f'brief malformed: {path}')
        return None


def save(brief: Brief, base: Optional[Path] = None) -> Path:
    """Write ``brief`` for its root/HEAD and prune: the project's directory
    to the ``KEEP`` newest files, the store to the ``KEEP_PROJECTS`` newest
    project directories. Returns the path written."""
    path = path_for(Path(brief.root), brief.head_sha, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(brief.to_dict(), ensure_ascii=False,
                               indent=1), encoding='utf-8')
    _prune(path.parent, keep=path)
    return path


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _prune(directory: Path, keep: Path) -> None:
    """Oldest-first removal: files beyond ``KEEP`` in ``directory``, then
    project directories beyond ``KEEP_PROJECTS`` under its parent (the
    directory just written to is never removed)."""
    briefs = sorted(directory.glob('*.json'), key=_mtime, reverse=True)
    for old in briefs[KEEP:]:
        if old != keep:
            try:
                old.unlink()
            except OSError:
                log.exc(f'brief prune failed: {old}')
    projects = sorted((d for d in directory.parent.iterdir() if d.is_dir()),
                      key=_mtime, reverse=True)
    for old in projects[KEEP_PROJECTS:]:
        if old.resolve() != directory.resolve():
            shutil.rmtree(old, ignore_errors=True)
