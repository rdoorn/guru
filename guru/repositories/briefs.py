"""Where project briefs persist: ``~/.guru/briefs/<project-key>/<head>.json``.

One JSON file per (project, HEAD); ``guru.domain.brief.current`` loads the
one for the checked-out HEAD and saves a fresh build. The project key is the
directory name plus a short hash of its absolute path, so two checkouts with
the same name do not share briefs. ``KEEP`` newest briefs are kept per
project; older ones are removed on save. This module satisfies
``brief.BriefStore``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from guru import config, log
from guru.domain.brief import Brief

BRIEFS_DIR = config.GURU_HOME / 'briefs'
KEEP = 5


def project_key(root: Path) -> str:
    """``<name>-<8 hex of the absolute path>`` for ``root``."""
    resolved = Path(root).resolve()
    digest = hashlib.sha1(str(resolved).encode('utf-8')).hexdigest()[:8]
    return f'{resolved.name or "root"}-{digest}'


def path_for(root: Path, head_sha: str, base: Optional[Path] = None) -> Path:
    """The brief file for ``root`` at ``head_sha`` under ``base``
    (default ``BRIEFS_DIR``)."""
    safe = ''.join(c for c in head_sha if c.isalnum()) or 'nogit'
    return Path(base or BRIEFS_DIR) / project_key(root) / f'{safe}.json'


def load(root: Path, head_sha: str,
         base: Optional[Path] = None) -> Optional[Brief]:
    """The stored brief, or None when absent or unreadable (logged)."""
    path = path_for(root, head_sha, base)
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
    """Write ``brief`` for its root/HEAD and prune the project's directory
    to the ``KEEP`` newest files. Returns the path written."""
    path = path_for(Path(brief.root), brief.head_sha, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(brief.to_dict(), ensure_ascii=False,
                               indent=1), encoding='utf-8')
    _prune(path.parent, keep=path)
    return path


def _prune(directory: Path, keep: Path) -> None:
    briefs = sorted(directory.glob('*.json'), key=lambda p: p.stat().st_mtime,
                    reverse=True)
    for old in briefs[KEEP:]:
        if old != keep:
            try:
                old.unlink()
            except OSError:
                log.exc(f'brief prune failed: {old}')
