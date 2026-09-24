"""Per-project sandbox image records under ``~/.guru/sandbox/<name>/``
(design plan §1, repository layer).

``image.json`` remembers what was built — tag, digest, the lockfile sha and
the sha of the generated Dockerfile, ``built_at`` — next to the Dockerfile
itself, so :func:`needs_build` can say whether the recorded image still
matches the project (a new lockfile or a changed Dockerfile means a
rebuild). :func:`record_sandbox_event` appends to the ``sandbox_events``
ledger stream.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from guru import config, log, session
from guru.domain import ledger
from guru.domain.sandbox import SandboxSpec, sha_text

RECORD_FILE = 'image.json'
DOCKERFILE = 'Dockerfile'
WORK_DIR = 'work'                      # working copies live under here
_RECORD_KEYS = ('tag', 'digest', 'lockfile_sha', 'built_at',
                'dockerfile_sha')


@dataclass(frozen=True)
class ImageRecord:
    """What ``image.json`` holds about the last successful build."""
    tag: str
    digest: str
    lockfile_sha: str
    built_at: str
    dockerfile_sha: str


def record_dir(spec: SandboxSpec) -> Path:
    """``~/.guru/sandbox/<name>/`` for ``spec`` (``config.SANDBOX_HOME``)."""
    return Path(config.SANDBOX_HOME) / spec.name


def dockerfile_path(spec: SandboxSpec) -> Path:
    """Where the generated Dockerfile is written."""
    return record_dir(spec) / DOCKERFILE


def work_root(spec: SandboxSpec) -> Path:
    """Parent directory for ``spec``'s working copies (a path Colima
    mounts into its VM, unlike a macOS temp dir)."""
    return record_dir(spec) / WORK_DIR


def write_dockerfile(spec: SandboxSpec, text: str) -> Path:
    """Write ``text`` to :func:`dockerfile_path` (creating the record dir)
    and return the path."""
    path = dockerfile_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def load_record(spec: SandboxSpec) -> Optional[ImageRecord]:
    """The stored :class:`ImageRecord`, or None when there is none or the
    file is unreadable/corrupt (logged; a rebuild follows)."""
    path = record_dir(spec) / RECORD_FILE
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        log.exc(f'sandbox: unreadable image record {path}')
        return None
    if not isinstance(data, dict) or not all(
            isinstance(data.get(k), str) for k in _RECORD_KEYS):
        log.warning('sandbox: malformed image record %s; ignoring', path)
        return None
    return ImageRecord(**{k: data[k] for k in _RECORD_KEYS})


def save_record(spec: SandboxSpec, record: ImageRecord) -> Path:
    """Write ``record`` as ``image.json`` and return its path."""
    path = record_dir(spec) / RECORD_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=1, sort_keys=True)
                    + '\n', encoding='utf-8')
    return path


def record_built(spec: SandboxSpec, dockerfile: str, digest: str
                 ) -> ImageRecord:
    """Record a successful build of ``spec`` from ``dockerfile`` (its text)
    that produced ``digest``; returns the saved record."""
    rec = ImageRecord(
        tag=spec.image_tag, digest=digest, lockfile_sha=spec.lockfile_sha,
        built_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        dockerfile_sha=sha_text(dockerfile))
    save_record(spec, rec)
    return rec


def needs_build(spec: SandboxSpec, dockerfile: str) -> bool:
    """True unless a record exists for ``spec.image_tag`` with the same
    lockfile sha and the sha of ``dockerfile``."""
    rec = load_record(spec)
    if rec is None:
        return True
    return (rec.tag != spec.image_tag
            or rec.lockfile_sha != spec.lockfile_sha
            or rec.dockerfile_sha != sha_text(dockerfile))


def _argv_head(argv: object) -> list[str]:
    if not isinstance(argv, (list, tuple)):
        return [str(argv)[:ledger.ARGS_HEAD]]
    return [str(a)[:ledger.ARGS_HEAD] for a in argv]


def record_sandbox_event(kind: str, argv_head: object, seconds: float,
                         ok: bool, detail: str) -> None:
    """Append one row to the ``sandbox_events`` stream: ``kind`` (``build``,
    ``run``, ``kill``, ``copy``, ``diff``), the argv with each element cut
    at ``ledger.ARGS_HEAD``, ``seconds``, ``ok`` and a short ``detail``
    (exit code, denial, digest). Session join keys as for tool events.
    Never raises."""
    try:
        ledger.submit('sandbox_events', {
            **ledger.base_row(), 'agent': session.agent_id,
            'task_id': session.task_id, 'turn_id': session.turn_id,
            'kind': str(kind), 'argv': _argv_head(argv_head),
            'seconds': float(seconds), 'ok': bool(ok),
            'detail': str(detail or '')[:ledger.ARGS_HEAD]})
    except Exception:                            # noqa: BLE001
        log.exc('ledger record_sandbox_event failed')
