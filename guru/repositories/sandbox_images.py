"""Per-project sandbox image records under ``~/.guru/sandbox/<name>/``
(design plan §1, repository layer).

``image.json`` remembers what was built — tag, digest, the lockfile sha and
the sha of the generated Dockerfile, ``built_at`` — next to the Dockerfile
itself, so :func:`needs_build` can say whether the recorded image still
matches the project (a new lockfile or a changed Dockerfile means a
rebuild). :func:`record_sandbox_event` appends to the ``sandbox_events``
ledger stream; :func:`record_net_events` aggregates the provisioning
proxy's access log into the ``net_events`` stream (one row per host, port,
method and outcome with a count). ``deps.json`` in the same directory holds
the pending :class:`~guru.domain.deps.DependencyRequest` records (S2).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from guru import config, log, session
from guru.domain import ledger
from guru.domain.deps import DependencyRequest, normalise
from guru.domain.sandbox import SandboxSpec, sha_text

RECORD_FILE = 'image.json'
DOCKERFILE = 'Dockerfile'
DEPS_FILE = 'deps.json'                # pending dependency requests
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


def _session_keys() -> dict:
    return {**ledger.base_row(), 'agent': session.agent_id,
            'task_id': session.task_id, 'turn_id': session.turn_id}


def record_net_events(events: list[dict], phase: str) -> int:
    """Aggregate the proxy access-log ``events`` (dicts with ``host``,
    ``port``, ``method``, ``allowed``, ``reason`` as
    ``guru.sandbox.proxy.parse_log`` yields them) into ``net_events`` rows:
    one per distinct (host, port, method, allowed, reason) with ``count``
    and the provisioning ``phase`` (``build``, ``uv add``). Returns the
    number of rows written (0 when nothing to record). Never raises."""
    try:
        counts: dict[tuple, int] = {}
        for ev in events or []:
            if not isinstance(ev, dict):
                continue
            key = (str(ev.get('host', ''))[:ledger.ARGS_HEAD],
                   int(ev.get('port') or 0),
                   str(ev.get('method', ''))[:16],
                   bool(ev.get('allowed')),
                   str(ev.get('reason', ''))[:ledger.ARGS_HEAD])
            counts[key] = counts.get(key, 0) + 1
        for (host, port, method, allowed, reason), n in sorted(
                counts.items()):
            ledger.submit('net_events', {
                **_session_keys(), 'phase': str(phase)[:32], 'host': host,
                'port': port, 'method': method, 'allowed': allowed,
                'reason': reason, 'count': n})
        return len(counts)
    except Exception:                            # noqa: BLE001
        log.exc('ledger record_net_events failed')
        return 0


# --- pending dependency requests ---------------------------------------------

def deps_path(spec: SandboxSpec) -> Path:
    """Where ``spec``'s pending dependency requests live."""
    return record_dir(spec) / DEPS_FILE


def pending_requests(spec: SandboxSpec) -> list[DependencyRequest]:
    """The recorded, not yet applied requests in request order; empty when
    the file is absent, unreadable or malformed (logged)."""
    path = deps_path(spec)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        log.exc(f'sandbox: unreadable dependency requests {path}')
        return []
    items = data.get('requests') if isinstance(data, dict) else None
    out: list[DependencyRequest] = []
    for item in items or []:
        if (isinstance(item, dict) and isinstance(item.get('name'), str)
                and isinstance(item.get('constraint', ''), str)):
            out.append(DependencyRequest(
                name=item['name'], constraint=item.get('constraint', ''),
                requested_at=str(item.get('requested_at', ''))))
    return out


def _save_requests(spec: SandboxSpec, requests: list[DependencyRequest]
                   ) -> None:
    path = deps_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {'requests': [asdict(r) for r in requests]}, indent=1,
        sort_keys=True) + '\n', encoding='utf-8')


def add_request(spec: SandboxSpec, request: DependencyRequest
                ) -> list[DependencyRequest]:
    """Record ``request`` (stamped ``requested_at``), replacing an earlier
    request for the same normalised name; returns the pending list."""
    stamped = DependencyRequest(
        name=request.name, constraint=request.constraint,
        requested_at=datetime.now(timezone.utc).isoformat(
            timespec='seconds'))
    pending = [r for r in pending_requests(spec) if r.key != stamped.key]
    pending.append(stamped)
    _save_requests(spec, pending)
    return pending


def remove_request(spec: SandboxSpec, name: str) -> bool:
    """Drop the request whose normalised name matches ``name``; True when
    one was removed."""
    key = normalise(name)
    pending = pending_requests(spec)
    kept = [r for r in pending if r.key != key]
    if len(kept) == len(pending):
        return False
    _save_requests(spec, kept)
    return True
