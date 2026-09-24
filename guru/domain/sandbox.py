"""Sandboxed execution — the domain half (design plan
``2026-09-24-sandbox-design-and-plan.md`` §1, chunk S1).

A :class:`SandboxSpec` names everything the runtime needs to build and run
one project's sandbox image: the project, a Docker-safe name, the pinned
base image, the image tag (derived from the lockfile sha), and the resource
limits. :func:`dockerfile_for` generates the Dockerfile from
``pyproject.toml`` + ``uv.lock`` — the only packages in the image are the
lockfile's (``uv sync --frozen``); the base image must be pinned by digest.
:func:`copy_excludes` lists what never enters the working copy (and thus
the image): noise directories, ``.env*`` and every small text file the
secret scanner flags — the scanner is a parameter (default: the bound
``policy`` scanner) so the endpoint can bind the project scanner and copy
hygiene never depends on whether routing bound one.

Stdlib only; imports ``guru.config``, ``guru.log`` and sibling domain
modules. The docker CLI lives in the endpoint ``guru.sandbox.colima``.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from guru.domain.policy import ContentScanner

from guru import log
from guru.domain import files, policy
from guru.domain.procs import DENIED_PREFIX

# The uv release installed into the image (pinned: the build runs on the
# provisioning network, the run phase has none). Version-pinned, not
# hash-pinned: ``pip --require-hashes`` would need the hash of every
# platform wheel the multi-arch base image may select (manylinux/musllinux
# x aarch64/x86_64) kept in step with the version; the reliable alternative
# is ``COPY --from=ghcr.io/astral-sh/uv:<ver>@sha256:… /uv`` (digest-pinned
# binary, no pip at all), a follow-up.
UV_VERSION = '0.11.6'
# argv[0] basenames a sandbox run may start. Shells are refused as
# everywhere; ``pip`` is absent on purpose (installs only via the lockfile).
RUNNERS = frozenset(('python', 'python3', 'pytest', 'uv', 'ruff', 'mypy',
                     'flake8', 'make'))
# Files larger than this are copied without a secret scan (the scanner is
# for source and config, not data); binary files are skipped too.
SCAN_MAX_BYTES = 64 * 1024
# Copy-time excludes that hold for every project.
BASE_EXCLUDES = ('.env', '.env*')
# Where the project lands inside the container.
WORKDIR = '/work'
# Build args a provisioning build passes so pip/uv route through the
# filtering proxy (S2). Docker predefines these names: they reach every
# RUN without an ARG line, are excluded from ``docker history`` and never
# persist in the image (the run phase has no network anyway).
PROXY_BUILD_ARGS = ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
                    'NO_PROXY', 'no_proxy')
VENV = '/opt/venv'
_DIGEST_RX = re.compile(r'@sha256:[0-9a-f]{64}$')
_NAME_RX = re.compile(r'[^a-z0-9._-]+')


@dataclass(frozen=True)
class SandboxSpec:
    """Everything needed to build and run one project's sandbox."""
    project: Path
    name: str
    base_image: str
    image_tag: str
    lockfile_sha: str
    cpus: float
    memory_mb: int
    pids: int
    timeout_s: int


def sha_text(text: str) -> str:
    """sha256 hex of ``text`` (Dockerfile identity for the image record)."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def lockfile_sha(project: Path) -> str:
    """sha256 hex of the project's ``uv.lock`` bytes; ``''`` when absent."""
    try:
        return hashlib.sha256(
            (Path(project) / 'uv.lock').read_bytes()).hexdigest()
    except OSError:
        return ''


def check_base_image(base_image: str) -> str:
    """``''`` when ``base_image`` is a digest-pinned reference
    (``repo:tag@sha256:<64 hex>``), else why it is refused."""
    if not isinstance(base_image, str) or not base_image.strip():
        return 'base image is empty; expected <image>@sha256:<digest>'
    if any(ch.isspace() for ch in base_image):
        return f'base image {base_image!r} contains whitespace'
    if not _DIGEST_RX.search(base_image):
        return (f'base image {base_image!r} is not pinned by digest; '
                'expected <image>@sha256:<64 hex>')
    return ''


def dockerfile_for(project: Path, base_image: str) -> str:
    """The Dockerfile for ``project``'s sandbox image.

    ``FROM`` is the digest-pinned ``base_image`` (``ValueError`` otherwise),
    uv is installed at ``UV_VERSION``, the dependency layer is
    ``uv sync --frozen --all-groups --no-install-project`` from the copied
    ``pyproject.toml`` + ``uv.lock`` (all groups: the sandbox exists to run
    the project's tests and linters, which live in ``dev``), then the tree
    is copied and synced again so a packaged project is installed too. The
    virtualenv is ``/opt/venv`` — outside ``/work``, which the working copy
    is bind-mounted over at run time — and goes first on ``PATH``. The
    final ``ENV`` makes uv offline and no-sync so nothing installs at run
    time; the image runs as ``1000:1000`` in ``/work``. No ``ARG`` lines for
    the proxy: :func:`proxy_build_args` uses Docker's predefined proxy build
    args, which reach ``RUN`` undeclared and stay out of ``docker history``
    (declaring them would put the values into the history). Raises
    ``ValueError`` when the project lacks ``pyproject.toml`` or ``uv.lock``.
    """
    problem = check_base_image(base_image)
    if problem:
        raise ValueError(problem)
    root = Path(project)
    for needed in ('pyproject.toml', 'uv.lock'):
        if not (root / needed).is_file():
            raise ValueError(f'{root} has no {needed}; the sandbox builds '
                             'from a uv lockfile')
    sha = lockfile_sha(root)
    return '\n'.join((
        '# Generated by guru (guru.domain.sandbox.dockerfile_for); do not '
        'edit.',
        f'# uv.lock sha256: {sha}',
        f'FROM {base_image}',
        'ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 '
        f'UV_PROJECT_ENVIRONMENT={VENV} UV_LINK_MODE=copy',
        f'RUN pip install --no-cache-dir uv=={UV_VERSION}',
        f'WORKDIR {WORKDIR}',
        'COPY pyproject.toml uv.lock ./',
        'RUN uv sync --frozen --all-groups --no-install-project',
        f'COPY . {WORKDIR}',
        'RUN uv sync --frozen --all-groups',
        f'ENV PATH={VENV}/bin:$PATH UV_OFFLINE=1 UV_NO_SYNC=1',
        'USER 1000:1000',
        f'WORKDIR {WORKDIR}',
        '',
    ))


def proxy_build_args(proxy_url: str) -> dict[str, str]:
    """``--build-arg`` values pointing every proxy variable (upper and
    lower case) at ``proxy_url`` and clearing ``NO_PROXY``/``no_proxy`` so
    nothing bypasses it. Bypass is impossible anyway (the build network is
    internal, without a route); the variables are what makes the build
    work."""
    return {var: ('' if var.lower() == 'no_proxy' else str(proxy_url))
            for var in PROXY_BUILD_ARGS}


def _is_text_head(path: Path) -> bool:
    try:
        with path.open('rb') as fh:
            return b'\x00' not in fh.read(2048)
    except OSError:
        return False


def flagged_files(project: Path, scanner: Optional[ContentScanner] = None
                  ) -> dict[str, list[str]]:
    """``{relative path: [finding kinds]}`` for every text file of at most
    ``SCAN_MAX_BYTES`` under ``project`` (noise dirs skipped) in which
    ``scanner`` (default the bound ``policy`` scanner) finds something.
    Empty without any scanner. Never raises for an unreadable file or a
    scanner that raises (logged, file kept)."""
    root = Path(project).resolve()
    active = scanner if scanner is not None else policy.scanner()
    if active is None:
        return {}
    out: dict[str, list[str]] = {}
    for path in files.walk_files(root):
        try:
            if path.stat().st_size > SCAN_MAX_BYTES:
                continue
            if not _is_text_head(path):
                continue
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        try:
            findings = active.scan(text)
        except Exception:                        # noqa: BLE001
            log.exc(f'sandbox: scanner failed on {path}')
            findings = []
        if findings:
            kinds = sorted({f.kind for f in findings})
            out[path.relative_to(root).as_posix()] = kinds
    return out


def copy_excludes(project: Path, scanner: Optional[ContentScanner] = None
                  ) -> list[str]:
    """What never enters the working copy: every noise directory name,
    ``.env`` and ``.env*``, and the relative path of each file
    :func:`flagged_files` reports with ``scanner`` (logged, so the exclusion
    is visible). Names (no ``/``) match any path component by glob;
    flagged paths match exactly."""
    flagged = flagged_files(project, scanner)
    for rel, kinds in sorted(flagged.items()):
        log.warning('sandbox: excluding %s from the copy (secret scan: %s)',
                    rel, ', '.join(kinds))
    return [*sorted(files.NOISE_DIRS), *BASE_EXCLUDES, *sorted(flagged)]


def check_argv(argv: object) -> str:
    """``''`` when ``argv`` is a non-empty list of strings whose ``argv[0]``
    basename is in ``RUNNERS``; else a ``DENIED_PREFIX`` text."""
    if (not isinstance(argv, list) or not argv
            or not all(isinstance(a, str) for a in argv)):
        return (f'{DENIED_PREFIX} sandbox argv must be a non-empty list '
                'of strings')
    head = Path(argv[0]).name
    if head not in RUNNERS:
        return (f"{DENIED_PREFIX} '{argv[0]}' is not a sandbox runner; "
                'argv[0] must be one of ' + ', '.join(sorted(RUNNERS)))
    return ''


def safe_name(name: str) -> str:
    """A Docker repository component for a project directory name:
    lowercase, ``[a-z0-9._-]`` only, runs of other characters collapsed to
    ``-``, no leading/trailing separators; ``project`` when nothing is
    left."""
    cleaned = _NAME_RX.sub('-', name.lower()).strip('.-_')
    return cleaned or 'project'


def project_key(root: Path) -> str:
    """The per-project state key ``<safe basename>-<sha256(resolved
    path)[:8]>``: two checkouts with the same directory name never share
    an image tag, record directory or working copies."""
    resolved = Path(root).expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode('utf-8')).hexdigest()[:8]
    return f'{safe_name(resolved.name)}-{digest}'


def spec_from(project: Path, settings: object) -> SandboxSpec:
    """A :class:`SandboxSpec` for ``project`` under ``settings`` (a
    ``SandboxSettings``: ``base_image``, ``cpus``, ``memory_mb``, ``pids``,
    ``timeout_s``). ``name`` is :func:`project_key`; the image tag is
    ``guru-sandbox/<name>:<lockfile sha[:12]>`` so a new lockfile is a new
    tag. Raises ``ValueError`` when the project has no ``uv.lock``."""
    root = Path(project).expanduser().resolve()
    sha = lockfile_sha(root)
    if not sha:
        raise ValueError(f'{root} has no uv.lock; the sandbox needs a uv '
                         'lockfile')
    name = project_key(root)
    return SandboxSpec(
        project=root, name=name,
        base_image=str(getattr(settings, 'base_image')),
        image_tag=f'guru-sandbox/{name}:{sha[:12]}',
        lockfile_sha=sha,
        cpus=float(getattr(settings, 'cpus')),
        memory_mb=int(getattr(settings, 'memory_mb')),
        pids=int(getattr(settings, 'pids')),
        timeout_s=int(getattr(settings, 'timeout_s')))


def spec_summary(spec: Optional[SandboxSpec]) -> str:
    """One line naming the spec for status output."""
    if spec is None:
        return '(none)'
    return (f'{spec.name}: {spec.image_tag} (lockfile {spec.lockfile_sha[:12]}'
            f', {spec.cpus} cpus, {spec.memory_mb} MB, {spec.pids} pids, '
            f'{spec.timeout_s}s)')
