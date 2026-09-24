"""Provisioning (design plan §2 steps 1 and 5, chunk S2): build the sandbox
image on the internal network behind the filtering proxy, and turn an
approved dependency request into a lockfile change plus a rebuild.

:func:`provision` is the only path that gives a build network access: it
first makes sure ``pypi.org`` and ``files.pythonhosted.org`` are approved
(``tools.ensure_domain_allowed`` asks per access mode), brings up the
internal network and the proxy with the allow-list generated from
``config.ALLOWED_DOMAINS``, runs ``colima.build`` with the proxy build
args, then tears both down in ``finally`` and writes the proxy's access
log to ``net_events``. It is skipped when the recorded image is current
(``images.needs_build``).

:func:`request_dependency` records a validated request and installs
nothing. :func:`apply_dependency` asks (ask mode; auto approves silently;
read-only refuses) through the pluggable :func:`set_approve_asker` seam,
runs ``uv add`` in a provisioning container on a *copy* of the project,
brings the resulting ``pyproject.toml``/``uv.lock`` back through
``patch.apply_patch`` (the write gate, all-or-nothing) with the lockfile
diff summary, and rebuilds. Nothing in the sandbox ever writes the real
tree directly.
"""
from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

from guru import config, log
from guru.domain import deps, patch, tools
from guru.domain import sandbox as sb
from guru.domain.deps import DependencyRequest
from guru.repositories import sandbox_images as images
from guru.repositories.sandbox_images import ImageRecord
from guru.repositories.settings import SandboxSettings, load_sandbox
from guru.sandbox import colima, proxy

# Hosts a uv/pip build needs; both are approved through the normal domain
# question (one list, no implicit additions).
REQUIRED_DOMAINS = ('pypi.org', 'files.pythonhosted.org')
PROXY_DIR = 'proxy'                    # config under ~/.guru/sandbox/<name>/
UV_CACHE = '/tmp/uv-cache'             # inside the container's tmpfs
LOCK_FILES = ('pyproject.toml', 'uv.lock')

_asker: Optional[Callable[[str], bool]] = None


class ProvisionError(RuntimeError):
    """Provisioning could not complete (domains denied, network, build)."""


# --- approval seam -----------------------------------------------------------

def set_approve_asker(fn: Optional[Callable[[str], bool]]) -> None:
    """Install the dependency-approval prompt ``fn(question) -> bool`` (the
    TUI installs its own; None restores the console default)."""
    global _asker
    _asker = fn


def _ask_console(question: str) -> bool:
    """Console prompt; default and any error mean no."""
    try:
        answer = input(f'{question}\n[y/N] ').strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return answer in ('y', 'yes')


def approve(question: str) -> bool:
    """Mode-aware approval: read-only never; auto (with
    ``config.AUTO_GRANT``) without asking; otherwise the installed asker
    (an asker that raises is a decline)."""
    if config.MODE == config.MODE_READ_ONLY:
        return False
    if config.MODE == config.MODE_AUTO and config.AUTO_GRANT:
        return True
    asker = _asker or _ask_console
    try:
        return bool(asker(question))
    except Exception:                            # noqa: BLE001
        log.exc('dependency approval asker failed; treating as a decline')
        return False


# --- naming ------------------------------------------------------------------

def network_name(spec: sb.SandboxSpec) -> str:
    """The internal network for ``spec``'s provisioning."""
    return f'guru-provision-{spec.name}'


def proxy_name(spec: sb.SandboxSpec) -> str:
    """The proxy container for ``spec`` (stable, so the build cache holds
    across provisions)."""
    return f'guru-proxy-{spec.name}'


def proxy_url(spec: sb.SandboxSpec) -> str:
    """The proxy URL as seen from the internal network."""
    return f'http://{proxy_name(spec)}:{proxy.PROXY_PORT}'


def _settings(settings: Optional[SandboxSettings]) -> SandboxSettings:
    return settings if settings is not None else load_sandbox()


def _ensure_domains() -> None:
    for domain in REQUIRED_DOMAINS:
        if not tools.ensure_domain_allowed(domain):
            raise ProvisionError(f"web access to '{domain}' was denied; the "
                                 'sandbox image cannot be provisioned '
                                 'without it')


@contextmanager
def _proxied(spec: sb.SandboxSpec, settings: SandboxSettings, phase: str
             ) -> Iterator[str]:
    """Bring up the internal network and the proxy (allow-list from
    ``config.ALLOWED_DOMAINS``), yield the network name, then record the
    access log as ``net_events`` and tear everything down."""
    net, name = network_name(spec), proxy_name(spec)
    proxy.network_up(net, cwd=spec.project)
    try:
        subnet = proxy.network_subnet(net, cwd=spec.project)
        allowlist = proxy.allowlist_from_domains(config.ALLOWED_DOMAINS)
        cfg = images.record_dir(spec) / PROXY_DIR
        proxy.write_config(cfg, allowlist, client_cidr=subnet)
        proxy.proxy_up(name, net, cfg, settings.proxy_image,
                       cwd=spec.project)
        try:
            yield net
        finally:
            events = proxy.tail_access_log(name, cwd=spec.project)
            images.record_net_events(events, phase)
            proxy.proxy_down(name, cwd=spec.project)
    finally:
        proxy.network_down(net, cwd=spec.project)


def _fresh_copy(spec: sb.SandboxSpec, prefix: str) -> Path:
    dest = images.work_root(spec) / f'{prefix}-{uuid.uuid4().hex[:8]}'
    return colima.prepare_copy(spec.project, dest,
                               sb.copy_excludes(spec.project))


def _discard(copy: Optional[Path]) -> None:
    if copy is None:
        return
    try:
        colima.remove_copy(copy)
    except (ValueError, OSError):
        log.exc(f'sandbox: could not remove working copy {copy}')
        shutil.rmtree(copy, ignore_errors=True)


# --- provisioning ------------------------------------------------------------

def provision(project: Path, settings: Optional[SandboxSettings] = None,
              force: bool = False) -> ImageRecord:
    """Build (or confirm) the sandbox image for ``project``.

    Returns the current :class:`ImageRecord` without touching docker when
    the record matches the lockfile and Dockerfile (``force`` rebuilds).
    Otherwise: approve ``REQUIRED_DOMAINS``, copy the project (excludes
    applied) as the build context, build on the internal network through
    the proxy, tear down, and return the new record. ``ProvisionError``
    when a domain is denied or the network/proxy/build fails; the
    context copy and the network are removed either way.
    """
    cfg = _settings(settings)
    spec = sb.spec_from(project, cfg)
    text = sb.dockerfile_for(spec.project, spec.base_image)
    if not force and not images.needs_build(spec, text):
        rec = images.load_record(spec)
        if rec is not None:
            return rec
    _ensure_domains()
    dockerfile = images.write_dockerfile(spec, text)
    copy: Optional[Path] = None
    try:
        copy = _fresh_copy(spec, 'build')
        with _proxied(spec, cfg, 'build') as net:
            res = colima.build(spec, dockerfile, copy, network=net,
                               build_args=sb.proxy_build_args(
                                   proxy_url(spec)))
    except proxy.ProxyError as e:
        raise ProvisionError(str(e)) from e
    finally:
        _discard(copy)
    if not res.ok:
        images.record_sandbox_event('provision', res.argv, res.seconds,
                                    False, res.denied or 'build failed')
        raise ProvisionError('docker build failed: '
                             + (res.denied or res.stderr.strip()[-1500:]
                                or res.stdout.strip()[-500:]))
    rec = images.load_record(spec)
    if rec is None:                              # pragma: no cover
        raise ProvisionError('build succeeded but no image record was '
                             'written')
    images.record_sandbox_event('provision', res.argv, res.seconds, True,
                                rec.digest)
    return rec


# --- dependency requests -----------------------------------------------------

def request_dependency(name: str, constraint: str = '',
                       project: Optional[Path] = None,
                       settings: Optional[SandboxSettings] = None) -> str:
    """Record a dependency request for ``project`` (default the current
    project) and install nothing. Returns a one-paragraph digest, or a
    ``Refused: …`` text for an invalid name/constraint."""
    root = Path(project) if project is not None \
        else config.PROJECT_GURU_DIR.parent
    try:
        req = deps.request_from(str(name), str(constraint or ''))
    except ValueError as e:
        images.record_sandbox_event('dep_request', ['request', str(name),
                                                    str(constraint)], 0.0,
                                    False, str(e))
        return f'Refused: {e}'
    try:
        spec = sb.spec_from(root, _settings(settings))
    except ValueError as e:
        return f'Refused: {e}'
    pending = images.add_request(spec, req)
    images.record_sandbox_event('dep_request', ['request', req.spec], 0.0,
                                True, f'{len(pending)} pending')
    return (f'Recorded dependency request {req.spec}; nothing was '
            'installed. The user approves it with `/sandbox deps apply '
            f'{req.name}`, which updates uv.lock and rebuilds the sandbox '
            f'image. Pending: {len(pending)}.')


def _read_lock_files(root: Path) -> dict[str, str]:
    return {name: (root / name).read_text(encoding='utf-8')
            for name in LOCK_FILES}


def apply_dependency(project: Path, request: DependencyRequest,
                     settings: Optional[SandboxSettings] = None) -> str:
    """Apply ``request`` to ``project`` after approval: ``uv add`` in a
    provisioning container on a copy, the ``pyproject.toml``/``uv.lock``
    diff back through ``apply_patch``, then :func:`provision` rebuilds.
    Returns a digest (lockfile diff summary, new image digest) or one of
    ``Refused: …`` / ``Declined: …`` / ``uv add failed: …`` / the patch
    refusal. The request is removed from the pending store once applied."""
    cfg = _settings(settings)
    try:
        spec = sb.spec_from(project, cfg)
    except ValueError as e:
        return f'Refused: {e}'
    if images.load_record(spec) is None:
        return ('Refused: the sandbox image is not built; run /sandbox '
                'provision first.')
    if config.MODE == config.MODE_READ_ONLY:
        return 'Refused: read-only mode. Change mode to add dependencies.'
    question = (f'Add dependency {request.spec} to uv.lock and rebuild the '
                'sandbox image?')
    if not approve(question):
        images.record_sandbox_event('dep_apply', ['uv', 'add', request.spec],
                                    0.0, False, 'declined')
        return f'Declined: {request.spec} was not added.'
    try:
        _ensure_domains()
    except ProvisionError as e:
        return f'Refused: {e}'
    before = _read_lock_files(spec.project)
    copy: Optional[Path] = None
    try:
        copy = _fresh_copy(spec, 'deps')
        env = {**{k: proxy_url(spec) for k in
                  ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy')},
               'NO_PROXY': '', 'UV_OFFLINE': '0', 'UV_CACHE_DIR': UV_CACHE}
        try:
            with _proxied(spec, cfg, 'uv add') as net:
                res = colima.run(spec, ['uv', 'add', '--no-sync',
                                        request.spec], copy, network=net,
                                 env=env)
        except proxy.ProxyError as e:
            return f'uv add failed: {e}'
        if res.returncode != 0 or res.denied:
            return ('uv add failed: '
                    + (res.denied or res.stderr.strip()[-1500:]
                       or f'exit {res.returncode}'))
        after = _read_lock_files(copy)
    finally:
        _discard(copy)
    diff = ''.join(deps.unified_diff(spec.project / name, before[name],
                                     after[name]) for name in LOCK_FILES)
    if not diff:
        images.remove_request(spec, request.name)
        return (f'{request.spec} changed nothing: pyproject.toml and uv.lock '
                'are unchanged (already present?). Request cleared.')
    summary = deps.summary_text(deps.lock_diff_summary(before['uv.lock'],
                                                       after['uv.lock']))
    applied = patch.apply_patch(diff)
    if not applied.startswith('Applied patch'):
        images.record_sandbox_event('dep_apply', ['uv', 'add', request.spec],
                                    res.seconds, False, applied[:200])
        return f'{applied}\nLockfile change not applied ({summary}).'
    images.remove_request(spec, request.name)
    rec = provision(spec.project, cfg)
    images.record_sandbox_event('dep_apply', ['uv', 'add', request.spec],
                                res.seconds, True, f'{summary}; {rec.digest}')
    return (f'Added {request.spec}. uv.lock: {summary}. Sandbox image '
            f'rebuilt: {rec.tag} digest {rec.digest}.')
