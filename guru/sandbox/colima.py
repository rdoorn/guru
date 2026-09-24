"""The Colima/Docker sandbox runtime (design plan §1 endpoint, chunk S1).

Every docker invocation is a fixed argv guru assembles here and starts
through ``guru.domain.procs.run`` (scrubbed environment, wall-clock kill).
Model text never enters a docker argv except as the trailing *container*
command of :func:`run`, whose ``argv[0]`` must be in
``guru.domain.sandbox.RUNNERS``. The container itself runs with no network,
as ``1000:1000``, all capabilities dropped, ``no-new-privileges``, a
read-only root, a tmpfs ``/tmp``, pid/memory/cpu limits and the working
copy bind-mounted at ``/work`` — the only writable path, and a copy: the
real tree is never mounted. A run that outlives its timeout is
``docker kill``\\ ed by the ``--name`` guru generated, because killing the
docker CLI's process group does not stop the container the daemon owns.

Working copies (:func:`prepare_copy`) are ``git init``\\ ed and committed so
:func:`diff` can return the sandbox's changes as a unified diff; they live
under ``~/.guru/sandbox/<name>/work/``, a path Colima mounts into its VM.
When the project is a git repository the copy is the *positive* list
``git ls-files --cached --others --exclude-standard`` — tracked and
untracked-but-not-ignored files only, so a gitignored ``secrets.yaml`` or
``.env`` never enters the copy or the image; only a non-repository falls
back to a filtered ``copytree``. On top of that :func:`copy_excludes_for`
applies :func:`guru.domain.sandbox.copy_excludes` with the project secret
scanner bound (``policy`` scanner if routing bound one, else
``guru.scanners.secrets.load_project_scanner``), so copy hygiene never
depends on routing.
"""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from guru import log
from guru.domain import policy, procs
from guru.domain.policy import ContentScanner
from guru.domain.procs import ProcResult
from guru.domain.sandbox import (WORKDIR, SandboxSpec, check_argv,
                                 copy_excludes)
from guru.repositories import sandbox_images as images
from guru.scanners.secrets import load_project_scanner

DOCKER = 'docker'
INFO_TIMEOUT_S = 10            # docker info
BUILD_TIMEOUT_S = 1800         # docker build (uv sync of a big lockfile)
KILL_TIMEOUT_S = 15            # docker kill after a run timeout
GIT_TIMEOUT_S = 120            # git init/add/commit/diff on the copy
STOP_TIMEOUT_S = 5             # --stop-timeout: SIGKILL this long after stop
BUILD_OUT_KB = 64
RUN_OUT_KB = 256
DIFF_OUT_KB = 2048
# Network modes BuildKit accepts; any other name is a user-defined network,
# which only the classic builder (DOCKER_BUILDKIT=0) can attach to.
BUILDKIT_NETWORKS = frozenset(('none', 'default', 'host'))
# Marker file prepare_copy writes; remove_copy refuses a tree without it.
COPY_MARKER = '.guru-sandbox-copy'
# Git identity for the copy's commits (the scrubbed HOME has no gitconfig).
_GIT_IDENTITY = ('-c', 'user.name=guru-sandbox',
                 '-c', 'user.email=sandbox@guru.local',
                 '-c', 'commit.gpgsign=false',
                 '-c', 'init.defaultBranch=main')
# Ignored inside the copy so caches the container writes stay out of diffs.
_GIT_EXCLUDE = ('__pycache__/\n.pytest_cache/\n.mypy_cache/\n.ruff_cache/\n'
                '*.pyc\n.venv/\n' + COPY_MARKER + '\n')
# Environment the docker CLI needs to find the Colima context: procs.run
# gives the child a temporary HOME, which would hide ~/.docker.
_DOCKER_PASSTHROUGH = ('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY',
                       'DOCKER_CERT_PATH')

_ARG_NAME_RX = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

_available: Optional[bool] = None


@dataclass
class BuildResult:
    """Outcome of :func:`build`: ``digest`` is the image id on success."""
    ok: bool
    tag: str
    digest: str
    seconds: float
    stdout: str
    stderr: str
    argv: list[str] = field(default_factory=list)
    denied: str = ''


@dataclass
class RunResult:
    """Outcome of :func:`run`, shaped like ``ProcResult`` plus the container
    name, the full docker argv and whether guru had to ``docker kill`` it.
    ``argv`` is the container command (what the model asked for)."""
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False
    truncated: bool = False
    denied: str = ''
    name: str = ''
    docker_argv: list[str] = field(default_factory=list)
    killed: bool = False


# --- docker plumbing ---------------------------------------------------------

def docker_env() -> dict[str, str]:
    """``env_extra`` for the docker CLI: ``DOCKER_CONFIG`` (from the
    environment or ``~/.docker``) so the selected context survives the
    scrubbed HOME, plus any ``DOCKER_HOST``/``DOCKER_CONTEXT``/TLS keys the
    user has set."""
    env = {'DOCKER_CONFIG': os.environ.get('DOCKER_CONFIG')
           or str(Path.home() / '.docker')}
    for key in _DOCKER_PASSTHROUGH:
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def cli_limits(timeout_s: int, out_kb: int) -> procs.Limits:
    """Limits for the docker CLI itself (a Go binary: generous address
    space, CPU at least the wall clock)."""
    return procs.Limits(timeout_s=int(timeout_s),
                        cpu_s=max(600, int(timeout_s)), mem_mb=8192,
                        fsize_mb=256, out_kb=int(out_kb))


def _docker(args: list[str], cwd: Path, timeout_s: int,
            out_kb: int = RUN_OUT_KB,
            env: Optional[dict[str, str]] = None) -> ProcResult:
    return procs.run([DOCKER, *args], cwd, cli_limits(timeout_s, out_kb),
                     env_extra={**docker_env(), **(env or {})})


def available(cwd: Optional[Path] = None, refresh: bool = False) -> bool:
    """Whether ``docker info`` succeeds (cached per process; ``refresh``
    re-probes). ``cwd`` is where the CLI runs (default the current
    directory; it must be under the read allow-list like any child)."""
    global _available
    if _available is None or refresh:
        res = _docker(['info', '--format', '{{.ServerVersion}}'],
                      Path(cwd) if cwd is not None else Path.cwd(),
                      INFO_TIMEOUT_S, out_kb=4)
        _available = res.returncode == 0 and not res.denied
        if not _available:
            log.info('sandbox: docker unavailable: %s',
                     res.denied or res.stderr.strip()[:200])
    return bool(_available)


def _image_id(tag: str, cwd: Path) -> str:
    res = _docker(['image', 'inspect', '--format', '{{.Id}}', tag], cwd,
                  INFO_TIMEOUT_S, out_kb=4)
    return res.stdout.strip() if res.returncode == 0 else ''


def build_args_argv(build_args: Optional[dict[str, str]]) -> list[str]:
    """``--build-arg K=V`` pairs (keys sorted) for ``build_args``; a key
    that is not an identifier raises ``ValueError`` (the values are
    guru's own proxy URL, but the argv stays a fixed shape)."""
    out: list[str] = []
    args = build_args or {}
    for key in sorted(args):
        if not isinstance(key, str) or not _ARG_NAME_RX.fullmatch(key):
            raise ValueError(f'build arg name {key!r} is not an identifier')
        value = str(args[key])
        if any(ch in value for ch in '\r\n'):
            raise ValueError(f'build arg {key} value contains a newline')
        out += ['--build-arg', f'{key}={value}']
    return out


def build(spec: SandboxSpec, dockerfile: Path, context_dir: Path,
          network: str = 'none',
          build_args: Optional[dict[str, str]] = None) -> BuildResult:
    """``docker build --network <network> [--build-arg K=V…] -t <tag> -f
    <dockerfile> <context_dir>``; on success the image id is looked up and
    the build is recorded (image record + ``sandbox_events``). ``network``
    is ``none`` unless the caller provides the provisioning network (S2);
    a user-defined network is only supported by the classic builder, so
    the CLI then runs with ``DOCKER_BUILDKIT=0`` (BuildKit rejects it
    outright). ``build_args`` are the proxy variables of
    :func:`guru.domain.sandbox.proxy_build_args`. The CLI runs in
    ``spec.project``."""
    extra = build_args_argv(build_args)
    argv = ['build', '--network', str(network), *extra, '-t', spec.image_tag,
            '-f', str(dockerfile), str(context_dir)]
    env = {} if network in BUILDKIT_NETWORKS else {'DOCKER_BUILDKIT': '0'}
    res = _docker(argv, spec.project, BUILD_TIMEOUT_S, out_kb=BUILD_OUT_KB,
                  env=env)
    ok = res.returncode == 0 and not res.denied
    digest = _image_id(spec.image_tag, spec.project) if ok else ''
    if ok:
        images.record_built(spec, dockerfile.read_text(encoding='utf-8'),
                            digest)
    images.record_sandbox_event(
        'build', res.argv, res.seconds, ok,
        res.denied or (digest if ok else f'exit {res.returncode}'
                       + (' (timed out)' if res.timed_out else '')))
    return BuildResult(ok=ok, tag=spec.image_tag, digest=digest,
                       seconds=res.seconds, stdout=res.stdout,
                       stderr=res.stderr, argv=res.argv, denied=res.denied)


def container_name() -> str:
    """A fresh ``guru-sb-<12 hex>`` name so a timed-out run can be
    ``docker kill``\\ ed."""
    return f'guru-sb-{uuid.uuid4().hex[:12]}'


def docker_run_argv(spec: SandboxSpec, argv: list[str], copy: Path,
                    name: str, network: str = 'none',
                    env: Optional[dict[str, str]] = None) -> list[str]:
    """The exact ``docker run`` argv for a sandbox run (pure). ``network``
    is ``none`` for the execution phase; provisioning passes the internal
    network and ``env`` (``-e K=V`` pairs: the proxy variables). Keys must
    be identifiers (``ValueError``)."""
    extra: list[str] = []
    variables = env or {}
    for key in variables:
        if not isinstance(key, str) or not _ARG_NAME_RX.fullmatch(key):
            raise ValueError(f'env name {key!r} is not an identifier')
        extra += ['-e', f'{key}={variables[key]}']
    return [DOCKER, 'run', '--rm', '--name', name,
            '--stop-timeout', str(STOP_TIMEOUT_S),
            '--network', str(network), '--user', '1000:1000',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--read-only', '--tmpfs', '/tmp',
            '--pids-limit', str(int(spec.pids)),
            '--memory', f'{int(spec.memory_mb)}m',
            '--cpus', str(spec.cpus),
            '-v', f'{copy}:{WORKDIR}', '-w', WORKDIR, *extra,
            spec.image_tag, *argv]


def run(spec: SandboxSpec, argv: list[str], workdir_copy: Path,
        network: str = 'none',
        env: Optional[dict[str, str]] = None) -> RunResult:
    """Run ``argv`` inside ``spec``'s image with ``workdir_copy`` mounted at
    ``/work`` (see the module docstring for the container flags).

    ``argv[0]`` must be a ``RUNNERS`` basename, else the run is denied
    (``returncode -1``, ``denied`` set) and nothing starts. ``network`` and
    ``env`` default to no network and no variables (the execution phase);
    provisioning (``uv add``) passes the internal network and the proxy
    variables. The wall clock is ``spec.timeout_s``; on expiry the CLI's
    process group is killed and the container is ``docker kill``\\ ed by
    name (``killed``). Every run lands in ``sandbox_events``. Never raises
    for a failing container.
    """
    denial = check_argv(argv)
    if denial:
        images.record_sandbox_event('run', argv, 0.0, False, denial)
        return RunResult(list(argv) if isinstance(argv, list) else [],
                         -1, '', '', 0.0, denied=denial)
    name = container_name()
    full = docker_run_argv(spec, argv, Path(workdir_copy), name,
                           network=network, env=env)
    res = procs.run(full, spec.project,
                    cli_limits(spec.timeout_s, RUN_OUT_KB),
                    env_extra=docker_env())
    killed = False
    if res.timed_out:
        kill = _docker(['kill', name], spec.project, KILL_TIMEOUT_S,
                       out_kb=4)
        killed = kill.returncode == 0
        log.warning('sandbox: run %s timed out after %ss; docker kill %s',
                    name, spec.timeout_s, 'ok' if killed else 'failed')
    ok = res.returncode == 0 and not res.denied
    images.record_sandbox_event(
        'run', argv, res.seconds, ok,
        res.denied or (f'exit {res.returncode}'
                       + (' (timed out, killed)' if res.timed_out else '')))
    return RunResult(list(argv), res.returncode, res.stdout, res.stderr,
                     res.seconds, timed_out=res.timed_out,
                     truncated=res.truncated, denied=res.denied, name=name,
                     docker_argv=full, killed=killed)


# --- working copies ----------------------------------------------------------

def project_scanner() -> ContentScanner:
    """The scanner copy hygiene uses: the bound ``policy`` scanner when
    routing installed one, else the project secret scanner
    (``.guru/sensitive_markers.txt`` + ``scan_allow.txt``)."""
    return policy.scanner() or load_project_scanner()


def copy_excludes_for(project: Path) -> list[str]:
    """:func:`guru.domain.sandbox.copy_excludes` with :func:`project_scanner`
    bound — the excludes every working copy is made with."""
    return copy_excludes(project, scanner=project_scanner())


def _split_excludes(excludes: list[str]) -> tuple[list[str], set[str]]:
    """``(names, paths)``: an exclude without ``/`` is a glob matched
    against any path component; one with ``/`` is an exact relative path
    (a flagged file — never a pattern, so ``secrets[1].py`` excludes that
    file and no other)."""
    names = [e for e in excludes if '/' not in e]
    paths = {e for e in excludes if '/' in e}
    return names, paths


def _excluded(rel: Path, names: list[str], paths: set[str]) -> bool:
    """Whether the project-relative ``rel`` is excluded: any component
    globs a name, or the whole path equals a flagged path."""
    return rel.as_posix() in paths or any(
        fnmatch.fnmatch(part, pat) for part in rel.parts for pat in names)


def _ignorer(root: Path, excludes: list[str]
             ) -> Callable[[str, list[str]], set[str]]:
    """A ``shutil.copytree`` ``ignore`` callable over :func:`_excluded`."""
    names, paths = _split_excludes(excludes)

    def ignore(directory: str, entries: list[str]) -> set[str]:
        here = Path(directory).resolve()
        rel_dir = here.relative_to(root) if here != root else Path()
        return {entry for entry in entries
                if _excluded(rel_dir / entry, names, paths)}
    return ignore


def _git_files(src: Path) -> Optional[list[str]]:
    """The repository's file list (tracked plus untracked-not-ignored,
    from ``git ls-files -z``) when ``src`` has a ``.git``; None when it is
    not a repository. ``RuntimeError`` when git fails or the list is
    truncated — a repository is never copied wholesale."""
    if not (src / '.git').exists():
        return None
    res = _git(['ls-files', '-z', '--cached', '--others',
                '--exclude-standard'], src, src, out_kb=DIFF_OUT_KB)
    if res.returncode != 0:
        raise RuntimeError(f'git ls-files failed in {src}: '
                           f'{res.denied or res.stderr.strip()}')
    if res.truncated:
        raise RuntimeError(f'git ls-files output for {src} exceeds '
                           f'{DIFF_OUT_KB} KB; refusing to copy a partial '
                           'tree')
    return [name for name in res.stdout.split('\0') if name]


def _copy_listed(src: Path, dst: Path, names: list[str],
                 excludes: list[str]) -> int:
    """Copy the listed relative paths from ``src`` to ``dst`` (symlinks as
    links, directories/gitlinks and vanished files skipped) minus
    ``excludes``; returns the number of files copied."""
    globs, paths = _split_excludes(excludes)
    copied = 0
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            continue
        if _excluded(rel, globs, paths):
            continue
        source = src / rel
        if not source.is_symlink() and not source.is_file():
            continue                         # deleted, or a submodule dir
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=False)
        copied += 1
    return copied


def _git(args: list[str], repo: Path, cwd: Path,
         out_kb: int = RUN_OUT_KB) -> ProcResult:
    """``git -C <repo> <args>`` through the runner, from ``cwd`` (an
    allow-listed directory — the copy itself is not)."""
    return procs.run(['git', *_GIT_IDENTITY, '-C', str(repo), *args], cwd,
                     cli_limits(GIT_TIMEOUT_S, out_kb))


def prepare_copy(project: Path, dest: Path, excludes: list[str]) -> Path:
    """Copy ``project`` to ``dest``, mark it, and ``git init`` + commit
    everything so :func:`diff` has a baseline.

    A git repository is copied from its ``git ls-files`` positive list
    (tracked + untracked-not-ignored; gitignored files never enter); a
    plain directory is ``copytree``\\ d. Either way ``excludes`` (as
    :func:`copy_excludes_for` shapes them) are applied and symlinks are
    kept as links. ``dest`` must not exist. Raises
    ``OSError``/``RuntimeError`` when the copy or git fails."""
    src = Path(project).expanduser().resolve()
    dst = Path(dest)
    if dst.exists():
        raise FileExistsError(f'{dst} already exists')
    dst.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    listed = _git_files(src)
    if listed is None:
        shutil.copytree(src, dst, symlinks=True,
                        ignore=_ignorer(src, excludes))
        how = 'copytree'
    else:
        dst.mkdir()
        n = _copy_listed(src, dst, listed, excludes)
        how = f'git ls-files ({n} files)'
    (dst / COPY_MARKER).write_text('working copy made by guru; safe to '
                                   'delete\n', encoding='utf-8')
    for args in (['init', '-q'], ['add', '-A'],
                 ['commit', '-q', '--allow-empty', '-m', 'sandbox copy']):
        if args[0] == 'add':
            exclude = dst / '.git' / 'info' / 'exclude'
            exclude.parent.mkdir(parents=True, exist_ok=True)
            exclude.write_text(_GIT_EXCLUDE, encoding='utf-8')
        res = _git(args, dst, src)
        if res.returncode != 0:
            shutil.rmtree(dst, ignore_errors=True)
            raise RuntimeError(f'git {args[0]} failed in {dst}: '
                               f'{res.denied or res.stderr.strip()}')
    images.record_sandbox_event('copy', ['copy', str(src), str(dst)],
                                time.monotonic() - started, True,
                                f'{how}; {len(excludes)} excludes')
    return dst


def diff(copy: Path, project: Optional[Path] = None) -> str:
    """The unified diff of ``copy``'s working tree against its baseline
    commit (``git diff HEAD``): edits, deletions and — via ``git add -N``
    — new files; caches excluded; ``''`` when unchanged. The git CLI
    runs from ``project`` (default the current directory)."""
    repo = Path(copy)
    cwd = Path(project) if project is not None else Path.cwd()
    add = _git(['add', '-A', '-N'], repo, cwd)
    if add.returncode != 0:
        raise RuntimeError(f'git add -N failed in {repo}: '
                           f'{add.denied or add.stderr.strip()}')
    res = _git(['diff', 'HEAD', '--no-color', '--no-ext-diff'], repo,
               cwd, out_kb=DIFF_OUT_KB)
    if res.returncode != 0:
        raise RuntimeError(f'git diff failed in {repo}: '
                           f'{res.denied or res.stderr.strip()}')
    return res.stdout + ('\n[diff truncated]\n' if res.truncated else '')


def remove_copy(copy: Path) -> None:
    """Delete a working copy :func:`prepare_copy` made. Refuses
    (``ValueError``) a directory without the ``COPY_MARKER`` file so a
    mistaken path can never delete a real tree."""
    target = Path(copy)
    if not (target / COPY_MARKER).is_file():
        raise ValueError(f'{target} is not a guru sandbox copy '
                         f'(no {COPY_MARKER}); refusing to delete')
    shutil.rmtree(target, ignore_errors=True)


def reset_cache() -> None:
    """Forget the cached :func:`available` probe (tests)."""
    global _available
    _available = None
