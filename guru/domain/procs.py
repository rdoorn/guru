"""Fixed-argv subprocess runner for the audited tools (design plan A1).

Every child guru starts goes through :func:`run`: an argv *list* guru built
(never a shell string; ``shell=True`` is never used and shell binaries are
refused as ``argv[0]``), a working directory that must be under the read
allow-list, an environment built from scratch (so a secret exported in
guru's own environment never reaches the child), and resource limits (CPU
seconds, address space, largest file, wall-clock timeout) so a runaway test
suite cannot take the machine with it.

Mechanics:

* The child runs in its **own session** (``start_new_session=True``), and
  the whole process group is ``SIGKILL``\\ ed once the leader has exited or
  timed out — a grandchild the tool left behind (a forked test worker, a
  ``sleep``) dies with it.
* stdout/stderr go to **files inside the temporary HOME**, not pipes, so
  ``RLIMIT_FSIZE`` bounds how much a child can write and guru reads at most
  ``out_kb`` KB of each head (``truncated`` flags a longer file). Output is
  therefore capped twice: on disk by the kernel, in memory by the reader.
* Resource limits are applied by a tiny **exec shim** — a second
  ``python -c`` that sets the rlimits and ``execv``\\ s the real argv — rather
  than ``preexec_fn``. Trade-off: one extra interpreter start (~30 ms) per
  call, and ``argv[0]`` is resolved on the *scrubbed* PATH by the shim; in
  return the parent never runs Python code between ``fork`` and ``exec``,
  which the stdlib documents as unsafe with threads (guru's turns run on
  worker threads).

Caveat: the child gets ``PYTHONPATH=<cwd>``, so a ``python -m <module>``
invocation resolves ``<module>`` in the project first — a project file named
``pytest.py`` / ``flake8.py`` / ``mypy.py`` shadows the real tool and runs
instead. That is project code the user already trusts to run its own tests,
and it runs under the same limits and scrubbed environment; callers that
need the tool itself must not rely on ``-m`` resolution being untouchable.

Stdlib only (``subprocess``, ``resource``, ``tempfile``); the allow-list
gate is ``guru.domain.files.ensure_path_allowed``.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from guru import config, log
from guru.domain import files

_DEFAULT_PATH = '/usr/bin:/bin:/usr/sbin:/sbin'
_DEFAULT_LANG = 'C.UTF-8'

# Prefix of every ``ProcResult.denied`` text (and of the tool result a verb
# builds from it), so ``tools.execute_tool``'s denial detection can key on
# one string.
DENIED_PREFIX = 'Denied:'
# argv[0] basenames refused outright — a fixed argv must never be a shell
# (design principle 1: no command strings, even indirectly).
SHELLS = frozenset(('sh', 'bash', 'zsh', 'dash', 'fish', 'ksh', 'csh',
                    'tcsh'))
# Environment keys ``env_extra`` may not override.
PROTECTED_ENV = frozenset(('PATH', 'HOME', 'PYTHONPATH',
                           'PYTHONDONTWRITEBYTECODE'))

# The exec shim (see the module docstring). argv: [limits-json, *argv].
# Each rlimit is set on its own and skipped when the OS refuses it
# (RLIMIT_AS is advisory on macOS). Exit 127 when argv[0] cannot be found
# on PATH, like a shell would.
_SHIM = r'''
import json, os, shutil, sys
limits = json.loads(sys.argv[1])
argv = sys.argv[2:]
try:
    import resource
    for name, value in (('RLIMIT_CPU', limits['cpu_s']),
                        ('RLIMIT_AS', limits['mem_mb'] * 1024 * 1024),
                        ('RLIMIT_FSIZE', limits['fsize_mb'] * 1024 * 1024)):
        kind = getattr(resource, name, None)
        if kind is None:
            continue
        try:
            resource.setrlimit(kind, (value, value))
        except (ValueError, OSError):
            pass
except ImportError:
    pass
exe = shutil.which(argv[0])
if exe is None:
    sys.stderr.write('guru procs: cannot find %r on PATH\n' % argv[0])
    sys.exit(127)
os.execv(exe, argv)
'''


@dataclass
class Limits:
    """Ceilings for one child process; defaults follow ``config.PROC_*``
    (settings.toml ``[tools.limits]``) at construction time."""
    timeout_s: int = field(default_factory=lambda: config.PROC_TIMEOUT_S)
    cpu_s: int = field(default_factory=lambda: config.PROC_CPU_S)
    mem_mb: int = field(default_factory=lambda: config.PROC_MEM_MB)
    fsize_mb: int = field(default_factory=lambda: config.PROC_FSIZE_MB)
    out_kb: int = field(default_factory=lambda: config.PROC_OUT_KB)

    @classmethod
    def from_dict(cls, overrides: Optional[dict]) -> 'Limits':
        """Config defaults with the known ``PROC_LIMIT_KEYS`` of
        ``overrides`` applied — the shape a project's ``.guru/tools.toml``
        ``[tools.limits]`` table has. Unknown keys are ignored; a value that
        is not a positive integer is logged and the default kept."""
        lim = cls()
        for key, value in (overrides or {}).items():
            if key not in config.PROC_LIMIT_KEYS:
                continue
            try:
                number = int(value)
                if isinstance(value, bool) or number <= 0:
                    raise ValueError(value)
            except (TypeError, ValueError):
                log.warning('procs: ignoring limit %s = %r; expected a '
                            'positive integer', key, value)
                continue
            setattr(lim, key, number)
        return lim

    def as_dict(self) -> dict[str, int]:
        """The rlimit-relevant values for the exec shim."""
        return {'cpu_s': int(self.cpu_s), 'mem_mb': int(self.mem_mb),
                'fsize_mb': int(self.fsize_mb)}


@dataclass
class ProcResult:
    """What one child did. ``returncode`` is -1 when the child never ran
    to completion: refused (``denied`` names why), timed out
    (``timed_out``) or could not be started (``stderr`` says why)."""
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False
    truncated: bool = False
    denied: str = ''


def _check_argv(argv: object) -> list[str]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, list):
        raise TypeError('argv must be a list of strings (never a shell '
                        'command string)')
    if not argv or not all(isinstance(a, str) for a in argv):
        raise TypeError('argv must be a non-empty list of strings')
    return list(argv)


def _environment(cwd: Path, home: str, env_extra: Optional[dict]
                 ) -> dict[str, str]:
    """The child's environment, built from scratch (design principle 3):
    nothing from ``os.environ`` leaks except ``PATH`` and ``LANG``, and
    ``env_extra`` may not override the ``PROTECTED_ENV`` keys."""
    env = {
        'PATH': os.environ.get('PATH') or _DEFAULT_PATH,
        'HOME': home,
        'LANG': os.environ.get('LANG') or _DEFAULT_LANG,
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONPATH': str(cwd),
    }
    for key, value in (env_extra or {}).items():
        if str(key) in PROTECTED_ENV:
            log.warning('procs: env_extra may not override %s; ignored', key)
            continue
        env[str(key)] = str(value)
    return env


def _read_head(path: Path, out_kb: int) -> tuple[str, bool]:
    """``(text, truncated)``: the first ``out_kb`` KB of ``path`` decoded
    leniently; ``truncated`` when the file is larger than that."""
    limit = max(1, int(out_kb)) * 1024
    try:
        size = path.stat().st_size
        with path.open('rb') as fh:
            data = fh.read(limit)
    except OSError:
        return '', False
    return data.decode('utf-8', errors='replace'), size > limit


def _resolve_cwd(cwd: Path) -> tuple[Optional[Path], str]:
    """``(resolved, denial)``: the real cwd, or why it is refused."""
    try:
        target = Path(cwd).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as e:
        return None, f'{DENIED_PREFIX} cwd {cwd} is not usable: {e}'
    if not target.is_dir():
        return None, f'{DENIED_PREFIX} cwd {target} is not a directory'
    if not files.ensure_path_allowed(target):
        return None, (f"{DENIED_PREFIX} cwd '{target}' is outside the "
                      'allowed directories')
    return target, ''


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's whole process group (it is the session leader)
    and reap the leader; silent when the group is already gone."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:            # pragma: no cover - kernel
        log.warning('procs: pid %d did not die after SIGKILL', proc.pid)


def run(argv: list[str], cwd: Path, limits: Optional[Limits] = None,
        env_extra: Optional[dict] = None) -> ProcResult:
    """Run ``argv`` in ``cwd`` under ``limits`` and return a ProcResult.

    ``argv`` must be a list of strings (a shell string raises TypeError;
    a shell binary as ``argv[0]`` is refused). ``cwd`` must resolve to a
    directory under the read allow-list (the usual mode-aware gate applies;
    a refusal yields ``returncode=-1`` with ``denied`` set — starting with
    ``DENIED_PREFIX`` — and no child started). The child runs in its own
    session with a fresh environment (``PATH``, a temporary ``HOME`` removed
    afterwards, ``LANG``, ``PYTHONDONTWRITEBYTECODE=1``,
    ``PYTHONPATH=<cwd>`` plus ``env_extra`` minus ``PROTECTED_ENV``), the
    rlimits of ``limits`` (via the exec shim) and a wall-clock timeout after
    which its whole process group is killed. stdout/stderr are captured to
    files under the temporary HOME, decoded with ``errors='replace'`` and
    read up to ``limits.out_kb`` each (``truncated`` flags a longer file).
    Never raises for a child that fails to start or times out.
    """
    argv = _check_argv(argv)
    limits = limits or Limits()
    if Path(argv[0]).name in SHELLS:
        return ProcResult(argv, -1, '', '', 0.0, denied=(
            f"{DENIED_PREFIX} '{argv[0]}' is a shell; tools run fixed "
            'argv lists only'))
    target, denial = _resolve_cwd(cwd)
    if target is None:
        return ProcResult(argv, -1, '', '', 0.0, denied=denial)
    home = tempfile.mkdtemp(prefix='guru-proc-')
    out_path, err_path = Path(home, 'stdout.log'), Path(home, 'stderr.log')
    shim_argv = [sys.executable, '-c', _SHIM, json.dumps(limits.as_dict()),
                 *argv]
    started = time.monotonic()
    timed_out = False
    stdout = stderr = ''
    truncated = False
    try:
        try:
            with out_path.open('wb') as out_fh, err_path.open('wb') as err_fh:
                proc = subprocess.Popen(
                    shim_argv, cwd=str(target), stdout=out_fh,
                    stderr=err_fh, stdin=subprocess.DEVNULL,
                    env=_environment(target, home, env_extra),
                    start_new_session=True, shell=False)
                try:
                    code = proc.wait(timeout=limits.timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    code = -1
                finally:
                    _kill_group(proc)      # strays die with the leader
        except (OSError, ValueError) as e:
            log.info('procs: cannot start %s: %s', argv[0], e)
            code = -1
            err_path.write_bytes(str(e).encode('utf-8'))
        stdout, cut_out = _read_head(out_path, limits.out_kb)
        stderr, cut_err = _read_head(err_path, limits.out_kb)
        truncated = cut_out or cut_err
    finally:
        shutil.rmtree(home, ignore_errors=True)
    seconds = time.monotonic() - started
    return ProcResult(argv, code, stdout, stderr, seconds,
                      timed_out=timed_out, truncated=truncated)
