"""Fixed-argv subprocess runner for the audited tools (design plan A1).

Every child guru starts goes through :func:`run`: an argv *list* guru built
(never a shell string; ``shell=True`` is never used), a working directory
that must be under the read allow-list, an environment built from scratch
(so a secret exported in guru's own environment never reaches the child),
and resource limits (CPU seconds, address space, largest file, wall-clock
timeout) so a runaway test suite cannot take the machine with it. Output is
capped, so a chatty child cannot flood the model's context either; the
caller's digest decides what the model actually sees.

Stdlib only (``subprocess``, ``resource``, ``tempfile``); the allow-list
gate is ``guru.domain.files.ensure_path_allowed``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from guru import config, log
from guru.domain import files

try:                                   # POSIX only; absent on Windows
    import resource
except ImportError:                    # pragma: no cover - platform
    resource = None                    # type: ignore[assignment]

_DEFAULT_PATH = '/usr/bin:/bin:/usr/sbin:/sbin'
_DEFAULT_LANG = 'C.UTF-8'


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
        ``overrides`` applied (unknown keys ignored) — the shape a project's
        ``.guru/tools.toml`` ``[tools.limits]`` table has."""
        lim = cls()
        for key, value in (overrides or {}).items():
            if key in config.PROC_LIMIT_KEYS:
                setattr(lim, key, int(value))
        return lim


@dataclass
class ProcResult:
    """What one child did. ``returncode`` is -1 when the child never ran
    to completion: refused (``denied`` names why), timed out
    (``timed_out``) or could not be started (``stderr`` says why)."""
    argv: list
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False
    truncated: bool = False
    denied: str = ''


def _check_argv(argv: object) -> list:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, list):
        raise TypeError('argv must be a list of strings (never a shell '
                        'command string)')
    if not argv or not all(isinstance(a, str) for a in argv):
        raise TypeError('argv must be a non-empty list of strings')
    return list(argv)


def _environment(cwd: Path, home: str, env_extra: Optional[dict]) -> dict:
    """The child's environment, built from scratch (design principle 3):
    nothing from ``os.environ`` leaks except ``PATH`` and ``LANG``."""
    env = {
        'PATH': os.environ.get('PATH') or _DEFAULT_PATH,
        'HOME': home,
        'LANG': os.environ.get('LANG') or _DEFAULT_LANG,
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONPATH': str(cwd),
    }
    for key, value in (env_extra or {}).items():
        env[str(key)] = str(value)
    return env


def _limiter(limits: Limits) -> Optional[Callable[[], None]]:
    """A ``preexec_fn`` applying the rlimits, or None where the platform
    has no ``resource`` module. Each limit is set on its own and skipped
    silently when the OS refuses it (RLIMIT_AS is advisory on macOS)."""
    if resource is None:
        return None
    wanted = [('RLIMIT_CPU', int(limits.cpu_s)),
              ('RLIMIT_AS', int(limits.mem_mb) * 1024 * 1024),
              ('RLIMIT_FSIZE', int(limits.fsize_mb) * 1024 * 1024)]

    def _apply() -> None:
        for name, value in wanted:
            kind = getattr(resource, name, None)
            if kind is None:
                continue
            try:
                resource.setrlimit(kind, (value, value))
            except (ValueError, OSError):
                pass
    return _apply


def _cap(raw: Optional[bytes], out_kb: int) -> tuple:
    """``(text, truncated)``: the first ``out_kb`` KB decoded leniently."""
    data = raw or b''
    limit = max(1, int(out_kb)) * 1024
    truncated = len(data) > limit
    return data[:limit].decode('utf-8', errors='replace'), truncated


def _resolve_cwd(cwd: Path) -> tuple:
    """``(resolved, denial)``: the real cwd, or why it is refused."""
    try:
        target = Path(cwd).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as e:
        return None, f'cwd {cwd} is not usable: {e}'
    if not target.is_dir():
        return None, f'cwd {target} is not a directory'
    if not files.ensure_path_allowed(target):
        return None, f"cwd '{target}' is outside the allowed directories"
    return target, ''


def run(argv: list, cwd: Path, limits: Optional[Limits] = None,
        env_extra: Optional[dict] = None) -> ProcResult:
    """Run ``argv`` in ``cwd`` under ``limits`` and return a ProcResult.

    ``argv`` must be a list of strings (a shell string raises TypeError).
    ``cwd`` must resolve to a directory under the read allow-list (the
    usual mode-aware gate applies; a refusal yields ``returncode=-1`` with
    ``denied`` set and no child started). The child gets a fresh
    environment (``PATH``, a temporary ``HOME`` removed afterwards,
    ``LANG``, ``PYTHONDONTWRITEBYTECODE=1``, ``PYTHONPATH=<cwd>`` plus
    ``env_extra``), the rlimits of ``limits`` and a wall-clock timeout.
    stdout/stderr are decoded with ``errors='replace'`` and capped at
    ``limits.out_kb`` each (``truncated`` flags a cut). Never raises for a
    child that fails to start or times out.
    """
    argv = _check_argv(argv)
    limits = limits or Limits()
    target, denial = _resolve_cwd(cwd)
    if target is None:
        return ProcResult(argv, -1, '', '', 0.0, denied=denial)
    home = tempfile.mkdtemp(prefix='guru-proc-')
    started = time.monotonic()
    timed_out = False
    raw_out: Optional[bytes]
    raw_err: Optional[bytes]
    try:
        try:
            proc = subprocess.run(
                argv, cwd=str(target), capture_output=True,
                env=_environment(target, home, env_extra),
                preexec_fn=_limiter(limits), timeout=limits.timeout_s,
                shell=False)
            raw_out, raw_err, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            raw_out, raw_err, code = e.stdout, e.stderr, -1
        except (OSError, ValueError) as e:
            log.info('procs: cannot start %s: %s', argv[0], e)
            raw_out, raw_err, code = b'', str(e).encode('utf-8'), -1
    finally:
        shutil.rmtree(home, ignore_errors=True)
    seconds = time.monotonic() - started
    stdout, cut_out = _cap(raw_out, limits.out_kb)
    stderr, cut_err = _cap(raw_err, limits.out_kb)
    return ProcResult(argv, code, stdout, stderr, seconds,
                      timed_out=timed_out, truncated=cut_out or cut_err)
