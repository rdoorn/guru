"""Lightweight logging so the codebase's exception guards are debuggable
instead of silent. No output by default (a NullHandler is attached); call
setup() once (from the CLI) to write to ~/.guru/guru.log, and to stderr too
when GURU_DEBUG is set."""
import logging
import os
from pathlib import Path

log = logging.getLogger('guru')
log.addHandler(logging.NullHandler())
log.setLevel(logging.DEBUG)
_configured = False


def setup() -> None:
    """Attach a file handler (~/.guru/guru.log) and, under GURU_DEBUG, stderr.
    Safe to call more than once."""
    global _configured
    if _configured:
        return
    _configured = True
    fmt = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s: %(message)s')
    try:
        path = Path(os.path.expanduser('~/.guru')) / 'guru.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding='utf-8')
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)
    except OSError:
        pass
    if os.environ.get('GURU_DEBUG'):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.DEBUG)
        log.addHandler(sh)


def exc(msg: str) -> None:
    """Log the current exception with traceback at debug level."""
    log.debug(msg, exc_info=True)
