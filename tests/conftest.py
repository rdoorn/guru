"""Shared test helpers: the in-memory ledger repository, a per-test reset
of the default session's ledger accumulators and an isolated skill catalog.

HOME points at a throwaway directory (under ``~/.guru-evals``) for the
whole run, set before guru is imported (``config`` resolves ``~/.guru`` at
import): no test, and no subprocess a test starts, writes to the user's
real home (``GURU.md``, ``adapters.toml``, ``guru.log`` and the matplotlib
font cache did).
"""
import atexit
import os
import shutil
import tempfile

_REAL_HOME = os.path.expanduser('~')
# The container runtime keeps its client config and VM sockets under the
# real home; the sandbox tests read them (never write), so they stay there.
for _var, _dir in (('DOCKER_CONFIG', '.docker'), ('COLIMA_HOME', '.colima')):
    if _var not in os.environ and os.path.isdir(
            os.path.join(_REAL_HOME, _dir)):
        os.environ[_var] = os.path.join(_REAL_HOME, _dir)
# Under the real home (Colima shares only that with its VM, and the sandbox
# tests mount their copies from HOME) but outside ~/.guru, next to the
# contained HOMEs of ``python -m guru.evals verify``; removed at exit.
_CONTAINED = os.path.join(_REAL_HOME, '.guru-evals')
os.makedirs(_CONTAINED, exist_ok=True)
_TEST_HOME = tempfile.mkdtemp(prefix='test-home-', dir=_CONTAINED)
os.environ['HOME'] = _TEST_HOME
os.environ['MPLCONFIGDIR'] = os.path.join(_TEST_HOME, '.matplotlib')
atexit.register(shutil.rmtree, _TEST_HOME, True)

import pytest  # noqa: E402

from guru import config, session, skills  # noqa: E402
from guru.domain import ledger  # noqa: E402


@pytest.fixture(scope='session', autouse=True)
def _isolated_eval_sandbox_root(tmp_path_factory):
    """Point the eval runner's stable sandbox workdir
    (``guru.evals.runner.sandbox_root()``) at a per-session directory.

    The runner's sandbox tests run ``run_case`` for real, which empties
    ``<tmp>/guru-eval-sandbox``; on the shared temp dir that deleted the
    copy of a live eval in another process (run 94fdc1bb11a5 — and the
    dogfood case runs this very suite inside its copy).
    """
    mp = pytest.MonkeyPatch()
    mp.setenv('GURU_EVAL_SANDBOX_ROOT',
              str(tmp_path_factory.mktemp('eval-sandbox-root')))
    try:
        yield
    finally:
        mp.undo()


@pytest.fixture(scope='session', autouse=True)
def _isolated_brief_store(tmp_path_factory):
    """Point the brief store (``guru.repositories.briefs.root()``) at a
    per-session directory: the runner's own tests run ``run_case`` for
    real and would otherwise leave one ``~/.guru/briefs/<fixture>-<hash>``
    directory per case (18 stray ones were found on 2026-09-26)."""
    mp = pytest.MonkeyPatch()
    mp.setenv('GURU_BRIEFS_DIR',
              str(tmp_path_factory.mktemp('brief-store')))
    try:
        yield
    finally:
        mp.undo()


@pytest.fixture(autouse=True)
def _no_log_file(monkeypatch):
    """``guru.log.setup`` is a no-op under the tests: the eval CLI (and
    any other entry point a test calls) must not attach a handler to the
    developer's real ``~/.guru/guru.log``."""
    from guru import log
    monkeypatch.setattr(log, 'setup', lambda: None)


@pytest.fixture(autouse=True)
def _isolated_skill_catalog(tmp_path, monkeypatch):
    """Keep ``skills.REGISTRY`` empty across tests and the catalog directory
    under ``tmp_path``.

    The bench and the eval runner call ``skills.ensure_loaded()``; without
    this a test that runs either would fill the process-wide registry from
    the developer's ``~/.guru/skills`` and every later test would see a
    catalog block on the system prompt.
    """
    monkeypatch.setattr(config, 'GURU_SKILLS_DIR', tmp_path / 'skills')
    saved = dict(skills.REGISTRY)
    skills.REGISTRY.clear()
    try:
        yield
    finally:
        skills.REGISTRY.clear()
        skills.REGISTRY.update(saved)


@pytest.fixture(autouse=True)
def _fresh_accumulators(monkeypatch):
    """Zero the bound session's call/cost/struggle counters for each test.

    Adapter and ledger tests feed ``record_call``/``bump`` on the shared
    default SessionState; without this reset one test's unknown-priced call
    would flip ``cost_known`` for every test after it.
    """
    monkeypatch.setattr(session, 'call_count', 0)
    monkeypatch.setattr(session, 'cost_usd', 0.0)
    monkeypatch.setattr(session, 'cost_known', True)
    monkeypatch.setattr(session, 'unpriced_calls', 0)
    monkeypatch.setattr(session, 'struggle', ledger.new_struggle())
    monkeypatch.setattr(session, 'last_error', '')
    ledger.forget_calls()            # the per-turn rows behind turn_summary


class FakeRepo:
    """In-memory LedgerRepository: ``rows`` is a list of (stream, row)."""

    def __init__(self) -> None:
        self.rows: list = []
        self.transcripts: dict = {}

    def append(self, stream: str, row: dict) -> None:
        self.rows.append((stream, row))

    def transcript_path(self, task_id: str) -> str:
        """Deterministic fake path for ``task_id``."""
        return f'/fake/transcripts/{task_id}.json.gz'

    def save_transcript(self, task_id: str, messages: list) -> str:
        """Keep the transcript in memory; return the fake path."""
        self.transcripts[task_id] = list(messages)
        return self.transcript_path(task_id)

    def stream(self, name: str) -> list:
        """Rows written to ``name``, in order."""
        return [r for s, r in self.rows if s == name]


@pytest.fixture
def fake_repo(monkeypatch):
    """Install a FakeRepo as the ledger backend (enabled) for one test."""
    repo = FakeRepo()
    ledger.set_repository(repo)
    monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
    try:
        yield repo
    finally:
        ledger.flush()
        ledger.set_repository(None)
