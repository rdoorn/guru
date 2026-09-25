"""Shared test helpers: the in-memory ledger repository, a per-test reset
of the default session's ledger accumulators and an isolated skill catalog."""
import pytest

from guru import config, session, skills
from guru.domain import ledger


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
