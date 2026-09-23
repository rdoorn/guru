"""Ledger: what every model call, user turn and sub-agent task cost.

Entities and a ``LedgerRepository`` Protocol live here; persistence is a
repository (``guru.repositories.jsonl_ledger``). Recording is fire-and-forget
on a single background worker (the decision seam has its own), so nothing
here blocks a turn; every failure is logged and swallowed. Design:
docs/plans/2026-09-23-routing-framework-design.md §3.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

import guru
from guru import config, log, session
from guru.domain import pricing

RUN_ID = uuid.uuid4().hex[:12]           # one per guru process
try:
    PROJECT = Path.cwd().name            # the directory guru was started in
except OSError:                          # cwd deleted underneath us
    PROJECT = ''


class LedgerRepository(Protocol):
    """Append a row to a named stream (calls, tasks, turns, decisions,
    labels).

    A repository may also offer ``save_transcript(task_id, messages)``
    returning a path, and ``rows(stream, run_id=None)`` returning the stored
    rows; the helpers here look them up with ``getattr`` so a minimal
    repository still satisfies the Protocol.
    """

    def append(self, stream: str, row: dict) -> None: ...


_repo: Optional[LedgerRepository] = None
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='guru-ledger')


def set_repository(repo: Optional[LedgerRepository]) -> None:
    """Install (or with None remove) the persistence backend."""
    global _repo
    _repo = repo


def repository() -> Optional[LedgerRepository]:
    """The installed persistence backend, or None."""
    return _repo


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def base_row() -> dict:
    """Columns every ledger row carries: ``ts``, ``run_id``, ``project``."""
    return {'ts': _now(), 'run_id': RUN_ID, 'project': PROJECT}


def submit(stream: str, row: dict) -> None:
    """Queue ``row`` for ``stream`` on the background worker.

    A no-op when no repository is installed or the ledger is disabled;
    never raises.
    """
    repo = _repo
    if repo is None or not config.LEDGER_ENABLED:
        return

    def _write() -> None:
        try:
            repo.append(stream, row)
        except Exception:                        # noqa: BLE001
            log.exc(f'ledger append failed ({stream})')
    try:
        _executor.submit(_write)
    except RuntimeError:
        log.exc('ledger worker unavailable')


def flush() -> None:
    """Block until queued rows are written (tests, exit); never raises."""
    try:
        _executor.submit(lambda: None).result()
    except RuntimeError:                         # executor already shut down
        pass


# --- per-session accumulators -----------------------------------------------

def new_struggle() -> dict:
    """A zeroed struggle-counter dict (one key per ``STRUGGLE_KEYS``)."""
    return {k: 0 for k in session.STRUGGLE_KEYS}


def bump(key: str, n: int = 1) -> None:
    """Add ``n`` (default 1) to ``session.struggle[key]`` for the bound
    session.

    Never raises: a broken counter must not take a turn down with it.
    """
    try:
        counters = session.struggle
        counters[key] = counters.get(key, 0) + n
    except Exception:                            # noqa: BLE001
        log.exc(f'ledger bump failed ({key})')


def struggle_delta(before: dict, after: dict) -> dict:
    """``after - before`` per counter, over the union of their keys."""
    keys = list(before) + [k for k in after if k not in before]
    return {k: after.get(k, 0) - before.get(k, 0) for k in keys}


def _accumulate(row: dict) -> None:
    """Fold one priced call row into the bound session's accumulators."""
    session.call_count += 1
    cost = row.get('cost_usd')
    if cost is None:
        session.cost_known = False
        session.unpriced_calls += 1
    else:
        session.cost_usd += cost


# --- environment ------------------------------------------------------------

def _git(*args: str) -> Optional[str]:
    """Stdout of ``git <args>`` in the cwd, or None on any failure."""
    try:
        proc = subprocess.run(['git', *args], capture_output=True,
                              text=True, timeout=2)
    except Exception:                            # noqa: BLE001
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _config_hash() -> str:
    """sha256[:16] of the sorted ``[routing]`` + ``[decisions]`` settings."""
    sections = {name: config.settings_section(name)
                for name in ('routing', 'decisions')}
    blob = json.dumps(sections, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def _guru_version() -> str:
    """Installed distribution version, else the package's ``__version__``."""
    try:
        return importlib.metadata.version('guru')
    except Exception:                            # noqa: BLE001
        return str(getattr(guru, '__version__', ''))


_static_env: Optional[dict] = None       # version + config hash, per process


def _static_environment() -> dict:
    """The per-process part of the snapshot (settings are read at start)."""
    global _static_env
    if _static_env is None:
        env = {'guru_version': '', 'config_hash': ''}
        try:
            env['guru_version'] = _guru_version()
        except Exception:                        # noqa: BLE001
            pass
        try:
            env['config_hash'] = _config_hash()
        except Exception:                        # noqa: BLE001
            log.exc('ledger config hash failed')
        _static_env = env
    return dict(_static_env)


def environment() -> dict:
    """Snapshot what a task ran under.

    Fields: git sha and dirty flag, cwd, guru version, routing/decisions
    config hash and the configured judge specs (``judge_models`` is the
    ``{point: spec}`` mapping). Version and config hash are computed once per
    process; the git part stays live (the dirty flag changes) but is cheap:
    two short git subprocesses with a 2 s timeout. Every field degrades to
    ``''``/``None`` on failure; the call itself never raises.
    """
    env: dict = {'git_sha': '', 'git_dirty': None, 'cwd': '',
                 'guru_version': '', 'config_hash': '', 'judge_models': {}}
    try:
        env['cwd'] = str(Path.cwd())
    except OSError:
        pass
    sha = _git('rev-parse', 'HEAD')
    if sha:
        env['git_sha'] = sha
        status = _git('status', '--porcelain')
        env['git_dirty'] = bool(status) if status is not None else None
    env.update(_static_environment())
    try:
        env['judge_models'] = {
            str(k): str(v) for k, v in config.DECISIONS_POINTS.items()}
    except Exception:                            # noqa: BLE001
        pass
    return env


# --- transcripts ------------------------------------------------------------

def save_transcript(task_id: str, messages: list) -> str:
    """Queue a task's conversation for the repository; return its path.

    Uses the repository's ``transcript_path(task_id)`` for the path (so the
    task row can carry it at once) and runs ``save_transcript`` on the ledger
    worker. Returns ``''`` when there is no repository, the ledger is
    disabled, or the repository lacks either method. ``messages`` must
    already be plain dicts (the caller converts). Never raises.
    """
    repo = _repo
    if repo is None or not config.LEDGER_ENABLED:
        return ''
    pather = getattr(repo, 'transcript_path', None)
    saver = getattr(repo, 'save_transcript', None)
    if pather is None or saver is None:
        return ''
    try:
        path = str(pather(task_id))
    except Exception:                            # noqa: BLE001
        log.exc(f'ledger transcript path failed ({task_id})')
        return ''

    def _write() -> None:
        try:
            saver(task_id, messages)
        except Exception:                        # noqa: BLE001
            log.exc(f'ledger transcript save failed ({task_id})')
    try:
        _executor.submit(_write)
    except RuntimeError:
        log.exc('ledger worker unavailable')
        return ''
    return path


# --- calls -------------------------------------------------------------------

@dataclass
class CallRecord:
    """One provider call."""
    adapter: str
    model: str
    usage: pricing.Usage
    seconds: float
    phase: str                              # 'step' | 'final' | 'summarise'
    local: bool = False
    cost_header: Optional[float] = None     # proxy-reported cost, if any
    load_s: Optional[float] = None
    prefill_s: Optional[float] = None
    generate_s: Optional[float] = None
    agent: str = ''
    task_id: str = ''
    turn_id: str = ''

    def to_row(self) -> dict:
        """Flatten to a ledger row, pricing the call as it goes."""
        cost: Optional[float]
        if self.cost_header is not None:
            cost, source = self.cost_header, 'header'
        elif self.local:
            cost, source = 0.0, 'local'
        else:
            cost = pricing.cost_usd(self.model, self.usage)
            source = 'table' if cost is not None else 'unknown'
        return {**base_row(), 'agent': self.agent, 'task_id': self.task_id,
                'turn_id': self.turn_id, 'adapter': self.adapter,
                'model': self.model, 'tokens_in': self.usage.input_tokens,
                'tokens_out': self.usage.output_tokens,
                'cache_read': self.usage.cache_read_tokens,
                'cache_write': self.usage.cache_write_tokens,
                'seconds': self.seconds, 'phase': self.phase,
                'load_s': self.load_s, 'prefill_s': self.prefill_s,
                'generate_s': self.generate_s,
                'cost_usd': cost, 'cost_source': source}


def record_call(*, adapter: str, model: str, usage: pricing.Usage,
                seconds: float, phase: str, local: bool = False,
                cost_header: Optional[float] = None,
                load_s: Optional[float] = None,
                prefill_s: Optional[float] = None,
                generate_s: Optional[float] = None) -> None:
    """Record one provider call for the current session's agent/task/turn."""
    try:
        rec = CallRecord(adapter=adapter, model=model, usage=usage,
                         seconds=seconds, phase=phase, local=local,
                         cost_header=cost_header, load_s=load_s,
                         prefill_s=prefill_s, generate_s=generate_s,
                         agent=session.agent_id, task_id=session.task_id,
                         turn_id=session.turn_id)
        row = rec.to_row()
        _accumulate(row)                 # caller thread: session is bound
        submit('calls', row)
    except Exception:                            # noqa: BLE001
        log.exc('ledger record_call failed')


# --- tasks -------------------------------------------------------------------

@dataclass
class TaskRecord:
    """One spawned sub-agent task; written at spawn and again at finish."""
    task_id: str
    parent: str
    task: str
    role: str = ''
    skill: str = ''
    kind: str = 'other'
    complexity: str = 'standard'
    turn_id: str = ''
    status: str = 'running'
    seconds: Optional[float] = None
    answer_len: Optional[int] = None
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: Optional[float] = None
    cost_known: bool = True
    tools_used: list = field(default_factory=list)
    adapter: str = ''
    model: str = ''
    struggle: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    transcript_path: str = ''
    prompt_sha: str = ''                    # sha256[:16] of the system prompt
    tools_active: list = field(default_factory=list)
    # Routing (guru.domain.routing / guru.orchestrator): the resolved
    # Route.as_dict() (None when routing was inert), every filter/fallback
    # that changed the outcome, the secret-scan finding count on the task
    # text, the spend confirmation in force, and the task this one retries.
    route: Optional[dict] = None
    reason: list = field(default_factory=list)
    findings: int = 0
    confirmation: str = ''
    retry_of: str = ''

    def to_row(self) -> dict:
        """Flatten to a ledger row plus a short hash of the task text."""
        row = {**base_row(), **asdict(self)}
        row['text_sha'] = hashlib.sha256(
            self.task.encode('utf-8')).hexdigest()[:16]
        return row


def prompt_hash(system_prompt: str) -> str:
    """sha256[:16] of a system prompt, for grouping tasks by prompt."""
    return hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()[:16]


def new_task(*, task: str, parent: str, role: str = '', skill: str = '',
             kind: str = 'other', complexity: str = 'standard',
             turn_id: str = '', adapter: Optional[str] = None,
             model: Optional[str] = None,
             env: Optional[dict] = None, prompt_sha: str = '',
             tools_active: Optional[list] = None,
             route: Optional[dict] = None, reason: Optional[list] = None,
             findings: int = 0, confirmation: str = '',
             retry_of: str = '') -> TaskRecord:
    """Create a running TaskRecord with a fresh id.

    ``turn_id`` defaults to the bound session's value when empty;
    ``adapter`` and ``model`` default to the session's when None (an
    explicit ``''`` is kept, e.g. for a refused task). Callers spawning on
    another thread (the orchestrator) pass them explicitly, plus the
    environment snapshot, system
    prompt hash and active tool names of the child, and the routing outcome
    (``route``, ``reason``, ``findings``, ``confirmation``, ``retry_of``).
    """
    return TaskRecord(task_id=uuid.uuid4().hex[:12], parent=parent, task=task,
                      role=role or '', skill=skill or '', kind=kind,
                      complexity=complexity,
                      turn_id=turn_id or session.turn_id,
                      adapter=(getattr(session.adapter, 'name', '')
                               if adapter is None else adapter),
                      model=session.model if model is None else model,
                      env=dict(env or {}), prompt_sha=prompt_sha,
                      tools_active=list(tools_active or []),
                      route=dict(route) if route is not None else None,
                      reason=list(reason or []), findings=int(findings),
                      confirmation=confirmation or '',
                      retry_of=retry_of or '')


def record_task(rec: TaskRecord) -> None:
    """Write the spawn row for ``rec``."""
    try:
        submit('tasks', rec.to_row())
    except Exception:                            # noqa: BLE001
        log.exc('ledger record_task failed')


def finish_task(rec: TaskRecord, *, status: str, seconds: float,
                answer_len: int, calls: int, tokens_in: int, tokens_out: int,
                cost_usd: Optional[float], tools_used: Optional[list] = None,
                cost_known: bool = True, struggle: Optional[dict] = None,
                transcript_path: str = '') -> None:
    """Write the completion row for ``rec`` (same task_id, new status).

    ``cost_usd`` is None (and ``cost_known`` False) when any of the task's
    calls could not be priced; ``struggle`` is copied.
    """
    try:
        rec.status, rec.seconds, rec.answer_len = status, seconds, answer_len
        rec.calls, rec.tokens_in, rec.tokens_out = calls, tokens_in, tokens_out
        rec.cost_usd, rec.tools_used = cost_usd, list(tools_used or [])
        rec.cost_known = bool(cost_known) and cost_usd is not None
        rec.struggle = dict(struggle or {})
        rec.transcript_path = transcript_path
        submit('tasks', rec.to_row())
    except Exception:                            # noqa: BLE001
        log.exc('ledger finish_task failed')


# --- turns -------------------------------------------------------------------

@dataclass
class TurnRecord:
    """One user message to the main agent, closed at its final answer."""
    turn_id: str
    request: str
    model: str
    seconds: float
    tasks_spawned: int
    tools_used: list
    tokens_in: int
    tokens_out: int
    cost_usd: Optional[float]
    controller_executed: bool
    agent: str = 'main'
    adapter: str = ''
    struggle: dict = field(default_factory=dict)  # this turn's deltas

    def to_row(self) -> dict:
        """Flatten to a ledger row."""
        return {**base_row(), **asdict(self)}


def record_turn(rec: TurnRecord) -> None:
    """Write the row for one completed user turn."""
    try:
        submit('turns', rec.to_row())
    except Exception:                            # noqa: BLE001
        log.exc('ledger record_turn failed')


def new_turn_id() -> str:
    """Fresh 12-hex-char id for a user turn."""
    return uuid.uuid4().hex[:12]


# --- labels ------------------------------------------------------------------

def record_label(target_id: str, labeller: str, label: str,
                 note: str = '') -> None:
    """Append a verdict to the ``labels`` stream.

    Two granularities share the stream: ``target_id`` is a turn or task id
    with label ``good``/``bad`` (``/good``, ``/bad``: an outcome verdict),
    or a decision key ``<point>:<question>:<input_sha>`` with label
    ``yes``/``no`` (``guru.ledger_cli review``: the correct answer to that
    judge question; see ``ledger_report.decision_key``). ``labeller`` is
    ``user``, ``judge:<name>`` or ``review``. The ledger is never edited: a
    later label for the same target supersedes an earlier one at read
    time. Never raises.
    """
    try:
        submit('labels', {**base_row(), 'target_id': target_id,
                          'labeller': labeller, 'label': label,
                          'note': note or ''})
    except Exception:                            # noqa: BLE001
        log.exc('ledger record_label failed')


# --- read-side aggregation ---------------------------------------------------

def model_key(row: dict) -> str:
    """``adapter|model`` (the bench's ``models.txt`` form) for a row."""
    return f"{row.get('adapter') or ''}|{row.get('model') or ''}"


def per_model_usage(calls_rows: list) -> dict:
    """Calls, tokens (in, out, cache read, cache write) and cost per
    ``adapter|model`` over ``calls_rows``.

    ``cost_usd`` is None for a model as soon as one of its calls could not
    be priced. Pure; used by :func:`run_summary` and the ledger report.
    """
    out: dict = {}
    for r in calls_rows:
        key = model_key(r)
        m = out.setdefault(key, {
            'adapter': r.get('adapter') or '', 'model': r.get('model') or '',
            'calls': 0, 'tokens_in': 0, 'tokens_out': 0, 'cache_read': 0,
            'cache_write': 0, 'cost_usd': 0.0})
        m['calls'] += 1
        for col in ('tokens_in', 'tokens_out', 'cache_read', 'cache_write'):
            m[col] += int(r.get(col) or 0)
        cost = r.get('cost_usd')
        if cost is None:
            m['cost_usd'] = None
        elif m['cost_usd'] is not None:
            m['cost_usd'] += float(cost)
    return out


def sum_cost(values: list) -> Optional[float]:
    """Sum of costs, or None when any is unknown."""
    total = 0.0
    for v in values:
        if v is None:
            return None
        total += float(v)
    return total


def run_summary(calls_rows: list, tasks_rows: list, run_id: str) -> dict:
    """What one run spent: a pure aggregation for ``/ledger``.

    Returns ``{'run_id', 'models', 'tasks', 'top_tasks', 'totals'}``:
    ``models`` per ``adapter|model`` (:func:`per_model_usage`), ``tasks``
    counts distinct task ids per ``adapter|model`` (a task's latest row
    decides), ``top_tasks`` are the three most expensive finished tasks
    (``status != 'running'``) by cost then seconds with unknown costs last,
    and ``totals`` sums calls, tokens (in/out/cache read/cache write), cost
    (None if any unknown) and tasks.
    Rows of other runs are ignored.
    """
    calls = [r for r in calls_rows if r.get('run_id') == run_id]
    tasks = [r for r in tasks_rows if r.get('run_id') == run_id]
    models = per_model_usage(calls)
    latest: dict = {}
    for r in tasks:
        latest[r.get('task_id')] = r
    per_route: dict = {}
    for r in latest.values():
        key = model_key(r)
        per_route[key] = per_route.get(key, 0) + 1
    finished = [r for r in latest.values() if r.get('status') != 'running']

    def _rank(r: dict) -> tuple:
        cost = r.get('cost_usd')
        return (cost is None, -(cost or 0.0), -(r.get('seconds') or 0.0))
    top = [{'task_id': r.get('task_id', ''), 'role': r.get('role', ''),
            'kind': r.get('kind', ''), 'adapter': r.get('adapter', ''),
            'model': r.get('model', ''), 'cost_usd': r.get('cost_usd'),
            'seconds': r.get('seconds'), 'status': r.get('status', ''),
            'task': ' '.join(str(r.get('task') or '').split())[:80]}
           for r in sorted(finished, key=_rank)[:3]]
    totals = {'calls': len(calls),
              **{col: sum(m[col] for m in models.values())
                 for col in ('tokens_in', 'tokens_out', 'cache_read',
                             'cache_write')},
              'cost_usd': sum_cost([m['cost_usd'] for m in models.values()]),
              'tasks': len(latest)}
    return {'run_id': run_id, 'models': models, 'tasks': per_route,
            'top_tasks': top, 'totals': totals}
