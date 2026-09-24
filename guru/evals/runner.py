"""Eval runner (endpoint): one case = one prompt through the headless bench.

Per case: copy the fixture into a temp dir (a ``[fixture_git]`` case is
``git archive``-d from its repository at the pinned ref) and ``git init``
it, chdir there
with the case's access mode and auto-deny askers (an unattended run must
never sit on a prompt), point the ledger at the run's ``ledger/`` dir, run
the prompt through :class:`guru.bench.BenchRun`, then collect the answer,
tools, sub-agents, stall nudges, the fixture diff (``git status``), the
fixture's own pytest verdict and the cost from the ledger rows this case
appended. Everything touched (cwd, ``config.MODE``, allow-lists, askers,
persistence of approvals, ledger repository) is restored afterwards.

Routing: :func:`run_suite` takes a :class:`RoutingSettings` (parsed from a
``[routing]`` file by :func:`load_routing_file`) and builds the adapter
registry over the suite's adapters, so :class:`guru.bench.BenchRun` routes
sub-agents exactly as the TUI would; ``routing.controller`` runs the main
agent as a controller and ``routing.secret_scan`` binds the secret scanner
for the duration (``config.SECRET_SCAN`` and the scanner are restored). A
case records the distinct ``Adapter|model`` its sub-agent tasks ran on
(``CaseResult.routes``). Remote spend is denied unless ``allow_spend`` is
set, which installs a granting spend asker per case.

Judges: an experiment file may also carry a ``[decisions]`` table
(:func:`load_decisions_file`); :func:`run_suite` then sets the decision
seam (``config.DECISIONS_MODE/POINTS/ACTIVE/THRESHOLDS``) from it, installs
the judges for the duration and records their names on the run; the seam
and the judge registry are restored (cleared) afterwards.

Process-global state: the cwd, ``guru.config`` and the domain-level asker
hooks are shared by the whole process, so cases run strictly sequentially
and :func:`run_case` must never run concurrently with the TUI, the bench or
another runner in the same process. Escalations (a directory outside the
copy, a web domain) are denied in every mode: the sandbox turns
``config.AUTO_GRANT`` off, so even ``auto`` cases consult the (denying)
askers for anything outside the copy, and nothing is persisted to the
developer's ``.guru`` allow-list files.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy as copymod
import fnmatch
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

from guru import bench, config, judges, log, session, skills
from guru.adapters import turn
from guru.adapters.base import Adapter
from guru.domain import conversation
from guru.domain import decisions as decision_seam
from guru.domain import files, ledger, policy, spend, tools
from guru.evals import cases, checks, runs
from guru.evals.cases import Case, GitFixture
from guru.evals.checks import Observed
from guru.evals.runs import CaseResult, Run
from guru.repositories import settings as routing_settings
from guru.repositories.adapters import AdapterRegistry, registry_from
from guru.repositories.jsonl_ledger import JsonlLedger
from guru.repositories.settings import DecisionsSettings, RoutingSettings
from guru.scanners.secrets import load_project_scanner

_IGNORE_PATTERNS = ('__pycache__', '.pytest_cache', '*.pyc', '.git')
_COPY_IGNORE = shutil.ignore_patterns(*_IGNORE_PATTERNS)
_GIT_EXCLUDE = '__pycache__/\n.pytest_cache/\n*.pyc\n'
_GIT_IDENTITY = ['-c', 'user.name=evals', '-c', 'user.email=evals@local',
                 '-c', 'commit.gpgsign=false']
FIXTURE_PYTEST_TIMEOUT_S = 300
WORKER_DRAIN_S = 30.0          # how long to wait for leftover worker threads
_WORKER_POLL_S = 0.2
_PERSISTERS = ('persist_read_dir', 'persist_write_dir', 'persist_domain')
DEFAULT_TRAJECTORY_DIR = cases.REPO_ROOT / 'evals'   # TRAJECTORY.md


def _deny(question: str) -> bool:
    return False


def _grant(question: str) -> bool:
    return True


@dataclass(frozen=True)
class Routing:
    """How a run routes sub-agents: the validated ``[routing]`` settings,
    the registry that resolves rung adapter names and the routing file's
    stem (recorded on the run)."""
    settings: RoutingSettings
    registry: AdapterRegistry
    name: str = ''


def _read_toml(path: Path) -> dict:
    """The parsed TOML of an experiment file; ``ValueError`` when the file
    cannot be read or is not valid TOML."""
    try:
        return tomllib.loads(path.read_text(encoding='utf-8'))
    except OSError as e:
        raise ValueError(f'{path}: cannot read routing file: {e}') from e
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f'{path}: invalid TOML: {e}') from e


def load_routing_file(path: Path) -> RoutingSettings:
    """Parse the ``[routing]`` table of a TOML file (same shape as
    ``settings.toml``). ``ValueError`` for a missing/unreadable file,
    invalid TOML, a missing table or an invalid key."""
    path = Path(path)
    data = _read_toml(path)
    section = data.get('routing')
    if not isinstance(section, dict) or not section:
        raise ValueError(f'{path}: no [routing] table')
    return routing_settings.load_routing(section=section)


def load_decisions_file(path: Path) -> Optional[DecisionsSettings]:
    """Parse the optional ``[decisions]`` table of an experiment file
    (``mode`` plus ``points``/``active``/``thresholds``). None when the
    file has no such table; ``ValueError`` as :func:`load_routing_file`
    and for an invalid key or value."""
    path = Path(path)
    data = _read_toml(path)
    section = data.get('decisions')
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError(f'{path}: [decisions] must be a table')
    try:
        return routing_settings.load_decisions(section)
    except ValueError as e:
        raise ValueError(f'{path}: {e}') from e


def _no_persist(value: str) -> None:
    """Stand-in for ``config.persist_*`` during a case: approve in memory
    only, never touch the developer's allow-list files."""


def _no_save_model_ctx(model: str, num_ctx: int) -> None:
    """Stand-in for ``config.save_model_ctx`` while an adapter activates: a
    pinned ``--num-ctx`` (or a fit made for the suite) must not become the
    developer's stored per-model context in ``~/.guru/model_ctx.json``."""


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(['git', *_GIT_IDENTITY, *args], cwd=repo,
                          capture_output=True, text=True, check=True)
    return proc.stdout


def _assert_no_running_loop(what: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(f'{what} is synchronous and drives its own asyncio '
                       'loop; do not call it from a running event loop')


# --- fixture copy -----------------------------------------------------------

def _ignored(member: str) -> bool:
    """True when any path component of ``member`` matches a cache pattern
    (the same set ``shutil.copytree`` skips for directory fixtures)."""
    return any(fnmatch.fnmatch(part, pat)
               for part in Path(member).parts for pat in _IGNORE_PATTERNS)


def _archive(git: GitFixture, dst: Path) -> None:
    """Extract ``git archive <ref>`` of the repository into ``dst``
    (skipping cache members); ``ValueError`` when git refuses the ref."""
    proc = subprocess.run(
        ['git', '-C', str(git.path), 'archive', '--format=tar', git.ref],
        capture_output=True)
    if proc.returncode != 0:
        raise ValueError(f'git archive {git.ref!r} in {git.path} failed: '
                         f'{proc.stderr.decode(errors="replace").strip()}')
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        members = [m for m in tar.getmembers() if not _ignored(m.name)]
        tar.extractall(dst, members=members, filter='data')


def _init_git(dst: Path) -> None:
    """``git init`` the copy and commit everything (caches excluded) so
    ``files_changed`` can diff against it."""
    _git(dst, 'init', '-q')
    exclude = dst / '.git' / 'info' / 'exclude'
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text(_GIT_EXCLUDE, encoding='utf-8')
    _git(dst, 'add', '-A')
    _git(dst, 'commit', '-q', '--allow-empty', '-m', 'fixture')


def prepare_fixture(source: Union[str, GitFixture], workdir: Path,
                    fixtures_dir: Optional[Path] = None) -> Path:
    """Materialise the fixture under ``workdir`` and commit it in a new git.

    A ``str`` names a directory under ``fixtures_dir`` (default
    ``evals/fixtures``), copied to ``workdir/<name>``; a
    :class:`GitFixture` is ``git archive``-d at its ref to
    ``workdir/<repo dir name>``. ``files_changed`` is later computed with
    ``git status --porcelain`` in the copy, so untracked files count as
    changes; caches are excluded either way.
    """
    if isinstance(source, GitFixture):
        dst = Path(workdir) / source.path.name
        _archive(source, dst)
    else:
        base = Path(fixtures_dir) if fixtures_dir is not None \
            else cases.FIXTURES_DIR
        src = base / source
        if not src.is_dir():
            raise ValueError(f'fixture {source!r} not found at {src}')
        dst = Path(workdir) / source
        shutil.copytree(src, dst, ignore=_COPY_IGNORE)
    _init_git(dst)
    return dst


def files_changed(repo: Path) -> list[str]:
    """Sorted paths modified, added, deleted or untracked since the commit."""
    out = _git(repo, 'status', '--porcelain', '-z', '--untracked-files=all')
    entries = [e for e in out.split('\0') if e]
    paths: set[str] = set()
    skip_next = False
    for entry in entries:
        if skip_next:                 # the old name of a rename/copy
            skip_next = False
            continue
        status, path = entry[:2], entry[3:]
        paths.add(path)
        skip_next = status[0] in 'RC'
    return sorted(paths)


def _fixture_env(repo: Path) -> dict:
    """The environment for the fixture's pytest: ``PYTHONPATH`` starts with
    the copy, so a copied package (a git fixture of a real project, which
    has no venv of its own) shadows any installed one."""
    env = dict(os.environ)
    prev = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(repo) + (os.pathsep + prev if prev else '')
    return env


def fixture_tests_pass(repo: Path,
                       timeout: float = FIXTURE_PYTEST_TIMEOUT_S) -> bool:
    """Run the fixture's own pytest in ``repo`` (this interpreter, the copy
    first on ``PYTHONPATH``); True when it exits 0."""
    try:
        proc = subprocess.run(
            [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider'],
            cwd=repo, capture_output=True, text=True, timeout=timeout,
            env=_fixture_env(repo))
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


# --- adapter / model resolution --------------------------------------------

def _split_spec(spec: str) -> tuple[str, str]:
    adapter, sep, model = spec.partition('|')
    if not sep or not adapter.strip() or not model.strip():
        raise ValueError(f"model spec {spec!r} must be 'Adapter|model'")
    return adapter.strip(), model.strip()


def _activate(state: session.SessionState, adapter: Adapter,
              model: str) -> None:
    """Activate ``model`` on ``adapter`` with ``state`` bound (adapters write
    the resolved context window to the bound session).
    ``config.save_model_ctx`` is a no-op for the duration so nothing lands
    in ``model_ctx.json``."""
    token = session.use(state)
    prev_save = config.save_model_ctx
    config.save_model_ctx = _no_save_model_ctx
    try:
        adapter.activate(model)
        state.model = model
    finally:
        config.save_model_ctx = prev_save
        session.reset(token)


def _state_for(spec: str, base_state: session.SessionState,
               adapters: list[Adapter]) -> session.SessionState:
    """The base state for a case: ``default`` -> ``base_state``; else a copy
    on the named adapter with ``model`` activated."""
    if spec == cases.DEFAULT_MODEL:
        return base_state
    name, model = _split_spec(spec)
    adapter = bench._adapter_for(name, adapters)
    if adapter is None:
        raise ValueError(f'no adapter {name!r} for case model {spec!r}')
    # A shallow copy keeps every context/ctx-window setting of the suite's
    # base state (only adapter + model differ); the orchestrator copies the
    # scalar fields from it and never shares the mutable ones.
    st = copymod.copy(base_state)
    st.adapter = adapter
    _activate(st, adapter, model)
    return st


def _default_model(adapter: Adapter) -> str:
    """Mirror ``cli._startup_select``: the CLI default when the adapter lists
    it, else the first listed model, else the CLI default anyway."""
    from guru import cli
    try:
        ids = [m.model_id for m in adapter.list_models()]
    except Exception:                                # noqa: BLE001
        ids = []
    if cli.DEFAULT_MODEL in ids or not ids:
        return cli.DEFAULT_MODEL
    return ids[0]


def resolve_base(model_spec: Optional[str], adapters: list[Adapter],
                 num_ctx: int = 0) -> tuple[session.SessionState, str]:
    """Build the suite's base state; return ``(state, 'Adapter|model')``.

    ``None``/``default`` picks the default (Ollama) adapter and its default
    model as the CLI would; otherwise ``Adapter|model`` is resolved through
    the bench's adapter lookup. ``ValueError`` for a bad or unknown spec.
    ``num_ctx`` > 0 pins the context window like the CLI's ``--num-ctx``:
    it is set as ``num_ctx_override`` before ``activate`` so the adapter
    skips the GPU auto-fit and loads the model at that size (a case with
    its own ``model`` inherits the pin through the copied state).
    """
    if not model_spec or model_spec == cases.DEFAULT_MODEL:
        adapter = bench._adapter_for(None, adapters)
        model = _default_model(adapter)
    else:
        name, model = _split_spec(model_spec)
        adapter = bench._adapter_for(name, adapters)
        if adapter is None:
            raise ValueError(f'no adapter {name!r} in {model_spec!r}')
    st = session.SessionState()
    st.adapter = adapter
    st.num_ctx_override = max(0, int(num_ctx))
    _activate(st, adapter, model)
    return st, f'{getattr(adapter, "name", "?")}|{model}'


# --- one case ---------------------------------------------------------------

class _Sandbox:
    """Result of running inside :func:`_sandbox`; ``leaked`` is set when
    worker threads outlived the run (the deny askers then stay installed)."""
    leaked: bool = False


@contextlib.contextmanager
def _sandbox(copy: Path, mode: str, repo: JsonlLedger,
             allow_spend: bool = False) -> Iterator[_Sandbox]:
    """cwd, access mode, allow-lists, askers, persistence and ledger for one
    case; all restored afterwards (askers excepted when workers leaked).
    The spend asker denies unless ``allow_spend``."""
    prev_cwd = os.getcwd()
    prev_mode = config.MODE
    prev_grant = config.AUTO_GRANT
    prev_ledger = config.LEDGER_ENABLED
    prev_read = set(config.ALLOWED_READ_DIRS)
    prev_write = set(config.ALLOWED_WRITE_DIRS)
    prev_domains = set(config.ALLOWED_DOMAINS)
    prev_persist = {n: getattr(config, n) for n in _PERSISTERS}
    prev_domain, prev_path = tools._domain_asker, files._path_asker
    prev_spend = spend._asker
    prev_repo = ledger.repository()
    key = str(copy.resolve())
    state = _Sandbox()
    try:                       # every mutation below is undone by finally
        os.chdir(copy)
        config.MODE = mode
        config.AUTO_GRANT = False      # auto cases: escalations hit _deny
        config.LEDGER_ENABLED = True
        config.ALLOWED_READ_DIRS.add(key)
        config.ALLOWED_WRITE_DIRS.add(key)
        for n in _PERSISTERS:
            setattr(config, n, _no_persist)
        tools.set_domain_asker(_deny)
        files.set_path_asker(_deny)
        # A case never pays for remote unless the run opted in.
        spend.set_spend_asker(_grant if allow_spend else _deny)
        spend.reset()
        ledger.set_repository(repo)
        yield state
    finally:
        ledger.flush()
        ledger.set_repository(prev_repo)
        if state.leaked:
            log.warning('evals: worker threads still running after %s; '
                        'deny askers left installed', copy.name)
        else:
            tools.set_domain_asker(prev_domain)
            files.set_path_asker(prev_path)
            spend.set_spend_asker(prev_spend)
        for n, fn in prev_persist.items():
            setattr(config, n, fn)
        config.ALLOWED_READ_DIRS.clear()
        config.ALLOWED_READ_DIRS.update(prev_read)
        config.ALLOWED_WRITE_DIRS.clear()
        config.ALLOWED_WRITE_DIRS.update(prev_write)
        config.ALLOWED_DOMAINS.clear()
        config.ALLOWED_DOMAINS.update(prev_domains)
        config.LEDGER_ENABLED = prev_ledger
        config.AUTO_GRANT = prev_grant
        config.MODE = prev_mode
        os.chdir(prev_cwd)


def _drain_workers(agents: list, limit: float = WORKER_DRAIN_S) -> bool:
    """Wait (bounded) for leftover worker threads; True when all are idle."""
    end = time.monotonic() + limit
    while any(a.busy for a in agents):
        if time.monotonic() >= end:
            return False
        time.sleep(_WORKER_POLL_S)
    return True


def _execute(case: Case, copy: Path, base_state: session.SessionState,
             adapters: list[Adapter], repo: JsonlLedger,
             routing: Optional[Routing] = None, allow_spend: bool = False
             ) -> tuple[list, float, str]:
    """Run the prompt inside the sandbox.

    Returns ``(agents, seconds, error)``; the clock covers only the model
    run (not the fixture copy). ``error`` is set when workers were still
    running after the bounded drain. ``routing`` makes the bench route
    sub-agents (inert when None).
    """
    registry = routing.registry if routing is not None else None
    settings = routing.settings if routing is not None else None
    with _sandbox(copy, case.mode, repo, allow_spend) as box:
        base = _state_for(case.model, base_state, adapters)
        token = session.use(base)
        t0 = time.monotonic()
        try:
            agents = asyncio.run(
                bench.BenchRun(base, registry=registry,
                               routing=settings).run(case.prompt,
                                                     timeout=case.timeout_s))
        finally:
            seconds = time.monotonic() - t0
            session.reset(token)
        error = ''
        if not _drain_workers(agents, WORKER_DRAIN_S):
            box.leaked = True
            error = 'workers still running after timeout'
        return agents, seconds, error


def _stall_nudges(agents: list) -> int:
    """How many times the turn loop had to nudge any agent into acting."""
    n = 0
    for a in agents:
        for m in a.state.messages:
            if conversation.msg_role(m) == 'user' and \
                    conversation.msg_content(m) == turn._NUDGE_TEXT:
                n += 1
    return n


def _observe(agents: list, seconds: float, timed_out: bool,
             changed: list[str], tests_pass: Optional[bool],
             error: str) -> Observed:
    answer = bench._final_answer(agents[0]) if agents else ''
    if not error and not timed_out and not answer:
        error = 'empty answer'
    tools_used: list[str] = []
    for a in agents:
        tools_used += bench._tool_names(a)
    return Observed(
        answer=answer, tools_used=tools_used,
        spawned=max(len(agents) - 1, 0),
        roles=[a.state.active_role for a in agents[1:]
               if a.state.active_role],
        stall_nudges=_stall_nudges(agents), seconds=round(seconds, 3),
        files_changed=changed, fixture_tests_pass=tests_pass,
        timed_out=timed_out, error=error)


def _cost(rows: list[dict]) -> Optional[float]:
    """Sum of ``cost_usd`` over call rows; None when empty or any unknown."""
    if not rows or any(r.get('cost_usd') is None for r in rows):
        return None
    return float(sum(r['cost_usd'] for r in rows))


def _routes(rows: list[dict]) -> list[str]:
    """Distinct ``Adapter|model`` of task rows, first-appearance order;
    rows without an adapter/model (refused tasks) are skipped."""
    out: list[str] = []
    for r in rows:
        adapter, model = r.get('adapter') or '', r.get('model') or ''
        if not adapter or not model:
            continue
        spec = f'{adapter}|{model}'
        if spec not in out:
            out.append(spec)
    return out


def _save_transcript(agents: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as fh:
        json.dump(bench.serialize_transcript(agents), fh,
                  ensure_ascii=False, default=str)


def run_case(case: Case, base_state: session.SessionState,
             adapters: list[Adapter], out_dir: Path,
             fixtures_dir: Optional[Path] = None,
             routing: Optional[Routing] = None,
             allow_spend: bool = False) -> CaseResult:
    """Run one case and evaluate it; never raises for a failing run.

    Synchronous entry point: drives its own asyncio loop, so it must not be
    called from a running event loop (``RuntimeError``), nor concurrently
    with anything else that uses the process cwd or ``guru.config``.
    Writes ``out_dir/transcripts/<case>.json.gz`` and appends this case's
    ledger rows under ``out_dir/ledger``. A timeout is detected as the
    bench does: wall time reached ``case.timeout_s``. ``routing`` routes
    sub-agents (see :class:`Routing`); ``allow_spend`` grants remote spend
    for the case. A ``[fixture_git]`` case records ``{'path', 'ref'}`` as
    ``observed.fixture_git``.
    """
    _assert_no_running_loop('run_case')
    out_dir = Path(out_dir)
    repo = JsonlLedger(out_dir / 'ledger')
    repo.dir.mkdir(parents=True, exist_ok=True)
    rows_before = len(repo.rows('calls'))
    tasks_before = len(repo.rows('tasks'))
    workdir = Path(tempfile.mkdtemp(prefix=f'guru-eval-{case.name}-'))
    agents: list = []
    error = ''
    seconds = 0.0
    changed: list[str] = []
    tests_pass: Optional[bool] = None
    try:
        copy = prepare_fixture(case.fixture_git or case.fixture, workdir,
                               fixtures_dir)
        try:
            agents, seconds, error = _execute(case, copy, base_state,
                                              adapters, repo, routing,
                                              allow_spend)
        except Exception as e:                       # noqa: BLE001
            error = str(e) or type(e).__name__
        changed = files_changed(copy)
        if case.expect.fixture_tests_pass is not None:
            tests_pass = fixture_tests_pass(copy)
    except Exception as e:                           # noqa: BLE001
        error = error or str(e) or type(e).__name__
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    timed_out = bool(case.timeout_s) and seconds >= case.timeout_s
    obs = _observe(agents, seconds, timed_out, changed, tests_pass, error)
    if case.fixture_git is not None:
        obs.fixture_git = {'path': str(case.fixture_git.path),
                           'ref': case.fixture_git.ref}
    tpath = out_dir / 'transcripts' / f'{case.name}.json.gz'
    _save_transcript(agents, tpath)
    results = checks.evaluate(case.expect, obs)
    return CaseResult(
        case=case.name, passed=checks.passed(results),
        checks=[asdict(r) for r in results], observed=asdict(obs),
        rubric=case.expect.rubric, transcript_path=str(tpath),
        cost_usd=_cost(repo.rows('calls')[rows_before:]),
        routes=_routes(repo.rows('tasks')[tasks_before:]))


# --- the suite --------------------------------------------------------------

def git_sha() -> str:
    """HEAD of the guru checkout, or '' when git is unavailable."""
    try:
        proc = subprocess.run(['git', 'rev-parse', 'HEAD'],
                              cwd=cases.REPO_ROOT, capture_output=True,
                              text=True)
    except Exception:                                # noqa: BLE001
        return ''
    return proc.stdout.strip() if proc.returncode == 0 else ''


@contextlib.contextmanager
def _scanner_for(routing: Optional[RoutingSettings]) -> Iterator[None]:
    """Bind the secret scanner as ``cli.load_routing`` does for the TUI:
    on when the table is present and ``secret_scan`` is set; both
    ``config.SECRET_SCAN`` and the scanner are restored afterwards. No-op
    without routing."""
    if routing is None:
        yield
        return
    prev_flag, prev_scanner = config.SECRET_SCAN, policy.scanner()
    scan = routing.present and routing.secret_scan
    config.SECRET_SCAN = scan
    policy.set_scanner(load_project_scanner() if scan else None)
    try:
        yield
    finally:
        config.SECRET_SCAN = prev_flag
        policy.set_scanner(prev_scanner)


@contextlib.contextmanager
def _judges_for(decisions: Optional[DecisionsSettings]
                ) -> Iterator[list[str]]:
    """Install the experiment's judges for the run.

    Sets ``config.DECISIONS_MODE/POINTS/ACTIVE/THRESHOLDS`` from
    ``decisions``, calls ``judges.install()`` and yields the installed
    judges as ``point=name``; afterwards the config values are restored
    and the judge registry is cleared. No-op (yields ``[]``, touches
    nothing) without a table.
    """
    if decisions is None:
        yield []
        return
    prev = (config.DECISIONS_MODE, config.DECISIONS_POINTS,
            config.DECISIONS_ACTIVE, config.DECISIONS_THRESHOLDS)
    config.DECISIONS_MODE = decisions.mode
    config.DECISIONS_POINTS = dict(decisions.points)
    config.DECISIONS_ACTIVE = dict(decisions.active)
    config.DECISIONS_THRESHOLDS = dict(decisions.thresholds)
    try:
        installed = judges.install()
        yield [f'{point}={name}' for point, name in installed.items()]
    finally:
        decision_seam.clear_judges()
        (config.DECISIONS_MODE, config.DECISIONS_POINTS,
         config.DECISIONS_ACTIVE, config.DECISIONS_THRESHOLDS) = prev


def run_suite(suite: list[Case], model_spec: Optional[str], out_root: Path,
              base_state: Optional[session.SessionState] = None,
              adapters: Optional[list[Adapter]] = None, note: str = '',
              on_result: Optional[Callable[[CaseResult], None]] = None,
              trajectory_dir: Path = DEFAULT_TRAJECTORY_DIR,
              num_ctx: int = 0, routing: Optional[RoutingSettings] = None,
              routing_name: str = '', allow_spend: bool = False,
              decisions: Optional[DecisionsSettings] = None) -> Run:
    """Run every case, save the run file and append the trajectory row.

    Synchronous; see :func:`run_case` for the loop and concurrency rules.
    Transcripts and the ledger go under ``out_root/<run_id>/``; the run
    JSON lands in ``out_root``; the row goes to
    ``trajectory_dir/TRAJECTORY.md`` (default ``evals/TRAJECTORY.md``).
    ``on_result`` runs after each case. ``num_ctx`` pins the context when
    the base state is resolved here (see :func:`resolve_base`); the run
    records the context the base state ended up with. ``routing`` (with
    ``routing_name``, the file stem recorded on the run) activates
    sub-agent routing over a registry of ``adapters`` and binds the secret
    scanner for the duration; ``allow_spend`` grants remote spend.
    ``decisions`` installs the experiment's judges for the run (see
    :func:`_judges_for`); their names land on ``Run.judges``.
    """
    _assert_no_running_loop('run_suite')
    skills.ensure_loaded()       # spawn(role=, skill=) needs the catalog
    out_root = Path(out_root)
    if adapters is None:
        adapters = bench._build_adapters()
    if base_state is None:
        base_state, model_spec = resolve_base(model_spec, adapters, num_ctx)
    elif not model_spec or model_spec == cases.DEFAULT_MODEL:
        model_spec = (f'{getattr(base_state.adapter, "name", "?")}'
                      f'|{base_state.model}')
    routed = (Routing(routing, registry_from(adapters), routing_name)
              if routing is not None else None)
    run_id = runs.new_run_id()
    ts = runs.now_ts()
    out_dir = out_root / run_id
    results: list[CaseResult] = []
    with _scanner_for(routing), _judges_for(decisions) as judge_names:
        for case in suite:
            res = run_case(case, base_state, adapters, out_dir,
                           routing=routed, allow_spend=allow_spend)
            results.append(res)
            if on_result is not None:
                on_result(res)
    run = Run(run_id=run_id, ts=ts, model=model_spec, git_sha=git_sha(),
              cases=results,
              num_ctx=base_state.num_ctx or base_state.num_ctx_override,
              routing=routing_name if routing is not None else '',
              controller=bool(routing is not None and routing.controller),
              judges=judge_names)
    runs.save(run, out_root)
    runs.append_trajectory(run, Path(trajectory_dir), note=note)
    return run
