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

Rubric grading: ``run_suite(rubric_spec='Adapter|model')`` grades every case
that has an ``[expect.rubric]`` with :mod:`guru.evals.rubric` after its
deterministic checks (:func:`grade_case`): the grade lands on the result
(``rubric_score``, ``rubric_reason``), in the run file and as a ``labels``
row of the run's ledger (``target_id = <run_id>:<case>``, labeller
``rubric:<model>``). A grade never fails the case by itself; ``rubric_min``
does (a score below it, or no grade, adds a failing ``rubric_min`` check).
The grading call's own cost goes to the run's ledger, not to the case's
``cost_usd``. :func:`default_rubric_spec` names the routing file's
cheapest rung for the CLI's default.

Sandbox cases (``sandbox = true``): the copy is made at a stable path
(``<tmp>/guru-eval-sandbox/<fixture>``, so the image record and tag are
reused across runs while the lockfile is unchanged) and provisioned with
``provision.provision`` before the prompt runs — ``pypi.org`` and
``files.pythonhosted.org`` are allowed for the case (the runner's own
build, not a model escalation), the build clock lands in
``observed.sandbox`` and never in the case seconds. Without Colima the
case is skipped with error ``sandbox unavailable`` and fails its checks.
``config.PROJECT_GURU_DIR`` points at the copy for every case, so the
``sandbox_*`` verbs (and anything else that resolves "the project" from
it) see the copy, never the developer's checkout. The approval asker of
``sandbox_submit`` (``provision.set_approve_asker``) denies unless
``allow_spend``, and then grants an ``intended`` verdict only — an
``unclear`` one is what a human would have to read, so unattended it is a
decline. The ``gate`` reviewer is the default one (``judges.set_registry``
over the suite's adapters and the routing file: the ladder's ``standard``
rung, else the session model); the verdicts of the case's submits are
read back from its ``sandbox_events`` rows (``observed.gate_verdicts``).

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
from guru.domain import files, gate, ledger, policy, spend, tools
from guru.domain import routing as routing_domain
from guru.evals import cases, checks, rubric, runs
from guru.evals.cases import Case, GitFixture
from guru.evals.checks import Observed
from guru.evals.runs import CaseResult, Run
from guru.repositories import settings as routing_settings
from guru.repositories.adapters import AdapterRegistry, registry_from
from guru.repositories.jsonl_ledger import JsonlLedger
from guru.repositories.settings import DecisionsSettings, RoutingSettings
from guru.sandbox import colima, provision, verbs
from guru.scanners.secrets import load_project_scanner

_IGNORE_PATTERNS = ('__pycache__', '.pytest_cache', '*.pyc', '.git')
_COPY_IGNORE = shutil.ignore_patterns(*_IGNORE_PATTERNS)
_GIT_EXCLUDE = '__pycache__/\n.pytest_cache/\n*.pyc\n'
_GIT_IDENTITY = ['-c', 'user.name=evals', '-c', 'user.email=evals@local',
                 '-c', 'commit.gpgsign=false']
FIXTURE_PYTEST_TIMEOUT_S = 300
WORKER_DRAIN_S = 30.0          # how long to wait for leftover worker threads
JUDGE_WARM_UP_S = 30.0         # model loading before the first case
_WORKER_POLL_S = 0.2
_PERSISTERS = ('persist_read_dir', 'persist_write_dir', 'persist_domain')
DEFAULT_TRAJECTORY_DIR = cases.REPO_ROOT / 'evals'   # TRAJECTORY.md
SANDBOX_UNAVAILABLE = 'sandbox unavailable'
SANDBOX_WORKDIR = 'guru-eval-sandbox'   # under the temp dir; stable path


def _deny(question: str) -> bool:
    return False


def _grant(question: str) -> bool:
    return True


def _grant_intended(question: str) -> bool:
    """The sandbox approval asker of an ``allow_spend`` run: yes to a
    ``sandbox_submit`` whose verdict is ``intended`` (what auto mode
    applies silently in the TUI), no to ``unclear`` and to anything else
    (dependency approvals) — nobody is there to read the reasons."""
    return verbs.question_verdict(question) == gate.INTENDED


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
    (``mode`` plus ``points``/``active``/``thresholds`` and
    ``labels_margin``). None when the
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


def _workdir(case: Case) -> Path:
    """A fresh temp dir for the case's copy. A sandbox case uses the
    stable ``<tmp>/guru-eval-sandbox`` (emptied first): the sandbox keys
    its image record and tag on the resolved project path, so a stable
    path means one image per fixture, rebuilt only when the lockfile
    changes."""
    if not case.sandbox:
        return Path(tempfile.mkdtemp(prefix=f'guru-eval-{case.name}-'))
    workdir = Path(tempfile.gettempdir()) / SANDBOX_WORKDIR
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    return workdir


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
    has no venv of its own) shadows any installed one, and bytecode writing
    is off so edits between two runs are never masked by a cached .pyc."""
    env = dict(os.environ)
    prev = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(repo) + (os.pathsep + prev if prev else '')
    # No bytecode in the copy: a stale .pyc (same size and mtime second as
    # an edited source) would make the fixture's tests report the old code.
    env['PYTHONDONTWRITEBYTECODE'] = '1'
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
    """cwd, project dir, access mode, allow-lists, askers, persistence and
    ledger for one case; all restored afterwards (askers excepted when
    workers leaked). The spend asker denies unless ``allow_spend``; the
    sandbox approval asker denies unless ``allow_spend``, and then grants
    ``intended`` submits only (:func:`_grant_intended`)."""
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
    prev_approve = provision._asker
    prev_repo = ledger.repository()
    prev_project = (config.PROJECT_GURU_DIR, config.SANDBOX_POLICY_PATH)
    key = str(copy.resolve())
    state = _Sandbox()
    try:                       # every mutation below is undone by finally
        os.chdir(copy)
        # "The project" is the copy: the sandbox verbs resolve it from here.
        config.PROJECT_GURU_DIR = copy.resolve() / '.guru'
        config.SANDBOX_POLICY_PATH = config.PROJECT_GURU_DIR / 'sandbox.toml'
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
        provision.set_approve_asker(_grant_intended if allow_spend
                                    else _deny)
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
            provision.set_approve_asker(prev_approve)
        config.PROJECT_GURU_DIR, config.SANDBOX_POLICY_PATH = prev_project
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


def provision_sandbox(copy: Path) -> dict:
    """Provision ``copy`` as a sandbox project (inside :func:`_sandbox`).

    ``RuntimeError(SANDBOX_UNAVAILABLE)`` when ``docker info`` fails (no
    Colima); the provisioning domains are allowed for the case so the
    build can reach the index through the proxy (restored with the
    allow-lists); ``ProvisionError``/``ValueError`` from
    ``provision.provision`` propagate. Returns ``{'image', 'digest',
    'build_seconds'}`` (``build_seconds`` is near 0 when the recorded
    image was current and nothing was built).
    """
    if not colima.available(copy):
        raise RuntimeError(SANDBOX_UNAVAILABLE)
    config.ALLOWED_DOMAINS.update(provision.REQUIRED_DOMAINS)
    t0 = time.monotonic()
    rec = provision.provision(copy)
    return {'image': rec.tag, 'digest': rec.digest,
            'build_seconds': round(time.monotonic() - t0, 3)}


def _execute(case: Case, copy: Path, base_state: session.SessionState,
             adapters: list[Adapter], repo: JsonlLedger,
             routing: Optional[Routing] = None, allow_spend: bool = False
             ) -> tuple[list, float, str, Optional[dict]]:
    """Run the prompt inside the sandbox.

    Returns ``(agents, seconds, error, sandbox)``; the clock covers only
    the model run (not the fixture copy, nor the sandbox provisioning
    that a ``case.sandbox`` case does first — ``sandbox`` is its
    :func:`provision_sandbox` record, None otherwise). ``error`` is set
    when workers were still running after the bounded drain. ``routing``
    makes the bench route sub-agents (inert when None).
    """
    registry = routing.registry if routing is not None else None
    settings = routing.settings if routing is not None else None
    with _sandbox(copy, case.mode, repo, allow_spend) as box:
        info = provision_sandbox(copy) if case.sandbox else None
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
        return agents, seconds, error, info


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
             error: str, gate_verdicts: Optional[list[str]] = None,
             sandbox: Optional[dict] = None) -> Observed:
    answer = bench._final_answer(agents[0]) if agents else ''
    skipped = error == SANDBOX_UNAVAILABLE
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
        timed_out=timed_out, error=error,
        gate_verdicts=list(gate_verdicts or []), sandbox=sandbox,
        skipped=skipped)


def gate_verdicts(rows: list[dict]) -> list[str]:
    """The gate states of the ``submit`` rows among ``sandbox_events``
    rows, in order (``detail`` is ``'<state>: reasons'``)."""
    out: list[str] = []
    for r in rows:
        if r.get('kind') != 'submit':
            continue
        state = str(r.get('detail') or '').partition(':')[0].strip()
        if state in gate.STATES:
            out.append(state)
    return out


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
    ``observed.fixture_git``. A ``sandbox`` case is provisioned first
    (:func:`provision_sandbox`; ``observed.sandbox``), its gate verdicts
    come from the ``sandbox_events`` rows it appended
    (``observed.gate_verdicts``), and without Colima it is skipped with
    error ``sandbox unavailable`` (``observed.skipped``) and fails.
    """
    _assert_no_running_loop('run_case')
    out_dir = Path(out_dir)
    repo = JsonlLedger(out_dir / 'ledger')
    repo.dir.mkdir(parents=True, exist_ok=True)
    rows_before = len(repo.rows('calls'))
    tasks_before = len(repo.rows('tasks'))
    events_before = len(repo.rows('sandbox_events'))
    workdir = _workdir(case)
    agents: list = []
    error = ''
    seconds = 0.0
    changed: list[str] = []
    tests_pass: Optional[bool] = None
    sandbox: Optional[dict] = None
    try:
        copy = prepare_fixture(case.fixture_git or case.fixture, workdir,
                               fixtures_dir)
        try:
            agents, seconds, error, sandbox = _execute(
                case, copy, base_state, adapters, repo, routing,
                allow_spend)
        except Exception as e:                       # noqa: BLE001
            error = str(e) or type(e).__name__
        finally:
            if case.sandbox:
                verbs.cleanup_all()      # the verbs' task copies
        changed = files_changed(copy)
        if case.expect.fixture_tests_pass is not None:
            tests_pass = fixture_tests_pass(copy)
    except Exception as e:                           # noqa: BLE001
        error = error or str(e) or type(e).__name__
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    timed_out = bool(case.timeout_s) and seconds >= case.timeout_s
    verdicts = gate_verdicts(repo.rows('sandbox_events')[events_before:])
    obs = _observe(agents, seconds, timed_out, changed, tests_pass, error,
                   gate_verdicts=verdicts, sandbox=sandbox)
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


# --- rubric grading ---------------------------------------------------------

def default_rubric_spec(routing: RoutingSettings) -> str:
    """The ``Adapter|model`` of the routing file's cheapest rung: the
    lowest rung of the ``default`` ladder, else of the first ladder in the
    table; ``''`` without any rung."""
    specs = routing.ladders.get(routing_domain.DEFAULT_LADDER)
    if not specs:
        specs = next((v for v in routing.ladders.values() if v), None)
    if not specs:
        return ''
    return f'{specs[0].adapter}|{specs[0].model}'


def _record_grade(repo: JsonlLedger, target_id: str, labeller: str,
                  grade: rubric.Grade) -> None:
    """One ``labels`` row for a grade in the run's ledger (the ledger is
    enabled and pointed at ``repo`` for the write, then restored)."""
    prev_repo, prev_enabled = ledger.repository(), config.LEDGER_ENABLED
    ledger.set_repository(repo)
    config.LEDGER_ENABLED = True
    try:
        ledger.record_label(target_id, labeller, str(grade.score),
                            note=grade.reason)
        ledger.flush()
    finally:
        ledger.set_repository(prev_repo)
        config.LEDGER_ENABLED = prev_enabled


def grade_case(case: Case, res: CaseResult, judge: rubric.Judge,
               repo: JsonlLedger, target_id: str,
               rubric_min: Optional[int] = None) -> None:
    """Grade ``res`` against the case's rubric and record it; never raises.

    A case without a rubric is left alone. An empty answer scores 0
    without asking the judge. The grade goes to ``res.rubric_score`` /
    ``res.rubric_reason`` and to a ``labels`` row in ``repo`` (target
    ``target_id``, labeller ``rubric:<model>``); a provider error or an
    unparsable reply leaves the score None with ``error: ...`` as the
    reason (logged, no label). The grade does not touch ``res.passed``
    unless ``rubric_min`` is set: then a score below it — or no grade —
    appends a failing ``rubric_min`` check and fails the case, and a
    sufficient one appends a passing check.
    """
    if not case.expect.rubric:
        return
    answer = str(res.observed.get('answer') or '')
    grade: Optional[rubric.Grade] = None
    try:
        if not answer.strip():
            grade = rubric.Grade(0, 'empty answer')
        else:
            grade = rubric.grade(case.prompt, case.expect.rubric, answer,
                                 judge)
    except Exception as e:                           # noqa: BLE001
        res.rubric_reason = f'error: {e}'
        log.warning('evals: rubric grading of %s failed: %s', case.name, e)
    if grade is not None:
        res.rubric_score, res.rubric_reason = grade.score, grade.reason
        _record_grade(repo, target_id, rubric.labeller(judge), grade)
    if rubric_min is None:
        return
    ok = res.rubric_score is not None and res.rubric_score >= rubric_min
    if ok:
        detail = ''
    elif res.rubric_score is None:
        detail = f'not graded ({res.rubric_reason or "no grade"})'
    else:
        detail = f'rubric {res.rubric_score} < {rubric_min}'
    res.checks.append(asdict(checks.CheckResult('rubric_min', ok, detail)))
    res.passed = res.passed and ok


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

    Sets ``config.DECISIONS_MODE/POINTS/ACTIVE/THRESHOLDS`` and
    ``config.DECISIONS_LABELS_MARGIN`` from ``decisions``, calls
    ``judges.install()``, warms the judges synchronously (so the first
    active decision is not spent loading a model) and yields the installed
    judges as ``point=name``; afterwards the config values are restored
    and the judge registry is cleared. No-op (yields ``[]``, touches
    nothing) without a table.
    """
    if decisions is None:
        yield []
        return
    prev = (config.DECISIONS_MODE, config.DECISIONS_POINTS,
            config.DECISIONS_ACTIVE, config.DECISIONS_THRESHOLDS,
            config.DECISIONS_LABELS_MARGIN)
    config.DECISIONS_MODE = decisions.mode
    config.DECISIONS_POINTS = dict(decisions.points)
    config.DECISIONS_ACTIVE = dict(decisions.active)
    config.DECISIONS_THRESHOLDS = dict(decisions.thresholds)
    config.DECISIONS_LABELS_MARGIN = decisions.labels_margin
    try:
        installed = judges.install(warm=False)
        if installed:
            judges.warm_up_all(timeout_s=JUDGE_WARM_UP_S)
        yield [f'{point}={name}' for point, name in installed.items()]
    finally:
        decision_seam.clear_judges()
        (config.DECISIONS_MODE, config.DECISIONS_POINTS,
         config.DECISIONS_ACTIVE, config.DECISIONS_THRESHOLDS,
         config.DECISIONS_LABELS_MARGIN) = prev


def run_suite(suite: list[Case], model_spec: Optional[str], out_root: Path,
              base_state: Optional[session.SessionState] = None,
              adapters: Optional[list[Adapter]] = None, note: str = '',
              on_result: Optional[Callable[[CaseResult], None]] = None,
              trajectory_dir: Path = DEFAULT_TRAJECTORY_DIR,
              num_ctx: int = 0, routing: Optional[RoutingSettings] = None,
              routing_name: str = '', allow_spend: bool = False,
              decisions: Optional[DecisionsSettings] = None,
              rubric_spec: str = '',
              rubric_min: Optional[int] = None) -> Run:
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
    :func:`_judges_for`); their names land on ``Run.judges``. The adapter
    registry (and the routing settings, empty without a file) are installed
    for the sandbox gate's default reviewer (``judges.set_registry``) and
    cleared afterwards. ``rubric_spec`` (``Adapter|model``, resolved
    through that registry; ``ValueError`` for an unknown adapter) grades
    every rubric case with :func:`grade_case` before ``on_result`` sees
    it, ``rubric_min`` making a low grade fail the case; the spec lands on
    ``Run.rubric``.
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
    registry = registry_from(adapters)
    routed = (Routing(routing, registry, routing_name)
              if routing is not None else None)
    run_id = runs.new_run_id()
    ts = runs.now_ts()
    out_dir = out_root / run_id
    results: list[CaseResult] = []
    # The gate reviewer resolves adapters and ladders here: the routing
    # file's standard rung, or (no file: empty settings) the session model.
    judges.set_registry(registry, routing if routing is not None
                        else RoutingSettings())
    try:
        judge: Optional[rubric.LLMJudge] = None
        if rubric_spec:
            judge = rubric.judge_from_spec(rubric_spec)
            if judge is None:
                raise ValueError(f'rubric judge {rubric_spec!r}: not '
                                 "'Adapter|model' or the adapter is not "
                                 'configured')
        with _scanner_for(routing), _judges_for(decisions) as judge_names:
            for case in suite:
                res = run_case(case, base_state, adapters, out_dir,
                               routing=routed, allow_spend=allow_spend)
                if judge is not None:
                    grade_case(case, res, judge,
                               JsonlLedger(out_dir / 'ledger'),
                               f'{run_id}:{case.name}', rubric_min)
                results.append(res)
                if on_result is not None:
                    on_result(res)
    finally:
        judges.set_registry(None)
    run = Run(run_id=run_id, ts=ts, model=model_spec, git_sha=git_sha(),
              cases=results,
              num_ctx=base_state.num_ctx or base_state.num_ctx_override,
              routing=routing_name if routing is not None else '',
              controller=bool(routing is not None and routing.controller),
              judges=judge_names, rubric=rubric_spec if judge else '')
    runs.save(run, out_root)
    runs.append_trajectory(run, Path(trajectory_dir), note=note)
    return run
