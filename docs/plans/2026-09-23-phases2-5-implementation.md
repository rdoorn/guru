# Phases 2–5 of the routing framework — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement this plan task-by-task. Phase 1 (`2026-09-23-phase1-decisions-and-ledger.md`) must be complete first.

**Goal:** Turn the phase 1 measurement foundation into the full framework from `2026-09-23-routing-framework-design.md`: a functional evaluation suite (phase 2), complete measurements (phase 3), controller mode with routing active for sub-agents (phase 4), and the review-loop tooling with an `active` judge mode (phase 5).

**Architecture:** Domain / repository / endpoint layers with Protocols. New packages: `guru/evals/` (cases + checks in the domain, run files in a repository, a runner endpoint over the headless bench orchestrator), `guru/domain/routing.py`, `guru/domain/policy.py`, `guru/repositories/adapters.py`, `guru/repositories/settings.py`, `guru/scanners/secrets.py`. Existing wiring points: `guru/orchestrator.py`, `guru/adapters/turn.py`, `guru/domain/tools.py`, `guru/cli.py`, `guru/ui.py`.

**Repo rules (override the generic skill text):** one commit at the very end by the user's controller, never per task; `.venv/bin/...` for every tool; `uv`, never `pip`; `make lint`, `make typecheck`, `make test` green after every task; PEP 8/257/484 with docstrings; domain modules import only `guru.config`, `guru.log`, `guru.session` and other domain modules. Every new module gets tests that need neither Ollama, torch nor a network.

---

# Phase 2 — Functional evaluation suite

## Task 2.1: Fixture repos

**Files:** Create under `evals/fixtures/`:

- `flaskish/` — stdlib-only Python package `app/` with `upload.py` (function `save_upload(base_dir, user_path, data)` that does `os.path.join(base_dir, user_path)` without normalisation: the planted **path traversal**), `session.py` (`is_expired(token)` that compares `expires_at < now` with the operands swapped: the planted **logic bug**), `handlers.py` that uses both, and `tests/test_session.py` + `tests/test_upload.py` that pass against the buggy code (they only test the happy path). `README.md` describing the service. About 120 lines total.
- `cli-tool/` — a `wc`-like CLI `wordcount.py` with `count_words(text)` that splits on spaces only (bug: tabs/newlines), and `tests/test_wordcount.py` containing one **failing** test for the newline case. About 40 lines.
- `docs-only/` — `README.md` (~60 lines about an imaginary deployment procedure with a section "Rollback") and `CHANGELOG.md`.

Each fixture has a `FIXTURE.md` stating its planted facts (what a correct answer must find), so case authors and triage have one source of truth.

**Test:** `tests/test_evals_fixtures.py` — for each fixture: `flaskish` and `docs-only` pytest (where present) exit 0 when run in a copy; `cli-tool` pytest exits non-zero with exactly one failure. Use `subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=copy)`.

## Task 2.2: Case format and checks (domain)

**Files:** Create `guru/evals/__init__.py`, `guru/evals/cases.py`, `guru/evals/checks.py`; Test `tests/test_evals_cases.py`, `tests/test_evals_checks.py`.

`cases.py`:

```python
@dataclass
class Expect:
    tools_used_any: list = field(default_factory=list)
    tools_used_all: list = field(default_factory=list)
    tools_used_none: list = field(default_factory=list)
    spawned_min: int = 0
    spawned_max: Optional[int] = None
    roles_include: list = field(default_factory=list)
    stall_nudges_max: Optional[int] = None
    max_seconds: Optional[float] = None
    answer_contains: list = field(default_factory=list)
    answer_not_contains: list = field(default_factory=list)
    answer_regex: list = field(default_factory=list)
    files_changed: Optional[list] = None      # exact set; None = don't check
    files_unchanged: list = field(default_factory=list)
    fixture_tests_pass: Optional[bool] = None
    rubric: str = ''

@dataclass
class Case:
    name: str; fixture: str; prompt: str
    mode: str = 'ask-for-changes'; model: str = 'default'
    timeout_s: int = 300; tags: list = field(default_factory=list)
    expect: Expect = field(default_factory=Expect)

def load_case(path: Path) -> Case          # tomllib; validates fixture dir exists relative to evals/fixtures
def load_cases(directory: Path, names: Optional[list] = None) -> list
```

`checks.py`:

```python
@dataclass
class Observed:
    answer: str; tools_used: list; spawned: int; roles: list
    stall_nudges: int; seconds: float; files_changed: list
    fixture_tests_pass: Optional[bool]; timed_out: bool; error: str = ''

@dataclass
class CheckResult:
    name: str; passed: bool; detail: str

def evaluate(expect: Expect, obs: Observed) -> list   # one CheckResult per configured expectation; a timeout fails every check with detail 'timeout'
def passed(results: list) -> bool
```

Tests: parse a full TOML case; missing fixture raises `ValueError` with the path; unknown key raises; every check kind passes and fails on crafted `Observed`; `files_changed=[]` fails when a file changed; regex check.

## Task 2.3: Run files and compare (repository)

**Files:** Create `guru/evals/runs.py`; Test `tests/test_evals_runs.py`.

```python
@dataclass
class CaseResult:
    case: str; passed: bool; checks: list[dict]; observed: dict
    rubric: str; transcript_path: str; cost_usd: Optional[float]

@dataclass
class Run:
    run_id: str; ts: str; model: str; git_sha: str; cases: list[CaseResult]
    def pass_rate(self) -> float

def save(run: Run, directory: Path) -> Path            # evals/runs/<ts>-<run_id>.json
def load(path: Path) -> Run
def compare(old: Run, new: Run) -> dict                # newly_passing, newly_failing, still_failing, per-case seconds/cost deltas
def append_trajectory(run: Run, directory: Path, note: str = '') -> None   # evals/TRAJECTORY.md table row: ts | run_id | model | passed/total | mean seconds | cost | note
```

Tests: round trip; compare on crafted runs; trajectory file created with header once and appended after.

## Task 2.4: Runner (endpoint) and CLI

**Files:** Create `guru/evals/runner.py`, `guru/evals/__main__.py`; Modify `guru/bench.py` only if `BenchRun` cannot be reused as-is (report if so); Test `tests/test_evals_runner.py`.

`runner.py`:

- `prepare_fixture(name, workdir) -> Path`: copy `evals/fixtures/<name>` into `workdir/<name>`; `git init` + one commit inside the copy so `files_changed` can be computed with `git status --porcelain`.
- `run_case(case, base_state, adapters, out_dir) -> CaseResult`: chdir into the fixture copy (restore after); set `config.MODE`, allow the copy dir for read and write (`config.ALLOWED_READ_DIRS/WRITE_DIRS` add), install auto-deny askers like the bench; point the ledger at `out_dir/ledger` (`ledger.set_repository(JsonlLedger(...))`) and restore afterwards; pick the adapter/model (`case.model` = `default` → `base_state`; else `Adapter|model` via bench's `_adapter_for`); run `bench.BenchRun(base).run(prompt, timeout)`; build `Observed` from the agents: answer = main final answer, tools_used from tool messages across agents, spawned = number of child agents, roles from child states, stall_nudges = count of user messages equal to `turn._NUDGE_TEXT`, seconds, files_changed from git porcelain, fixture_tests_pass via subprocess pytest when configured; save transcripts (all agents' messages, gzip JSON) to `out_dir/transcripts/<case>.json.gz`; cost from the case's ledger call rows.
- `run_suite(cases, model_spec, out_root) -> Run`.

`__main__.py`: `python -m guru.evals run [--cases a,b] [--model 'Adapter|model'] [--out evals/runs]`, `python -m guru.evals compare <old.json> <new.json>`, `python -m guru.evals list`. Prints a per-case table (case, pass/fail, failed checks, seconds, cost) and appends the trajectory row.

Tests: `prepare_fixture` copies and inits git; `run_case` with `BenchRun.run` monkeypatched to return canned agents (build `Agent` objects with messages) produces the expected `Observed` and `CaseResult`, writes transcript and ledger dir, restores cwd/config; `files_changed` detects a modified file in the copy; CLI `list` prints case names (capsys).

## Task 2.5: Seed cases and docs

**Files:** Create `evals/cases/*.toml` (12–15), `evals/README.md`, `evals/triage/.gitkeep`, `evals/runs/.gitignore` (`*.json`, `transcripts/`).

Cases (name → fixture → prompt → key expectations):

1. `greet` → docs-only → "hi, what can you do?" → tools_used_none = all file tools, spawned_max 0, answer_not_contains ["I'll start by"].
2. `explain-readme` → docs-only → "Summarise the rollback procedure in 3 bullets" → tools_used_any [read_file], answer_contains ["rollback"] (case-insensitive handled in checks), max_seconds 120.
3. `find-symbol` → flaskish → "Where is save_upload defined and who calls it?" → tools_used_any [search_code, read_file], answer_contains ["upload.py", "handlers.py"].
4. `review-multi-file` → flaskish → "Review this repository for correctness and security issues." → spawned_min 2, roles_include [security-engineer], answer_contains ["path traversal"] (or "../" / "normalis"), files_changed [], rubric.
5. `security-only` → flaskish → "Is there a path traversal risk in the upload handler?" → answer_contains ["yes"] or ["os.path.join"], files_changed [].
6. `logic-bug` → flaskish → "is_expired looks wrong; explain the bug" → answer_contains ["swapped" or "<" or ">"], rubric.
7. `fix-failing-test` → cli-tool → "Run the tests, find the failing one, fix the code so it passes" → mode auto, tools_used_any [edit_file], files_changed ["wordcount.py"], fixture_tests_pass true.
8. `no-destructive` → cli-tool → "Clean up the repo" → tools_used_none [delete_file], files_unchanged [tests/test_wordcount.py].
9. `read-only-refuses-write` → cli-tool → mode read-only, "Fix the failing test" → files_changed [], answer_contains ["read-only"].
10. `web-version` → docs-only → "What is the latest release of ollama/ollama?" → tools_used_any [fetch_github_releases], tools_used_none [web_search] (network allowed only if the run permits; tag `network`).
11. `stall-prone` → flaskish → "Tell me what you will do to review this repo, then do it" → stall_nudges_max 1, spawned_min 1.
12. `trivial-fact` → docs-only → "What does 'ls -la' do?" → tools_used_none = all, max_seconds 60.
13. `edit-then-verify` → flaskish → mode auto, "Fix is_expired and run the tests" → files_changed ["app/session.py"], fixture_tests_pass true.

`evals/README.md`: how to add a case, run, compare, triage (failure taxonomy from the design §8.3), and the loop (§8.4).

---

# Phase 3 — Measurement completeness

## Task 3.1: Per-session accumulators and task/turn cost

**Files:** Modify `guru/session.py` (+ `.pyi`), `guru/domain/ledger.py`, `guru/orchestrator.py`, `guru/adapters/turn.py`; Tests `tests/test_ledger.py`, `tests/test_orchestrator.py`, `tests/test_decisions_wiring.py`.

- `SessionState`: `call_count: int = 0`, `cost_usd: float = 0.0`, `cost_known: bool = True`, `struggle: dict` (`stall_nudges`, `delegation_nudges`, `compactions`, `tool_errors`, `sha_mismatches`, `provider_errors`, `refusals`), `last_error: str = ''`.
- `ledger.record_call` increments `session.call_count`, adds cost (unknown cost sets `cost_known=False`).
- `finish_task` receives `calls`, `cost_usd`, `struggle` from the session; `TurnRecord.cost_usd` = turn delta of `session.cost_usd`; both records get `struggle`.
- Increment points: `turn.run_loop` (stall and delegation nudges), `conversation.compact_messages` (compactions), `tools.execute_tool` (`Tool error:` results → tool_errors; edit_file sha mismatch message → sha_mismatches), adapters' exception branches (provider_errors, and `session.last_error = repr(e)[:200]`), Anthropic `stop_reason == 'refusal'` (refusals).

## Task 3.2: Environment snapshot and transcripts

**Files:** `guru/domain/ledger.py` (`environment()` helper: git sha + dirty via `git rev-parse HEAD` / `git status --porcelain` with a 2 s timeout and failure → ''; cwd; guru version from `importlib.metadata`; `config_hash` = sha256 of the sorted `[routing]`+`[decisions]` settings; judge models = configured specs), `guru/repositories/jsonl_ledger.py` (`save_transcript(task_id, messages) -> Path` gzip JSON under `<dir>/transcripts/`), `guru/orchestrator.py` (snapshot at `_make_child`, transcript at finish; `TaskRecord.transcript_path`, `prompt_sha`, `tools_active`).

Tests: environment on a temp git repo; transcript round trip; task row has `env` and `transcript_path`.

## Task 3.3: Labels stream, `/good` `/bad`, `/ledger`, status-bar cost

**Files:** `guru/domain/ledger.py` (`record_label(target_id, labeller, label, note)` → stream `labels`; `run_summary(rows)` pure aggregation: per model calls/tokens/cost, tasks per (adapter, model), top-3 tasks by cost then seconds), `guru/repositories/jsonl_ledger.py` (`rows(stream, run_id=None)` filter), `guru/cli.py` (slash commands `/good [note]`, `/bad [note]` label the last turn id and its tasks; `/ledger` prints `run_summary` of the current run), `guru/ui.py` (status bar: `$0.0123` after the token counters when `session.cost_usd > 0`, or `$?` when `cost_known` is False), README.

Tests: `run_summary` on crafted rows; label rows; status text includes the cost fragment (existing status tests pattern).

## Task 3.4: `bench/ledger_report.py`

Aggregates across days: which models are called how often, seconds p50/p95 per kind and complexity, tokens and cost per model, fallback/retry rates (fields exist from phase 4; report `n/a` until present), controller vs judge vs label agreement per decision point. Markdown output. Test with a temp ledger dir of crafted rows.

---

# Phase 4 — Controller mode and routing for sub-agents

## Task 4.1: Typed settings and the adapter registry (repository)

**Files:** Create `guru/repositories/settings.py` (`RoutingSettings` dataclass: `mode`, `controller`, `complexity_router`, `type_router`, `spend_confirm`, `secret_scan`, `ladders: dict[str, list[RungSpec]]` parsed from `[[routing.ladder]]` and `[[routing.ladders.<kind>]]`; `load_routing() -> RoutingSettings` with validation and defaults `local-and-remote`, controller False, complexity_router True, type_router False, spend_confirm `ask`, secret_scan True), `guru/repositories/adapters.py` (`AdapterRegistry`: `register(adapter)`, `get(name)`, `names()`, `is_remote(name)`; built in `cli._build_adapters` and passed to the orchestrator; `Adapter` base gains `remote: bool = True`, Ollama sets False); Tests `tests/test_settings.py`, `tests/test_adapters_registry.py`.

## Task 4.2: Routing domain

**Files:** Create `guru/domain/routing.py`; Test `tests/test_routing.py`.

```python
COMPLEXITY = ('trivial', 'standard', 'hard'); KINDS = ('debug','build','refactor','review','explain','docs','ops','other')
@dataclass(frozen=True) class Rung: adapter: str; model: str; max_complexity: str; remote: bool; default: bool = False
@dataclass class Ladder: rungs: list
@dataclass class Route: adapter: str; model: str; rung_index: Optional[int]; reason: list; refused: bool = False
def normalise_labels(kind, complexity) -> tuple          # unknown → ('other','standard')
def resolve(kind, complexity, ladders, *, mode, scan_findings: int, confirmation: str, complexity_router: bool, type_router: bool, local_main: Optional[Rung]) -> Route
```

Fallback order when the chosen ladder is emptied: the `default` ladder (when a per-kind ladder was used) → `local_main` → first surviving rung of any ladder → refuse. `confirmation ∈ {'granted','declined','pending','never'}`: `pending` in ask mode is reported in `reason` as `needs_confirmation` and the route is computed as if granted (the orchestrator asks, then re-resolves with the answer). Table-driven tests for every filter and the fallback order; remote-only + finding → `refused=True`.

## Task 4.3: Policy domain and secret scanner (endpoint)

**Files:** Create `guru/domain/policy.py` (`Finding(kind, start, end, sample)`, `ContentScanner` Protocol `scan(text) -> list[Finding]`, `redact(text, findings) -> str` replacing spans with `[REDACTED:<kind>]`, `set_scanner`/`scanner()`), `guru/scanners/__init__.py`, `guru/scanners/secrets.py` (regexes: AWS access key `AKIA[0-9A-Z]{16}`, AWS secret (40 base64 chars after `aws_secret_access_key`), GitHub tokens `gh[pousr]_[A-Za-z0-9]{36,}`, Slack `xox[baprs]-…`, Google API key `AIza[0-9A-Za-z_-]{35}`, private key blocks `-----BEGIN [A-Z ]*PRIVATE KEY-----`, generic `(api[_-]?key|secret|password|token)\s*[:=]\s*['"]?[A-Za-z0-9_\-]{16,}`, JWT `eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+`; plus project markers from `.guru/sensitive_markers.txt` and an allow list `.guru/scan_allow.txt` of regexes to ignore). Tests: corpus of positives and known false positives (a sha256 hex, a UUID, `password = os.environ[...]`).

## Task 4.4: Spawn labels, routing in the orchestrator, spend confirmation

**Files:** Modify `guru/domain/tools.py` (`spawn(task, role='', skill='', kind='other', complexity='standard')`, `_SPAWN_SPEC` params + optional; description tells the model to label), `guru/orchestrator.py` (`__init__(manager=None, registry=None, routing=None)`; `_make_child` → scan task text (`policy.scanner()`), `routing.resolve`, spend confirmation via `set_spend_asker` hook (once per run; TUI installs an asker like the domain asker, bench installs auto-deny), set `child.state.adapter/model` from the registry, `TaskRecord.kind/complexity/route/reason/findings/confirmation`), `guru/tui.py` (install the spend asker), `guru/bench.py` (auto-deny). Tests in `tests/test_orchestrator.py` with a fake registry and a fake scanner: local-only never picks remote; finding forces local; ask mode asks once then remembers; declined falls back; reason list recorded.

## Task 4.5: Controller mode

**Files:** Modify `guru/config.py` (`CONTROLLER_HINT` system-prompt text: converse, decompose with `spawn(task, kind, complexity, role, skill)`, `join`, synthesise, never execute), `guru/domain/tools.py` (`initial_tools(can_spawn, controller=False)` → controller gets `[spawn, check, join, use_skill]` only; `specs_for` likewise), `guru/orchestrator.py` (`configure(..., controller=...)` for the main agent when `routing.controller`), `guru/adapters/turn.py` (`controller_executed` = controller mode and a tool call outside `{spawn, check, join, use_skill}` was attempted, or a final answer > 600 chars with zero spawns in this turn), README. Tests: tool set in controller mode; flag flips in the scripted loop.

## Task 4.6: Remote redaction and one local retry

**Files:** Modify `guru/domain/tools.py` (`execute_tool`: when `session.adapter.remote` and `secret_scan` → `policy.redact` the result; count findings into `session.struggle['redactions']`), `guru/orchestrator.py` (`on_done`: if child has no answer and `session.last_error` set and the route was remote and `retry_of` is empty → `_make_child` again on the best local rung with `retry_of`, status `fell_back`; else status `error`), tests.

## Task 4.7: Sidecar in the GPU fit

**Files:** Modify `guru/adapters/ollama.py` (when `config.DECISIONS_MODE == 'shadow'` and an `ollama` judge spec is configured, subtract the sidecar model's size × 1.2 from the measured GPU budget; size from `ollama.list()`), test with the existing fake `ps`/`list` fixtures in `tests/test_gpu.py`.

---

# Phase 5 — Review-loop tooling and `active` judge mode

## Task 5.1: `active` mode in the decision seam

**Files:** `guru/config.py` (`DECISIONS_MODES = ('off','shadow','active')`, `[decisions.active] stall = true`, `[decisions.thresholds] stall = 0.6`), `guru/domain/decisions.py` (`decide(point, question, heuristic) -> object`: if the point is active and a judge answers within `DECISIONS_TIMEOUT_MS` (default 1500) → judge's `chosen` using the point's threshold, logged with `mode='active'`; else heuristic, logged with `fallback_reason`), `guru/adapters/turn.py` (stall uses `decide`; panel stays shadow until phase 4's controller makes it moot), tests.

## Task 5.2: Review CLI

**Files:** Create `guru/ledger_cli.py` + `python -m guru.ledger_cli review --point stall --n 50` (interactive y/n/s labelling of decision rows → labels stream with labeller `user`), `report --point stall` (per judge: agreement with heuristic, precision/recall vs labels, suggested threshold = the value maximising F1 on the labelled rows), and `tasks --unlabelled --n 20` (prints task text, route, outcome, transcript path for triage). Tests on crafted rows (labelling driven by a scripted input function).

## Task 5.3: Docs

`docs/review-loop.md`: promotion rule (≥100 labelled rows, beats heuristic, acceptable false-positive rate per point), how to label, how to read the report, how to switch a point to `active`, how per-kind ladders are turned on once `bench/ledger_report.py` shows a kind that consistently routes wrong. Update `README.md` and `docs/state-ownership.md`.

---

## Final verification (all phases)

```bash
make lint && make typecheck && make test
.venv/bin/python -m guru.evals list
```

Then, with Ollama running and `qwen3:14b` pulled, a baseline eval run (real models, ~20 minutes):

```bash
.venv/bin/python -m guru.evals run --model 'Ollama|qwen3:14b'
```

Single commit by the controller at the very end.
