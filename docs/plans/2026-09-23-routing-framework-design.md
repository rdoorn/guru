# Routing framework: controller, ledger, judges, ladder — design

Status: approved in review on 2026-09-23 (sections 1–6). Phases 1–5
(`2026-09-23-phase1-decisions-and-ledger.md`,
`2026-09-23-phases2-5-implementation.md`) are implemented on branch
`feat/routing-framework`, pending final review and commit; the review-loop
procedure is `docs/review-loop.md`. Supersedes
the routing parts of `2026-09-23-system-one-decisions-design.md`, which remains
the record of the judge measurements.

## 0. Decisions taken during the design review

| # | Question | Decision |
|---|---|---|
| 1 | What does "remote" mean for data in local-and-remote mode | Policy-gated: a deterministic secret/sensitive-content scan must pass before content goes remote. Spending follows the access-mode pattern: `ask` confirms once per run, `auto` is pre-approved, a decline falls back to local. Remote-only refuses or redacts instead of falling back. |
| 2 | Where routing acts | Sub-agents only. The main agent is a **controller**: it converses, decomposes, routes and synthesises, and never executes a task itself. |
| 3 | What the controller may do itself | Conversation only: greetings, clarifying questions, synthesis of sub-agent results. Anything needing a tool or producing substantive content is a sub-agent task. |
| 4 | Cost | Price table + per-call cost in the ledger and status bar. **No cap** and no refusal on cost; insight only. LiteLLM cost header wins over the table. |
| 5 | Candidate models | A configurable **ladder** of rungs (adapter, model, max complexity). Per-kind ladders are supported in the data model but off (`type_router = false`) until measured. |
| 6 | Who labels complexity today | The controller labels (`spawn(kind=…, complexity=…)`); judges shadow. Move to independent judges when the log shows they are reliable. Every component has an enable flag and a model setting. |
| 7 | Secret found in a remote sub-agent's tool output | Redact and continue, with a typed marker and a ledger finding. |
| 8 | Code structure | Domain / repository / endpoint layers with Protocols between them; new code never imports an endpoint from the domain. |
| 9 | Plan order | Ledger core folded into phase 1 so shadow rows are joinable from day one. |
| 10 | Who runs the work (2026-09-23, after the local-worker runs) | **Claude is the controller and every worker.** The ladder's rungs are Claude tiers (Haiku for trivial, Sonnet for standard, Opus for hard). Local models act only as small judges (encoders) on the decision seam; the local LLM workers and the local controller were dropped after the 2026-09-23 runs (`evals/triage/1f4f8262a80a.md`: timeouts, hallucinated syntheses, a one-file fix nudged into a panel). The sidecar LLM judge is optional and off by default. |

## 1. Components and layers

Every component is a Protocol in the domain layer, one or more
implementations in a repository or endpoint layer, and an enable flag plus
model setting in the `[routing]` table. Disabling a component binds its
Protocol to a no-op implementation; the rest of the code never checks flags.

**Domain (`guru/domain/`)** — pure rules and entities, no I/O

- `decisions.py` — Question, Answer, Judge protocol, shadow logging (phase 1 plan).
- `ledger.py` — `CallRecord`, `TaskRecord`, `TurnRecord`, `LedgerRepository` protocol, `record_*()` functions, `RUN_ID`.
- `pricing.py` — default price table, settings override, `cost_usd(model, usage)`.
- `routing.py` — `Rung`, `Ladder`, `WorkingMode`, `Route`, `resolve(...)`.
- `policy.py` — `Finding`, `ContentScanner` protocol, `redact()`; rule "any finding forces local".

**Repository (`guru/repositories/`)** — persistence only

- `jsonl_ledger.py` — append-only JSONL under `~/.guru/ledger/`, one file per day per stream (`calls`, `tasks`, `turns`, `decisions`, `labels`), compressed transcripts under `transcripts/<task_id>.json.gz`.
- `settings.py` — typed loaders for `[routing]`, `[pricing]`, `[decisions]` (new tables; existing `config._apply_settings` stays for the old ones).
- `adapters.py` — the adapter registry (name → Adapter), moved out of the CLI so the orchestrator can resolve a route.

**Endpoint** — external I/O

- `guru/adapters/*` — unchanged role; each emits one `CallRecord` per provider call.
- `guru/judges/*` — sidecar JSON judge, encoder judges (phase 1 plan).
- `guru/scanners/secrets.py` — regex `ContentScanner` (keys, tokens, private-key blocks, project marker list).
- `guru/orchestrator.py` — the only caller of `routing.resolve`, in `spawn` / `spawn_panel`.

Config sketch:

```toml
[routing]
mode = "local-and-remote"        # local-only | local-and-remote | remote-only
controller = true                # main agent is a controller: spawn/check/join only
complexity_router = true         # off = every task goes to the ladder's default rung
type_router = false              # per-kind ladders; off until measured
spend_confirm = "ask"            # ask (once per run) | auto | never
secret_scan = true

[[routing.ladder]]               # default ladder, lowest rung first:
adapter = "SBP Litellm"          # the Claude tiers behind a LiteLLM adapter
model = "aws/claude-4-5-haiku"
max_complexity = "trivial"

[[routing.ladder]]
adapter = "SBP Litellm"
model = "aws/claude-5-sonnet"
max_complexity = "standard"
default = true

[[routing.ladder]]
adapter = "SBP Litellm"
model = "aws/claude-5-5-opus"
max_complexity = "hard"

# [[routing.ladders.review]]  ... per-kind ladders: plural key, same rung shape

[pricing."claude-sonnet-5"]      # overrides the bundled table
input_per_m = 2.0
output_per_m = 10.0
```

### Default price table (USD per million tokens; verified on the live pricing page 2026-09-23)

| Model ID | Input | Cache write 5m | Cache write 1h | Cache read | Output |
|---|---|---|---|---|---|
| claude-fable-5-1 | 10.00 | 12.50 | 20.00 | 0.25 | 50.00 |
| claude-fable-5 | 10.00 | 12.50 | 20.00 | 1.00 | 50.00 |
| claude-opus-5-5 | 4.00 | 5.00 | 8.00 | 0.20 | 20.00 |
| claude-opus-5 | 5.00 | 6.25 | 10.00 | 0.50 | 25.00 |
| claude-opus-4-8 | 5.00 | 6.25 | 10.00 | 0.50 | 25.00 |
| claude-sonnet-5 | 2.00 | 2.50 | 4.00 | 0.20 | 10.00 |
| claude-haiku-4-5 | 1.00 | 1.25 | 2.00 | 0.10 | 5.00 |

Matching: exact model ID first, then the longest table key contained in the
model ID (so `anthropic/claude-sonnet-5` via LiteLLM resolves). Unknown
remote models cost `null` (unknown), local Ollama models cost 0. Cache read
and cache write tokens are recorded separately because they are priced
differently. A LiteLLM `x-litellm-response-cost` header, when present, wins.

## 2. Data flow of one request (controller mode, local-and-remote)

1. The user message reaches the controller. Tool set: `spawn`, `check`,
   `join`. System prompt: converse, decompose, synthesise, never execute.
   `spawn` gains `kind` and `complexity` arguments the controller must fill.
2. The turn loop runs the controller; shadow judges log as in phase 1. The
   controller replies conversationally or calls `spawn` one or more times.
3. `Orchestrator.spawn` creates a `TaskRecord` (task text, labels, parent,
   environment snapshot) and runs the scanner over the task text. A finding
   forces the local ladder for this task.
4. `routing.resolve` picks the lowest rung of the kind's ladder that covers
   the complexity and survives the mode, scan and confirmation filters. A
   remote pick in `ask` mode triggers the once-per-run question; a decline
   falls back to the best local rung and is recorded on the task.
5. The child agent is configured with the route's adapter and model (via the
   adapter registry).
6. The child runs. Every provider call emits a `CallRecord`. When the child's
   adapter is remote, tool results pass through the scanner; findings are
   **redacted** with a typed marker and logged; the task continues.
7. On completion the `TaskRecord` gets its outcome and struggle signals; the
   transcript is saved compressed; the result reaches the controller through
   the existing mailbox; the controller synthesises. A `TurnRecord` closes
   the user turn with total cost and time.

## 3. Ledger

Append-only JSONL streams under `~/.guru/ledger/`, one file per day per
stream; every row carries `project`, `run_id`, `ts`. Nothing is aggregated
in the hot path.

**TurnRecord** (one per user message to the main/controller agent): request
text in full, controller model, tasks spawned, `controller_executed` flag,
seconds to final answer, tools used, tokens, total cost of the turn.

**TaskRecord** (one per spawned sub-agent):

- identity: `task_id`, `parent`, `turn_id`, `retry_of`, `siblings`
- input: task text (full), `text_sha`, controller `kind`/`complexity`, judge labels (shadow), role, skill, active tool set, system prompt hash
- route: adapter, model, rung, `reason[]` (every filter that changed the outcome), scan findings, confirmation outcome
- environment: git sha, git dirty, cwd, guru version, routing config hash, judge models
- outcome: status (done/error/cancelled/fell_back), seconds, calls, tokens in/out/cache, cost, tools used, answer length, transcript path
- struggle: stall nudges, delegation nudges, compactions, tool errors, sha mismatches, provider errors, refusal stop reasons
- `outcome_label`: empty; verdicts live in the labels stream

**CallRecord** (one per provider call): agent, `task_id`, `turn_id`, adapter,
model, tokens in/out/cache read/cache write, seconds, `load_s`, `prefill_s`,
`generate_s`, `ttft_s` where the provider reports them, `round` (tool /
final), cost, cost source (table / header / local).

**Labels stream** (`labels.jsonl`): `task_id`, labeller (user / judge:<name>
/ review), label, note, ts. Fed by `/good`, `/bad` and review passes. The
ledger itself is never edited.

**Transcripts**: full sub-agent conversation, gzip JSON, keyed by `task_id`.
Required for re-grading and for offline replay on another rung.

Reading: status bar shows run cost; `/ledger` prints the current run (calls
and cost per model, tasks per rung, three most expensive tasks);
`bench/ledger_report.py` aggregates across days and compares controller
labels, judge labels and outcome labels.

Questions the data must answer: cost and p95 time per completed task per
rung and kind; fallback and retry rates; controller vs judge agreement and
each vs outcome labels; the right default rung per kind; which judge is ready
to take over labelling.

Left out deliberately: sampled shadow execution on a second rung (replay from
transcripts + environment snapshot gives the counterfactual later at zero
live cost).

## 4. Routing policy

`resolve(task, ladders, mode, scan, confirmation) -> Route` is a pure
function.

- Labels: `kind ∈ {debug, build, refactor, review, explain, docs, ops,
  other}`, `complexity ∈ {trivial, standard, hard}`; missing → `other`,
  `standard`.
- Ladders: `default` plus optional per-kind; only `default` is used while
  `type_router = false`.
- Filters: local-only strips remote rungs; remote-only strips local rungs; a
  scan finding strips remote; in `ask` mode the first remote pick asks once,
  a decline strips remote for the run.
- Choice: with `complexity_router = true`, the lowest surviving rung whose
  `max_complexity ≥ complexity`; otherwise the ladder's `default` rung.
- Fallback order: local main model as configured → first surviving rung of
  any ladder → refuse with reason (only remote-only + scan finding can
  refuse).
- `Route.reason` lists every filter that changed the outcome; it is written
  to the task record verbatim.

Non-goals now: learned thresholds, probability gating, per-task cost
estimates. Memory: with Claude tiers as rungs (decision 10) nothing on the
ladder runs locally; the only local load is the encoder judges (small).
The sidecar LLM judge (`ollama` spec under `[decisions.points]`) is
optional and off by default; when it is configured the GPU fit reserves
its size before the main model's context is fitted, and a local rung that
does not fit is a runtime warning.

## 5. Error handling and fallbacks

- Judges and ledger writes run on the background worker; failures log and
  never block. Unwritable ledger → disabled for the run with one notice.
- Remote provider failure in a task: SDK retries; on final failure the task
  is respawned **once** on the best local rung with `retry_of`; a second
  failure is delivered to the controller as an error result.
- Missing rung model: ladder validated at startup against the registry;
  invalid rungs dropped with a warning.
- Scanner false positives: typed marker + ledger finding; per-project
  pattern allow list.
- Spend confirmation uses the existing asker hook; a dismissed prompt is a
  decline.
- Controller that executes anyway: existing nudge stays; `controller_executed`
  is measured, not punished.
- Cancellation propagates as today; task status `cancelled` with cost so far.

## 6. Testing

- Domain: pure unit tests; table-driven `resolve`; exact-cent pricing incl.
  cache tokens and header override; scanner corpus with known false positives.
- Repository: JSONL against `tmp_path`, corrupt lines, daily rollover,
  transcript round trip.
- Endpoint: fake clients (as today); each provider call emits exactly one
  `CallRecord`.
- Orchestrator: fake registry; child gets the resolved adapter/model; task
  record carries the route reason.
- Controller mode: scripted turn; tool set is spawn/check/join; flag flips
  when the controller executes.
- No test needs Ollama, torch or network; the bench gains `--route` later.

## 7. Phases

1. **Seam + ledger core, shadow mode** (`2026-09-23-phase1-decisions-and-ledger.md`): decision seam, judges behind the extra, pricing table, ledger entities + JSONL repository, per-call hook in all adapters, basic Turn/Task records with join keys, shadow wiring (stall, panel, injection). No behaviour change.
2. **Functional evaluation suite** (§8): fixture repo, case format, headless runner over the orchestrator, run files, `compare`. Seed with 10–15 cases covering conversation, single-file read, multi-file review, edit-with-tests, web lookup, and the delegation and stall behaviours. Establish the baseline run before any other change.
3. **Measurement completeness**: full Task/Turn fields, transcripts, environment snapshot, timing decomposition, struggle signals, labels stream + `/good` `/bad`, `/ledger`, run cost in the status bar, `bench/ledger_report.py`. Validated with the eval suite.
4. **Controller + routing active for sub-agents**: controller mode, `spawn(kind, complexity)`, adapter registry, ladder config, `routing.resolve`, working mode, spend confirmation, scanner + redaction, GPU-fit accounting for the sidecar. Each step is an eval iteration (§8.4).
5. **Review loop**: label the first 100–200 tasks; compare controller vs judges vs outcomes; promote judges per decision point; per-kind ladders if the data supports it; optional encoder fine-tuning on the labels.
6. **Adjacent**: symbol/AST tools (backlog) before any embedding retrieval; embeddings hybrid for `search_tools` when the registry grows.

## 8. Functional evaluation suite and the improvement loop

Decisions (2026-09-23): cases run against a **frozen fixture repo** checked
into `evals/fixtures/`, copied per run; **rubric cases are graded by the
reviewing assistant reading transcripts during triage**, not by a model, so
the harness has no judge dependency.

### 8.1 Cases

`evals/cases/<name>.toml`, one prompt each:

```toml
name = "review-multi-file"
fixture = "flaskish"                 # evals/fixtures/flaskish/
prompt = "Review this repository for correctness and security issues."
mode = "ask-for-changes"             # access mode for the run
model = "default"                    # or a specific adapter|model
timeout_s = 300

[expect.behaviour]                   # deterministic, about guru's choices
tools_used_any = ["read_file", "search_code"]
tools_used_none = ["delete_file"]
spawned_min = 2                      # delegation happened
roles_include = ["security-engineer"]
stall_nudges_max = 0
max_seconds = 240

[expect.content]                     # deterministic, about the answer/repo
answer_contains = ["path traversal"]
answer_not_contains = ["I'll start by"]
files_changed = []                   # review must not edit
fixture_tests_pass = true            # run the fixture's own pytest after

[expect.rubric]                      # manual grade 0–2 during triage
text = "Names the unchecked user path in upload.py and the missing CSRF check."
```

Fixtures are small, self-contained repos with their own tests (a Flask-like
service with one planted security bug and one planted logic bug; a CLI tool
with a failing test; a docs-only repo). They never change except by a
deliberate commit, so runs are comparable across weeks.

### 8.2 Runner

`guru/evals/` (domain: `cases.py` parsing + `checks.py` pure assertions;
repository: `runs.py` run files under `evals/runs/<ts>.json`; endpoint: the
headless orchestrator from `guru.bench`, generalised). Per case: copy the
fixture to a temp dir, set cwd and allow-lists, run the prompt through the
orchestrator (controller mode when enabled), collect the final answer, the
ledger rows for that `run_id`, the transcripts and the fixture diff; evaluate
behaviour and content checks; store rubric text + transcript path for triage.
`python -m guru.evals run [--cases a,b] [--model ...]` and
`python -m guru.evals compare <run1> <run2>` (newly passing / failing, time
and cost deltas per case, trajectory table appended to `evals/TRAJECTORY.md`).

### 8.3 Failure taxonomy (used in triage)

`did_not_delegate`, `wrong_tool`, `stalled`, `over_read` (read many files
where a search would do), `wrong_route` (phase 4+), `wrong_answer`,
`unsafe_edit` (changed a file it should not), `timeout`, `model_limit` (the
model cannot do it; not a guru bug). Every failing case gets exactly one tag
in the triage notes so fixes can be traced to causes.

### 8.4 The loop

1. Run the suite; record the run.
2. Triage: the reviewing assistant reads failures and rubric transcripts,
   tags causes, grades rubrics, writes `evals/triage/<run>.md` with proposed
   fixes (prompt text, config default, code) and the cases each fix targets.
3. The user approves the fixes; they are applied.
4. Rerun and `compare`; the trajectory table gains a row.
5. Stop when the pass rate plateaus or the remaining failures are
   `model_limit`. Then the suite is the regression gate for the next phase.

Guardrails: never edit a case to make it pass without a note in the triage
file; add a new case for every bug found in real use (the ledger's turn
records are the source of new prompts); keep the suite short enough
(`--tags fast` for the gate, the full suite under ~20 minutes) so it is run
often.

### 8.5 Experiment matrix (Claude-only, decision 10)

Files under `evals/routing/`; commands in `evals/routing/README.md`. Every
run uses `--model 'SBP Litellm|aws/claude-5-sonnet'` and `--allow-spend`.

| config | main agent | workers | file | measures |
|---|---|---|---|---|
| 0 | plain Sonnet, no routing | Sonnet (inherited) | none | baseline pass rate, time, cost |
| 1 | Sonnet controller | Claude tiers by complexity (Haiku / Sonnet / Opus) | `claude-tiers.toml` | does controller + ladder keep the pass rate and cut cost |
| 2 | Sonnet controller | as 1 | `claude-tiers-judges.toml` | as 1 plus shadow `panel` (encoder) and `injection` judges: data for the review loop, no behaviour change |

The runner installs config 2's judges for the run only (`[decisions]` in
the experiment file, `judges.install()`, restored afterwards) and records
the installed judges on the run file. The earlier matrix (local 8B
workers, local controller) is superseded by decision 10 and kept only in
the triage notes.
