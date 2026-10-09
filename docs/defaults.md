# Defaults audit

The rule (Ronald, 2026-09-25): *we should only use parameters to disable or
tweak things; the defaults should be clear.* Every user-facing knob added by
the routing, decisions, ledger, tools, sandbox and evals work is listed here
with its default and a class:

- **(a) enables** — the knob had to be set to get the intended behaviour.
  A violation: the default is now on and the knob is the off switch.
- **(b) disables** — an off switch for something that is on by default.
- **(c) tweaks** — a number or a choice; fine while the default is stated.
- **(d) redundant** — nothing read it; removed.

Counts: 3 × (a) fixed, 1 × (d) removed, everything else (b) or (c) with its
default stated below. Effective defaults, not just code defaults: where guru
writes a block for you (the `[routing]` table) the written value is the one
that counts and is named as such.

## `~/.guru/settings.toml`

| Knob | Where | Default | Purpose | Class | Action |
|------|-------|---------|---------|-------|--------|
| `[routing]` table | settings.toml | **written at startup** when an enabled `litellm`/`anthropic` adapter exists and no table is present (the measured Claude-tier block); never rewritten | route sub-agent tasks over Claude tiers | b | keep; `/routing off` (or `mode = "off"`) is the off switch |
| `mode` | `[routing]` | `local-and-remote` | which rungs may be picked; `off` disables the table | c | keep |
| `controller` | `[routing]` | retired: accepted and ignored (the main agent is always the lead) | — | **d → removed** | older settings files keep loading |
| `complexity_router` | `[routing]` | `true` | lowest rung whose `max_complexity` covers the task | b | keep |
| `type_router` | `[routing]` | was `false` — a `[[routing.ladders.review]]` table was silently ignored until the key was set; **now on when any per-kind ladder is configured** (the written block also says `true`) | review-kind tasks take their own ladder | **a → fixed** | `type_router = false` is the off switch |
| `spend_confirm` | `[routing]` | `ask` — asks **once per run** before the first remote spend; `auto` never asks; `never` records that no confirmation applies | money gate | c | keep `ask` |
| `secret_scan` | `[routing]` | `true` | findings force a local rung; remote tool output redacted | b | keep |
| `[[routing.ladder]]`, `[[routing.ladders.<kind>]]` | settings.toml | Haiku (trivial) / Sonnet (standard, default rung) / Opus (hard); `review` ladder Sonnet → Opus | the tiers | c | keep |
| `mode` | `[decisions]` | `off` in code; **`active`** in the written default block (remote adapter present) | judges observe (`shadow`) or decide (`active`) | c | keep; stated |
| `sidecar_model`, `sidecar_url` | `[decisions]` | `qwen3:4b`, `http://localhost:11434` | the `ollama` judge | c | keep |
| `timeout_ms`, `gate_timeout_ms` | `[decisions]` | `1500`, `60000` | active-judge and gate-reviewer budgets | c | keep |
| `breaker_timeouts`, `breaker_cooldown_s` | `[decisions]` | `5`, `60` | per-point circuit breaker | c | keep |
| `answer_check` | `[decisions]` | `true` | the lead's answer to work it did is checked against the changes (below) | b | keep; `false` to measure without it |
| `gate_review` | `[decisions]` | `true` | the sandbox gate's LLM reviewer reads every submitted diff after the rules; `false` lets the deterministic rules alone decide (`gate.decide_unreviewed`): **a diff no rule flags then lands without asking in auto mode** (fail-open; with the reviewer on, a missing review asks) | b | keep; `false` only to measure without it |
| `lead`, `worker` | `[thinking]` | `high`, `medium` | extended thinking per round as a provider-neutral effort (`low`/`medium`/`high`, `off`); LiteLLM sends the level as `reasoning_effort`; the Anthropic adapter sends adaptive thinking (the model picks the depth; only `off` changes it); Ollama thinks when the model can, `off` disables | b | keep |
| `labels_margin` | `[decisions]` | `0.15` | labels judge must beat its runner-up by this to override | c | keep |
| `[decisions.points]`, `[decisions.active]`, `[decisions.thresholds]` | settings.toml | written block: `labels` on the decide judge (GLiNER2.5-Decide; 0.83 vs 0.46 for NLI on 325 real tasks, `bench/primitives/README.md`) and `panel` on the encoder judge (`labels` active when the `judge` extra is installed, else `false` with a note), `injection` shadow; threshold `0.5`. An untouched `labels = "encoder"` line from an earlier written block is migrated to `decide` at startup (`migrate_labels_judge`) once the `judge` extra with gliner2 is installed; the decide judge cuts its input at 2000 characters to stay inside the 1500 ms active timeout Memory: the written block loads GLiNER2.5-Decide (DeBERTa-v3-large) for `labels` plus NLI-base for `panel`, about 1.7 GB fp32 resident | which judge per point, which points decide | c | keep; `labels = "encoder"` is the NLI alternative |
| `enabled` | `[ledger]` | `true` | append-only JSONL ledger under `~/.guru/ledger/` | b | keep |
| `turn_line` | `[ledger]` | `true` (verified: `config.LEDGER_TURN_LINE`) | per-turn cost line and exit summary | b | keep |
| `[pricing."<model>"]` | settings.toml | bundled Anthropic price table | per-field price overrides | c | keep |
| `preactivate` | `[tools]` | `list_dir, list_tree, read_file, search_code, outline, find_symbol, run_tests, check_syntax` | tools every agent gets without `search_tools` | c | keep |
| `flat` | `[tools]` | `false` | whole registry pre-activated (costs prompt tokens; large-context models) | c | keep |
| `[tools.limits]` `timeout_s`, `cpu_s`, `mem_mb`, `fsize_mb`, `out_kb` | settings.toml (project `.guru/tools.toml` overrides) | `120`, `120`, `2048`, `64`, `256` | subprocess ceilings for the audited tools | c | keep |
| `web_summarize_over_chars`, `outline_file_over_chars` | `[context]` | `6000`, `8000` | tool-output retention thresholds | c | keep |
| `[sampling]`, `[sampling."<model>"]` | settings.toml | empty (modelfile defaults) | sampling overrides | c | keep |
| `model_timeout` | `[bench]` | `600` (`0` disables) | per-model ceiling in `guru.bench` | c | keep |
| `model`, `num_ctx` | `[evals]` | `''` (guru's default model), `8192` (`0` = auto-fit) | eval suite defaults; flags win | c | keep |
| `base_image`, `proxy_image` | `[sandbox]` | digest-pinned `python:3.12-slim`, `kalaksi/tinyproxy` | images; must be pinned | c | keep |
| `cpus`, `memory_mb`, `pids`, `timeout_s` | `[sandbox]` | `2.0`, `2048`, `256`, `600` | container limits | c | keep |
| `runtime` | `[sandbox]` | `"docker"` (the only value) | nothing read it | **d → removed** | key rejected as unknown like any typo |

## `~/.guru/adapters.toml` (`[[adapter]]` records)

| Knob | Default | Purpose | Class | Action |
|------|---------|---------|-------|--------|
| `enable` | `true` | provider shown in `/models`, usable by ladders | b | keep |
| `cache` | `true` (verified: `cli._instantiate`) | prompt-cache markers on Anthropic/LiteLLM requests | b | keep |
| `thinking` | `true` (anthropic) | adaptive thinking; off for endpoints that lack it | b | keep |
| `models` | unset (queried from the endpoint) | model allowlist | c | keep |
| `auth`, `url`, `base_url`, `api_key_env`, `api_key`, `profile` | `api_key`, `http://localhost:11434`, — | connection details | c | keep |

## Project files (`.guru/`)

| File | Default when absent | Purpose | Class | Action |
|------|---------------------|---------|-------|--------|
| `tools.toml` | everything enabled, runner `pytest`, global limits | `enabled = [...]` allowlist, `disabled = [...]`, `[tools.tests] runner`, `[tools.limits]`; an invalid file **fails closed** (every registry tool off) | b/c | keep |
| `sandbox.toml` | **sandbox on** for a project with a provisioned image (`/sandbox provision`); the file was the opt-in (`enabled = true` required, and nothing but `/sandbox status` read the flag) | `enabled = false` turns the sandbox off for the project (verbs hidden and refused, provisioning refused); other keys tweak `[sandbox]` | **a → fixed** | optional file: off switch or tweak |
| `domains_allow.txt`, `read_dirs_allow.txt`, `write_dirs_allow.txt` | empty; filled by the approval prompts | allow-lists | c | keep |
| `sensitive_markers.txt`, `scan_allow.txt` | empty | secret-scanner markers / suppressions | c | keep |
| `GURU.md`, `settings.json`, `memory/` | — | project prompt, last-used model, saved conversations | c | keep |

Why the sandbox does not provision itself: provisioning builds a Docker
image (minutes), needs Colima running and asks to allow `pypi.org` and
`files.pythonhosted.org`. Doing that silently at startup for every
uv-managed project is neither small nor safe, so the explicit
`/sandbox provision` stays the one enabling action; from then on the
sandbox is on with no file needed.

## Environment variables

| Variable | Default | Purpose | Class | Action |
|----------|---------|---------|-------|--------|
| `GURU_DEBUG` | unset | log to stderr as well as `~/.guru/guru.log`; `ui.debug` lines | c | keep |
| `GURU_DUMP_REQUESTS=<dir>` | unset | dump every remote request's kwargs as JSON (cache triage) | c | keep |
| `GURU_EVAL_SANDBOX_ROOT` | the system temp dir | where the eval runner's stable sandbox copies live | c | keep |

`GURU_ANTHROPIC_API_KEY` is only the example value of `api_key_env` in the
adapter template, not a knob guru reads by name.

## Command-line flags and slash commands

| Knob | Default | Purpose | Class | Action |
|------|---------|---------|-------|--------|
| `guru --model`, `--num-ctx`, `--reset-skills` | last used model, `0` (GPU auto-fit), off | startup overrides | c | keep |
| `/mode` | `ask-for-changes` | read-only / ask / auto | c | keep |
| `/routing on\|off` | on (see `[routing]`) | flip `mode` in settings.toml and reload | b | keep |
| `/sandbox status\|provision [--force]\|gate\|deps …` | `status` | inspect, build, review, dependency requests | c | keep |
| `evals run --allow-spend` | deny | grants the once-per-run spend question and lets sandbox cases apply an `intended` submit; the headless runner cannot ask, so the flag is the answer | b (safety gate) | keep |
| `evals run --rubric SPEC` | was: a judge only with `--allow-spend` **and** `--routing`; **now whenever `--allow-spend` is given** — the routing file's cheapest rung, else the cheapest Claude tier of the first enabled remote adapter (`cheapest_remote_spec`); `--rubric none` turns grading off | model-graded rubric cases | **a → fixed** | `none` is the off switch |
| `evals run --rubric-min N` | unset (a grade is reported, never fails a case) | gate on the grade | c | keep |
| `evals run/grade --samples N` | `1` | ask the judge N times, record the median | c | keep; documented |
| `evals run --repeat N` | `1` | run the selection N times; gate ceil(N/2) | c | keep |
| `evals run --routing FILE` | none — the runner deliberately ignores `settings.toml`'s `[routing]` so a run is reproducible from the file | the experiment | c | keep |
| `evals run --model`, `--num-ctx` | `[evals]` settings, else guru's default model / `8192` | model under test | c | keep |
| `evals run/list --tags`, `--cases` | all cases | selection | c | keep |
| `evals run/grade --out` | `evals/runs` | run directory | c | keep |
| `evals grade --rubric SPEC` (repeatable), `--show`, `--labels FILE` | required unless `--show`; off; `evals/rubric-labels.toml` | offline re-grading | c | keep |

## Code constants (not settings keys)

Read only from `guru/config.py`; listed so their defaults are on record:
`MODE = ask-for-changes`, `AUTO_GRANT = True` (the eval runner turns it off
so auto cases still hit its denying asker), `DELEGATION_NUDGE_MIN_READS = 3`
(`0` disables it; the former `OVER_READ_LIMIT` over-read guard is gone —
`read_file` is structural), `COMPACT_AT = 0.85`,
`KEEP_RECENT_GROUPS = 4`, `DEFAULT_NUM_CTX = 4096`, `GPU_FIT_SAFETY = 0.95`,
`SECRET_SCAN` (mirrors `[routing] secret_scan` while a table is present).
Every constant in `config.py` is read by at least one module; none was
removed.

## Startup output (no knob)

Always on, nothing to set: `./start.sh` prints one line per startup step
(`settings, skills, ledger`, `adapters`, `main model`, `judges`) with what
it found, where each model runs (`local` / `remote` and the host; `GPU` or
`CPU spill` for an Ollama model) and how long the step took. Judges warm
on a background thread after the prompt opens; the statusline shows
`judges: loading <name>…`, then `judges ready Ns` for
`READY_SHOWN_S = 10` seconds (`guru/domain/startup.py`), or `judges:
<names> failed (see log)` until restart. Anything the judge libraries
print or log during warm-up (the gliner2 config banner, transformers'
`Device set to use mps`, attention-implementation warnings) goes to
`~/.guru/guru.log` at debug level instead of the terminal.

## Lead, workers and the turn loop (no knob)

Always on, code constants (2026-10-09: the controller, its `plan` tool,
forced tool calls, round caps, read budgets, deliverables and design-first
rules were removed after three dashboard eval cycles showed them costing
more than they caught):

- **The lead** (`config.LEAD_HINT`): the main agent keeps the overview,
  works itself where that is quicker, spawns workers (`spawn` with
  kind/complexity, routed over the ladders) for parts that can run in
  parallel, joins them, reviews and integrates what they return and
  verifies the whole before it answers. It sees the project map and rules.
  A worker gets `config.WORKER_HINT`: do the task fully, test and lint
  what it changed, and end with a report (files changed, what was
  verified, what is left).
- **No forced tool calls**: a reply without a tool call ends the turn (the
  lead's answer, a worker's report). Forcing `tool_choice` silently turns
  Anthropic's thinking off, so it is never sent.
- **Stall monitor instead of a round cap** (`guru/adapters/turn.py`): a
  round makes progress when a file changed or a tool returned something
  not seen before in the turn (timings, clock times and hex addresses
  ignored). `STALL_ROUNDS = 20` rounds without progress put one warning on
  the next tool result; `STALL_GRACE = 5` more end the turn — a worker's
  with a handoff built in code (files changed, files read, its last text;
  task status `stalled`), the lead's with its last text and a note. A
  sandbox script (`sandbox_python`) with new code counts as progress even
  when its output repeats. Struggle columns `stall_warnings`, `stalls`.
  Nothing else bounds a turn's cost in the TUI (the spend confirmation is
  asked once per run); an eval case's `max_cost_usd` cancels its run.
- **Provider failure mid-task**: a worker whose provider call fails after
  it read or changed files (the context full, an outage) ends with the
  same code-built handoff (`stopped: the provider failed …`, status
  `stalled`) instead of no answer, so its copy and its work reach the
  lead. The history is not compacted mid-turn.
- **Re-runs after an edit**: once a round changes files, earlier calls
  are no longer answered as duplicates (a test run, a read may be stale).
  A refused or duplicate test run does not count as testing.
- **Whole-suite test runs**: `run_tests` with no target and no `-k` gets
  at least `SUITE_TIMEOUT_S = 600` seconds of wall clock and CPU
  (`guru/config.py`; the `[tools.limits]` defaults stay 120 s for every
  other run); a timed-out whole suite tells the agent to run the tests
  for what it changed.
- **Child process width**: every audited subprocess (tests, lint, git)
  sees `COLUMNS=200` (`procs.CHILD_COLUMNS`): without a terminal, rich
  wrapped at 80 columns and a long scratch-path line broke a test only
  under guru's `run_tests`.
- **Verify before reporting**: an agent that changed files and has not
  run `run_tests` (or `sandbox_run`/`sandbox_python`) and — when the
  project's `lint` tool is enabled — `lint` since has its answer sent back
  once, naming what is missing (`verify_sendbacks`).
- **Project rules**: the first of `AGENTS.md`, `CLAUDE.md`,
  `.guru/rules.md` at the project root (`brief.RULES_FILES`, cut at
  `MAX_RULES_CHARS = 6000`) is read fresh into the lead's and every
  worker's system context as `[project rules — follow them]`.
- **Answer check** (`guru/domain/claims.py`, endpoint
  `guru/judges/claims.py`): when the lead answers a request it worked on
  (a `spawn`, a write, or workers' results in the request's history), a
  reviewer on the routing's `standard` review rung (secret scan,
  local-only mode and the spend confirmation apply, as for the gate)
  compares the answer with the request and the repository's changes
  (`git diff HEAD` plus new files, `MAX_EVIDENCE_CHARS = 32000`), once per
  request. Problems (at most `MAX_PROBLEMS = 5`) send the answer back once;
  a reviewer error or garbage reply delivers the answer unchanged.
  Installed by the CLI and the eval runner; `[decisions] answer_check`.
- **Sandbox projects**: a worker edits its own copy and does not submit
  (`sandbox_submit` is hidden from workers); when it finishes, its copy's
  diff (cut at `DIFF_REPORT_CHARS = 30000`) rides along with its report.
  The lead reviews it and calls `apply_work(<worker>)` to merge it into
  its own copy, tests there, and submits the integrated change once
  through the gate. The merge is all or nothing: the lead's own edits are
  staged first and `git apply --3way` runs; a conflict or failure rolls
  the copy back and names the conflicting files, keeping the worker's
  copy; a truncated worker diff is refused. The diff in the report sits in
  a code fence longer than any backtick run in it. Every copy is removed
  when the TUI exits. A finished worker's copy lives
  until it is applied or the user's next request; a failed one's goes at
  once.
- **Write tools up front**: a writing task kind starts with `write_file`,
  `edit_file` and `apply_patch` active (project policy, kind hiding and
  the sandbox rule still apply).
- **Thinking room**: a LiteLLM turn round may use up to
  `TURN_MAX_TOKENS = 32000` output tokens (thinking spends them before the
  reply). A model that rejects `reasoning_effort` is retried once without
  it and remembered for the run.
- **Cut-off replies**: when the provider reports the reply hit its output
  limit (LiteLLM `finish_reason = length`, Anthropic `stop_reason =
  max_tokens`), that round's tool calls are not run (`OUTPUT_CUT_REFUSAL`:
  write large content in parts). A call whose result failed (invalid
  arguments, refused, tool error, not run) may be retried identically; it
  is not answered as a duplicate.
- **Agent tabs** (TUI): a sub-agent that finished `done` leaves the tab
  bar (archived; `check`/`join` still find it, the transcript is in the
  ledger); one that ended `stalled` or `error`, or that is on screen when
  it finishes, stays until its tab has been viewed and left. Titles count
  up for the whole run and are never reused.

Eval runner: `max_cost_usd` (per case) cancels the run once its priced
calls pass the cap and fails the check; `task_status_none` fails a case
whose sub-tasks ended in a listed status; `fixture_home_clean` fails a
case whose fixture tests left files in their private HOME beyond
`fixture_home_allow`; `fixture_lint_pass` runs flake8 (`guru bench tests
evals`, those present) and mypy (`guru`) in the copy; every run that changed the
fixture writes its patch to `evals/runs/<run_id>/diffs/<case>.patch`.

`[evals] home` (settings.toml, default `''` = `~/.guru-evals`): the root
under which `python -m guru.evals verify <ref>` makes a fresh HOME
(`verify-<timestamp>-…`) for a candidate branch's flake8 (`guru bench tests
evals`, as `make lint`), mypy (`guru`) and pytest, so nothing they write
lands in the real `~/.guru` and no run sees another's files; the HOME is
kept and its files are listed.

`[decisions] answer_check` (default `true`): the answer check above; `false`
turns it off (the CLI and the eval runner then install no checker).
`[decisions] gate_review` (default `true`): the sandbox gate's LLM
reviewer; `false` leaves the verdict to the deterministic rules. An eval
experiment file's `[decisions]` table takes the same two keys for its run
(`evals/routing/claude-tiers-nojudges.toml` switches both off; the run's
judges list then says `answer_check=off`, `gate_review=off`).

## Usage store and dashboard

| Knob | Where | Default | Purpose | Class |
|------|-------|---------|---------|-------|
| `usage_db` | `[ledger]` | `true` | every model call, each request's topic and each sub-agent task in `~/.guru/usage.db` (SQLite, shared by every guru process; file 0600, a new directory 0700) | b |
| `GURU_USAGE_DB` | env | unset (`~/.guru/usage.db`) | another file for the store (tests, a second profile) | c |
| `enabled` | `[dashboard]` | `true` | the usage dashboard on `127.0.0.1`, served by the first guru to bind the port; the others show where it is and retry every 30 s (±30 %) | b |
| `port` | `[dashboard]` | `7340` | dashboard port (1024-65535; anything else is ignored and logged) | c |
| `topic_labels` | `[dashboard]` | `true` | a 3-6 word label per user request from the cheapest routed rung (`explain`/`trivial`), in the background on a session of its own; remote rungs only, never the main model or a local rung (no queueing behind the turn, no second local model); without one (no ladder, local-only mode, a secret-scan finding, a pending spend confirmation) the topic is the request text | b |

Code constants: the request text kept as a topic and a task goal are
redacted — the project scanner's findings (`policy.redact`), then
token-shaped strings (provider keys, `key=`/`token=`/`password=` values,
40+ character opaque runs; `usage._TOKEN_RES`) — and then cut to
`TOPIC_CHARS = 200`; the label model only ever sees the redacted text; the store
waits at most `BUSY_TIMEOUT_S = 1.0` for a lock, drops a row that meets a
busy database and disables itself on any other error; schema changes add
columns and never lower `user_version` (`SCHEMA_VERSION = 1`). Eval runs
write with `source = eval`; the dashboard's Source filter separates them.
Data is kept forever. `/dashboard` prints where the dashboard is served.
A retried sub-agent keeps its original request's topic; `/review` opens
a topic of its own. Known limit: workers' results delivered after the
user has already started a new request are synthesised under the new
request's topic (the delivery carries no topic).
