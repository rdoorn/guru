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
| `controller` | `[routing]` | **on when any ladder rung is configured**, else off | the main agent only spawns/checks/joins | c | keep (auto default) |
| `complexity_router` | `[routing]` | `true` | lowest rung whose `max_complexity` covers the task | b | keep |
| `type_router` | `[routing]` | was `false` — a `[[routing.ladders.review]]` table was silently ignored until the key was set; **now on when any per-kind ladder is configured** (the written block also says `true`) | review-kind tasks take their own ladder | **a → fixed** | `type_router = false` is the off switch |
| `spend_confirm` | `[routing]` | `ask` — asks **once per run** before the first remote spend; `auto` never asks; `never` records that no confirmation applies | money gate | c | keep `ask` |
| `secret_scan` | `[routing]` | `true` | findings force a local rung; remote tool output redacted | b | keep |
| `[[routing.ladder]]`, `[[routing.ladders.<kind>]]` | settings.toml | Haiku (trivial) / Sonnet (standard, default rung) / Opus (hard); `review` ladder Sonnet → Opus | the tiers | c | keep |
| `mode` | `[decisions]` | `off` in code; **`active`** in the written default block (remote adapter present) | judges observe (`shadow`) or decide (`active`) | c | keep; stated |
| `sidecar_model`, `sidecar_url` | `[decisions]` | `qwen3:4b`, `http://localhost:11434` | the `ollama` judge | c | keep |
| `timeout_ms`, `gate_timeout_ms` | `[decisions]` | `1500`, `60000` | active-judge and gate-reviewer budgets | c | keep |
| `breaker_timeouts`, `breaker_cooldown_s` | `[decisions]` | `5`, `60` | per-point circuit breaker | c | keep |
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
