# Routing experiment: Claude tiers and shadow judges

Claude is the controller and every worker; local models are only small
judges (design decision 10). Three runs of the same suite, differing only
in how sub-agent tasks are routed and whether the judges observe. Compare
pass rate, seconds and cost (the run file, the `compare` output and the
`TRAJECTORY.md` row carry all three; the table's `routes:` detail shows
which `Adapter|model` each case's sub-agents ran on, and the summary's
`judges` field lists the judges a config-2 run installed).

| config | main agent | workers | file |
|---|---|---|---|
| 0 | plain Sonnet, no routing | Sonnet (inherited) | none |
| 1 | Sonnet controller | Claude tiers by complexity | `claude-tiers.toml` |
| 2 | Sonnet controller | Claude tiers by complexity | `claude-tiers-judges.toml` |

Configs 1 and 2 carry the same ladder: trivial tasks on
`SBP Litellm|aws/claude-4-5-haiku`, standard on
`SBP Litellm|aws/claude-5-sonnet` (the default rung), hard on
`SBP Litellm|aws/claude-5-5-opus`. Both set `controller = true` (the main
agent only spawns, checks and joins), `spend_confirm = "auto"` (no prompt;
`--allow-spend` is still required, see below) and `secret_scan = true`.
Config 2 adds a `[decisions]` table: `mode = "shadow"` with the `panel`
point on the `encoder` judge and the `injection` point on the `injection`
judge, no sidecar LLM. In shadow mode the judges only log their verdicts
next to the heuristic's in the ledger `decisions` stream (data collection
for the review loop); they never change what the run does.

The `adapter` in every rung must match the `name` of an `[[adapter]]` in
`~/.guru/adapters.toml` exactly (here `SBP Litellm`); a rung on an unknown
adapter is dropped with a warning and the ladder shrinks.

## Requirements

- An `SBP Litellm` adapter in `~/.guru/adapters.toml` that serves the
  three model ids above.
- Config 2 only: the judge extra (`uv sync --extra judge`) for the encoder
  judges. Without it the judges are skipped with a log line, the run
  records `judges: []` and is otherwise identical to config 1.

## Commands

Remote spend is denied by default in the eval runner; `--allow-spend`
grants it for the run. Without it every remote rung is skipped, and since
every rung here is remote, sub-agents fall back to the controller's own
(pre-approved) model.

```sh
# 0: plain Sonnet, no routing (baseline)
.venv/bin/python -m guru.evals run --model 'SBP Litellm|aws/claude-5-sonnet' \
    --allow-spend --note '0: baseline, no routing'

# 1: Sonnet controller, Claude tiers for the workers
.venv/bin/python -m guru.evals run --model 'SBP Litellm|aws/claude-5-sonnet' \
    --routing evals/routing/claude-tiers.toml \
    --allow-spend --note '1: claude tiers'

# 2: as 1, plus panel/injection judges in shadow
.venv/bin/python -m guru.evals run --model 'SBP Litellm|aws/claude-5-sonnet' \
    --routing evals/routing/claude-tiers-judges.toml \
    --allow-spend --note '2: claude tiers + shadow judges'

# then
.venv/bin/python -m guru.evals compare evals/runs/<0>.json evals/runs/<1>.json
.venv/bin/python -m guru.evals compare evals/runs/<1>.json evals/runs/<2>.json
```

Add `--tags fast` to any of them for the short gate. The run's model label
reads `Adapter|model+routed:<file stem>+controller` for configs 1 and 2;
the run file's `judges` list (and the CLI summary) names the judges config
2 installed, as `point=judge`.

## Reading the cost

A case's cost is the sum of its ledger `calls` rows. Remote ids such as
`aws/claude-5-sonnet` are priced through the first-party table by alias
(`guru/domain/pricing.py`); Bedrock partner pricing may differ, so treat
the figure as an approximation and override per model with
`[pricing."aws/claude-5-sonnet"]` in `settings.toml` when the exact rate
matters. Config 2's judges run locally and cost nothing; their rows are in
the run's `ledger/decisions` stream, not in `calls`.

## What changed since the local-worker runs

The 2026-09-23 runs with local 8B/14B workers (triage
`evals/triage/1f4f8262a80a.md`) showed the local rungs timing out and
hallucinating; the ladder is now Claude tiers only. Two loop fixes from
that triage apply to every config here: a `join` that opens a barrier
ends the controller's turn (no more polling rounds), and the delegation
nudge fires only after three distinct files were read on a request that
is not a single-file edit, never for a controller.

## Runs (2026-09-23, git 8296e89)

| config | run id | result | mean s | cost |
|---|---|---|---|---|
| 0 | `04dbcf7566f9` | 9/9 | 35.0 | $0.94 |
| 1 | `3f9ab125dbeb` | 7/9 | 18.8 | $0.38 |
| 2 | `f1929d55c41a` | 7/9 | 26.8 | $0.54 |

Triage: `evals/triage/2026-09-23-claude-tiers.md`. Short version: the
controller asked "which repository?" on two cases instead of delegating
(fixed since: a `[project]` block in the system context and a "never ask
which repository" rule in the controller hint), every spawned task was
labelled `standard` so the Haiku and Opus rungs never ran, and on the cases
where work happened the controller cost more than plain Sonnet. The saving
against the pre-fix baseline (384761c577f4, $4.12) came from the `join`
polling fix, not from routing. Rerun config 1 after the fixes before
drawing further conclusions.

- `ad9f5e9caece` — config 3: Haiku 4.5 controller + Claude tiers: 9/9, 29.1 s, $0.769 (beats plain Sonnet). See the triage note.
- `c7473cbe8524` — config 1 rerun after the controller-context fix: 8/9, 38.9 s, $1.205.

- Real cases on guru (2026-09-24): see `evals/triage/2026-09-24-real-cases.md` — Haiku controller + tiers matched plain Opus quality at 22% of its cost.
