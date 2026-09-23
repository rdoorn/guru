# Routing experiment: does local steering cut remote cost?

Three runs of the same suite, differing only in who plans and who works.
Compare pass rate, seconds and cost (the run file, the `compare` output
and the `TRAJECTORY.md` row carry all three; the table's `routes:` detail
shows which `Adapter|model` each case's sub-agents ran on).

| run | controller (main agent) | workers | routing file |
|---|---|---|---|
| A | remote, no routing | remote (same model, inherited) | none |
| B | remote (`--model`) | local 8B; hard tasks remote | `remote-controller-local-workers.toml` |
| C | local (`--model`) | local 8B; hard tasks remote | `local-controller-remote-hard.toml` |

B and C carry the same ladder (trivial/standard on
`Ollama|huihui_ai/qwen3-abliterated:8b`, hard on
`SBP Litellm|aws/claude-5-sonnet`); the controller model is whatever
`--model` gives, so the two files exist to keep the run's `routing` label
honest. Both set `controller = true` (the main agent only spawns, checks
and joins), `spend_confirm = "auto"` (no prompt; `--allow-spend` is still
required, see below) and `secret_scan = true`.

The `adapter` in every rung must match the `name` of an `[[adapter]]` in
`~/.guru/adapters.toml` exactly (here `Ollama` and `SBP Litellm`); a rung on
an unknown adapter is dropped with a warning and the ladder shrinks.

## Commands

Remote spend is denied by default in the eval runner; `--allow-spend`
grants it for the run. Without it every remote rung is skipped and B/C
degrade to all-local.

```sh
# A: all remote, no routing (baseline cost)
.venv/bin/python -m guru.evals run --model 'SBP Litellm|aws/claude-5-sonnet' \
    --allow-spend --note 'A: all remote'

# B: remote controller, local workers (hard -> remote)
.venv/bin/python -m guru.evals run --model 'SBP Litellm|aws/claude-5-sonnet' \
    --routing evals/routing/remote-controller-local-workers.toml \
    --allow-spend --note 'B: remote controller, local workers'

# C: local controller, remote only for hard tasks
.venv/bin/python -m guru.evals run --model 'Ollama|huihui_ai/qwen3-abliterated:8b' \
    --routing evals/routing/local-controller-remote-hard.toml \
    --allow-spend --note 'C: local controller, remote hard'

# then
.venv/bin/python -m guru.evals compare evals/runs/<A>.json evals/runs/<B>.json
.venv/bin/python -m guru.evals compare evals/runs/<A>.json evals/runs/<C>.json
```

Add `--tags fast` to any of them for the short gate. The run's model label
reads `Adapter|model@ctx+routed:<file stem>+controller` for B and C.

## Reading the cost

A case's cost is the sum of its ledger `calls` rows; local calls are $0.
Remote ids such as `aws/claude-5-sonnet` are priced through the
first-party table by alias (`guru/domain/pricing.py`); Bedrock partner
pricing may differ, so treat the figure as an approximation and override
per model with `[pricing."aws/claude-5-sonnet"]` in `settings.toml` when
the exact rate matters.
