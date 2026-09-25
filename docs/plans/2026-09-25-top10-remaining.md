# Remaining approved top-10 items — implementation plan

Status (2026-09-25): approved by Ronald on 2026-09-24 ("go do the approved
top-10 items"); items 2 (audited tools + sandbox) and 7 (symbol tools) are
done on `feat/audited-tools`; item 9 (streaming) rejected; item 10 (retire
local-worker docs) folded into item 1's template change. This plan covers
1, 3, 4, 5, 6, 8 and the tool-usage smell report (3b). Chunk A (items 1,
6, 10) implemented 2026-09-25 (`settings.default_routing_toml` /
`ensure_default_routing` / `switch_routing`, `mode = "off"`, `/routing`,
the review ladder via `type_router = true`, the panel judge's security
worker in `Orchestrator.spawn`; the `/routing` dispatch line in
`guru/tui.py` is pending); Chunk B (items 3, 4) implemented 2026-09-25,
probe result below; Chunk C (items 5, 8, 3b) implemented 2026-09-25
(`guru/evals/rubric.py`, `--rubric` / `--rubric-min` / `--repeat`,
`make eval-fast`, "Tool usage smells" in the ledger report; spread =
sample standard deviation, grading cost kept out of the case cost); all
uncommitted.

Assumptions taken without asking (change them if wrong):

- Item 1 writes the default `[routing]` block only when no `[routing]`
  exists and at least one remote adapter is enabled; `/routing` toggles
  `mode` between `off` and the written config, never deletes user edits.
- Item 5 grades with Haiku (cheapest rung) on a 0–2 scale per case, the
  same scale used by hand so far (6/6 over three cases).
- Item 8's gate: every fast case must pass in at least 2 of 3 repeats;
  cost and time are reported as mean ± spread and never fail the gate.

## Chunk A — routing defaults and the panel judge (items 1, 6)

- `guru/repositories/settings.py`: `default_routing_toml(adapter, kind)`
  renders the measured configuration (controller Haiku, tiers Haiku /
  Sonnet / Opus, labels judge active with margin 0.15, panel judge active,
  `[[routing.ladders.review]]` starting at Sonnet) with model ids per
  adapter kind (litellm/Bedrock ids vs Anthropic ids); `ensure_default_routing()`
  writes it into `~/.guru/settings.toml` when the file has no `[routing]`
  and a remote adapter is enabled; returns what it did.
- `guru/cli.py` startup calls it once and prints one line when it wrote
  the block. `/routing` shows the active ladder and `mode`, `/routing off`
  / `/routing on` flips `mode` in the file and reloads.
- Item 6: `[decisions]` `panel` point promoted to `active` in the default
  block; the orchestrator, for `review`-kind tasks, adds a security
  worker when the panel judge says `needs_security` and no security role
  was spawned; review-kind tasks resolve on the `review` ladder.
- README: "Routing" section rewritten around the default; local-worker
  ladder examples removed (item 10).

## Chunk B — prompt caching and the per-turn cost line (items 3, 4)

- `guru/adapters/anthropic.py`: `cache_control: {type: ephemeral}` on the
  system prompt and the last tool schema; usage already records cache
  tokens.
- `guru/adapters/litellm.py`: the same markers in the OpenAI-compatible
  shape the LiteLLM proxy forwards to Anthropic/Bedrock (content-part
  `cache_control`); verified against the proxy with a two-call probe
  (`cache_read_tokens > 0` on the second call) recorded in the plan.
- `guru/domain/ledger.py`: `turn_summary(turn_id)` → cost, calls, models,
  cache read share; `guru/cli.py` and `guru/tui.py` print one dim line
  after each answer, plus a session summary at exit; `[ledger]
  turn_line = true|false`.

Probe result (2026-09-25, `SBP Litellm` → `aws/claude-4-5-haiku`, two
identical calls through `LiteLLMAdapter`'s helpers, system prompt as a
content part with `cache_control` plus the marker on the last of two
tools): **the proxy honours `cache_control`.** Call 1: input 360, output
47, cache_read 0, cache_write 6386, cost header $0.00944. Call 2: input
360, output 49, cache_read 6386, cache_write 0, cost header $0.00137
(−85%). The numbers arrive in Anthropic-style `usage.cache_read_input_tokens`
/ `usage.cache_creation_input_tokens` and, mirrored, in
`usage.prompt_tokens_details.cached_tokens` / `.cache_creation_tokens`;
`usage.prompt_tokens` (6746) counts the cached tokens too, so the adapter
prices `prompt_tokens − read − write` as uncached input. Two caveats:
Haiku 4.5's minimum cacheable prefix is 4096 tokens, so the first attempt
with a ~3700-token prompt (the plan's "~3000 tokens") wrote and read
nothing — a short guru system prompt on Haiku will not cache until tools
plus prompt pass 4096 tokens (Sonnet 5 needs 1024, Opus 5 512); and the
proxy enforces a thinking budget, so `max_tokens` below ~8k is a 400
(the adapter's 16384 is fine). A control pair without markers showed no
implicit caching (cost $0.0070 both calls). Anthropic's own adapter
(`cache_control` on the system block and the last tool) was not probed
live; it is the documented first-party shape. Pricing table already
carried cache columns (reads 10% of input except Fable 5.1 at 2.5%,
5-minute writes 125%), verified against the claude-api reference.

## Chunk C — rubric judge, ×3 fast gate, tool-usage smells (items 5, 8, 3b)

- `guru/evals/rubric.py`: `grade(case, answer, judge_spec) -> (score 0–2,
  reason)` through `Adapter.complete()` with a fixed prompt; recorded as a
  `labels` row (`labeller = rubric:<model>`); runner flag `--rubric
  'Adapter|model'` (default: the routing file's cheapest rung when
  `--allow-spend`); table column `rubric: 2/2`; summary `rubric N/M`.
- `--repeat N` in the runner: N runs of the selection, one run file per
  repeat plus a `compare`-style aggregate: pass count per case, cost and
  seconds mean ± spread; exit 1 when a case passes fewer than
  `ceil(N/2)` times. `make eval-fast` = `--tags fast --repeat 3`.
- `bench/ledger_report.py`: "Tool usage smells" — whole-file reads after
  an outline of the same file, `search_tools` for a pre-activated tool,
  repeated identical calls, refused calls, bytes shown vs produced ratio.

Gate per chunk: `make lint && make typecheck && make test`; one commit per
chunk; then a measured run (fast ×3 + real cases) compared with
`evals/triage/2026-09-24-sandbox.md`.
