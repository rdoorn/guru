# Structural round — make guru cheaper and smarter than a single-model CLI, on any model

Ronald (2026-09-26): "get back to the basics and make sure that what I type
in guru ends up cheaper and smarter than using the Claude CLI"; we had
fixed model failures (wrong tools, getting stuck) with prompt workarounds;
"a plan with zero tasks is rejected" is wrong — a simple question has no
task. Approved: all items below.

Principle: enforce behaviour in code, request nothing in prose that can be
checked. A model is qualified by measurement, not trusted by prompt.

Headline metrics from now on (model-agnostic): tokens per task, tool bytes
per task, turns per task; dollars second.

## Package A — control plane (typed plan + turn contract)

Files: guru/orchestrator.py, guru/adapters/turn.py, guru/adapters/{base,
anthropic,litellm,ollama}.py (tool forcing), guru/domain/routing.py,
guru/domain/plan.py (new), guru/config.py (hint removal), tests.

1. **Typed controller plan.** The controller answers a user turn with ONE
   forced tool call `plan` whose schema is:
   `{outcome: "answer"|"delegate", answer?: str, tasks?: [{goal: str,
   kind: <routing KINDS>, complexity: <labels>, files?: [str],
   role?: str, skill?: str}]}`.
   - `answer`: the controller's text is the reply; no worker runs (simple
     questions, follow-ups on delivered results, clarifications).
   - `delegate`: guru validates and then spawns every task itself (the
     existing `spawn_panel` path with the plan's kind/complexity → ladder),
     joins, ends the turn; results arrive through the mailbox and the
     controller answers with a second `plan` whose outcome is `answer`
     (or delegates again).
   - Validation in code (`plan.validate(request, plan) -> list[str]`):
     kinds and complexities must be known; `delegate` needs ≥1 task; when
     the request names several concerns from a fixed vocabulary
     (correctness, security, performance, reliability, design, tests,
     docs) every named concern must be covered by a task goal, else ONE
     re-ask listing the missing concern; a malformed plan (neither
     outcome, missing fields) → one re-ask, then a protocol error
     recorded in the turn row. Never reject `answer`.
   - The controller's tool set becomes `plan` only (plus `ask_user` if it
     exists). `spawn`/`join`/`check` stay for non-controller delegation.
   - Remove the now-redundant prose: decomposition sentences of
     `CONTROLLER_HINT`, the "one worker per named concern" rule, the
     join/check polling rule; keep the `[project]` block.
2. **Turn contract.** A worker turn ends with a tool call or a
   `final_answer(text)` tool call. Adapters that support tool forcing
   (Anthropic `tool_choice any`, LiteLLM/OpenAI `required`) force it;
   Ollama falls back to text-as-answer. A text-only reply on a forcing
   adapter is a protocol violation: one deterministic re-prompt
   ("call a tool or final_answer"), then the text is taken as the answer
   and the turn row gets `protocol_violation = 1`. Remove the preamble
   regex stall path and the stall nudge; keep the ledger `stall_nudges`
   column reading zero for old rows.
3. Tests: plan schema/validation matrix, re-ask once, answer outcome runs
   no worker, delegate spawns exactly the plan, review tasks resolve on
   the review ladder, final_answer forcing per adapter, violation
   re-prompt then record, mailbox synthesis via `plan`. Evals must still
   pass on the fake adapters.

## Package C — tool surface + project brief + model qualification

Files: guru/domain/tools.py, guru/domain/code.py, guru/domain/files.py,
guru/domain/brief.py (new), guru/repositories/briefs.py (new),
guru/domain/toolpolicy.py, bench/tool_contract.py (new), README tool
sections, tests. Do not edit config.py (Package A); put constants in the
new modules.

1. **Structural read.** `read_file` on a file over `READ_OUTLINE_LINES`
   (200) without `lines=` returns the outline (defs/classes with line
   ranges, first 20 lines) and says how to fetch a range; whole-file
   reads of large files become impossible instead of discouraged. Remove
   the over-read guard and the "prefer outline" prose (report the prose
   lines to the coordinator; config.py is Package A's).
2. **Strict arguments, corrective errors.** Every tool validates its
   arguments against its schema before running (types, required, enum)
   and returns one line naming the expected shape and an example.
   `sandbox_run` argv must be a list (already documented).
3. **Review tasks are read-only.** A task of kind `review` runs with the
   write tools hidden and refused (`toolpolicy` per task kind), so a
   reviewer cannot wander into edits.
4. **Project brief.** `brief.build(root, head_sha)`: file map (dirs with
   counts, top-level modules), outline per Python module (capped), symbol
   index (defs → path:line), how to run tests (pytest/unittest, Makefile
   targets), conventions (pyproject tool sections). Stored under
   `~/.guru/briefs/<project-key>/<head_sha>.json`, rebuilt when HEAD
   changes or on `/brief refresh`. `brief.slice(brief, task_text,
   max_tokens=1500)`: the files whose names or symbols appear in the task
   plus the map and the test command. Workers get the slice in their task
   text; the controller gets the map for planning (Package A calls
   `brief.slice`; agree on the function signature above).
5. **Tool contract benchmark.** `bench/tool_contract.py --model
   'Adapter|model'`: one canned mini-task per tool (fixture `cli-tool`),
   forcing a call; records call success, schema errors, retries, tokens,
   seconds; writes `evals/models/<slug>.json`. Run it on the Haiku, Sonnet
   and one local Ollama model (qwen3:4b or 8b, small only) and record the
   numbers in this plan.

### Package C — delivered 2026-09-26 (branch feat/structural-C)

Tool contract (`bench/tool_contract.py`, 20 tools: the always-on five plus
every registry tool except the web three (`--web`) and the sandbox six
(no image on the fixture); forcing on through the LiteLLM proxy):

| model | ok | called | schema errors | retries | tokens in+out | seconds | USD |
|---|---|---|---|---|---|---|---|
| SBP Litellm\|aws/claude-4-5-haiku | 20/20 | 20 | 0 | 0 | 15195+3126 | 31.5 | 0.0339 |
| SBP Litellm\|aws/claude-5-sonnet | 20/20 | 20 | 0 | 0 | 15684+1160 | 40.0 | 0.0473 |
| local model | not run (Ronald's request) | | | | | | |

Results: `evals/models/sbp-litellm-aws-claude-4-5-haiku.json`,
`evals/models/sbp-litellm-aws-claude-5-sonnet.json`. Both Claude tiers
meet the contract on every tool at the first attempt; the validator saw no
schema error, so the corrective line was never needed. Haiku spends ~2.7x
Sonnet's output tokens (it narrates before calling).

Project brief on guru itself (222 files, 134 Python, 123 modules outlined,
3418 symbols): build 0.20-0.56 s (cold 0.56 s), `git rev-parse` 0.46 s
through `procs.run`, stored JSON ~390 KB; the 3 s outlining budget is
never reached. A slice for a two-file review task is ~1240 tokens.

## Package E — metrics, matrix, gate hygiene

Files: guru/evals/*, guru/domain/ledger.py (metrics only),
guru/domain/ledger_report.py, bench/ledger_report.py, evals/routing/*,
evals/cases/*, Makefile, evals/README.md, tests.

1. **Model-agnostic metrics.** Per case: tokens in/out, cache read, tool
   bytes shown, turns, calls; the eval table gains `tok` (k tokens) and
   `turns`; the summary and TRAJECTORY row gain tokens per case; the
   aggregate table (`--repeat`) reports tokens mean ± spread. Ledger report
   "per task" section with the same numbers.
2. **Matrix.** `python -m guru.evals matrix --models 'A|m1,B|m2' [--tags
   fast]`: runs the selection once per model as the worker/main model
   (controller from the routing file when given) and prints one row per
   model: passed, tokens, tool bytes, seconds, cost. Reads
   `evals/models/<slug>.json` (Package C) for a `contract` column when
   present.
3. **Checks that measure preference become smells.** `planted-failure-
   digest` and `find-symbol-outline`: drop `tools_used_none = [read_file]`
   (triage note); the table gets a `smells` column (whole-file reads,
   repeated calls, refused calls per case) from the tool_events smells.
4. **Gate.** `make eval-gate` = fast ×3 + sandbox cases ×3 + dogfood ×3;
   `make eval-fast` unchanged.
5. **Judges.** `panel = false` in the default routing block and
   `claude-tiers-panel.toml` renamed to `claude-tiers-review.toml`
   (review ladder only); the panel point stays shadow until labelled rows
   say otherwise (README sentence). Labels judge unchanged.

## Measurement after the merge

Suite as before (fast ×3, real + dogfood, sandbox, review-multi-file ×3)
with tokens per task as the headline, compared with suite 3 of
`evals/triage/2026-09-25-loop-1.md`; tool contract numbers for three
models; independent review of the diff; triage note
`evals/triage/2026-09-26-structural.md`. Budget ≈ $10.
