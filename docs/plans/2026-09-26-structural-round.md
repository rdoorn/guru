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
   seconds; writes `evals/models/<slug>.json`. Run it on Haiku and Sonnet
   and record the numbers in this plan. No local Ollama model is run
   unless Ronald asks (2026-09-26).

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

### Package A — delivered 2026-09-26 (branch feat/structural-A, merged 89eb935)

Typed plan (`guru/domain/plan.py`): `parse`/`schema_errors`/
`missing_concerns`/`evaluate` return a `Verdict`; the handler
(`Orchestrator.do_plan`) re-asks once, then runs a coverage-gapped plan as
given and never a malformed one; `answer` is never rejected. Turn contract
(`guru/adapters/turn.py`): `forced_tool()` per round, `Adapter.forces`
per adapter (Anthropic `tool_choice: any`/`tool`, LiteLLM `required`/
named; Ollama cannot), one re-prompt then `protocol_violation`; a lone
`final_answer`/`plan` round collapses into the assistant's text so every
reader of the final answer sees it once. The controller's tool set is
`plan` alone (`tools.CONTROLLER_TOOLS`); guru spawns the plan's tasks
through `spawn_panel` with the plan's kind/complexity. Prose removed from
`guru/config.py` (enforced in code instead): the `SYSTEM_PROMPT` sentences
"Prefer outline ... over read_file on a whole file; then read_file only the
line range you need" and "After editing a .py file, verify with
check_syntax and run_tests before you report the change"; the
`DELEGATION_HINT` sentences "Use check to poll and join to be resumed when
a group finishes" and "Prefer outline and find_symbol over read_file on
whole files, and verify edits with check_syntax/run_tests before reporting
them"; the whole `CONTROLLER_HINT` decomposition text ("You are a
CONTROLLER ... DECOMPOSE every piece of actual work", the
`spawn(task, kind, complexity, role, skill)` label lists, "Every task that
edits code must say: verify ...", the complexity-tier sentence, "use check
to poll and join to be resumed ... then SYNTHESISE", "When a request names
several concerns ... spawn one worker per named concern", "Reply directly,
briefly, for greetings ..."), and the over-read guard (`OVER_READ_LIMIT`,
the preamble stall path and its nudge). Tests at the review round:
`tests/test_plan.py` 76, `tests/test_turn_contract.py` 43,
`tests/test_orchestrator_plan.py` 15, `tests/test_adapter_forcing.py` 22.

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

### Package E — delivered 2026-09-26 (branch feat/structural-E, merged 89eb935)

Per case `metrics` (`ledger.usage_metrics` over the case's `calls` and
`tool_events` rows: tokens in/out, cache read, tool bytes shown, turns,
calls) and `smells` (`case_smells`: whole-file reads, repeated calls,
refused calls); the eval table has `tok`, `turns` and `smells` columns,
the summary and TRAJECTORY row carry tokens per case, `--repeat` reports
tokens mean ± spread, and the ledger report has the same numbers per
task. `python -m guru.evals matrix --models 'A|m1,B|m2'` runs the
selection once per model and prints passed, tokens, tool kB, seconds,
cost and the `contract` column read from `evals/models/<slug>.json`
(`summary.ok/summary.tools`; one slug implementation, `runs.model_slug`,
shared with the bench). `planted-failure-digest` and
`find-symbol-outline` no longer forbid `read_file`. `make eval-gate` =
fast + sandbox cases + dogfood, `--repeat 3`. `panel = false` in the
default routing block; `claude-tiers-panel.toml` is
`claude-tiers-review.toml`. Tests at the review round:
`tests/test_evals_runs.py` 55, `tests/test_evals_runner.py` 183,
`tests/test_evals_checks.py` 37, `tests/test_evals_cases.py` 48,
`tests/test_ledger_report.py` 62.

### Review round — delegate loop (open item)

Baseline runs 7730e6c1c39c and 689aecc6283a: the controller re-delegated
seven times on the dogfood case because every follow-up worker starts on
a fresh sandbox task copy (`verbs._task_key` keys copies by task id, and
`Orchestrator.on_done` removes a task's copies through
`verbs.cleanup_task` when the worker finishes), so the previous worker's
edits were gone. Done in code: `plan.MAX_DELEGATE_ROUNDS = 3` per request
(refusal text says to answer from the results) and the controller's
contract in the `plan` tool description. Open: a task that carries
`continue: "<agent title>"` and reuses that worker's sandbox copy. It is
not a small change — the copy would have to survive `cleanup_task`
(retention until the *request* ends, not the task), be re-keyed to the
new task id (or the child given the previous task id, which is the ledger
row's identity), and be released when the controller answers. Design it
against `verbs._copies`/`_tally` keying and the on_done cleanup before
touching either.

## Measurement after the merge

Suite as before (fast ×3, real + dogfood, sandbox, review-multi-file ×3)
with tokens per task as the headline, compared with suite 3 of
`evals/triage/2026-09-25-loop-1.md`; tool contract numbers for three
models; independent review of the diff; triage note
`evals/triage/2026-09-26-structural.md`. Budget ≈ $10. Note when reading
cache numbers: every worker's system prompt now differs by its brief slice
(and the controller's by the project map), so the system-block cache is
written once per worker instead of shared across them — expect a cache
write per worker and read the tokens-per-task headline with that in mind.
