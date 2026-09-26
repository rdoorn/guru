# Triage: structural round (2026-09-26)

Plan: `docs/plans/2026-09-26-structural-round.md`. Measurement results
are appended after the merge; this note opens with the case changes of
Package E so a run before and after them can be read side by side.

## Case changes

**`planted-failure-digest` and `find-symbol-outline` lose
`tools_used_none = ["read_file"]`.** The check measured a preference (the
model did not open a file), not the outcome the case exists for (naming
the failing test from the `run_tests` digest; locating `save_upload` and
its callers). Across the loop-1..3 suites (`2026-09-25-loop-1.md`) it was
the only reason a fast case ever failed: `planted-failure-digest` read the
whole file once in suites 2 and 3 and still named the failing test and
why (2/3 each, gate ok), and every graded answer of both cases scored 2/2.
A pass/fail on tool choice hides that. The prompts are unchanged ("Do not
read the source files" / "do not read whole files"), so runs remain
comparable; what changes is where a whole-file read lands:

- the table's new `smells` column counts, per case, whole-file
  `read_file` calls after an `outline` of the same path, identical
  repeated calls and refused calls (`ledger_report.tool_smells` over the
  case's `tool_events` rows), and the summary line totals them;
- the `tok`/`turns` columns show what the extra read cost, which is the
  number that matters — a read that costs 2k tokens and still answers is
  a smell to triage, not a failure to gate on;
- with Package C's structural read, a whole-file `read_file` of a file
  over 200 lines returns the outline anyway, so the old check would have
  passed vacuously.

Unchanged: `tools_used_all = ["run_tests"]` on `planted-failure-digest`
(an outcome: the digest must be the source of the answer) and
`tools_used_any = ["find_symbol", "outline"]` on `find-symbol-outline`.

## Routing file

`evals/routing/claude-tiers-panel.toml` is now `claude-tiers-review.toml`
with `panel = false`: the review ladder is what it measures, and the panel
point stays shadow until labelled rows say otherwise (the default routing
block written on first run says the same). Rows in `evals/TRAJECTORY.md`
under `+routed:claude-tiers-panel` predate the rename and ran with the
panel active.

## Gate

`make eval-gate` = `run --tags fast,sandbox --repeat 3` (eight fast cases,
the three sandbox cases and the `guru-sandbox-ledger-origin` dogfood
case, one selection so one aggregate and one exit code); ≈ $4 with a
Haiku controller and the rubric judge, needs Colima and
`EVAL_ARGS='--routing evals/routing/<file>.toml --allow-spend'`.
`make eval-fast` is unchanged.

## Suite S1 — structural round before the review fixes (merge 89eb935)

Runs d704579d4055 / 0b4ee5f1947b / fb4730bca891 (fast ×3), 7730e6c1c39c
(real + dogfood), 689aecc6283a (sandbox), 14b0b51fee72 / 6e97e8c2ebfd /
7970b9516698 (review-multi-file ×3 on `claude-tiers-review.toml`).
`tok` = all tokens the provider processed (in + out + cache read + cache
write): the model-agnostic context volume.

| section | passed | tok / case | calls | vs suite 3 (2026-09-25) |
|---|---|---|---|---|
| fast ×3 | 24/24 | 19.5k | 110 | 20.9k, 123 calls, 22/24 |
| real (3 guru cases) | 3/3 | — | — | 3/3 |
| dogfood | FAIL | 577k | 44 | applied in suite 3 rerun |
| sandbox (3) | 2/3 | 39.7k | 20 | 3/3 |
| review-multi-file ×3 | 2/3 | 59k | 25 | 2/3, 65k, 29 calls |

Fast subset: first suite with 24/24, 7% fewer tokens and 11% fewer calls
per case than suite 3; smells 1 across 24 cases (was a failing check).
`protocol_violation` = 0 over the whole suite. Cache read share: Haiku
16%, Sonnet 80%, Opus 84%.

Two failures are new control-plane behaviour, not harness bugs:

1. **Delegate loop (dogfood).** The controller answered every mailbox
   delivery with another `delegate`: 7 plan rounds, 7 workers, 577k
   tokens, no `sandbox_submit`. Each worker gets a fresh sandbox task
   copy, so the follow-up worker found the earlier edits gone and the
   controller re-delegated. There was no cap on delegate rounds per
   request. Fix in the review round: `MAX_DELEGATE_ROUNDS`, refusal text
   at the cap, `answer` fallback.
2. **Answer instead of delegate (dependency request).** Haiku returned
   `plan(outcome=answer, answer="I need to search for the sandbox tool…")`
   and stopped: the controller has only `plan`, and the removed hint had
   said so. The contract now lives in the plan tool's schema description
   ("you have no other tools; anything that needs a file, command,
   package, test or the sandbox must be delegated").

Independent review of the merged diff: FAIL on one Critical (duplicate
`plan` calls suppressed without running the handler, no round cap →
unbounded paid rounds; the "second identical plan accepted" path was
unreachable), plus Important 2–9 (empty `answer` rejected, matrix
contract column shape, brief `ConfigParser` interpolation, brief store
growth per fixture copy, controller never got the map, coverage
vocabulary false positives, range cap dropped 10×, second plan in a
round). All in the fix round; suite S2 after it.

## Suite S2 — after the review fixes (merge 84632da)

Runs 98e82daaac3d / 01b9b1acdf34 / b73678df582f (fast ×3), e855e6c39eba
(real + dogfood), b522cb0cff20 (sandbox), 3be4e7e72c6a / 9f179b1764e0 /
d140b0541df1 (review-multi-file ×3). Gate before the run: 2465 unit tests,
6 Colima tests, lint and mypy clean.

| section | S1 passed → S2 | S1 tok/case → S2 | S1 calls → S2 | suite 3 (25th) |
|---|---|---|---|---|
| fast ×3 | 24/24 → 24/24 | 19.5k → 17.9k | 110 → 104 | 22/24, 20.9k, 123 |
| real (3 guru cases) | 3/3 → 3/3 | — | — | 3/3 |
| dogfood | FAIL → PASS | 577k (no submit) → 791k (2 intended submits) | 44 → 59 | pass after rerun |
| sandbox (3) | 2/3 → 3/3 | 39.7k → 46.9k | 20 → 26 | 3/3 |
| review-multi-file ×3 | 2/3 → 0/3 | 59k → 42k (one worker) | 25 → 18 | 2/3 |

- `protocol_violation` = 0 across S2. Haiku rubric judge: 2/2 on all three
  real cases (was 1/2 each in the 25th's suites) and on the dogfood case.
- Dogfood: the delegate cap held (3 delegate rounds, the 4th refused, then
  `answer`); two submits, both `intended`, applied; fixture tests pass.
  Still 791k tokens because each follow-up worker starts on a fresh
  sandbox copy — the copy handover (`continue: <agent>`) is the open item
  that would cut this to one worker.
- Dependency request: passed; the plan schema now states the controller
  has no other tools.
- review-multi-file 0/3: the controller sent ONE task whose goal named
  both "correctness and security". That satisfied the coverage rule as
  written (every concern appears in some goal) but not its intent. Fixed
  after S2 (commit 5da3718): a delegate plan with fewer tasks than
  coordinated concerns is re-asked once. Rerun below.
- Cache read share: Haiku 10%, Sonnet 90%, Opus 81%.

### review-multi-file after the undersplit rule (runs 93e122e2f82c, 447a79aa6909, e2337608fbc5)

3/3 for the first time in any suite (previous best 2/3): the controller
now spawns a correctness worker and a security worker for "correctness and
security" in every repeat, because a plan with fewer tasks than
coordinated concerns is re-asked once. 69k tokens per case (two workers),
62 s, $0.26, rubric 1.7/2.
