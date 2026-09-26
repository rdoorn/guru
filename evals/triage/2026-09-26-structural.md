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
