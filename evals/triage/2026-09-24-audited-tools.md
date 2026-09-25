# Triage: audited tools v1 (run 69bc45821a7d)

- Recorded: 2026-09-24T17:59Z, git 6bed035 (audited verbs, tool_events
  stream, `.guru/tools.toml` policy, disabled tools not advertised)
- Config: `evals/routing/claude-tiers-judges.toml` — Haiku 4.5 controller,
  Claude tiers for the workers, labels judge as tie-breaker; 13 cases
  (`fast` + the edit cases + the three `real` guru cases)
- Result: 11/13, mean 37.6 s, $2.53. Both failures are check artefacts (the
  answers were correct); no taxonomy tag applies, the cases are adjusted
  below per the README guardrail (never edit a case without a note here).

## Results

| case | result | s | cost | worker rung | failed check |
|---|---|---|---|---|---|
| greet | PASS | 4.3 | $0.005 | — | |
| trivial-fact | PASS | 3.1 | $0.005 | — | |
| find-symbol | FAIL | 8.6 | $0.022 | haiku | `tools_used_any` |
| find-symbol-outline | PASS | 22.3 | $0.134 | opus | |
| explain-readme | PASS | 16.5 | $0.125 | opus | |
| logic-bug | PASS | 19.0 | $0.057 | sonnet | |
| security-only | PASS | 47.6 | $0.109 | sonnet | |
| planted-failure-digest | PASS | 20.0 | $0.103 | opus | |
| fix-failing-test | PASS | 30.7 | $0.215 | haiku, opus | |
| edit-then-verify | FAIL | 43.8 | $0.304 | opus | `files_changed` |
| guru-explain-gpu-fit | PASS | 63.8 | $0.359 | sonnet | |
| guru-review-adapters | PASS | 101.2 | $0.707 | opus | |
| guru-add-version-flag | PASS | 108.6 | $0.385 | sonnet | |
| **total** | **11/13** | **37.6 mean** | **$2.53** | | |

Rubrics (hand-graded, 0–2): edit-then-verify 2 (comparison became
`now >= expires_at`, answer quotes "all 7 tests pass"), fix-failing-test 2
(`text.split()`, test file untouched), guru-explain-gpu-fit 2 (two probe
loads, KV slope, `config.save_model_ctx` → `~/.guru/model_ctx.json`),
guru-review-adapters 2 (five concrete divergences with the ledger risk for
each), guru-add-version-flag 2 (`action="version"` on `guru.__version__`,
new test, reports the run_tests digest). 6/6 on the guru cases again.

## The two check artefacts

- **find-symbol** — `tools_used_any = ["search_code", "read_file"]` failed
  because the Haiku worker used `find_symbol` alone (evidence:
  `tools_used: spawn, join, find_symbol`) and answered correctly
  (`app/upload.py` line 5; callers in `handlers.py:32`, `test_upload.py`).
  The case predates the audited verbs; `find_symbol` is exactly the tool we
  want here. Fix applied: `find_symbol` and `outline` added to the list.
- **edit-then-verify** — `files_changed = ["app/session.py"]` failed on
  `unexpected ['tests/test_session.py']`. The Opus worker fixed the
  comparison and added two real expiry tests (`test_is_expired_false_before_
  expiry`, `test_is_expired_true_at_and_after_expiry`) because, as it
  reported, "the tests weren't actually failing ... the existing test only
  checks that is_expired returns a bool". That is the fixture's design
  (FIXTURE.md: a reviewer who adds an expiry test will see it fail against
  the planted code) and the right developer behaviour; nothing else changed
  and the suite was green (7 passed). Fix applied: the exact set is replaced
  by `files_unchanged = ["app/handlers.py", "app/upload.py",
  "tests/test_upload.py"]`; the rubric and `fixture_tests_pass` still pin
  the session.py change.

## Comparison with the 2026-09-24 real-case runs

Same three guru cases, same Haiku-controller configuration; the difference
is the audited toolset (outline, find_symbol, run_tests, check_syntax, …)
and the prompt lines that steer workers to them.

| | Haiku v2 mean of 3 (660eb…/4dd3…/39a1…) | today 69bc45821a7d |
|---|---|---|
| passed | 7/9 | 3/3 |
| cost (three guru cases) | $2.48 | $1.45 ($0.385 + $0.359 + $0.707) |
| mean s | 105 | 91.2 |
| rubric | 5.7 | 6 |

- Cost down 42% with the same rubric quality. The adapter review went from
  $1.60–2.23 to $0.71: the reviewer read ranges (three chunks of
  `ollama.py`, one each of `litellm.py`/`anthropic.py`) instead of files.
- Yesterday's worker slip (v2 repeat 1: the version-flag test was written as
  a `python -m guru.cli` subprocess and never run, suite red) did not recur
  in effect: the worker again wrote a subprocess test, but `run_tests` ran
  (987 passed, `test_misc.py` 3/3) and `check_syntax` was called after each
  edit, so the case passed with the fixture suite green. `run_tests` was
  called 10 times across the run; every edit case verified before reporting.
- Labels: the controller labelled `find-symbol` trivial (Haiku, $0.022,
  8.6 s) and its outline twin standard→Opus ($0.134) — same question, 6x
  the cost; tier examples in the hint still leave room.

## Tool audit (ledger `tool_events`, 190 KB shown in total)

| tool | calls | bytes shown | per call |
|---|---|---|---|
| read_file | 32 | 92.7 KB | 2.9 KB |
| search_tools | 11 | 53.8 KB | 4.9 KB |
| search_code | 10 | 21.3 KB | 2.1 KB |
| outline | 15 | 12.0 KB | 0.8 KB |
| find_symbol | 7 | 4.3 KB | 0.6 KB |
| spawn / join | 13 / 13 | 1.6 KB / 1.1 KB | ~0.1 KB |
| run_tests | 10 | 1.0 KB | 104 B |
| edit_file | 6 | 1.0 KB | 168 B |
| check_syntax | 5 | 0.7 KB | 132 B |
| list_tree | 2 | 0.6 KB | 0.3 KB |

- `read_file` and `search_tools` are 77% of the bytes. 20 of the 32 reads
  were ranged (the "outline first, then read the range" steer works: 15
  outline calls), but the review case's ranges were 150–260 lines each
  (8–13 KB); 12 whole-file reads were all small files (< 3.1 KB).
- `search_tools` cost 1.9–6.7 KB per call because it repeated every matched
  tool's description and parameter block — and a broad phrase ("read file
  contents", "outline source file and find symbol") fell back to listing
  all 18 tools. The schema reaches the model through the activated spec
  anyway, so this was pure duplication.
- The verification verbs are cheap: `run_tests` and `check_syntax` digests
  are ~100 B per call; 15 calls cost 1.7 KB together. The old failure mode
  (assert instead of verify) is gone at no context cost.

## Fix applied in this triage

- `search_tools` now returns a digest: at most six `name — first sentence`
  rows plus the "These tools are now active" line, and activates exactly the
  six it lists (`SEARCH_TOOLS_LIMIT`, `_top_matches`). Output for any query
  is < 800 chars (was 1.9–6.7 KB); with 11 calls per run that is ~45 KB of
  worker context saved. Tests in `tests/test_tools.py::TestSearchToolsDigest`,
  docs in `docs/tools.md`.

## Recommendations

1. Keep the audited verbs preactivated for workers (already the case:
   `PREACTIVATE_TOOLS` holds list_dir/list_tree/read_file/search_code/
   outline/find_symbol/run_tests/check_syntax). 9 of the 11 `search_tools`
   queries ("read file contents", "grep search code" x2, "find symbol
   definition" x2, "outline ..." x2, "list directory files", "run tests")
   asked for a verb the worker already held; only the two "edit file"
   queries were needed. The system prompt still says "each turn you begin
   with a single tool: search_tools ... never call a tool it has not
   returned", which contradicts preactivation and buys a round-trip per
   hop. Applied in this triage: `SYSTEM_PROMPT` now says the listed tools
   are directly callable and `search_tools` is for a capability that is
   not listed (editing, web access); pinned by
   `tests/test_tools.py::test_prompt_makes_listed_tools_directly_callable`.
2. Revisit `read_file` volume next: it is half the bytes. Outline-first is
   being used (15 calls) and 20/32 reads were ranged, so the remaining lever
   is range size on reviews — a `read_file` cap (e.g. 120 lines per call
   unless `lines` is explicit) or an outline-based nudge when a range
   exceeds it.
3. Rerun this suite twice more for variance before comparing further
   configurations; the three guru cases varied ±20% in cost yesterday.
