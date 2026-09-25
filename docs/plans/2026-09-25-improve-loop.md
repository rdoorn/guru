# Improvement loop — plan (2026-09-25)

Ronald's ask: fix the five "what's next" items, review the result, improve
based on it, and repeat a few times; every round must end with the tests
run and a confident verdict on them.

Loop contract (every iteration):

1. Build on a fresh branch `feat/improve-loop-N` from `main`.
2. Gate: `make check` = lint + typecheck + unit tests + Colima integration
   tests. No commit without it; the report quotes the counts, never "green".
3. Measure: `make eval-fast` (fast ×3), the real guru cases, the sandbox
   cases, `review-multi-file` ×3 with the panel judge active, and the
   dogfood case. Budget: ≤ $10 per iteration, ≤ 3 iterations unless the
   last one still found something worth fixing.
4. Review: an independent agent reviews the whole diff (correctness,
   security, layering) and the run transcripts; findings become the next
   iteration's list together with what the measurements showed.
5. Triage note per iteration in `evals/triage/`, memory updated.

## Iteration 1 — findings that drive it

- Haiku never cached: 69 controller calls / 211k input tokens with 0 cache
  reads (fast ×3 + real runs of 2026-09-25). The controller prefix (prompt
  + few tool schemas) is under Haiku's 4096-token minimum, and no marker
  sits on the conversation, so the growing history is re-sent uncached on
  every step. Opus workers cached 25% of their input for the same reason.
- The Haiku rubric judge graded 2/6 where hand grades were 6/6; it sees
  only the answer text.
- The panel security worker has never fired in a run (eval routing file
  keeps `panel` in shadow).
- S2 integration test `test_provision_and_uv_add_through_the_proxy` flaked
  once (`hosts == set()`: a cached image build makes no network calls).
- A `destructive` rule kind (renames, mass deletions) is still open; the
  code-health rule from the CodeScene reading is not started.

## Iteration 1 — work packages (parallel, disjoint files)

**A. Evals** (`guru/evals/*`, `evals/*`, `Makefile`, `tests/test_evals_*`)
- Rubric packet carries evidence: files changed, fixture tests pass, tools
  used (names + counts), gate verdicts, spawned roles, cost and seconds —
  as a guru-computed block outside the untrusted answer fence; the grading
  instructions say the evidence is authoritative over the answer's claims.
- `python -m guru.evals grade RUN_ID --rubric SPEC [--rubric SPEC2]`:
  re-grades a stored run from its transcripts and observed data (no model
  re-run), prints a per-case table with one column per judge and the
  hand grade when `evals/rubric-labels.toml` has one; records labels rows.
  Ship `evals/rubric-labels.toml` with the 2026-09-24 hand grades (three
  real cases: 2, 2, 2) as the first entries and document how to add one.
- Dogfood case `guru-sandbox-ledger-origin`: fixture `git:guru@0256f9e`,
  `sandbox = true`, prompt "Add the tasks stream's `origin` column to
  `guru.ledger_cli tasks` output, with a test, run the tests, then submit";
  expects `sandbox_submit`, gate `intended`, files_changed includes
  `guru/ledger_cli.py` and a test file, fixture_tests_pass. Check first
  that a guru image builds in reasonable time (uv.lock without the `judge`
  extra); if it does not, report and leave the case tagged `slow`.
- `make check` target and README/evals README mention.

**B. Gate: code health + destructive rules** (`guru/domain/health.py` new,
`guru/domain/gate.py`, `guru/sandbox/verbs.py`, `guru/domain/tools.py`,
tests, README sandbox section)
- `health.py`: pure AST metrics per Python function — lines, cyclomatic
  complexity (branches + boolean ops), max nesting depth, arguments,
  return count — and a per-file summary. `function_health(node)`,
  `file_health(source)`, `delta(before_src, after_src)` → changed and new
  functions with before/after metrics and a verdict per function:
  `improved | unchanged | degraded` against fixed thresholds (lines > 60,
  complexity > 10, nesting > 4, args > 6 — a function crossing a
  threshold it was under before is `degraded`; a new function over a
  threshold is `degraded`; pre-existing debt untouched is not flagged).
- Gate rule kind `health`: computed from the diff plus the original file
  contents (the sandbox submit has both: the copy's baseline via
  `git show HEAD:path` and the working tree); informational reasons per
  degraded function, and `SUSPICIOUS_KINDS` unchanged; a degraded
  function makes `unclear` at most (asks), never auto-applies. Reviewer
  packet gets a "Code health" block (guru-computed, outside the fences).
- Verb `code_health(path='')` inside the sandbox project: the file's or
  the copy's changed functions' metrics and verdicts, so a worker can
  self-correct before `sandbox_submit`; digest ≤ 600 chars.
- Rule kind `destructive`: renames, more than 3 files deleted or more than
  200 lines removed net in one submit, deletion of any test file →
  `unclear` (ask), reasons name the files. Update `GATE_QUESTIONS` sha test.

**D. Caching + flaky test** (`guru/adapters/*`, `tests/test_adapters.py`,
`tests/test_sandbox_integration.py`, `guru/sandbox/provision.py` only if
the flake needs it)
- Conversation breakpoint: `cache_control` on the last message of the
  conversation (last user or tool-result block) for both adapters, in
  addition to system and tools; document the 4-breakpoint limit. Live
  probe on `aws/claude-4-5-haiku`: a controller-shaped conversation (guru
  system prompt + controller tools + 6 turns) → cache reads appear once
  the prefix passes 4096 tokens; record the numbers in this plan.
- Flake: make the assertion conditional on the build having happened
  (`rec.built` / build seconds > 0) or force a rebuild in the test so the
  proxy is exercised deterministically; run the sandbox suite three
  times in a row and report all three.

**C. Panel measurement** (coordinator): `evals/routing/claude-tiers-panel.toml`
= judges file with `panel = true` under `[decisions.active]`; run
`review-multi-file` ×3 after the build; count `origin = panel` rows.

### D — results (2026-09-25)

Conversation breakpoint shipped in both adapters: `anthropic.cached_messages`
puts `cache_control` on the last content block of the last user message
(the user's text, or the last `tool_result` of a tool round);
`litellm.cached_messages` turns the last user *or tool* message into a
one-part content list carrying the marker. Together with the system block
and the last tool definition that is 3 of Anthropic's 4 breakpoints per
request; the marker moves forward every round, so the previous round's
position stays a read point and each request writes only what was
appended. `cache = false` on the adapter record sends no markers at all.

Live probe (`~/.claude/jobs/eb164789/tmp/cache_probe_conversation.py`,
`aws/claude-4-5-haiku` through the SBP LiteLLM proxy): real
`build_system_prompt()` (~690 tokens) + the four controller tool schemas
(~470 tokens) + one user request, then six `check` rounds with ~800-token
tool results; every step sent twice. The proxy accepted the marked text
part on `role: tool` messages (forwarded as a marked `tool_result`), so
tool messages are marked, not only user messages.

| step (last msg) | prompt | pass 1 uncached / write / read | pass 2 uncached / write / read | cost p1 → p2 |
|---|---|---|---|---|
| 1 (user) | 1871 | 1871 / 0 / 0 | 1871 / 0 / 0 | $0.0053 → $0.0058 |
| 2 (tool) | 3487 | 3487 / 0 / 0 | 3487 / 0 / 0 | $0.0065 → $0.0064 |
| 3 (tool) | 5130 | 5 / 5125 / 0 | 5 / 0 / 5125 | $0.0073 → $0.0009 |
| 4 (tool) | 6774 | 5 / 1644 / 5125 | 5 / 0 / 6769 | $0.0054 → $0.0016 |
| 5 (tool) | 8420 | 5 / 1646 / 6769 | 5 / 0 / 8415 | $0.0059 → $0.0018 |
| 6 (tool) | 10065 | 5 / 1645 / 8415 | 5 / 0 / 10060 | $0.0058 → $0.0042 |

Steps 1-2 sit under Haiku 4.5's 4096-token minimum and are not cached
(as the findings predicted: nothing about the ~1.2k-token controller
prefix is cacheable on Haiku). From step 3 the whole history is written
once and read back: the first pass of each later step reads the previous
step's prefix and writes only the ~1.6k appended tokens; the repeat pass
reads everything and leaves 5 uncached tokens. Baseline with the system
marker only (`mode=none`, same conversation): 0 cache writes and 0 reads on
all twelve calls, 1871 → 10065 uncached tokens per call, $0.0059-$0.0138
per call at steps 3-6 versus $0.0009-$0.0042 for the cached repeat.
Sonnet/Opus workers (1024-2048 minimum) cross the threshold a step or two
earlier. Cost header (`x-litellm-response-cost`) present on every call.

Flake `test_provision_and_uv_add_through_the_proxy`: the failure is not a
provision() cache hit but Docker's layer cache. The fixture had no
dependencies, so the only network layer was `RUN pip install uv`, which is
shared with every image built from the same base — including the real guru
project's sandbox image (`guru-sandbox/guru-*`, built by package A's
dogfood case). With that image present the build made no network calls at
all and the test failed on every run (reproduced: 1 failed, 4 passed in
31.5 s). Fix in the test only: the fixture depends on `iniconfig==2.0.0`
(a real uv.lock entry, wheel from `files.pythonhosted.org`), and
`pyproject.toml` carries a per-run comment so the `COPY pyproject.toml
uv.lock` layer never hits a leftover fixture image's cache; the build
assertion is `'files.pythonhosted.org' in hosts <= {pypi.org,
files.pythonhosted.org}` (the `pypi.org` index hit belongs to the
cacheable pip layer). `make test-sandbox` three times in a row with the
guru image present: 5 passed in 43.7 s, 5 passed in 43.0 s, 5 passed in
53.2 s. No `built` field was needed in provision.py / sandbox_images.py.
