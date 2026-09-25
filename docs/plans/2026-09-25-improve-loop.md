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
  *(Iteration 2 changed "net in one submit" to "gross, per submit and
  cumulative over the task"; see below.)*

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

## Iteration 2 — gate fixes (2026-09-25)

An independent review of package B found the new gate rules weaker than
documented. Each finding, what changed, and the test that pins it
(`tests/test_gate.py`, `tests/test_sandbox_verbs.py`,
`tests/test_sandbox_colima.py` new, one test added to
`tests/test_sandbox_integration.py`).

1. **Thresholds were per submit; the copy is discarded after every apply,
   so a worker could split a mass deletion across submits.** `verbs` now
   keeps an in-memory `gate.Tally` per `(project, task)` — files deleted
   (emptied included), lines removed gross, files touched — fed on every
   *applied* submit and cleared by `cleanup_task`/`cleanup_all` (not by
   the copy's removal). `gate.rules(..., prior=tally)` →
   `destructive_flags(..., prior=...)` checks both thresholds against
   `now + prior` and names both numbers: `deletes 2 files now, 5 in this
   task (more than 3): c.py, d.py`. The `apply` event detail carries the
   running total (`applied; task so far: 2 file(s) touched, 2 deleted, 2
   line(s) removed`). Pinned by
   `test_deletions_split_across_submits_are_counted`,
   `test_removed_lines_accumulate_across_submits`,
   `test_cumulative_deleted_files_over_a_task`,
   `test_cumulative_removed_lines_over_a_task`.
2. **Net-removed was defeated by padding (300 `-` + 300 `+#`).** The
   threshold is now *gross* removed lines per submit
   (`DESTRUCTIVE_REMOVED = 200`, replacing `DESTRUCTIVE_NET_REMOVED`);
   added lines offset nothing. The flag still reports the added lines and
   how many of them are content (`gate.substantive`: neither blank nor
   comment-only) so the reader sees padding: `removes 300 lines (more than
   200); +300 added, 10 of them content`. The plan text above said "net";
   this is the recorded change. README updated. Pinned by
   `test_gross_removed_lines`,
   `test_padding_with_blank_and_comment_lines_does_not_offset`.
3. **Emptying a file evaded the delete flag, `deletions_requested` and the
   destructive counts.** `_sections` now yields `gate.Section` objects
   with an `emptied` flag (hunks remove every line, add none, no context
   line, not `+++ /dev/null`, not a new file — in both the parsed and the
   raw-scan path); `Section.gone` (deleted or emptied) drives the delete
   flag (`empties tests/test_x.py (3 lines; file kept)`), the deleted-files
   count, the test-file rule (`empties test file …`), `deleted_paths`
   (`path (emptied)`), and `stat`/`stat_text` (now 5-tuples with an
   `emptied` column; `| +0 -3 emptied`, `1 file(s) emptied: …` in the
   packet summary). Pinned by `test_emptied_file_is_a_delete_flag_and_
   destructive`, `test_emptied_needs_every_line_gone`,
   `test_emptied_in_an_unparsable_diff`, `test_stat_renders_emptied_files`,
   verbs `test_emptied_test_file_asks`.
4. **The baseline and the diff came from a `.git` the worker could rewrite
   (the copy was mounted rw including `.git`).** Two belts. (a)
   `colima.run` mounts the copy's `.git` read-only on top of the rw copy
   mount (`-v <copy>/.git:/work/.git:ro`, only when the copy has a `.git`
   directory); the integration test
   `test_git_dir_is_read_only_in_the_container` shows a write to
   `.git/HEAD` and an `rmtree('.git')` fail inside the container while a
   normal write succeeds and the host's `git diff` sees it. (b)
   `prepare_copy` records the baseline sha as a `baseline: <sha>` line in
   `COPY_MARKER`; `colima.check_baseline` compares it with `git rev-parse
   HEAD`, and both `diff` and `show_baseline` call it first — a mismatch
   raises `colima.BaselineChanged` (a `RuntimeError`), which every verb
   turns into `Refused: sandbox copy baseline changed; the copy was
   discarded — run a sandbox verb to make a fresh one` and discards the
   copy. A marker without a sha is refused too. Caveat recorded: the
   marker itself sits in the rw copy, so belt (b) alone would not survive
   a worker that rewrites both `.git` and the marker; with belt (a) the
   worker cannot rewrite `.git` at all. Pinned by
   `TestGitReadOnlyMount`, `TestBaseline` (real temp repos: a commit or an
   amend in the copy is refused by both readers), verbs
   `test_baseline_changed_is_refused_and_the_copy_discarded`.
5. **`code_health` read through symlinks.** The unchanged-file branch
   refuses when `target.resolve()` is not under `copy.resolve()`
   (`Refused: 'pkg/link.py' resolves outside the sandbox copy.`); a link
   that stays inside the copy still works. Pinned by
   `test_symlink_out_of_the_copy_is_refused`.
6. **`colima.show_baseline` had no unit tests.** `TestShowBaseline` on a
   temp repository: absolute path → None, `..` → None, `''` → None,
   missing file / file created after the baseline → None, content over
   `DIFF_OUT_KB` (truncated) → None, normal → the baseline content.
7. **Minor.** `_renames` runs only when `patch.parse` failed (a parsed
   diff has no rename, so a `-- foo` body line can never be read as a
   header pair; the `--- a / +++ b` header-pair rename test is kept and
   `test_renames_only_looked_for_in_an_unparsable_diff` shows the scan's
   false positive is never consulted). `is_test_path` also matches
   `tests_*/`, `testing/` and `*_tests.py` (`pkg/testing.py`, `mytests/`,
   `tests_/` stay non-tests). Dead code removed: `gate.health_flags` and
   `health.notable`; the single path is `gate.health_flags_from(deltas)`
   over `gate.health_deltas(...)`, used by `verbs._verdict`. Health deltas
   are computed only when the deterministic flags are not already
   suspicious, and at most `HEALTH_MAX_FILES = 20` Python files are
   measured per diff. `code_health(path=...)` passes `paths={path}` to
   `health_deltas`, which filters the parsed diff before reading any
   baseline. The anthropic `cached_messages` empty-last-block case belongs
   to another package and was skipped.

Coordinator addition (dogfood run): the container had no writable `HOME`
(uid 1000, read-only root), so guru's own tests could not run; `colima.run`
now sets `-e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache` (`CONTAINER_ENV`,
tmpfs, non-persistent) ahead of the caller's env, the rest of the
environment stays scrubbed. The `sandbox_run` tool description says argv
must be a JSON list of strings (a plain string is split on whitespace
only). Pinned by `TestContainerEnv` and an assertion in the new
integration test. The exact-argv assertions in `tests/test_sandbox.py`
and `tests/test_sandbox_provision.py` were updated for the two `-e`
pairs, and the provision `fake_run` fixture answers faked git calls with
a sha so `prepare_copy` can record a baseline.

`GATE_QUESTIONS` is unchanged; the pinned sha `c8292068e5e9fd25` stands.

## Iteration 3 — controller caching (2026-09-25)

Starting point (triage loop-1, "Caching"): over a suite the Haiku
controller made 104 calls, 274,915 uncached input tokens, 62,123 cache
writes and **0 cache reads**, while the Sonnet/Opus workers read 85% of
their input from cache. The live probe with a static prefix read fine from
step 3, so guru itself was changing the prefix between controller calls.

### Request dumping

`GURU_DUMP_REQUESTS=<dir>` (`guru.adapters.base.dump_request`, wired into
`litellm._complete` and `AnthropicAdapter._create`) writes each outgoing
request's kwargs as `<utc-ts>-<adapter>-<n>.json`. The kwargs are the
request body; the key lives in the SDK client (pinned by
`TestRequestDump`: an adapter constructed with `api_key='sk-…'` dumps a
file that does not contain it). Documented in README "Ledger and
decisions".

### Diff finding

One `explain-readme` run (Haiku controller, spend allowed, rubric off,
$0.051) produced six requests; the controller's were #1 (spawn), #2 (join,
turn ends waiting) and #6 (the mailbox synthesis turn). System block and
tools list were byte-identical across all three (`sys 5e883899`,
`tools e8786610`) — the system tail and tool activation were **not** the
cause. The messages were:

```
#2  system | user:"Summarise the rollback…" | assistant:null tool_calls=[spawn id=tooluse_xM3…]
           | tool(tool_call_id=tooluse_xM3…):"Spawned agent1 …"          <- cache_control
#6  system | user:"Summarise the rollback…" | assistant:"(used tools)"
           | user:"[tool spawn result]\nSpawned agent1 …"
           | assistant:"I've delegated this to a sub-agent. Let me wait…"
           | user:"[tool join result]\nWaiting for agent1 …"
           | user:"[joined results] …"                                     <- cache_control
```

Root cause: at the start of every turn the adapters rebuilt their native
history from the neutral messages with `to_openai_messages` /
`to_anthropic_messages`, which *flattened* past tool rounds to text
(`(used tools)` + `[tool X result]` user messages) because the neutral
messages carried no provider ids. Within a turn the native `tool_calls` /
`tool` messages were sent; on the next turn (the mailbox synthesis, and
every later user turn in the TUI) the same history arrived in a different
shape, so the cache written at the end of the previous turn was never a
prefix of the next request. In the TUI `conversation.apply_retention` made
it worse: it dropped every text-less assistant tool-call step after the
turn, so the tool results lost their calls.

A second, structural finding from the same dumps and the whole-suite
ledgers: the controller's requests are 2.7k–3.4k tokens, below Haiku 4.5's
4,096-token minimum cacheable prompt. Nothing is written until the mailbox
turn (5–7k), which is the controller's *last* call in an eval case — hence
"62k writes, 0 reads". No prefix fix can make Haiku read on those calls;
a controller on Sonnet (1,024 minimum) does read (below).

### Fix

- The neutral messages keep the provider's ids: `id` (and, for LiteLLM,
  the model's `raw_arguments` string) on each `tool_calls` entry,
  `tool_call_id` on the tool message. `litellm.native_round` /
  `anthropic.native_round` rebuild a round whose ids match as the native
  `tool_calls` + `tool` messages / `tool_use` + `tool_result` blocks; a
  round without ids (Ollama history, older transcripts) or with a missing
  result still flattens. `message_to_dict` persists `tool_call_id` so a
  resumed conversation keeps the shape.
- `apply_retention` keeps text-less assistant steps that carry tool calls
  (a few tokens each); truly empty steps are still dropped.
- Pinned by `TestNativeRoundRebuild` (`tests/test_adapters.py`): a whole
  LiteLLM / Anthropic turn with a tool call, then the next turn's
  translation equals the last request's messages up to the marker.

### Numbers

Same case, dumping on, after the fix (`b7adcd223af0`, $0.034): the
mailbox request #6 is now
`system | user | assistant:null tool_calls=[spawn] | tool(id) | assistant:"…" tool_calls=[join] | tool(id) | user:"[joined results]"`
and its prefix equals request #2 byte-for-byte (system, tools and messages,
marker position aside). Haiku controller ledger: 2,721 / 2,909 / 3,430
uncached, 0 read, 0 write — under the 4,096 minimum as predicted.

Sonnet as controller, same case (`8274b6029b7b`, $0.032; the case failed
its regex because the trivial-routed Haiku worker's answer missed it, not
caching-related):

| call | uncached | cache read | cache write |
|---|---|---|---|
| #1 spawn | 2 | 0 | 3,240 |
| #2 join (same turn) | 2 | 3,240 | 302 |
| #6 mailbox synthesis (next turn) | 2 | 3,542 | 438 |

The mailbox turn reads the whole previous turn (3,240 + 302 = 3,542) and
writes only the delivery; before the fix the shape change would have
limited the read to the first breakpoint (system + tools + request). The
Haiku "0 reads" is now a model-minimum question, not a prefix bug: either
accept it for the eval controller or run the controller on Sonnet.

### Also in this iteration

- Panel judge: `needs_security` is asked over the parent's request and the
  spawned task (`orchestrator.panel_text`, request first, blank-line
  separated; the task alone when the request is empty or identical), so a
  controller that strips "security" from the task it writes still
  triggers the security worker. Heuristic stays *no*. `turn.request_in`
  reads any agent's history. Pinned by `TestPanelText`.
- Spawn/join race: `Orchestrator.spawn` registers the children in a
  pending set synchronously (`_register_pending`, child marked busy) and
  `_start` drains it after appending; `do_join` / `do_check` use
  `children_of(parent)` = registered + pending, so a `join` in the same
  tool round as the `spawn` (headless front-ends run it inline on the
  worker thread) opens the barrier instead of answering "None of those
  are your sub-agents". `spawn_panel` uses the same set. Pinned by
  `TestSpawnJoinRace`, including an end-to-end run on an instant fake
  model that spawns and joins in one round.

Not done: Anthropic direct with `thinking` on resends previous turns'
tool rounds without their thinking blocks (the API strips those itself, so
the cache should still hit; not measured live). `apply_retention`'s
summarize/outline compaction still rewrites large tool results after a
turn by design, which changes the prefix from that point on.
