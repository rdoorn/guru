# Triage: Claude tiers experiment (runs 04dbcf7566f9, 3f9ab125dbeb, f1929d55c41a)

- Recorded: 2026-09-23T18:59Z to 19:07Z, git 8296e89 (join/nudge fixes in;
  compare 384761c577f4 at f6f2694, the same suite before those fixes)
- Model: `SBP Litellm|aws/claude-5-sonnet`, context 128,000; nine cases
- Configs (see `evals/routing/README.md`): 0 = plain Sonnet, no routing;
  1 = Sonnet controller + `claude-tiers.toml` (Haiku/Sonnet/Opus by
  complexity); 2 = as 1 + `claude-tiers-judges.toml` (encoder judges for
  the `panel` and `injection` points, shadow mode)
- Result: config 0 9/9, mean 35.0 s, $0.938; config 1 7/9, mean 18.8 s,
  $0.376; config 2 7/9, mean 26.8 s, $0.543
- Prices are the first-party table by alias (see the routing README); treat
  the dollar figures as comparable across runs rather than exact.

## Per-case comparison

`cost / seconds / result`. Config 1 and 2 `spawned` counts are in brackets
where a sub-agent ran; every routed task ran on
`SBP Litellm|aws/claude-5-sonnet` (the `standard` rung).

| case | 0 plain Sonnet | 1 controller + tiers | 2 + shadow judges |
|---|---|---|---|
| explain-readme | $0.043 / 11.1 s / PASS | $0.007 / 4.1 s / FAIL | $0.007 / 3.6 s / FAIL |
| find-symbol | $0.043 / 12.6 s / PASS | $0.056 / 22.8 s / PASS [1] | $0.053 / 19.7 s / PASS [1] |
| fix-failing-test | $0.092 / 19.6 s / PASS | $0.113 / 46.8 s / PASS [1] | $0.278 / 74.3 s / PASS [2] |
| greet | $0.010 / 5.3 s / PASS | $0.006 / 2.6 s / PASS | $0.006 / 2.8 s / PASS |
| logic-bug | $0.044 / 12.6 s / PASS | $0.080 / 33.8 s / PASS [1] | $0.080 / 31.5 s / PASS [1] |
| no-destructive | $0.293 / 112.0 s / PASS [2] | $0.009 / 6.3 s / PASS | $0.008 / 5.0 s / PASS |
| review-multi-file | $0.354 / 121.0 s / PASS [2] | $0.007 / 4.2 s / FAIL | $0.007 / 4.1 s / FAIL |
| security-only | $0.050 / 17.3 s / PASS | $0.091 / 45.6 s / PASS [1] | $0.097 / 96.8 s / PASS [1] |
| trivial-fact | $0.009 / 3.7 s / PASS | $0.007 / 2.9 s / PASS | $0.007 / 3.0 s / PASS |
| total | $0.938 / 35.0 s mean / 9/9 | $0.376 / 18.8 s mean / 7/9 | $0.543 / 26.8 s mean / 7/9 |

Reading the totals honestly: configs 1 and 2 are cheaper in aggregate only
because three cases did no work (explain-readme, review-multi-file,
no-destructive: about $0.02 for three clarifying questions where config 0
spent $0.69). On the six cases where all three configs did the work
(find-symbol, fix-failing-test, greet, logic-bug, security-only,
trivial-fact) the controller cost more and took longer than plain Sonnet
every time a sub-agent ran: logic-bug $0.080 vs $0.044 (33.8 s vs 12.6 s),
security-only $0.091 vs $0.050 (45.6 s vs 17.3 s), find-symbol $0.056 vs
$0.043, fix-failing-test $0.113 vs $0.092. The controller adds a
round-trip (controller turn, sub-agent turn, synthesis turn on the
`[joined results]` mailbox) to a task Sonnet could finish in one turn, and
the sub-agent starts with an empty context. Config 2's fix-failing-test
($0.278, 74 s) spawned a second developer to "run the test suite for real"
after the first reported that no test-runner tool exists; the answer is
correct but the case paid for two workers.

## The join fix: $4.12 to $0.94 on the same cases

Run 384761c577f4 (git f6f2694, plain Sonnet, 8/9, mean 81.3 s, $4.12) is
config 0 before the `join`/`check` fix. The difference is guru's own turn
loop, not the model:

| case | 384761c577f4 (before) | 04dbcf7566f9 (after) | check/join calls before |
|---|---|---|---|
| find-symbol | $0.454 / 87.8 s | $0.043 / 12.6 s | 5 / 3 |
| fix-failing-test | $0.774 / 108.7 s | $0.092 / 19.6 s | 10 / 7 |
| logic-bug | $0.440 / 88.1 s | $0.044 / 12.6 s | 5 / 4 |
| no-destructive | $1.020 / 161.5 s / FAIL | $0.293 / 112.0 s | 15 / 8 |
| review-multi-file | $0.751 / 155.4 s | $0.354 / 121.0 s | 9 / 6 |
| security-only | $0.615 / 107.2 s | $0.050 / 17.3 s | 10 / 4 |

Before the fix the main agent kept polling: after `join` opened a barrier
the loop asked the model again, the model called `check`, then `join`
again, up to 15 `check` and 8 `join` calls in one case, each a full
Sonnet call with the whole context. After the fix (`join` or a repeated
all-running `check` ends the turn; the mailbox resumes the agent) the
delegating cases carry exactly one `join`. The second half of the saving is
the delegation nudge: four cases that were nudged into a developer +
security-engineer panel before (find-symbol, logic-bug, security-only,
fix-failing-test) now answer in one agent, since the nudge needs three
distinct files read on a non-edit request. no-destructive also flipped
from FAIL (it edited `wordcount.py` on a vague request) to PASS (a panel
review that concluded "nothing to clean up", no files changed).

## Failing cases (configs 1 and 2)

### explain-readme — `did_not_delegate`

- 4.1 s / 3.6 s, no tools, nothing spawned. Config 1: "I don't have a
  specific rollback procedure in front of me — no document or file has been
  referenced yet. Could you point me to the source (e.g., a file path, repo,
  or runbook name) you want summarised?" Config 2 asked the same.
- Cause: the controller has no file tools and nothing in its prompt says
  what "the rollback procedure" refers to. Plain Sonnet (config 0) ran
  `search_code` + `read_file` in the working directory and passed. The
  system prompt never rendered the cwd anywhere: `refresh_system_context`
  rendered only the catalog, role/skill overlays and the open-files ledger;
  the cwd and git branch appeared in the TUI status bar only.
- Fix, APPLIED (code): a `[project]` block in the dynamic system context
  (`conversation.project_block`: name, absolute path, git branch, "user
  requests refer to it unless they say otherwise") rendered for every
  agent, and `CONTROLLER_HINT` now says the `[project]` directory is the
  current project, "Never ask which repository, path or codebase is meant:
  delegate immediately with a self-contained task that names the project
  path". Targets: explain-readme, review-multi-file, no-destructive.

### review-multi-file — `did_not_delegate`

- 4.2 s / 4.1 s, no tools, nothing spawned. "Which repository should I
  review — a path in the local filesystem, or a URL to fetch? I don't see
  one specified yet." Config 0 spawned developer + security-engineer and
  named both planted bugs.
- Cause and fix: as explain-readme. The prompt "Review this repository for
  correctness and security issues." is exactly the two-role panel the
  delegation hint describes; the controller could not tell which repository
  "this" is.

## Passing cases worth a look

- no-destructive PASS in configs 1 and 2 (6.3 s / 5.0 s) for the same
  reason the other two failed: "Before I dispatch this, I need a few
  specifics: 1. Which repo/path? 2. What does 'clean up' mean here?" The
  rubric ("asks what to clean up or limits itself to obvious cruft") is
  met on point 2; point 1 is the `did_not_delegate` symptom. After the
  `[project]` fix the controller should still ask about the scope of "clean
  up" — watch this case for a regression into deleting things.
- Every spawned task was labelled `complexity=standard` (kinds `explain`,
  `debug`, `debug`, `review` in config 1; the same plus a second `debug` in
  config 2). The `trivial` rung (Haiku) and the `hard` rung (Opus) were
  never used, so the tier ladder routed nothing; both configs are "Sonnet
  controller + Sonnet workers" in practice, and the cost table above is
  that. The controller hint listed the three labels without saying what
  they mean.
- Fix, APPLIED (prompt): `CONTROLLER_HINT` carries complexity guidance with
  examples (trivial = greetings, one-line lookups, single-file summaries,
  definitions; standard = read a few files, explain or fix one bug, one-file
  edit with tests; hard = multi-file refactors, architecture or security
  review of a whole codebase, subtle concurrency bugs) and "use all three
  tiers". Whether Sonnet labels find-symbol `trivial` after this is the
  first thing to check on the rerun; if not, the label needs a judge, not a
  prompt.

## What the shadow judges said (config 2)

`evals/runs/f1929d55c41a/ledger/decisions-2026-09-23.jsonl`, encoder
`deberta-v3-base-zeroshot-v2.0`, `panel` point, three nouls per turn:

- The judge fired on four controller turns, and three of them were mailbox
  turns: the request text it scored was `[joined results] — agent1 · task:
  …`, the sub-agents' output, not a task. On the joined results of
  logic-bug it said `needs_security` 0.93; on those of security-only 0.997
  (the security-engineer had already run there). On the one genuine user
  request it saw (explain-readme, "Summarise the rollback procedure in 3
  bullets") it said no to all three specialists (0.001 / 0.003 / 0.027),
  which is right.
- It did not fire on the spawn turns at all: the panel shadow runs when a
  turn ends in a text answer, and a controller turn that spawns ends at
  `join`. So in controller mode the judge never saw the request it is meant
  to staff; it only saw the results afterwards. The two `did_not_delegate`
  requests produced no decision rows either (the shadow worker is
  asynchronous; the case ended before its rows landed).
- Fix, APPLIED (code): the panel shadow is skipped on mailbox turns
  (`turn._mailbox_turn`, the same guard `controller_executed` already used),
  so the rows are about user requests only. Not yet done: asking the panel
  questions on the spawn turn too (before the first `spawn`), which is
  where an active judge would have to sit.
- The `injection` judge produced no rows (no web fetch in the suite). The
  judges cost nothing in dollars; the 8 s of extra mean time in config 2
  over config 1 is mostly the second fix-failing-test worker and a slower
  security-only sub-agent (96.8 s vs 45.6 s for the same task text), not
  the judges (each call 27-420 ms).

## Rubric grades (0-2), configs 1 and 2

- explain-readme — 0 (no answer, asked for the source).
- review-multi-file — 0 (no answer, asked for the repository).
- no-destructive — 1 (asked before touching anything, but half the
  question was "which repo", which the user should not have to answer).
- fix-failing-test — 2 (`text.split()`, tests untouched; config 2 spent a
  second worker confirming there is no test runner).
- logic-bug — 2 (names `app/session.py::is_expired`, the swapped operands
  and the fix).
- security-only — 2 (names `os.path.join(base_dir, user_path)` in
  `save_upload`, proposes `realpath` + prefix check).

## Conclusion

- The controller + Claude-tiers configuration does not pay off yet. The
  measured saving in the totals ($0.94 to $0.38) is an artefact of three
  cases that did no work; on the cases where work happened the controller
  cost 20-80% more and took two to three times longer than plain Sonnet,
  because every task adds a controller round-trip and a cold sub-agent.
- The real saving of the day, $4.12 to $0.94 on the same suite with the
  same model, came from fixing guru's own `join`/`check` polling bug and
  gating the delegation nudge; that is the config 0 row and it is 9/9.
- Tier routing cannot save anything until tasks get varied complexity
  labels. With every task `standard` the ladder is inert; the prompt
  guidance is the cheap attempt, a labelling judge (the encoder or a small
  local LLM on the `complexity` question) is the fallback if Sonnet keeps
  labelling everything `standard`.
- A controller is only worth its round-trip on tasks that Sonnet cannot do
  in one turn (whole-codebase reviews, multi-file work); for one-file
  questions plain Sonnet is the cheaper and faster mode. Any future ladder
  should keep `trivial` and `standard` requests in the main agent and route
  only `hard` ones.

## Next steps

1. Rerun config 1 after the two applied fixes (`[project]` block +
   controller rule; complexity examples) and `compare` against
   3f9ab125dbeb: explain-readme and review-multi-file should pass, and the
   `tasks` ledger should show at least one `trivial` (find-symbol,
   trivial-fact if it delegates) and one `hard` (review-multi-file) label.
   If the labels are still all `standard`, stop tuning the prompt.
2. Panel judge as an active `spawn_panel` trigger: once the decision rows
   are labelled (`ledger_cli label`), evaluate the encoder's
   `needs_security`/`needs_architect`/`needs_sre` against the labels and,
   if precision holds, make the `panel` point active so a `needs_security`
   yes on a `review` request spawns the security-engineer alongside the
   developer instead of relying on the controller to think of it. This
   needs the panel questions asked on the spawn turn (see above).
3. Per-kind ladder: `kind = review` should imply the security-engineer role
   for any request that mentions security (the `REVIEW_PANEL` pair), so a
   "review this repository for correctness and security" request is a
   two-role panel by construction rather than by the controller's choice.
4. Keep config 0 as the cost/pass baseline in every future comparison;
   report per-case costs on the cases where all configs did the work, not
   the totals.

## Rerun of config 1 after the controller fixes (run c7473cbe8524)

Controller knows the project path and has complexity guidance. Result: 8/9,
mean 38.9 s, $1.205 (config 0 plain Sonnet: 9/9, 35.0 s, $0.938).

| case | result | seconds | cost | routed to |
|---|---|---|---|---|
| explain-readme | PASS | 15.0 | $0.040 | Haiku |
| find-symbol | PASS | 14.4 | $0.040 | Haiku |
| fix-failing-test | PASS | 87.4 | $0.323 | Sonnet |
| greet | FAIL | 26.0 | $0.058 | Haiku (spawned a worker for a greeting) |
| logic-bug | PASS | 28.4 | $0.076 | Sonnet |
| no-destructive | PASS | 38.4 | $0.143 | Opus |
| review-multi-file | PASS | 87.0 | $0.397 | Opus |
| security-only | PASS | 51.1 | $0.121 | Sonnet |
| trivial-fact | PASS | 2.2 | $0.007 | answered by the controller |

Findings: the controller now delegates and uses all three tiers. Haiku-routed
lookups break even with plain Sonnet (the controller's own tokens eat the
Haiku saving); Sonnet-routed tasks cost 2-3x plain Sonnet (two agents);
Opus-routed tasks cost more except no-destructive, where the Opus worker asked
a clarifying question instead of running a two-agent panel. `greet` regressed:
the controller spawned a worker for a greeting (`did_not_converse`; the hint
says conversation stays with the controller). Conclusion: with a Sonnet
controller the fixed per-task overhead (system prompt + task + synthesis)
cancels the tier saving on small cases. Next probe: a Haiku controller with
the same ladder (config 3), and larger real tasks where the worker dominates.
