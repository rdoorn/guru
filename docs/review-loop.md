# The review loop: from shadow judges to active decisions

guru's small closed-form decisions (is this reply a stall, does this task
need a security / architecture / reliability reviewer, is a fetched page a
prompt injection) are made by a heuristic while a small local judge answers
the same question in *shadow*. Both answers land in the ledger's `decisions`
stream. This note is the procedure for turning a shadow judge into the
deciding one, point by point, and for turning on per-kind routing ladders
once the data supports it. Design: `plans/2026-09-23-routing-framework-design.md`
(§3 ledger, §7 phase 5); the seam is `guru/domain/decisions.py`.

## 1. The promotion rule

A judge takes over a decision point only when all of these hold on the
labelled rows for that point:

- **at least 100 labelled rows** for the point (`labelled` in the report),
  so precision and recall are not noise;
- **the judge beats the heuristic** on both precision and recall against the
  labels (or equals one and beats the other) at the threshold you intend to
  configure;
- **the false-positive rate is acceptable for that point**. A false positive
  costs something different at each point: for `stall` it is one extra nudge
  (cheap; a few percent is fine), for `panel` it is a spawned specialist
  that was not needed (moderate), for `injection` a redacted or refused page
  (expensive to the user; keep it low).

Judge agreement with the heuristic (`agreement with heuristic`) is context,
not a criterion: a judge that always agrees adds nothing, one that disagrees
a lot is either much better or much worse, and only labels tell which.

## 2. Labelling with the CLI

Run guru with judges in shadow (`[decisions] mode = "shadow"` plus the
`[decisions.points]` table) for a while, then label:

```bash
.venv/bin/python -m guru.ledger_cli review --point stall --n 50
```

Each row shows the point and question, the judge's `P(yes)` and verdict, the
heuristic's answer, which one was used, and the input head (the reply or
task text the question was about). Answer with the **correct answer to the
question**, not with whether guru behaved well:

- `y` — the answer to the question is yes (for `stall`: the reply *is* a
  stalled preamble)
- `n` — the answer is no
- `s` — skip (unsure; the row stays in the queue)
- `q` — quit

Every `y`/`n` writes a `labels` row through `ledger.record_label`:
`target_id = <point>:<question>:<input_sha>` (the *decision key*), labeller
`user`, label `yes`/`no`, note `point:<point>;question:<id>`. The key is
derived from the input, so rows for an identical input share one label and
a labelled row is never shown again. Turn and task labels from `/good` and
`/bad` (`target_id` = turn or task id, label `good`/`bad`) live in the same
stream but are outcome verdicts; the report keeps the two apart.

`review` shows the newest unlabelled rows first. Aim for 100 or more per
point before reading the report; label in several sessions so that the rows
span different days, models and projects.

## 3. Reading the report

```bash
.venv/bin/python -m guru.ledger_cli report --point stall     # or: make ledger-report
```

One block per point and judge (a point that switched judges gets one block
per judge, scored on its own rows):

```
stall / ollama-json:qwen3:4b
  rows 312  used judge=0 heuristic=312  fallbacks none
  agreement with heuristic 81%
  queued_ms p50 2  p95 15  (n 312)
  labelled 120
  judge     precision 91%  recall 87%  f1 89%  fpr 5%  (tp 40 fp 4 fn 6 tn 70 undecided 0)
  heuristic precision 72%  recall 93%  f1 81%  fpr 23%  (tp 43 fp 17 fn 3 tn 57 undecided 0)
  threshold  suggested 0.60 (f1 92% on 120 rows)
```

- `rows` / `used` / `fallbacks`: how many decisions the point saw, how many
  the judge decided (only in active mode) and why it fell back (`timeout`,
  `error`, `no_judge`; `not_active` marks shadow rows of a point that is not
  switched on while the global mode is `active`).
- `queued_ms` is how long the rows waited for their judge worker (put to
  call start), separate from the judge's own `ms`. Shadow batches and
  active decisions run on different workers, so a high `queued_ms` on an
  active point means active decisions are arriving faster than the sidecar
  answers them; a high value with a normal `ms` is "worker busy", not
  "judge slow".
- `judge` and `heuristic` are scored against the `yes`/`no` labels with
  positive = `yes`: `fpr` is the false-positive rate, `undecided` counts
  labelled rows where that side had no answer (they are excluded from the
  four counts). The judge line uses the verdict as recorded (`chosen`): in
  shadow and active rows alike the judge's `P(yes)` is compared with the
  point's configured threshold at write time (the row's `threshold`
  field), so shadow rows already preview a threshold you set in
  `[decisions.thresholds]` before promoting the point.
- `threshold` is the value in 0.05..0.95 (step 0.05) that maximises F1 when
  the recorded `P(yes)` is re-thresholded on the labelled rows; ties break
  towards 0.5. Treat it as a starting point: check the F1 it reports against
  the judge line above, and prefer a slightly higher threshold when false
  positives are the expensive error for that point.

`bench/ledger_report.py` prints the same numbers as a Markdown table
("Judge vs review labels") next to the cross-day latency, cost and
fallback tables.

## 4. Switching a point to active

Once the rule in §1 holds, in `~/.guru/settings.toml`:

```toml
[decisions]
mode = "active"                 # judges decide the points listed below
timeout_ms = 1500               # max wait per decision; heuristic after that
[decisions.points]
stall = "ollama"
panel = "encoder"
[decisions.active]
stall = true                    # promoted; panel stays shadow
[decisions.thresholds]
stall = 0.6                     # from the report; default 0.5
```

In `active` mode `decisions.decide` runs the point's judge synchronously on
the *active* judge worker (shadow batches have their own worker, so a
queued panel or injection batch never delays an active decision) and waits
at most `timeout_ms`; the judge's `P(yes)` is compared with the point's
threshold and that verdict drives guru (for `stall`: whether the turn gets
a nudge). On timeout, a judge error or an undecided answer the heuristic
decides and the row says so (`used = "heuristic"`, `fallback_reason`).
Points not listed under `[decisions.active]` keep shadowing, so the same
ledger keeps collecting evidence for the next promotion.

Watch `fallbacks` in the report after switching. A `timeout` can mean two
things, and `queued_ms` tells them apart: a small `queued_ms` with timeouts
means the judge itself is too slow for the budget (raise `timeout_ms` or
pick a smaller sidecar model); a `queued_ms` near `timeout_ms` means the
active worker was still busy with the previous decision when this one was
queued (active decisions arriving faster than the sidecar answers - the
same fix, or fewer active points).

A per-point circuit breaker keeps a dead sidecar from costing `timeout_ms`
on every turn: after `breaker_timeouts` consecutive timeouts (default 5)
the judge is skipped for `breaker_cooldown_s` seconds (default 60) and the
rows say `fallback_reason = "breaker"`; one warning is logged when it
opens. After the cooldown the next decision tries the judge again, and a
decision that returns in time resets the count. A `breaker` count in the
report means the sidecar was down or overloaded for a stretch, not that
the judge answered wrongly.

Demotion is the same edit in reverse (`stall = false` or back to
`mode = "shadow"`); nothing in the ledger is rewritten.

## 5. Per-kind ladders

Routing uses one default ladder (`[[routing.ladder]]`) for every task kind.
Per-kind ladders (`[[routing.ladders.<kind>]]`) exist in the data model but
are off (`type_router = false`) until the ledger shows a kind that
consistently routes wrong. The evidence is in `bench/ledger_report.py`:

- **Task latency** per `(kind, complexity)`: a kind whose p95 sits far above
  the others at the same complexity is landing on a rung that is too small.
- **Fallbacks and retries**: fallbacks concentrated in one kind mean its
  default rung keeps failing there.
- **Judge vs labels** / `tasks --unlabelled` triage: `/bad` labels and
  `wrong_route` triage tags (evals §8.3) clustering on one kind.

When one kind stands out across at least a week of rows (not one bad day),
add a ladder for it and switch `type_router = true`:

```toml
[routing]
type_router = true
[[routing.ladders.review]]      # kind from the controller's KINDS
adapter = "SBP Litellm"
model = "aws/claude-5-sonnet"
max_complexity = "standard"
[[routing.ladders.review]]
adapter = "SBP Litellm"
model = "aws/claude-5-5-opus"
max_complexity = "hard"
```

Kinds without their own ladder keep using the default. Re-run the eval
suite (§6) before and after; the trajectory table shows whether the change
moved time, cost and pass rate for that kind.

## 6. The eval suite is the regression gate

Every change in this loop (a promoted judge, a threshold, a new ladder) is
a behaviour change, so it goes through `python -m guru.evals`:

```bash
.venv/bin/python -m guru.evals run --model 'SBP Litellm|aws/claude-4-5-haiku' --routing evals/routing/claude-tiers-judges.toml --allow-spend
.venv/bin/python -m guru.evals compare evals/runs/<before>.json evals/runs/<after>.json
```

The cases in `evals/cases/` pin the delegation, stall and tool behaviours
against frozen fixture repos; `compare` lists newly passing and failing
cases with time and cost deltas and appends to `evals/TRAJECTORY.md`. A
promotion that makes a case fail is reverted, whatever the report said.
Triage of failures uses `tasks --unlabelled` to find the transcripts:

```bash
.venv/bin/python -m guru.ledger_cli tasks --unlabelled --n 20
```

which prints, newest first, each finished task without a label: task text,
route (`adapter|model`) and the reasons that changed the route, status,
seconds, cost and the transcript path.
