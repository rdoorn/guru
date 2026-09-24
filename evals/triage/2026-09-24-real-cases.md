# Real cases on guru itself: five configurations (2026-09-24)

Cases: `guru-explain-gpu-fit`, `guru-review-adapters`, `guru-add-version-flag`
(git-pinned to guru @ dc0cd31). One run per configuration. Rubrics graded by
hand from the final answers (0–2 each, 6 max). All runs via "SBP Litellm".

| config | run | passed | mean s | cost | rubric |
|---|---|---|---|---|---|
| plain Sonnet 5 (guru as before: delegation hint, no controller) | 807a96827d02 | 2/3 | 352.8 | $8.20 | 5 |
| plain Opus 5.5 | 65a0ce16e651 | 3/3 | 196.2 | $12.38 | 6 |
| Haiku 4.5 controller + tiers + judges | 125934108305 | 3/3 | 166.7 | $2.78 | 6 |
| Sonnet 5 controller + tiers + judges | 475fe1576de0 | 3/3 | 184.4 | $4.71 | 6 |
| Opus 5.5 controller + tiers + judges | 974716735cf6 | 2/3* | 131.5 | $3.77 | 6 |

\* failed only `answer_regex` (wrote "recorded" instead of `record_call`); the
review itself was complete. The case regex is loosened to `record(_call|ed|ing)`.

## Per case

| case | plain Sonnet | plain Opus | Haiku ctrl | Sonnet ctrl | Opus ctrl |
|---|---|---|---|---|---|
| explain-gpu-fit | $0.24 / 37 s | $2.30 / 178 s | $0.51 / 73 s (Opus worker) | $0.68 / 114 s | $0.65 / 115 s |
| review-adapters | $4.76 / 530 s TIMEOUT | $8.51 / 294 s | $1.60 / 320 s (Sonnet worker) | $2.47 / 188 s (Opus worker) | $2.23 / 170 s (Opus worker) |
| add-version-flag | $3.20 / 492 s | $1.57 / 117 s | $0.67 / 108 s | $1.56 / 252 s | $0.89 / 110 s |

## Findings

- **Plain Sonnet over-reads.** On the review it read 87 files itself
  (42 `search_code`, 45 `read_file`) for ~450 s before spawning three
  reviewers, which the case timeout then cancelled after 47–72 s each
  (tag `over_read`). The controller design removes this by construction: a
  controller has no file tools.
- **A cheap controller is the win.** Haiku as controller with Sonnet/Opus
  workers matched plain Opus on rubric quality (6/6) at 22% of its cost and
  34% of plain Sonnet's, and was the fastest complete configuration.
- **Controller strength did not change output quality here.** Sonnet and
  Opus controllers wrote slightly more specific tasks (Opus: two-step
  implement-then-verify) but all three controllers reached 6/6; they cost
  1.4–1.7x the Haiku controller.
- **Labels are the weak spot.** Haiku labelled the GPU explanation `hard`
  (→ Opus, $0.51) and the adapter review `standard` (→ Sonnet); the encoder
  labels judge disagreed on both (review → hard 0.57; explain → 0.44/0.43,
  undecided). Sonnet controller: judge agreed 3/4. The judge's confidence is
  low (0.4–0.57), so it is a second opinion to log, not yet a router.
- **The judge is free.** Encoder judges added no measurable cost or time.

## Next

1. Repeat the Haiku-controller configuration three times (variance is real:
   the same case varied 2x in cost between runs).
2. Improve labelling: give the controller hint concrete examples per tier
   drawn from these cases, or promote the labels judge to break ties.
3. Add `over_read` guard for non-controller mode (stop reading after N
   distinct files and delegate) — or make controller mode the default when a
   ladder is configured.
