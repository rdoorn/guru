# Eval trajectory

One row per recorded run (appended by `python -m guru.evals run`).

| ts | run_id | model | passed/total | mean seconds | cost | note |
|---|---|---|---|---|---|---|
| 2026-09-23T15:45:06+00:00 | 088625f91b8c | Ollama\|qwen3:14b | 5/6 | 48.8 | $0.00 | baseline subset qwen3:14b |
| 2026-09-23T16:08:29+00:00 | 1f4f8262a80a | Ollama\|huihui_ai/qwen3-abliterated:8b | 3/8 | 167.7 | $0.00 | iteration 2 on 8B: glob fix; remaining cases |
| 2026-09-23T17:05:52+00:00 | 384761c577f4 | SBP Litellm\|aws/claude-5-sonnet@125k | 8/9 | 81.3 | $4.12 | A: all remote sonnet-5 |
| 2026-09-23T18:59:24+00:00 | 04dbcf7566f9 | SBP Litellm\|aws/claude-5-sonnet@125k | 9/9 | 35.0 | $0.94 | config 0: plain sonnet-5, join+nudge fixes |
| 2026-09-23T19:04:43+00:00 | 3f9ab125dbeb | SBP Litellm\|aws/claude-5-sonnet@125k+routed:claude-tiers+controller | 7/9 | 18.8 | $0.38 | config 1: sonnet controller, claude tiers |
| 2026-09-23T19:07:36+00:00 | f1929d55c41a | SBP Litellm\|aws/claude-5-sonnet@125k+routed:claude-tiers-judges+controller | 7/9 | 26.8 | $0.54 | config 2: claude tiers + encoder judges shadow |
| 2026-09-23T20:18:12+00:00 | 3bff8cb9eb58 | SBP Litellm\|aws/claude-5-sonnet@125k+routed:claude-tiers+controller | 0/9 | 0.6 | n/a | config 1 rerun: controller knows cwd, complexity guidance |
| 2026-09-23T20:21:20+00:00 | c7473cbe8524 | SBP Litellm\|aws/claude-5-sonnet@125k+routed:claude-tiers+controller | 8/9 | 38.9 | $1.21 | config 1 rerun: controller knows cwd, complexity guidance |
| 2026-09-23T20:28:43+00:00 | ad9f5e9caece | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers+controller | 9/9 | 29.1 | $0.77 | config 3: haiku controller, claude tiers |
| 2026-09-24T07:57:12+00:00 | 807a96827d02 | SBP Litellm\|aws/claude-5-sonnet@125k | 2/3 | 352.8 | $8.20 | config 0 real cases: plain sonnet |
| 2026-09-24T08:07:47+00:00 | 65a0ce16e651 | SBP Litellm\|aws/claude-5-5-opus@125k | 3/3 | 196.2 | $12.38 | config 0-opus real cases: plain opus 5.5 |
| 2026-09-24T08:07:59+00:00 | 475fe1576de0 | SBP Litellm\|aws/claude-5-sonnet@125k+routed:claude-tiers-judges+controller | 3/3 | 184.4 | $4.71 | sonnet controller + tiers + judges, real cases |
| 2026-09-24T08:15:16+00:00 | 125934108305 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/3 | 166.7 | $2.78 | config 3+judges real cases: haiku controller |
| 2026-09-24T08:18:00+00:00 | 974716735cf6 | SBP Litellm\|aws/claude-5-5-opus@125k+routed:claude-tiers-judges+controller | 2/3 | 131.5 | $3.77 | opus controller + tiers + judges, real cases |
| 2026-09-24T08:55:47+00:00 | 660eb67121bb | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 1/3 | 96.9 | $1.95 | haiku controller v2 (examples+tie-break), repeat 1 |
| 2026-09-24T09:01:25+00:00 | 4dd3f4cb1061 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/3 | 102.1 | $2.58 | haiku controller v2 (examples+tie-break), repeat 2 |
| 2026-09-24T09:07:01+00:00 | 39a114b8a034 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/3 | 115.5 | $2.90 | haiku controller v2 (examples+tie-break), repeat 3 |
| 2026-09-24T17:59:40+00:00 | 69bc45821a7d | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 11/13 | 37.6 | $2.53 | audited tools v1: haiku controller, fast+edit+real cases |
| 2026-09-24T20:48:23+00:00 | 6f6fb88ab45b | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 2/3 | 42.0 | $0.70 | sandbox S4: three sandbox cases, first run (case expectations wrong) |
| 2026-09-24T20:51:29+00:00 | f2a4787b9f81 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 12/14 | 33.2 | $2.64 | S3/S4 comparison run: fast+edit+real vs 69bc45821a7d |
| 2026-09-24T20:53:35+00:00 | 09051dcb6083 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 1/1 | 56.7 | $0.29 | sandbox-unrelated-change with corrected expectations |
| 2026-09-24T21:01:52+00:00 | d0c352bbb675 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/3 | 43.4 | $0.80 | sandbox cases with SANDBOX_RULE in the prompt |
| 2026-09-25T06:08:07+00:00 | e2cc0483103a | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/3 | 36.3 | $0.54 | deletions through the gate; unrelated-change re-planted as a bait note |
| 2026-09-25T07:13:02+00:00 | 16ff3cc08cde | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 8/8 | 16.2 | $0.22 | top-10 gate: fast x3 with caching + rubric (repeat 1/3; rubric errored, max_tokens) |
| 2026-09-25T07:15:39+00:00 | a9353706790c | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 7/8 | 17.0 | $0.25 | top-10 gate: fast x3 (repeat 2/3) |
| 2026-09-25T07:18:04+00:00 | dd0ff914337d | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 7/8 | 17.5 | $0.25 | top-10 gate: fast x3 (repeat 3/3) |
| 2026-09-25T07:22:01+00:00 | 871dc26f9efa | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 1/1 | 26.4 | $0.07 | rubric judge verified after the max_tokens floor (logic-bug 2/2) |
| 2026-09-25T07:22:58+00:00 | fa5c42d05059 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/3 | 66.8 | $1.39 | top-10: real guru cases with caching + Haiku rubric judge |
| 2026-09-25T09:47:42+00:00 | 8bf355893d82 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 8/8 | 16.2 | $0.25 | loop 1: fast x3 (1/3), conversation cache breakpoint |
| 2026-09-25T09:50:41+00:00 | 9f122a20774b | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 8/8 | 17.0 | $0.24 | loop 1: fast x3 (2/3) |
| 2026-09-25T09:53:36+00:00 | 644b68facb64 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 8/8 | 20.1 | $0.25 | loop 1: fast x3 (3/3) |
| 2026-09-25T09:56:50+00:00 | 1de2ea5a68fa | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/4 | 122.3 | $1.21 | loop 1: real + dogfood; sandbox checks errored (runner bug) |
| 2026-09-25T10:07:46+00:00 | 94fdc1bb11a5 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 1/3 | 33.3 | $0.15 | loop 1: sandbox cases; checks errored (runner bug), stall answers |
| 2026-09-25T10:10:11+00:00 | 61c2e3bb32d3 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-panel+controller | 1/1 | 79.5 | $0.32 | loop 1: review-multi-file x3 with panel active (1/3) |
| 2026-09-25T10:12:03+00:00 | cc4411c400e3 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-panel+controller | 0/1 | 78.4 | $0.18 | loop 1: panel (2/3), spawned_min miss |
| 2026-09-25T10:13:39+00:00 | c1362a8ac39a | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-panel+controller | 1/1 | 86.2 | $0.32 | loop 1: panel (3/3) |
| 2026-09-25T10:50:18+00:00 | 94437405450e | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 8/8 | 15.8 | $0.27 | loop 2: fast x3 (1/3) |
| 2026-09-25T10:53:26+00:00 | c33bed3bb374 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 8/8 | 14.4 | $0.20 | loop 2: fast x3 (2/3) |
| 2026-09-25T10:55:55+00:00 | 30e56fd72375 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 7/8 | 18.2 | $0.29 | loop 2: fast x3 (3/3), one whole-file read miss |
| 2026-09-25T10:56:39+00:00 | 1279f9e29da6 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 1/1 | 18.5 | $0.05 |  |
| 2026-09-25T10:59:07+00:00 | fe9e21c94f43 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/4 | 145.7 | $1.05 | loop 2: real + dogfood; dogfood applied, fixture check wrong (TMPDIR inside repo) |
| 2026-09-25T11:00:54+00:00 | b7adcd223af0 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 1/1 | 20.1 | $0.03 |  |
| 2026-09-25T11:01:30+00:00 | 8274b6029b7b | SBP Litellm\|aws/claude-5-sonnet@125k+routed:claude-tiers-judges+controller | 0/1 | 15.1 | $0.03 |  |
| 2026-09-25T11:12:06+00:00 | c47045f45c31 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/3 | 51.7 | $0.23 | loop 2: sandbox 3/3 after the runner fix |
| 2026-09-25T11:15:41+00:00 | 9da2e374999f | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-panel+controller | 0/1 | 69.6 | $0.17 | loop 2: panel x3 (1/3), single worker |
| 2026-09-25T11:17:57+00:00 | 714c3ae9f649 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-panel+controller | 0/1 | 75.0 | $0.18 | loop 2: panel x3 (2/3), single worker |
| 2026-09-25T11:19:44+00:00 | a213997e4d9d | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-panel+controller | 0/1 | 67.2 | $0.16 | loop 2: panel x3 (3/3), single worker |
| 2026-09-25T12:04:28+00:00 | aadc848e66e6 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 7/8 | 26.0 | $0.35 | loop 3: fast x3 (1/3), security-only timed out once |
| 2026-09-25T12:08:42+00:00 | 7bedee6b8e36 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 8/8 | 15.3 | $0.23 | loop 3: fast x3 (2/3) |
| 2026-09-25T12:11:16+00:00 | 0e5cf459ac0a | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 7/8 | 14.8 | $0.22 | loop 3: fast x3 (3/3) |
| 2026-09-25T12:13:52+00:00 | ac6c18d35849 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/4 | 204.9 | $1.51 | loop 3: real + dogfood; dogfood applied, fixture tests fail |
| 2026-09-25T12:31:01+00:00 | 472fdc6f8ac1 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/3 | 38.3 | $0.19 | loop 3: sandbox 3/3 |
| 2026-09-25T12:33:45+00:00 | d13ae86ca63a | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-panel+controller | 1/1 | 83.3 | $0.32 | loop 3: panel x3 (1/3), two workers after the decomposition hint |
| 2026-09-25T12:35:53+00:00 | 68444551eccf | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-panel+controller | 1/1 | 60.2 | $0.26 | loop 3: panel x3 (2/3), two workers |
| 2026-09-25T12:37:28+00:00 | b9c241f9deab | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-panel+controller | 0/1 | 71.9 | $0.19 | loop 3: panel x3 (3/3), single worker |
| 2026-09-25T12:49:12+00:00 | c38e907de6a2 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 0/1 | 211.1 | $0.34 | dogfood rerun with pytest tail: guru tests see the provisioned image |
| 2026-09-25T13:09:38+00:00 | 04f76149b129 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 1/1 | 137.5 | $0.31 | dogfood rerun with private HOME: PASS, rubric 2/2 |
