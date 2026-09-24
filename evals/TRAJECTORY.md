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
| 2026-09-24T08:07:59+00:00 | 475fe1576de0 | SBP Litellm\|aws/claude-5-sonnet@125k+routed:claude-tiers-judges+controller | 3/3 | 184.4 | $4.71 | sonnet controller + tiers + judges, real cases |
| 2026-09-24T08:07:47+00:00 | 65a0ce16e651 | SBP Litellm\|aws/claude-5-5-opus@125k | 3/3 | 196.2 | $12.38 | config 0-opus real cases: plain opus 5.5 |
| 2026-09-24T08:15:16+00:00 | 125934108305 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/3 | 166.7 | $2.78 | config 3+judges real cases: haiku controller |
| 2026-09-24T08:18:00+00:00 | 974716735cf6 | SBP Litellm\|aws/claude-5-5-opus@125k+routed:claude-tiers-judges+controller | 2/3 | 131.5 | $3.77 | opus controller + tiers + judges, real cases |
| 2026-09-24T08:55:47+00:00 | 660eb67121bb | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 1/3 | 96.9 | $1.95 | haiku controller v2 (examples+tie-break), repeat 1 |
| 2026-09-24T09:01:25+00:00 | 4dd3f4cb1061 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/3 | 102.1 | $2.58 | haiku controller v2 (examples+tie-break), repeat 2 |
| 2026-09-24T09:07:01+00:00 | 39a114b8a034 | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 3/3 | 115.5 | $2.90 | haiku controller v2 (examples+tie-break), repeat 3 |
| 2026-09-24T17:59:40+00:00 | 69bc45821a7d | SBP Litellm\|aws/claude-4-5-haiku@125k+routed:claude-tiers-judges+controller | 11/13 | 37.6 | $2.53 | audited tools v1: haiku controller, fast+edit+real cases |
