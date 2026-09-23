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
