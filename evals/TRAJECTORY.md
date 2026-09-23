# Eval trajectory

One row per recorded run (appended by `python -m guru.evals run`).

| ts | run_id | model | passed/total | mean seconds | cost | note |
|---|---|---|---|---|---|---|
| 2026-09-23T15:45:06+00:00 | 088625f91b8c | Ollama\|qwen3:14b | 5/6 | 48.8 | $0.00 | baseline subset qwen3:14b |
| 2026-09-23T16:08:29+00:00 | 1f4f8262a80a | Ollama\|huihui_ai/qwen3-abliterated:8b | 3/8 | 167.7 | $0.00 | iteration 2 on 8B: glob fix; remaining cases |
