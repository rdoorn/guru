# Coding-model benchmark (stage 1) — design

Date: 2026-08-15

## Goal

Run several local Ollama coding models through guru on one fixed prompt —
"i want you to inspect current code in this repository, and tell me something
about code quality" — measure cost/behavior, and plot speed vs accuracy.

## Metrics per model

- tools called (names + count)
- tokens used (in / out) and tokens/sec
- time taken (wall-clock seconds)
- agents used (+ each agent's role/skill), and spawn-call count
- the full result text (for accuracy review)
- accuracy (0-100) — filled in a second, manual/Claude step
- error (if the run failed)

## Architecture

Headless — reuse guru's real turn loop (`adapter.run_turn`) so we measure the
actual search_tools/tool pipeline, not raw Ollama.

**Components**

1. `bench/models.txt` — curated, editable list (one `model[:tag]` per line;
   `#` comments). Seeded with the current coding models. If missing, the
   runner falls back to the coding models found in `ollama list`.

2. `guru/bench.py` — headless runner. Per model:
   - Fresh `SessionState`; `session.adapter = OllamaAdapter(); adapter.activate(
     model)` (uses the real GPU-fit context); a **quiet capture console** bound
     via `ui.use_console` (no TUI, output discarded/collected).
   - Build `messages = [system, user(PROMPT)]`, `tools.reset_active_tools()`,
     reset token counters.
   - Run the prompt through the **headless orchestrator** (below); time it.
   - Collect metrics and append a record to `bench/results-<ts>.json`
     (`accuracy: null`). Continue on per-model errors (record `error`).

3. **Headless orchestrator (compact, in `bench.py`)** — a small asyncio
   coordinator mirroring the TUI mailbox, with no-op UI callbacks:
   - `AgentManager` + a `_configure`-equivalent (fresh state/console per agent).
   - `spawn(task, role, skill)` → create a child agent, run its turn in a
     thread (`run_in_executor`), deliver its result to the parent's queue on
     completion; `check`/`join`/barriers as in the TUI.
   - Runs until the main agent and all sub-agents are idle with empty queues.
   - Tracks every agent created (title, model, role, skill) for the metric.
   - This duplicates ~100 lines of the TUI coordinator on purpose (keeps
     tui.py untouched, lower risk). A later refactor can extract a shared
     `guru/orchestrator.py` used by both.

4. Accuracy review (two-step): after a run, Claude reads the captured `result`
   fields, scores each 0-100 against the actual repo, and writes the scores
   back into the JSON (or a sidecar `scores-<ts>.json`).

5. `guru/bench_plot.py` — reads results (+accuracy) and renders **two** scatter
   PNGs into `bench/`: `speed_tokens.png` (x = tokens/sec) and
   `speed_walltime.png` (x = seconds). Both: y = accuracy (0-100), one dot per
   model, annotated `model (ctx)`. Missing accuracy → point skipped with a
   logged note.

## Data flow

`models.txt` → runner (per model: activate → orchestrated turn → metrics) →
`results-<ts>.json` → Claude fills `accuracy` → `bench_plot.py` → two PNGs.

## Metrics JSON schema (per model)

```json
{
  "model": "qwen3:14b",
  "num_ctx": 40960, "ctx_ceiling": 40960,
  "seconds": 42.1, "tokens_in": 5321, "tokens_out": 812,
  "tokens_per_sec": 19.3,
  "tools_called": ["search_tools", "list_tree", "read_file"],
  "tool_count": 7,
  "agents_used": 1,
  "agents": [{"title": "main", "model": "qwen3:14b",
              "role": null, "skill": null}],
  "spawn_calls": 0,
  "result": "…full answer text…",
  "accuracy": null,
  "error": null
}
```

## Error handling

- Per-model failure (pull/activate/turn) → record `error`, continue.
- A model that never produces an answer → `result` empty, `error` noted.
- Plot skips points with `accuracy == null`.

## Testing

- `models.txt` load: parse, comments, missing-file fallback.
- Metrics parsing: given a fake adapter turn that appends known `tool`
  messages and sets token counts, assert the collected record (tool list,
  counts, tokens, tokens/sec).
- Headless orchestrator: a fake adapter whose turn calls `spawn` → the child
  runs and its result is delivered; `agents_used == 2`; `join` returns the
  child result.
- Plot data-prep: results+scores → list of `(x, y, label)` points; null
  accuracy skipped; both axes computed.

## Dependencies

Add `matplotlib` (dev/bench dependency) via `uv`.

## Out of scope (later)

Multiple prompts / task suites, repeated trials + variance, automated accuracy
scoring, richer plots (context-size as dot size, cost axis), CI integration.
