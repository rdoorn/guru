# Backlog

Low-priority ideas parked for later. Not scheduled.

## `(yarn)` annotation on extended context options — LOW

**Idea:** In `/context` (and maybe `/models`), label ceiling entries that are
YaRN-extended as `(yarn)` — e.g. `393,216 (yarn)` — so the user knows the top
of the range is extrapolated positional scaling (and recall may fade near the
max). Do not select it by default; it's the user's choice.

**Detect it from GGUF metadata** (via `ollama.show(model).modelinfo`):
`{arch}.rope.scaling.type == 'yarn'` (with `.factor` /
`.original_context_length`). Only annotate; do not change the values.

**Crucial caveat (why we did NOT do the naive version):** raising `num_ctx`
past a model's native length does **not** enable YaRN in Ollama — it just runs
past the trained length and degrades. YaRN must be baked into the GGUF, and
when it is, the build already reports the extended length as its
`context_length` — which guru already reads as the ceiling and already offers.
So this is purely a *labeling* enhancement over the already-offered range, NOT
a "2× the registered max" option. Offering an extended option for a model
without YaRN metadata (e.g. `qwen3:14b`, which only has `rope.freq_base`) would
produce degraded output and must be avoided.

**Effort:** small (read one metadata key, format the label).

---

## Other deferred items (from prior designs)

- **Narrower code-search / symbol tools** (`outline` / `find_symbol`) so the
  model fetches a function instead of a whole file — limits context at
  acquisition time rather than compacting after the fact. (From the
  tool-output-retention design.)
- **Non-thinking mode toggle** for agentic turns (option 2 from the Qwen
  tool-use investigation) — try if pre-activation + sampling don't lift local
  models enough.
- **Read-only "explore / fan-out" execution mode** and per-role/skill tool
  restriction — the third axis deferred from the roles/skills design.
- **Extract a shared orchestrator** from `tui.py` so the TUI and the benchmark
  share one spawn/check/join mailbox (the bench currently mirrors it).
