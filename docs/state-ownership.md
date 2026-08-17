# State ownership

guru keeps runtime state in a few well-defined places rather than threading a
settings object through every call. This note records *which* state lives
*where* and *why*, so the boundaries are intentional and not an accident.

There are three categories.

## 1. Per-agent runtime state — isolated (ContextVar)

The state that a parallel sub-agent MUST NOT share with its parent — the model,
the conversation (`messages`), token counts, the active tool set, the cancel
flag — lives in a `SessionState` held by a `contextvars.ContextVar`:

- `guru.session` is a module proxy over the current `SessionState`. A worker
  thread binds its agent's state with `session.use(agent.state)` for the
  duration of the turn, so `session.model`, `session.messages`, etc. resolve to
  *that* agent's state. See `guru/session.py` (and the typed facade in
  `guru/session.pyi`).
- `guru.ui.console` is the same pattern for output: each agent's turn binds its
  own rich Console via `ui.use_console(agent.console)`.

This is the only state that genuinely needs isolation, and it already is.
Sub-agents run in background threads concurrently; the ContextVar gives each
one its own view without any shared mutable globals.

## 2. Pluggable seams — injected at runtime

Behaviour that differs between front-ends (the TUI vs. the headless benchmark)
is injected through setters rather than hard-coded:

- `tools.set_spawn_handler` / `set_check_handler` / `set_join_handler` — the
  delegation mailbox, supplied by `guru.orchestrator.Orchestrator`.
- `tools.set_domain_asker` / `files.set_path_asker` — the permission prompts.
  The TUI installs an interactive asker; the benchmark installs an auto-deny.

These are the dependency-injection points, and they already exist where
front-ends actually diverge.

## 3. Process-wide configuration — module globals (deliberately)

Single-valued, process-wide configuration lives as module-level state in
`guru.config` and is mutated in place when the user changes it:

- `config.MODE` (read-only / ask / auto),
- `config.ALLOWED_READ_DIRS` / `ALLOWED_WRITE_DIRS` / `ALLOWED_DOMAINS`
  (per-project allow-lists),
- `config.SAMPLING` / `SAMPLING_PER_MODEL`, `config.PREACTIVATE_TOOLS`,
  `config.FLAT_TOOLS`, `config.BENCH_MODEL_TIMEOUT`, the GPU-fit constants.

These are the same for every agent in the process. guru is a single-user,
single-process CLI, so there is never more than one value of each. Making them
"injectable" would isolate nothing that needs isolating; it would only add a
settings object threaded through nearly every module (and every registry tool
function) for no behavioural gain, at real regression risk. Tests already
override these cleanly with `monkeypatch`.

## Why not full dependency injection?

DI's payoff — concurrent instances with different configs, multi-tenant
isolation, hidden-dependency clarity — does not apply to a single-process CLI
whose only state needing isolation (category 1) is already ContextVar-isolated
and whose front-end differences (category 2) already have injection seams. The
remaining globals (category 3) are genuinely process config. So the boundaries
above are the design, not a stepping stone to a settings-object rewrite.
