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
- `spend.set_spend_asker` — the once-per-run "allow remote model spend?"
  question (`[routing] spend_confirm = "ask"`). The TUI installs an
  interactive prompt; the benchmark and the eval runner install an auto-deny
  (a run that never pays). `guru.domain.spend` remembers the answer for the
  run.
- `policy.set_scanner` — the content scanner (`guru.scanners.secrets`) that
  finds secrets in task text (forces a local rung) and in tool results bound
  for a remote adapter (redacted with a typed marker). Installed by the CLI
  at startup when `[routing] secret_scan` is on; None disables scanning.
- `decisions.set_judge` — the judge per decision point (shadow or active
  mode), installed from settings by `guru.judges.install()` at startup.
  Whether a point *acts* on its judge is process config
  (`config.DECISIONS_ACTIVE`, category 3), not a property of the judge.
- `ledger.set_repository` — the ledger persistence backend, a JSONL
  repository in the CLI/TUI, a fake in tests. `guru.ledger_cli review`
  installs the JSONL repository of the directory it labels for the duration
  of the command and restores the previous one afterwards.
- `tools.set_policy` — the project's tool policy (`.guru/tools.toml`, loaded
  by `guru.repositories.settings.load_tools_policy`): which registry tools
  are enabled/disabled, the test runner and any subprocess-limit overrides.
  The CLI installs it at startup; the default (no file, or `set_policy(None)`)
  enables everything. `tools.execute_tool` consults it through
  `tools.is_enabled` and writes a `denied = "policy"` `tool_events` row for a
  refused call. The always-on tools (`search_tools`, `use_skill`, `spawn`,
  `check`, `join`) are never subject to it.

These are the dependency-injection points, and they already exist where
front-ends actually diverge.

## 3. Process-wide configuration — module globals (deliberately)

Single-valued, process-wide configuration lives as module-level state in
`guru.config` and is mutated in place when the user changes it:

- `config.MODE` (read-only / ask / auto),
- `config.ALLOWED_READ_DIRS` / `ALLOWED_WRITE_DIRS` / `ALLOWED_DOMAINS`
  (per-project allow-lists),
- `config.SAMPLING` / `SAMPLING_PER_MODEL`, `config.PREACTIVATE_TOOLS`,
  `config.FLAT_TOOLS`, `config.BENCH_MODEL_TIMEOUT`, the GPU-fit constants,
- `config.DECISIONS_MODE` / `DECISIONS_ACTIVE` / `DECISIONS_THRESHOLDS` /
  `DECISIONS_TIMEOUT_MS` — the decision seam's mode, which points act on
  their judge, the per-point `P(yes)` threshold and the active-mode wait
  budget (`[decisions]` in settings.toml),
- `config.PROC_TIMEOUT_S` / `PROC_CPU_S` / `PROC_MEM_MB` / `PROC_FSIZE_MB` /
  `PROC_OUT_KB` — the subprocess ceilings `guru.domain.procs.Limits` defaults
  to (`[tools.limits]` in settings.toml; a project's `.guru/tools.toml`
  overrides per call through the installed policy),
- `config.SECRET_SCAN` — mirrors `[routing] secret_scan` for the tool layer,
  and stays off (no scanner bound) when no `[routing]` table is configured
  (the typed `RoutingSettings` itself travels with the `Orchestrator`, which
  also holds the `AdapterRegistry` it resolves routes against).

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
