# Startup progress Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Show what guru is doing during startup (step, where each model
runs, how long it took), warm judges in the background with their state in
the TUI statusline, and keep library output out of the chat box.

**Architecture:** Domain (`guru/domain/startup.py`): the `Progress`
Protocol, host classification (local/remote) and the warm-up status value
with its statusline text. Endpoints: the Rich step reporter
(`guru/startup.py`), the thread-scoped stdout/stderr router
(`guru/judges/quiet.py`), `Adapter.location()` and judge
`describe()`. `cli.main` wires the steps; `tui` reads the warm-up status.

**Tech Stack:** Python 3.12, rich (spinner/status), prompt_toolkit
(statusline), pytest, flake8, mypy.

---

## Design (approved 2026-10-07)

- Foreground steps before the TUI opens, one line each, spinner while
  running, then `✓ title  detail  N.Ns` (or `✗` with the error):
  `settings, skills, ledger` → `adapters` (each enabled adapter with
  `local`/`remote` and host) → `main model` (model, adapter, local/remote,
  ctx, GPU / CPU spill for Ollama) → `judges` (each judge's `describe()`,
  "warming in background").
- Adapter notices (`Fitting…`, `Pulling…`, spill warnings, routing and
  migration notices) keep printing through `ui.console`; the spinner draws
  them above itself. Only the `Loading…` / `ready.` pair of the Ollama
  preload becomes a step detail (redundant with the step line).
- Judge warm-up stays on its daemon thread. `judges.warm_status()` is
  `idle` / `loading <name>` / `ready <secs>` / `failed <names>`; a change
  calls the registered listener (the TUI invalidates). Statusline segment:
  `judges: loading decide…`, `judges ready 8.1s` (for 10 s), `judges:
  decide failed (see log)` (stays).
- Failure = a `guru` log record carrying an exception from the warm-up
  thread while that judge warms (judges' `warm_up()` never raises by
  contract; they `log.exc`, which logs at debug), or a raising `warm_up`.
- Noise: a stdout/stderr router sends writes from the `guru-judge-warm-up`
  thread to the guru log (debug) while every other thread passes through.
  That one mechanism covers the gliner2 banner (`print`), transformers'
  `Device set to use mps` (a StreamHandler bound to the `sys.stderr` seen at
  import, so the router is installed first thing in `cli.main`) and the
  sdpa RuntimeWarning (`warnings` writes to `sys.stderr`); no verbosity
  change or warnings filter was needed. `patch_stdout` swaps in its own
  proxy during each `[main]` prompt, so the router is installed around
  that proxy too.
- Always on, no new setting (good defaults); `docs/defaults.md` states it.

## Tasks

### Task 1: Domain — `guru/domain/startup.py`

- `Progress` Protocol: `step(title) -> ContextManager[StepHandle]`,
  `StepHandle.detail(text)`, `paused() -> ContextManager`.
- `host_location(url: str | None, remote_default: bool) -> Location`
  (`Location(kind: 'local'|'remote', host: str)`; localhost, 127.0.0.0/8,
  ::1, 0.0.0.0 are local; empty url uses `remote_default`).
- `WarmStatus` (frozen dataclass: `state`, `name`, `seconds`, `failed`,
  `at`) and `status_text(status, now) -> str` (empty for idle and for
  ready older than `READY_SHOWN_S = 10`).
- Tests: `tests/test_startup_domain.py` — classification table, every
  status text, ready expiry.

### Task 2: Endpoint — Rich reporter `guru/startup.py`

- `RichProgress(console)`: `step()` shows `console.status` spinner, prints
  `  ✓ title  detail  0.4s` on exit, `  ✗ title  error` on exception
  (re-raised). `paused()` stops the spinner (for `ollama pull`).
- `PlainProgress(console)`: the outside-startup default — `detail()`
  prints the dim line it replaces; `step()` prints nothing extra.
- Module current reporter: `current()`, `use(progress)` context manager.
- Tests: `tests/test_startup.py` with a recording `Console(file=StringIO)`.

### Task 3: Adapter and judge descriptions

- `Adapter.location()` default: `host_location(getattr(self, 'url', None)
  or getattr(self, 'base_url', None), self.remote)`; Anthropic without
  base_url → `remote, api.anthropic.com` (`default_host`).
- `Adapter.describe() -> str`: `Ollama (local, localhost:11434)`.
- Judges `describe()`: decide/encoder/injection → `<name> (local,
  in-process)`; ollama-json → `<name> (Ollama, local|remote, host)`;
  llm → `<name> (<adapter location>)`.
- Ollama `_preload_and_fit` uses `startup.current().detail(...)`; pull
  wrapped in `startup.current().paused()`.
- Tests in `tests/test_adapters*.py` / `tests/test_judges.py`.

### Task 4: Warm-up status + noise control

- `guru/judges/quiet.py`: `install()` (idempotent per stream object) —
  `ThreadRouter` on sys.stdout and sys.stderr keyed on the warm-up thread
  name.
- `judges._warm_all` updates status per judge, detects failures via a
  thread-filtered log handler, calls the listener; `warm_status()`,
  `set_warm_listener()`.
- Tests: router routes by thread, status transitions, failure detection,
  listener calls.

### Task 5: Wiring

- `cli.main`: steps around the phases using `startup.RichProgress`.
- `tui`: statusline appends `status_text(judges.warm_status())`; listener
  invalidates the app; one timer redraw after `READY_SHOWN_S` drops the
  ready notice without a keypress. The router is also installed inside
  each `patch_stdout` block.
- `docs/defaults.md`: a "Startup output (no knob)" section.
- Verify: `.venv/bin/python -m flake8 guru tests` and
  `.venv/bin/python -m pytest -q`; one commit at the end.
