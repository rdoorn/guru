# Worker budget, capped handoffs, deliverables, agent retirement

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Workers turn exploration into written files inside their round
budget; a worker that still hits the cap hands its findings back instead
of nothing; plan tasks own explicit deliverables; finished sub-agents
leave the tab bar.

**Evidence (dogfood 2026-10-07, usage-dashboard request):** three build
workers (2x Opus, 1x Sonnet) each hit the 40-round cap in ~95 s: 0
writes, 39-46 reads, 1.7-2.2M input tokens, ~$4.14 total. Write tools
were active from ~round 23 (`search_tools("write file")`). Each returned
the 61-character cap line with `status: done`; the controller had to send
a fourth worker to find out nothing landed. Sub-agent tabs are never
removed (`AgentManager` has no remove).

**Architecture:** turn contract (`guru/adapters/turn.py`) owns the round
footer, the checkpoints and the capped handoff; one shared
`turn.tool_result()` replaces the three adapters' copies of
skip-or-execute. Plan schema (`guru/domain/plan.py`) gains
`deliverables`; the orchestrator checks them at the end of a task and
records `capped` / `incomplete` statuses honestly. `AgentManager` gains
an archive; the TUI orchestrator retires finished children.

---

## Task 1 — honest status (2b)

- `session.capped` set by the cap branch of `_drive`; reset per turn.
- `Orchestrator.on_done`: status `capped` when the child's last turn hit
  the cap. Ledger `finish_task` accepts it (status is free text; check
  readers: ledger report, evals metrics).
- Delivery header names it: `[result from agent7 · capped after 40
  rounds · task: …]`.
- Tests: orchestrator with a fake child whose session is capped.

## Task 2 — capped handoff (2a)

Scope (review): workers only — a sub-agent executing a task
(`session.task_id`); the main agent keeps the plain cap. A change counts
only once its tool succeeded (read off the result: `(sha:` for
write/edit, `Applied patch` targets for apply_patch/sandbox_submit,
`Deleted` for delete_file); `sandbox_run`/`sandbox_python` suppress the
checkpoint (editing the sandbox copy).

- Round footer on the first tool result of every worker round:
  `[guru] round 23/40 · files changed: 0`.
- Checkpoint (once) for a writing kind with no file changed at
  `CHECKPOINT_AT = 0.5` of the cap: stop exploring; write now or
  final_answer naming the blocker.
- Last call (once) at `cap - HANDOFF_ROUNDS` (2): your next call must be
  final_answer with a handoff (changed, found with file:line, remaining).
- If the cap still hits: the capped text is built in code — files read,
  files changed, the model's last text — so the controller gets a
  usable handoff with zero extra calls.
- Struggle key `budget_nudges` counts checkpoint + last-call notes.
- Tests: fake step/run_tools driving the loop past each threshold.

## Task 3 — write tools up front (1c)

- `tools.initial_tools(kind=...)`: for `WRITE_KINDS` (build, refactor,
  debug, docs) the pre-activated set adds `write_file`, `edit_file`,
  `apply_patch` (still subject to the project policy and kind hiding).

## Task 4 — deliverables (3b)

- `Task.deliverables: list[str]`; schema field "files this task creates
  or changes; it owns them and changes nothing else".
- Soft plan problems (re-asked once, then the plan runs, like coverage;
  checked on follow-up turns too): a build/refactor/docs task without
  deliverables; more than `MAX_DELIVERABLES = 3`; a path owned by two
  tasks of the same plan; a directory or `~` path; deliverables on a
  read-only kind. Soft so a stubborn controller is never trapped.
- Worker task text: `Deliverables (you own these; change nothing
  else): …`.
- At the end of a task (orchestrator): each deliverable must exist and be
  modified after the task started (mtime); otherwise status `incomplete`
  and the delivery lists the missing ones.
- Tests: plan parse/validate, task text, orchestrator check with tmp files.

## Task 5 — retire finished agents (4b)

- `AgentManager`: `archived` list, `all_agents()`, `archive(agent)`,
  `next_title()` (monotonic; titles never reused — barriers key on them).
- Orchestrator lookups (`agent_for_state`, `children_of`,
  `_security_seen`) use `all_agents()`; titles from `next_title()`.
- `Orchestrator.retire(agent)` hook, no-op by default (bench/evals keep
  every agent for metrics); the TUI archives a reported child whose
  status is `done`; `capped`/`error`/`incomplete` children stay until
  their tab has been viewed.
- Tests: manager archive + titles; TUI-style retire policy.

## Task 6 — controller facts (5)

- `CONTROLLER_HINT`: workers have a 40-round budget; size each task to
  one deliverable set (≤3 files) that a worker can write in that budget;
  sequence dependent tasks instead of running them in parallel; a capped
  or incomplete result says so — re-delegate the remaining part with the
  handoff, do not re-explore.

## Task 7 — the real use case as an eval

- Case: the usage-dashboard request, phase-1 slice (SQLite store +
  ledger wiring + tests), run against a snapshot of this repo; graded on
  deliverables existing, their tests passing, no edits outside scope,
  cost and rounds. Budget: $20 per run, ~$80 total.
- Baseline on `main`-equivalent code, then this branch; iterate on the
  logic while the result is not good enough; each run's diff kept on a
  separate branch.

## Verify

`make lint typecheck test` after each task; one commit at the end.
