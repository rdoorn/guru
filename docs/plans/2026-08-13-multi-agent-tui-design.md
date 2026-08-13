# Design: multi-agent, multi-viewport async TUI

Date: 2026-08-13

## Goal

Turn guru from a synchronous REPL into a full-screen TUI where submitting a
message runs an agent **in the background** while the prompt stays live, and
multiple agents each own a viewport you can switch between. A main agent can
spawn/delegate to sub-agents that run in parallel.

## Concurrency: asyncio (not threads)

The workload is I/O-bound (LLM/HTTP waits). The best foundation is **asyncio**:

- prompt_toolkit is asyncio-native (`Application.run_async`); agents as tasks
  on the same loop means no cross-thread marshaling and no locks around UI
  state.
- Clean cancellation (`task.cancel()`) maps directly to per-agent Ctrl+C and a
  main agent cancelling sub-tasks — threads can't cancel a blocking call.
- Cheap fan-out for a main agent spawning many sub-agents (`asyncio.gather`).

Adapters' `run_turn` become `async def` using the async SDK clients
(`AsyncAnthropic`, `AsyncOpenAI`, `ollama.AsyncClient`). Blocking tools
(`web_search`/`web_fetch`) run via `loop.run_in_executor`.

## Core objects

- **Agent** — replaces the single global `session`. Owns: id, title (2-word
  summary), adapter+model, conversation (neutral messages), active tools,
  context/token counters, an output **buffer** (scrollback), a status
  (`idle`/`thinking`/`error`), an input queue, and its current asyncio task.
- **AgentManager** — the list of agents, the active index, spawn/cancel, and
  message routing to the active agent's queue.
- **Buffer** — a growable list of formatted lines per agent; the output pane
  renders the active agent's buffer. Adapters write here instead of the global
  console (rich rendered to formatted text / string).

## UI (full-screen Application)

Layout, top to bottom:
- **Output pane** — scrollable `FormattedTextControl` bound to the active
  agent's buffer; updates live as its background task appends output.
- **Tab line** — `[main] [agent1] [2-word summary] …`, active highlighted.
- **Status line** — the existing model/ctx/tokens/dir/branch bar.
- **Input line** — always interactive; submit routes to the active agent.

Key bindings:
- Enter → queue the message to the active agent (ignored if empty).
- Ctrl+Left / Ctrl+Right → switch active viewport (swap the rendered buffer).
- Ctrl+C → cancel the active agent's current task only.
- Ctrl+D → quit.

## Execution flow

1. Submit → `manager.active.enqueue(text)`.
2. Each Agent has a worker coroutine: awaits its queue, runs
   `adapter.run_turn(agent)` (async), appending `* thinking…`, `* tool …`,
   and the answer to the agent's buffer as they happen.
3. Multiple agents' workers run concurrently on the loop.
4. A main agent can call a `spawn(task)` tool → creates a sub-Agent, enqueues
   the task, returns its id; results flow back when the sub-agent finishes.

## Phasing

- **Phase A** — full-screen async TUI with **one** agent: submit runs it in a
  background task, prompt stays live, output streams into a scroll buffer,
  Ctrl+C cancels the running task. Adapters converted to async. No tabs yet.
- **Phase B** — multiple agents + tab line + Ctrl+Left/Right + per-viewport
  buffers + input routing + a `spawn`/delegate tool for the main agent.

## Risks

- Interactive TUI behaviour can't be verified headlessly — the user is the
  test harness; expect iteration.
- Converting adapters to async touches all three providers; the neutral
  message format and tool translation stay the same.
- rich Markdown rendering must be captured to text for the buffer (render via
  a string Console) or replaced with prompt_toolkit formatting.
