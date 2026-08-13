# Session continuity — architecture and what comes next

## What this project is

`guru` is a local-first LLM chat agent with an on-demand tool directory and
pluggable provider adapters. It runs as a Python package (`python -m guru` or
`./start.sh`).

## Architecture

- `guru/session.py` — shared runtime state (active adapter, model, context
  accounting, conversation, active tools).
- `guru/config.py` — paths (`~/.guru/`, project `./.guru/`), `adapters.toml`,
  `GURU.md` assembly, network allow-list.
- `guru/ui.py` — console, status bar (scroll-region + prompt toolbar), model
  picker, key bindings, terminal modes, memory detection.
- `guru/domain/tools.py` — tool directory, `search_tools`, provider-neutral
  `active_specs()`, and `execute_tool` (allow-list gate + run).
- `guru/domain/conversation.py` — save/resume and hybrid compaction on the
  neutral message-dict format.
- `guru/adapters/base.py` — `Adapter` interface + `ModelInfo`.
- `guru/adapters/ollama.py` — Ollama provider (daemon check, on-demand pull,
  context/param-size/memory).
- `guru/adapters/anthropic.py` — Anthropic provider, `api_key` or `oauth`
  auth, neutral↔Messages-API translation, tool parity.
- `guru/cli.py` — prompt loop, slash commands, cross-adapter `/models`.

Providers are configured in `~/.guru/adapters.toml`. Selecting a model in
`/models` switches the active adapter+model. `/models` shows Ollama model
memory footprints, red when over 80% of system memory.

## Status

- Package refactor (Phase 1) and Anthropic adapter (Phase 2) are complete.
- **Not yet verified live:** actual Anthropic turns against a real endpoint
  (both `api_key` local and `oauth` enterprise) and the interactive `/models`
  rendering. The translation/loop follows the documented Messages-API shapes.

## Candidate next work

- Verify + debug the Anthropic adapter against real endpoints.
- Streaming the final answer (tool-call rounds stay non-streaming).
- `/clear` (reset conversation) and `/history` (show prior turns).
- More tools: `run_python`, `read_file`, `list_files`, `ollama_ps`.
- Prompt caching for the Anthropic adapter (stable system + tools prefix).
