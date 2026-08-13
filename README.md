# guru

Local Ollama chat agent with an on-demand tool directory. The model starts
each conversation with one meta-tool (`search_tools`) and discovers further
tools as it needs them — designed to scale to 200+ tools without loading every
schema into context on every request.

## Quick start

```bash
uv sync          # one-time: creates .venv and installs deps
./start.sh       # launch against the default model
./start.sh --model qwen3:8b   # override the model
```

Requires the Ollama app running in the menu bar.

## Input

| Key | Action |
|-----|--------|
| `Enter` | Submit |
| `Shift+Enter` | New line (requires iTerm2, Kitty, WezTerm, or similar) |
| `Escape` → `Enter` | New line (works in any terminal) |
| `Ctrl+C` | Cancel current input |
| `Ctrl+D` | Exit |
| `↑` / `↓` | History (persisted to `~/.guru_history`) |

## Slash commands

| Command | Action |
|---------|--------|
| `/search <query>` | Call `web_search` directly and optionally `web_fetch` a result |
| `/models` (or `/model`) | Interactive model selector (↑/↓, Enter, Esc) |
| `/save` | Save the current conversation to disk |
| `/resume` | Restore a previously saved conversation (interactive selector) |
| `/compact` | Shrink the conversation to free up context now |
| `exit` / `quit` | Exit |

## Context management

A status bar pinned to the bottom of the screen shows session state:

```
🤖 qwen3-abliterated-32k | 💪 8.2B | 🧠 28% ███░░░░░░░ | ↓ 289898 | ↑ 784 | 📁 guru | 🌿 main
```

model · parameter size · context fullness (coloured green/yellow/red) ·
session input tokens · output tokens · current directory · git branch. During
model generation it stays fixed via a terminal scroll region; at the input
prompt the same info is shown in the prompt toolbar.

- **Window size** — guru resolves the effective `num_ctx` from the model's
  modelfile (falling back to `4096`), capped at the model's architecture
  ceiling. Override with `--num-ctx N`.
- **Auto-compaction** — when occupancy crosses 85%, guru compacts between
  turns: it drops old reasoning traces, evicts stale tool outputs, and, if
  still too large, folds the oldest turns into a summary. Recent turns and the
  system prompt are always kept. Trigger it manually with `/compact`.

## Configuration

**Global** — `~/.guru/`:

- `GURU.md` — the base system prompt, appended to the built-in one. Edit it to
  change guru's behaviour everywhere. Auto-created on first run.

**Per-project** — a `.guru/` folder in the current directory, so project
state travels with the project (created lazily on first write):

- `.guru/GURU.md` — project-specific instructions, appended after the global
  `GURU.md`.
- `.guru/domains_allow.txt` — this project's network allow-list (see below).
- `.guru/memory/*.memory` — saved conversations, one JSON file per `/save`.

## Network safeguard

All outbound network access is blocked by default, **per project**. The first
time `web_search` or `web_fetch` needs a domain, guru asks for approval on the
command line:

```
[ACCESS] Request to access example.com.
Allow access to 'example.com'? [y/N]
```

- **Yes** → the domain is added to `.guru/domains_allow.txt` and never asked
  again in this project.
- **No** (or Ctrl+C) → the model is told the request was denied.

Matching is on the hostname only (port ignored). `web_search` gates on the
search-engine backend (`duckduckgo.com`), so you approve internet access once
per project.

## Tool directory

See [`docs/ollama-wrapper.md`](docs/ollama-wrapper.md) for full details on how
the tool directory works and how to add new tools.

Current tools: `web_search`, `web_fetch`, `fetch_github_releases`.

## Architecture

`guru` is a Python package with a domain layer and pluggable provider adapters.
Run it with `./start.sh` or `python -m guru`.

| Path | Purpose |
|------|---------|
| `guru/cli.py` | Prompt loop, slash commands, cross-adapter `/models` |
| `guru/session.py` | Shared runtime state (active adapter, model, context, conversation) |
| `guru/config.py` | Paths, `~/.guru/adapters.toml`, GURU.md assembly, allow-list |
| `guru/ui.py` | Console, status bar, model picker, key bindings, terminal modes |
| `guru/domain/tools.py` | Tool directory, discovery, gating, execution |
| `guru/domain/conversation.py` | Save/resume and compaction (provider-neutral) |
| `guru/adapters/base.py` | `Adapter` interface + `ModelInfo` |
| `guru/adapters/ollama.py` | Ollama provider (daemon check + on-demand pull) |
| `start.sh` | Thin launcher → `python -m guru` |
| `docs/plans/` | Design docs |

Provider adapters are configured in `~/.guru/adapters.toml`. The Ollama adapter
is built in; Anthropic (API-key and OAuth) adapters are planned.
