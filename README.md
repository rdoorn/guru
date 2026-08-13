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
| `exit` / `quit` | Exit |

## Tool directory

See [`docs/ollama-wrapper.md`](docs/ollama-wrapper.md) for full details on how
the tool directory works and how to add new tools.

Current tools: `web_search`, `web_fetch`, `fetch_github_releases`.

## Files

| File | Purpose |
|------|---------|
| `guru.py` | Main agent — tool registry, input loop, key bindings |
| `start.sh` | Launcher — checks Ollama, pulls model if needed, runs guru.py |
| `pyproject.toml` | Dependencies (managed by uv) |
| `docs/ollama-wrapper.md` | Full technical documentation |
| `docs/next-session.md` | Context for continuing development |
