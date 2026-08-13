# Ollama Wrapper — local chat agent with on-demand tools

A custom chat CLI that runs against any local Ollama model and gives it
access to a tool directory. The model discovers tools on demand rather than
having every schema loaded upfront — designed to scale to 200+ tools without
bloating the context window on every request.

```bash
./start-local-cli-abliterated-wrapper.sh
```

---

## Input controls

| Key | Action |
|-----|--------|
| `Enter` | Submit the message |
| `Shift+Enter` | Insert a newline (multi-line input) |
| `Escape` then `Enter` | Insert a newline (fallback, works in any terminal) |
| `Ctrl+C` | Cancel the current input |
| `Ctrl+D` | Exit |
| Up / Down arrows | Navigate input history (persisted to `~/.ollama_wrapper_history`) |

### Shift+Enter terminal requirement

Terminals send the same byte (`\r`) for both Enter and Shift+Enter by default.
The wrapper enables `modifyOtherKeys` mode (via `\x1b[>4;2m`) each time the
prompt opens. In this mode iTerm2, Kitty, WezTerm, and other modern terminals
send `\x1b[13;2u` for Shift+Enter — a sequence that the wrapper maps to a
newline action.

**Supported out of the box:** iTerm2, Kitty, WezTerm, Ghostty, Windows Terminal.

**Not supported:** VS Code integrated terminal, macOS Terminal.app. Use
`Escape` + `Enter` instead.

---

## Slash commands

| Command | Action |
|---------|--------|
| `/search <query>` | Bypass the model and call `web_search` directly; prompts to `web_fetch` one of the results |
| `exit` / `quit` | Exit |

---

## Tool directory

The model starts each conversation with one meta-tool: `search_tools`. When
it decides it needs a capability it calls `search_tools` with a short
action-oriented phrase — not the user's question.

```
search_tools("search the web")
search_tools("fetch webpage url")
search_tools("get latest github release version")
```

`search_tools` returns matching tool names, descriptions, and parameter
signatures. The matched tools are added to the model's active set and can be
called directly on the next turn. Tools remain active for the rest of the
conversation — no re-discovery needed.

### How matching works

Each tool entry in `TOOL_REGISTRY` has a `description`, `tags` list, and
`parameters` dict. `search_tools` scores every tool across all fields:

| Match location | Score per keyword hit |
|---|---|
| Tool name | +5 |
| Tag | +3 |
| Description | +2 |
| Parameter description | +1 |

Common stop words (`for`, `the`, `this`, …) are filtered before matching so
they don't cause false positives. If no tool scores above zero, all tools are
returned (the fallback).

### Adding a tool

1. Write a function with a clear one-line docstring.
2. Add an entry to `TOOL_REGISTRY` in `ollama_wrapper.py`:

```python
TOOL_REGISTRY["my_tool"] = {
    "fn": my_tool_function,
    "description": "One sentence: what it does and when to use it.",
    "tags": ["keyword1", "keyword2", "action phrase", ...],
    "parameters": {
        "param_name": "Human-readable description of the parameter",
    },
}
```

Tags should cover the natural-language actions a model might use when
searching for this capability. Err on the side of more tags rather than fewer.

---

## Response footer

After each answer the wrapper prints a stats line:

```
── qwen3-abliterated-32k:latest  ·  ctx 2,341 / 32,768  ·  in 2,341  ·  out 312 ──
```

- **ctx** — tokens used in the context window (in / window size)
- **in** — total prompt tokens across all tool-call rounds in this turn
- **out** — total generated tokens across all tool-call rounds in this turn

---

## Technical notes

### Why `Keys.F13` for Shift+Enter

`prompt_toolkit` validates key names against a fixed `Keys` enum.
`'shift-enter'` and `'s-enter'` are not in that enum, so binding to them
raises `ValueError`. `Keys.F13` (`'f13'`) is a valid enum member that is never
sent by normal terminal use, making it a safe internal alias for Shift+Enter.

`ANSI_SEQUENCES` (prompt_toolkit's escape-sequence lookup table) is a plain
dict read at runtime on every keypress. Adding entries before the first
`session.prompt()` call takes effect immediately — no trie rebuild required.

### Why `modifyOtherKeys` must be enabled

Without it, pressing Shift+Enter sends the same byte as Enter (`\r`). The
`pre_run` hook fires after prompt_toolkit has set up raw mode but before the
user types, making it the correct place to write `\x1b[>4;2m` to stdout. The
`termios`-based save/restore that prompt_toolkit performs does not cover
escape-sequence-controlled modes, so the setting persists through each prompt
session. On exit, `\x1b[>4;0m` resets it via `atexit`.

### Why `search_tools` queries should be action phrases

The model reasons about what it wants to DO, not about what the user said.
Querying with the user's raw words (e.g. `search_tools("kubernetes version")`)
produces unreliable matches — stop words cause false positives, and relevant
tools may score zero if they don't mention the subject domain. Action phrases
(`"get latest github release version"`) map directly to what tools do, giving
the scoring a fighting chance.
