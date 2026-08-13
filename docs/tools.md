# Tool directory — on-demand tools with `search_tools`

guru gives the model a tool directory it discovers on demand rather than
loading every tool schema upfront — designed to scale to 200+ tools without
bloating the context window on every request. The mechanism is
provider-agnostic: it works the same on Ollama and Anthropic models.

## How it works

The model starts each conversation with one meta-tool: `search_tools`. When it
needs a capability it calls `search_tools` with a short action-oriented phrase
— not the user's question.

```
search_tools("search the web")
search_tools("fetch webpage url")
search_tools("get latest github release version")
```

`search_tools` returns matching tool names, descriptions, and parameter
signatures, and adds the matched tools to the model's active set. They stay
active for the rest of the conversation — no re-discovery — and are restored
when a saved conversation is resumed.

Implementation: `guru/domain/tools.py`. Each adapter translates the active tool
set to its provider's native tool schema (`active_specs()` provides the
provider-neutral form), and routes the model's tool calls through the shared
`execute_tool`, which applies the network allow-list gate and runs the tool.

## How matching works

Each entry in `TOOL_REGISTRY` has a `description`, `tags` list, and
`parameters` dict. `search_tools` scores every tool across all fields:

| Match location | Score per keyword hit |
|---|---|
| Tool name | +5 |
| Tag | +3 |
| Description | +2 |
| Parameter description | +1 |

Stop words (`for`, `the`, `this`, …) are filtered before matching. If no tool
scores above zero, all tools are returned (the fallback).

Action phrases work better than the user's raw words: the model reasons about
what it wants to DO, and action phrases map directly to what tools do, giving
the scoring a fighting chance.

## Adding a tool

1. Write a function with a clear one-line docstring in `guru/domain/tools.py`.
2. Add an entry to `TOOL_REGISTRY`:

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

Tags should cover the natural-language actions a model might use when searching
for this capability — err on the side of more tags. Parameters are treated as
required string inputs when translated to a provider's tool schema.

## Network allow-list

Web-facing tools (`web_search`, `web_fetch`) are gated by a per-project
allow-list — the first access to a new domain prompts for approval. See the
README's *Network safeguard* section.

## Input handling notes (`guru/ui.py`)

### Shift+Enter / keyboard Enter vs pasted newlines

Terminals send `\r` for both Enter and pasted newlines. guru enables
`modifyOtherKeys` mode (`\x1b[>4;2m`) each time the prompt opens so modern
terminals (iTerm2, Kitty, WezTerm, Ghostty, Windows Terminal) send distinct
sequences for Shift+Enter and keyboard Enter. Pasted newlines are told apart
from a deliberate Enter by the prompt_toolkit input queue: keys still queued
behind a `\r` mean it is part of a paste (insert), an empty queue means a
deliberate Enter (submit). `Escape`+`Enter` is a universal submit fallback.

### Why `Keys.F13` / `Keys.F14`

`prompt_toolkit` validates key names against a fixed `Keys` enum, and
`'shift-enter'` is not a member. `F13`/`F14` are valid enum members never sent
by normal terminal use, so they are safe internal aliases mapped from the
escape sequences via `ANSI_SEQUENCES`.
