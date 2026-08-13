# Session continuity — what was built and what comes next

## What this project is

`guru` is a local Ollama chat agent with a tool directory. It was extracted
from `local-ai` (parent project at `/Users/rdoorn/projects/local-ai`) where it
was developed iteratively as `ollama_wrapper.py`. The rename to `guru.py`
happened at project extraction.

## Architecture

The agent runs a single-file loop (`guru.py`):

1. **Input** — `prompt_toolkit` PromptSession with custom key bindings.
   Shift+Enter inserts a newline; Enter submits. `modifyOtherKeys` mode
   (`\x1b[>4;2m`) is enabled via `pre_run` hook so the terminal sends
   distinguishable escape sequences for Shift+Enter.

2. **Tool directory** — `TOOL_REGISTRY` dict. Each entry has `fn`, `description`,
   `tags`, and `parameters`. The model starts each turn with only `search_tools`
   (the meta-tool). It calls `search_tools("action phrase")` to discover tools,
   which are then added to `active_tools` for the rest of the conversation.
   Matching is weighted: name (+5), tags (+3), description (+2), parameters (+1).
   Stop words are filtered before matching.

3. **Agent loop** — calls `ollama.chat()` in a loop. Tool results are appended
   as messages. If the model returns empty content and no tool calls (happens
   when qwen3 reasons through an answer in `<think>` but fails to emit the call),
   one nudge is sent before giving up.

4. **Stats footer** — after each response, prints model name, context window
   usage, and cumulative in/out token counts for the turn.

## Key implementation details

### Shift+Enter
- `ANSI_SEQUENCES['\x1b[13;2u'] = Keys.F13` — maps the CSI u escape sequence
  to a valid prompt_toolkit Keys enum member (F13 is unused in normal terminals)
- `@_kb.add('f13')` inserts a newline
- `pre_run=_enable_modify_other_keys` sends `\x1b[>4;2m` before each prompt,
  putting the terminal into modifyOtherKeys mode 2 so Shift+Enter sends the
  CSI u sequence instead of plain `\r`
- `atexit` sends `\x1b[>4;0m` to reset on exit

### Tool discovery (why action phrases work)
The model generates the `search_tools` query based on what it wants to DO, not
what the user said. Example: user asks "what's the weather?" → model calls
`search_tools("search the web current data")`. This maps reliably to tags and
descriptions. Querying with the user's raw words produces poor matches.

### active_tools persistence
`active_tools` is defined outside the per-turn loop — tools discovered in turn 1
remain available in turn 2. Without this, the model "remembers" tools from
conversation history and tries to call them, but they aren't in the tool list,
causing a silent empty response from qwen3.

### Empty response nudge
qwen3 sometimes generates a correct reasoning chain in `<think>` but emits no
tool call or content. One nudge ("please continue and complete the task") is
sent before giving up. The nudge message uses `search_tools` as a hint in case
the tool wasn't discovered yet.

## What to work on next

Rough priority order as of session end:

1. **GitHub repo** — project was just created locally; needs a remote.
   `gh repo create rdoorn/guru --public --source=. --remote=origin --push`

2. **More tools** — the registry currently has 3 tools. Candidates:
   - `run_python(code)` — execute Python snippets locally (sandboxed)
   - `read_file(path)` — read a local file and return contents
   - `list_files(directory)` — list files in a directory
   - `ollama_list()` — list locally available Ollama models
   - `ollama_ps()` — show currently loaded models and memory usage

3. **Streaming output** — currently uses non-streaming `ollama.chat()` for all
   rounds including the final answer. The final text response should stream so
   the user sees tokens as they arrive instead of waiting for the full response.
   Tool-call rounds should remain non-streaming (more reliable tool detection).

4. **Conversation commands** — `/clear` to reset the message history without
   restarting, `/history` to show prior turns.

5. **Model config file** — instead of hardcoding the default model in the
   argparse default, read it from a `~/.guru/config.toml` or `.guru.toml`
   in the project root.

6. **Tests** — no tests currently. At minimum: `_match_tools` scoring logic,
   stop word filtering, and the ANSI sequence registration.
