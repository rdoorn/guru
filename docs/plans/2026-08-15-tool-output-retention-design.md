# Tool-output retention — design

Date: 2026-08-15

## Problem

`conversation.prune_tool_exchanges` runs after every turn and drops **all**
tool output. That keeps context lean but forces the model to re-acquire
everything on a follow-up question. Meanwhile `compact_messages` already keeps
the last `KEEP_RECENT_GROUPS` turns and evicts *older* tool output on a size
threshold — but the blanket prune deletes everything first, so the age-based
logic never has anything to work with.

Not all tool output is equal: a `web_search`/`web_fetch` can return a large
page of which ~1% is relevant, while a `read_file` is almost always relevant to
a coding follow-up. Retention should be per-tool.

## Principle

**Size-gated, eager, per-tool retention.** Small results are kept untouched (no
wasted cycles); only large ones are compacted, right after the turn that used
them (the full content was present while answering, so the answer itself loses
nothing). Age-tiering (`compact_messages`) remains the backstop for overflow.

## Configuration

New global settings file `~/.guru/settings.toml` (defaults baked into
`config.py`; the file overrides; loaded once at startup):

```toml
[context]
web_summarize_over_chars = 6000   # web_search/web_fetch above this -> summary
outline_file_over_chars  = 8000   # read_file above this -> code outline
```

- `config.load_settings()` parses it (tomllib), merges over defaults, tolerates
  a missing/invalid file.
- Defaults: `WEB_SUMMARIZE_OVER_CHARS = 6000`, `OUTLINE_FILE_OVER_CHARS = 8000`.

## Per-tool policy

A `retain` field on each `TOOL_REGISTRY` entry (default `keep`):

| Tool | retain |
|---|---|
| `web_search`, `web_fetch` | `summarize` |
| `read_file` | `outline` |
| `search_code`, `list_dir`, `list_tree` | `keep` |
| `write_file`, `edit_file`, `delete_file` | `keep` |
| (unknown / default) | `keep` |

## after_turn (replaces the blanket prune)

`prune_tool_exchanges` is replaced by `apply_retention(messages)`:

1. Drop text-less assistant tool-call steps (unchanged cleanup).
2. For each `tool` message from the just-finished turn, look up its tool's
   `retain` policy (via `tool_name`) and apply:
   - `keep`: leave as-is.
   - `summarize`: if `len(content) > WEB_SUMMARIZE_OVER_CHARS`, replace with a
     **query-focused summary** — the adapter is asked to keep only the parts of
     the content relevant to that turn's user question. Prefixed
     `[<tool> summary · query: <q>]`. Otherwise keep verbatim.
   - `outline`: if `len(content) > OUTLINE_FILE_OVER_CHARS`, replace with a
     **code skeleton** (see below). Prefixed `[read_file outline · <path>]`.
     Otherwise keep verbatim.
3. `after_turn` then runs the existing estimate + `compact_messages`.

"That turn's user question" = the most recent `user` message at or before the
tool message.

### Query-focused summary

New adapter helper (e.g. `summarise_focused(question, content)`), used by the
`summarize` policy. Prompt intent: "Given the question `<q>`, extract only the
parts of the following content relevant to answering it; keep concrete facts,
values, and identifiers; be concise; output only the extract." Reuses the
session model. Falls back to a plain truncate if the call fails.

### Code outline

`_outline_code(read_output)`:
- The stored `read_file` result is line-numbered (`"{n:>6}\t{line}"`) with a
  header line. Reconstruct the source by stripping the `"{n}\t"` prefixes.
- If the header path ends in `.py` and the reconstructed source parses with
  `ast`: emit module docstring (first line), imports, and every top-level and
  class-level `def`/`class` **signature + its one-line docstring**; drop bodies.
- Otherwise (partial read, non-Python, or `SyntaxError`): fall back to a
  regex outline (keep lines matching `^\s*(def |class |async def |@|import |
  from )`), and if that yields little, truncate to the first N lines.
- Always keep the original header so the model still knows the path / sha /
  total line count and can re-read.

## Age-tiering backstop

`compact_messages` is unchanged. Because retention now keeps most tool output,
the existing "keep recent groups, evict older tool output on threshold" logic
finally has material to compact when context actually fills.

## Error handling

- Missing/invalid `settings.toml` -> defaults.
- Summary LLM call fails -> truncate fallback (never lose the turn).
- Outline parse fails -> regex/truncate fallback.
- Unknown `tool_name` -> `keep`.

## Testing

- `load_settings`: missing file -> defaults; override from a temp file; invalid
  file -> defaults.
- Policy dispatch: correct `retain` resolved per tool name; unknown -> keep.
- `summarize`: large web output -> replaced by summary (mock adapter);
  small -> kept verbatim; summary failure -> truncated.
- `outline`: large `.py` read -> signatures + docstrings kept, bodies gone,
  header preserved; small -> kept; non-`.py` large -> truncate fallback.
- `apply_retention` integration: web-large summarized, file-large outlined,
  file-small kept, `search_code` kept, text-less assistant step dropped.

## Out of scope (future)

Narrower code-search / symbol tools (e.g. `outline`/`find_symbol`) so the model
fetches a function instead of a whole file, limiting context at acquisition
time rather than compacting after the fact.
