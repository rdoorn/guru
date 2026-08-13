# Design: config, memory, and domain safeguards

Date: 2026-08-13

## Goal

Add three capabilities to `guru`:

1. A configurable system prompt via `GURU.md` files.
2. Conversation persistence (`/save`, `/resume`).
3. A domain allow-list that gates all network access.

## 1. Config and setup (`~/.guru/`)

On startup guru ensures `~/.guru/` exists, writes a default `GURU.md` if
absent, and touches `domains_allow.txt`.

The system prompt is assembled as:

```
built-in SYSTEM_PROMPT
+ "\n\n" + ~/.guru/GURU.md            (if non-empty)
+ "\n\n" + ./.GURU.md                 (if present and non-empty)
```

The built-in prompt is always kept first because it teaches the
`search_tools` meta-tool mechanism the whole agent depends on. The local
`.GURU.md` extends (appends to) the global one.

## 2. Domain safeguard

- `~/.guru/domains_allow.txt` — one domain per line, loaded into a set at
  startup. Comparison is on hostname only, lowercased, port ignored.
- `_ensure_domain_allowed(domain)` returns True if already allowed;
  otherwise prompts `Allow access to '<domain>'? [y/N]`. On yes it adds the
  domain to the in-memory set and appends it to the file. On no, empty, or
  Ctrl+C it denies.
- `web_fetch` gates on the target URL's hostname. Denial returns
  `"Access to domain '<x>' was denied by the user."` to the model.
- `web_search` gates on a per-engine backend domain constant
  (`duckduckgo.com` for now; structured so future engines declare their own).
  This is the "allow internet access at least once" gate. Denial returns a
  denial string to the model.
- The `/search` slash command goes through the same gate.

## 3. Memory: `/save` and `/resume`

- Project memory directory: `~/.guru/<encoded-cwd>/`, where `<encoded-cwd>`
  is the absolute working directory with path separators replaced by `-`
  (e.g. `-Users-rdoorn-projects-guru`).
- `/save` writes `messages[1:]` (skipping the system prompt) to
  `<dir>/<uuid4>.memory` as JSON. Each message is normalised to a clean dict
  (`role`, `content`, plus `tool_name` / `tool_calls` when present).
- `/resume` shows an arrow-key selector (shared with `/models`) listing this
  project's `*.memory` files by modified-time and first-user-message title.
  On selection the conversation is loaded as
  `messages = [system] + loaded`, and active tools are re-derived by scanning
  the restored messages for known tool names (prevents the documented
  empty-response bug). The model is not restored; the current model continues.

## Decisions

- Ctrl+C at the domain prompt counts as deny.
- `/save` skips the system prompt so a resumed conversation always uses the
  current `GURU.md`.
- Resume keeps the currently-active model rather than forcing a default.

## Tests

Unit tests for the pure functions: `_domain_of`, cwd encoding, and system
prompt assembly. `flake8` and `pytest` must pass before committing.
