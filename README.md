# guru

Local LLM chat agent with an on-demand tool directory and pluggable provider
adapters. The model starts each conversation with one meta-tool
(`search_tools`) and discovers further tools as it needs them — designed to
scale to 200+ tools without loading every schema into context on every request.

Providers: **Ollama** (local), **Anthropic** (API key or enterprise OAuth), and
**LiteLLM** (any OpenAI-compatible proxy, e.g. an AWS/enterprise gateway) — all
selectable from `/models`.

## Quick start

```bash
uv sync          # one-time: creates .venv and installs deps
./start.sh       # launch against the default model
./start.sh --model qwen3:8b   # override the model
```

Requires the Ollama app running in the menu bar (for local models).

## Input

| Key | Action |
|-----|--------|
| `Enter` | Submit |
| `Shift+Enter` | New line (requires iTerm2, Kitty, WezTerm, or similar) |
| `Escape` → `Enter` | New line (works in any terminal) |
| `Ctrl+C` | Cancel current input (double-press to exit) |
| `Ctrl+D` | Exit |
| `Ctrl+N` | Spawn a new agent and view it |
| `Shift+Right` | Enter the sub-agent viewer |
| `Shift+Left` | Cycle sub-agents / return to `[main]` |
| `Shift+Tab` | Cycle the access mode (read-only / ask / auto) |
| `↑` / `↓` | History (persisted to `~/.guru/history`) |

## Slash commands

| Command | Action |
|---------|--------|
| `/mode [name]` | Set the access mode (read-only / ask / auto), or cycle with no arg |
| `/role [name]` | Set the active persona; `off`/`none` clears it |
| `/skill [name]` | Set the active method; `off`/`none` clears it |
| `/models` (or `/model`) | Interactive model selector (↑/↓, Enter, Esc) |
| `/context` | Pick the context window (halves of the model's max, down to 4k) |
| `/adapters` | Enable/disable providers (Space toggles, Enter verifies + saves) |
| `/save` | Save the current conversation to disk |
| `/resume` | Restore a previously saved conversation (interactive selector) |
| `/compact` | Shrink the conversation to free up context now |
| `/search <query>` | Call `web_search` directly and optionally `web_fetch` a result |
| `exit` / `quit` | Exit |

## Roles & skills

A **role** is a persona (WHO the agent is — `developer`, `architect`,
`security-engineer`, `SRE`); a **skill** is a method (HOW it works —
`code-review`, `brainstorming`, `systematic-debugging`,
`test-driven-development`). Both are plain Markdown files with a small
`key: value` frontmatter block, stored under `~/.guru/skills/*.md` and seeded
with baked-in defaults on first run (`--reset-skills` overwrites the defaults;
your own extra files are never touched). One role and one skill are active at a
time and are rendered onto the system prompt on demand — they are prompt
overlays only and do not change tool access. Switch them with `/role <name>`
and `/skill <name>`; the model can adopt a skill itself with the `use_skill`
tool, and a spawned sub-agent can be given both via `spawn(role=…, skill=…)`.

## Context management

A status bar pinned to the bottom of the screen shows session state:

```
🤖 qwen3-abliterated-32k | 💪 8.2B | 🔐 ask | 🎭 developer/code-review | 🧠 28% ███░░░░░░░ · 📊 sys:1.2k tl:0.8k in:2.1k out:0.4k | ↓ 289898 | ↑ 784 | 📁 guru | 🌿 main
```

model · parameter size · access mode · role/skill · context fullness (coloured
green/yellow/red) · context breakdown (system / tools / input / output tokens) ·
session input tokens · output tokens · current directory · git branch. During
model generation it stays fixed; at the input prompt the same info is shown in
the prompt toolbar.

- **Window size** — guru resolves the effective `num_ctx` from the model's
  modelfile (falling back to `4096`), capped at the model's architecture
  ceiling. Override with `--num-ctx N` or the `/context` command.
- **GPU auto-fit (Ollama)** — the first time a model is selected with no
  explicit `--num-ctx`/`/context` override, guru measures the largest context
  that stays entirely on the GPU. It probes two loads and reads Ollama's own
  memory report (`ollama.ps`) to recover the real weights, the real per-token
  KV cost (correct for f16 or q8_0), and — on a spill — the true GPU budget,
  then persists that per-model choice to `~/.guru/model_ctx.json` and reuses it
  on the next launch. A manual `/context` or `--num-ctx` always wins; the
  reported architecture max is never changed.
- **Auto-compaction** — when occupancy crosses 85%, guru compacts between
  turns: it drops old reasoning traces, evicts stale tool outputs, and, if
  still too large, folds the oldest turns into a summary. Recent turns and the
  system prompt are always kept. Trigger it manually with `/compact`.

## Multi-agent

guru runs a hybrid multi-agent UI: the main agent lives in the normal terminal
buffer; sub-agents live in a full-screen viewer. The main agent can delegate a
self-contained task to a sub-agent that runs in parallel in its own context
with the `spawn` tool, poll it with `check`, and be resumed once a group of
sub-agents finishes with `join`. Sub-agents read bulk tool output in their own
context and return only their conclusion, keeping the main context small.
`Ctrl+N` spawns and views a new agent; `Shift+Right`/`Shift+Left` move between
viewers.

## Providers

Adapters are configured in `~/.guru/adapters.toml` (auto-created with an Ollama
entry plus commented Anthropic and LiteLLM templates). Each `[[adapter]]`
becomes a group in `/models`; selecting a model switches the active provider and
model. Full tool parity — the tool directory works on every provider.

```toml
[[adapter]]
name = "Ollama"
type = "ollama"
url  = "http://localhost:11434"

[[adapter]]
name = "Anthropic (local)"
type = "anthropic"
auth = "api_key"
base_url = "http://localhost:8080"      # local endpoint speaking the Messages API
api_key_env = "GURU_ANTHROPIC_API_KEY"  # key read from this env var
# models = ["my-local-model"]           # optional; else queried from the endpoint
# thinking = false                       # disable adaptive thinking for endpoints that lack it

[[adapter]]
name = "Anthropic Enterprise"
type = "anthropic"
auth = "oauth"        # no API key; token from `ant auth login` (ant CLI required)
profile = "default"   # optional ant profile name

[[adapter]]
name = "LiteLLM"
type = "litellm"                       # any OpenAI-compatible proxy
base_url = "https://proxy.example/v1"  # include /v1
api_key_env = "LITELLM_KEY"            # env var holding the virtual key
# api_key = "sk-..."                   # or inline (used if the env var is unset)
# models = ["azure/gpt-4.1"]           # optional allowlist; else queried from /v1/models
```

Each adapter has an `enable` flag (missing = enabled). Manage them with the
**`/adapters`** command: Space toggles, Enter saves the flags back to
`adapters.toml` and **verifies** each enabled adapter — a connectivity check
for Ollama / API-key / LiteLLM providers, and for an enterprise OAuth provider
the one-time browser login (`ant auth login --profile <profile>`) if it hasn't
been done yet. After the first login, the SDK refreshes and re-stores the token
automatically; you only re-login when the refresh token hard-expires.

Opening `/models` **logs in / verifies every enabled adapter** first, so the
list is complete and usable. guru remembers the last adapter + model you used
per project in `.guru/settings.json` and restores + re-authenticates it on the
next startup.

Secrets are never stored in the file — API keys come from the environment (or
an inline `api_key` for LiteLLM) and the OAuth profile is managed by the `ant`
CLI.

> The enterprise OAuth login needs the `ant` CLI:
> `brew install anthropics/tap/ant`, then run `/adapters` and enable the
> enterprise provider to trigger the login.

In `/models`, Ollama models also show their estimated memory footprint,
coloured **red** when it exceeds 80% of system memory (won't fit comfortably).
Remote models are queried for their context window; no memory is shown.

## Configuration

**Global** — `~/.guru/`:

- `GURU.md` — the base system prompt, appended to the built-in one. Edit it to
  change guru's behaviour everywhere. Auto-created on first run.
- `adapters.toml` — provider configuration (see **Providers** above).
- `skills/*.md` — role and skill overlays (see **Roles & skills** above).
- `model_ctx.json` — per-model chosen context sizes (written by the GPU
  auto-fit and by `/context`).
- `settings.toml` — optional global user settings (see below). Not created
  automatically; add it yourself to override defaults.

`~/.guru/settings.toml` sections:

- `[context]` — tool-output retention thresholds (chars): a result below the
  threshold is kept verbatim, above it web results are query-summarized and
  large file reads are outlined.
  - `web_summarize_over_chars` (default `6000`)
  - `outline_file_over_chars` (default `8000`)
- `[tools]`
  - `preactivate = [...]` — core tools pre-activated on every agent so weaker
    models can call them directly without the `search_tools` hop (default
    `["list_dir", "list_tree", "read_file", "search_code"]`).
  - `flat = true` — pre-activate the ENTIRE registry on every agent, so a
    capable, large-context model gets the whole toolset up front (costs more
    prompt tokens; off by default).
- `[sampling]` — sampling overrides applied on top of a model's own modelfile
  defaults. Scalar keys here are global (all models); a `[sampling."<model>"]`
  sub-table holds per-model overrides (per-model wins). Empty by default.
- `[bench]`
  - `model_timeout` — per-model wall-clock ceiling (seconds) for the headless
    benchmark; a model that stalls past it is cancelled and recorded as a
    timeout (default `600`; `0` disables the guard).

**Per-project** — a `.guru/` folder in the current directory, so project
state travels with the project (created lazily on first write):

- `.guru/GURU.md` — project-specific instructions, appended after the global
  `GURU.md`.
- `.guru/settings.json` — the last-used adapter + model for this project.
- `.guru/domains_allow.txt` — this project's network allow-list (see below).
- `.guru/read_dirs_allow.txt` / `.guru/write_dirs_allow.txt` — approved
  file-read / file-write directories.
- `.guru/memory/*.memory` — saved conversations, one JSON file per `/save`.

## Access modes & safeguards

An access mode governs whether tool-driven changes prompt, auto-approve, or are
refused (cycle it with `Shift+Tab` or `/mode`):

- **read-only** — refuses file writes.
- **ask-for-changes** (default) — prompts once per not-yet-allowed
  domain/directory.
- **auto** — approves silently, filling the allow-lists.

All outbound network access is blocked by default, **per project**. The first
time `web_search` or `web_fetch` needs a domain, guru asks for approval:

```
Allow web access to 'example.com'? [Y/n]
```

Approving adds the domain to `.guru/domains_allow.txt` and never asks again in
this project. Matching is on the hostname only (port ignored). `web_search`
gates on the search-engine backend (`duckduckgo.com`), so you approve internet
access once per project. File reads and writes are gated the same way, against
separate per-directory allow-lists.

## Tool directory

See [`docs/tools.md`](docs/tools.md) for full details on how the tool directory
works and how to add new tools. The model discovers tools at runtime by calling
`search_tools` with a phrase describing the action it wants; matched tools
become active and can then be called directly.

Registry tools:

- **Filesystem** — `list_dir`, `list_tree`, `read_file`, `search_code`
  (grep), `write_file`, `edit_file`, `delete_file`. All are restricted to
  allowed directories and gated by the access mode.
- **Web** — `web_search`, `web_fetch`, `fetch_github_releases`.

`search_tools`, `use_skill`, and (for delegation-capable agents) `spawn`,
`check`, `join` are always available and not part of the registry.

## Architecture

`guru` is a Python package with a domain layer and pluggable provider adapters.
Run it with `./start.sh` or `python -m guru`.

| Path | Purpose |
|------|---------|
| `guru/cli.py` | Entry point: adapter wiring, model selection, slash-command helpers |
| `guru/tui.py` | Hybrid multi-agent UI (main agent in the normal buffer, sub-agents in a full-screen viewer) |
| `guru/tui_io.py` | TUI output writers and status-bar formatting (split out of `tui.py`) |
| `guru/orchestrator.py` | Shared spawn/check/join mailbox for the TUI and the benchmark |
| `guru/agents.py` | `Agent` / `AgentManager` — viewports and sub-agent spawning |
| `guru/session.py` | Per-context runtime state (adapter, model, context, conversation), routed for parallelism |
| `guru/config.py` | Paths, `adapters.toml`, `settings.toml`, GPU-fit constants, GURU.md assembly, allow-lists |
| `guru/skills.py` | Roles & skills registry (persona / method overlays) |
| `guru/log.py` | Lightweight logging to `~/.guru/guru.log` |
| `guru/bench.py` | Headless coding-model benchmark |
| `guru/ui.py` | Console, status bar, model picker, key bindings, terminal modes |
| `guru/domain/tools.py` | Tool directory, discovery, gating, execution |
| `guru/domain/files.py` | Filesystem tools (list / read / grep / write / edit / delete) |
| `guru/domain/conversation.py` | Save/resume and compaction (provider-neutral) |
| `guru/adapters/base.py` | `Adapter` interface + `ModelInfo` |
| `guru/adapters/turn.py` | Shared provider-agnostic tool-calling turn loop |
| `guru/adapters/ollama.py` | Ollama provider (daemon check, on-demand pull, GPU auto-fit) |
| `guru/adapters/anthropic.py` | Anthropic provider (API-key or OAuth) |
| `guru/adapters/litellm.py` | LiteLLM / OpenAI-compatible provider |
| `start.sh` | Thin launcher → `python -m guru` |
| `docs/plans/` | Design docs |

Provider adapters are configured in `~/.guru/adapters.toml` (see **Providers**).

## Development

Make targets (local; there is no CI):

- `make test` — run the test suite (pytest).
- `make lint` — flake8 over `guru bench tests`.
- `make typecheck` — mypy over `guru`.
- `make bench` — run the headless coding-model benchmark, writing
  `bench/results-<timestamp>.json` plus a companion `transcript-<timestamp>.json`.
- `make bench-plot` — plot the latest results (override with `RESULTS=...`).
