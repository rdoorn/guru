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
| `/good [note]`, `/bad [note]` | Label the last completed turn (and the sub-agent tasks it spawned) in the ledger's `labels` stream |
| `/ledger` | Print this run's spend: calls, tokens and cost per model, tasks per model, the three most expensive tasks |
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

## Ledger and decisions (shadow and active modes)

guru keeps an append-only ledger under `~/.guru/ledger/`, one JSONL file per
stream per UTC day (`calls-2026-09-23.jsonl`, `turns-…`, `tasks-…`,
`decisions-…`, `labels-…`). It records one row per model call (adapter, model, tokens
including cache reads and writes, seconds, cost), per user turn (request,
model, tools used, tasks spawned, time, tokens, cost) and per spawned
sub-agent task (task text, role/skill, status, time, tokens; a task row is
written at spawn and again when it finishes, with its call count, tokens and
cost filled in from the sub-agent's session accumulators — cost is null when
any of its calls hit a model whose price is unknown). Rows are written on
a background worker and never block a turn; a failed write is logged and the
ledger switches itself off for the rest of the run.

Cost comes from a bundled Anthropic price table (USD per million tokens),
overridable per field under `[pricing."<model>"]` in `~/.guru/settings.toml`
(`input_per_m`, `output_per_m`, `cache_read_per_m`, `cache_write_5m_per_m`,
`cache_write_1h_per_m`; a model not in the table needs at least `input_per_m`
and `output_per_m`). Local Ollama models cost zero; when a LiteLLM proxy
reports a per-response cost, that figure wins over the table; a model the
table does not know gets a null cost. Disable the whole ledger with
`[ledger] enabled = false`.

Reading it back: the status bar shows the run's spend after the token
counters (`$0.0123`, or `$?` once any call could not be priced); `/ledger`
prints the current run (calls, tokens and cost per model, tasks per model,
the three most expensive tasks); `/good [note]` and `/bad [note]` label the
last completed turn and its tasks in the `labels` stream (the ledger itself
is never edited). `.venv/bin/python bench/ledger_report.py [--dir DIR]`
aggregates across days as Markdown: calls per model, task latency p50/p95
per kind and complexity, fallback/retry rates, and judge-vs-heuristic and
judge-vs-label agreement per decision point.

The review loop has its own CLI, `python -m guru.ledger_cli`
(`docs/review-loop.md`): `review --point stall [--n 50]` shows the newest
unlabelled decision rows for a point and takes `y`/`n`/`s`/`q` (the correct
answer to the judge's question; each answer is a `labels` row keyed
`<point>:<question>:<input_sha>`), `report [--point stall]` (also
`make ledger-report`) scores every judge against the heuristic and the
labels (precision, recall, F1, false-positive rate) and suggests the
`P(yes)` threshold that maximises F1, and `tasks --unlabelled [--n 20]`
lists recent unlabelled tasks with route, status, cost and transcript path
for triage.

The same ledger holds a `decisions` stream: guru's small closed-form
decisions — is this reply a stall, does this task need a security /
architecture / reliability reviewer, is a fetched page a prompt-injection
attempt — can be handed to a fast local judge alongside the built-in
heuristic. In `shadow` mode judges only *observe*: the heuristic still
decides, and both answers are logged so the first few hundred can be
reviewed before any judge is trusted. In `active` mode the points listed
under `[decisions.active]` take the judge's answer: the judge runs
synchronously (on its own worker, apart from the shadow batches) with a
`timeout_ms` budget, a yes/no question is decided by
`P(yes) >= [decisions.thresholds].<point>` (default 0.5; shadow rows apply
the same threshold so they preview it), and on timeout, error or a missing
judge the heuristic decides and the row records why (`used`,
`fallback_reason`, `queued_ms`). After `breaker_timeouts` consecutive
timeouts (default 5) a point's judge is skipped for `breaker_cooldown_s`
(default 60). Points not listed stay in shadow. Off by default; enable in
`~/.guru/settings.toml`:

```toml
[decisions]
mode = "shadow"                 # off | shadow | active
sidecar_model = "qwen3:4b"      # Ollama model for the "ollama" judge (~3.5 GB resident)
sidecar_url = "http://localhost:11434"
timeout_ms = 1500               # active: max wait for a judge per decision
[decisions.points]              # decision point -> judge spec
stall = "ollama"                # small decoder, JSON-constrained answer
panel = "encoder"               # zero-shot NLI encoder (needs the extra)
injection = "injection"         # prompt-injection classifier (needs the extra)
[decisions.active]              # active mode only: which points the judge decides
stall = true
[decisions.thresholds]          # P(yes) at or above which a judge says yes
stall = 0.6
```

Today only the `stall` point can act on a judge's verdict (whether a turn is
nudged); `panel` and `injection` are shadow-only until the review loop
promotes them. The promotion rule (100+ labelled rows, judge beats the
heuristic, acceptable false-positive rate) and the labelling procedure are in
`docs/review-loop.md`.

Judge specs are `ollama` or `ollama:<model>`, `encoder` or
`encoder:<hf-model>` (default `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`)
and `injection` or `injection:<hf-model>` (default
`protectai/deberta-v3-base-prompt-injection-v2`). The encoder and injection
judges need `uv sync --extra judge` (torch + transformers, about 1 GB);
without it they are skipped with a log line and the heuristic runs alone.
Design and measurements: `docs/plans/2026-09-23-routing-framework-design.md`,
`bench/primitives/README.md`.

## Multi-agent

guru runs a hybrid multi-agent UI: the main agent lives in the normal terminal
buffer; sub-agents live in a full-screen viewer. The main agent can delegate a
self-contained task to a sub-agent that runs in parallel in its own context
with the `spawn` tool, poll it with `check`, and be resumed once a group of
sub-agents finishes with `join`. Sub-agents read bulk tool output in their own
context and return only their conclusion, keeping the main context small.
`Ctrl+N` spawns and views a new agent; `Shift+Right`/`Shift+Left` move between
viewers.

## Routing (controller and ladder)

Sub-agent tasks can be *routed*: the model that runs a spawned task is picked
from a ladder of rungs (cheapest first) by the task's labels rather than
inherited from the parent. The `spawn` tool carries two labels the model
fills in — `kind` (`debug`, `build`, `refactor`, `review`, `explain`, `docs`,
`ops`, `other`) and `complexity` (`trivial`, `standard`, `hard`) — and
`[routing]` in `~/.guru/settings.toml` says what to do with them:

```toml
[routing]
mode = "local-and-remote"   # local-only | local-and-remote | remote-only
controller = false          # main agent only coordinates (see below)
complexity_router = true    # pick the lowest rung that covers the complexity
type_router = false         # use the per-kind ladders below (default: no)
spend_confirm = "ask"       # ask | auto | never
secret_scan = true          # findings force local + redact remote tool output

[[routing.ladder]]          # the default ladder, lowest rung first
adapter = "Ollama"          # an adapter name from adapters.toml
model = "qwen3:14b"
max_complexity = "standard" # the hardest task this rung should take
default = true              # used when complexity_router = false

[[routing.ladder]]
adapter = "Anthropic"
model = "claude-sonnet-5"
max_complexity = "hard"

[[routing.ladders.review]]  # optional per-kind ladder (only with type_router)
adapter = "Anthropic"
model = "claude-opus-5"
max_complexity = "hard"
```

Without a `[routing]` table guru behaves exactly as before: children run on
the parent's adapter and model, nothing is scanned or redacted, and no spend
question is asked. With one, secret scan and redaction default on. A rung
naming an adapter that is not configured is dropped with a warning at
startup; an invalid table logs a warning and the defaults apply. The
parent's own adapter/model is always *pre-approved*: it is the "no change"
fallback, it never needs a spend confirmation and neither a scan finding nor
a declined confirmation takes it away (the parent already runs that model
and already saw the task text); only `mode = "local-only"` refuses to fall
back to a remote parent model.

**Working modes.** `local-only` never runs a task on an adapter that sends
content off-machine (Ollama is local; Anthropic and LiteLLM are remote);
`remote-only` never runs one locally; `local-and-remote` uses the whole
ladder. With `complexity_router` on, a task takes the lowest surviving rung
whose `max_complexity` is at least its complexity; off, the ladder's
`default` rung. When the chosen ladder is emptied by the filters the task
falls back to the default ladder, then to the parent's own model
(pre-approved, see above), then to the first surviving rung of any ladder.
A spawn is *refused* only when no permitted rung is left after those steps:
in `remote-only` mode (where the parent model is never a fallback) when the
filters strip every remote rung — a secret-scan finding, or a declined spend
confirmation — or in `local-only` mode when the parent itself runs remotely
and no ladder has a local rung. The spawn tool reports why and a `refused`
task row is written. Every filter that changed the outcome is listed
verbatim in the task row's `reason` (and its `route`).

**Controller mode.** `controller = true` turns the main agent into a
coordinator: it converses, clarifies, decomposes with
`spawn(task, kind, complexity, role, skill)`, polls with `check`, waits with
`join` and synthesises — and never executes a task itself. Its tool set is
exactly `spawn`, `check`, `join`, `use_skill` (no file or web tools). A
controller that does the work anyway is measured, not punished: the turn row
carries `controller_executed = true` when it attempted any other tool or
answered with more than 600 characters without spawning.

**Spend confirmation.** In `ask` mode the first task that would run on a
remote (paid) *ladder rung* asks once per run — "Allow remote model spend for this
run?" — and the answer is remembered; a decline strips remote rungs for the
rest of the run and the task falls back to the best local rung. `auto`
never asks; `never` records that no confirmation applies. The headless
benchmark and the eval runner always decline.

**Secret scan and redaction.** With `secret_scan = true` a regex scanner
(AWS/GitHub/Slack/Google keys, private-key blocks, JWTs, generic
`password = …` assignments, plus your own markers from
`.guru/sensitive_markers.txt`; false positives are silenced with regexes in
`.guru/scan_allow.txt`) runs over every task text: any finding forces a
local rung (`findings` on the task row). While a sub-agent runs on a remote
adapter, every tool result passes through the same scanner and findings are
replaced with `[REDACTED:<kind>]` before the text leaves the machine; the
count lands in the task's `struggle.redactions`.

**Retry rule.** A task that ran on a remote rung and ended without an answer
after a provider error is respawned once on the best local rung (mode forced
to `local-only` for that pick); the original row closes as `fell_back` and
the retry carries `retry_of = <original task id>`. A retry that fails too
is delivered to the parent as an error; there is never a second retry, and
a cancelled task is not retried.

**GPU fit.** Whenever an `ollama` judge is configured under `[decisions]`
(and decisions are not `off`), the sidecar model's size x 1.2 is reserved
out of the measured GPU budget before the main model's context is fitted,
so the two do not spill each other.

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
- `[routing]`, `[[routing.ladder]]`, `[[routing.ladders.<kind>]]` — see
  **Routing (controller and ladder)** above.

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
