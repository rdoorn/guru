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
| `/tools` | Print the last turn's tool calls from the audit stream: tool, args head, seconds, bytes shown/produced, denials |
| `/brief`, `/brief refresh`, `/brief slice <text>` | Show the project brief for the checked-out HEAD (built once, stored under `~/.guru/briefs/`), rebuild it, or preview the slice a worker gets for a task (see **Project brief**) |
| `/routing`, `/routing off`, `/routing on` | Show the active routing configuration (mode, ladders, judges) or flip `mode` in `settings.toml` and reload it |
| `/sandbox [status \| provision [--force] \| gate \| deps …]` | Inspect the project's sandbox, build its image, review the pending gate, manage dependency requests (see **Sandbox**) |
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
decisions — does this task need a security / architecture / reliability
reviewer, is a fetched page a prompt-injection attempt — can be handed to a
fast local judge alongside the built-in heuristic (the `stall` point is
retired in favour of the turn loop's stall monitor; its config keys are
accepted and ignored). In `shadow` mode judges only *observe*: the heuristic still
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
labels_margin = 0.15            # active labels: judge's top tier must beat the
                                #   runner-up by this much to override
[decisions.points]              # decision point -> judge spec
stall = "ollama"                # small decoder, JSON-constrained answer
panel = "encoder"               # zero-shot NLI encoder (needs the extra)
injection = "injection"         # prompt-injection classifier (needs the extra)
labels = "decide"               # the lead's kind/complexity labels
[decisions.active]              # active mode only: which points the judge decides
stall = true
labels = true
[decisions.thresholds]          # P(yes) at or above which a judge says yes
stall = 0.6
```

One point acts on a judge's verdict: `labels`, a margin-gated
*tie-breaker* for the complexity label
the lead puts on a spawned task — the judge's tier routes the task
only when it differs from the lead's and beats the runner-up by
`labels_margin` (0.15); the task row's `reason` then says
`labels:judge override standard->hard (0.57 vs 0.33)`, and a judge that
lost on margin leaves a row with `fallback_reason = "margin"` (the kind
label is only observed). `panel` asks the same judge `needs_security` over
a review-kind task spawned without a security worker and, when active,
has the orchestrator add one (`reason` starts with `origin:panel`); the
default routing block keeps it **shadow** (`panel = false`) until
labelled rows say otherwise — in three eval suites it answered no on
every "correctness and security" review request, so its verdicts are
recorded and measured, nothing acts on them. `injection` stays shadow-only
until the review loop promotes it. The promotion rule (100+ labelled
rows, judge beats the heuristic, acceptable false-positive rate) and the
labelling procedure are in `docs/review-loop.md`.

Judge specs are `ollama` or `ollama:<model>`, `encoder` or
`encoder:<hf-model>` (zero-shot NLI, default
`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`), `decide` or
`decide:<hf-model>` (schema classifier for choice/score questions,
default `fastino/GLiNER2.5-Decide`) and `injection` or
`injection:<hf-model>` (default
`protectai/deberta-v3-base-prompt-injection-v2`). `labels` defaults to
`decide`: on 325 real sub-agent tasks it picked the complexity tier right
0.83 of the time against 0.46 for `encoder`, and as the tie-breaker it
lifts routed accuracy from 0.60 to 0.83; `labels = "encoder"` switches
back. A settings file that still carries the untouched
`labels = "encoder"` line guru wrote earlier is moved to `decide` at
startup, with a notice, once the `judge` extra (now including gliner2) is
installed; until then it keeps the working NLI judge. `decide` declines yes/no questions, so `panel` and
`stall` stay on `encoder` / `ollama`. The encoder, decide and injection
judges need `uv sync --extra judge` (torch + transformers<5 + gliner2,
about 1.5 GB); without it they are skipped with a log line and the
heuristic runs alone.
Design and measurements: `docs/plans/2026-09-23-routing-framework-design.md`,
`bench/primitives/README.md`.

Debugging what a remote adapter sends (prompt-cache misses, prefix
stability): `GURU_DUMP_REQUESTS=<dir>` makes the LiteLLM and Anthropic
adapters write every outgoing request's keyword arguments — model,
messages, tools, system, `cache_control` markers — as JSON to
`<dir>/<utc-timestamp>-<adapter>-<n>.json` (`n` counts requests in the
process, so a listing is the request sequence). The files hold the request
body only; the API key is held by the SDK client and never appears in
them — but the body *is* the whole conversation: system prompt, every
file the agent read, every gate diff and tool result. Treat the dump
directory as sensitive and delete it after use. Diff two consecutive
files of the same agent to see what moved in the cached prefix; the
`calls` ledger stream has the matching `cache_read` / `cache_write`
counts.

## Usage dashboard

Every model call guru makes — adapter, model, tokens, cost, project and
the request's topic — goes to a SQLite store at `~/.guru/usage.db` (file
0600), shared by every guru process; sub-agents carry their request's
topic, eval runs are marked `source = eval`. Each request gets a short
topic label from the cheapest routed remote model in the background (the
request text when none can be used). One guru among many serves a
read-only dashboard on `http://127.0.0.1:7340` (`[dashboard] port`): the
first to bind serves, the others show where it is and retry, taking over
when it stops. `/dashboard` prints where it is served; `[ledger] usage_db
= false` or `[dashboard] enabled = false` turn the parts off. Details in
`docs/defaults.md` (Usage store and dashboard).

## Multi-agent

guru runs a hybrid multi-agent UI: the main agent lives in the normal terminal
buffer; sub-agents live in a full-screen viewer. The main agent can delegate a
self-contained task to a sub-agent that runs in parallel in its own context
with the `spawn` tool, poll it with `check`, and be resumed once a group of
sub-agents finishes with `join`. Sub-agents read bulk tool output in their own
context and return only their conclusion, keeping the main context small.
`Ctrl+N` spawns and views a new agent; `Shift+Right`/`Shift+Left` move between
viewers.

## Routing (the lead and the ladder)

Sub-agent tasks are *routed*: the model that runs a spawned task is picked
from a ladder of Claude tiers (cheapest first) by the task's labels rather
than inherited from the parent. The `spawn` tool carries two labels the
lead fills in — `kind` (`debug`, `build`, `refactor`, `review`,
`explain`, `docs`, `ops`, `other`) and `complexity` (`trivial`, `standard`,
`hard`) — and `[routing]` in `~/.guru/settings.toml` says what to do with
them.

**The default configuration.** On startup, when `~/.guru/settings.toml`
has no `[routing]` table and at least one remote adapter (`litellm` or
`anthropic`) is enabled in `adapters.toml`, guru appends the measured
default block (`evals/routing/claude-tiers-judges.toml`, triage notes
`evals/triage/2026-09-24-*`) to the file and prints one line saying so.
The block is written once; an existing `[routing]` table — even an empty
one — is never touched, other tables are left as they are, and the file is
created if missing. For a LiteLLM adapter named `SBP Litellm` it reads:

```toml
[routing]
mode = "local-and-remote"   # local-only | local-and-remote | remote-only | off
complexity_router = true    # lowest rung whose max_complexity covers the task
type_router = true          # review tasks use [[routing.ladders.review]]
spend_confirm = "ask"       # ask (once per run) | auto | never
secret_scan = true          # findings force local; remote tool output redacted

[[routing.ladder]]          # trivial: lookups, one-file summaries
adapter = "SBP Litellm"
model = "aws/claude-4-5-haiku"
max_complexity = "trivial"

[[routing.ladder]]          # standard: a few files, one bug, one edit
adapter = "SBP Litellm"
model = "aws/claude-5-sonnet"
max_complexity = "standard"
default = true

[[routing.ladder]]          # hard: multi-file work, whole-repo reviews
adapter = "SBP Litellm"
model = "aws/claude-5-5-opus"
max_complexity = "hard"

[[routing.ladders.review]]  # review-kind tasks: never Haiku
adapter = "SBP Litellm"
model = "aws/claude-5-sonnet"
max_complexity = "standard"
default = true

[[routing.ladders.review]]
adapter = "SBP Litellm"
model = "aws/claude-5-5-opus"
max_complexity = "hard"

[decisions]
mode = "active"
labels_margin = 0.15
[decisions.points]
labels = "decide"           # complexity tie-breaker; "encoder" = the NLI judge
panel = "encoder"           # needs_security: one extra security reviewer
injection = "injection"     # shadow: fetched pages checked for injection
[decisions.active]
labels = true
panel = false               # shadow until labelled rows say otherwise
```

For an `anthropic` adapter the model ids are the first-party ones
(`claude-haiku-4-5`, `claude-sonnet-5`, `claude-opus-5-5`). The
`[decisions]` part is skipped when the file already has one, and without
the `judge` extra (`uv sync --extra judge`) `labels` is written `false`
with a note: the encoder judge is then only observed (`panel` is shadow
either way).
The routing table does not set the main (lead) model: pick it in
`/models`.

**Inspecting and turning it off.** `/routing` prints the mode and flags,
every ladder's rungs and the judge per decision point (active / shadow,
installed or not). `/routing off` writes `mode = "off"` into the table
(remembering the previous mode in a trailing `# was "…"` comment, touching
no other line) and reloads: routing, secret scan and redaction all behave
as if no `[routing]` table existed. `/routing on` restores the previous
mode. Both take effect in the running process.

**How a task is routed.** With `complexity_router` on, a task takes the
lowest surviving rung whose `max_complexity` is at least its complexity;
off, the ladder's `default` rung. With `type_router` on, a task whose
`kind` has a per-kind ladder (`[[routing.ladders.<kind>]]`) uses it — the
default block gives `review` its own ladder starting at Sonnet, so a
trivial-labelled review never lands on Haiku; every other kind uses the
default ladder. `type_router` needs no setting: it is on
whenever a per-kind ladder is configured and `type_router = false` turns it
off. A rung naming an adapter that is not configured is dropped
with a warning at startup; an invalid table logs a warning and the defaults
apply. The parent's own adapter/model is always *pre-approved*: it is the
"no change" fallback, it never needs a spend confirmation and neither a scan
finding nor a declined confirmation takes it away (the parent already runs
that model and already saw the task text); only `mode = "local-only"`
refuses to fall back to a remote parent model.

**Working modes.** `local-only` never runs a task on an adapter that sends
content off-machine (Ollama is local; Anthropic and LiteLLM are remote);
`remote-only` never runs one locally; `local-and-remote` uses the whole
ladder. When the chosen ladder is emptied by the filters the task falls
back to the default ladder, then to the parent's own model (pre-approved,
see above), then to the first surviving rung of any ladder. A spawn is
*refused* only when no permitted rung is left after those steps: in
`remote-only` mode (where the parent model is never a fallback) when the
filters strip every remote rung — a secret-scan finding, or a declined
spend confirmation — or in `local-only` mode when the parent itself runs
remotely and no ladder has a local rung. The spawn tool reports why and a
`refused` task row is written. Every filter that changed the outcome is
listed verbatim in the task row's `reason` (and its `route`). Ollama stays
the local-only option: point the rungs at an Ollama adapter for a
`local-only` setup, and at the sidecar for the `ollama` judge.

**The lead.** The main agent is the lead (`config.LEAD_HINT`): it keeps
the overview of the request, does small work itself and spawns workers for
parts that can run in parallel, each with a self-contained task (goal,
files and interfaces, what it already knows). Workers report back; the
lead joins them, reviews what they changed, integrates the parts and
verifies the whole (full tests, lint, the running application uses the new
code) before it answers. In a sandbox project each worker's copy diff
comes back with its report; the lead merges it into its own copy with
`apply_work(<worker>)`, tests there and submits the integrated change once
through the gate. (The former controller mode — a tool-less main agent
answering with one forced `plan` call — is retired; a `controller` key in
`[routing]` is ignored.)

**The turn loop.** Nothing is forced: a reply without a tool call ends the
turn — the lead's answer, a worker's report — so a thinking model keeps
its reasoning on every round (forcing `tool_choice` turns Anthropic's
extended thinking off without an error). Thinking is on by default:
`[thinking] lead = "high"`, `worker = "medium"` (`low`/`medium`/`high`/
`off`); LiteLLM sends `reasoning_effort`, Anthropic adaptive thinking.
There is no round cap. The stall monitor counts rounds without progress
(no file changed and no tool result not seen before in the turn; timings
ignored): after 20 the agent gets one warning, after 5 more the turn ends
— a worker's with a handoff (files changed and read, its last text; task
status `stalled`) so the lead decides what next. An answer from an agent
that changed files and did not run the tests (and lint, when enabled)
since is sent back once; the lead's answer to a request it worked on goes
through the answer check (`[decisions] answer_check`).

**Judges on the routing seam.** Two decision points can act on the routing
seam (`[decisions] mode = "active"`, see **Ledger and decisions**); the
default block activates `labels` and keeps `panel` shadow.
`labels` is a margin-gated tie-breaker for the lead's complexity
label — the encoder judge's tier routes the task only when it differs from
the lead's and beats the runner-up by `labels_margin` (the task
row's `reason` then says `labels:judge override standard->hard (0.57 vs
0.33)`); `panel`, when active, asks the same judge `needs_security` over every
`review`-kind task the lead spawns without a security reviewer (role
`security-engineer` or a skill containing `security`), and on *yes* guru
spawns one extra `security-engineer` worker on the same task with the
`/review` panel's security focus — once per parent turn, routed like the
task it shadows, its row's `reason` opening with `origin:panel
(needs_security)`, and the lead told to join it. A shadow judge, a
timeout, an error or a missing judge adds nothing. `injection` stays shadow.

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

**Defaults.** guru works without any settings file: every feature that is
meant to be used is on by default, and a parameter exists only to turn it
off or to tweak a number or a choice — never to switch the intended
behaviour on. The routing block is written for you when a remote adapter
is enabled, the ledger records from the first call, a provisioned project
is sandboxed without a `.guru/sandbox.toml`, and an eval that may spend
grades its rubric cases. [`docs/defaults.md`](docs/defaults.md) lists every
knob with its default, what it does and why it exists; the tables below
name the defaults inline.

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
    `["list_dir", "list_tree", "read_file", "search_code", "outline",
    "find_symbol", "run_tests", "check_syntax"]`).
  - `flat = true` — pre-activate the ENTIRE registry on every agent, so a
    capable, large-context model gets the whole toolset up front (costs more
    prompt tokens; off by default).
  - `[tools.limits]` — ceilings for every subprocess the audited tools start
    (`guru/domain/procs.py`): `timeout_s` (default `120`), `cpu_s` (`120`),
    `mem_mb` (`2048`), `fsize_mb` (`64`), `out_kb` (`256`, per output
    stream). A project's `.guru/tools.toml` can override them again.
- `[sampling]` — sampling overrides applied on top of a model's own modelfile
  defaults. Scalar keys here are global (all models); a `[sampling."<model>"]`
  sub-table holds per-model overrides (per-model wins). Empty by default.
- `[bench]`
  - `model_timeout` — per-model wall-clock ceiling (seconds) for the headless
    benchmark; a model that stalls past it is cancelled and recorded as a
    timeout (default `600`; `0` disables the guard).
- `[routing]`, `[[routing.ladder]]`, `[[routing.ladders.<kind>]]` — see
  **Routing (the lead and the ladder)** above (written for you at startup when
  a remote adapter is enabled; `/routing off` turns it off).
- `[decisions]`, `[ledger]` (`enabled`, `turn_line`, both `true`),
  `[pricing."<model>"]` — see **Ledger and decisions** above.
- `[evals]` (`model`, `num_ctx = 8192`) — see `evals/README.md`.
- `[sandbox]` — global container limits and pinned images; see **Sandbox**
  below.

**Per-project** — a `.guru/` folder in the current directory, so project
state travels with the project (created lazily on first write):

- `.guru/GURU.md` — project-specific instructions, appended after the global
  `GURU.md`.
- `.guru/settings.json` — the last-used adapter + model for this project.
- `.guru/domains_allow.txt` — this project's network allow-list (see below).
- `.guru/read_dirs_allow.txt` / `.guru/write_dirs_allow.txt` — approved
  file-read / file-write directories.
- `.guru/tools.toml` — the project's tool policy (see **Audited tools**).
- `.guru/sandbox.toml` — optional: `enabled = false` turns the sandbox off
  for this project, other keys tweak its limits (see **Sandbox**).
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
  allowed directories and gated by the access mode. `read_file` is
  **structural** on long files: a file up to 200 lines
  (`files.READ_OUTLINE_LINES`) reads whole; a longer one without `lines=`
  returns its def/class outline with line ranges, the first 20 lines and
  how to fetch a span -- never the text -- and one `lines='a-b'` call
  returns at most 800 lines (`files.READ_RANGE_SPAN`; an explicit span is
  a deliberate choice, so it is wider than the 200-line structural
  threshold). A whole-file read of a long file is impossible, not
  discouraged.
- **Web** — `web_search`, `web_fetch`, `fetch_github_releases`.
- **Code** — the eight audited verbs below.
- **Sandbox** — `sandbox_run`, `sandbox_python`, `sandbox_diff`,
  `code_health`, `sandbox_submit`, `request_dependency`; advertised only
  in a project with a provisioned sandbox image (see Sandbox below).

`search_tools`, `use_skill` and (for delegation-capable agents) `spawn`,
`check`, `join` — plus `apply_work` in a sandbox project — are always
available and not part of the registry.

**Strict arguments.** Every call -- registry or always-on tool -- is
validated against the tool's spec before anything runs
(`tools.validate_arguments`): an object of named parameters, no unknown
parameter, every required one present, each value of its declared type
(`str` by default; `int`, `bool` and `list` are coerced from the strings a
provider sends, so `depth="3"` and `argv='["pytest", "-q"]'` are fine but
`argv="pytest -q"` is not) and within its enum (`find_symbol kind`,
`spawn kind`/`complexity`, `lint detail`). A rejected call returns ONE line
-- the problem, the expected shape and an example, e.g.
`Invalid arguments: read_file is missing required 'path'. Expected
read_file(path: str, lines?: str), e.g. read_file(path='src/app.py',
lines='40-80')` -- counts as a tool error and is audited with `ok = false`.

**Review tasks are read-only.** `toolpolicy.for_kind(kind)` names the tools
a task of that kind neither sees nor may call; for `review` that is every
write tool (`write_file`, `edit_file`, `apply_patch`, `delete_file`,
`sandbox_run`, `sandbox_python`, `sandbox_submit`, `request_dependency`) —
a reviewer reads and reports; it neither edits nor runs commands in the
sandbox copy (`sandbox_diff` and `code_health` stay). The orchestrator
passes the kind to `tools.initial_tools(..., kind=)` and sets
`session.task_kind`; a hidden tool named anyway answers
`Refused: <tool> is not available to a review task (read-only) ...` and is
recorded with `denied = "kind"`.

### Project brief

`guru.domain.brief` builds, once per checked-out HEAD, what a worker needs
before its first tool call: the file map (directories with counts,
top-level modules), a capped def/class outline per Python module, a symbol
index (name -> `path:line`), how to run the tests (`make test` when the
Makefile has the target, else pytest/unittest as configured) and the
conventions declared in `pyproject.toml` `[tool.*]` sections (plus
`[flake8]` from `.flake8`/`setup.cfg`). The walk stops outlining after 3 s
(`BUILD_BUDGET_S`) and keeps counting; on guru itself a build takes about
0.2-0.6 s. Briefs are stored as JSON under
`~/.guru/briefs/<project-key>/<head_sha>.json` (`guru.repositories.briefs`;
`GURU_BRIEFS_DIR` overrides the root, five briefs per project and twenty
project directories kept, oldest removed on save) and rebuilt when HEAD
changes or on `/brief refresh` (`guru.briefcmd`). The eval runner points
`GURU_BRIEFS_DIR` inside each case's workdir, so fixture copies never
leave briefs under your home. Outline signatures show defaults as `...`
(`def f(a, b=...)`) so no project string literal lands in a system prompt.

`brief.slice(brief, task_text, max_tokens=1500)` is what a worker gets in
its system context (`[project brief]`): the map, the test command, then
the outline of every module the task names (by path, basename or stem) and
the location of every code-like symbol it names, cut at a 4-chars-per-token
estimate. The lead gets `brief.render_map(brief)` in its system
context (`[project map]`). The orchestrator loads the brief once per
(root, HEAD) and hands the same object to the lead and every child
(`Orchestrator.project_brief`); a brief that cannot be built is logged and
left out — it is a saving, never a dependency.

### Tool contract benchmark

`bench/tool_contract.py --model 'Adapter|model' [--out evals/models]
[--web] [--only a,b]` qualifies a model by measurement: one canned
mini-task per tool on a fresh copy of the `cli-tool` fixture, the call
forced where the provider allows it (Anthropic and OpenAI-style
`tool_choice`; Ollama is asked), up to three attempts with the corrective
line fed back. Per tool it records whether a call came, whether it was the
right tool, schema errors, retries, whether the executed call did the job,
tokens, USD and seconds; the result lands in `evals/models/<slug>.json`
(the eval matrix reads it for its `contract` column) and prints as a table.
Web tools need `--web`; the sandbox verbs are listed as skipped until the
fixture has an image.

### Audited tools

guru has no shell tool. Coding and verification go through fixed Python
procedures: the model chooses a **verb and a target**, guru builds the argv
list, runs it under limits, and hands back a short digest. Design and plan:
[`docs/plans/2026-09-24-audited-tools-plan.md`](docs/plans/2026-09-24-audited-tools-plan.md).

The verbs:

- `outline(path)` — def/class map of a file with line ranges (non-Python:
  the first 40 numbered lines), so the model can pick lines to read instead
  of reading whole files.
- `find_symbol(name, kind='')` — definitions (AST) and references (word
  grep) across the project's `.py` files; `kind` filters `def`/`ref`.
- `run_tests(target='', k='', maxfail=1, detail='')` — pytest (or unittest,
  per policy) via a fixed argv; the digest is the summary line plus the
  failing test ids with their assertion line.
- `check_syntax(path)` — `py_compile` in-process; `ok` or the SyntaxError.
- `lint(path='')` — flake8, and mypy when configured; skips a linter that
  is not installed and says so.
- `git_status()` — changed files (porcelain); read-only.
- `git_diff(path='', detail=False)` — `--stat` digest, or the unified diff
  for one path; read-only, never `add`/`commit`/`checkout`.
- `apply_patch(diff)` — a unified diff for one or more project files,
  validated hunk by hunk before anything is written, all-or-nothing, through
  the same write gates and sha ledger as `edit_file`; a deletion
  (`+++ /dev/null`) is applied when its body equals the file exactly.

**Digest and detail.** Every verb returns at most a few hundred characters
by default; a `detail` argument expands one item (one failing test, one
file's diff, one linter's issues). The full subprocess output goes to the
log and the ledger transcript, never to the model — that is what keeps a
test run from flooding the context.

**Limits.** Every subprocess runs with a fixed argv (a shell binary as the
program is refused), in its own process group (killed whole on timeout,
strays included), with an environment built from scratch (a secret in
guru's own environment never reaches the child), rlimits on CPU, memory and
file size, and output captured to files under a throw-away `HOME` so
`fsize_mb` bounds it on disk and only the first `out_kb` reaches guru. The
ceilings come from `[tools.limits]` in `settings.toml` (`timeout_s`,
`cpu_s`, `mem_mb`, `fsize_mb`, `out_kb`) and a project can lower them again
in its policy file. The runner refuses a working directory outside the read
allow-list; that refusal is recorded like an access-mode denial.

**Policy file.** A project can narrow the toolset with `.guru/tools.toml`:

```toml
[tools]
enabled = ["read_file", "search_code", "run_tests"]   # non-empty: allowlist
disabled = ["web_search", "web_fetch"]                 # always wins

[tools.tests]
runner = "pytest"          # pytest | unittest

[tools.limits]
timeout_s = 60             # per-project subprocess ceilings (see [tools.limits])
```

No file means everything is enabled. `disabled` wins over `enabled`; a
non-empty `enabled` list is an allowlist for registry tools. The always-on
tools (`search_tools`, `use_skill`, `spawn`, `check`, `join`,
`apply_work`) are never subject to it. A disabled tool is not advertised at all — it is not
pre-activated, `search_tools` does not return it and it is absent from the
tool schemas the model sees — and if the model names it anyway the call
answers `Tool '<name>' is disabled by .guru/tools.toml` and is recorded with
`denied = "policy"`.

**Fail closed.** If the policy file is present but invalid (unknown key,
unknown runner, bad limit, broken TOML) or unreadable, guru reports the
problem at startup (naming the file) and disables *every* registry tool
until it is fixed or removed — only `search_tools`, `use_skill`,
`spawn`, `check`, `join`, `apply_work` remain. A policy meant to restrict tools can never widen
them by mistake.

**Audit.** Every tool call — allowed, refused or unknown — writes one row to
the ledger's `tool_events` stream (tool, args head, seconds, bytes produced
vs shown to the model, files touched, denial: `policy`, `mode` or
`kind`). `/tools` shows the last turn's rows; `bench/ledger_report.py`
aggregates them per tool in its **Tools** section.

### Sandbox

The sandbox runs the model's code changes in a container on a **copy** of
the project, and lets them back into the real tree only through a quality
gate. Design and plan:
[`docs/plans/2026-09-24-sandbox-design-and-plan.md`](docs/plans/2026-09-24-sandbox-design-and-plan.md).

**Requirements.** Docker CLI against [Colima](https://github.com/abiosoft/colima)
(Apple silicon; `docker info` must succeed) and a uv-managed project:
`pyproject.toml` + `uv.lock`. Only lockfile-declared packages exist in the
image; the model never installs anything.

**Default: on once provisioned.** No file is needed: as soon as the
project has a provisioned image (below) the sandbox verbs replace the
direct write tools. Guru does not provision by itself (an image build takes
minutes, needs Colima and asks to allow the PyPI hosts), so
`/sandbox provision` is the one explicit step. `.guru/sandbox.toml` is
optional — the off switch for a project, or a per-project tweak of the
`[sandbox]` limits in `~/.guru/settings.toml` (the project file wins key by
key; `enabled` is read from the project file only):

```toml
[sandbox]
# enabled = false                            # turn the sandbox off for this project
# base_image = "python:3.12-slim@sha256:…"   # digest-pinned, or refused
# cpus = 2.0
# memory_mb = 2048
# pids = 256
# timeout_s = 600                            # wall clock per container run
```

With `enabled = false` the verbs are hidden (and refused if named), the
write tools come back and `/sandbox provision` refuses.

**Provision.** `/sandbox provision` (or `--force`) generates a Dockerfile
from `pyproject.toml` + `uv.lock`, asks once to allow `pypi.org` and
`files.pythonhosted.org` (the normal web-access question), and builds the
image on an *internal* Docker network whose only other member is a
digest-pinned tinyproxy with an allow-list generated from
`.guru/domains_allow.txt` (CONNECT to allow-listed hosts on 443 only; every
request is logged as `net_events`). The image record lives under
`~/.guru/sandbox/<project>/`; it is rebuilt only when the lockfile changes.
The `sandbox_*` verbs are advertised to the model only while that record
exists — and while it does, the direct write tools (`write_file`,
`edit_file`, `apply_patch`, `delete_file`) are hidden from the model and
refused if named anyway: in a sandboxed project the gate is the only write
path.

**The verbs.**

- `sandbox_run(argv)` — a fixed argv as a JSON list (`argv[0]` one of
  `python`, `pytest`, `uv`, `ruff`, `mypy`, `flake8`, `make`; shells
  refused; a plain string is split on whitespace only) in the task's
  copy inside the container: `--network none`, unprivileged user, all
  capabilities dropped, read-only root, tmpfs `/tmp` (also `HOME`),
  `.git` read-only, cpu/memory/pid limits, wall-clock kill. Digest: exit
  code and the first lines of output; `detail` returns the last 4 KB.
- `sandbox_python(code)` — runs a Python snippet the same way (arbitrary
  code is fine *inside* the sandbox; that is what it is for).
- `sandbox_diff()` — per-file `+/-` counts of the copy against the project.
- `code_health(path='')` — the copy's changed or new Python functions with
  their metrics (lines, cyclomatic complexity, nesting depth, arguments,
  returns) and a verdict per function — `degraded | improved | unchanged`
  against the gate's thresholds — so a worker can fix a degraded function
  before it submits; `path` narrows to one file (an unchanged file shows
  its current metrics). Digest capped at 600 chars.
- `sandbox_submit(intent)` — the only way changes reach the real tree: the
  copy's diff goes through the gate with the agent's stated intent.
- `request_dependency(name, constraint)` — records a package request;
  installs nothing.

**The gate.** Two stages. Deterministic rules first: paths inside the
project and outside the noise dirs, a size cap, the secret scanner over
added lines, red-flag patterns (process/network/eval primitives,
encoded blobs, skipped tests, removed asserts, CI/config/conftest edits),
the `destructive` rules and the code-health rules (both below).
Then an AI reviewer (the configured `gate` judge, else the routing ladder's
`standard` rung, else the session model) answers a fixed question set over
the user's request, the task, the intent, guru's change summary and
code-health rows, and the diff. Three verdicts:

- `intended` — applied via `apply_patch` (auto mode; ask mode shows the
  diff and asks first).
- `unclear` — the user is asked, with the reviewer's reasons, in every mode
  (auto never waves it through).
- `suspicious` — refused; nothing is applied and the reasons are printed.

File deletions go through the same gate: a file deleted in the copy shows
up as a `delete` flag (`deletes <path> (N lines)`, informational) and in
the change summary the reviewer sees, and the reviewer answers whether
every deletion is something the *user's request* asked for
(`deletions_requested`); a `no` makes the verdict `unclear`, a `yes` with
an otherwise clean review is applied like any edit. A file *emptied* in
the copy (every line removed, the file kept) is treated as a deletion
throughout: `empties <path> (N lines; file kept)`, an `emptied` marker in
the change summary, and the destructive counts below. `apply_patch`
accepts deletions (`+++ /dev/null`) whose body equals the file exactly,
inside and outside the sandbox — one patch algebra.

*Destructive changes* are never auto-applied: a rename (the patch algebra
refuses them; the headers are scanned only when the diff failed to
parse), more than 3 files deleted or emptied, more than 200 lines removed
*gross* — added lines offset nothing, so padding a removal with blank or
comment lines (or anything else) buys nothing; the flag reports how many
added lines were content — or the deletion or emptying of any test file
(under a `tests/`, `test/`, `testing/` or `tests_*/` directory, or named
`test_*.py`, `*_test.py`, `*_tests.py`, `conftest.py`) raises a
`destructive` flag naming the files, and the verdict is `unclear` at
most — the user is asked, whatever the reviewer says. The two thresholds
hold over the *whole task*, not per submit: guru keeps a per-task tally
of the files deleted and lines removed by the submits it already
applied, and a submit that would push the running total over a threshold
is flagged with both numbers (`deletes 2 files now, 5 in this task (more
than 3)`). Once a task is over the removed-lines threshold, a later
submit is flagged for it only when it removes at least one substantive
(non-blank, non-comment) line itself, and a submit that deletes no file
is never flagged for the task's earlier deletions. The tally is cleared
when the task ends; for an agent that runs no task (the main agent in a
hands-on session) the tally is the current user turn's, so a new turn
starts from zero while the agent's working copy lives on. A file reduced
to blank and comment lines only (a `# removed` left behind) counts as
emptied.

*Code health* (`guru/domain/health.py`, pure `ast`): for every Python
function the diff changes or adds, the gate compares the copy's baseline
(`git show HEAD:<path>` in the copy) with the result and measures lines,
cyclomatic complexity (branches, `except`, `with`, `assert`, comprehension
`if`s, ternaries, `match` cases and boolean operands), maximum nesting
depth, argument count (`self`/`cls` excluded) and return count. A function
is `degraded` when a metric crosses a threshold it was under before
(lines > 60, complexity > 10, nesting > 4, args > 6), when a new function
is over one, or when a metric already over its threshold grows by more
than 20%; `improved` when an over-threshold metric shrank; `unchanged`
otherwise — pre-existing debt the change leaves alone is not flagged.
Each degraded function is a `health` flag (`unclear` at most, never
`suspicious`), and the reviewer's packet carries a "Code health" block
computed by guru outside the untrusted fences listing the degraded and
improved functions with before→after metrics. A source that does not
parse is skipped, never an error; at most 20 Python files per diff are
measured, and none when the deterministic rules already found the change
suspicious (the verdict is settled). `code_health(path)` measures only
that file (no other baseline is read) and refuses a path that resolves
outside the copy — a symlink out of it included.

*The baseline is guru's, not the worker's.* The gate compares the copy
with the commit `prepare_copy` made, so that commit must be out of the
container's reach: the copy is mounted read-write, but its `.git` is
mounted read-only on top of it (`-v <copy>/.git:/work/.git:ro`), and the
baseline commit's sha is kept in guru's memory next to the copy's path
(the copy's marker file records it too, as documentation — nothing reads
it back once the copy is made). `sandbox_diff`, `code_health` and
`sandbox_submit` all verify `git rev-parse HEAD` against the kept sha
before reading anything; a copy whose baseline moved is refused
(`Refused: sandbox copy baseline changed`) and discarded, and so is one
whose marker the worker removed or rewrote (`Refused: the sandbox copy's
marker file was removed or changed`) — the directory is removed either
way. The diff is read with `git diff --text`, so a `.gitattributes` the
worker writes cannot turn it into `Binary files differ`. Inside the
container `HOME` and
`XDG_CACHE_HOME` point at the tmpfs `/tmp` (uid 1000 has no home and the
root is read-only), so tool caches and tests that write under `HOME`
work and nothing persists.

In read-only mode a submit reports the diff and stops before the reviewer
is consulted (nothing could be applied, so the diff never leaves the
machine for no decision).

Every submit is a `sandbox_events` row and a `decisions` row for the
reviewer; `/sandbox gate` lists this run's verdicts.

**Dependency requests.** `request_dependency` only records. The user runs
`/sandbox deps` to list, `/sandbox deps apply <name>` to approve: guru runs
`uv add` in a provisioning container through the proxy on a copy, shows the
lockfile diff (packages added/removed/changed), brings `pyproject.toml` +
`uv.lock` back through `apply_patch`, and rebuilds. If the rebuild fails
after the lockfile landed, the digest says so and the request stays pending
for a retry. `/sandbox deps request <spec>` records one by hand.

**What is and is not contained.** Execution has no network at all;
provisioning reaches only allow-listed hosts through the logged proxy. The
container sees the copy, never the real tree, your home, your environment
or any credential; the copy is the `git ls-files` positive list minus noise
dirs, `.env*` and anything the secret scanner flags. Not contained: a
kernel escape lands in the Colima VM (accepted for local development); the
build container runs as root with a writable root, so the proxy allow-list
and the lockfile are the controls there; the internal network still reaches
the VM's own gateway-address listeners.

**Commands.** `/sandbox status` (runtime, settings, image, pending
requests, task copies), `/sandbox provision [--force]`, `/sandbox gate`,
`/sandbox deps [request <spec> | apply <name>]`. `make test-sandbox` runs
the container integration tests (skipped without Colima); the eval suite
has three `sandbox` cases (`python -m guru.evals list --tags sandbox`, see
`evals/README.md`).

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

- `make check` — the gate before a commit: `lint`, `typecheck`, `test`
  and `test-sandbox` in that order (needs Colima for the last).
- `make test` — run the test suite (pytest; container tests excluded).
- `make test-sandbox` — the container integration tests against the
  local Colima (skipped when `docker info` fails).
- `make lint` — flake8 over `guru bench tests evals`.
- `make typecheck` — mypy over `guru`.
- `make bench` — run the headless coding-model benchmark, writing
  `bench/results-<timestamp>.json` plus a companion `transcript-<timestamp>.json`.
- `make bench-plot` — plot the latest results (override with `RESULTS=...`).
