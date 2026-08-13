# Design: services/domain refactor with pluggable provider adapters

Date: 2026-08-13

## Goal

Move `guru` from a single `guru.py` script to a package with a domain layer and
pluggable provider adapters, then add Anthropic providers alongside Ollama.

Three adapters, all shown in `/models` grouped by adapter; selecting a model
switches the active adapter + model:

- **Ollama** — local, existing behaviour.
- **Anthropic (local)** — the `anthropic` SDK with an API key + `base_url`
  pointed at a local endpoint that speaks the Anthropic Messages API.
- **Anthropic Enterprise** — the `anthropic` SDK with OAuth (no API key), token
  obtained from the `ant` CLI profile; sends the `oauth-2025-04-20` beta header.

Full tool parity: the tool directory (`web_search`, `web_fetch`,
`fetch_github_releases`, discovered via `search_tools`) works on every adapter.

## Package layout

```
guru/
  __main__.py        # python -m guru → cli.main()
  cli.py             # prompt loop, slash commands, cross-adapter /models
  config.py          # paths, ~/.guru/adapters.toml, GURU.md assembly, allow-list
  ui.py              # Console+theme, status bar, _pick, key bindings, terminal modes
  domain/
    tools.py         # TOOL_REGISTRY, search_tools, executor, domain allow-list gating
    conversation.py  # message helpers, /save, /resume, compaction (hybrid D)
  adapters/
    base.py          # Adapter ABC + ModelInfo + Usage
    ollama.py        # existing ollama logic + daemon check/model pull
    anthropic.py     # Messages API; one class, two configured instances
tests/
start.sh             # thin launcher → uv run python -m guru
```

## Neutral message format

The neutral conversation format is the **normalized message dict** already used
by `_message_to_dict`: `{role, content, tool_calls?, tool_name?}`. Both adapters
translate to/from it. This keeps `/save`, `/resume`, and compaction unchanged and
provider-independent (a chat saved on Ollama can resume on Claude).

- Ollama: dicts pass through nearly as-is; `options={num_ctx}`.
- Anthropic: `system` → top-level `system`; user/assistant → content blocks;
  `tool_calls` → assistant `tool_use` blocks; `tool_name`+content → user
  `tool_result` blocks; adaptive thinking.

## Adapter interface

```python
@dataclass
class ModelInfo: adapter, model_id, label, context_window, size
@dataclass
class Usage: input_tokens, output_tokens, context_used

class Adapter(ABC):
    name: str
    def available(self) -> bool: ...
    def list_models(self) -> list[ModelInfo]: ...
    def run_turn(self, messages, active_tools, execute_tool, ui) -> Usage: ...
```

`run_turn` owns the provider's tool-calling loop for one user turn: translate,
call the provider, invoke `execute_tool(name, args)` on each tool request (domain
callback that does allow-list gating + `search_tools` activation + runs the tool),
feed results back, loop to final text. Rendering goes through the shared `ui`
object. Tool execution and gating stay provider-agnostic.

## Config — `~/.guru/adapters.toml`

```toml
[[adapter]]
name = "Ollama"
type = "ollama"
url  = "http://localhost:11434"

[[adapter]]
name = "Anthropic (local)"
type = "anthropic"
auth = "api_key"
base_url = "http://localhost:8080"
api_key_env = "GURU_ANTHROPIC_API_KEY"

[[adapter]]
name = "Anthropic Enterprise"
type = "anthropic"
auth = "oauth"
profile = "default"
```

Secrets stay in env / the `ant` profile, never in the file. First run writes the
Ollama entry plus commented Anthropic templates. OAuth token via
`ant auth print-credentials --access-token [--profile X]`, refreshed per session.

## /models — cross-adapter selector

`_pick` renders a grouped list: adapter headers (skipped by arrows) + models with
context window and ✓ on active. Only reachable adapters are queried; an
unreachable adapter shows "unavailable". Selecting sets active adapter+model and
refreshes the status bar. Anthropic context windows from `client.models.list()`
(`max_input_tokens`); Ollama from `show()`.

## Preserved features

Status bar, `/save`/`/resume`/`/compact`/`/search`, domain allow-list,
Ctrl+C/paste/scroll-region handling — all preserved, relocated into `ui.py` /
`domain/`. Adapters report `Usage` per turn (Ollama `prompt_eval_count`,
Anthropic `usage.input_tokens`); context window from `ModelInfo`. `💪` size =
param size (Ollama) / model tier or blank (Anthropic). `num_ctx` is Ollama-only.
The allow-list gates only model-initiated web tools, not the provider transport.

## Phasing

- **Phase 1** — carve `guru.py` into the package; all current behaviour behind
  the Ollama adapter. Zero feature change. flake8 + pytest + manual run confirm
  parity. `start.sh` becomes a thin launcher; ollama daemon check/pull moves into
  the ollama adapter.
- **Phase 2** — Anthropic adapter (both auth modes), neutral translation + tool
  parity, cross-adapter `/models`, config file.
