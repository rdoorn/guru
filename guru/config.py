"""Paths, constants, setup, system-prompt assembly, and the domain allow-list.

This module owns filesystem locations and pure configuration. It imports no
other guru module, so everything else can depend on it freely.
"""
import json
import os
from pathlib import Path
from urllib.parse import urlparse

# Global config lives in ~/.guru; project-specific state lives in a .guru/
# folder inside the current project so it travels with the project.
GURU_HOME = Path(os.path.expanduser('~/.guru'))
GURU_MD_PATH = GURU_HOME / 'GURU.md'                 # global base persona
ADAPTERS_PATH = GURU_HOME / 'adapters.toml'          # adapter configuration
GURU_SKILLS_DIR = GURU_HOME / 'skills'               # roles & skills overlays
MODEL_CTX_PATH = GURU_HOME / 'model_ctx.json'        # per-model chosen context
GLOBAL_SETTINGS_PATH = GURU_HOME / 'settings.toml'   # global user settings

PROJECT_GURU_DIR = Path.cwd() / '.guru'
PROJECT_GURU_MD = PROJECT_GURU_DIR / 'GURU.md'       # appended to the global
DOMAINS_ALLOW_PATH = PROJECT_GURU_DIR / 'domains_allow.txt'  # per-project
READ_DIRS_ALLOW_PATH = PROJECT_GURU_DIR / 'read_dirs_allow.txt'   # read list
WRITE_DIRS_ALLOW_PATH = PROJECT_GURU_DIR / 'write_dirs_allow.txt'  # write list
PROJECT_MEMORY_DIR = PROJECT_GURU_DIR / 'memory'     # saved conversations

# Access mode (session-level policy). Separate from the allow-lists: it decides
# whether we prompt, auto-approve, or refuse. read-only refuses writes; ask
# prompts per not-yet-allowed target; auto approves silently (filling the
# lists). Path resolution / escape checks apply in every mode.
MODE_READ_ONLY = 'read-only'
MODE_ASK = 'ask-for-changes'
MODE_AUTO = 'auto'
MODES = (MODE_READ_ONLY, MODE_ASK, MODE_AUTO)
MODE = MODE_ASK
# Remembers the last-used adapter + model for this project.
PROJECT_SETTINGS_PATH = PROJECT_GURU_DIR / 'settings.json'

# Search-engine backend host. web_search gates on this so "allow internet
# access at least once" maps to approving the engine. Structured as a
# constant so additional engines can each declare their own backend host.
SEARCH_BACKEND_DOMAIN = 'duckduckgo.com'

# ollama's default context window when a model's modelfile does not set one.
DEFAULT_NUM_CTX = 4096
# Compact the conversation when occupancy crosses this fraction of num_ctx.
COMPACT_AT = 0.85
# Number of most-recent turn-groups kept verbatim during compaction.
KEEP_RECENT_GROUPS = 4

# Tool-output retention thresholds (chars). Overridable via settings.toml's
# [context] section. Below the threshold a tool result is kept verbatim; above
# it, web results are query-summarized and large code reads are outlined.
WEB_SUMMARIZE_OVER_CHARS = 6000
OUTLINE_FILE_OVER_CHARS = 8000

# Tools pre-activated on every agent so weaker models can call them directly
# without first calling search_tools (which they often "announce" instead of
# doing). Overridable via settings.toml's [tools] preactivate = [...].
PREACTIVATE_TOOLS = ['list_dir', 'list_tree', 'read_file', 'search_code']

# Flat toolset: when true, EVERY registry tool is pre-activated on each agent,
# so a capable model gets the whole toolset up front and never needs the
# search_tools discovery hop. Costs more prompt tokens per turn (all schemas
# are always sent), so it's off by default and best for large-context models.
# Overridable via settings.toml's [tools] flat = true.
FLAT_TOOLS = False

# Sampling overrides applied on top of a model's own modelfile defaults (the
# authoritative per-model source). Empty by default so each model keeps its
# author-tuned params. settings.toml [sampling] holds global scalar overrides;
# [sampling."<model>"] sub-tables hold per-model overrides (per-model wins).
SAMPLING: dict = {}              # global scalar overrides (all models)
SAMPLING_PER_MODEL: dict = {}    # {model_id: {param: value}}

# Per-model wall-clock ceiling for the headless benchmark (guru.bench). A model
# that stalls past this is cancelled cooperatively (the adapters' cancel path —
# Ollama aborts mid-stream) and recorded as a timeout, so one slow/thrashing
# model can't hang the whole suite. Overridable via settings.toml [bench]
# model_timeout (seconds). Set above the slowest legitimate run (a real 24B run
# can take ~400s); 0 disables the guard.
BENCH_MODEL_TIMEOUT = 600

# GPU auto-fit: when a model is first selected (and the user gave no explicit
# --num-ctx), guru picks the largest context that stays entirely on the GPU.
# It is only a default: a stored per-model choice or a manual /context or
# --num-ctx always wins, and the reported architecture max is never touched.
# The fit is measured from Ollama's own memory report (ollama.ps) rather than
# guessed from a RAM fraction: two probe loads reveal the real weights, the
# real per-token KV cost (so it is correct for f16 OR q8_0), and — on a spill —
# the true GPU budget. GPU_FIT_SAFETY stays a little under the measured budget.
GPU_FIT_SAFETY = 0.95         # use 95% of the measured GPU budget
CTX_PROBE_HIGH = 32768        # upper probe context for the fit measurement
# The constants below are only used by the metadata fallback estimate, taken
# when ollama.ps is unavailable (e.g. a remote daemon that hides memory).
GPU_MEM_HEADROOM = 0.20       # fraction of GPU memory left free (other apps)
MAC_GPU_FRACTION = 0.75       # Apple-Silicon Metal working set ~= 75% of RAM
KV_CACHE_BYTES = 2.0          # bytes per KV element (f16); q8_0 ~= 1.0
FIT_OVERHEAD_BYTES = 512 * 1024 * 1024   # compute buffers / activations slack

# The GURU.md contents are appended verbatim to the model's system prompt, so
# this default holds only model-directed instructions — no human-facing notes
# (those belong in the README, not in every request's context).
DEFAULT_GURU_MD = """## Persona

- Be concise and direct.
- Do not use emoji or icons in responses.
- Cite sources when you use a tool result.

## Rules

- Do not invent facts. If a tool did not return something, say so.
"""

DEFAULT_ADAPTERS_TOML = """# guru adapter configuration.
# Each [[adapter]] block is a provider shown in /models. Secrets are never
# stored here — use environment variables or an `ant auth login` profile.

[[adapter]]
name = "Ollama"
type = "ollama"
url  = "http://localhost:11434"

# Uncomment and configure to add Anthropic providers:
#
# [[adapter]]
# name = "Anthropic (local)"
# type = "anthropic"
# auth = "api_key"
# base_url = "http://localhost:8080"
# api_key_env = "GURU_ANTHROPIC_API_KEY"
#
# [[adapter]]
# name = "Anthropic Enterprise"
# type = "anthropic"
# auth = "oauth"
# profile = "guru"   # one-time: `ant auth login --profile guru`
#
# [[adapter]]
# name = "LiteLLM"
# type = "litellm"                       # OpenAI-compatible proxy
# base_url = "https://proxy.example/v1"  # include /v1
# api_key_env = "LITELLM_KEY"            # env var holding the virtual key
# api_key = "sk-..."                     # or inline (used if the env is unset)
# models = ["azure/gpt-4.1"]            # optional allowlist
"""

SYSTEM_PROMPT = """
You are a helpful assistant with a tool directory. Each turn you begin with a
single tool: search_tools. To do anything else, call search_tools with a short
phrase naming the ACTION you want — not the user's question. It returns
matching tools; call those directly by name, and never call a tool it has not
returned.

You DO have web and local filesystem access, through these tools. Never say
you cannot access the internet or files — call search_tools for the capability
first, then use the tool it returns. Act rather than explaining how.

Examples (question → search_tools phrase):
  list files here → "list directory files"
  read lines 40-60 of cli.py → "read file lines"
  find where a function is defined → "grep search code"
  create or write a file → "write file"
  change or replace text in a file → "edit file"
  delete or remove a file → "delete file"
  fetch this URL / query an endpoint → "fetch webpage url"

Do not use tools for math, logic, coding, or stable facts from your training.
If results are weak, refine and search again. Cite sources and state only what
the results show. If a needed detail (a name, a location) is missing, ask.
Before concluding code or a feature is missing, grep for its definition and
read the file that defines it; when reviewing a file, follow its local
imports. Never infer that something is absent from a single file.

To create, change, or delete a file you MUST call write_file, edit_file, or
delete_file in this turn and wait for it to return success — never state that
a file was written, changed, or deleted unless a tool call did it. Do not
restate the file's contents afterwards; the change is shown to the user. If no
tool exists for a request, say so and stop.
Never state a file's contents from memory — read it. edit_file needs the
file's sha: reuse the one your most recent read_file, write_file, or edit_file
of that file returned (they all return it) — you need not re-read if you
already hold a current sha. An '[open files]' list may be present with current
shas for files you have touched; reuse those directly for edit_file. If
edit_file reports a sha mismatch, the file changed underneath you; read it
again to refresh the sha, then retry.
"""

# Appended to the system prompt of delegation-capable agents (TUI only), to
# steer heavy tool output out of the main context and into sub-agents.
DELEGATION_HINT = (
    "When a request spans multiple files or several concerns (correctness,"
    " security, design, reliability, tests), DECOMPOSE it instead of"
    " inspecting everything yourself: spawn one sub-agent per concern, in"
    " parallel, each with the role+skill that fits, then join and synthesise"
    " their findings. Each sub-agent reads the bulk in its own context and"
    " returns only its conclusion, keeping yours small.\n"
    "Example — to review this codebase, spawn in parallel:\n"
    "  spawn(task='review the code for correctness, readability, tests',"
    " role='developer', skill='code-review')\n"
    "  spawn(task='review the code for injection, authz, secrets, path"
    " traversal, vulnerable deps', role='security-engineer',"
    " skill='code-review')\n"
    "then join both and write one consolidated report. Add an architect"
    " (design) or SRE (reliability) sub-agent when those concerns apply."
    " Use check to poll and join to be resumed when a group finishes."
    " Prefer delegating a domain panel over reading many files yourself."
)

# Deterministic code-review panel (the /review command) and the target of the
# delegation steering: each entry is (role, skill, focus) — one specialist
# sub-agent to spawn in parallel. Kept small on purpose; architect/SRE are
# available in the catalog for the model to add when design/ops matter.
REVIEW_PANEL = [
    ('developer', 'code-review',
     'correctness, readability, tests, and maintainability'),
    ('security-engineer', 'code-review',
     'security: injection, authz, secrets, path traversal, vulnerable deps'),
]

# Delegation nudge: if a delegation-capable MAIN agent answers a broad task
# (>= this many file reads) having spawned no sub-agent, nudge it once to
# decompose into a parallel domain panel. Set 0 to disable the nudge.
DELEGATION_NUDGE_MIN_READS = 3
DELEGATION_READ_TOOLS = {'read_file', 'search_code', 'list_dir', 'list_tree'}


def review_prompt(area: str = 'the repository') -> str:
    """A delegation-first review instruction built from REVIEW_PANEL: spawn the
    fixed domain panel in parallel, then join and synthesise one report. Used
    by the /review command so the multi-agent path is exercised on demand
    without the user hand-writing the spawn calls."""
    lines = [
        f"Review {area}. Delegate a domain panel — spawn these sub-agents in"
        " parallel, then join them and write ONE consolidated report grouped"
        " by severity. Spawn exactly:"]
    for role, skill, focus in REVIEW_PANEL:
        lines.append(
            f"- spawn(task='Review {area} for {focus}. Give concrete findings"
            f" with file:line and a suggested fix.', role='{role}',"
            f" skill='{skill}')")
    lines.append(
        "Spawn all of them now, then join and synthesise their findings."
        " Do not review everything yourself.")
    return "\n".join(lines)


# Domains approved for model-initiated web access, loaded at startup.
ALLOWED_DOMAINS: set = set()
# Directories approved for model-initiated file READS (resolved absolute
# paths). Loaded from the per-project allow-list; new ones (including the
# working directory) are approved once and persisted. See load below.
ALLOWED_READ_DIRS: set = set()
# Directories approved for model-initiated file WRITES — a separate list, so
# read access never implies write access.
ALLOWED_WRITE_DIRS: set = set()


def ensure_setup() -> None:
    """Create the global ~/.guru dir, a default GURU.md, and adapters.toml.

    Project state (.guru/ in the current directory) is created lazily on
    first write so read-only sessions do not litter arbitrary directories.
    """
    GURU_HOME.mkdir(parents=True, exist_ok=True)
    if not GURU_MD_PATH.exists():
        GURU_MD_PATH.write_text(DEFAULT_GURU_MD, encoding='utf-8')
    if not ADAPTERS_PATH.exists():
        ADAPTERS_PATH.write_text(DEFAULT_ADAPTERS_TOML, encoding='utf-8')


def load_allowed_domains() -> set:
    """Read the allow-list file into a set of lowercased domains."""
    try:
        lines = DOMAINS_ALLOW_PATH.read_text(encoding='utf-8').splitlines()
    except OSError:
        return set()
    return {ln.strip().lower() for ln in lines if ln.strip()}


def persist_domain(domain: str) -> None:
    """Append a newly approved domain to the project allow-list file."""
    DOMAINS_ALLOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DOMAINS_ALLOW_PATH.open('a', encoding='utf-8') as fh:
        fh.write(domain + '\n')


def _load_dir_list(path) -> set:
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return set()
    return {ln.strip() for ln in lines if ln.strip()}


def _append_dir(path, directory: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(directory + '\n')


def load_allowed_read_dirs() -> set:
    """Read the read-access allow-list into a set of resolved path strings."""
    return _load_dir_list(READ_DIRS_ALLOW_PATH)


def persist_read_dir(directory: str) -> None:
    """Append a newly approved read directory to the project allow-list."""
    _append_dir(READ_DIRS_ALLOW_PATH, directory)


def load_allowed_write_dirs() -> set:
    """Read the write-access allow-list into a set of resolved path strings."""
    return _load_dir_list(WRITE_DIRS_ALLOW_PATH)


def persist_write_dir(directory: str) -> None:
    """Append a newly approved write directory to the project allow-list."""
    _append_dir(WRITE_DIRS_ALLOW_PATH, directory)


def domain_of(url: str) -> str:
    """Return the lowercased hostname of a URL, port stripped."""
    host = urlparse(url).hostname
    if not host:
        # Bare host without a scheme (e.g. "example.com/path").
        host = urlparse('//' + url).hostname
    return (host or url).lower()


def load_settings() -> dict:
    """Load the project's last-used adapter + model, or {} if none."""
    try:
        return json.loads(
            PROJECT_SETTINGS_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    """Persist the project's last-used adapter + model."""
    PROJECT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROJECT_SETTINGS_PATH.write_text(
        json.dumps(data, indent=2), encoding='utf-8')


def build_system_prompt() -> str:
    """Assemble the system prompt: built-in + global GURU.md + project GURU.md.

    The built-in prompt is always first so the search_tools mechanism is
    never lost. The project .guru/GURU.md extends (appends to) the global one.
    """
    parts = [SYSTEM_PROMPT.strip()]
    for path in (GURU_MD_PATH, PROJECT_GURU_MD):
        try:
            text = path.read_text(encoding='utf-8').strip()
        except OSError:
            continue
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def load_adapter_configs() -> list:
    """Parse adapters.toml into a list of adapter config dicts."""
    try:
        import tomllib
    except ModuleNotFoundError:                       # Python < 3.11
        import tomli as tomllib                        # type: ignore
    try:
        data = tomllib.loads(ADAPTERS_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    return data.get('adapter', [])


def settings_section(name: str) -> dict:
    """Return a top-level table from ~/.guru/settings.toml (or {})."""
    try:
        import tomllib
    except ModuleNotFoundError:                       # Python < 3.11
        import tomli as tomllib                        # type: ignore
    try:
        data = tomllib.loads(
            GLOBAL_SETTINGS_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    section = data.get(name, {}) if isinstance(data, dict) else {}
    return section if isinstance(section, dict) else {}


def load_context_settings() -> dict:
    """Return the [context] table from ~/.guru/settings.toml (or {})."""
    return settings_section('context')


def _apply_settings() -> None:
    """Apply settings.toml overrides (retention, pre-activation, sampling)."""
    global WEB_SUMMARIZE_OVER_CHARS, OUTLINE_FILE_OVER_CHARS
    global PREACTIVATE_TOOLS, SAMPLING, SAMPLING_PER_MODEL
    global BENCH_MODEL_TIMEOUT, FLAT_TOOLS
    ctx = load_context_settings()
    try:
        WEB_SUMMARIZE_OVER_CHARS = int(
            ctx.get('web_summarize_over_chars', WEB_SUMMARIZE_OVER_CHARS))
        OUTLINE_FILE_OVER_CHARS = int(
            ctx.get('outline_file_over_chars', OUTLINE_FILE_OVER_CHARS))
    except (TypeError, ValueError):
        pass
    tl = settings_section('tools')
    pre = tl.get('preactivate')
    if isinstance(pre, list):
        PREACTIVATE_TOOLS = [str(x) for x in pre]
    FLAT_TOOLS = bool(tl.get('flat', FLAT_TOOLS))
    sampling = settings_section('sampling')
    # Scalar keys are global overrides; sub-tables are per-model overrides.
    SAMPLING = {k: v for k, v in sampling.items()
                if not isinstance(v, dict)}
    SAMPLING_PER_MODEL = {k: v for k, v in sampling.items()
                          if isinstance(v, dict)}
    bench = settings_section('bench')
    try:
        BENCH_MODEL_TIMEOUT = int(
            bench.get('model_timeout', BENCH_MODEL_TIMEOUT))
    except (TypeError, ValueError):
        pass


def _toml_value(value: object) -> str:
    """Serialize a scalar/list value to TOML."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return '[' + ', '.join(_toml_value(v) for v in value) + ']'
    text = str(value).replace('\\', '\\\\').replace('"', '\\"')
    return f'"{text}"'


def save_adapter_configs(configs: list) -> None:
    """Write adapter config dicts back to adapters.toml.

    Reformats the file (inline comments are not preserved). Only structural
    config is written — secrets stay in env vars / the ant profile.
    """
    order = ('name', 'type', 'enable', 'auth', 'url', 'base_url',
             'api_key_env', 'profile', 'models', 'thinking')
    lines = [
        '# guru adapter configuration.',
        '# Managed by the /adapters command. Secrets are never stored here —',
        '# use environment variables or an `ant auth login` profile.',
        '',
    ]
    for cfg in configs:
        lines.append('[[adapter]]')
        for key in order:
            if key in cfg and cfg[key] is not None:
                lines.append(f'{key} = {_toml_value(cfg[key])}')
        # Preserve any keys not in the known order.
        for key, val in cfg.items():
            if key not in order and val is not None:
                lines.append(f'{key} = {_toml_value(val)}')
        lines.append('')
    GURU_HOME.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(lines).rstrip() + '\n'
    ADAPTERS_PATH.write_text(text, encoding='utf-8')


def load_model_ctx() -> dict:
    """Return the per-model chosen context sizes ({model_id: num_ctx})."""
    try:
        data = json.loads(MODEL_CTX_PATH.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_model_ctx(model: str, num_ctx: int) -> None:
    """Remember the context ``model`` was last run at, for next selection.

    Stored globally (a fit is machine+model specific, not project specific) so
    the size is reused on the next load instead of being recomputed.
    """
    if not model or not num_ctx:
        return
    data = load_model_ctx()
    if data.get(model) == int(num_ctx):
        return
    data[model] = int(num_ctx)
    try:
        GURU_HOME.mkdir(parents=True, exist_ok=True)
        MODEL_CTX_PATH.write_text(
            json.dumps(data, indent=2), encoding='utf-8')
    except OSError:
        pass


# Create global config on import; project state stays lazy.
ensure_setup()
ALLOWED_DOMAINS = load_allowed_domains()
# Directories previously approved for this project. Nothing is allowed by
# default — the first file access (including the working directory) prompts
# once, then the approval is persisted here.
ALLOWED_READ_DIRS = load_allowed_read_dirs()
ALLOWED_WRITE_DIRS = load_allowed_write_dirs()
_apply_settings()
