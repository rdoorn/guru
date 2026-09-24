"""Paths, constants, setup, system-prompt assembly, and the domain allow-list.

This module owns filesystem locations and pure configuration. It imports no
other guru module, so everything else can depend on it freely.
"""
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from guru import log
from guru.domain import routing as _routing

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
# Secret scanner (guru/scanners/secrets.py): project code names that must
# never reach a remote model (one literal per line), and regexes whose
# matches suppress a finding (known test fixtures, sample keys).
SENSITIVE_MARKERS_PATH = PROJECT_GURU_DIR / 'sensitive_markers.txt'
SCAN_ALLOW_PATH = PROJECT_GURU_DIR / 'scan_allow.txt'

# Access mode (session-level policy). Separate from the allow-lists: it decides
# whether we prompt, auto-approve, or refuse. read-only refuses writes; ask
# prompts per not-yet-allowed target; auto approves silently (filling the
# lists). Path resolution / escape checks apply in every mode.
MODE_READ_ONLY = 'read-only'
MODE_ASK = 'ask-for-changes'
MODE_AUTO = 'auto'
MODES = (MODE_READ_ONLY, MODE_ASK, MODE_AUTO)
MODE = MODE_ASK
# Escalation policy for auto mode: when True (the default) auto grants and
# persists any new directory/domain without asking. When False, auto mode
# still consults the asker for escalations outside the allow-lists, so a
# sandbox (the eval runner) can deny them while auto-approving everything
# inside its own allow-lists. Domain-level knob; not a settings key.
AUTO_GRANT = True
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

# Decision seam (guru/domain/decisions.py). In "shadow" mode the existing
# heuristics keep deciding; a configured judge answers the same question in
# the background and both answers are logged. In "active" mode the points
# listed under [decisions.active] take the judge's answer (bounded by
# timeout_ms, falling back to the heuristic); every other configured point
# stays shadow. settings.toml:
#   [decisions]
#   mode = "shadow"              # off | shadow | active
#   sidecar_model = "qwen3:4b"   # Ollama model for the "ollama" judge
#   sidecar_url = "http://localhost:11434"
#   timeout_ms = 1500            # active: max wait for a judge per decision
#   breaker_timeouts = 5         # active: consecutive timeouts that open the
#   breaker_cooldown_s = 60      #   per-point breaker, and for how long
#   [decisions.points]           # decision point -> judge spec
#   stall = "ollama"             # ollama | ollama:<model> | encoder |
#   panel = "encoder"            # encoder:<hf-model> | injection
#   injection = "injection"
#   [decisions.active]           # active mode: which points the judge decides
#   stall = true
#   [decisions.thresholds]       # noul: judge says yes when P(yes) >= this
#   stall = 0.6                  # (default 0.5)
DECISIONS_MODES = ('off', 'shadow', 'active')
JUDGING_MODES = ('shadow', 'active')      # modes in which judges run at all
DECISIONS_MODE = 'off'
DECISIONS_SIDECAR_MODEL = 'qwen3:4b'
DECISIONS_SIDECAR_URL = 'http://localhost:11434'
DECISIONS_POINTS: dict = {}
DECISIONS_ACTIVE: dict = {}
DECISIONS_THRESHOLDS: dict = {}
DECISIONS_TIMEOUT_MS = 1500
DECISIONS_BREAKER_TIMEOUTS = 5
DECISIONS_BREAKER_COOLDOWN_S = 60.0

# Routing (guru/domain/routing.py, guru/repositories/settings.py): the typed
# [routing] table is loaded by the CLI at startup. SECRET_SCAN mirrors its
# secret_scan flag for the tool layer, which redacts tool results bound to a
# remote adapter (guru.domain.policy) when it is on. Off until a [routing]
# table is configured (guru behaves exactly as before without one).
SECRET_SCAN = False

# Ledger (guru/domain/ledger.py): append-only JSONL streams of every model
# call, turn and sub-agent task under ~/.guru/ledger/. [ledger] enabled=false
# turns it off. [pricing."<model>"] overrides the bundled price table
# (input_per_m, output_per_m, cache_write_5m_per_m, cache_write_1h_per_m,
# cache_read_per_m; USD per million tokens).
LEDGER_DIR = GURU_HOME / 'ledger'
LEDGER_ENABLED = True
PRICING_OVERRIDES: dict = {}

# Eval suite (guru/evals): the default ``Adapter|model`` spec ('' = guru's
# default model, as the CLI would pick) and the context window the model is
# pinned to for a run (0 = the GPU auto-fit; 8192 keeps a 24 GB Mac from
# loading an 8B at 40k and crawling). settings.toml:
#   [evals]
#   model = "Ollama|qwen3:14b"
#   num_ctx = 8192
# ``python -m guru.evals run --model/--num-ctx`` override both.
EVALS_MODEL = ''
EVALS_NUM_CTX = 8192

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

# Appended instead of DELEGATION_HINT when [routing] controller = true: the
# main agent only converses and coordinates; every task runs in a routed
# sub-agent (design doc §2).
CONTROLLER_HINT = (
    "You are a CONTROLLER. You converse with the user, ask clarifying"
    " questions when the request is ambiguous, and DECOMPOSE every piece of"
    " actual work into sub-agent tasks — you never execute a task yourself."
    " Your only tools are spawn, check, join and use_skill; never call file,"
    " code or web tools (you do not have them).\n"
    "The working directory in the [project] block of the active context"
    " (name, absolute path, git branch) is the current project; the user's"
    " requests refer to it unless they say otherwise. 'This repository',"
    " 'the codebase', 'the tests', 'the README' all mean that project."
    " Never ask which repository, path or codebase is meant: delegate"
    " immediately with a self-contained task that names the project path,"
    " and let the sub-agent look around (it has the file tools you lack).\n"
    "For each task call spawn(task, kind, complexity, role, skill): write a"
    " clear, self-contained task; label kind as one of debug, build,"
    " refactor, review, explain, docs, ops, other and complexity as one of"
    " trivial, standard, hard (the labels pick the model that runs it);"
    " add the role (persona) and skill (method) from the catalog that fit."
    " Complexity: " + '; '.join(
        f'{tier} = {desc}'
        for tier, desc in _routing.COMPLEXITY_DESCRIPTIONS.items())
    + ". Use all three tiers — a task that a"
    " small model can do on trivial, one that needs care on hard.\n"
    "Spawn independent tasks in parallel, use check to poll and join to be"
    " resumed when a group finishes, then SYNTHESISE the results into one"
    " answer for the user. Reply directly, briefly, for greetings, questions"
    " about yourself, or clarifications that need no work."
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
# (>= this many DISTINCT paths read with the read tools, and a request
# that is not a single-file edit) having spawned no sub-agent, nudge it
# once to decompose into a parallel domain panel. Never for a controller.
# Set 0 to disable the nudge.
DELEGATION_NUDGE_MIN_READS = 3
DELEGATION_READ_TOOLS = {'read_file', 'search_code', 'list_dir', 'list_tree'}


def review_tasks(area: str = 'the repository') -> list:
    """The (task, role, skill) list for the /review panel, one per REVIEW_PANEL
    member. guru spawns these directly (see orchestrator.spawn_panel), so the
    multi-agent path runs deterministically rather than depending on the model
    choosing to delegate."""
    return [
        (f"Review {area} for {focus}. Give concrete findings with file:line"
         f" and a suggested fix; be specific.", role, skill)
        for role, skill, focus in REVIEW_PANEL]


def review_synthesis(area: str = 'the repository') -> str:
    """The synthesis lead-in delivered to the parent once the panel joins."""
    return (
        f"Below are independent sub-agent reviews of {area}. Synthesise them"
        " into ONE consolidated report grouped by severity (blocker, major,"
        " minor, nit), de-duplicating overlaps and keeping file:line refs.")


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
    """Apply settings.toml overrides (retention, tools, sampling, bench,
    decisions, ledger, pricing, evals)."""
    global WEB_SUMMARIZE_OVER_CHARS, OUTLINE_FILE_OVER_CHARS
    global EVALS_MODEL, EVALS_NUM_CTX
    global PREACTIVATE_TOOLS, SAMPLING, SAMPLING_PER_MODEL
    global BENCH_MODEL_TIMEOUT, FLAT_TOOLS
    global DECISIONS_MODE, DECISIONS_SIDECAR_MODEL, DECISIONS_SIDECAR_URL
    global DECISIONS_POINTS, DECISIONS_ACTIVE, DECISIONS_THRESHOLDS
    global DECISIONS_TIMEOUT_MS, DECISIONS_BREAKER_TIMEOUTS
    global DECISIONS_BREAKER_COOLDOWN_S, LEDGER_ENABLED, PRICING_OVERRIDES
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
    dec = settings_section('decisions')
    mode = str(dec.get('mode', DECISIONS_MODE))
    if mode not in DECISIONS_MODES:
        log.info('ignoring unknown [decisions] mode %r; expected one of %s',
                 mode, ', '.join(DECISIONS_MODES))
    DECISIONS_MODE = mode if mode in DECISIONS_MODES else 'off'
    DECISIONS_SIDECAR_MODEL = str(
        dec.get('sidecar_model', DECISIONS_SIDECAR_MODEL))
    DECISIONS_SIDECAR_URL = str(dec.get('sidecar_url', DECISIONS_SIDECAR_URL))
    points = dec.get('points')
    DECISIONS_POINTS = ({str(k): str(v) for k, v in points.items()}
                        if isinstance(points, dict) else {})
    active = dec.get('active')
    DECISIONS_ACTIVE = ({str(k): v for k, v in active.items()
                         if isinstance(v, bool)}
                        if isinstance(active, dict) else {})
    thresholds = dec.get('thresholds')
    DECISIONS_THRESHOLDS = (
        {str(k): float(v) for k, v in thresholds.items()
         if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if isinstance(thresholds, dict) else {})
    try:
        DECISIONS_TIMEOUT_MS = int(dec.get('timeout_ms', DECISIONS_TIMEOUT_MS))
    except (TypeError, ValueError):
        pass
    try:
        DECISIONS_BREAKER_TIMEOUTS = int(
            dec.get('breaker_timeouts', DECISIONS_BREAKER_TIMEOUTS))
        DECISIONS_BREAKER_COOLDOWN_S = float(
            dec.get('breaker_cooldown_s', DECISIONS_BREAKER_COOLDOWN_S))
    except (TypeError, ValueError):
        pass
    ev = settings_section('evals')
    model = ev.get('model', EVALS_MODEL)
    if isinstance(model, str):
        EVALS_MODEL = model.strip()
    num_ctx = ev.get('num_ctx', EVALS_NUM_CTX)
    if isinstance(num_ctx, int) and not isinstance(num_ctx, bool) \
            and num_ctx >= 0:
        EVALS_NUM_CTX = num_ctx
    LEDGER_ENABLED = bool(settings_section('ledger').get('enabled', True))
    PRICING_OVERRIDES = {
        str(k): {str(f): float(v) for f, v in tbl.items()
                 if isinstance(v, (int, float))}
        for k, tbl in settings_section('pricing').items()
        if isinstance(tbl, dict)}


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
