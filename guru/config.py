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

PROJECT_GURU_DIR = Path.cwd() / '.guru'
PROJECT_GURU_MD = PROJECT_GURU_DIR / 'GURU.md'       # appended to the global
DOMAINS_ALLOW_PATH = PROJECT_GURU_DIR / 'domains_allow.txt'  # per-project
DIRS_ALLOW_PATH = PROJECT_GURU_DIR / 'dirs_allow.txt'  # file-tool allow-list
PROJECT_MEMORY_DIR = PROJECT_GURU_DIR / 'memory'     # saved conversations
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
  latest kubernetes version → "get latest github release"
  fetch this URL / query an endpoint → "fetch webpage url"

Do not use tools for math, logic, coding, or stable facts from your training.
If results are weak, refine and search again. Cite sources and state only what
the results show. If a needed detail (a name, a location) is missing, ask.
Before concluding code or a feature is missing, grep for its definition and
read the file that defines it; when reviewing a file, follow its local
imports. Never infer that something is absent from a single file.
"""

# Appended to the system prompt of delegation-capable agents (TUI only), to
# steer heavy tool output out of the main context and into sub-agents.
DELEGATION_HINT = (
    "You can run work in parallel by delegating to sub-agents with the spawn"
    " tool. Prefer this for any subtask that produces large tool output you"
    " do not need in full — fetching web pages, reading big files, or broad"
    " multi-step research. The sub-agent reads the bulk in its own context"
    " and returns only its conclusion to you, keeping your own context small."
    " Use check to poll a sub-agent and join to be resumed once a group"
    " finishes."
)

# Domains approved for model-initiated web access, loaded at startup.
ALLOWED_DOMAINS: set = set()
# Directories approved for model-initiated file access (resolved absolute
# paths). Loaded from the per-project allow-list; new ones (including the
# working directory) are approved once and persisted. See load below.
ALLOWED_DIRS: set = set()


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


def load_allowed_dirs() -> set:
    """Read the file-access allow-list into a set of resolved path strings."""
    try:
        lines = DIRS_ALLOW_PATH.read_text(encoding='utf-8').splitlines()
    except OSError:
        return set()
    return {ln.strip() for ln in lines if ln.strip()}


def persist_dir(directory: str) -> None:
    """Append a newly approved directory to the project allow-list file."""
    DIRS_ALLOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DIRS_ALLOW_PATH.open('a', encoding='utf-8') as fh:
        fh.write(directory + '\n')


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


# Create global config on import; project state stays lazy.
ensure_setup()
ALLOWED_DOMAINS = load_allowed_domains()
# Directories previously approved for this project. Nothing is allowed by
# default — the first file access (including the working directory) prompts
# once, then the approval is persisted here.
ALLOWED_DIRS = load_allowed_dirs()
