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

DEFAULT_GURU_MD = """# GURU.md

Instructions for the guru assistant. Edit this file to change guru's
behaviour globally. Add a `.guru/GURU.md` inside a project to append
project-specific instructions.

## Persona

- Be concise and direct.
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
# name = "Claude Code"
# type = "anthropic"
# auth = "claude_code"   # reuse Claude Code's Keychain OAuth token (macOS)
"""

SYSTEM_PROMPT = """
You are a helpful assistant with access to a tool directory.

You start each turn with one tool: search_tools.
When you need a capability, call search_tools with a short phrase describing
the ACTION you want to perform — not the user's question.

Correct usage:
  User asks "what is the weather in Amsterdam?"
    → search_tools("search the web current data")

  User asks "what is the latest kubernetes version?"
    → search_tools("get latest github release version")

  User asks "read this URL for me: https://..."
    → search_tools("fetch webpage url")

  User asks "query the /metrics endpoint on localhost"
    → search_tools("fetch url endpoint http")

After search_tools returns matching tools, call them directly by name.
Do not try to call a tool that has not been returned by search_tools first.

When the user asks you to DO something (fetch a URL, search for something,
query an endpoint), use the tool — do not describe how the user could do it
themselves.

Guidelines:
- Never use tools for math, logic, coding, or stable facts from training data.
- If the first search returns poor results, refine the query and search again.
- Always cite sources. Report only what tool results explicitly state.
- If a question requires a location or name not provided, ask first.
"""

# Domains approved for model-initiated web access, loaded at startup.
ALLOWED_DOMAINS: set = set()


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
