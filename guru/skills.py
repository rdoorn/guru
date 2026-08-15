"""Roles and skills: persona (WHO) and procedure (HOW) prompt overlays.

A *role* is an agent identity (a persona/lens); a *skill* is a methodology
(a procedure). Both are small Markdown files with a simple ``key: value``
frontmatter block, stored under ``~/.guru/skills/``. They are loaded once into
a registry and rendered onto the system prompt on demand (see
``guru.domain.conversation.refresh_system_context``). In v1 they are pure
prompt overlays -- they do not change tool access, which stays governed by the
read-only / ask / auto access mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Body budget: ~512 tokens at ~4 chars/token. Longer bodies are truncated so a
# user's over-long edit cannot blow the (small) context window.
_MAX_BODY_TOKENS = 512
_MAX_BODY_CHARS = _MAX_BODY_TOKENS * 4
_TRUNC = "\n…[truncated]"

ROLE = 'role'
SKILL = 'skill'
_KINDS = (ROLE, SKILL)


@dataclass
class SkillEntry:
    name: str
    kind: str
    description: str
    model: str
    body: str


# Baked-in defaults: name -> (kind, description, model, body). Seeded to disk
# on first run; refined copies ship via ``--reset-skills``.
DEFAULTS: dict = {
    'developer': (
        ROLE,
        'General coding with an eye for correctness, readability, and tests',
        '',
        "You are a senior software developer. Prioritize correctness,"
        " readability, and maintainability. Match the surrounding code's"
        " style. Prefer small, well-named functions and clear data flow."
        " Handle edge cases and error paths, and add tests for new behavior."
        " Call out risky assumptions. Avoid over-engineering (YAGNI) and"
        " premature abstraction."),
    'architect': (
        ROLE,
        'System design: components, boundaries, trade-offs before code',
        '',
        "You are a software architect. Think in components, boundaries, and"
        " data flow before code. Weigh trade-offs (coupling, cohesion,"
        " scalability, operational cost) and state them explicitly. Prefer"
        " simple, evolvable designs over clever ones. Identify where the"
        " design will strain under load or change. Recommend one option and"
        " say why."),
    'security-engineer': (
        ROLE,
        'Security lens: hostile inputs, authz, secrets, vulnerabilities',
        '',
        "You are a security engineer. Assume inputs are hostile. Focus on"
        " injection, authentication/authorization, secrets handling, unsafe"
        " deserialization, SSRF, path traversal, and vulnerable"
        " dependencies. For each finding state the impact and a concrete"
        " fix. Prefer secure defaults and least privilege. Flag anything"
        " that widens the attack surface."),
    'SRE': (
        ROLE,
        'Reliability & operability: failure modes, observability, rollback',
        '',
        "You are a site reliability engineer. Optimize for operability:"
        " observability (logs, metrics, traces), failure modes, blast"
        " radius, graceful degradation, and safe rollback. Ask how this"
        " behaves under partial failure and load. Prefer boring,"
        " well-instrumented solutions. Call out alerting and on-call"
        " implications."),
    'code-review': (
        SKILL,
        'Systematic review; findings by severity with concrete fixes',
        '',
        "Apply a systematic code review. Read the whole diff before"
        " commenting. Check: correctness and edge cases; error handling;"
        " tests (present and meaningful); readability and naming; security;"
        " performance hot spots; and consistency with the surrounding code."
        " Report findings grouped by severity (blocker / major / minor /"
        " nit) with file:line and a concrete suggested fix. Do not restate"
        " unchanged code."),
    'brainstorming': (
        SKILL,
        'Turn an idea into an approved design via Q&A before coding',
        '',
        "Turn the idea into a design through dialogue before any code. Ask"
        " questions one at a time; prefer multiple-choice. Understand"
        " purpose, constraints, and success criteria. Propose 2-3 approaches"
        " with trade-offs and a recommendation. Apply YAGNI ruthlessly."
        " Present the design in small sections and get agreement on each"
        " before moving on. Do not implement until the design is approved."),
    'systematic-debugging': (
        SKILL,
        'Root-cause-first debugging: reproduce, hypothesize, verify',
        '',
        "Find the root cause before proposing any fix. Reproduce reliably"
        " and read the exact error. Trace the bad value backward to its"
        " source; instrument component boundaries if the path is unclear."
        " Form one hypothesis, test it with the smallest possible change,"
        " and verify. Write a failing test that captures the bug before"
        " fixing. If three fixes fail, question the architecture rather than"
        " trying a fourth."),
    'test-driven-development': (
        SKILL,
        'Test-first: failing test, minimal code, refactor',
        '',
        "Work test-first. Write the smallest failing test for the next"
        " behavior; run it and confirm it fails for the right reason. Write"
        " the minimal code to pass; run and confirm green. Refactor with"
        " tests green. One behavior per test, clear names, assert on"
        " observable outcomes not implementation. Never write production"
        " code without a failing test driving it."),
}


def _cap(body: str) -> str:
    if len(body) > _MAX_BODY_CHARS:
        return body[:_MAX_BODY_CHARS] + _TRUNC
    return body


def parse_entry(text: str) -> "SkillEntry | None":
    """Parse a '---'-delimited frontmatter file into a SkillEntry, or None."""
    if not text.startswith('---'):
        return None
    parts = text.split('---', 2)
    if len(parts) < 3:
        return None
    meta: dict = {}
    for line in parts[1].strip().splitlines():
        if ':' in line:
            key, _, val = line.partition(':')
            meta[key.strip()] = val.strip()
    name = meta.get('name', '').strip()
    kind = meta.get('kind', '').strip()
    if not name or kind not in _KINDS:
        return None
    return SkillEntry(
        name=name, kind=kind,
        description=meta.get('description', '').strip(),
        model=meta.get('model', '').strip(),
        body=_cap(parts[2].strip()))


def _render_default(name: str) -> str:
    kind, desc, model, body = DEFAULTS[name]
    return (f"---\nname: {name}\nkind: {kind}\n"
            f"description: {desc}\nmodel: {model}\n---\n{body}\n")


def seed_defaults(directory: Path, reset: bool = False) -> None:
    """Write baked-in defaults into ``directory``.

    Missing files are always written. With ``reset`` True, existing default
    files are overwritten too (user-authored extra files are never touched).
    """
    directory.mkdir(parents=True, exist_ok=True)
    for name in DEFAULTS:
        path = directory / f"{name}.md"
        if reset or not path.exists():
            path.write_text(_render_default(name), encoding='utf-8')


def load_registry(directory: Path) -> dict:
    """Parse every ``*.md`` under ``directory`` into ``{name: SkillEntry}``."""
    registry: dict = {}
    if not directory.exists():
        return registry
    for path in sorted(directory.glob('*.md')):
        try:
            entry = parse_entry(path.read_text(encoding='utf-8'))
        except OSError:
            continue
        if entry is not None:
            registry[entry.name] = entry
    return registry


def names(registry: dict, kind: str) -> list:
    """Names of a given kind in ``registry`` (insertion order)."""
    return [n for n, e in registry.items() if e.kind == kind]


# Process-wide registry, populated by setup() at startup.
REGISTRY: dict = {}


def setup(reset: bool = False) -> None:
    """Seed defaults, then load the registry into the module global."""
    from guru import config
    seed_defaults(config.GURU_SKILLS_DIR, reset=reset)
    REGISTRY.clear()
    REGISTRY.update(load_registry(config.GURU_SKILLS_DIR))


def get(name: str) -> "SkillEntry | None":
    return REGISTRY.get(name)


def catalog_block() -> str:
    """One-line-per-entry catalog for the system prompt, or '' if empty."""
    if not REGISTRY:
        return ''
    roles = [REGISTRY[n] for n in names(REGISTRY, ROLE)]
    sk = [REGISTRY[n] for n in names(REGISTRY, SKILL)]
    lines = ["Available specialists — set a role (persona) and/or a skill"
             " (method) with /role, /skill, use_skill, or spawn(role, skill):"]
    if roles:
        lines.append("Roles: general-purpose (default)")
        for e in roles:
            lines.append(f"  - {e.name} — {e.description}")
    if sk:
        lines.append("Skills:")
        for e in sk:
            lines.append(f"  - {e.name} — {e.description}")
    return "\n".join(lines)
