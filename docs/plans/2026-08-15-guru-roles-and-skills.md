# Guru Roles & Skills Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add composable persona (role) and methodology (skill) prompt overlays to guru, selectable by the model or the user, tuned for small-context local models.

**Architecture:** A new `guru/skills.py` holds a registry loaded from `~/.guru/skills/*.md` (baked-in defaults seeded on first run). Roles and skills are pure system-prompt overlays rendered onto `messages[0]` each turn by a unified `conversation.refresh_system_context()` (which absorbs the existing `[open files]` ledger). At most one role + one skill are active per agent, tracked on `SessionState`. The model selects via `use_skill` and `spawn(role, skill, task)`; the user via `/role` and `/skill`.

**Tech Stack:** Python 3.11+, prompt_toolkit, rich; `uv` for deps; `pytest` + `flake8` (79-col, PEP8/257/484). No new third-party dependency (frontmatter parsed by hand).

**Project conventions (override the skill's defaults):**
- **Commit ONCE at the very end** (Task 8), not per task. Keep the test-first cadence within each task, but do not `git commit` until all tasks pass.
- Always run via `.venv`: `source .venv/bin/activate`.
- `flake8 guru tests` must be clean and `pytest` green before the final commit.
- No emojis in code/messages unless already present in that surface.

---

### Task 1: Skills registry module (`guru/skills.py`)

**Files:**
- Create: `guru/skills.py`
- Test: `tests/test_guru.py` (new class `TestSkillsRegistry`)

**Step 1: Write failing tests**

Add to `tests/test_guru.py` (top imports already include `config`, `session`; add `from guru import skills` at the top with the other imports):

```python
class TestSkillsRegistry:
    """Frontmatter parsing, seeding, token cap, lookup."""

    def test_parse_entry_reads_frontmatter_and_body(self) -> None:
        text = (
            "---\n"
            "name: developer\n"
            "kind: role\n"
            "description: General coding\n"
            "---\n"
            "Be a developer.\n")
        e = skills.parse_entry(text)
        assert e.name == 'developer' and e.kind == 'role'
        assert e.description == 'General coding'
        assert e.body == 'Be a developer.'

    def test_parse_entry_rejects_missing_frontmatter(self) -> None:
        assert skills.parse_entry("no frontmatter here") is None

    def test_body_is_capped(self) -> None:
        big = "x" * (skills._MAX_BODY_CHARS + 500)
        text = f"---\nname: t\nkind: skill\ndescription: d\n---\n{big}"
        e = skills.parse_entry(text)
        assert len(e.body) <= skills._MAX_BODY_CHARS + len(skills._TRUNC)
        assert e.body.endswith(skills._TRUNC)

    def test_seed_writes_missing_then_load(self, tmp_path) -> None:
        skills.seed_defaults(tmp_path, reset=False)
        reg = skills.load_registry(tmp_path)
        assert 'developer' in reg and reg['developer'].kind == 'role'
        assert 'code-review' in reg and reg['code-review'].kind == 'skill'

    def test_seed_does_not_overwrite_user_edit(self, tmp_path) -> None:
        skills.seed_defaults(tmp_path, reset=False)
        f = tmp_path / 'developer.md'
        f.write_text(f.read_text() + "\nUSER EDIT\n", encoding='utf-8')
        skills.seed_defaults(tmp_path, reset=False)          # again
        assert 'USER EDIT' in f.read_text()

    def test_reset_overwrites_defaults_not_extras(self, tmp_path) -> None:
        skills.seed_defaults(tmp_path, reset=False)
        dev = tmp_path / 'developer.md'
        dev.write_text("---\nname: developer\nkind: role\n"
                       "description: d\n---\nMANGLED\n", encoding='utf-8')
        extra = tmp_path / 'my-role.md'
        extra.write_text("---\nname: my-role\nkind: role\n"
                         "description: mine\n---\nkeep\n", encoding='utf-8')
        skills.seed_defaults(tmp_path, reset=True)
        assert 'MANGLED' not in dev.read_text()
        assert extra.read_text().endswith('keep\n')

    def test_names_by_kind(self, tmp_path) -> None:
        reg = {'a': skills.SkillEntry('a', 'role', 'd', '', 'b'),
               'c': skills.SkillEntry('c', 'skill', 'd', '', 'b')}
        assert skills.names(reg, 'role') == ['a']
        assert skills.names(reg, 'skill') == ['c']
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_guru.py::TestSkillsRegistry -q`
Expected: FAIL (`ModuleNotFoundError: guru.skills` / attributes missing).

**Step 3: Implement `guru/skills.py`**

```python
"""Roles and skills: persona (WHO) and procedure (HOW) prompt overlays.

A *role* is an agent identity (a persona/lens); a *skill* is a methodology
(a procedure). Both are small Markdown files with a simple ``key: value``
frontmatter block, stored under ``~/.guru/skills/``. They are loaded once into
a registry and rendered onto the system prompt on demand (see
``guru.domain.conversation.refresh_system_context``). In v1 they are pure
prompt overlays — they do not change tool access, which stays governed by the
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
    """Sorted-by-insertion names of a given kind in ``registry``."""
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
```

**Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_guru.py::TestSkillsRegistry -q`
Expected: PASS. Then `flake8 guru/skills.py` clean.

---

### Task 2: Config path + CLI seeding & `--reset-skills`

**Files:**
- Modify: `guru/config.py` (add `GURU_SKILLS_DIR` near the other `GURU_HOME` paths, ~line 15)
- Modify: `guru/cli.py:329-350` (`main`: add flag, call `skills.setup`)
- Test: `tests/test_guru.py::TestSkillsRegistry` already covers seeding; add a config-path assertion.

**Step 1: Add the path constant** — in `guru/config.py` after `ADAPTERS_PATH`:

```python
GURU_SKILLS_DIR = GURU_HOME / 'skills'               # roles & skills overlays
```

**Step 2: Wire CLI** — in `guru/cli.py` `main`, after the `--num-ctx` argument:

```python
    parser.add_argument(
        "--reset-skills", action="store_true",
        help="Overwrite the baked-in default roles/skills on startup",
    )
```

and after `args, _ = parser.parse_known_args()` (before `tui.run()`), add:

```python
    from guru import skills
    skills.setup(reset=args.reset_skills)
```

**Step 3: Add a smoke test** to `TestSkillsRegistry`:

```python
    def test_setup_populates_registry(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'GURU_SKILLS_DIR', tmp_path / 'skills')
        skills.setup(reset=True)
        assert 'architect' in skills.REGISTRY
        skills.REGISTRY.clear()          # leave global clean for other tests
```

**Step 4: Run** `.venv/bin/python -m pytest tests/test_guru.py::TestSkillsRegistry -q` → PASS; `flake8 guru` clean.

---

### Task 3: SessionState fields

**Files:**
- Modify: `guru/session.py:44-50` (add fields after `active_tool_names`/`file_shas`)

**Step 1: Add fields**

```python
        # Active persona (role) and methodology (skill) overlay names, or None
        # for general-purpose / no skill. Rendered onto the system prompt each
        # turn (see conversation.refresh_system_context).
        self.active_role = None
        self.active_skill = None
```

**Step 2: Verify import still loads** — `.venv/bin/python -c "import guru.session"` (no error). No dedicated test; covered by Task 4/6.

---

### Task 4: Unified `refresh_system_context` (absorbs the ledger)

**Files:**
- Modify: `guru/domain/conversation.py` — rename `refresh_file_ledger` → `refresh_system_context`, generalize; add `from guru import skills` to imports.
- Modify: existing ledger tests in `tests/test_guru.py::TestFileShaLedger` — rename the 5 `conversation.refresh_file_ledger()` calls to `conversation.refresh_system_context()`.
- Test: new class `TestSystemContext`.

**Step 1: Write failing tests** (`tests/test_guru.py`):

```python
class TestSystemContext:
    """Catalog + role + skill overlays rendered onto messages[0]."""

    def _reg(self):
        return {
            'developer': skills.SkillEntry(
                'developer', 'role', 'dev', '', 'BE A DEV'),
            'code-review': skills.SkillEntry(
                'code-review', 'skill', 'review', '', 'REVIEW METHOD'),
        }

    def test_catalog_rendered_when_registry_present(
            self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(session, 'active_role', None)
        monkeypatch.setattr(session, 'active_skill', None)
        monkeypatch.setattr(
            session, 'messages', [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        body = session.messages[0]['content']
        assert body.startswith('BASE')
        assert 'Available specialists' in body and 'developer' in body

    def test_role_and_skill_bodies_rendered(self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(session, 'active_role', 'developer')
        monkeypatch.setattr(session, 'active_skill', 'code-review')
        monkeypatch.setattr(
            session, 'messages', [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        body = session.messages[0]['content']
        assert '[role: developer]' in body and 'BE A DEV' in body
        assert '[skill: code-review]' in body and 'REVIEW METHOD' in body

    def test_idempotent_across_calls(self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(session, 'active_role', 'developer')
        monkeypatch.setattr(session, 'active_skill', None)
        monkeypatch.setattr(
            session, 'messages', [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        conversation.refresh_system_context()
        body = session.messages[0]['content']
        assert body.count('[role: developer]') == 1
        assert body.startswith('BASE')

    def test_unknown_active_name_ignored(self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(session, 'active_role', 'nonexistent')
        monkeypatch.setattr(session, 'active_skill', None)
        monkeypatch.setattr(
            session, 'messages', [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()      # must not raise
        assert '[role:' not in session.messages[0]['content']
```

**Step 2: Run** → FAIL (`refresh_system_context` missing).

**Step 3: Implement** — in `guru/domain/conversation.py` add `from guru import skills` to the top imports, then replace the `_LEDGER_SEP` constant + `refresh_file_ledger` function with:

```python
# Boundary between the base system prompt and the per-turn dynamic context
# (catalog, active role/skill overlays, open-files ledger) re-rendered on it.
_DYN_SEP = "\n\n--- active context ---\n"


def _ledger_block() -> str:
    """The '[open files]' sha list, or '' when nothing is tracked."""
    ledger = session.file_shas
    if not ledger:
        return ''
    cwd = Path.cwd()
    lines = ["[open files]"]
    for key, sha in ledger.items():
        p = Path(key)
        try:
            shown = p.relative_to(cwd)
        except ValueError:
            shown = p
        lines.append(f"- {shown} (sha:{sha})")
    lines.append(
        "Reuse a sha for edit_file instead of re-reading; if edit_file"
        " reports a mismatch, that file changed — re-read it.")
    return "\n".join(lines)


def refresh_system_context() -> None:
    """Rebuild the dynamic tail of the system prompt (messages[0]) in place.

    Renders, in order, the roles/skills catalog, the active role overlay, the
    active skill overlay, and the open-files sha ledger — each a single copy
    that survives pruning/compaction (message 0 is always kept) and is counted
    in the 'sys' context bucket. Idempotent: the previous tail is stripped
    first, so calling it every turn never doubles it.
    """
    msgs = session.messages
    if not msgs or not isinstance(msgs[0], dict) \
            or msgs[0].get('role') != 'system':
        return
    base = (msgs[0].get('content') or '').split(_DYN_SEP)[0]

    sections = []
    catalog = skills.catalog_block()
    if catalog:
        sections.append(catalog)
    role = skills.get(session.active_role) if session.active_role else None
    if role is not None and role.kind == skills.ROLE:
        sections.append(f"[role: {role.name}]\n{role.body}")
    skill = skills.get(session.active_skill) if session.active_skill else None
    if skill is not None and skill.kind == skills.SKILL:
        sections.append(f"[skill: {skill.name}]\n{skill.body}")
    ledger = _ledger_block()
    if ledger:
        sections.append(ledger)

    if sections:
        msgs[0]['content'] = base + _DYN_SEP + "\n\n".join(sections)
    else:
        msgs[0]['content'] = base
```

**Step 4: Update the existing ledger tests** — in `tests/test_guru.py::TestFileShaLedger`, rename every `conversation.refresh_file_ledger()` to `conversation.refresh_system_context()`. (The `[open files]` assertions still hold — the ledger block keeps that header.) The `test_refresh_noop_without_system_message` test stays valid.

**Step 5: Run** `.venv/bin/python -m pytest tests/test_guru.py::TestSystemContext tests/test_guru.py::TestFileShaLedger -q` → PASS. `flake8 guru` clean.

---

### Task 5: `use_skill` tool + `spawn(role, skill)`

**Files:**
- Modify: `guru/domain/tools.py` — add `use_skill` fn + `_USE_SKILL_SPEC`, register in `specs_for`/`reset_active_tools`, handle in `execute_tool`; widen `spawn` + `_SPAWN_SPEC`.
- Test: new class `TestSkillTools`.

**Step 1: Write failing tests** (`tests/test_guru.py`):

```python
class TestSkillTools:
    def _reg(self):
        return {
            'developer': skills.SkillEntry(
                'developer', 'role', 'dev', '', 'BE A DEV'),
            'code-review': skills.SkillEntry(
                'code-review', 'skill', 'review', '', 'REVIEW'),
        }

    def test_use_skill_sets_active_skill(self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'active_skill', None)
        out = tools.use_skill('code-review')
        assert session.active_skill == 'code-review' and 'code-review' in out

    def test_use_skill_rejects_role_or_unknown(self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'active_skill', None)
        assert 'No skill' in tools.use_skill('developer')   # wrong kind
        assert 'No skill' in tools.use_skill('nope')
        assert session.active_skill is None

    def test_spawn_passes_role_and_skill(self, monkeypatch) -> None:
        seen = {}

        def handler(task, role, skill):
            seen.update(task=task, role=role, skill=skill)
            return "ok"
        tools.set_spawn_handler(handler)
        try:
            tools.spawn('do it', role='developer', skill='code-review')
            assert seen == {'task': 'do it', 'role': 'developer',
                            'skill': 'code-review'}
        finally:
            tools.set_spawn_handler(None)

    def test_spawn_defaults_role_skill_empty(self, monkeypatch) -> None:
        seen = {}
        tools.set_spawn_handler(
            lambda task, role, skill: seen.update(
                role=role, skill=skill) or "ok")
        try:
            tools.spawn('t')
            assert seen == {'role': '', 'skill': ''}
        finally:
            tools.set_spawn_handler(None)
```

**Step 2: Run** → FAIL.

**Step 3: Implement** in `guru/domain/tools.py`:

- Add near the top imports: `from guru import skills` (alongside `from guru import config, session, ui` — check the actual import line and extend it).

- Widen `spawn` (replace lines 44-59):

```python
def spawn(task: str, role: str = '', skill: str = '') -> str:
    """
    Delegate a self-contained task to a new sub-agent that runs in parallel.

    Optionally give it a ``role`` (persona) and/or ``skill`` (method) from the
    catalog — e.g. role='security-engineer', skill='code-review'. The
    sub-agent works in its own viewport and context and returns only its
    conclusion; it cannot spawn further agents.
    """
    if _spawn_handler is None:
        return (
            "Spawning sub-agents is not available in this mode."
            " Handle this task yourself instead.")
    return _spawn_handler(task, role, skill)
```

- Extend `_SPAWN_SPEC` parameters + `optional`:

```python
_SPAWN_SPEC = {
    'name': 'spawn',
    'description': (
        'Delegate a self-contained task to a new sub-agent that runs in'
        ' parallel in its own viewport and context. Optionally set role'
        ' (persona) and skill (method) from the catalog. The sub-agent cannot'
        ' spawn further agents. Returns immediately — its result is delivered'
        ' back to you automatically when it finishes.'),
    'parameters': {
        'task': 'A clear, self-contained instruction for the sub-agent',
        'role': 'Optional persona name from the catalog (or empty)',
        'skill': 'Optional method name from the catalog (or empty)',
    },
    'optional': ['role', 'skill'],
}
```

- Add the `use_skill` tool + spec (place after the `_SPAWN_SPEC`/`check` block):

```python
def use_skill(name: str) -> str:
    """
    Adopt a methodology (skill) from the catalog for the current task, e.g.
    'code-review' or 'systematic-debugging'. It stays active until you switch
    skills. Roles (personas) are set with spawn(role=...) or the user's /role.
    """
    entry = skills.get(name)
    if entry is None or entry.kind != skills.SKILL:
        avail = ', '.join(skills.names(skills.REGISTRY, skills.SKILL)) or 'none'
        return f"No skill '{name}'. Available skills: {avail}."
    session.active_skill = name
    return f"Skill '{name}' is now active for this agent."


_USE_SKILL_SPEC = {
    'name': 'use_skill',
    'description': (
        'Adopt a methodology (skill) from the catalog for the current task'
        ' (e.g. code-review, systematic-debugging). Stays active until'
        ' switched. Use the catalog names shown in your context.'),
    'parameters': {'name': 'A skill name from the catalog'},
}
```

- In `specs_for`, make `use_skill` always available — change `specs = [_SEARCH_TOOLS_SPEC]` to:

```python
    specs = [_SEARCH_TOOLS_SPEC, _USE_SKILL_SPEC]
```

- In `reset_active_tools`, add `use_skill` to the always-on base — change `base = [search_tools]` to `base = [search_tools, use_skill]`.

- In `execute_tool`, add a branch before the `spawn` branch:

```python
    if name == "use_skill":
        return use_skill(**arguments)
```

**Step 4: Run** `.venv/bin/python -m pytest tests/test_guru.py::TestSkillTools -q` → PASS. `flake8 guru` clean.

*Note:* the `check`/`join` handler wiring is unchanged; only `spawn`'s handler signature changes (Task 6 updates `_spawn_agent` to match).

---

### Task 6: TUI wiring — `_configure`, `_spawn_agent`, `_work`, `/role`, `/skill`

**Files:**
- Modify: `guru/tui.py` — `_configure` (342), `_new_agent` (360), `_spawn_agent` (367), `_work` (248), `_handle_command` (662), `_greet` (748), and `_status_from` (138) for a small indicator.

**Step 1: Widen `_configure`** (line 342) to accept role/skill:

```python
    def _configure(agent, base, can_spawn: bool,
                   role=None, skill=None) -> None:
        st = agent.state
        st.messages = [
            {'role': 'system', 'content': config.build_system_prompt()}]
        st.active_tools = [tools.search_tools, tools.use_skill]
        if can_spawn:
            st.messages[0]['content'] += "\n\n" + config.DELEGATION_HINT
            st.active_tools.extend([tools.spawn, tools.check, tools.join])
        st.active_tool_names = set()
        st.active_role = role or None
        st.active_skill = skill or None
        st.model = base.model
        st.adapter = base.adapter
        st.num_ctx = base.num_ctx
        st.ctx_ceiling = base.ctx_ceiling
        st.model_size = base.model_size
        st.git_branch = base.git_branch
        st.can_spawn = can_spawn
        _attach_console(agent)
```

**Step 2: Thread role/skill through `_spawn_agent`** (line 367):

```python
    def _spawn_agent(task: str, role: str = '', skill: str = '') -> str:
        base = session.current()
        title = f"agent{len(manager.agents)}"
        child = Agent(id=title, title=title)
        _configure(child, base, can_spawn=False, role=role, skill=skill)
        child.task = task
        child.append(f"[{title}] spawned · task: {task}")
        child.append(f"> {task}")
        child.queue.append(task)
        ...
```

(the `_add_and_start`/return body is unchanged). `set_spawn_handler(_spawn_agent)` already exists near line 440 — its signature now matches `spawn`'s `_spawn_handler(task, role, skill)`.

**Step 3: Call the unified refresh in `_work`** (line ~248) — replace `conversation.refresh_file_ledger()` with:

```python
                conversation.refresh_system_context()
```

**Step 4: Add `/role` and `/skill` commands** in `_handle_command` (after the `/mode` block, ~line 670). Add these two helper closures near `_set_mode` and the command branches:

```python
    def _set_role(arg: str) -> None:
        arg = (arg or '').strip()
        if arg in ('', 'off', 'none', 'general-purpose'):
            main.state.active_role = None
            main.console.print("[green]Role[/green] → general-purpose")
            return
        entry = skills.get(arg)
        if entry is None or entry.kind != skills.ROLE:
            avail = ', '.join(skills.names(skills.REGISTRY, skills.ROLE))
            main.console.print(f"[red]No role '{arg}'.[/red] Roles: {avail}")
            return
        main.state.active_role = arg
        main.console.print(f"[green]Role[/green] → {arg}")

    def _set_skill(arg: str) -> None:
        arg = (arg or '').strip()
        if arg in ('', 'off', 'none'):
            main.state.active_skill = None
            main.console.print("[green]Skill[/green] → none")
            return
        entry = skills.get(arg)
        if entry is None or entry.kind != skills.SKILL:
            avail = ', '.join(skills.names(skills.REGISTRY, skills.SKILL))
            main.console.print(f"[red]No skill '{arg}'.[/red] Skills: {avail}")
            return
        main.state.active_skill = arg
        main.console.print(f"[green]Skill[/green] → {arg}")
```

and the branches inside `_handle_command`:

```python
        if text == '/role' or text.startswith('/role '):
            _set_role(text[5:].strip())
            return True
        if text == '/skill' or text.startswith('/skill '):
            _set_skill(text[6:].strip())
            return True
```

Add `from guru.domain import ... ` already imports `tools`; add `skills` to the tui imports (`from guru import config, session, ui` → also `skills`). Confirm `skills` is imported at top of `tui.py`.

**Step 5: Small status indicator** — in `_status_from` (line 138), after the `mode` line, add role/skill when set:

```python
    rs = st.active_role or 'general'
    if st.active_skill:
        rs += f"/{st.active_skill}"
    left = (f"🤖 {model} | 💪 {st.model_size or '?'} | 🔐 {mode}"
            f" | 🎭 {rs} | ")
```

(replace the existing `left = ...` line).

**Step 6: Update greet hint** (line 754) to include the new commands:

```python
        main.console.print(
            "[dim]/mode /role /skill /models /context /adapters /save"
            " /resume /compact /search[/dim]\n")
```

**Step 7: Run the full suite** `.venv/bin/python -m pytest -q` → all PASS. `flake8 guru tests` clean. Manually sanity-check import: `.venv/bin/python -c "import guru.tui"`.

---

### Task 7: Manual smoke test (no code)

Run `./start.sh` and verify:
- `/role security-engineer` then `/skill code-review` → status shows `🎭 security-engineer/code-review`.
- Ask it to review a small diff → response reflects the security lens.
- `/role off`, `/skill off` → indicator returns to `general`.
- Model self-selection: ask a debugging question → it may `use_skill('systematic-debugging')` from the catalog.
- `./start.sh` with the guru CLI `--reset-skills` (via the launcher) rewrites `~/.guru/skills/*.md` defaults; a hand-edited extra file survives.

Record any issues as follow-up; do not fix silently.

---

### Task 8: Final verification + single commit

**Step 1:** `source .venv/bin/activate && flake8 guru tests` → clean.
**Step 2:** `pytest -q` → all green (existing + new).
**Step 3:** Commit everything from this session (color fix, sha ledger, roles/skills, both design docs) in one commit:

```bash
git add -A
git commit -m "feat: add composable role/skill prompt overlays with on-demand catalog"
```

(End commit body with the project's required trailer; do not mention AI/LLM.)
**Step 4:** Remind the user to `ssh-add ~/.ssh/id_ed25519` then `git push origin main`.

---

## Notes for the executor
- **DRY:** the ledger logic now lives only in `_ledger_block`; do not leave a second copy in `refresh_file_ledger`.
- **YAGNI:** no tool-restriction, no read-only "explore", no `Ctrl+N` role picker in v1 — all deferred (see the design doc's "Open items").
- **Context cost is the whole point:** keep default bodies well under 512 tokens; the cap is a backstop, not a target.
- If `skills.REGISTRY` is empty (no files), everything degrades gracefully: no catalog, `use_skill`/`/role`/`/skill` report "none available", `spawn` ignores empty role/skill.
