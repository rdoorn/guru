"""Tests for guru.skills registry and the skill/role tools."""
from guru import config, session, skills
from guru.domain import tools


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

    def test_setup_populates_registry(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'GURU_SKILLS_DIR', tmp_path / 'skills')
        skills.setup(reset=True)
        assert 'architect' in skills.REGISTRY
        skills.REGISTRY.clear()          # leave global clean for other tests


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
