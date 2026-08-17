"""Tests for guru.config settings, prompts, and persistence."""
from pathlib import Path

from guru import config


class TestDomainOf:
    """Tests for config.domain_of."""

    def test_strips_scheme_and_lowercases(self) -> None:
        assert config.domain_of('https://Example.COM/path') == 'example.com'

    def test_strips_port(self) -> None:
        assert config.domain_of('http://example.com:8443/x') == 'example.com'

    def test_bare_host_without_scheme(self) -> None:
        assert config.domain_of('example.com/path') == 'example.com'

    def test_subdomain_preserved(self) -> None:
        assert config.domain_of('https://api.github.com') == 'api.github.com'


class TestProjectMemoryDir:
    """The project memory dir lives under ./.guru/memory."""

    def test_points_into_project_guru_dir(self) -> None:
        assert config.PROJECT_MEMORY_DIR == Path.cwd() / '.guru' / 'memory'


class TestBuildSystemPrompt:
    """Tests for config.build_system_prompt assembly."""

    def test_appends_global_and_local(self, tmp_path, monkeypatch) -> None:
        global_md = tmp_path / 'GURU.md'
        local_md = tmp_path / '.GURU.md'
        global_md.write_text('GLOBAL RULES', encoding='utf-8')
        local_md.write_text('LOCAL RULES', encoding='utf-8')
        monkeypatch.setattr(config, 'GURU_MD_PATH', global_md)
        monkeypatch.setattr(config, 'PROJECT_GURU_MD', local_md)

        prompt = config.build_system_prompt()

        assert config.SYSTEM_PROMPT.strip() in prompt
        assert 'GLOBAL RULES' in prompt
        assert 'LOCAL RULES' in prompt
        assert prompt.index('GLOBAL RULES') < prompt.index('LOCAL RULES')

    def test_missing_files_are_skipped(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'GURU_MD_PATH', tmp_path / 'nope.md')
        monkeypatch.setattr(config, 'PROJECT_GURU_MD', tmp_path / 'nope2.md')
        assert config.build_system_prompt() == config.SYSTEM_PROMPT.strip()


class TestReviewPanel:
    """The /review command's deterministic panel (config helpers)."""

    def test_review_tasks_one_per_panel_member(self) -> None:
        tasks = config.review_tasks('the repo')
        assert len(tasks) == len(config.REVIEW_PANEL)
        for (task, role, skill), (prole, pskill, _focus) in zip(
                tasks, config.REVIEW_PANEL):
            assert role == prole and skill == pskill
            assert 'the repo' in task and 'file:line' in task

    def test_review_synthesis_mentions_area(self) -> None:
        s = config.review_synthesis('the repo')
        assert 'the repo' in s and 'consolidate' in s.lower()


class TestModelCtxStore:
    """Per-model context persistence (~/.guru/model_ctx.json)."""

    def _isolate(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'GURU_HOME', tmp_path)
        monkeypatch.setattr(
            config, 'MODEL_CTX_PATH', tmp_path / 'model_ctx.json')

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch) -> None:
        self._isolate(tmp_path, monkeypatch)
        config.save_model_ctx('qwen3:14b', 32768)
        config.save_model_ctx('devstral', 65536)
        assert config.load_model_ctx() == {
            'qwen3:14b': 32768, 'devstral': 65536}

    def test_save_ignores_empty(self, tmp_path, monkeypatch) -> None:
        self._isolate(tmp_path, monkeypatch)
        config.save_model_ctx('', 100)
        config.save_model_ctx('m', 0)
        assert config.load_model_ctx() == {}

    def test_load_missing_returns_empty(self, tmp_path, monkeypatch) -> None:
        self._isolate(tmp_path, monkeypatch)
        assert config.load_model_ctx() == {}


class TestSettings:
    """Tests for config.load_settings / save_settings."""

    def test_round_trip(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / '.guru' / 'settings.json'
        monkeypatch.setattr(config, 'PROJECT_SETTINGS_PATH', path)
        assert config.load_settings() == {}
        config.save_settings({'adapter': 'Claude Code', 'model': 'x'})
        loaded = config.load_settings()
        assert loaded['adapter'] == 'Claude Code'
        assert loaded['model'] == 'x'


class TestContextSettings:
    """Global ~/.guru/settings.toml overrides context thresholds."""

    def test_load_missing_returns_empty(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            config, 'GLOBAL_SETTINGS_PATH', tmp_path / 'settings.toml')
        assert config.load_context_settings() == {}

    def test_load_reads_context_section(
            self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'settings.toml'
        p.write_text(
            "[context]\nweb_summarize_over_chars = 1234\n", encoding='utf-8')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', p)
        assert config.load_context_settings() == {
            'web_summarize_over_chars': 1234}

    def test_load_invalid_returns_empty(
            self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'settings.toml'
        p.write_text("not = valid = toml", encoding='utf-8')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', p)
        assert config.load_context_settings() == {}

    def test_apply_overrides_defaults(
            self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'settings.toml'
        p.write_text(
            "[context]\nweb_summarize_over_chars = 999\n"
            "outline_file_over_chars = 111\n", encoding='utf-8')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', p)
        monkeypatch.setattr(config, 'WEB_SUMMARIZE_OVER_CHARS', 6000)
        monkeypatch.setattr(config, 'OUTLINE_FILE_OVER_CHARS', 8000)
        config._apply_settings()
        assert config.WEB_SUMMARIZE_OVER_CHARS == 999
        assert config.OUTLINE_FILE_OVER_CHARS == 111


class TestToolsAndSamplingSettings:
    """settings.toml [tools] preactivate + [sampling] global/per-model."""

    def test_apply_reads_preactivate_and_sampling(
            self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'settings.toml'
        p.write_text(
            '[tools]\npreactivate = ["read_file", "search_code"]\n\n'
            '[sampling]\ntemperature = 0.7\n\n'
            '[sampling."batiai/qwen3.6-27b:q3"]\n'
            'temperature = 0.6\ntop_p = 0.95\n', encoding='utf-8')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', p)
        monkeypatch.setattr(config, 'PREACTIVATE_TOOLS', ['x'])
        monkeypatch.setattr(config, 'SAMPLING', {})
        monkeypatch.setattr(config, 'SAMPLING_PER_MODEL', {})
        config._apply_settings()
        assert config.PREACTIVATE_TOOLS == ['read_file', 'search_code']
        assert config.SAMPLING == {'temperature': 0.7}
        assert config.SAMPLING_PER_MODEL == {
            'batiai/qwen3.6-27b:q3': {'temperature': 0.6, 'top_p': 0.95}}

    def test_apply_reads_flat_tools(self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'settings.toml'
        p.write_text('[tools]\nflat = true\n', encoding='utf-8')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', p)
        monkeypatch.setattr(config, 'FLAT_TOOLS', False)
        config._apply_settings()
        assert config.FLAT_TOOLS is True
