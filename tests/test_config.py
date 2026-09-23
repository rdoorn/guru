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


class TestControllerHint:
    """CONTROLLER_HINT tells the controller which project requests refer
    to and how to label complexity (triage 2026-09-23-claude-tiers)."""

    def test_never_asks_which_repository(self) -> None:
        hint = config.CONTROLLER_HINT
        assert 'Never ask which repository' in hint
        assert '[project]' in hint
        assert 'refer to it' in hint

    def test_task_names_the_project_path(self) -> None:
        assert 'project path' in config.CONTROLLER_HINT

    def test_complexity_examples(self) -> None:
        hint = config.CONTROLLER_HINT
        for word in ('trivial', 'standard', 'hard', 'greetings',
                     'one-file edit', 'multi-file refactor', 'concurrency',
                     'whole codebase'):
            assert word in hint, word


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


class TestDecisionsAndLedgerSettings:
    """settings.toml [decisions], [ledger], [pricing] tables."""

    def _apply(self, tmp_path, monkeypatch, text):
        p = tmp_path / 'settings.toml'
        p.write_text(text, encoding='utf-8')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', p)
        for name, default in (('DECISIONS_MODE', 'off'),
                              ('DECISIONS_POINTS', {}),
                              ('DECISIONS_ACTIVE', {}),
                              ('DECISIONS_THRESHOLDS', {}),
                              ('DECISIONS_TIMEOUT_MS', 1500),
                              ('DECISIONS_SIDECAR_MODEL', 'qwen3:4b'),
                              ('LEDGER_ENABLED', True),
                              ('PRICING_OVERRIDES', {})):
            monkeypatch.setattr(config, name, default)
        config._apply_settings()

    def test_unknown_mode_is_ignored_and_logged(
            self, tmp_path, monkeypatch, caplog) -> None:
        with caplog.at_level('INFO', logger='guru'):
            self._apply(tmp_path, monkeypatch,
                        '[decisions]\nmode = "autopilot"\n')
        assert config.DECISIONS_MODE == 'off'
        assert any('autopilot' in r.getMessage() for r in caplog.records)

    def test_defaults(self) -> None:
        assert config.DECISIONS_MODE == 'off'
        assert config.DECISIONS_POINTS == {}
        assert config.LEDGER_ENABLED is True
        assert config.LEDGER_DIR == config.GURU_HOME / 'ledger'
        assert config.PRICING_OVERRIDES == {}

    def test_reads_all_three_tables(self, tmp_path, monkeypatch) -> None:
        self._apply(tmp_path, monkeypatch, (
            '[decisions]\nmode = "shadow"\nsidecar_model = "qwen3:1.7b"\n'
            '[decisions.points]\nstall = "ollama"\npanel = "encoder"\n'
            '[ledger]\nenabled = false\n'
            '[pricing."claude-sonnet-5"]\ninput_per_m = 2.5\n'
            'output_per_m = 11.0\n'))
        assert config.DECISIONS_MODE == 'shadow'
        assert config.DECISIONS_SIDECAR_MODEL == 'qwen3:1.7b'
        assert config.DECISIONS_POINTS == {'stall': 'ollama',
                                           'panel': 'encoder'}
        assert config.LEDGER_ENABLED is False
        assert config.PRICING_OVERRIDES == {
            'claude-sonnet-5': {'input_per_m': 2.5, 'output_per_m': 11.0}}

    def test_unknown_mode_falls_back_to_off(self, tmp_path, monkeypatch):
        self._apply(tmp_path, monkeypatch, '[decisions]\nmode = "yolo"\n')
        assert config.DECISIONS_MODE == 'off'

    def test_active_defaults(self) -> None:
        assert config.DECISIONS_MODES == ('off', 'shadow', 'active')
        assert config.JUDGING_MODES == ('shadow', 'active')
        assert config.DECISIONS_ACTIVE == {}
        assert config.DECISIONS_THRESHOLDS == {}
        assert config.DECISIONS_TIMEOUT_MS == 1500
        assert config.DECISIONS_BREAKER_TIMEOUTS == 5
        assert config.DECISIONS_BREAKER_COOLDOWN_S == 60.0

    def test_reads_breaker_settings(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'DECISIONS_BREAKER_TIMEOUTS', 5)
        monkeypatch.setattr(config, 'DECISIONS_BREAKER_COOLDOWN_S', 60.0)
        self._apply(tmp_path, monkeypatch, (
            '[decisions]\nmode = "active"\nbreaker_timeouts = 3\n'
            'breaker_cooldown_s = 10\n'))
        assert config.DECISIONS_BREAKER_TIMEOUTS == 3
        assert config.DECISIONS_BREAKER_COOLDOWN_S == 10.0

    def test_reads_active_tables(self, tmp_path, monkeypatch) -> None:
        self._apply(tmp_path, monkeypatch, (
            '[decisions]\nmode = "active"\ntimeout_ms = 800\n'
            '[decisions.points]\nstall = "ollama"\npanel = "encoder"\n'
            '[decisions.active]\nstall = true\npanel = false\n'
            '[decisions.thresholds]\nstall = 0.6\npanel = 1\n'))
        assert config.DECISIONS_MODE == 'active'
        assert config.DECISIONS_TIMEOUT_MS == 800
        assert config.DECISIONS_ACTIVE == {'stall': True, 'panel': False}
        assert config.DECISIONS_THRESHOLDS == {'stall': 0.6, 'panel': 1.0}

    def test_bad_active_values_are_skipped(self, tmp_path, monkeypatch):
        self._apply(tmp_path, monkeypatch, (
            '[decisions]\nmode = "active"\ntimeout_ms = "soon"\n'
            '[decisions.active]\nstall = "yes"\npanel = true\n'
            '[decisions.thresholds]\nstall = "high"\npanel = 0.7\n'))
        assert config.DECISIONS_TIMEOUT_MS == 1500
        assert config.DECISIONS_ACTIVE == {'panel': True}
        assert config.DECISIONS_THRESHOLDS == {'panel': 0.7}


class TestEvalsSettings:
    """settings.toml [evals]: default model spec and pinned context."""

    def _apply(self, tmp_path, monkeypatch, text):
        p = tmp_path / 'settings.toml'
        p.write_text(text, encoding='utf-8')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', p)
        monkeypatch.setattr(config, 'EVALS_MODEL', '')
        monkeypatch.setattr(config, 'EVALS_NUM_CTX', 8192)
        config._apply_settings()

    def test_defaults(self) -> None:
        assert config.EVALS_MODEL == ''
        assert config.EVALS_NUM_CTX == 8192

    def test_apply_reads_model_and_num_ctx(self, tmp_path,
                                           monkeypatch) -> None:
        self._apply(tmp_path, monkeypatch,
                    '[evals]\nmodel = "Ollama|huihui_ai/qwen3-abliterated:8b"'
                    '\nnum_ctx = 16384\n')
        assert config.EVALS_MODEL == 'Ollama|huihui_ai/qwen3-abliterated:8b'
        assert config.EVALS_NUM_CTX == 16384

    def test_missing_table_keeps_defaults(self, tmp_path, monkeypatch):
        self._apply(tmp_path, monkeypatch, '[tools]\nflat = false\n')
        assert config.EVALS_MODEL == ''
        assert config.EVALS_NUM_CTX == 8192

    def test_bad_num_ctx_is_ignored(self, tmp_path, monkeypatch) -> None:
        self._apply(tmp_path, monkeypatch,
                    '[evals]\nnum_ctx = "lots"\nmodel = 3\n')
        assert config.EVALS_NUM_CTX == 8192
        assert config.EVALS_MODEL == ''
