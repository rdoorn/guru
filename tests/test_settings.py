"""Typed [routing] settings loader (guru.repositories.settings)."""
import pytest

from guru import config
from guru.domain.routing import Ladder, Rung
from guru.repositories.adapters import AdapterRegistry
from guru.repositories.settings import (
    RoutingSettings, RungSpec, ladders_from_settings, load_decisions,
    load_routing)
from tests.test_adapters_registry import FakeAdapter


class TestDefaults:
    def test_empty_section_gives_documented_defaults(self) -> None:
        s = load_routing({})
        assert s == RoutingSettings(
            mode='local-and-remote', controller=False,
            complexity_router=True, type_router=False,
            spend_confirm='ask', secret_scan=True, ladders={})

    def test_present_only_with_a_configured_table(
            self, tmp_path, monkeypatch) -> None:
        assert load_routing({}).present is False
        assert load_routing({'mode': 'local-only'}).present is True
        p = tmp_path / 'settings.toml'
        p.write_text('[tools]\nflat = true\n', encoding='utf-8')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', p)
        assert load_routing().present is False
        p.write_text('[routing]\nsecret_scan = false\n', encoding='utf-8')
        s = load_routing()
        assert s.present is True and s.secret_scan is False

    def test_none_reads_settings_toml(self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'settings.toml'
        p.write_text('[routing]\nmode = "local-only"\ncontroller = true\n',
                     encoding='utf-8')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', p)
        s = load_routing()
        assert s.mode == 'local-only'
        assert s.controller is True

    def test_missing_settings_toml_gives_defaults(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            config, 'GLOBAL_SETTINGS_PATH', tmp_path / 'absent.toml')
        assert load_routing() == load_routing({})


class TestScalars:
    def test_all_scalars_parsed(self) -> None:
        s = load_routing({
            'mode': 'remote-only', 'controller': True,
            'complexity_router': False, 'type_router': True,
            'spend_confirm': 'never', 'secret_scan': False})
        assert (s.mode, s.controller, s.complexity_router, s.type_router,
                s.spend_confirm, s.secret_scan) == (
            'remote-only', True, False, True, 'never', False)

    @pytest.mark.parametrize('mode', ['local', 'LOCAL-ONLY', 'remote', ''])
    def test_bad_mode_raises(self, mode) -> None:
        with pytest.raises(ValueError, match='mode'):
            load_routing({'mode': mode})

    @pytest.mark.parametrize('value', ['always', 'yes', 'Ask'])
    def test_bad_spend_confirm_raises(self, value) -> None:
        with pytest.raises(ValueError, match='spend_confirm'):
            load_routing({'spend_confirm': value})

    @pytest.mark.parametrize('key', ['controller', 'complexity_router',
                                     'type_router', 'secret_scan'])
    def test_non_bool_flag_raises(self, key) -> None:
        with pytest.raises(ValueError, match=key):
            load_routing({key: 'true'})

    def test_unknown_keys_named(self) -> None:
        with pytest.raises(ValueError) as exc:
            load_routing({'mode': 'local-only', 'bogus': 1, 'zzz': 2})
        assert 'bogus' in str(exc.value)
        assert 'zzz' in str(exc.value)


class TestLadders:
    RUNGS = [
        {'adapter': 'Ollama', 'model': 'qwen3:4b',
         'max_complexity': 'trivial'},
        {'adapter': 'Ollama', 'model': 'qwen3:14b',
         'max_complexity': 'standard', 'default': True},
        {'adapter': 'Anthropic', 'model': 'claude-sonnet-5',
         'max_complexity': 'hard'},
    ]

    def test_default_ladder_from_routing_ladder(self) -> None:
        s = load_routing({'ladder': self.RUNGS})
        assert list(s.ladders) == ['default']
        assert s.ladders['default'] == [
            RungSpec('Ollama', 'qwen3:4b', 'trivial'),
            RungSpec('Ollama', 'qwen3:14b', 'standard', default=True),
            RungSpec('Anthropic', 'claude-sonnet-5', 'hard'),
        ]

    def test_per_kind_ladders_from_routing_ladders(self) -> None:
        s = load_routing({
            'ladder': self.RUNGS[:1],
            'ladders': {'review': self.RUNGS[2:], 'docs': self.RUNGS[:1]}})
        assert set(s.ladders) == {'default', 'review', 'docs'}
        assert s.ladders['review'] == [
            RungSpec('Anthropic', 'claude-sonnet-5', 'hard')]

    def test_per_kind_ladder_named_default_rejected(self) -> None:
        with pytest.raises(ValueError, match='default'):
            load_routing({'ladders': {'default': self.RUNGS[:1]}})

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match='ladders.poetry'):
            load_routing({'ladders': {'poetry': self.RUNGS[:1]}})

    def test_rung_missing_adapter_or_model(self) -> None:
        with pytest.raises(ValueError, match='model'):
            load_routing({'ladder': [{'adapter': 'Ollama',
                                      'max_complexity': 'hard'}]})
        with pytest.raises(ValueError, match='adapter'):
            load_routing({'ladder': [{'model': 'x',
                                      'max_complexity': 'hard'}]})

    def test_rung_bad_max_complexity(self) -> None:
        with pytest.raises(ValueError, match='max_complexity'):
            load_routing({'ladder': [{'adapter': 'Ollama', 'model': 'x',
                                      'max_complexity': 'insane'}]})

    def test_rung_missing_max_complexity(self) -> None:
        with pytest.raises(ValueError, match='max_complexity'):
            load_routing({'ladder': [{'adapter': 'Ollama', 'model': 'x'}]})

    def test_rung_unknown_key_named(self) -> None:
        with pytest.raises(ValueError, match='weight'):
            load_routing({'ladder': [{'adapter': 'Ollama', 'model': 'x',
                                      'max_complexity': 'hard',
                                      'weight': 3}]})

    def test_two_default_rungs_rejected(self) -> None:
        rungs = [dict(r, default=True) for r in self.RUNGS[:2]]
        with pytest.raises(ValueError, match='default'):
            load_routing({'ladder': rungs})

    def test_ladder_must_be_list_of_tables(self) -> None:
        with pytest.raises(ValueError, match='ladder'):
            load_routing({'ladder': {'adapter': 'Ollama'}})
        with pytest.raises(ValueError, match='ladders'):
            load_routing({'ladders': [1, 2]})

    def test_non_bool_default_rejected(self) -> None:
        with pytest.raises(ValueError, match='default'):
            load_routing({'ladder': [{'adapter': 'Ollama', 'model': 'x',
                                      'max_complexity': 'hard',
                                      'default': 'yes'}]})

    def test_full_toml_round_trip(self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'settings.toml'
        p.write_text(
            '[routing]\n'
            'mode = "local-and-remote"\n'
            'type_router = true\n'
            '[[routing.ladder]]\n'
            'adapter = "Ollama"\nmodel = "qwen3:14b"\n'
            'max_complexity = "standard"\ndefault = true\n'
            '[[routing.ladders.review]]\n'
            'adapter = "Anthropic"\nmodel = "claude-sonnet-5"\n'
            'max_complexity = "hard"\n', encoding='utf-8')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', p)
        s = load_routing()
        assert s.type_router is True
        assert s.ladders['default'][0].default is True
        assert s.ladders['review'][0].adapter == 'Anthropic'


class TestLaddersFromSettings:
    def _registry(self) -> AdapterRegistry:
        reg = AdapterRegistry()
        reg.register(FakeAdapter('Ollama', remote=False))
        reg.register(FakeAdapter('Anthropic', remote=True))
        return reg

    def test_builds_domain_ladders_with_remote_flag(self) -> None:
        s = load_routing({
            'ladder': TestLadders.RUNGS,
            'ladders': {'review': TestLadders.RUNGS[2:]}})
        ladders = ladders_from_settings(s, self._registry())
        assert set(ladders) == {'default', 'review'}
        assert isinstance(ladders['default'], Ladder)
        assert ladders['default'].rungs == [
            Rung('Ollama', 'qwen3:4b', 'trivial', remote=False),
            Rung('Ollama', 'qwen3:14b', 'standard', remote=False,
                 default=True),
            Rung('Anthropic', 'claude-sonnet-5', 'hard', remote=True),
        ]
        assert ladders['review'].rungs[0].remote is True

    def test_unknown_adapter_rung_dropped_with_warning(self, caplog) -> None:
        s = load_routing({'ladder': [
            {'adapter': 'Nope', 'model': 'x', 'max_complexity': 'hard'},
            TestLadders.RUNGS[0]]})
        with caplog.at_level('WARNING', logger='guru'):
            ladders = ladders_from_settings(s, self._registry())
        assert [r.model for r in ladders['default'].rungs] == ['qwen3:4b']
        assert 'Nope' in caplog.text

    def test_ladder_with_no_valid_rungs_omitted(self) -> None:
        s = load_routing({'ladder': [
            {'adapter': 'Nope', 'model': 'x', 'max_complexity': 'hard'}]})
        assert ladders_from_settings(s, self._registry()) == {}

    def test_empty_settings_gives_empty_ladders(self) -> None:
        assert ladders_from_settings(load_routing({}), self._registry()) == {}


class TestDecisionsSettings:
    """``load_decisions``: the ``[decisions]`` table of an experiment file
    (mode + points/active/thresholds sub-tables)."""

    def test_empty_section_gives_off_defaults(self) -> None:
        from guru.repositories.settings import DecisionsSettings
        s = load_decisions({})
        assert s == DecisionsSettings()
        assert s.mode == 'off' and s.points == {} and s.active == {} \
            and s.thresholds == {}

    def test_full_table_parsed(self) -> None:
        s = load_decisions({
            'mode': 'shadow',
            'points': {'panel': 'encoder', 'injection': 'injection'},
            'active': {'stall': True},
            'thresholds': {'stall': 0.7, 'panel': 1}})
        assert s.mode == 'shadow'
        assert s.points == {'panel': 'encoder', 'injection': 'injection'}
        assert s.active == {'stall': True}
        assert s.thresholds == {'stall': 0.7, 'panel': 1.0}

    @pytest.mark.parametrize('mode', ['bogus', 3, ''])
    def test_bad_mode_raises(self, mode) -> None:
        with pytest.raises(ValueError, match='mode'):
            load_decisions({'mode': mode})

    def test_unknown_keys_named(self) -> None:
        with pytest.raises(ValueError, match='sidecar_model'):
            load_decisions({'mode': 'shadow', 'sidecar_model': 'x'})

    def test_points_must_map_to_strings(self) -> None:
        with pytest.raises(ValueError, match='points'):
            load_decisions({'points': {'stall': 3}})
        with pytest.raises(ValueError, match='points'):
            load_decisions({'points': 'encoder'})

    def test_active_must_map_to_bools(self) -> None:
        with pytest.raises(ValueError, match='active'):
            load_decisions({'active': {'stall': 'yes'}})

    def test_thresholds_must_map_to_numbers(self) -> None:
        with pytest.raises(ValueError, match='thresholds'):
            load_decisions({'thresholds': {'stall': True}})
        with pytest.raises(ValueError, match='thresholds'):
            load_decisions({'thresholds': {'stall': '0.7'}})

    def test_labels_margin_defaults_and_parses(self) -> None:
        assert load_decisions({}).labels_margin == 0.15
        assert load_decisions({'labels_margin': 0.3}).labels_margin == 0.3
        assert load_decisions({'labels_margin': 0}).labels_margin == 0.0

    @pytest.mark.parametrize('value', ['0.3', True, -0.1])
    def test_labels_margin_must_be_a_non_negative_number(self, value):
        with pytest.raises(ValueError, match='labels_margin'):
            load_decisions({'labels_margin': value})


class TestControllerDefault:
    """``controller`` defaults to on when any ladder rung is configured;
    an explicit value always wins."""

    _RUNG = {'adapter': 'A', 'model': 'm', 'max_complexity': 'hard'}

    def test_ladder_without_controller_key_turns_it_on(self) -> None:
        assert load_routing({'ladder': [self._RUNG]}).controller is True
        assert load_routing(
            {'ladders': {'review': [self._RUNG]}}).controller is True

    def test_no_ladder_leaves_it_off(self) -> None:
        assert load_routing({}).controller is False
        assert load_routing({'mode': 'local-only'}).controller is False
        assert load_routing({'ladder': []}).controller is False

    def test_explicit_false_with_ladder_stays_off(self) -> None:
        s = load_routing({'controller': False, 'ladder': [self._RUNG]})
        assert s.controller is False

    def test_explicit_true_without_ladder_stays_on(self) -> None:
        assert load_routing({'controller': True}).controller is True

    def test_dataclass_resolves_the_default_too(self) -> None:
        rung = RungSpec('A', 'm', 'hard')
        assert RoutingSettings(ladders={'default': [rung]}).controller is True
        assert RoutingSettings().controller is False
        assert RoutingSettings(controller=False,
                               ladders={'default': [rung]}).controller is False


class TestToolsPolicyLoader:
    """load_tools_policy parses a project's .guru/tools.toml (A3)."""

    def _load(self, tmp_path, text: str):
        from guru.repositories.settings import load_tools_policy
        p = tmp_path / 'tools.toml'
        p.write_text(text, encoding='utf-8')
        return load_tools_policy(p)

    def test_missing_file_is_the_default_policy(self, tmp_path) -> None:
        from guru.repositories.settings import ToolsPolicy, load_tools_policy
        pol = load_tools_policy(tmp_path / 'absent.toml')
        assert pol == ToolsPolicy()
        assert pol.enabled == set() and pol.disabled == set()
        assert pol.test_runner == 'pytest' and pol.limits == {}

    def test_default_path_is_config(self, tmp_path, monkeypatch) -> None:
        from guru.repositories.settings import load_tools_policy
        p = tmp_path / 'tools.toml'
        p.write_text('[tools]\ndisabled = ["web_fetch"]\n', encoding='utf-8')
        monkeypatch.setattr(config, 'TOOLS_POLICY_PATH', p)
        assert load_tools_policy().disabled == {'web_fetch'}

    def test_full_table(self, tmp_path) -> None:
        pol = self._load(tmp_path,
                         '[tools]\nenabled = ["read_file", "run_tests"]\n'
                         'disabled = ["web_search"]\n'
                         '[tools.tests]\nrunner = "unittest"\n'
                         '[tools.limits]\ntimeout_s = 30\nout_kb = 8\n')
        assert pol.enabled == {'read_file', 'run_tests'}
        assert pol.disabled == {'web_search'}
        assert pol.test_runner == 'unittest'
        assert pol.limits == {'timeout_s': 30, 'out_kb': 8}

    def test_unknown_key_names_the_path(self, tmp_path) -> None:
        with pytest.raises(ValueError, match='tools.toml') as e:
            self._load(tmp_path, '[tools]\nenable = ["x"]\n')
        assert 'enable' in str(e.value)

    def test_unknown_top_level_table(self, tmp_path) -> None:
        with pytest.raises(ValueError, match='tools.toml'):
            self._load(tmp_path, '[routing]\nmode = "x"\n')

    def test_bad_runner(self, tmp_path) -> None:
        with pytest.raises(ValueError, match='tools.toml') as e:
            self._load(tmp_path, '[tools.tests]\nrunner = "nose"\n')
        assert 'nose' in str(e.value) and 'pytest' in str(e.value)

    def test_bad_limits(self, tmp_path) -> None:
        with pytest.raises(ValueError, match='tools.toml'):
            self._load(tmp_path, '[tools.limits]\ntimeout_s = "slow"\n')
        with pytest.raises(ValueError, match='tools.toml'):
            self._load(tmp_path, '[tools.limits]\nram = 4\n')

    def test_lists_must_hold_strings(self, tmp_path) -> None:
        with pytest.raises(ValueError, match='tools.toml'):
            self._load(tmp_path, '[tools]\nenabled = "read_file"\n')

    def test_invalid_toml_names_the_path(self, tmp_path) -> None:
        with pytest.raises(ValueError, match='tools.toml'):
            self._load(tmp_path, '[tools\nx = \n')

    def test_unreadable_file_raises_not_defaults(self, tmp_path) -> None:
        from guru.repositories.settings import load_tools_policy
        d = tmp_path / 'tools.toml'
        d.mkdir()                     # exists but cannot be read as a file
        with pytest.raises(ValueError, match='tools.toml'):
            load_tools_policy(d)


class TestSandboxSettings:
    """load_sandbox merges settings.toml [sandbox] with .guru/sandbox.toml
    (project wins; ``enabled`` only from the project file)."""

    PIN = 'python:3.12-slim@sha256:' + 'b' * 64

    def _project_file(self, tmp_path, text: str):
        p = tmp_path / 'sandbox.toml'
        p.write_text(text, encoding='utf-8')
        return p

    def test_defaults(self, tmp_path) -> None:
        from guru.repositories.settings import (
            DEFAULT_BASE_IMAGE, DEFAULT_PROXY_IMAGE, SandboxSettings,
            load_sandbox)
        s = load_sandbox({}, tmp_path / 'absent.toml')
        assert s == SandboxSettings()
        assert s.enabled is False and s.runtime == 'docker'
        assert s.base_image == DEFAULT_BASE_IMAGE
        assert '@sha256:' in DEFAULT_BASE_IMAGE
        assert '@sha256:' in DEFAULT_PROXY_IMAGE
        assert (s.cpus, s.memory_mb, s.pids, s.timeout_s) == (
            2.0, 2048, 256, 600)

    def test_default_paths_are_config(self, tmp_path, monkeypatch) -> None:
        from guru.repositories.settings import load_sandbox
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH',
                            tmp_path / 'settings.toml')
        (tmp_path / 'settings.toml').write_text(
            '[sandbox]\ncpus = 4\n', encoding='utf-8')
        p = self._project_file(tmp_path, '[sandbox]\nenabled = true\n')
        monkeypatch.setattr(config, 'SANDBOX_POLICY_PATH', p)
        s = load_sandbox()
        assert s.enabled is True and s.cpus == 4.0

    def test_project_file_wins_key_by_key(self, tmp_path) -> None:
        from guru.repositories.settings import load_sandbox
        p = self._project_file(
            tmp_path, f'[sandbox]\nenabled = true\nmemory_mb = 512\n'
                      f'base_image = "{self.PIN}"\n')
        s = load_sandbox({'cpus': 1.5, 'memory_mb': 4096, 'pids': 32}, p)
        assert s.enabled is True
        assert s.cpus == 1.5 and s.pids == 32          # global kept
        assert s.memory_mb == 512                        # project wins
        assert s.base_image == self.PIN

    def test_enabled_only_from_the_project_file(self, tmp_path) -> None:
        from guru.repositories.settings import load_sandbox
        with pytest.raises(ValueError, match='enabled') as e:
            load_sandbox({'enabled': True}, tmp_path / 'absent.toml')
        assert 'sandbox.toml' in str(e.value)
        p = self._project_file(tmp_path, '[sandbox]\nenabled = "yes"\n')
        with pytest.raises(ValueError, match='enabled'):
            load_sandbox({}, p)

    def test_absent_project_file_stays_disabled(self, tmp_path) -> None:
        from guru.repositories.settings import load_sandbox
        assert load_sandbox({'cpus': 1}, tmp_path / 'nope.toml').enabled \
            is False

    def test_unknown_key_named(self, tmp_path) -> None:
        from guru.repositories.settings import load_sandbox
        with pytest.raises(ValueError, match='cpu_count'):
            load_sandbox({'cpu_count': 2}, tmp_path / 'absent.toml')
        p = self._project_file(tmp_path, '[sandbox]\nmemory = 1\n')
        with pytest.raises(ValueError, match='sandbox.toml') as e:
            load_sandbox({}, p)
        assert 'memory' in str(e.value)

    def test_unknown_table_in_project_file(self, tmp_path) -> None:
        from guru.repositories.settings import load_sandbox
        p = self._project_file(tmp_path, '[tools]\nx = 1\n')
        with pytest.raises(ValueError, match='sandbox.toml'):
            load_sandbox({}, p)

    def test_invalid_toml_and_unreadable_file(self, tmp_path) -> None:
        from guru.repositories.settings import load_sandbox
        p = self._project_file(tmp_path, '[sandbox\nx = \n')
        with pytest.raises(ValueError, match='sandbox.toml'):
            load_sandbox({}, p)
        d = tmp_path / 'dir.toml'
        d.mkdir()
        with pytest.raises(ValueError, match='dir.toml'):
            load_sandbox({}, d)

    @pytest.mark.parametrize('key', ['base_image', 'proxy_image'])
    @pytest.mark.parametrize('value', ['python:3.12-slim', 'x@sha256: y', 7])
    def test_images_must_be_digest_pinned(self, tmp_path, key, value):
        from guru.repositories.settings import load_sandbox
        with pytest.raises(ValueError, match=key):
            load_sandbox({key: value}, tmp_path / 'absent.toml')

    def test_bad_runtime(self, tmp_path) -> None:
        from guru.repositories.settings import load_sandbox
        with pytest.raises(ValueError, match='runtime') as e:
            load_sandbox({'runtime': 'podman'}, tmp_path / 'absent.toml')
        assert 'docker' in str(e.value)

    @pytest.mark.parametrize('key,value', [
        ('cpus', 0), ('cpus', '2'), ('cpus', True), ('memory_mb', 0),
        ('memory_mb', 2.5), ('memory_mb', True), ('pids', -1),
        ('timeout_s', 'soon'), ('timeout_s', 0)])
    def test_bad_numbers(self, tmp_path, key, value) -> None:
        from guru.repositories.settings import load_sandbox
        with pytest.raises(ValueError, match=key):
            load_sandbox({key: value}, tmp_path / 'absent.toml')

    def test_cpus_accepts_int_and_float(self, tmp_path) -> None:
        from guru.repositories.settings import load_sandbox
        assert load_sandbox({'cpus': 3}, tmp_path / 'x').cpus == 3.0
        assert load_sandbox({'cpus': 0.5}, tmp_path / 'x').cpus == 0.5
