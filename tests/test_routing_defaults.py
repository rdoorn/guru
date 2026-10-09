"""The default [routing] block (settings.default_routing_toml /
ensure_default_routing), ``mode = "off"`` and the ``/routing`` command."""
import re
import tomllib

import pytest

from guru import cli, config
from guru.domain import decisions, policy
from guru.domain.pricing import DEFAULT_PRICES
from guru.repositories import settings as rs

LITELLM = {'name': 'SBP Litellm', 'type': 'litellm', 'enable': True}
ANTHROPIC = {'name': 'Claude', 'type': 'anthropic', 'enable': True}
OLLAMA = {'name': 'Ollama', 'type': 'ollama', 'enable': True}


@pytest.fixture
def config_snapshot():
    """Restore every ``config`` global afterwards: ``ensure_default_routing``
    re-applies the settings file, which rewrites tools, sampling,
    decisions and ledger globals from whatever the test wrote."""
    saved = {k: v for k, v in vars(config).items() if k.isupper()}
    yield
    for k, v in saved.items():
        setattr(config, k, v)
    policy.set_scanner(None)


@pytest.fixture
def settings_file(tmp_path, monkeypatch, config_snapshot):
    """A tmp settings.toml wired as ``config.GLOBAL_SETTINGS_PATH``."""
    path = tmp_path / 'settings.toml'
    monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', path)
    return path


class TestOffMode:
    def test_off_yields_inert_defaults_after_validating(self) -> None:
        section = {'mode': 'off', 'ladder': [
            {'adapter': 'A', 'model': 'm', 'max_complexity': 'hard'}]}
        s = rs.load_routing(section)
        assert s == rs.RoutingSettings(off=True)
        assert s.present is False

    def test_off_still_rejects_a_broken_table(self) -> None:
        with pytest.raises(ValueError, match='unknown keys'):
            rs.load_routing({'mode': 'off', 'nope': 1})

    def test_full_keeps_the_ladders_for_display(self) -> None:
        section = {'mode': 'off', 'ladder': [
            {'adapter': 'A', 'model': 'm', 'max_complexity': 'hard'}]}
        s = rs.load_routing(section, full=True)
        assert s.off is True and s.present is True
        assert s.mode == 'local-and-remote'
        assert [r.model for r in s.ladders['default']] == ['m']


class TestDefaultRoutingToml:
    def _parse(self, **kw) -> dict:
        return tomllib.loads(rs.default_routing_toml(
            'SBP Litellm', 'litellm', judges_available=True, **kw))

    def test_renders_the_measured_configuration(self) -> None:
        data = self._parse()
        s = rs.load_routing(data['routing'])
        assert s.mode == 'local-and-remote'
        assert 'controller' not in data['routing']   # retired key
        assert s.complexity_router is True and s.type_router is True
        assert s.spend_confirm == 'ask' and s.secret_scan is True
        default = s.ladders['default']
        assert [(r.model, r.max_complexity, r.default) for r in default] == [
            ('aws/claude-4-5-haiku', 'trivial', False),
            ('aws/claude-5-sonnet', 'standard', True),
            ('aws/claude-5-5-opus', 'hard', False)]
        assert all(r.adapter == 'SBP Litellm' for r in default)
        review = s.ladders['review']
        assert [(r.model, r.max_complexity) for r in review] == [
            ('aws/claude-5-sonnet', 'standard'), ('aws/claude-5-5-opus',
                                                  'hard')]
        assert review[0].default is True

    def test_judges_active_when_the_extra_is_available(self) -> None:
        d = rs.load_decisions(self._parse()['decisions'])
        assert d.mode == 'active' and d.labels_margin == 0.15
        assert d.points == {'labels': 'decide', 'panel': 'encoder',
                            'injection': 'injection'}
        # panel stays shadow until labelled rows say otherwise
        assert d.active == {'labels': True, 'panel': False}

    def test_panel_is_shadow_regardless_of_the_extra(self) -> None:
        for available in (True, False):
            text = rs.default_routing_toml('SBP Litellm', 'litellm',
                                           judges_available=available)
            d = rs.load_decisions(tomllib.loads(text)['decisions'])
            assert d.active['panel'] is False
            assert d.points['panel'] == 'encoder'     # configured, observed
        assert 'panel = false               # shadow until labelled rows' \
            in text

    def test_judges_shadow_with_a_note_without_the_extra(self) -> None:
        text = rs.default_routing_toml('SBP Litellm', 'litellm',
                                       judges_available=False)
        d = rs.load_decisions(tomllib.loads(text)['decisions'])
        assert d.active == {'labels': False, 'panel': False}
        assert 'uv sync --extra judge' in text

    def test_labels_comment_names_the_nli_alternative(self) -> None:
        text = rs.default_routing_toml('X', 'litellm', judges_available=True)
        line = next(ln for ln in text.splitlines()
                    if ln.startswith('labels = "decide"'))
        assert '"encoder"' in line

    def test_extra_probe_needs_gliner2(self, monkeypatch) -> None:
        import importlib.util
        present = {'torch', 'transformers'}
        monkeypatch.setattr(importlib.util, 'find_spec',
                            lambda name: object() if name in present
                            else None)
        assert rs._judge_extra_available() is False
        present.add('gliner2')
        assert rs._judge_extra_available() is True

    def test_extra_probe_is_used_when_not_told(self, monkeypatch) -> None:
        monkeypatch.setattr(rs, '_judge_extra_available', lambda: False)
        text = rs.default_routing_toml('X', 'litellm')
        assert 'labels = false' in text

    def test_anthropic_ids_are_priced_first_party_ids(self) -> None:
        text = rs.default_routing_toml('Claude', 'anthropic',
                                       judges_available=True)
        s = rs.load_routing(tomllib.loads(text)['routing'])
        models = [r.model for r in s.ladders['default']]
        assert models == ['claude-haiku-4-5', 'claude-sonnet-5',
                          'claude-opus-5-5']
        assert all(m in DEFAULT_PRICES for m in models)
        assert all(r.adapter == 'Claude' for r in s.ladders['review'])

    def test_decisions_table_can_be_omitted(self) -> None:
        data = self._parse(decisions=False)
        assert 'decisions' not in data and 'routing' in data

    def test_unknown_kind_and_bad_name_raise(self) -> None:
        with pytest.raises(ValueError, match='ollama'):
            rs.default_routing_toml('Ollama', 'ollama')
        with pytest.raises(ValueError, match='TOML string'):
            rs.default_routing_toml('bad "name"', 'litellm')

    def test_header_comment_explains_the_switch(self) -> None:
        text = rs.default_routing_toml('X', 'litellm', judges_available=True)
        assert '/routing off' in text and 'never rewrites' in text


OLD_LABELS = ('labels = "encoder"          # complexity tie-breaker for the'
              " controller's label")


def _old_block() -> str:
    """The [decisions] block guru wrote before labels moved to decide."""
    return ('[routing]\nmode = "local-and-remote"\n\n'
            '[decisions]\nmode = "active"\n\n'
            f'[decisions.points]\n{OLD_LABELS}\n'
            'panel = "encoder"           # needs_security\n\n'
            '[decisions.active]\nlabels = true\n')


class TestMigrateLabelsJudge:
    @pytest.fixture(autouse=True)
    def _extra(self, monkeypatch):
        monkeypatch.setattr(rs, '_judge_extra_available', lambda: True)

    def test_untouched_default_moves_to_decide(self, settings_file) -> None:
        settings_file.write_text(_old_block())
        assert rs.migrate_labels_judge() == 'migrated'
        text = settings_file.read_text()
        assert OLD_LABELS not in text
        d = rs.load_decisions(tomllib.loads(text)['decisions'])
        assert d.points['labels'] == 'decide'
        assert d.points['panel'] == 'encoder'
        assert d.active == {'labels': True}
        # every other line is kept byte for byte
        assert text.replace(rs.LABELS_DECIDE_LINE, OLD_LABELS) == \
            _old_block()

    def test_new_line_matches_the_written_default(self) -> None:
        text = rs.default_routing_toml('X', 'litellm', judges_available=True)
        assert rs.LABELS_DECIDE_LINE in text.splitlines()

    def test_idempotent(self, settings_file) -> None:
        settings_file.write_text(_old_block())
        rs.migrate_labels_judge()
        once = settings_file.read_text()
        assert rs.migrate_labels_judge() == 'unchanged'
        assert settings_file.read_text() == once

    def test_edited_line_is_left_alone(self, settings_file) -> None:
        for line in ('labels = "encoder"', 'labels = "encoder"  # mine',
                     'labels = "encoder:org/Other"'):
            text = _old_block().replace(OLD_LABELS, line)
            settings_file.write_text(text)
            assert rs.migrate_labels_judge() == 'unchanged'
            assert settings_file.read_text() == text

    def test_same_line_outside_decisions_points_is_left_alone(
            self, settings_file) -> None:
        text = f'[other]\n{OLD_LABELS}\n'
        settings_file.write_text(text)
        assert rs.migrate_labels_judge() == 'unchanged'
        assert settings_file.read_text() == text

    def test_missing_or_invalid_file(self, settings_file) -> None:
        assert rs.migrate_labels_judge() == 'unchanged'
        settings_file.write_text('[decisions.points\n' + OLD_LABELS)
        assert rs.migrate_labels_judge() == 'invalid'

    def test_without_the_extra_the_nli_judge_is_kept(
            self, settings_file, monkeypatch) -> None:
        # decide would not load; the working encoder tie-breaker stays
        monkeypatch.setattr(rs, '_judge_extra_available', lambda: False)
        settings_file.write_text(_old_block())
        assert rs.migrate_labels_judge() == 'unchanged'
        assert settings_file.read_text() == _old_block()

    def test_crlf_line_endings_are_kept(self, settings_file) -> None:
        crlf = _old_block().replace('\n', '\r\n').encode()
        settings_file.write_bytes(crlf)
        assert rs.migrate_labels_judge() == 'migrated'
        assert settings_file.read_bytes() == crlf.replace(
            OLD_LABELS.encode(), rs.LABELS_DECIDE_LINE.encode())

    def test_array_of_tables_header_is_not_the_table(
            self, settings_file) -> None:
        text = f'[[decisions.points]]\n{OLD_LABELS}\n'
        settings_file.write_text(text)
        assert rs.migrate_labels_judge() == 'unchanged'
        assert settings_file.read_text() == text

    def test_line_inside_a_multiline_string_is_left_alone(
            self, settings_file) -> None:
        text = (f'[notes]\ntext = """\n[decisions.points]\n{OLD_LABELS}\n'
                '"""\n')
        settings_file.write_text(text)
        assert rs.migrate_labels_judge() == 'unchanged'
        assert settings_file.read_text() == text

    def test_reapplies_settings_after_migrating(self, settings_file,
                                                monkeypatch) -> None:
        settings_file.write_text(_old_block())
        applied = []
        monkeypatch.setattr(config, '_apply_settings',
                            lambda: applied.append(1))
        rs.migrate_labels_judge()
        assert applied == [1]


class TestEnsureDefaultRouting:
    def test_writes_into_a_missing_file(self, settings_file) -> None:
        out = rs.ensure_default_routing([OLLAMA, LITELLM],
                                        judges_available=True)
        assert out == 'written'
        data = tomllib.loads(settings_file.read_text(encoding='utf-8'))
        assert rs.load_routing(data['routing']).present is True
        assert data['decisions']['mode'] == 'active'
        assert data['routing']['ladder'][0]['adapter'] == 'SBP Litellm'
        # the process picked the new [decisions] up right away
        assert config.DECISIONS_MODE == 'active'
        assert config.DECISIONS_ACTIVE == {'labels': True, 'panel': False}

    def test_creates_parent_directories(self, tmp_path, monkeypatch,
                                        config_snapshot) -> None:
        path = tmp_path / 'deep' / 'er' / 'settings.toml'
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', path)
        assert rs.ensure_default_routing(
            [LITELLM], judges_available=True) == 'written'
        assert path.is_file()

    def test_appends_after_other_tables_untouched(self, settings_file):
        before = ('[tools]\nflat = true   # keep me\n\n'
                  '[sampling]\ntemperature = 0.2\n')
        settings_file.write_text(before, encoding='utf-8')
        assert rs.ensure_default_routing([LITELLM],
                                         judges_available=True) == 'written'
        text = settings_file.read_text(encoding='utf-8')
        assert text.startswith(before)
        data = tomllib.loads(text)
        assert data['tools'] == {'flat': True}
        assert data['sampling'] == {'temperature': 0.2}
        assert 'routing' in data and 'decisions' in data

    def test_file_without_trailing_newline_stays_valid(self, settings_file):
        settings_file.write_text('[tools]\nflat = true', encoding='utf-8')
        assert rs.ensure_default_routing([LITELLM],
                                         judges_available=True) == 'written'
        data = tomllib.loads(settings_file.read_text(encoding='utf-8'))
        assert data['tools'] == {'flat': True} and 'routing' in data

    def test_existing_decisions_table_is_kept_and_not_duplicated(
            self, settings_file) -> None:
        settings_file.write_text('[decisions]\nmode = "shadow"\n',
                                 encoding='utf-8')
        assert rs.ensure_default_routing([LITELLM],
                                         judges_available=True) == 'written'
        data = tomllib.loads(settings_file.read_text(encoding='utf-8'))
        assert data['decisions'] == {'mode': 'shadow'}
        assert 'routing' in data

    def test_existing_routing_table_is_never_overwritten(self, settings_file):
        text = '[routing]\nmode = "local-only"\n'
        settings_file.write_text(text, encoding='utf-8')
        assert rs.ensure_default_routing([LITELLM]) == 'exists'
        assert settings_file.read_text(encoding='utf-8') == text

    def test_empty_routing_table_counts_as_existing(self, settings_file):
        settings_file.write_text('[routing]\n', encoding='utf-8')
        assert rs.ensure_default_routing([LITELLM]) == 'exists'

    def test_no_remote_adapter_writes_nothing(self, settings_file) -> None:
        disabled = {**LITELLM, 'enable': False}
        assert rs.ensure_default_routing([OLLAMA, disabled]) == 'no-remote'
        assert rs.ensure_default_routing([]) == 'no-remote'
        assert not settings_file.exists()

    def test_first_enabled_remote_adapter_wins(self, settings_file) -> None:
        anthropic_off = {**ANTHROPIC, 'enable': False}
        rs.ensure_default_routing([OLLAMA, anthropic_off, LITELLM, ANTHROPIC],
                                  judges_available=True)
        data = tomllib.loads(settings_file.read_text(encoding='utf-8'))
        assert data['routing']['ladder'][0]['adapter'] == 'SBP Litellm'

    def test_anthropic_adapter_gets_first_party_ids(self, settings_file):
        rs.ensure_default_routing([ANTHROPIC], judges_available=True)
        data = tomllib.loads(settings_file.read_text(encoding='utf-8'))
        assert data['routing']['ladder'][1]['model'] == 'claude-sonnet-5'

    def test_invalid_toml_is_left_alone(self, settings_file) -> None:
        settings_file.write_text('[tools\nflat = ', encoding='utf-8')
        assert rs.ensure_default_routing([LITELLM]) == 'invalid'
        assert settings_file.read_text(encoding='utf-8') == '[tools\nflat = '

    def test_defaults_to_the_configured_adapters(self, settings_file,
                                                 monkeypatch) -> None:
        monkeypatch.setattr(config, 'load_adapter_configs',
                            lambda: [LITELLM])
        assert rs.ensure_default_routing(judges_available=True) == 'written'


class TestSwitchRouting:
    def _write(self, path, text: str) -> None:
        path.write_text(text, encoding='utf-8')

    def test_off_then_on_round_trips_and_keeps_everything_else(
            self, settings_file) -> None:
        rs.ensure_default_routing([LITELLM], judges_available=True)
        before = settings_file.read_text(encoding='utf-8')
        assert rs.switch_routing(False) == 'off'
        after = settings_file.read_text(encoding='utf-8')
        assert re.search(r'^mode = "off" +# was "local-and-remote"', after,
                         re.MULTILINE)
        assert rs.load_routing().off is True
        # only the mode line differs
        diff = [(a, b) for a, b in zip(before.splitlines(),
                                       after.splitlines()) if a != b]
        assert len(diff) == 1 and diff[0][0].startswith('mode = ')
        assert rs.switch_routing(True) == 'local-and-remote'
        assert settings_file.read_text(encoding='utf-8') == before

    def test_remembers_a_non_default_mode(self, settings_file) -> None:
        self._write(settings_file,
                    '[routing]\nmode = "remote-only"\ncontroller = true\n')
        assert rs.switch_routing(False) == 'off'
        assert rs.switch_routing(True) == 'remote-only'
        assert settings_file.read_text(encoding='utf-8') == (
            '[routing]\nmode = "remote-only"\ncontroller = true\n')

    def test_missing_mode_line_is_inserted_under_the_header(
            self, settings_file) -> None:
        self._write(settings_file, '[tools]\nflat = true\n\n'
                                   '[routing]\ncontroller = true\n')
        assert rs.switch_routing(False) == 'off'
        text = settings_file.read_text(encoding='utf-8')
        assert text == ('[tools]\nflat = true\n\n'
                        '[routing]\nmode = "off"\ncontroller = true\n')
        assert rs.load_routing().off is True
        assert rs.switch_routing(True) == 'local-and-remote'
        assert 'mode = "local-and-remote"' in settings_file.read_text(
            encoding='utf-8')

    def test_idempotent(self, settings_file) -> None:
        self._write(settings_file, '[routing]\nmode = "local-only"\n')
        assert rs.switch_routing(True) == 'local-only'
        assert rs.switch_routing(False) == 'off'
        assert rs.switch_routing(False) == 'off'
        assert settings_file.read_text(encoding='utf-8').count('was') == 1

    def test_mode_line_in_another_table_is_not_touched(self, settings_file):
        self._write(settings_file,
                    '[decisions]\nmode = "active"\n\n'
                    '[routing]\ncontroller = true\n\n'
                    '[sandbox]\nmode = "x"\n')
        rs.switch_routing(False)
        data = tomllib.loads(settings_file.read_text(encoding='utf-8'))
        assert data['decisions']['mode'] == 'active'
        assert data['sandbox']['mode'] == 'x'
        assert data['routing']['mode'] == 'off'

    def test_plain_off_without_marker_restores_the_default(
            self, settings_file) -> None:
        self._write(settings_file, '[routing]\nmode = "off"\n')
        assert rs.switch_routing(True) == 'local-and-remote'

    def test_no_routing_table_raises(self, settings_file) -> None:
        self._write(settings_file, '[tools]\nflat = true\n')
        with pytest.raises(ValueError, match='no \\[routing\\] table'):
            rs.switch_routing(False)
        with pytest.raises(ValueError, match='cannot read'):
            rs.switch_routing(False, settings_file.parent / 'absent.toml')


class TestRoutingCommand:
    @pytest.fixture(autouse=True)
    def _judges(self, monkeypatch):
        monkeypatch.setattr(config, 'DECISIONS_MODE', 'active')
        monkeypatch.setattr(config, 'DECISIONS_POINTS',
                            {'labels': 'encoder', 'panel': 'encoder'})
        monkeypatch.setattr(config, 'DECISIONS_ACTIVE', {'labels': True})
        decisions.clear_judges()
        yield
        decisions.clear_judges()

    def test_format_lists_flags_ladders_and_judges(self, tmp_path) -> None:
        text = rs.default_routing_toml('SBP Litellm', 'litellm',
                                       judges_available=True)
        full = rs.load_routing(tomllib.loads(text)['routing'], full=True)

        class J:
            name = 'enc'

            def ask(self, qs):
                return []
        decisions.set_judge('labels', J())
        out = cli._format_routing(full, tmp_path / 's.toml')
        assert out.splitlines()[0] == 'routing: on (mode local-and-remote)'
        assert 'type router on' in out and 'controller' not in out
        assert 'ladder default:' in out and 'ladder review (kind review):' \
            in out
        assert 'SBP Litellm | aws/claude-4-5-haiku' in out
        assert 'up to standard  (default rung)' in out
        assert 'labels active (encoder, installed)' in out
        assert 'panel shadow (encoder, not installed)' in out
        assert out.splitlines()[-1] == f'file: {tmp_path / "s.toml"}'

    def test_format_off_and_absent(self, tmp_path) -> None:
        off = rs.load_routing({'mode': 'off', 'ladder': [
            {'adapter': 'A', 'model': 'm', 'max_complexity': 'hard'}]},
            full=True)
        out = cli._format_routing(off, tmp_path)
        assert out.startswith('routing: off (mode = "off"')
        assert 'A | m' in out                    # ladders still shown
        absent = cli._format_routing(rs.RoutingSettings(), tmp_path)
        assert absent.startswith('routing: not configured')

    def test_plain_command_prints_the_state(self, settings_file, capsys):
        rs.ensure_default_routing([LITELLM], judges_available=True)
        assert cli._routing_command('') is None
        out = capsys.readouterr().out
        assert 'routing: on' in out and 'aws/claude-5-5-opus' in out

    def test_off_toggles_file_and_reloads(self, settings_file, capsys):
        rs.ensure_default_routing([LITELLM], judges_available=True)
        config.SECRET_SCAN = True
        reloaded = cli._routing_command('off')
        assert reloaded is not None and reloaded.off is True
        assert reloaded.present is False
        assert config.SECRET_SCAN is False and policy.scanner() is None
        assert 'mode = "off"' in settings_file.read_text(encoding='utf-8')
        out = capsys.readouterr().out
        assert 'routing off' in out and 'routing: off' in out

    def test_on_restores_and_rebinds(self, settings_file, capsys) -> None:
        rs.ensure_default_routing([LITELLM], judges_available=True)
        rs.switch_routing(False)
        reloaded = cli._routing_command('ON')
        assert reloaded is not None and reloaded.present is True
        assert reloaded.off is False and reloaded.type_router is True
        assert config.SECRET_SCAN is True and policy.scanner() is not None
        assert 'routing: on (mode local-and-remote)' in capsys.readouterr().out

    def test_unknown_word_prints_usage(self, settings_file, capsys) -> None:
        assert cli._routing_command('maybe') is None
        assert cli._ROUTING_USAGE in capsys.readouterr().out

    def test_switch_without_table_is_one_line(self, settings_file, capsys):
        settings_file.write_text('[tools]\nflat = true\n', encoding='utf-8')
        assert cli._routing_command('off') is None
        out = ' '.join(capsys.readouterr().out.split())
        assert 'no [routing] table' in out
