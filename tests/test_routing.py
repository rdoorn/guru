"""Pure routing rules (guru.domain.routing): labels, filters, fallbacks."""
import dataclasses

import pytest

from guru.domain import routing
from guru.domain.routing import Ladder, Route, Rung, normalise_labels, resolve

L4 = Rung('Ollama', 'qwen3:4b', 'trivial', remote=False)
L14 = Rung('Ollama', 'qwen3:14b', 'standard', remote=False, default=True)
R_SONNET = Rung('Anthropic', 'claude-sonnet-5', 'hard', remote=True)
R_OPUS = Rung('Anthropic', 'claude-opus-5', 'hard', remote=True)
MAIN = Rung('Ollama', 'main-model', 'hard', remote=False)
R_MAIN = Rung('Anthropic', 'remote-main', 'hard', remote=True)

DEFAULT = Ladder([L4, L14, R_SONNET])
REVIEW = Ladder([R_OPUS])
LADDERS = {'default': DEFAULT, 'review': REVIEW}


def _resolve(kind='build', complexity='standard', ladders=LADDERS, *,
             mode='local-and-remote', scan_findings=0,
             confirmation='granted', complexity_router=True,
             type_router=False, local_main=MAIN) -> Route:
    return resolve(kind, complexity, ladders, mode=mode,
                   scan_findings=scan_findings, confirmation=confirmation,
                   complexity_router=complexity_router,
                   type_router=type_router, local_main=local_main)


class TestConstants:
    def test_complexity_and_kinds(self) -> None:
        assert routing.COMPLEXITY == ('trivial', 'standard', 'hard')
        assert routing.KINDS == ('debug', 'build', 'refactor', 'review',
                                 'explain', 'docs', 'ops', 'other')
        assert routing.MODES == ('local-only', 'local-and-remote',
                                 'remote-only')
        assert routing.CONFIRMATIONS == ('granted', 'declined', 'pending',
                                         'never')

    def test_rung_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            L4.model = 'x'  # type: ignore[misc]


class TestNormaliseLabels:
    @pytest.mark.parametrize('kind, complexity, expected', [
        ('debug', 'hard', ('debug', 'hard')),
        ('Review', 'TRIVIAL', ('review', 'trivial')),
        ('  docs ', ' standard ', ('docs', 'standard')),
        ('poetry', 'standard', ('other', 'standard')),
        ('build', 'insane', ('build', 'standard')),
        ('', '', ('other', 'standard')),
        (None, None, ('other', 'standard')),
        (42, 3.5, ('other', 'standard')),
    ])
    def test_table(self, kind, complexity, expected) -> None:
        assert normalise_labels(kind, complexity) == expected


class TestComplexityChoice:
    @pytest.mark.parametrize('complexity, model, idx', [
        ('trivial', 'qwen3:4b', 0),
        ('standard', 'qwen3:14b', 1),
        ('hard', 'claude-sonnet-5', 2),
    ])
    def test_lowest_rung_that_fits(self, complexity, model, idx) -> None:
        r = _resolve(complexity=complexity)
        assert (r.adapter, r.model, r.rung_index, r.refused) == (
            'Anthropic' if idx == 2 else 'Ollama', model, idx, False)
        assert r.reason == []

    def test_unknown_labels_normalised(self) -> None:
        r = _resolve(kind='poetry', complexity='insane')
        assert r.model == 'qwen3:14b'
        assert r.kind == 'other' and r.complexity == 'standard'

    def test_router_off_picks_default_rung(self) -> None:
        r = _resolve(complexity='hard', complexity_router=False)
        assert (r.model, r.rung_index) == ('qwen3:14b', 1)
        assert r.reason == []

    def test_router_off_no_default_marked_picks_first(self) -> None:
        ladders = {'default': Ladder([L4, R_SONNET])}
        r = _resolve(ladders=ladders, complexity='hard',
                     complexity_router=False)
        assert (r.model, r.rung_index) == ('qwen3:4b', 0)

    def test_route_carries_ladder_name(self) -> None:
        assert _resolve().ladder == 'default'


class TestTypeRouter:
    def test_off_ignores_kind_ladder(self) -> None:
        r = _resolve(kind='review', complexity='hard')
        assert (r.ladder, r.model) == ('default', 'claude-sonnet-5')

    def test_on_uses_kind_ladder(self) -> None:
        r = _resolve(kind='review', complexity='hard', type_router=True)
        assert (r.ladder, r.model, r.rung_index) == (
            'review', 'claude-opus-5', 0)
        assert r.reason == []

    def test_on_without_kind_ladder_falls_to_default(self) -> None:
        r = _resolve(kind='docs', complexity='hard', type_router=True)
        assert (r.ladder, r.model) == ('default', 'claude-sonnet-5')
        assert r.reason == []

    def test_kind_ladder_lower_than_complexity_climbs_to_top(self) -> None:
        # No rung in the kind ladder fits: the highest surviving rung of that
        # ladder is used rather than silently switching ladders.
        ladders = {'default': DEFAULT, 'docs': Ladder([L4])}
        r = _resolve(kind='docs', complexity='hard', type_router=True,
                     ladders=ladders)
        assert (r.ladder, r.model, r.rung_index) == ('docs', 'qwen3:4b', 0)
        assert 'complexity:hard exceeds ladder; using top rung' in r.reason


class TestModeFilters:
    def test_local_only_strips_remote(self) -> None:
        r = _resolve(complexity='hard', mode='local-only')
        assert (r.adapter, r.model, r.rung_index) == (
            'Ollama', 'qwen3:14b', 1)
        assert ('mode:local-only stripped 1 remote rung (ladder default)'
                in r.reason)
        assert 'complexity:hard exceeds ladder; using top rung' in r.reason

    def test_local_only_no_change_no_reason(self) -> None:
        ladders = {'default': Ladder([L4, L14])}
        r = _resolve(ladders=ladders, mode='local-only')
        assert r.reason == []

    def test_remote_only_strips_local(self) -> None:
        r = _resolve(complexity='trivial', mode='remote-only')
        assert (r.adapter, r.model, r.rung_index) == (
            'Anthropic', 'claude-sonnet-5', 2)
        assert ('mode:remote-only stripped 2 local rungs (ladder default)'
                in r.reason)

    def test_rung_index_refers_to_original_ladder(self) -> None:
        r = _resolve(complexity='trivial', mode='remote-only')
        assert LADDERS['default'].rungs[r.rung_index] is R_SONNET

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match='mode'):
            _resolve(mode='hybrid')


class TestScanFilter:
    def test_finding_forces_local(self) -> None:
        r = _resolve(complexity='hard', scan_findings=1)
        assert (r.model, r.rung_index) == ('qwen3:14b', 1)
        assert 'scan:1 finding forced local (ladder default)' in r.reason

    def test_findings_plural(self) -> None:
        r = _resolve(complexity='hard', scan_findings=3)
        assert 'scan:3 findings forced local (ladder default)' in r.reason

    def test_finding_with_no_remote_rungs_no_reason(self) -> None:
        ladders = {'default': Ladder([L4, L14])}
        r = _resolve(ladders=ladders, scan_findings=2)
        assert r.reason == []

    def test_finding_after_local_only_not_double_counted(self) -> None:
        r = _resolve(complexity='hard', mode='local-only', scan_findings=1)
        assert ('mode:local-only stripped 1 remote rung (ladder default)'
                in r.reason)
        assert not [x for x in r.reason if x.startswith('scan:')]


class TestConfirmation:
    def test_granted_is_silent(self) -> None:
        r = _resolve(complexity='hard', confirmation='granted')
        assert (r.model, r.reason) == ('claude-sonnet-5', [])

    def test_never_is_silent(self) -> None:
        r = _resolve(complexity='hard', confirmation='never')
        assert (r.model, r.reason) == ('claude-sonnet-5', [])

    def test_declined_strips_remote(self) -> None:
        r = _resolve(complexity='hard', confirmation='declined')
        assert (r.model, r.rung_index) == ('qwen3:14b', 1)
        assert 'confirmation:declined (ladder default)' in r.reason

    def test_declined_with_no_remote_pick_is_silent(self) -> None:
        ladders = {'default': Ladder([L4, L14])}
        r = _resolve(ladders=ladders, confirmation='declined')
        assert r.reason == []

    def test_pending_computes_as_granted_and_flags(self) -> None:
        r = _resolve(complexity='hard', confirmation='pending')
        assert (r.model, r.rung_index) == ('claude-sonnet-5', 2)
        assert r.reason == ['needs_confirmation']
        assert r.needs_confirmation is True

    def test_pending_with_local_pick_is_silent(self) -> None:
        r = _resolve(complexity='trivial', confirmation='pending')
        assert r.reason == []
        assert r.needs_confirmation is False

    def test_pending_after_scan_strip_is_silent(self) -> None:
        # A finding already forced local: nothing remote is left to confirm.
        r = _resolve(complexity='hard', confirmation='pending',
                     scan_findings=1)
        assert 'needs_confirmation' not in r.reason

    def test_unknown_confirmation_raises(self) -> None:
        with pytest.raises(ValueError, match='confirmation'):
            _resolve(confirmation='maybe')


class TestFallbacks:
    def test_no_ladders_uses_local_main(self) -> None:
        r = _resolve(ladders={})
        assert (r.adapter, r.model, r.rung_index, r.ladder) == (
            'Ollama', 'main-model', None, '')
        assert routing.LOCAL_MAIN_TAKEN in r.reason
        assert r.refused is False

    def test_local_only_empties_ladder_uses_local_main(self) -> None:
        ladders = {'default': Ladder([R_SONNET])}
        r = _resolve(ladders=ladders, mode='local-only')
        assert r.model == 'main-model'
        assert r.reason == [
            'mode:local-only stripped 1 remote rung (ladder default)',
            routing.LOCAL_MAIN_TAKEN]

    def test_remote_local_main_skipped_only_by_local_only(self) -> None:
        # The parent runs on a remote model: an explicit local-only mode is
        # the one filter that skips it; the next step (any ladder with a
        # local rung) is taken instead.
        ladders = {'default': Ladder([R_SONNET]), 'review': Ladder([L4])}
        r = _resolve(ladders=ladders, local_main=R_MAIN, mode='local-only')
        assert (r.adapter, r.model, r.ladder, r.refused) == (
            'Ollama', 'qwen3:4b', 'review', False)
        assert routing.LOCAL_MAIN_SKIPPED in r.reason
        assert routing.LOCAL_MAIN_TAKEN not in r.reason
        assert r.reason[-1] == 'fallback:ladder review'

    def test_remote_local_main_skipped_then_refused(self) -> None:
        r = _resolve(ladders={'default': Ladder([R_SONNET])},
                     local_main=R_MAIN, mode='local-only')
        assert r.refused is True
        assert routing.LOCAL_MAIN_SKIPPED in r.reason
        assert r.reason[-1].startswith('refused:')

    @pytest.mark.parametrize('filters', [
        {'scan_findings': 2},
        {'confirmation': 'declined'},
        {'scan_findings': 1, 'confirmation': 'declined'},
    ])
    def test_remote_local_main_is_pre_approved(self, filters) -> None:
        # Decision A: the parent's own model already saw the task text and
        # is already paid for — findings and a decline never skip it.
        r = _resolve(ladders={'default': Ladder([R_SONNET])},
                     local_main=R_MAIN, **filters)
        assert (r.model, r.refused) == ('remote-main', False)
        assert r.reason[-1] == routing.LOCAL_MAIN_TAKEN
        assert routing.NEEDS_CONFIRMATION not in r.reason

    def test_remote_local_main_never_needs_confirmation(self) -> None:
        r = _resolve(ladders={}, local_main=R_MAIN, confirmation='pending')
        assert (r.model, r.refused) == ('remote-main', False)
        assert r.reason == [routing.LOCAL_MAIN_TAKEN]
        assert r.needs_confirmation is False

    @pytest.mark.parametrize('filters', [
        {'mode': 'local-only'},
        {'scan_findings': 1},
        {'confirmation': 'declined'},
    ])
    def test_local_local_main_still_taken(self, filters) -> None:
        r = _resolve(ladders={'default': Ladder([R_SONNET])},
                     local_main=MAIN, **filters)
        assert (r.model, r.refused) == ('main-model', False)
        assert r.reason[-1] == routing.LOCAL_MAIN_TAKEN

    def test_type_router_kind_ladder_emptied_falls_to_default(self) -> None:
        r = _resolve(kind='review', complexity='hard', type_router=True,
                     mode='local-only')
        assert (r.ladder, r.model) == ('default', 'qwen3:14b')
        assert r.reason == [
            'mode:local-only stripped 1 remote rung (ladder review)',
            'mode:local-only stripped 1 remote rung (ladder default)',
            'fallback:ladder default',
            'complexity:hard exceeds ladder; using top rung']

    def test_no_local_main_first_surviving_rung_of_any_ladder(self) -> None:
        ladders = {'default': Ladder([R_SONNET]), 'review': Ladder([L4])}
        r = _resolve(ladders=ladders, mode='local-only', local_main=None)
        assert (r.adapter, r.model, r.ladder, r.rung_index) == (
            'Ollama', 'qwen3:4b', 'review', 0)
        assert r.reason == [
            'mode:local-only stripped 1 remote rung (ladder default)',
            'fallback:ladder review']

    def test_any_ladder_fallback_records_that_ladders_filters(self) -> None:
        ladders = {'default': Ladder([R_SONNET]),
                   'review': Ladder([R_OPUS, L4]),
                   'docs': Ladder([R_OPUS])}
        r = _resolve(ladders=ladders, mode='local-only', local_main=None)
        assert (r.ladder, r.model, r.rung_index) == ('review', 'qwen3:4b', 1)
        # The emptied docs ladder is not mentioned; the chosen one is.
        assert r.reason == [
            'mode:local-only stripped 1 remote rung (ladder default)',
            'mode:local-only stripped 1 remote rung (ladder review)',
            'fallback:ladder review']

    def test_local_main_ignored_in_remote_only(self) -> None:
        ladders = {'default': Ladder([L4]), 'review': Ladder([R_OPUS])}
        r = _resolve(ladders=ladders, mode='remote-only')
        assert r.model == 'claude-opus-5'
        assert 'fallback:local_main' not in r.reason

    def test_remote_only_plus_finding_refuses(self) -> None:
        r = _resolve(complexity='hard', mode='remote-only', scan_findings=1)
        assert r.refused is True
        assert (r.adapter, r.model, r.rung_index) == ('', '', None)
        assert ('mode:remote-only stripped 2 local rungs (ladder default)'
                in r.reason)
        assert 'scan:1 finding forced local (ladder default)' in r.reason
        assert r.reason[-1].startswith('refused:')

    def test_remote_only_plus_declined_refuses(self) -> None:
        r = _resolve(complexity='hard', mode='remote-only',
                     confirmation='declined')
        assert r.refused is True
        assert 'confirmation:declined (ladder default)' in r.reason

    def test_nothing_configured_no_main_refuses(self) -> None:
        r = _resolve(ladders={}, local_main=None)
        assert r.refused is True
        assert r.reason[-1].startswith('refused:')

    def test_inputs_not_mutated(self) -> None:
        ladders = {'default': Ladder([L4, L14, R_SONNET])}
        _resolve(ladders=ladders, mode='local-only', complexity='hard')
        assert ladders['default'].rungs == [L4, L14, R_SONNET]


class TestRouteRecord:
    def test_as_dict_is_json_ready(self) -> None:
        r = _resolve(complexity='hard', mode='local-only')
        d = r.as_dict()
        assert d['adapter'] == 'Ollama'
        assert d['model'] == 'qwen3:14b'
        assert d['rung_index'] == 1
        assert d['ladder'] == 'default'
        assert d['refused'] is False
        assert isinstance(d['reason'], list)
