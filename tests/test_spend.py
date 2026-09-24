"""Tests for the once-per-run spend confirmation (guru.domain.spend)."""
import pytest

from guru.domain import spend


@pytest.fixture(autouse=True)
def _fresh_state():
    spend.reset()
    spend.set_spend_asker(None)
    yield
    spend.reset()
    spend.set_spend_asker(None)


class TestSpendConfirmation:
    """confirmation(mode) maps the spend_confirm setting + the remembered
    answer to a routing confirmation value."""

    def test_never_and_auto_do_not_ask(self) -> None:
        calls: list = []
        spend.set_spend_asker(lambda q: calls.append(q) or True)
        assert spend.confirmation('never') == 'never'
        assert spend.confirmation('auto') == 'granted'
        assert calls == []
        assert spend.status('never') == 'never'
        assert spend.status('auto') == 'granted'

    def test_status_peeks_without_asking(self) -> None:
        calls: list = []
        spend.set_spend_asker(lambda q: calls.append(q) or True)
        assert spend.status('ask') == 'pending'
        assert calls == []

    def test_ask_asks_once_and_remembers_grant(self) -> None:
        calls: list = []
        spend.set_spend_asker(lambda q: calls.append(q) or True)
        assert spend.confirmation('ask') == 'granted'
        assert spend.confirmation('ask') == 'granted'
        assert spend.status('ask') == 'granted'
        assert len(calls) == 1 and 'remote' in calls[0].lower()

    def test_ask_remembers_decline(self) -> None:
        calls: list = []
        spend.set_spend_asker(lambda q: calls.append(q) or False)
        assert spend.confirmation('ask') == 'declined'
        assert spend.confirmation('ask') == 'declined'
        assert len(calls) == 1

    def test_default_asker_denies(self) -> None:
        assert spend.confirmation('ask') == 'declined'

    def test_raising_asker_is_a_decline(self) -> None:
        def boom(q: str) -> bool:
            raise RuntimeError('tty gone')
        spend.set_spend_asker(boom)
        assert spend.confirmation('ask') == 'declined'

    def test_reset_forgets_the_answer(self) -> None:
        spend.set_spend_asker(lambda q: True)
        assert spend.confirmation('ask') == 'granted'
        spend.reset()
        assert spend.status('ask') == 'pending'

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            spend.confirmation('sometimes')
