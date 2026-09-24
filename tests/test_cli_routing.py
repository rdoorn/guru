"""cli.load_routing: the [routing] table is inert unless configured."""
import pytest

from guru import cli, config
from guru.domain import policy
from guru.repositories import settings as rs


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    monkeypatch.setattr(config, 'SECRET_SCAN', config.SECRET_SCAN)
    yield
    policy.set_scanner(None)


def test_absent_table_leaves_scan_off(monkeypatch) -> None:
    monkeypatch.setattr(rs, 'load_routing', lambda: rs.RoutingSettings())
    routing = cli.load_routing()
    assert routing.present is False
    assert config.SECRET_SCAN is False and policy.scanner() is None


def test_present_table_turns_scan_on_by_default(monkeypatch) -> None:
    monkeypatch.setattr(rs, 'load_routing',
                        lambda: rs.RoutingSettings(present=True))
    routing = cli.load_routing()
    assert routing.secret_scan is True
    assert config.SECRET_SCAN is True and policy.scanner() is not None


def test_present_table_can_turn_scan_off(monkeypatch) -> None:
    monkeypatch.setattr(
        rs, 'load_routing',
        lambda: rs.RoutingSettings(present=True, secret_scan=False))
    cli.load_routing()
    assert config.SECRET_SCAN is False and policy.scanner() is None


def test_invalid_table_warns_and_uses_defaults(monkeypatch) -> None:
    def boom():
        raise ValueError('[routing] mode = "x"')
    monkeypatch.setattr(rs, 'load_routing', boom)
    routing = cli.load_routing()
    assert routing == rs.RoutingSettings()
    assert config.SECRET_SCAN is False and policy.scanner() is None
