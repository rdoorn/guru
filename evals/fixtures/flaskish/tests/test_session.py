"""Happy-path tests for session tokens."""
from app import session


def test_create_sets_user_and_expiry() -> None:
    token = session.create('alice', ttl=60, now=1000.0)
    assert token['user'] == 'alice'
    assert token['expires_at'] == 1060.0
    assert len(token['id']) == 16


def test_lookup_returns_the_created_token() -> None:
    token = session.create('bob')
    assert session.lookup(token['id']) is token
    assert session.lookup('missing') is None


def test_is_expired_returns_a_bool() -> None:
    token = session.create('carol', ttl=60, now=1000.0)
    assert isinstance(session.is_expired(token, now=1030.0), bool)
