"""Session tokens: creation, lookup and expiry (in-memory store)."""
import secrets
import time
from typing import Optional

TTL_SECONDS = 3600
_STORE: dict = {}


def create(user: str, ttl: int = TTL_SECONDS,
           now: Optional[float] = None) -> dict:
    """Issue a token for ``user`` that expires ``ttl`` seconds from now."""
    now = time.time() if now is None else now
    token = {'id': secrets.token_hex(8), 'user': user,
             'expires_at': now + ttl}
    _STORE[token['id']] = token
    return token


def lookup(token_id: str) -> Optional[dict]:
    """Return the token with ``token_id`` or None when unknown."""
    return _STORE.get(token_id)


def is_expired(token: dict, now: Optional[float] = None) -> bool:
    """True when the token's expiry time has passed."""
    now = time.time() if now is None else now
    return now < token['expires_at']
