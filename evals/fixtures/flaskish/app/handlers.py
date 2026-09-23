"""HTTP-ish handlers. A request is a dict with ``headers``, ``path`` and
``body``; a response is ``(status, body)``."""
from typing import Optional

from app import session, upload

UPLOAD_ROOT = '/srv/uploads'


def _authenticate(request: dict) -> Optional[dict]:
    token_id = request.get('headers', {}).get('X-Token', '')
    token = session.lookup(token_id)
    if token is None or session.is_expired(token):
        return None
    return token


def handle_login(request: dict) -> tuple:
    """POST /login with ``body['user']``: returns a fresh token."""
    user = request.get('body', {}).get('user')
    if not user:
        return 400, {'error': 'user required'}
    return 200, {'token': session.create(user)['id']}


def handle_upload(request: dict, root: str = UPLOAD_ROOT) -> tuple:
    """PUT /upload/<name>: store the body under the caller's directory."""
    token = _authenticate(request)
    if token is None:
        return 401, {'error': 'invalid or expired token'}
    name = request['path'].removeprefix('/upload/')
    stored = upload.save_upload(root, f"{token['user']}/{name}",
                                request.get('body', b''))
    return 201, {'stored': stored}


def handle_whoami(request: dict) -> tuple:
    """GET /whoami: the user behind the token."""
    token = _authenticate(request)
    if token is None:
        return 401, {'error': 'invalid or expired token'}
    return 200, {'user': token['user']}
