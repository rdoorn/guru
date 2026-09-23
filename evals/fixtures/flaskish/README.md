# Flaskish

A minimal file-upload service written against the standard library only, in
the style of a small Flask app but without the framework: handlers take a
request dict and return `(status, body)`.

## Endpoints

- `POST /login` with `{"user": "<name>"}` returns a session token
  (`X-Token` header on later requests). Tokens live one hour.
- `PUT /upload/<name>` stores the request body under
  `/srv/uploads/<user>/<name>`. `<name>` may contain sub-directories.
- `GET /whoami` returns the user owning the token.

## Layout

- `app/session.py` - token creation, lookup, expiry.
- `app/upload.py` - writing and reading stored files.
- `app/handlers.py` - the three endpoints; authentication lives in
  `_authenticate`.
- `tests/` - happy-path tests; run with `python -m pytest -q`.
