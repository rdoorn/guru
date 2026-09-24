# Fixture: flaskish

Frozen eval fixture. Do not change it except by a deliberate commit; the
eval cases and their expected answers depend on the facts below.

## Planted facts

1. **Path traversal** in `app/upload.py::save_upload`: the user-controlled
   `user_path` is joined with `os.path.join(base_dir, user_path)` and never
   normalised or checked to stay below `base_dir`. `../` segments (or an
   absolute `user_path`, which `os.path.join` lets replace the base) write
   outside the upload root. Reached from `app/handlers.py::handle_upload`.
2. **Swapped comparison** in `app/session.py::is_expired`: it returns
   `now < token['expires_at']`, i.e. True for tokens that are still valid
   and False for expired ones. Every authenticated handler therefore
   rejects fresh tokens and accepts expired ones.

## Behaviour of the fixture's own tests

`python -m pytest -q` passes (5 tests). The tests only cover the happy
path, so a correct fix of either bug must keep them green; a reviewer who
adds an expiry or traversal test will see it fail against the planted code.

## Expected in a correct answer

- A review names both bugs, with file and function.
- A security-only question about uploads says yes, path traversal, and
  points at the unnormalised `os.path.join` (fix: `os.path.normpath` /
  `os.path.realpath` plus a prefix check, or reject `..` and absolute paths).
- A question about `is_expired` says the operands are swapped (should be
  `token['expires_at'] < now` or equivalently `now >= expires_at`).
