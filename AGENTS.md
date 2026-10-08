# Working on guru

Rules for anyone changing this repository, human or agent.

## Layers
- `guru/domain` holds pure rules and entities. It never imports
  `guru.repositories`, `guru.adapters` or other endpoints, not even lazily.
- Persistence (JSONL, SQLite, settings files) lives in `guru/repositories`
  behind a Protocol declared in the domain; external I/O (model providers,
  HF models, scanners) lives in endpoint modules behind Protocols.
- `guru/cli.py` (and the eval runner) install the implementations, e.g.
  `ledger.set_repository(JsonlLedger(config.LEDGER_DIR))`. Without an
  installed implementation the domain does nothing.

## Defaults
- Features are on by default; settings only disable or tweak them.
- Every new setting, env var or code default is listed in
  `docs/defaults.md` in the same change.

## Tests
- Tests never touch the real `~/.guru` or `$HOME`. A new on-disk store gets
  a path override (config value or env var) and an autouse fixture in
  `tests/conftest.py`, next to the existing briefs/skills/log isolation.
- Cover the failure modes: unavailable or locked storage, bad input.
- `make lint`, `make typecheck` and `make test` pass before you report.

## Data guru keeps
- Files guru creates under `~/.guru` are private: files 0600, directories
  0700.
- Text that comes from users or tasks (requests, task goals, tool output)
  is redacted before it is persisted —
  `policy.redact(text, policy.scan(text))` from `guru.domain.policy` — and
  length-capped.
- Telemetry never breaks a run and never stalls it: a transient error drops
  the record (logged), only a structural error disables the sink, and the
  hot path never waits more than about a second.

## Style
- PEP 8, PEP 257, PEP 484; flake8 clean (`make lint`), mypy clean
  (`make typecheck`).
- Package management with `uv`, never `pip`.
- Commit messages: `(fix|feat|BREAKING_CHANGE): <one-line summary>`.
