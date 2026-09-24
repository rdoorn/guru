# Fixture: docs-only

Frozen eval fixture with no code and no tests: two Markdown files about an
imaginary deployment procedure.

## Planted facts

1. `README.md` has a section titled **Rollback** with a five-step
   procedure: pause the rollout, redeploy the previous tag (about four
   minutes), do not roll back the database, confirm on the dashboards and
   announce, open a post-mortem within a day. Trigger: error rate above 1%
   or p99 regression above 300 ms.
2. `CHANGELOG.md` lists v2.4.1 (2026-09-10) as the latest release.

## Behaviour of the fixture's own tests

There are none; `python -m pytest -q` collects nothing.

## Expected in a correct answer

- "Summarise the rollback procedure" reads `README.md` and mentions
  rollback, the previous tag and that the database is not rolled back.
- Greetings and general questions ("what does `ls -la` do?") need no file
  tool at all.
