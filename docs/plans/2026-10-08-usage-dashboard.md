# Usage dashboard (branch `opus`) — reference implementation

Approved design (2026-10-08). The full original request: an on-disk usage
store, a localhost dashboard, and one server among many gurus.

## Decisions (from the user)
- Scope: the whole feature. Topic: a short generated label per user turn,
  by the cheapest routed rung (fallback: the request text). Serving: every
  guru tries the port; the first wins; the others show where it is and
  retry. Frontend: stdlib only, SVG charts. Port 7340, configurable.
  Views: overview + cost per day, per model/project/topic, recent calls,
  time range filter. History: start empty. Evals: recorded, marked
  `source = eval`. Retention: forever.
- Done: `make check` green, the acceptance probe passes (main-agent topic,
  sub-agent grouping, 3 concurrent processes, server election), an
  independent review on the same 5 dimensions as guru's branches.

## Layers
- `guru/domain/usage.py`: topic text (redact then cap), label prompt and
  parser, time ranges, the `UsageQueries` and `TopicLabeler` Protocols,
  the topic lifecycle (`begin_topic`: `session.topic_id`, the `topics`
  stream, the background label).
- `guru/domain/ledger.py`: calls and tasks carry `topic_id`.
- `guru/repositories/usage_sqlite.py`: `SqliteUsage` (calls, topics,
  tasks; WAL, 1 s busy timeout, busy -> drop row, other errors -> disable;
  migrations by column; 0600 file, 0700 new dir) and its queries.
- `guru/repositories/fanout.py`: `FanOutLedger` (JSONL + usage store).
- `guru/judges/topic.py`: `RoutedTopicLabeler` (cheapest rung through the
  routing rules; no rung -> no label).
- `guru/dashboard/`: stdlib HTTP server on 127.0.0.1 (Host check,
  GET only, quiet log), the single-server election and retry thread, one
  HTML page.
- Wiring: `cli.py` (store, labeler, dashboard startup step, `/dashboard`),
  `adapters/turn.py` (begin a topic on a user turn), `orchestrator.py`
  (children inherit `topic_id`), eval runner (`source = eval`).
- Settings: `[ledger] usage_db`, `[dashboard] enabled`, `port`,
  `topic_labels`; `GURU_USAGE_DB`. All in `docs/defaults.md`.
