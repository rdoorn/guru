"""The usage store: every model call, the topic of each user request and
each sub-agent task, in one SQLite file shared by every guru process
(``config.USAGE_DB_PATH``, default ``~/.guru/usage.db``).

A ``LedgerRepository`` for the ``calls``, ``topics`` and ``tasks``
streams (other streams are ignored), installed next to the JSONL ledger
through ``guru.repositories.fanout.FanOutLedger``; and the dashboard's
``UsageQueries``.

Several gurus write at once: WAL mode, one short transaction per row, a
``BUSY_TIMEOUT_S`` wait. Telemetry never breaks or stalls a run: a row
that meets a busy or locked database is dropped (logged), any other error
disables the store for this process (logged once); writes come from the
ledger's single worker thread, never the caller's. Schema changes are
additive: missing columns are added on open, so gurus of different ages
can share the file, and ``user_version`` never goes down. The file is
0600; a directory it creates is 0700.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from guru import log
from guru.domain import usage

SCHEMA_VERSION = 1
BUSY_TIMEOUT_S = 1.0
OPEN_ATTEMPTS = 8           # ~0.7 s of retries for a contended first open

# table -> [(column, declaration)]; the first column is the key.
TABLES: dict = {
    'calls': [
        ('id', 'INTEGER PRIMARY KEY AUTOINCREMENT'),
        ('ts', 'TEXT NOT NULL'), ('source', 'TEXT'), ('run_id', 'TEXT'),
        ('project', 'TEXT'), ('project_path', 'TEXT'), ('agent', 'TEXT'),
        ('task_id', 'TEXT'), ('turn_id', 'TEXT'), ('topic_id', 'TEXT'),
        ('adapter', 'TEXT'), ('model', 'TEXT'), ('phase', 'TEXT'),
        ('tokens_in', 'INTEGER'), ('tokens_out', 'INTEGER'),
        ('cache_read', 'INTEGER'), ('cache_write', 'INTEGER'),
        ('seconds', 'REAL'), ('cost_usd', 'REAL'), ('cost_source', 'TEXT'),
    ],
    'topics': [
        ('topic_id', 'TEXT PRIMARY KEY'), ('ts', 'TEXT'), ('source', 'TEXT'),
        ('project', 'TEXT'), ('project_path', 'TEXT'), ('request', 'TEXT'),
        ('label', 'TEXT'),
    ],
    'tasks': [
        ('task_id', 'TEXT PRIMARY KEY'), ('ts', 'TEXT'), ('source', 'TEXT'),
        ('topic_id', 'TEXT'), ('turn_id', 'TEXT'), ('kind', 'TEXT'),
        ('goal', 'TEXT'), ('status', 'TEXT'), ('model', 'TEXT'),
        ('cost_usd', 'REAL'),
    ],
}
INDEXES = (
    ('calls_ts', 'calls', 'ts'), ('calls_topic', 'calls', 'topic_id'),
    ('calls_project', 'calls', 'project'), ('calls_model', 'calls', 'model'),
)
_CALL_COLUMNS = [c for c, _ in TABLES['calls'][1:]]
_INT_COLUMNS = {'tokens_in', 'tokens_out', 'cache_read', 'cache_write'}
_REAL_COLUMNS = {'seconds', 'cost_usd'}
_TRANSIENT = ('locked', 'busy')


def _number(value: object, kind: type) -> Optional[float]:
    """``value`` as int/float for a numeric column, else None."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return kind(value)
    except (TypeError, ValueError):
        return None


def _text(value: object) -> Optional[str]:
    return None if value is None else str(value)


class SqliteUsage:
    """The usage store at ``path``; rows written are marked ``source``
    (``cli`` for interactive guru, ``eval`` for the eval runner)."""

    def __init__(self, path: Path, source: str = 'cli') -> None:
        self.path = Path(path)
        self.source = source
        self.disabled = False
        self._local = threading.local()
        self._lock = threading.Lock()
        self._ready = False

    # --- connection ----------------------------------------------------------

    def _prepare_file(self) -> None:
        parent = self.path.parent
        if not parent.exists():
            parent.mkdir(mode=0o700, parents=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        if os.stat(self.path).st_mode & 0o077:
            os.chmod(self.path, 0o600)       # our own file: never readable

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            return conn
        with self._lock:
            if not self._ready:
                self._prepare_file()
        conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_S,
                               isolation_level=None)
        conn.execute(f'PRAGMA busy_timeout = {int(BUSY_TIMEOUT_S * 1000)}')
        with self._lock:
            if not self._ready:
                self._open_retrying(conn)
                self._ready = True
        self._local.conn = conn
        return conn

    def _open_retrying(self, conn: sqlite3.Connection) -> None:
        """Migrate, retrying a contended first open: several gurus that
        create the file at once race for the WAL switch and the schema,
        which do not wait on the busy handler. Bounded by
        ``OPEN_ATTEMPTS`` short sleeps; a lasting lock still raises."""
        for attempt in range(OPEN_ATTEMPTS):
            try:
                self._migrate(conn)
                return
            except sqlite3.OperationalError as e:
                if (attempt == OPEN_ATTEMPTS - 1 or not any(
                        w in str(e).lower() for w in _TRANSIENT)):
                    raise
                time.sleep(0.02 * (attempt + 1))

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Create what is missing — tables, columns, indexes — and raise
        ``user_version`` to ``SCHEMA_VERSION`` (never lower it)."""
        if conn.execute('PRAGMA journal_mode = WAL').fetchone()[0] != 'wal':
            log.info('usage store: WAL unavailable for %s', self.path)
        conn.execute('PRAGMA synchronous = NORMAL')
        conn.execute('BEGIN IMMEDIATE')
        try:
            for table, cols in TABLES.items():
                decl = ', '.join(f'{c} {d}' for c, d in cols)
                conn.execute(f'CREATE TABLE IF NOT EXISTS {table} ({decl})')
                have = {r[1] for r in conn.execute(
                    f'PRAGMA table_info({table})')}
                for col, d in cols:
                    if col not in have:
                        kind = d.split()[0]
                        conn.execute(f'ALTER TABLE {table} ADD COLUMN'
                                     f' {col} {kind}')
            for name, table, col in INDEXES:
                conn.execute(f'CREATE INDEX IF NOT EXISTS {name} ON'
                             f' {table}({col})')
            version = conn.execute('PRAGMA user_version').fetchone()[0]
            if version < SCHEMA_VERSION:
                conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
            conn.execute('COMMIT')
        except BaseException:
            if conn.in_transaction:      # BEGIN itself may be what failed
                conn.execute('ROLLBACK')
            raise

    def close(self) -> None:
        """Close this thread's connection (others close at exit)."""
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # --- writing (LedgerRepository) ------------------------------------------

    def append(self, stream: str, row: dict) -> None:
        """Store a ``calls``, ``topics`` or ``tasks`` row; ignore the rest.
        Never raises."""
        if self.disabled or stream not in ('calls', 'topics', 'tasks'):
            return
        try:
            conn = self._conn()
            getattr(self, f'_write_{stream}')(conn, row)
        except sqlite3.OperationalError as e:
            if any(word in str(e).lower() for word in _TRANSIENT):
                log.warning('usage store busy; %s row dropped', stream)
                return
            self._disable(e)
        except (sqlite3.Error, OSError) as e:
            self._disable(e)

    def _disable(self, e: Exception) -> None:
        self.disabled = True
        log.warning('usage store disabled (%s): %s', self.path, e)

    def _common(self, row: dict) -> dict:
        return {'source': self.source, 'project': _text(row.get('project')),
                'project_path': _cwd()}

    def _write_calls(self, conn: sqlite3.Connection, row: dict) -> None:
        values = {**self._common(row)}
        for col in _CALL_COLUMNS:
            if col in values:
                continue
            raw = row.get(col)
            if col in _INT_COLUMNS:
                values[col] = _number(raw, int)
            elif col in _REAL_COLUMNS:
                values[col] = _number(raw, float)
            else:
                values[col] = _text(raw)
        cols = ', '.join(_CALL_COLUMNS)
        marks = ', '.join('?' for _ in _CALL_COLUMNS)
        conn.execute(f'INSERT INTO calls ({cols}) VALUES ({marks})',
                     [values[c] for c in _CALL_COLUMNS])

    def _write_topics(self, conn: sqlite3.Connection, row: dict) -> None:
        topic_id = _text(row.get('topic_id'))
        if not topic_id:
            return
        common = self._common(row)
        conn.execute(
            'INSERT INTO topics (topic_id, ts, source, project, project_path,'
            ' request, label) VALUES (?, ?, ?, ?, ?, ?, ?)'
            ' ON CONFLICT(topic_id) DO UPDATE SET'
            " request = CASE WHEN excluded.request <> ''"
            ' THEN excluded.request ELSE topics.request END,'
            " label = CASE WHEN excluded.label <> ''"
            ' THEN excluded.label ELSE topics.label END',
            (topic_id, _text(row.get('ts')), common['source'],
             common['project'], common['project_path'],
             usage.topic_text(str(row.get('request') or '')),
             _text(row.get('label')) or ''))

    def _write_tasks(self, conn: sqlite3.Connection, row: dict) -> None:
        task_id = _text(row.get('task_id'))
        if not task_id:
            return
        goal = usage.topic_text(str(row.get('task') or ''))
        conn.execute(
            'INSERT INTO tasks (task_id, ts, source, topic_id, turn_id, kind,'
            ' goal, status, model, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?,'
            ' ?, ?) ON CONFLICT(task_id) DO UPDATE SET'
            ' status = excluded.status, model = excluded.model,'
            ' cost_usd = excluded.cost_usd',
            (task_id, _text(row.get('ts')), self.source,
             _text(row.get('topic_id')), _text(row.get('turn_id')),
             _text(row.get('kind')), goal, _text(row.get('status')),
             _text(row.get('model')), _number(row.get('cost_usd'), float)))

    # --- reading (UsageQueries) ----------------------------------------------

    def _where(self, since: Optional[str], source: str,
               alias: str = 'c') -> tuple:
        clauses, args = [], []
        if since:
            clauses.append(f'{alias}.ts >= ?')
            args.append(since)
        if source != 'all':
            clauses.append(f'{alias}.source = ?')
            args.append(source)
        where = (' WHERE ' + ' AND '.join(clauses)) if clauses else ''
        return where, args

    def _query(self, sql: str, args: list) -> list:
        if self.disabled or not self.path.exists():
            return []
        cur = self._conn().execute(sql, args)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def totals(self, since: Optional[str], source: str = 'all') -> dict:
        where, args = self._where(since, source)
        rows = self._query(
            'SELECT COUNT(*) AS calls, COALESCE(SUM(c.cost_usd), 0) AS cost,'
            ' SUM(c.cost_usd IS NULL) AS unpriced,'
            ' COALESCE(SUM(c.tokens_in), 0) AS tokens_in,'
            ' COALESCE(SUM(c.tokens_out), 0) AS tokens_out,'
            " COUNT(DISTINCT NULLIF(c.topic_id, '')) AS topics,"
            " COUNT(DISTINCT NULLIF(c.project, '')) AS projects,"
            ' COUNT(DISTINCT c.model) AS models'
            f' FROM calls c{where}', args)
        total = rows[0] if rows else {}
        return {k: (total.get(k) or 0) for k in (
            'calls', 'cost', 'unpriced', 'tokens_in', 'tokens_out',
            'topics', 'projects', 'models')}

    def daily(self, since: Optional[str], source: str = 'all') -> list:
        """Cost, tokens and calls per local day and model."""
        where, args = self._where(since, source)
        return self._query(
            "SELECT date(c.ts, 'localtime') AS day, c.model AS model,"
            ' COALESCE(SUM(c.cost_usd), 0) AS cost, COUNT(*) AS calls,'
            ' COALESCE(SUM(c.tokens_in), 0) + COALESCE(SUM(c.tokens_out), 0)'
            f' AS tokens FROM calls c{where} GROUP BY day, model'
            ' ORDER BY day, model', args)

    def groups(self, by: str, since: Optional[str], source: str = 'all',
               limit: int = 50) -> list:
        """Cost, tokens and calls grouped ``by`` model, project or topic,
        costliest first. A topic is its label, else its request text."""
        if by not in usage.GROUPS:
            raise ValueError(f'by must be one of {", ".join(usage.GROUPS)}')
        key = {'model': 'c.model', 'project': 'c.project',
               'topic': "COALESCE(NULLIF(t.label, ''), NULLIF(t.request, ''),"
                        " '(no topic)')"}[by]
        where, args = self._where(since, source)
        return self._query(
            f'SELECT {key} AS key, COALESCE(SUM(c.cost_usd), 0) AS cost,'
            ' SUM(c.cost_usd IS NULL) AS unpriced, COUNT(*) AS calls,'
            ' COALESCE(SUM(c.tokens_in), 0) AS tokens_in,'
            ' COALESCE(SUM(c.tokens_out), 0) AS tokens_out,'
            ' MAX(c.ts) AS last'
            ' FROM calls c LEFT JOIN topics t ON t.topic_id = c.topic_id'
            f'{where} GROUP BY key ORDER BY cost DESC, calls DESC LIMIT ?',
            args + [max(1, min(int(limit), 500))])

    def calls(self, since: Optional[str], source: str = 'all',
              limit: int = 50, offset: int = 0) -> list:
        """The latest calls, newest first, with their topic."""
        where, args = self._where(since, source)
        return self._query(
            'SELECT c.ts, c.source, c.project, c.agent, c.adapter, c.model,'
            ' c.phase, c.tokens_in, c.tokens_out, c.cost_usd,'
            " COALESCE(NULLIF(t.label, ''), t.request, '') AS topic,"
            ' k.goal AS task'
            ' FROM calls c LEFT JOIN topics t ON t.topic_id = c.topic_id'
            ' LEFT JOIN tasks k ON k.task_id = c.task_id'
            f'{where} ORDER BY c.ts DESC, c.id DESC LIMIT ? OFFSET ?',
            args + [max(1, min(int(limit), 500)), max(0, int(offset))])


def _cwd() -> Optional[str]:
    try:
        return os.getcwd()
    except OSError:
        return None


def wait_ready(store: SqliteUsage, timeout_s: float = 2.0) -> bool:
    """Open ``store`` now (tests, the startup step); False when it is
    disabled or could not open within ``timeout_s``."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            store._conn()
            return not store.disabled
        except sqlite3.OperationalError as e:
            if not any(w in str(e).lower() for w in _TRANSIENT):
                store._disable(e)
                return False
            time.sleep(0.05)
        except (sqlite3.Error, OSError) as e:
            store._disable(e)
            return False
    return False
