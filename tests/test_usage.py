"""The usage store and its rules: guru.domain.usage (topic text, labels,
ranges, the topic lifecycle), guru.repositories.usage_sqlite and
guru.repositories.fanout."""
from __future__ import annotations

import os
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from guru import config, session
from guru.domain import ledger, policy, usage
from guru.repositories import usage_sqlite
from guru.repositories.fanout import FanOutLedger
from guru.repositories.usage_sqlite import SqliteUsage


def _call(**kw) -> dict:
    row = {'ts': '2026-10-08T10:00:00.000+00:00', 'run_id': 'r',
           'project': 'guru', 'agent': 'main', 'task_id': '',
           'turn_id': 't1', 'topic_id': 't1', 'adapter': 'Anthropic',
           'model': 'claude-sonnet-5', 'tokens_in': 1000, 'tokens_out': 100,
           'cache_read': 0, 'cache_write': 0, 'seconds': 1.0,
           'phase': 'final', 'cost_usd': 0.01, 'cost_source': 'table'}
    row.update(kw)
    return row


@pytest.fixture
def store(tmp_path) -> SqliteUsage:
    s = SqliteUsage(tmp_path / 'g' / 'usage.db')
    yield s
    s.close()


class _Secret:
    def scan(self, text):
        from guru.domain.policy import Finding
        i = text.find('sk-')
        return [Finding('secret', i, i + 12)] if i >= 0 else []


class TestTopicText:
    def test_redacts_before_cutting(self, monkeypatch) -> None:
        policy.set_scanner(_Secret())
        try:
            text = 'x' * (usage.TOPIC_CHARS - 5) + ' sk-123456789 tail'
            out = usage.topic_text(text)
        finally:
            policy.set_scanner(None)
        assert 'sk-1234' not in out
        assert len(out) <= usage.TOPIC_CHARS

    @pytest.mark.parametrize('secret', [
        'sk-ant-api03-abcdefghijklmnopqrstuv', 'sk-DUMMYDUMMY1234',
        'ghp_abcdefghijklmnopqrstuvwxyz0123', 'xoxb-1234567890-abcdef',
        'AKIAIOSFODNN7EXAMPLE', 'password=hunter2', 'API_KEY: abc123',
        'token=eyJhbGciOiJIUzI1NiJ9', 'a' * 48,
    ])
    def test_token_shaped_strings_are_redacted(self, secret) -> None:
        out = usage.topic_text(f'deploy with {secret} please')
        assert secret not in out and usage.REDACTED in out
        assert out.startswith('deploy with') and out.endswith('please')

    def test_ordinary_text_is_kept(self) -> None:
        for text in ('Fix guru/domain/ledger.py line 412 (record_call)',
                     'see guru/repositories/usage_sqlite_and_more/paths/x'):
            assert usage.topic_text(text) == text

    def test_whitespace_collapsed_and_cut(self) -> None:
        out = usage.topic_text('fix   the\n\nlogin ' + 'word ' * 100)
        assert out.startswith('fix the login word')
        assert len(out) <= usage.TOPIC_CHARS and out.endswith('…')


class TestLabel:
    @pytest.mark.parametrize('reply, label', [
        ('{"topic": "usage dashboard store"}', 'usage dashboard store'),
        ('Sure: {"topic": " Fix login redirect. "}', 'Fix login redirect'),
        ('{"topic": ""}', None), ('no json', None), ('{"topic": 3}', None),
        ('{"topic": "' + 'word ' * 12 + '"}', None),
    ])
    def test_parse(self, reply, label) -> None:
        assert usage.parse_label(reply) == label

    def test_prompt_carries_the_request(self) -> None:
        prompt = usage.label_prompt('add a flag')
        assert prompt.endswith('REQUEST:\nadd a flag')


class TestRanges:
    def test_since(self) -> None:
        now = datetime(2026, 10, 8, 15, 30, tzinfo=timezone.utc)
        assert usage.since('all', now) is None
        assert usage.since('7d', now) == '2026-10-01T15:30:00.000+00:00'
        today = usage.since('today', now)
        local = now.astimezone()
        midnight = datetime.combine(local.date(), datetime.min.time()
                                    ).astimezone()
        assert today == midnight.astimezone(timezone.utc).isoformat(
            timespec='milliseconds')
        with pytest.raises(ValueError):
            usage.since('forever', now)


class TestBeginTopic:
    def test_topic_row_and_background_label(self, monkeypatch,
                                            fake_repo) -> None:
        done = threading.Event()

        class Labeler:
            def label(self, request):
                self.seen = (request, session.topic_id)
                done.set()
                return 'login redirect fix'

        labeler = Labeler()
        usage.set_labeler(labeler)
        monkeypatch.setattr(session, 'call_count', 5)
        monkeypatch.setattr(session, 'turn_id', 'turn-9')
        monkeypatch.setattr(session, 'topic_id', '')
        try:
            usage.begin_topic('please fix the login redirect')
            assert done.wait(5)
        finally:
            usage.set_labeler(None)
        time.sleep(0.05)
        ledger.flush()
        rows = fake_repo.stream('topics')
        assert session.topic_id == 'turn-9'
        assert rows[0]['request'] == 'please fix the login redirect'
        assert rows[0]['topic_id'] == rows[-1]['topic_id'] == 'turn-9'
        assert rows[-1]['label'] == 'login redirect fix'
        assert labeler.seen == ('please fix the login redirect', 'turn-9')
        assert session.call_count == 5         # the label's own session

    def test_labels_off_or_no_labeler(self, monkeypatch, fake_repo) -> None:
        monkeypatch.setattr(session, 'turn_id', 'turn-1')
        monkeypatch.setattr(config, 'TOPIC_LABELS', False)

        class Never:
            def label(self, request):
                raise AssertionError('labelled')
        usage.set_labeler(Never())
        try:
            usage.begin_topic('x')
        finally:
            usage.set_labeler(None)
        ledger.flush()
        assert [r['label'] for r in fake_repo.stream('topics')] == ['']


class TestStoreWrites:
    def test_schema_modes_and_a_call(self, store) -> None:
        store.append('calls', _call())
        assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(store.path.parent).st_mode) == 0o700
        con = sqlite3.connect(store.path)
        row = con.execute('SELECT source, model, tokens_in, cost_usd,'
                          ' project_path FROM calls').fetchone()
        assert row[:4] == ('cli', 'claude-sonnet-5', 1000, 0.01)
        assert row[4] == os.getcwd()
        assert con.execute('PRAGMA user_version').fetchone()[0] == \
            usage_sqlite.SCHEMA_VERSION
        assert con.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'

    def test_an_existing_loose_file_is_tightened(self, tmp_path) -> None:
        path = tmp_path / 'usage.db'
        path.write_bytes(b'')
        os.chmod(path, 0o644)
        s = SqliteUsage(path)
        s.append('calls', _call())
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    def test_bad_values_become_null_not_errors(self, store) -> None:
        store.append('calls', _call(tokens_in='many', cost_usd={'x': 1},
                                    seconds=True))
        assert not store.disabled
        row = store.calls(None)[0]
        assert row['tokens_in'] is None and row['cost_usd'] is None

    def test_topic_upsert_keeps_request_and_label(self, store) -> None:
        store.append('topics', {'topic_id': 't1', 'ts': 'x',
                                'request': 'fix the login', 'label': ''})
        store.append('topics', {'topic_id': 't1', 'label': 'login fix'})
        con = sqlite3.connect(store.path)
        assert con.execute('SELECT request, label FROM topics').fetchall() \
            == [('fix the login', 'login fix')]

    def test_task_spawn_then_finish_is_one_row(self, store) -> None:
        task = {'task_id': 'k1', 'ts': 'x', 'topic_id': 't1',
                'task': 'review auth', 'kind': 'review', 'status': 'running'}
        store.append('tasks', task)
        store.append('tasks', {**task, 'status': 'done', 'cost_usd': 0.2})
        con = sqlite3.connect(store.path)
        assert con.execute('SELECT status, cost_usd, goal FROM tasks'
                           ).fetchall() == [('done', 0.2, 'review auth')]

    def test_other_streams_are_ignored(self, store) -> None:
        store.append('decisions', {'x': 1})
        assert not store.path.exists()


class TestStoreReads:
    def _seed(self, store) -> None:
        store.append('topics', {'topic_id': 't1', 'request': 'fix login',
                                'label': 'login fix'})
        store.append('topics', {'topic_id': 't2', 'request': 'add a flag',
                                'label': ''})
        store.append('calls', _call(cost_usd=0.5))
        store.append('calls', _call(topic_id='t2', model='claude-4-5-haiku',
                                    cost_usd=0.1, project='other'))
        store.append('calls', _call(topic_id='t2', cost_usd=None))
        eval_store = SqliteUsage(store.path, source='eval')
        eval_store.append('calls', _call(topic_id='', cost_usd=2.0))

    def test_totals_and_source_filter(self, store) -> None:
        self._seed(store)
        all_ = store.totals(None, 'all')
        assert all_['calls'] == 4 and all_['cost'] == pytest.approx(2.6)
        assert all_['unpriced'] == 1
        assert store.totals(None, 'cli')['cost'] == pytest.approx(0.6)
        assert store.totals(None, 'eval')['calls'] == 1

    def test_no_topic_and_no_project_are_not_counted(self, store) -> None:
        store.append('calls', _call(topic_id='t1', project='guru'))
        store.append('calls', _call(topic_id='', project=''))
        totals = store.totals(None)
        assert (totals['topics'], totals['projects']) == (1, 1)

    def test_groups_by_topic_label_then_request(self, store) -> None:
        self._seed(store)
        rows = {r['key']: r for r in store.groups('topic', None, 'cli')}
        assert set(rows) == {'login fix', 'add a flag'}
        assert rows['login fix']['cost'] == pytest.approx(0.5)
        assert rows['add a flag']['unpriced'] == 1
        assert [r['key'] for r in store.groups('topic', None, 'eval')] == \
            ['(no topic)']
        with pytest.raises(ValueError):
            store.groups('user; DROP TABLE calls', None)

    def test_groups_by_model_and_project(self, store) -> None:
        self._seed(store)
        assert [r['key'] for r in store.groups('model', None, 'cli')] == \
            ['claude-sonnet-5', 'claude-4-5-haiku']
        assert {r['key'] for r in store.groups('project', None)} == \
            {'guru', 'other'}

    def test_daily_and_calls(self, store) -> None:
        self._seed(store)
        days = store.daily(None, 'cli')
        models = {d['model'] for d in days}
        assert models == {'claude-sonnet-5', 'claude-4-5-haiku'}
        page = store.calls(None, 'all', limit=2, offset=0)
        assert len(page) == 2 and 'topic' in page[0]
        assert store.calls(None, 'all', limit=2, offset=3)[0]['source'] \
            in ('cli', 'eval')

    def test_since_filters(self, store) -> None:
        store.append('calls', _call(ts='2026-01-01T00:00:00.000+00:00'))
        store.append('calls', _call(ts='2026-10-08T00:00:00.000+00:00'))
        assert store.totals('2026-06-01T00:00:00.000+00:00')['calls'] == 1

    def test_reading_a_missing_store_is_empty(self, tmp_path) -> None:
        s = SqliteUsage(tmp_path / 'none.db')
        assert s.groups('model', None) == [] and s.totals(None)['calls'] == 0
        assert not (tmp_path / 'none.db').exists()


class TestStoreRobustness:
    def test_missing_columns_are_added_and_version_kept(self,
                                                        tmp_path) -> None:
        path = tmp_path / 'usage.db'
        con = sqlite3.connect(path)
        con.execute('CREATE TABLE calls (id INTEGER PRIMARY KEY, ts TEXT,'
                    ' model TEXT)')
        con.execute('PRAGMA user_version = 7')       # a newer guru
        con.commit()
        con.close()
        s = SqliteUsage(path)
        s.append('calls', _call())
        con = sqlite3.connect(path)
        cols = {r[1] for r in con.execute('PRAGMA table_info(calls)')}
        assert {'topic_id', 'cost_usd', 'source'} <= cols
        assert con.execute('PRAGMA user_version').fetchone()[0] == 7
        assert s.totals(None)['calls'] == 1

    def test_a_busy_first_open_retries_then_waits(self, tmp_path,
                                                  monkeypatch) -> None:
        s = SqliteUsage(tmp_path / 'usage.db')
        real = SqliteUsage._migrate
        calls: list = []

        def flaky(self, conn):
            calls.append(1)
            if len(calls) < 3:
                raise sqlite3.OperationalError('database is locked')
            return real(self, conn)
        monkeypatch.setattr(SqliteUsage, '_migrate', flaky)
        s.append('calls', _call())
        assert len(calls) == 3 and not s.disabled
        assert s.totals(None)['calls'] == 1

    def test_a_busy_database_drops_the_row(self, tmp_path,
                                           monkeypatch, caplog) -> None:
        monkeypatch.setattr(usage_sqlite, 'BUSY_TIMEOUT_S', 0.1)
        s = SqliteUsage(tmp_path / 'usage.db')
        s.append('calls', _call())
        holder = sqlite3.connect(s.path, isolation_level=None)
        holder.execute('BEGIN EXCLUSIVE')
        try:
            with caplog.at_level('WARNING', logger='guru'):
                t0 = time.monotonic()
                s.append('calls', _call())
                assert time.monotonic() - t0 < 2
        finally:
            holder.execute('ROLLBACK')
            holder.close()
        assert not s.disabled
        assert any('row dropped' in r.getMessage() for r in caplog.records)
        s.append('calls', _call())
        assert s.totals(None)['calls'] == 2

    def test_an_unopenable_path_disables(self, tmp_path) -> None:
        blocker = tmp_path / 'file'
        blocker.write_text('x')
        s = SqliteUsage(blocker / 'usage.db')     # parent is a file
        s.append('calls', _call())
        assert s.disabled
        s.append('calls', _call())                # no retry, no raise

    def test_a_corrupt_file_disables(self, tmp_path) -> None:
        path = tmp_path / 'usage.db'
        path.write_bytes(b'not a database' * 100)
        s = SqliteUsage(path)
        s.append('calls', _call())
        assert s.disabled

    def test_three_processes_write_every_row(self, tmp_path) -> None:
        path = tmp_path / 'usage.db'
        code = ('import sys\n'
                'from guru.repositories.usage_sqlite import SqliteUsage\n'
                'from tests.test_usage import _call\n'
                's = SqliteUsage(sys.argv[1])\n'
                'for i in range(40):\n'
                "    s.append('calls', _call())\n"
                'assert not s.disabled\n')
        root = Path(__file__).resolve().parents[1]
        env = {**os.environ, 'PYTHONPATH': str(root)}
        procs = [subprocess.Popen([sys.executable, '-c', code, str(path)],
                                  cwd=root, env=env) for _ in range(3)]
        assert [p.wait(60) for p in procs] == [0, 0, 0]
        assert SqliteUsage(path).totals(None)['calls'] == 120


class TestFanOut:
    def test_every_sink_gets_the_row_and_a_failure_is_contained(
            self) -> None:
        got: list = []

        class Primary:
            dir = 'here'

            def append(self, stream, row):
                got.append(('p', stream))

            def rows(self, stream):
                return ['r']

        class Broken:
            def append(self, stream, row):
                raise RuntimeError('boom')

        class Extra:
            def append(self, stream, row):
                got.append(('e', stream))

        f = FanOutLedger(Primary(), Broken(), None, Extra())
        f.append('calls', {})
        assert got == [('p', 'calls'), ('e', 'calls')]
        assert f.rows('calls') == ['r'] and f.dir == 'here'
