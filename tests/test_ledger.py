"""Tests for the ledger domain + JSONL repository."""
import gzip
import json
import subprocess

import pytest
from typing import Optional

from guru import config, session
from guru.domain import ledger, pricing
from guru.repositories.jsonl_ledger import JsonlLedger


class TestEntities:
    """Run id and CallRecord rows."""

    def test_run_id_is_short_hex(self) -> None:
        assert len(ledger.RUN_ID) == 12
        int(ledger.RUN_ID, 16)

    def test_project_is_computed_once(self) -> None:
        assert isinstance(ledger.PROJECT, str)
        assert ledger.base_row()['project'] == ledger.PROJECT

    def test_task_record_text_sha_is_stable(self) -> None:
        a = ledger.TaskRecord(task_id='1', parent='main', task='review auth')
        b = ledger.TaskRecord(task_id='2', parent='main', task='review auth')
        sha = a.to_row()['text_sha']
        assert len(sha) == 16 and int(sha, 16) >= 0
        assert sha == b.to_row()['text_sha']
        assert sha != ledger.TaskRecord(task_id='3', parent='main',
                                        task='other').to_row()['text_sha']

    def test_call_record_to_row_has_keys(self) -> None:
        rec = ledger.CallRecord(adapter='Ollama', model='qwen3:14b',
                                usage=pricing.Usage(10, 5), seconds=1.2,
                                phase='step', local=True)
        row = rec.to_row()
        for k in ('ts', 'run_id', 'project', 'agent', 'task_id', 'turn_id',
                  'adapter', 'model', 'tokens_in', 'tokens_out',
                  'cache_read', 'cache_write', 'seconds', 'phase', 'cost_usd',
                  'cost_source'):
            assert k in row, k
        assert row['cost_usd'] == 0.0 and row['cost_source'] == 'local'


class ExplodingRepo:
    """LedgerRepository whose append always fails."""

    def append(self, stream: str, row: dict) -> None:
        raise OSError('disk on fire')


class TestRecording:
    """record_* functions route rows through the installed repository."""

    @pytest.fixture(autouse=True)
    def _repo(self, fake_repo) -> None:
        pass

    def test_new_task_explicit_keys_win_over_session(
            self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'turn_id', 'S')
        monkeypatch.setattr(session, 'model', 'session-model')
        t = ledger.new_task(task='x', parent='main', turn_id='P',
                            adapter='Ollama', model='parent-model')
        assert (t.turn_id, t.adapter, t.model) == \
            ('P', 'Ollama', 'parent-model')
        d = ledger.new_task(task='x', parent='main')
        assert (d.turn_id, d.model) == ('S', 'session-model')

    def test_record_call_prices_remote_and_carries_session_keys(
            self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        monkeypatch.setattr(session, 'agent_id', 'agent2')
        monkeypatch.setattr(session, 'task_id', 't1')
        monkeypatch.setattr(session, 'turn_id', 'u9')
        ledger.record_call(adapter='Anthropic', model='claude-sonnet-5',
                           usage=pricing.Usage(1_000_000, 0), seconds=2.0,
                           phase='final')
        ledger.flush()
        [(stream, row)] = ledger.repository().rows
        assert stream == 'calls' and row['cost_usd'] == 2.0
        assert row['agent'] == 'agent2' and row['task_id'] == 't1'
        assert row['turn_id'] == 'u9' and row['cost_source'] == 'table'

    def test_header_cost_wins(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        ledger.record_call(adapter='LiteLLM', model='azure/gpt-4.1',
                           usage=pricing.Usage(100, 100), seconds=1.0,
                           phase='final', cost_header=0.0123)
        ledger.flush()
        [(_, row)] = ledger.repository().rows
        assert row['cost_usd'] == 0.0123 and row['cost_source'] == 'header'

    def test_disabled_ledger_records_nothing(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        ledger.record_call(adapter='Ollama', model='m', usage=pricing.Usage(),
                           seconds=0.1, phase='final', local=True)
        ledger.flush()
        assert ledger.repository().rows == []

    def test_task_and_turn_records(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        t = ledger.new_task(task='review auth', parent='main', role='dev',
                            skill='code-review', kind='review',
                            complexity='standard')
        assert t.task_id and t.status == 'running'
        ledger.record_task(t)
        ledger.finish_task(t, status='done', seconds=3.0, answer_len=120,
                           calls=2, tokens_in=50, tokens_out=20, cost_usd=0.0)
        ledger.record_turn(ledger.TurnRecord(
            turn_id='u1', request='hi', model='qwen3:14b', seconds=0.5,
            tasks_spawned=0, tools_used=[], tokens_in=5, tokens_out=3,
            cost_usd=0.0, controller_executed=False))
        ledger.flush()
        rows = ledger.repository().rows
        assert [s for s, _ in rows] == ['tasks', 'tasks', 'turns']
        spawned, done = rows[0][1], rows[1][1]
        # The spawn row was built before finish_task mutated the record.
        assert spawned['status'] == 'running' and spawned['seconds'] is None
        assert done['status'] == 'done' and done['answer_len'] == 120
        assert done['seconds'] == 3.0
        assert spawned['text_sha'] == done['text_sha']
        assert len(done['text_sha']) == 16

    def test_failing_repository_never_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        ledger.set_repository(ExplodingRepo())
        ledger.record_call(adapter='Ollama', model='m', usage=pricing.Usage(),
                           seconds=0.1, phase='final', local=True)
        ledger.flush()

    def test_no_repository_is_a_noop(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        ledger.set_repository(None)
        assert ledger.repository() is None
        ledger.record_call(adapter='Ollama', model='m', usage=pricing.Usage(),
                           seconds=0.1, phase='final', local=True)
        ledger.flush()

    def test_seconds_and_cost_are_not_rounded(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        ledger.record_call(adapter='Anthropic', model='claude-sonnet-5',
                           usage=pricing.Usage(7, 3), seconds=1.23456789,
                           phase='step')
        ledger.flush()
        [(_, row)] = ledger.repository().rows
        assert row['seconds'] == 1.23456789
        assert row['cost_usd'] == (7 * 2.0 + 3 * 10.0) / 1_000_000


class TestJsonlLedger:
    """Append-only JSONL files, one per stream per UTC day."""

    def test_appends_per_stream_per_day_and_reads_back(self, tmp_path):
        repo = JsonlLedger(tmp_path)
        repo.append('calls', {'a': 1})
        repo.append('tasks', {'b': 2})
        files = sorted(p.name for p in tmp_path.iterdir())
        assert len(files) == 2 and files[0].startswith('calls-')
        assert files[0].endswith('.jsonl')
        assert repo.rows('calls') == [{'a': 1}]

    def test_skips_corrupt_lines(self, tmp_path):
        repo = JsonlLedger(tmp_path)
        repo.append('calls', {'a': 1})
        p = next(tmp_path.glob('calls-*.jsonl'))
        p.write_text(p.read_text() + 'garbage\n[1, 2]\n"str"\n')
        assert repo.rows('calls') == [{'a': 1}]

    def test_unwritable_dir_disables_quietly(self, tmp_path):
        blocker = tmp_path / 'file'
        blocker.write_text('x')
        repo = JsonlLedger(blocker / 'ledger')      # parent is a file
        repo.append('calls', {'a': 1})              # must not raise
        assert repo.disabled is True


class TestAccumulators:
    """record_call feeds the per-session counters; bump never raises."""

    @pytest.fixture(autouse=True)
    def _repo(self, fake_repo) -> None:
        pass                    # accumulators reset by conftest (autouse)

    def test_session_state_defaults(self) -> None:
        st = session.SessionState()
        assert (st.call_count, st.cost_usd, st.cost_known) == (0, 0.0, True)
        assert st.unpriced_calls == 0 and st.last_error == ''
        assert st.struggle == {k: 0 for k in session.STRUGGLE_KEYS}
        assert set(session.STRUGGLE_KEYS) == {
            'stall_nudges', 'delegation_nudges', 'over_read', 'compactions',
            'tool_errors', 'sha_mismatches', 'provider_errors', 'refusals',
            'redactions'}

    def test_record_call_counts_and_adds_known_cost(self) -> None:
        ledger.record_call(adapter='Anthropic', model='claude-sonnet-5',
                           usage=pricing.Usage(1_000_000, 0), seconds=1.0,
                           phase='step')
        ledger.record_call(adapter='Ollama', model='qwen3:14b',
                           usage=pricing.Usage(10, 10), seconds=1.0,
                           phase='step', local=True)
        assert session.call_count == 2
        assert session.cost_usd == 2.0 and session.cost_known is True

    def test_unknown_price_marks_cost_unknown(self) -> None:
        ledger.record_call(adapter='LiteLLM', model='mystery-model',
                           usage=pricing.Usage(10, 10), seconds=1.0,
                           phase='step')
        assert session.call_count == 1 and session.cost_known is False
        assert session.cost_usd == 0.0 and session.unpriced_calls == 1
        # Later priced calls still accumulate; the flag stays False.
        ledger.record_call(adapter='LiteLLM', model='x', cost_header=0.5,
                           usage=pricing.Usage(1, 1), seconds=1.0,
                           phase='step')
        assert session.cost_usd == 0.5 and session.cost_known is False
        assert session.unpriced_calls == 1

    def test_accumulators_update_with_ledger_disabled(
            self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        ledger.record_call(adapter='LiteLLM', model='x', cost_header=0.25,
                           usage=pricing.Usage(1, 1), seconds=1.0,
                           phase='step')
        ledger.flush()
        assert ledger.repository().rows == []
        assert session.call_count == 1 and session.cost_usd == 0.25

    def test_bump_increments_and_never_raises(self, monkeypatch) -> None:
        ledger.bump('refusals')
        ledger.bump('refusals')
        assert session.struggle['refusals'] == 2
        ledger.bump('not-a-key')                    # tolerated, recorded
        assert session.struggle['not-a-key'] == 1
        monkeypatch.setattr(session, 'struggle', None)
        ledger.bump('refusals')                     # must not raise

    def test_struggle_delta(self) -> None:
        before = {'stall_nudges': 1, 'compactions': 0}
        after = {'stall_nudges': 3, 'compactions': 0, 'refusals': 1}
        assert ledger.struggle_delta(before, after) == {
            'stall_nudges': 2, 'compactions': 0, 'refusals': 1}

    def test_finish_task_carries_struggle_and_cost_known(self) -> None:
        t = ledger.new_task(task='x', parent='main')
        struggle = {'tool_errors': 2}
        ledger.finish_task(t, status='done', seconds=1.0, answer_len=1,
                           calls=3, tokens_in=1, tokens_out=1,
                           cost_usd=None, cost_known=False, struggle=struggle)
        struggle['tool_errors'] = 99               # a copy was taken
        ledger.flush()
        [(_, row)] = ledger.repository().rows
        assert row['calls'] == 3 and row['cost_usd'] is None
        assert row['cost_known'] is False
        assert row['struggle'] == {'tool_errors': 2}

    def test_finish_task_defaults(self) -> None:
        t = ledger.new_task(task='x', parent='main')
        ledger.finish_task(t, status='done', seconds=1.0, answer_len=1,
                           calls=0, tokens_in=0, tokens_out=0, cost_usd=0.0)
        ledger.flush()
        [(_, row)] = ledger.repository().rows
        assert row['struggle'] == {} and row['cost_known'] is True
        assert row['transcript_path'] == ''
        assert row['env'] == {} and row['prompt_sha'] == ''
        assert row['tools_active'] == []

    def test_finish_task_cost_none_is_never_known(self) -> None:
        t = ledger.new_task(task='x', parent='main')
        ledger.finish_task(t, status='done', seconds=1.0, answer_len=1,
                           calls=0, tokens_in=0, tokens_out=0, cost_usd=None)
        ledger.flush()
        [(_, row)] = ledger.repository().rows
        assert row['cost_known'] is False

    def test_turn_record_has_struggle(self) -> None:
        row = ledger.TurnRecord(
            turn_id='u1', request='hi', model='m', seconds=0.5,
            tasks_spawned=0, tools_used=[], tokens_in=5, tokens_out=3,
            cost_usd=0.0, controller_executed=False,
            struggle={'compactions': 1}).to_row()
        assert row['struggle'] == {'compactions': 1}


def _git(cwd, *args) -> str:
    return subprocess.run(
        ['git', '-c', 'user.name=t', '-c', 'user.email=t@x', *args],
        cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class TestEnvironment:
    """ledger.environment() snapshots git, cwd, version and config."""

    @pytest.fixture(autouse=True)
    def _uncached(self, monkeypatch) -> None:
        monkeypatch.setattr(ledger, '_static_env', None)

    def test_git_repo_clean_then_dirty(self, tmp_path, monkeypatch) -> None:
        _git(tmp_path, 'init', '-q')
        (tmp_path / 'a.txt').write_text('a')
        _git(tmp_path, 'add', 'a.txt')
        _git(tmp_path, 'commit', '-q', '-m', 'init')
        monkeypatch.chdir(tmp_path)
        env = ledger.environment()
        assert env['git_sha'] == _git(tmp_path, 'rev-parse', 'HEAD')
        assert len(env['git_sha']) == 40 and env['git_dirty'] is False
        assert env['cwd'] == str(tmp_path.resolve())
        (tmp_path / 'b.txt').write_text('b')
        assert ledger.environment()['git_dirty'] is True

    def test_non_git_dir(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path.parent))
        env = ledger.environment()
        assert env['git_sha'] == '' and env['git_dirty'] is None
        assert env['cwd'] == str(tmp_path.resolve())

    def test_never_raises(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise RuntimeError('no git')
        monkeypatch.setattr(ledger.subprocess, 'run', boom)
        monkeypatch.setattr(config, 'settings_section',
                            lambda name: 1 / 0)
        env = ledger.environment()
        assert env['git_sha'] == '' and env['git_dirty'] is None
        assert env['config_hash'] == ''

    def test_version_config_hash_and_judges(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'DECISIONS_POINTS',
                            {'stall': 'encoder:x', 'panel': 'injection'})
        sections = {'routing': {'b': 1, 'a': 2}, 'decisions': {'mode': 'x'}}
        monkeypatch.setattr(config, 'settings_section',
                            lambda name: dict(sections.get(name, {})))
        env = ledger.environment()
        assert env['judge_models'] == {'stall': 'encoder:x',
                                       'panel': 'injection'}
        assert isinstance(env['guru_version'], str) and env['guru_version']
        assert len(env['config_hash']) == 16
        int(env['config_hash'], 16)
        h1 = env['config_hash']
        ledger._static_env = None
        sections['routing'] = {'a': 2, 'b': 1}         # same content
        assert ledger.environment()['config_hash'] == h1
        ledger._static_env = None
        sections['routing'] = {'a': 3}
        assert ledger.environment()['config_hash'] != h1

    def test_version_and_hash_cached_per_process(self, monkeypatch) -> None:
        calls: list = []

        def section(name):
            calls.append(name)
            return {}
        monkeypatch.setattr(config, 'settings_section', section)
        h = ledger.environment()['config_hash']
        assert ledger.environment()['config_hash'] == h
        assert sorted(calls) == ['decisions', 'routing']    # read once

    def test_version_falls_back_to_package_constant(
            self, monkeypatch) -> None:
        import guru

        def missing(name):
            raise ledger.importlib.metadata.PackageNotFoundError(name)
        monkeypatch.setattr(ledger.importlib.metadata, 'version', missing)
        assert ledger.environment()['guru_version'] == guru.__version__
        assert guru.__version__


class TestTranscripts:
    """Transcripts are gzip JSON keyed by task id; saving never raises."""

    def test_jsonl_ledger_round_trip(self, tmp_path) -> None:
        repo = JsonlLedger(tmp_path)
        msgs = [{'role': 'user', 'content': 'héllo'},
                {'role': 'tool', 'tool_name': 'x', 'content': 'y'}]
        path = repo.save_transcript('t1', msgs)
        assert path == tmp_path / 'transcripts' / 't1.json.gz'
        assert repo.transcript_path('t1') == path
        with gzip.open(path, 'rt', encoding='utf-8') as fh:
            assert json.load(fh) == msgs

    def test_jsonl_ledger_disabled_raises(self, tmp_path) -> None:
        repo = JsonlLedger(tmp_path)
        repo.disabled = True
        with pytest.raises(OSError):
            repo.save_transcript('t1', [])

    def test_domain_helper_uses_repository(self, fake_repo) -> None:
        path = ledger.save_transcript('t1', [{'role': 'user', 'content': 'q'}])
        assert path == fake_repo.transcript_path('t1')
        ledger.flush()                      # the write is off-thread
        assert fake_repo.transcripts['t1'] == [
            {'role': 'user', 'content': 'q'}]

    def test_domain_helper_writes_real_file_off_thread(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        repo = JsonlLedger(tmp_path)
        ledger.set_repository(repo)
        try:
            path = ledger.save_transcript('t2', [{'role': 'user',
                                                  'content': 'q'}])
            assert path == str(tmp_path / 'transcripts' / 't2.json.gz')
            ledger.flush()
            with gzip.open(path, 'rt', encoding='utf-8') as fh:
                assert json.load(fh) == [{'role': 'user', 'content': 'q'}]
        finally:
            ledger.flush()
            ledger.set_repository(None)

    def test_domain_helper_without_support_or_repo(
            self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        ledger.set_repository(ExplodingRepo())       # no save_transcript
        try:
            assert ledger.save_transcript('t1', []) == ''
            ledger.set_repository(None)
            assert ledger.save_transcript('t1', []) == ''
        finally:
            ledger.set_repository(None)

    def test_domain_helper_swallows_failures(self, monkeypatch) -> None:
        class Bad:
            def append(self, stream, row): ...

            def transcript_path(self, task_id):
                return f'/x/{task_id}.json.gz'

            def save_transcript(self, task_id, messages):
                raise OSError('disk on fire')

        class BadPath(Bad):
            def transcript_path(self, task_id):
                raise OSError('no dir')
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        ledger.set_repository(Bad())
        try:
            # The path is known up front; the failing write is logged later.
            assert ledger.save_transcript('t1', []) == '/x/t1.json.gz'
            ledger.flush()                              # must not raise
            ledger.set_repository(BadPath())
            assert ledger.save_transcript('t1', []) == ''
        finally:
            ledger.flush()
            ledger.set_repository(None)

    def test_domain_helper_noop_when_disabled(
            self, monkeypatch, fake_repo) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        assert ledger.save_transcript('t1', []) == ''
        assert fake_repo.transcripts == {}


class TestLabels:
    """record_label writes one row to the labels stream; never raises."""

    def test_label_row(self, fake_repo) -> None:
        ledger.record_label('t123', 'user', 'good', 'nice')
        ledger.flush()
        rows = fake_repo.stream('labels')
        assert len(rows) == 1
        r = rows[0]
        assert r['target_id'] == 't123' and r['labeller'] == 'user'
        assert r['label'] == 'good' and r['note'] == 'nice'
        assert r['run_id'] == ledger.RUN_ID and 'ts' in r and 'project' in r

    def test_default_note_and_exploding_repo(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        ledger.set_repository(ExplodingRepo())
        try:
            ledger.record_label('x', 'judge:fake', 'bad')     # must not raise
            ledger.flush()
        finally:
            ledger.set_repository(None)


def _call(model: str, adapter: str = 'A', run_id: str = 'r1', *,
          tin: int = 10, tout: int = 5, cread: int = 0, cwrite: int = 0,
          cost: Optional[float] = 0.01) -> dict:
    return {'run_id': run_id, 'adapter': adapter, 'model': model,
            'tokens_in': tin, 'tokens_out': tout, 'cache_read': cread,
            'cache_write': cwrite, 'cost_usd': cost}


def _task(task_id: str, status: str = 'done', *, model: str = 'm1',
          adapter: str = 'A', run_id: str = 'r1',
          cost: Optional[float] = 0.0, seconds: Optional[float] = 1.0,
          task: str = 'do it', role: str = '',
          kind: str = 'other') -> dict:
    return {'run_id': run_id, 'task_id': task_id, 'status': status,
            'adapter': adapter, 'model': model, 'cost_usd': cost,
            'seconds': seconds, 'task': task, 'role': role, 'kind': kind}


class TestRunSummary:
    """run_summary is a pure aggregation of calls + tasks rows of one run."""

    def test_per_model_and_totals(self) -> None:
        calls = [_call('m1', cread=7, cwrite=3),
                 _call('m1', tin=20, tout=1, cost=0.02, cread=1),
                 _call('m2', adapter='B', cost=0.5),
                 _call('m9', run_id='other', cost=99.0)]
        s = ledger.run_summary(calls, [], 'r1')
        assert s['run_id'] == 'r1'
        assert s['models']['A|m1'] == {
            'adapter': 'A', 'model': 'm1', 'calls': 2, 'tokens_in': 30,
            'tokens_out': 6, 'cache_read': 8, 'cache_write': 3,
            'cost_usd': pytest.approx(0.03)}
        assert s['models']['B|m2']['calls'] == 1
        assert 'other' not in ''.join(s['models'])
        assert s['totals']['calls'] == 3
        assert s['totals']['tokens_in'] == 40
        assert s['totals']['tokens_out'] == 11
        assert s['totals']['cache_read'] == 8
        assert s['totals']['cache_write'] == 3
        assert s['totals']['cost_usd'] == pytest.approx(0.53)

    def test_unknown_cost_propagates_as_none(self) -> None:
        calls = [_call('m1'), _call('m1', cost=None), _call('m2', cost=0.1)]
        s = ledger.run_summary(calls, [], 'r1')
        assert s['models']['A|m1']['cost_usd'] is None
        assert s['models']['A|m2']['cost_usd'] == pytest.approx(0.1)
        assert s['totals']['cost_usd'] is None

    def test_tasks_per_adapter_model_counts_task_ids_once(self) -> None:
        tasks = [_task('t1', 'running'), _task('t1', 'done'),
                 _task('t2', 'running', model='m2'), _task('t2', 'error',
                                                           model='m2'),
                 _task('t3', 'running'),                 # still running
                 _task('t4', 'done', run_id='other')]
        s = ledger.run_summary([], tasks, 'r1')
        assert s['tasks'] == {'A|m1': 2, 'A|m2': 1}
        assert s['totals']['tasks'] == 3

    def test_top_tasks_by_cost_then_seconds(self) -> None:
        tasks = [_task('a', cost=0.1, seconds=1),
                 _task('b', cost=0.5, seconds=1),
                 _task('c', cost=None, seconds=9),
                 _task('d', cost=0.1, seconds=7),
                 _task('e', cost=0.3, seconds=2),
                 _task('f', 'running', cost=9.0)]
        top = ledger.run_summary([], tasks, 'r1')['top_tasks']
        assert [t['task_id'] for t in top] == ['b', 'e', 'd']
        assert top[0]['cost_usd'] == 0.5 and top[0]['status'] == 'done'
        assert set(top[0]) >= {'task_id', 'role', 'kind', 'model',
                               'adapter', 'cost_usd', 'seconds', 'status',
                               'task'}

    def test_unknown_cost_sorts_last_and_task_text_is_shortened(self):
        tasks = [_task('c', cost=None, seconds=9, task='x' * 200),
                 _task('a', cost=0.0, seconds=1)]
        top = ledger.run_summary([], tasks, 'r1')['top_tasks']
        assert [t['task_id'] for t in top] == ['a', 'c']
        assert len(top[1]['task']) <= 80

    def test_empty(self) -> None:
        s = ledger.run_summary([], [], 'r1')
        assert s['models'] == {} and s['tasks'] == {} and s['top_tasks'] == []
        assert s['totals'] == {'calls': 0, 'tokens_in': 0, 'tokens_out': 0,
                               'cache_read': 0, 'cache_write': 0,
                               'cost_usd': 0.0, 'tasks': 0}


class TestJsonlLedgerRunFilter:
    """rows(stream, run_id=...) keeps only that run's rows."""

    def test_filter(self, tmp_path) -> None:
        repo = JsonlLedger(tmp_path)
        repo.append('calls', {'run_id': 'r1', 'a': 1})
        repo.append('calls', {'run_id': 'r2', 'a': 2})
        repo.append('calls', {'a': 3})                   # no run_id at all
        assert repo.rows('calls') == [{'run_id': 'r1', 'a': 1},
                                      {'run_id': 'r2', 'a': 2}, {'a': 3}]
        assert repo.rows('calls', run_id='r2') == [{'run_id': 'r2', 'a': 2}]
        assert repo.rows('calls', run_id='zz') == []


class TestToolEvents:
    """record_tool_event writes the ``tool_events`` audit stream (A2)."""

    def test_row_has_session_keys_and_fields(self, fake_repo,
                                             monkeypatch) -> None:
        monkeypatch.setattr(session, 'agent_id', 'agent3')
        monkeypatch.setattr(session, 'task_id', 't7')
        monkeypatch.setattr(session, 'turn_id', 'u1')
        ledger.record_tool_event(
            'read_file', {'path': 'a.py', 'lines': '1-5'}, seconds=0.25,
            ok=True, produced_bytes=900, shown_bytes=300,
            files_touched=['/abs/a.py'])
        ledger.flush()
        [(stream, row)] = fake_repo.rows
        assert stream == 'tool_events'
        assert row['agent'] == 'agent3' and row['task_id'] == 't7'
        assert row['turn_id'] == 'u1' and row['run_id'] == ledger.RUN_ID
        assert row['tool'] == 'read_file'
        assert row['args'] == {'path': 'a.py', 'lines': '1-5'}
        assert row['seconds'] == 0.25 and row['ok'] is True
        assert (row['produced_bytes'], row['shown_bytes']) == (900, 300)
        assert row['files_touched'] == ['/abs/a.py']
        assert row['denied'] == ''
        assert 'ts' in row and 'project' in row

    def test_args_are_stringified_and_headed(self, fake_repo) -> None:
        ledger.record_tool_event(
            'write_file', {'path': 'x', 'content': 'y' * 5000, 'depth': 3},
            seconds=0.0, ok=False, produced_bytes=0, shown_bytes=0,
            files_touched=[], denied='policy')
        ledger.flush()
        [(_, row)] = fake_repo.rows
        assert len(row['args']['content']) == ledger.ARGS_HEAD
        assert row['args']['depth'] == '3'
        assert row['denied'] == 'policy' and row['ok'] is False

    def test_never_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        ledger.set_repository(ExplodingRepo())
        try:
            ledger.record_tool_event('t', {}, seconds=0, ok=True,
                                     produced_bytes=0, shown_bytes=0,
                                     files_touched=[])
            ledger.flush()
        finally:
            ledger.set_repository(None)
