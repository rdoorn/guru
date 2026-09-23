"""Tests for run files, compare and the trajectory table (guru.evals.runs)."""
import json
from pathlib import Path

import pytest

from guru.evals import runs
from guru.evals.runs import CaseResult, Run


def result(case: str, ok: bool, seconds: float = 10.0,
           cost=0.5) -> CaseResult:
    return CaseResult(
        case=case, passed=ok,
        checks=[{'name': 'answer_contains', 'passed': ok,
                 'detail': '' if ok else 'answer lacks [\'yes\']'}],
        observed={'answer': 'yes' if ok else 'no', 'tools_used': [],
                  'spawned': 0, 'roles': [], 'stall_nudges': 0,
                  'seconds': seconds, 'files_changed': [],
                  'fixture_tests_pass': None, 'timed_out': False,
                  'error': ''},
        rubric='', transcript_path=f'transcripts/{case}.json.gz',
        cost_usd=cost)


def run(run_id: str = 'abc123', ts: str = '2026-09-23T10:00:00+00:00',
        model: str = 'Ollama|qwen3:14b', cases=None) -> Run:
    if cases is None:
        cases = [result('greet', True), result('review', False)]
    return Run(run_id=run_id, ts=ts, model=model, git_sha='deadbeef',
               cases=cases)


class TestRun:
    def test_pass_rate(self) -> None:
        assert run().pass_rate() == 0.5
        assert run(cases=[]).pass_rate() == 0.0
        assert run(cases=[result('a', True)]).pass_rate() == 1.0

    def test_new_run_id_and_now_ts(self) -> None:
        rid = runs.new_run_id()
        assert len(rid) == 12 and int(rid, 16) >= 0
        assert runs.now_ts().endswith('+00:00')


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        r = run()
        path = runs.save(r, tmp_path / 'runs')
        assert path.parent == tmp_path / 'runs'
        assert path.name == '20260923T100000Z-abc123.json'
        assert runs.load(path) == r

    def test_file_is_readable_json_with_summary(self, tmp_path: Path):
        path = runs.save(run(), tmp_path)
        data = json.loads(path.read_text(encoding='utf-8'))
        assert data['run_id'] == 'abc123'
        assert data['passed'] == 1 and data['total'] == 2
        assert [c['case'] for c in data['cases']] == ['greet', 'review']

    def test_save_rejects_unserialisable_content(self, tmp_path: Path):
        r = run(cases=[result('a', True)])
        r.cases[0].observed['bad'] = object()
        with pytest.raises(TypeError):
            runs.save(r, tmp_path)

    def test_load_rejects_unknown_shape(self, tmp_path: Path) -> None:
        p = tmp_path / 'x.json'
        p.write_text('{"run_id": "x"}')
        with pytest.raises(ValueError, match='x.json'):
            runs.load(p)

    def test_load_ignores_summary_keys(self, tmp_path: Path) -> None:
        path = runs.save(run(), tmp_path)
        data = json.loads(path.read_text())
        data['future_field'] = 1
        path.write_text(json.dumps(data))
        assert runs.load(path).run_id == 'abc123'


class TestCompare:
    def test_categories_and_deltas(self) -> None:
        old = run(run_id='old', cases=[
            result('a', True, seconds=10, cost=0.5),
            result('b', False, seconds=20, cost=1.0),
            result('c', False, seconds=30, cost=None),
            result('gone', True)])
        new = run(run_id='new', cases=[
            result('a', False, seconds=12, cost=0.75),
            result('b', True, seconds=15, cost=0.5),
            result('c', False, seconds=33, cost=None),
            result('fresh', True)])
        d = runs.compare(old, new)
        assert d['newly_passing'] == ['b']
        assert d['newly_failing'] == ['a']
        assert d['still_failing'] == ['c']
        assert d['still_passing'] == []
        assert d['added'] == ['fresh'] and d['removed'] == ['gone']
        assert d['pass_rate'] == {'old': 0.5, 'new': 0.5}
        assert d['deltas']['a'] == {'seconds': 2.0, 'cost_usd': 0.25}
        assert d['deltas']['b'] == {'seconds': -5.0, 'cost_usd': -0.5}
        assert d['deltas']['c'] == {'seconds': 3.0, 'cost_usd': None}
        assert 'fresh' not in d['deltas'] and 'gone' not in d['deltas']
        assert d['old'] == 'old' and d['new'] == 'new'

    def test_identical_runs(self) -> None:
        d = runs.compare(run(), run())
        assert d['newly_passing'] == [] and d['newly_failing'] == []
        assert d['still_failing'] == ['review']
        assert d['still_passing'] == ['greet']
        assert d['deltas']['greet'] == {'seconds': 0.0, 'cost_usd': 0.0}


def _last_row(directory: Path) -> str:
    text = (directory / 'TRAJECTORY.md').read_text(encoding='utf-8')
    return text.strip().splitlines()[-1]


class TestTrajectory:
    def test_header_once_then_rows(self, tmp_path: Path) -> None:
        runs.append_trajectory(run(), tmp_path, note='baseline')
        path = tmp_path / 'TRAJECTORY.md'
        text = path.read_text(encoding='utf-8')
        assert text.count('| ts |') == 1
        lines = text.strip().splitlines()
        assert lines[-1].startswith('| 2026-09-23T10:00:00+00:00 | abc123 |')
        assert '| Ollama\\|qwen3:14b |' in lines[-1]
        assert '| 1/2 |' in lines[-1]
        assert '| 10.0 |' in lines[-1]
        assert '| $1.00 |' in lines[-1] and lines[-1].endswith('| baseline |')
        runs.append_trajectory(run(run_id='def456'), tmp_path)
        text = path.read_text(encoding='utf-8')
        assert text.count('| ts |') == 1
        rows = [ln for ln in text.splitlines() if ln.startswith('| 2026')]
        assert len(rows) == 2 and rows[1].endswith('|  |')

    def test_unknown_cost_and_empty_run(self, tmp_path: Path) -> None:
        r = run(cases=[result('a', True, cost=None),
                       result('b', True, cost=2.0)])
        assert r.total_cost() is None
        runs.append_trajectory(r, tmp_path)
        assert '| n/a |' in _last_row(tmp_path)
        assert run().total_cost() == 1.0
        runs.append_trajectory(run(cases=[]), tmp_path)
        assert '| 0/0 |' in _last_row(tmp_path)

    def test_note_pipes_are_escaped(self, tmp_path: Path) -> None:
        runs.append_trajectory(run(), tmp_path, note='a | b')
        assert _last_row(tmp_path).endswith('| a \\| b |')


class TestNumCtx:
    def test_default_is_zero_and_old_files_load(self, tmp_path: Path):
        r = run()
        assert r.num_ctx == 0
        path = runs.save(r, tmp_path)
        data = json.loads(path.read_text(encoding='utf-8'))
        del data['num_ctx']                      # a run file from before
        path.write_text(json.dumps(data), encoding='utf-8')
        assert runs.load(path) == r

    def test_round_trip_keeps_num_ctx(self, tmp_path: Path) -> None:
        r = run()
        r.num_ctx = 8192
        assert runs.load(runs.save(r, tmp_path)).num_ctx == 8192

    @pytest.mark.parametrize('n, label', [
        (0, 'Ollama|qwen3:14b'), (8192, 'Ollama|qwen3:14b@8k'),
        (40960, 'Ollama|qwen3:14b@40k'), (5000, 'Ollama|qwen3:14b@5000')])
    def test_model_label(self, n: int, label: str) -> None:
        r = run()
        r.num_ctx = n
        assert r.model_label() == label

    def test_trajectory_model_column_carries_context(self, tmp_path: Path):
        r = run()
        r.num_ctx = 8192
        runs.append_trajectory(r, tmp_path)
        assert '| Ollama\\|qwen3:14b@8k |' in _last_row(tmp_path)


class TestRoutingFields:
    """``routing``/``controller`` on Run and ``routes`` on CaseResult."""

    def test_defaults_and_old_files_load(self, tmp_path: Path) -> None:
        r = run()
        assert r.routing == '' and r.controller is False
        assert r.cases[0].routes == []
        path = runs.save(r, tmp_path)
        data = json.loads(path.read_text())
        for key in ('routing', 'controller'):
            del data[key]
        for c in data['cases']:
            del c['routes']
        path.write_text(json.dumps(data))
        loaded = runs.load(path)
        assert loaded.routing == '' and loaded.controller is False
        assert all(c.routes == [] for c in loaded.cases)

    def test_round_trip(self, tmp_path: Path) -> None:
        c = result('a', True)
        c.routes = ['Ollama|qwen3:8b', 'SBP Litellm|aws/claude-5-sonnet']
        r = Run(run_id='r1', ts='2026-09-23T10:00:00+00:00', model='m',
                git_sha='', cases=[c], num_ctx=8192, routing='exp-b',
                controller=True)
        loaded = runs.load(runs.save(r, tmp_path))
        assert loaded == r
        assert loaded.cases[0].routes == c.routes

    def test_model_label_carries_routing(self) -> None:
        r = run()
        r.num_ctx = 8192
        r.routing = 'exp-b'
        assert r.model_label() == 'Ollama|qwen3:14b@8k+routed:exp-b'
        r.controller = True
        assert r.model_label() == \
            'Ollama|qwen3:14b@8k+routed:exp-b+controller'
        r.num_ctx = 0
        assert r.model_label() == 'Ollama|qwen3:14b+routed:exp-b+controller'

    def test_trajectory_row_carries_routing(self, tmp_path: Path) -> None:
        r = run()
        r.routing = 'exp-b'
        runs.append_trajectory(r, tmp_path)
        text = (tmp_path / runs.TRAJECTORY_FILE).read_text()
        assert '| Ollama\\|qwen3:14b+routed:exp-b |' in text
