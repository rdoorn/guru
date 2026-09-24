"""Tests for guru.domain.ledger_report and the bench/ledger_report.py script
on a temp ledger dir of crafted rows (no Ollama, torch or network)."""
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

from guru.domain import ledger_report as lr
from guru.repositories.jsonl_ledger import JsonlLedger

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / 'bench' / 'ledger_report.py'


def _write(directory: Path, stream: str, day: str, rows: list) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'{stream}-{day}.jsonl').open('a') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')


def _task(task_id: str, status: str = 'done', *, kind: str = 'review',
          complexity: str = 'standard', seconds: Optional[float] = 1.0,
          adapter: str = 'Ollama', model: str = 'qwen3:8b',
          cost: Optional[float] = 0.0, **extra: object) -> dict:
    row = {'task_id': task_id, 'status': status, 'kind': kind,
           'complexity': complexity, 'seconds': seconds, 'adapter': adapter,
           'model': model, 'cost_usd': cost, 'turn_id': 'turn1'}
    row.update(extra)
    return row


def _decision(point: str, chosen: Optional[bool],
              heuristic: Optional[bool], *, task_id: str = '',
              turn_id: str = 'turn1', judge: str = 'fake') -> dict:
    agree = (chosen == heuristic) if (chosen is not None
                                      and heuristic is not None) else None
    return {'point': point, 'chosen': chosen, 'heuristic': heuristic,
            'agree': agree, 'task_id': task_id, 'turn_id': turn_id,
            'judge': judge}


class TestFinishRows:
    def test_latest_non_running_row_per_task(self) -> None:
        rows = [_task('a', 'running'), _task('a', 'done', seconds=2.0),
                _task('b', 'running'), _task('c', 'error'),
                _task('c', 'done', seconds=5.0)]
        fin = lr.finish_rows(rows)
        assert [(r['task_id'], r['status'], r['seconds']) for r in fin] == [
            ('a', 'done', 2.0), ('c', 'done', 5.0)]


class TestPercentile:
    def test_nearest_rank(self) -> None:
        vals = [float(v) for v in range(1, 11)]        # 1..10
        assert lr.percentile(vals, 50) == 5.0
        assert lr.percentile(vals, 95) == 10.0
        assert lr.percentile([3.0], 95) == 3.0
        assert lr.percentile([], 50) is None
        assert lr.percentile([2.0, 1.0], 50) == 1.0     # unsorted input


class TestLatency:
    def test_per_kind_and_complexity_from_finish_rows(self) -> None:
        rows = [_task('a', seconds=1.0), _task('b', seconds=3.0),
                _task('c', seconds=2.0), _task('d', 'running', seconds=99.0),
                _task('e', kind='code', complexity='hard', seconds=10.0),
                _task('f', seconds=None)]                # ignored: no time
        lat = lr.latency_by_kind(rows)
        assert lat[('review', 'standard')] == {'n': 3, 'p50': 2.0,
                                               'p95': 3.0}
        assert lat[('code', 'hard')] == {'n': 1, 'p50': 10.0, 'p95': 10.0}

    def test_rows_without_kind_bucket_as_unlabelled(self) -> None:
        bare = {'task_id': 'z', 'status': 'done', 'seconds': 2.0}
        lat = lr.latency_by_kind([bare, _task('a', seconds=1.0)])
        assert lat[('unlabelled', 'unlabelled')] == {'n': 1, 'p50': 2.0,
                                                     'p95': 2.0}
        assert ('other', 'standard') not in lat


class TestFallbackRetry:
    def test_na_until_the_fields_exist(self) -> None:
        r = lr.fallback_retry([_task('a'), _task('b', 'error')])
        assert r == {'tasks': 2, 'fell_back': 0, 'fell_back_rate': 0.0,
                     'retries': None, 'retry_rate': None}

    def test_rates_when_present(self) -> None:
        rows = [_task('a', retry_of=''),
                _task('b', 'fell_back', retry_of=''),
                _task('c', retry_of='a'),
                _task('d', 'running', retry_of='')]
        r = lr.fallback_retry(rows)
        assert r['tasks'] == 3 and r['fell_back'] == 1
        assert r['fell_back_rate'] == pytest.approx(1 / 3)
        assert r['retries'] == 1 and r['retry_rate'] == pytest.approx(1 / 3)

    def test_empty(self) -> None:
        assert lr.fallback_retry([])['tasks'] == 0
        assert lr.fallback_retry([])['fell_back_rate'] is None


class TestAgreement:
    def test_judge_vs_heuristic_per_point(self) -> None:
        rows = [_decision('stall', True, True),
                _decision('stall', False, True),
                _decision('stall', None, True),
                _decision('delegate', True, True)]
        a = lr.judge_vs_heuristic(rows)
        assert a['stall'] == {'n': 3, 'agree': 1, 'disagree': 1,
                              'undecided': 1, 'rate': 0.5}
        assert a['delegate']['rate'] == 1.0

    def test_judge_vs_labels_matches_task_or_turn(self) -> None:
        decisions = [
            _decision('stall', True, True, task_id='tA'),        # good
            _decision('stall', False, True, task_id='tA'),       # good
            _decision('stall', True, False, task_id='tB'),       # bad
            _decision('stall', None, False, turn_id='turn9'),    # good (turn)
            _decision('stall', True, True, task_id='tX'),        # unlabelled
            _decision('delegate', True, True, task_id='tB')]     # bad
        labels = [{'target_id': 'tA', 'label': 'good'},
                  {'target_id': 'tB', 'label': 'bad'},
                  {'target_id': 'turn9', 'label': 'bad'},
                  {'target_id': 'turn9', 'label': 'good'}]        # last wins
        a = lr.judge_vs_labels(decisions, labels)
        assert a['stall']['good'] == {'n': 3, 'chosen_true': 1,
                                      'chosen_false': 1, 'undecided': 1,
                                      'agree_heuristic': 1}
        assert a['stall']['bad'] == {'n': 1, 'chosen_true': 1,
                                     'chosen_false': 0, 'undecided': 0,
                                     'agree_heuristic': 0}
        assert a['delegate']['bad']['n'] == 1
        assert 'good' not in a['delegate']


def _judged(point: str, p_yes: float, heuristic: bool, *, sha: str,
            question: str = '', judge: str = 'fake', ts: str = '',
            used: str = 'heuristic', reason: str = '') -> dict:
    """A decision row with a full ``dist`` (what the review loop needs)."""
    row = _decision(point, p_yes >= 0.5, heuristic, judge=judge)
    row.update({'question': question or point, 'input_sha': sha,
                'input_head': f'Reply:\n{sha}', 'dist': {'yes': p_yes},
                'ts': ts, 'used': used, 'fallback_reason': reason})
    return row


def _label(target: str, label: str, ts: str = '') -> dict:
    return {'target_id': target, 'label': label, 'labeller': 'user',
            'ts': ts, 'note': ''}


class TestDecisionKey:
    def test_key_joins_point_question_and_input(self) -> None:
        row = _judged('panel', 0.7, True, sha='abc', question='needs_sre')
        assert lr.decision_key(row) == 'panel:needs_sre:abc'
        assert lr.decision_key({}) == '::'

    def test_judge_vs_labels_joins_on_decision_key_first(self) -> None:
        rows = [_judged('stall', 0.9, False, sha='s1') | {'task_id': 'tA'}]
        labels = [_label('tA', 'good'), _label('stall:stall:s1', 'no')]
        a = lr.judge_vs_labels(rows, labels)
        assert list(a['stall']) == ['no']
        assert a['stall']['no']['chosen_true'] == 1


class TestPrf:
    def test_counts_and_rates(self) -> None:
        m = lr.prf(tp=3, fp=1, fn=1, tn=5)
        assert m['precision'] == pytest.approx(0.75)
        assert m['recall'] == pytest.approx(0.75)
        assert m['f1'] == pytest.approx(0.75)
        assert m['fpr'] == pytest.approx(1 / 6)

    def test_undefined_rates_are_none(self) -> None:
        m = lr.prf(tp=0, fp=0, fn=0, tn=2)
        assert m['precision'] is None and m['recall'] is None
        assert m['f1'] is None and m['fpr'] == 0.0


class TestSuggestThreshold:
    def test_picks_the_f1_maximising_threshold(self) -> None:
        # Truth yes at 0.6/0.7/0.9, no at 0.1/0.3/0.55: any t in
        # (0.55, 0.6] is perfect; the grid point is 0.6.
        pairs = [(0.6, True), (0.7, True), (0.9, True),
                 (0.1, False), (0.3, False), (0.55, False)]
        s = lr.suggest_threshold(pairs)
        assert s == {'threshold': 0.6, 'f1': 1.0, 'n': 6}

    def test_ties_break_towards_half(self) -> None:
        pairs = [(0.9, True), (0.1, False)]      # every t in (0.1, 0.9] ties
        assert lr.suggest_threshold(pairs)['threshold'] == 0.5

    def test_empty(self) -> None:
        assert lr.suggest_threshold([]) is None


class TestJudgeMetrics:
    def _rows(self) -> tuple:
        rows = [
            _judged('stall', 0.9, True, sha='a', used='judge'),    # tp
            _judged('stall', 0.8, False, sha='b', used='judge'),   # fp
            _judged('stall', 0.2, True, sha='c', used='heuristic',
                    reason='timeout'),                             # fn
            _judged('stall', 0.1, False, sha='d'),                 # tn
            _judged('stall', 0.7, False, sha='e'),                 # unlabelled
            _judged('stall', 0.6, True, sha='f', judge='other'),   # other
            _judged('panel', 0.6, True, sha='g', question='needs_sre')]
        labels = [_label('stall:stall:a', 'yes'),
                  _label('stall:stall:b', 'no'),
                  _label('stall:stall:c', 'yes'),
                  _label('stall:stall:d', 'no'),
                  _label('stall:stall:d', 'yes', ts='2'),        # last wins
                  _label('stall:stall:d', 'no', ts='3'),
                  _label('turnX', 'good')]                        # not review
        return rows, labels

    def test_per_point_and_judge(self) -> None:
        rows, labels = self._rows()
        m = lr.judge_metrics(rows, labels)
        assert set(m) == {'stall', 'panel'}
        assert set(m['stall']) == {'fake', 'other'}
        st = m['stall']['fake']
        assert st['rows'] == 5 and st['labelled'] == 4
        assert st['used'] == {'judge': 2, 'heuristic': 3}
        assert st['fallbacks'] == {'timeout': 1}
        assert st['agree_rate'] == pytest.approx(2 / 5)
        assert st['judge']['tp'] == 1 and st['judge']['fp'] == 1
        assert st['judge']['fn'] == 1 and st['judge']['tn'] == 1
        assert st['judge']['precision'] == pytest.approx(0.5)
        assert st['judge']['recall'] == pytest.approx(0.5)
        # heuristic: True on a (yes) and c (yes), False on b, d (no)
        assert st['heuristic']['precision'] == 1.0
        assert st['heuristic']['recall'] == 1.0
        assert st['threshold']['n'] == 4
        # t in (0.1, 0.2] catches both yes rows with one fp: F1 0.8;
        # ties (0.15, 0.2) break towards 0.5.
        assert st['threshold']['threshold'] == 0.2
        assert st['threshold']['f1'] == pytest.approx(0.8)
        assert m['panel']['fake']['labelled'] == 0
        assert m['panel']['fake']['threshold'] is None
        assert st['queued_ms'] == {'n': 0, 'p50': None, 'p95': None}

    def test_queued_ms_percentiles(self) -> None:
        rows = [_judged('stall', 0.9, True, sha=str(i)) | {'queued_ms': q}
                for i, q in enumerate((5, 10, 400, None))]
        q = lr.judge_metrics(rows, [])['stall']['fake']['queued_ms']
        assert q == {'n': 3, 'p50': 10.0, 'p95': 400.0}
        text = lr.render_metrics(lr.judge_metrics(rows, []))
        assert 'queued_ms p50 10  p95 400  (n 3)' in text

    def test_point_filter(self) -> None:
        rows, labels = self._rows()
        assert list(lr.judge_metrics(rows, labels, point='panel')) == [
            'panel']

    def test_undecided_rows_are_counted_not_scored(self) -> None:
        rows = [_judged('stall', 0.9, True, sha='a')]
        rows[0]['chosen'] = None
        rows[0]['dist'] = {}
        m = lr.judge_metrics(rows, [_label('stall:stall:a', 'yes')])
        j = m['stall']['fake']['judge']
        assert j['undecided'] == 1 and j['tp'] == 0
        assert m['stall']['fake']['threshold'] is None

    def test_render_text(self) -> None:
        rows, labels = self._rows()
        text = lr.render_metrics(lr.judge_metrics(rows, labels))
        assert 'stall' in text and 'fake' in text
        assert 'precision' in text and 'threshold' in text


class TestReviewQueue:
    def test_recent_unlabelled_rows_newest_first_deduped(self) -> None:
        rows = [_judged('stall', 0.9, True, sha='a', ts='1'),
                _judged('stall', 0.9, True, sha='a', ts='2'),   # same input
                _judged('stall', 0.2, False, sha='b', ts='3'),
                _judged('stall', 0.5, True, sha='c', ts='4'),
                _judged('panel', 0.5, True, sha='d', ts='5'),
                {'point': 'stall', 'ts': '6'}]                  # no sha
        labels = [_label('stall:stall:c', 'no')]
        q = lr.review_queue(rows, labels, point='stall', n=5)
        assert [r['input_sha'] for r in q] == ['b', 'a']
        assert q[1]['ts'] == '2'
        assert [r['input_sha'] for r in
                lr.review_queue(rows, labels, point='stall', n=1)] == ['b']


class TestUnlabelledTasks:
    def test_filters_labelled_and_running(self) -> None:
        tasks = [_task('a', ts='1', task='review x', reason=['local-only']),
                 _task('b', ts='2', task='fix y'),
                 _task('c', 'running', ts='3', task='still going'),
                 _task('d', 'error', ts='4', task='boom', seconds=None,
                       cost=None, transcript_path='/t/d.json.gz')]
        labels = [_label('b', 'good')]
        out = lr.unlabelled_tasks(tasks, labels, n=5)
        assert [t['task_id'] for t in out] == ['d', 'a']
        assert out[0]['status'] == 'error' and out[0]['seconds'] is None
        assert out[0]['transcript_path'] == '/t/d.json.gz'
        assert out[1]['route'] == 'Ollama|qwen3:8b'
        assert out[1]['reason'] == ['local-only']
        assert out[1]['task'] == 'review x'
        assert len(lr.unlabelled_tasks(tasks, labels, n=1)) == 1

    def test_missing_kind_is_empty(self) -> None:
        bare = {'task_id': 'z', 'status': 'done', 'ts': '1'}
        [t] = lr.unlabelled_tasks([bare], [], n=5)
        assert t['kind'] == '' and t['complexity'] == ''


class TestReport:
    def _rows(self) -> dict:
        calls = [{'adapter': 'Ollama', 'model': 'qwen3:8b', 'tokens_in': 100,
                  'tokens_out': 20, 'cache_read': 0, 'cost_usd': 0.0},
                 {'adapter': 'Anthropic', 'model': 'opus', 'tokens_in': 10,
                  'tokens_out': 5, 'cache_read': 3, 'cost_usd': 0.5},
                 {'adapter': 'Anthropic', 'model': 'opus', 'tokens_in': 10,
                  'tokens_out': 5, 'cache_read': 0, 'cost_usd': None}]
        tasks = [_task('a', seconds=1.0), _task('b', seconds=3.0, cost=0.5)]
        turns = [{'turn_id': 'turn1', 'seconds': 4.0, 'cost_usd': 0.5,
                  'controller_executed': False, 'tasks_spawned': 2},
                 {'turn_id': 'turn2', 'seconds': 1.0, 'cost_usd': None,
                  'controller_executed': True, 'tasks_spawned': 0}]
        decisions = [_decision('stall', True, True, task_id='a')]
        labels = [{'target_id': 'a', 'label': 'good'}]
        return {'calls': calls, 'tasks': tasks, 'turns': turns,
                'decisions': decisions, 'labels': labels}

    def test_build_and_render(self) -> None:
        rep = lr.build_report(**self._rows())
        assert rep['models']['Anthropic|opus']['calls'] == 2
        assert rep['models']['Anthropic|opus']['cost_usd'] is None
        assert rep['models']['Ollama|qwen3:8b']['cost_usd'] == 0.0
        assert rep['turns'] == {'n': 2, 'cost_usd': None, 'seconds_p50': 1.0,
                                'seconds_p95': 4.0, 'controller_executed': 1,
                                'tasks_spawned': 2}
        md = lr.render_markdown(rep)
        assert md.startswith('# Ledger report')
        for heading in ('## Calls per model', '## Task latency',
                        '## Fallbacks and retries', '## Judge vs heuristic',
                        '## Judge vs labels', '## Turns'):
            assert heading in md
        assert '| review | standard | 2 | 1.00 | 3.00 |' in md
        assert 'n/a' in md                  # retry rate absent
        assert '$?' in md                   # unknown opus cost
        assert rep['controller_labelled'] is True
        assert 'Controller vs judge agreement: n/a' not in md

    def test_controller_agreement_na_without_kind(self) -> None:
        rows = self._rows()
        rows['tasks'] = [{'task_id': 'a', 'status': 'done', 'seconds': 1.0}]
        rep = lr.build_report(**rows)
        assert rep['controller_labelled'] is False
        md = lr.render_markdown(rep)
        assert 'Controller vs judge agreement: n/a (fields arrive with ' \
               'routing)' in md
        assert '| unlabelled | unlabelled | 1 |' in md

    def test_render_empty(self) -> None:
        md = lr.render_markdown(lr.build_report(
            calls=[], tasks=[], turns=[], decisions=[], labels=[]))
        assert '(no rows)' in md


class TestScript:
    def test_reads_a_ledger_dir_across_days(self, tmp_path: Path) -> None:
        d = tmp_path / 'ledger'
        _write(d, 'calls', '2026-09-22', [
            {'adapter': 'Ollama', 'model': 'qwen3:8b', 'tokens_in': 1,
             'tokens_out': 1, 'cost_usd': 0.0}])
        _write(d, 'calls', '2026-09-23', [
            {'adapter': 'Anthropic', 'model': 'opus', 'tokens_in': 1,
             'tokens_out': 1, 'cost_usd': 0.02}])
        _write(d, 'tasks', '2026-09-23', [_task('a', seconds=2.0)])
        _write(d, 'decisions', '2026-09-23',
               [_decision('stall', True, True, task_id='a')])
        _write(d, 'labels', '2026-09-23',
               [{'target_id': 'a', 'label': 'bad'}])
        assert JsonlLedger(d).rows('calls')[0]['model'] == 'qwen3:8b'
        proc = subprocess.run([sys.executable, str(SCRIPT), '--dir', str(d)],
                              capture_output=True, text=True, cwd=REPO_ROOT)
        assert proc.returncode == 0, proc.stderr
        out = proc.stdout
        assert '| Ollama | qwen3:8b | 1 | 1 | 1 | 0 | 0 | $0.0000 |' in out
        assert '| Anthropic | opus | 1 |' in out
        assert '| stall | bad | 1 |' in out

    def test_missing_dir_is_an_empty_report(self, tmp_path: Path) -> None:
        proc = subprocess.run([sys.executable, str(SCRIPT), '--dir',
                               str(tmp_path / 'nope')],
                              capture_output=True, text=True, cwd=REPO_ROOT)
        assert proc.returncode == 0, proc.stderr
        assert '(no rows)' in proc.stdout


def _choice(question: str, chosen: Optional[str], heuristic: str, *,
            sha: str, judge: str = 'enc', point: str = 'labels') -> dict:
    agree = (chosen == heuristic) if chosen is not None else None
    return {'point': point, 'question': question, 'kind': 'choice',
            'chosen': chosen, 'heuristic': heuristic, 'agree': agree,
            'input_sha': sha, 'judge': judge, 'dist': {}, 'used': 'heuristic',
            'fallback_reason': '', 'task_id': '', 'turn_id': 't1'}


class TestChoiceRows:
    """Choice questions (the ``labels`` point): agreement is
    ``chosen == heuristic``; a decision label is the correct option key."""

    def _rows(self) -> list:
        return [_choice('complexity', 'hard', 'hard', sha='a'),
                _choice('complexity', 'trivial', 'standard', sha='b'),
                _choice('kind', 'debug', 'debug', sha='c'),
                _choice('kind', None, 'review', sha='d')]

    def test_judge_vs_heuristic_counts_choice_agreement(self) -> None:
        a = lr.judge_vs_heuristic(self._rows())
        assert a['labels'] == {'n': 4, 'agree': 2, 'disagree': 1,
                               'undecided': 1, 'rate': 2 / 3}

    def test_judge_metrics_agree_rate_and_option_labels(self) -> None:
        labels = [_label('labels:complexity:a', 'hard'),      # judge+heur ok
                  _label('labels:complexity:b', 'trivial'),   # judge ok
                  _label('labels:kind:c', 'review'),          # both wrong
                  _label('labels:kind:d', 'review')]          # heur ok
        m = lr.judge_metrics(self._rows(), labels)['labels']['enc']
        assert m['rows'] == 4 and m['labelled'] == 4
        assert m['agree_rate'] == pytest.approx(2 / 3)
        # yes/no scoring does not apply: nothing is coerced to a bool
        assert m['judge']['tp'] == m['judge']['fp'] == 0
        assert m['heuristic']['tp'] == m['heuristic']['fp'] == 0
        assert m['threshold'] is None
        assert m['choice'] == {'labelled': 4, 'judge_correct': 2,
                               'heuristic_correct': 2, 'judge_undecided': 1}

    def test_noul_rows_have_no_choice_block(self) -> None:
        rows = [_judged('stall', 0.9, True, sha='a')]
        m = lr.judge_metrics(rows, [_label('stall:stall:a', 'yes')])
        assert m['stall']['fake']['choice'] is None
        assert m['stall']['fake']['judge']['tp'] == 1

    def test_render_shows_choice_accuracy(self) -> None:
        labels = [_label('labels:complexity:a', 'hard')]
        text = lr.render_metrics(lr.judge_metrics(self._rows(), labels))
        assert 'choice' in text and 'judge 1/1' in text
        assert 'heuristic 1/1' in text

    def test_review_labels_keep_yes_no_as_bools(self) -> None:
        truth = lr.decision_labels([_label('k1', 'yes'), _label('k2', 'no'),
                                    _label('k3', 'hard'),
                                    {'target_id': 'k4', 'label': ''}])
        assert truth == {'k1': True, 'k2': False, 'k3': 'hard'}


def _event(tool: str, *, seconds: float = 1.0, produced: int = 100,
           shown: int = 50, denied: str = '') -> dict:
    return {'tool': tool, 'seconds': seconds, 'produced_bytes': produced,
            'shown_bytes': shown, 'denied': denied, 'ok': not denied,
            'turn_id': 'turn1', 'args': {}}


class TestToolsSummary:
    def test_per_tool_counts_means_bytes_and_denials(self) -> None:
        rows = [_event('read_file', seconds=1.0, produced=1000, shown=200),
                _event('read_file', seconds=3.0, produced=500, shown=500),
                _event('write_file', seconds=0.5, denied='mode'),
                _event('web_fetch', seconds=2.0, produced=0, shown=0,
                       denied='policy')]
        s = lr.tools_summary(rows)
        assert list(s) == ['read_file', 'web_fetch', 'write_file']
        assert s['read_file'] == {'calls': 2, 'mean_seconds': 2.0,
                                  'produced_bytes': 1500, 'shown_bytes': 700,
                                  'denials': 0}
        assert s['write_file']['denials'] == 1
        assert s['web_fetch'] == {'calls': 1, 'mean_seconds': 2.0,
                                  'produced_bytes': 0, 'shown_bytes': 0,
                                  'denials': 1}

    def test_missing_fields_default_to_zero(self) -> None:
        s = lr.tools_summary([{'tool': 'x'}, {}])
        assert s['x'] == {'calls': 1, 'mean_seconds': 0.0,
                          'produced_bytes': 0, 'shown_bytes': 0,
                          'denials': 0}
        assert '?' in s

    def test_report_and_markdown_section(self) -> None:
        rep = lr.build_report(calls=[], tasks=[], turns=[], decisions=[],
                              labels=[], tool_events=[
                                  _event('read_file', seconds=2.0,
                                         produced=1000, shown=250),
                                  _event('lint', denied='policy')])
        assert rep['tools']['read_file']['calls'] == 1
        assert rep['counts']['tool_events'] == 2
        md = lr.render_markdown(rep)
        assert '## Tools' in md
        assert '| read_file | 1 | 2.00 | 250 | 1000 | 25% | 0 |' in md
        assert '| lint | 1 | 1.00 | 50 | 100 | 50% | 1 |' in md

    def test_report_without_tool_events_still_renders(self) -> None:
        rep = lr.build_report(calls=[], tasks=[], turns=[], decisions=[],
                              labels=[])
        assert rep['tools'] == {} and rep['counts']['tool_events'] == 0
        assert '## Tools' in lr.render_markdown(rep)

    def test_script_prints_tools_section(self, tmp_path: Path) -> None:
        d = tmp_path / 'ledger'
        _write(d, 'tool_events', '2026-09-24',
               [_event('run_tests', seconds=4.0, produced=8000, shown=400)])
        proc = subprocess.run([sys.executable, str(SCRIPT), '--dir', str(d)],
                              capture_output=True, text=True, cwd=REPO_ROOT)
        assert proc.returncode == 0, proc.stderr
        assert '## Tools' in proc.stdout
        assert '| run_tests | 1 | 4.00 | 400 | 8000 | 5% | 0 |' in proc.stdout
