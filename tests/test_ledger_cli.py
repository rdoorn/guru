"""Tests for ``guru.ledger_cli`` (review / report / tasks) on crafted rows in
a temp ledger dir; the interactive loop is driven by a scripted ``ask``."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from guru import config, ledger_cli
from guru.domain import ledger
from guru.repositories.jsonl_ledger import JsonlLedger

REPO_ROOT = Path(__file__).resolve().parents[1]
DAY = '2026-09-23'


def _write(directory: Path, stream: str, rows: list, day: str = DAY) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'{stream}-{day}.jsonl').open('a') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')


def _decision(sha: str, p_yes: float, heuristic: bool, *, ts: str,
              point: str = 'stall', question: str = 'stall',
              head: str = '') -> dict:
    return {'point': point, 'question': question, 'input_sha': sha,
            'input_head': head or f'Reply:\nLet me look at {sha}',
            'dist': {'yes': p_yes, 'no': round(1 - p_yes, 2)},
            'chosen': p_yes >= 0.5, 'heuristic': heuristic,
            'agree': (p_yes >= 0.5) == heuristic, 'judge': 'fake',
            'used': 'heuristic', 'fallback_reason': '', 'ts': ts,
            'task_id': '', 'turn_id': 'turn1', 'mode': 'shadow'}


def _task(task_id: str, *, ts: str, status: str = 'done', **extra) -> dict:
    row = {'task_id': task_id, 'status': status, 'task': f'do {task_id}',
           'adapter': 'Ollama', 'model': 'qwen3:8b', 'seconds': 2.5,
           'cost_usd': 0.0, 'kind': 'review', 'complexity': 'standard',
           'reason': ['local-only'], 'transcript_path': f'/t/{task_id}.gz',
           'turn_id': 'turn1', 'ts': ts}
    row.update(extra)
    return row


class Out:
    """Collects printed lines."""

    def __init__(self) -> None:
        self.lines: list = []

    def __call__(self, *parts: object) -> None:
        self.lines.append(' '.join(str(p) for p in parts))

    @property
    def text(self) -> str:
        return '\n'.join(self.lines)


def _ask(answers: list):
    it = iter(answers)
    prompts: list = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return next(it)
    ask.prompts = prompts                                   # type: ignore
    return ask


@pytest.fixture
def ledger_dir(tmp_path: Path) -> Path:
    d = tmp_path / 'ledger'
    _write(d, 'decisions', [
        _decision('s1', 0.9, False, ts='2026-09-23T10:00:00'),
        _decision('s2', 0.2, True, ts='2026-09-23T10:01:00'),
        _decision('s3', 0.7, True, ts='2026-09-23T10:02:00'),
        _decision('p1', 0.6, True, ts='2026-09-23T10:03:00', point='panel',
                  question='needs_sre', head='Task: review deploys')])
    return d


def _labels(d: Path) -> list:
    return JsonlLedger(d).rows('labels')


class TestReview:
    def test_labels_rows_newest_first_and_skips(self, ledger_dir) -> None:
        out, ask = Out(), _ask(['y', 's', 'n'])
        rc = ledger_cli.main(['review', '--point', 'stall', '--dir',
                              str(ledger_dir)], ask=ask, out=out)
        assert rc == 0
        rows = _labels(ledger_dir)
        assert [(r['target_id'], r['label']) for r in rows] == [
            ('stall:stall:s3', 'yes'), ('stall:stall:s1', 'no')]
        assert all(r['labeller'] == 'user' for r in rows)
        assert all(r['note'] == 'point:stall;question:stall' for r in rows)
        assert 'Let me look at s3' in out.text
        assert 'P(yes)=0.70' in out.text and 'heuristic=True' in out.text
        assert 'labelled 2' in out.text and 'skipped 1' in out.text

    def test_quit_stops_without_labels(self, ledger_dir) -> None:
        out, ask = Out(), _ask(['q'])
        assert ledger_cli.main(['review', '--point', 'stall', '--dir',
                                str(ledger_dir)], ask=ask, out=out) == 0
        assert _labels(ledger_dir) == [] and len(ask.prompts) == 1

    def test_eof_quits(self, ledger_dir) -> None:
        def ask(prompt: str) -> str:
            raise EOFError
        assert ledger_cli.main(['review', '--point', 'stall', '--dir',
                                str(ledger_dir)], ask=ask, out=Out()) == 0
        assert _labels(ledger_dir) == []

    def test_unknown_answer_reasks(self, ledger_dir) -> None:
        out, ask = Out(), _ask(['maybe', 'y', 'q'])
        ledger_cli.main(['review', '--point', 'stall', '--n', '2', '--dir',
                         str(ledger_dir)], ask=ask, out=out)
        assert len(ask.prompts) == 3
        assert [r['label'] for r in _labels(ledger_dir)] == ['yes']

    def test_already_labelled_rows_are_not_shown(self, ledger_dir) -> None:
        _write(ledger_dir, 'labels', [
            {'target_id': 'stall:stall:s3', 'label': 'yes',
             'labeller': 'user', 'ts': 'x', 'note': ''}])
        out, ask = Out(), _ask(['n', 'n'])
        ledger_cli.main(['review', '--point', 'stall', '--dir',
                         str(ledger_dir)], ask=ask, out=out)
        assert 's3' not in out.text
        written = [r['target_id'] for r in _labels(ledger_dir)
                   if r.get('note')]
        assert written == ['stall:stall:s2', 'stall:stall:s1']

    def test_n_limits_the_queue(self, ledger_dir) -> None:
        ask = _ask(['y'])
        ledger_cli.main(['review', '--point', 'stall', '--n', '1', '--dir',
                         str(ledger_dir)], ask=ask, out=Out())
        assert len(ask.prompts) == 1

    def test_nothing_to_review(self, ledger_dir) -> None:
        out = Out()
        assert ledger_cli.main(['review', '--point', 'nope', '--dir',
                                str(ledger_dir)], ask=_ask([]), out=out) == 0
        assert 'nothing to review' in out.text

    def test_restores_the_ledger_repository(self, ledger_dir,
                                            monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        ledger.set_repository(None)
        ledger_cli.main(['review', '--point', 'stall', '--n', '1', '--dir',
                         str(ledger_dir)], ask=_ask(['y']), out=Out())
        assert ledger.repository() is None
        assert config.LEDGER_ENABLED is False
        assert len(_labels(ledger_dir)) == 1     # written despite the flag


class TestReport:
    def test_numbers_and_threshold(self, ledger_dir) -> None:
        _write(ledger_dir, 'labels', [
            {'target_id': 'stall:stall:s1', 'label': 'yes', 'ts': '1'},
            {'target_id': 'stall:stall:s2', 'label': 'no', 'ts': '2'},
            {'target_id': 'stall:stall:s3', 'label': 'no', 'ts': '3'}])
        out = Out()
        assert ledger_cli.main(['report', '--dir', str(ledger_dir)],
                               out=out) == 0
        text = out.text
        assert 'stall / fake' in text and 'panel / fake' in text
        assert 'rows 3' in text and 'labelled 3' in text
        # judge: s1 tp, s3 fp, s2 tn -> precision 50%, recall 100%
        assert 'judge     precision 50%  recall 100%' in text
        # heuristic: s1 fn, s2 fp, s3 fp -> precision 0%, recall 0%
        assert 'heuristic precision 0%  recall 0%' in text
        # only s1 (0.9) is yes; 0.7 is no -> t in (0.7, 0.9] -> 0.75
        assert 'suggested 0.75' in text
        assert 'agreement with heuristic 33%' in text     # s3 agrees

    def test_point_filter(self, ledger_dir) -> None:
        out = Out()
        ledger_cli.main(['report', '--point', 'panel', '--dir',
                         str(ledger_dir)], out=out)
        assert 'panel / fake' in out.text and 'stall' not in out.text
        assert 'threshold  n/a' in out.text

    def test_missing_dir(self, tmp_path) -> None:
        out = Out()
        assert ledger_cli.main(['report', '--dir', str(tmp_path / 'x')],
                               out=out) == 0
        assert 'no decision rows' in out.text


class TestTasks:
    def test_unlabelled_listing(self, tmp_path) -> None:
        d = tmp_path / 'ledger'
        _write(d, 'tasks', [
            _task('a', ts='1'), _task('b', ts='2'),
            _task('c', ts='3', status='running'),
            _task('d', ts='4', status='error', seconds=None, cost_usd=None)])
        _write(d, 'labels', [{'target_id': 'b', 'label': 'good'}])
        out = Out()
        assert ledger_cli.main(['tasks', '--unlabelled', '--dir', str(d)],
                               out=out) == 0
        text = out.text
        assert text.index('do d') < text.index('do a')
        assert 'do b' not in text and 'do c' not in text
        assert 'Ollama|qwen3:8b' in text and 'local-only' in text
        assert '/t/d.gz' in text and 'error' in text
        assert '2.5s' in text and '$0.0000' in text
        assert 'n/a' in text                                # d's seconds

    def test_missing_kind_prints_dashes(self, tmp_path) -> None:
        d = tmp_path / 'ledger'
        _write(d, 'tasks', [{'task_id': 'z', 'status': 'done', 'ts': '1',
                             'task': 'old row'}])
        out = Out()
        ledger_cli.main(['tasks', '--dir', str(d)], out=out)
        assert '  -/-  ' in out.text and 'unlabelled' not in out.text

    def test_n_and_all_tasks(self, tmp_path) -> None:
        d = tmp_path / 'ledger'
        _write(d, 'tasks', [_task('a', ts='1'), _task('b', ts='2')])
        _write(d, 'labels', [{'target_id': 'b', 'label': 'good'}])
        out = Out()
        ledger_cli.main(['tasks', '--n', '1', '--dir', str(d)], out=out)
        assert 'do b' in out.text and 'do a' not in out.text

    def test_empty(self, tmp_path) -> None:
        out = Out()
        assert ledger_cli.main(['tasks', '--unlabelled', '--dir',
                                str(tmp_path)], out=out) == 0
        assert 'no tasks' in out.text


class TestModuleEntry:
    def test_python_m_help(self) -> None:
        proc = subprocess.run([sys.executable, '-m', 'guru.ledger_cli',
                               '--help'], capture_output=True, text=True,
                              cwd=REPO_ROOT)
        assert proc.returncode == 0, proc.stderr
        for cmd in ('review', 'report', 'tasks'):
            assert cmd in proc.stdout
