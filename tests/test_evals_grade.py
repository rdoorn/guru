"""Tests for offline re-grading (guru.evals.grading and ``python -m
guru.evals grade``): a stored run, fake judges, the labels file, the
run's ledger and the CLI table. No model, no case re-run."""
import gzip
import json
from pathlib import Path

import pytest

from guru import config, judges
from guru.domain import ledger
from guru.evals import grading, labels, rubric, runs
from guru.evals.__main__ import main as cli_main
from guru.evals.runs import CaseResult, Run
from guru.repositories.jsonl_ledger import JsonlLedger


class FakeJudge:
    """A :class:`rubric.Judge` with a canned reply (or an exception)."""

    def __init__(self, reply, model: str = 'judge-model') -> None:
        self.reply, self.model, self.prompts = reply, model, []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _transcript(path: Path, prompt: str = 'the prompt',
                answer: str = 'the answer') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{'title': 'main', 'model': 'm', 'messages': [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': prompt},
        {'role': 'tool', 'tool_name': 'read_file', 'content': 'x'},
        {'role': 'assistant', 'content': ''},
        {'role': 'assistant', 'content': answer}]},
        {'title': 'agent1', 'model': 'm', 'messages': [
            {'role': 'user', 'content': 'task'},
            {'role': 'assistant', 'content': 'child answer'}]}]
    with gzip.open(path, 'wt', encoding='utf-8') as fh:
        json.dump(data, fh)


def _case(out_root: Path, run_id: str, name: str, rubric_text: str = 'r',
          answer: str = 'the answer', with_transcript: bool = True,
          **observed) -> CaseResult:
    tpath = out_root / run_id / 'transcripts' / f'{name}.json.gz'
    if with_transcript:
        _transcript(tpath, prompt=f'prompt for {name}', answer='t-answer')
    obs = {'answer': answer, 'seconds': 12.34,
           'files_changed': ['guru/cli.py'], 'fixture_tests_pass': True,
           'tools_used': ['edit_file', 'run_tests'], 'gate_verdicts': [],
           'spawned': 0, 'roles': [], 'timed_out': False, 'error': ''}
    obs.update(observed)
    return CaseResult(case=name, passed=True, checks=[], observed=obs,
                      rubric=rubric_text, transcript_path=str(tpath),
                      cost_usd=0.25)


def _run(out_root: Path, run_id: str = 'abc123abc123', **kw) -> Run:
    run = Run(run_id=run_id, ts='2026-09-25T07:22:58+00:00', model='A|m',
              git_sha='deadbeef', cases=[
                  _case(out_root, run_id, 'graded'),
                  _case(out_root, run_id, 'plain', rubric_text=''),
                  _case(out_root, run_id, 'empty', answer='',
                        with_transcript=False),
                  _case(out_root, run_id, 'from-transcript', answer='')],
              **kw)
    runs.save(run, out_root)
    return run


HAND = labels.parse_labels('''
[[label]]
case = "graded"
run = "*"
score = 2
note = "hand two"

[[label]]
case = "graded"
run = "otherrun00000"
score = 0

[[label]]
case = "empty"
score = 1
''')


class TestRunLookupAndTranscripts:
    def test_find_run_by_id_or_path(self, tmp_path: Path) -> None:
        run = _run(tmp_path)
        path = runs.find_run(tmp_path, run.run_id)
        assert path.name.endswith(f'-{run.run_id}.json')
        assert runs.find_run(tmp_path, str(path)) == path
        assert runs.load(path).run_id == run.run_id

    def test_find_run_errors(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no run 'nope'"):
            runs.find_run(tmp_path, 'nope')
        (tmp_path / '20260101T000000Z-dup.json').write_text('{}')
        (tmp_path / '20260102T000000Z-dup.json').write_text('{}')
        with pytest.raises(ValueError, match='ambiguous'):
            runs.find_run(tmp_path, 'dup')

    def test_transcript_prompt_and_answer(self, tmp_path: Path) -> None:
        p = tmp_path / 't.json.gz'
        _transcript(p, prompt='ask this', answer='final words')
        t = runs.load_transcript(p)
        assert runs.transcript_prompt(t) == 'ask this'
        assert runs.transcript_answer(t) == 'final words'
        assert runs.transcript_prompt([]) == ''
        assert runs.transcript_answer([{'title': 'm', 'model': 'x',
                                        'messages': []}]) == ''

    def test_load_transcript_errors(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='not a transcript'):
            runs.load_transcript(tmp_path / 'missing.json.gz')
        bad = tmp_path / 'bad.json.gz'
        with gzip.open(bad, 'wt') as fh:
            json.dump({'not': 'a list'}, fh)
        with pytest.raises(ValueError, match='not a transcript'):
            runs.load_transcript(bad)


class TestAnswerAndPrompt:
    def test_answer_from_observed_prompt_from_transcript(self, tmp_path):
        res = _case(tmp_path, 'rid', 'c1')
        assert grading.answer_and_prompt(res) == ('the answer',
                                                  'prompt for c1')

    def test_falls_back_to_transcript_answer(self, tmp_path: Path) -> None:
        res = _case(tmp_path, 'rid', 'c2', answer='')
        assert grading.answer_and_prompt(res) == ('t-answer',
                                                  'prompt for c2')

    def test_no_transcript_uses_case_file_prompt(self, tmp_path: Path):
        cdir = tmp_path / 'cases'
        cdir.mkdir()
        (cdir / 'c3.toml').write_text(
            'name = "c3"\nfixture = "docs-only"\nprompt = "from file"\n')
        res = _case(tmp_path, 'rid', 'c3', with_transcript=False)
        assert grading.answer_and_prompt(res, cases_dir=cdir) == \
            ('the answer', 'from file')
        res = _case(tmp_path, 'rid', 'unknown-case', with_transcript=False)
        assert grading.answer_and_prompt(res, cases_dir=cdir) == \
            ('the answer', '')


class TestRegrade:
    def test_grades_every_rubric_case_per_judge_and_records_labels(
            self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        prev_repo = ledger.repository()
        run = _run(tmp_path)
        haiku = FakeJudge('{"score": 0, "reason": "no code shown"}',
                          model='haiku')
        sonnet = FakeJudge('{"score": 2, "reason": "evidence shows it"}',
                           model='sonnet')
        out = grading.regrade(run, tmp_path, [('A|haiku', haiku),
                                              ('A|sonnet', sonnet)], HAND)
        assert out.run_id == run.run_id
        assert out.specs == ['A|haiku', 'A|sonnet']
        assert [r.case for r in out.rows] == ['graded', 'empty',
                                              'from-transcript']
        graded = out.rows[0]
        assert graded.grades['A|haiku'] == rubric.Grade(0, 'no code shown')
        assert graded.grades['A|sonnet'] == rubric.Grade(
            2, 'evidence shows it')
        assert graded.hand == labels.HandLabel('graded', '*', 2, 'hand two')
        # The empty answer scores 0 without a call, on both judges.
        empty = out.rows[1]
        assert empty.grades == {'A|haiku': rubric.Grade(0, 'empty answer'),
                                'A|sonnet': rubric.Grade(0, 'empty answer')}
        assert empty.hand is not None and empty.hand.score == 1
        assert out.rows[2].hand is None
        # Two graded cases with an answer x two judges = 4 calls.
        assert len(haiku.prompts) == 2 and len(sonnet.prompts) == 2
        # Agreement: graded (0 vs 2, 2 vs 2) and empty (0 vs 1) compared.
        assert out.agreement == {'A|haiku': (0, 2), 'A|sonnet': (1, 2)}
        # The packet: prompt from the transcript, evidence, fenced answer.
        packet = haiku.prompts[0]
        assert 'prompt for graded' in packet
        assert rubric.EVIDENCE_HEADER in packet
        assert '- files changed: guru/cli.py' in packet
        assert '- cost: $0.250' in packet
        assert '<<<ANSWER ' in packet and 'the answer' in packet
        # Labels rows in the run's ledger dir: judges and hand.
        assert out.ledger_dir == tmp_path / run.run_id / 'ledger'
        rows = JsonlLedger(out.ledger_dir).rows('labels')
        by = {(r['target_id'], r['labeller']): (r['label'], r['note'])
              for r in rows}
        rid = run.run_id
        assert by[(f'{rid}:graded', 'rubric:haiku')] == (
            '0', 'no code shown')
        assert by[(f'{rid}:graded', 'rubric:sonnet')] == (
            '2', 'evidence shows it')
        assert by[(f'{rid}:graded', 'hand')] == ('2', 'hand two')
        assert by[(f'{rid}:empty', 'hand')] == ('1', '')
        assert (f'{rid}:from-transcript', 'hand') not in by
        assert (f'{rid}:plain', 'rubric:haiku') not in by
        assert len(rows) == 2 * 3 + 2
        assert out.cost_usd is None                  # fake judges: no calls
        # Restored.
        assert ledger.repository() is prev_repo
        assert config.LEDGER_ENABLED is False

    def test_failing_judge_is_an_error_cell_not_a_crash(self, tmp_path,
                                                        monkeypatch):
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        run = _run(tmp_path)
        boom = FakeJudge(RuntimeError('provider down'), model='boom')
        garbage = FakeJudge('not json', model='garbage')
        out = grading.regrade(run, tmp_path, [('A|boom', boom),
                                              ('A|garbage', garbage)], [])
        graded = out.rows[0]
        assert graded.grades == {'A|boom': None, 'A|garbage': None}
        assert graded.errors['A|boom'] == 'provider down'
        assert 'no JSON object' in graded.errors['A|garbage']
        assert out.agreement == {'A|boom': (0, 0), 'A|garbage': (0, 0)}
        rows = JsonlLedger(out.ledger_dir).rows('labels')
        # Only the empty-answer zeros were recorded (no judge call).
        assert {r['target_id'] for r in rows} == {f'{run.run_id}:empty'}

    def test_run_specific_hand_grade_wins(self, tmp_path: Path,
                                          monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        run = _run(tmp_path, run_id='otherrun00000')
        out = grading.regrade(run, tmp_path,
                              [('A|j', FakeJudge('{"score": 0}'))], HAND)
        assert out.rows[0].hand == labels.HandLabel('graded',
                                                    'otherrun00000', 0)
        assert out.agreement == {'A|j': (1, 2)}


class TestShowText:
    def test_label_stub(self) -> None:
        assert grading.label_stub('c', 'rid000000000') == (
            '[[label]]\ncase = "c"\nrun = "rid000000000"\n'
            '# score = 0 | 1 | 2   (uncomment and pick one)\n'
            'note = "hand grade <date>: <why>"')
        stub = grading.label_stub('c', 'rid', score=2, note='why')
        assert stub == '[[label]]\ncase = "c"\nrun = "rid"\nscore = 2\n' \
            'note = "why"'
        assert labels.parse_labels(stub) == [labels.HandLabel('c', 'rid', 2,
                                                              'why')]
        # The placeholder stub is valid TOML too (the score line is a
        # comment), so a pasted-but-unfinished entry fails loudly on the
        # missing score rather than on a syntax error.
        with pytest.raises(ValueError, match='score must be one of'):
            labels.parse_labels(grading.label_stub('c', 'rid'))

    def test_show_text_names_hand_grade_scope(self, tmp_path: Path) -> None:
        run = _run(tmp_path)
        res = run.cases[0]
        text = grading.show_text(run, res, 'ans', 'ask',
                                 hand=labels.HandLabel('graded', run.run_id,
                                                       1))
        assert text.startswith(f'=== graded (run {run.run_id}) — hand '
                               f'grade 1 for run {run.run_id}\n')
        text = grading.show_text(run, res, 'ans', 'ask')
        assert text.startswith(f'=== graded (run {run.run_id})\n')
        assert 'hand grade' not in text.splitlines()[0]
        assert rubric.evidence(res.observed, res.cost_usd) in text
        assert 'Rubric:\nr\n' in text and 'Answer:\nans' in text


class _Adapter:
    name = 'Fake'
    remote = False

    def complete(self, prompt, max_tokens=1024, model=''):
        return '{"score": 1, "reason": "half"}'


class TestResolveJudges:
    @pytest.fixture(autouse=True)
    def _clear(self):
        yield
        judges.set_registry(None)

    def test_resolves_specs_over_the_adapters_and_clears_registry(self):
        got = grading.resolve_judges(['Fake|a', 'Fake|b'],
                                     adapters=[_Adapter()])  # type: ignore
        assert [spec for spec, _ in got] == ['Fake|a', 'Fake|b']
        assert [j.model for _, j in got] == ['a', 'b']
        assert rubric.judge_from_spec('Fake|a') is None   # registry cleared

    @pytest.mark.parametrize('specs, message', [
        ([], 'at least one --rubric'),
        (['Fake|a', 'Fake|a'], 'given twice'),
        (['Other|a'], "rubric judge 'Other|a'"),
        (['Fake'], "rubric judge 'Fake'"),
    ])
    def test_errors(self, specs, message) -> None:
        adapters = [_Adapter()]
        with pytest.raises(ValueError, match=message):
            grading.resolve_judges(specs, adapters=adapters)  # type: ignore


class TestCli:
    def _setup(self, tmp_path: Path, monkeypatch) -> Run:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        run = _run(tmp_path)
        replies = {'haiku': '{"score": 0, "reason": "terse"}',
                   'sonnet': '{"score": 2, "reason": "full"}'}

        def fake_resolve(specs, adapters=None):
            return [(s, FakeJudge(replies[s.partition('|')[2]],
                                  model=s.partition('|')[2])) for s in specs]
        monkeypatch.setattr(grading, 'resolve_judges', fake_resolve)
        labels_file = tmp_path / 'labels.toml'
        labels_file.write_text(
            '[[label]]\ncase = "graded"\nscore = 2\nnote = "n"\n')
        self.labels_file = labels_file
        return run

    def test_table_one_column_per_judge_plus_hand(self, tmp_path, capsys,
                                                  monkeypatch) -> None:
        run = self._setup(tmp_path, monkeypatch)
        code = cli_main(['grade', run.run_id, '--rubric', 'A|haiku',
                         '--rubric', 'A|sonnet', '--out', str(tmp_path),
                         '--labels', str(self.labels_file)])
        assert code == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert lines[0].startswith(f'run {run.run_id} (2026-09-25')
        header = lines[1].split()
        assert header == ['case', 'haiku', 'sonnet', 'hand']
        rows = {ln.split()[0]: ln.split()[1:] for ln in lines[3:6]}
        assert rows['graded'] == ['0', '2', '2']
        assert rows['empty'] == ['0', '0', '-']
        assert rows['from-transcript'] == ['0', '2', '-']
        assert 'plain' not in rows                      # no rubric
        assert 'agreement with hand: A|haiku 0/1 (0%)' in out
        assert 'agreement with hand: A|sonnet 1/1 (100%)' in out
        assert f'labels rows recorded under {tmp_path / run.run_id}' in out
        assert 'grading cost n/a' in out
        rows_written = JsonlLedger(tmp_path / run.run_id / 'ledger').rows(
            'labels')
        assert {r['labeller'] for r in rows_written} == {
            'rubric:haiku', 'rubric:sonnet', 'hand'}

    def test_same_model_on_two_adapters_keeps_full_spec_header(
            self, tmp_path, capsys, monkeypatch) -> None:
        run = self._setup(tmp_path, monkeypatch)
        code = cli_main(['grade', run.run_id, '--rubric', 'A|haiku',
                         '--rubric', 'B|haiku', '--out', str(tmp_path),
                         '--labels', str(self.labels_file)])
        assert code == 0
        header = capsys.readouterr().out.splitlines()[1]
        assert 'A|haiku' in header and 'B|haiku' in header

    def test_error_cell_and_message(self, tmp_path, capsys,
                                    monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        run = _run(tmp_path)
        monkeypatch.setattr(
            grading, 'resolve_judges',
            lambda specs, adapters=None: [
                (s, FakeJudge(RuntimeError('down'), model='x'))
                for s in specs])
        code = cli_main(['grade', run.run_id, '--rubric', 'A|x', '--out',
                         str(tmp_path), '--labels',
                         str(tmp_path / 'none.toml')])
        assert code == 0
        out = capsys.readouterr().out
        assert 'graded' in out and 'err' in out
        assert 'error: graded / A|x: down' in out
        assert 'agreement with hand: A|x 0/0' in out

    @pytest.mark.parametrize('argv_tail, message', [
        (['nope'], "no run 'nope'"),
        (['RUN', '--labels', 'BAD'], 'score must be one of'),
        (['RUN', '--rubric', 'Other|m'], "rubric judge 'Other|m'"),
    ])
    def test_usage_errors_exit_2(self, tmp_path, capsys, monkeypatch,
                                 argv_tail, message) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        run = _run(tmp_path)
        bad = tmp_path / 'bad.toml'
        bad.write_text('[[label]]\ncase = "a"\nscore = 7\n')
        real_resolve = grading.resolve_judges
        monkeypatch.setattr(grading, 'resolve_judges',
                            lambda specs, adapters=None:
                            real_resolve(specs, adapters=[]))
        argv = ['grade', '--out', str(tmp_path)]
        tail = [run.run_id if a == 'RUN' else str(bad) if a == 'BAD' else a
                for a in argv_tail]
        if '--rubric' not in tail:
            tail += ['--rubric', 'A|m']
        assert cli_main(argv + tail) == 2
        assert message in capsys.readouterr().err

    def test_rubric_or_show_is_required(self, tmp_path, capsys) -> None:
        assert cli_main(['grade', 'rid', '--out', str(tmp_path)]) == 2
        err = capsys.readouterr().err
        assert 'needs --rubric SPEC' in err and '--show' in err

    def test_show_prints_the_packet_per_case_without_a_judge(
            self, tmp_path, capsys, monkeypatch) -> None:
        run = self._setup(tmp_path, monkeypatch)
        calls: list = []
        monkeypatch.setattr(grading, 'regrade',
                            lambda *a, **k: calls.append(a))
        code = cli_main(['grade', run.run_id, '--show', '--out',
                         str(tmp_path), '--labels', str(self.labels_file)])
        assert code == 0 and calls == []                 # no grading
        out = capsys.readouterr().out
        # One block per rubric case, in run order; the plain case is out.
        heads = [ln for ln in out.splitlines() if ln.startswith('=== ')]
        assert heads == [
            f'=== graded (run {run.run_id}) — hand grade 2 (any run)',
            f'=== empty (run {run.run_id})',
            f'=== from-transcript (run {run.run_id})']
        block = out.partition('=== graded')[2].partition('=== empty')[0]
        assert 'Prompt:\nprompt for graded\n' in block
        assert 'Rubric:\nr\n' in block
        assert rubric.EVIDENCE_HEADER in block
        assert '- files changed: guru/cli.py' in block
        assert '- cost: $0.250' in block
        assert 'Answer:\nthe answer\n' in block
        assert '<<<ANSWER' not in block                  # no fence needed
        assert 'Scale: 2 = meets the intent' in block
        assert ('[[label]]\ncase = "graded"\nrun = "%s"\n'
                '# score = 0 | 1 | 2' % run.run_id) in block
        assert 'note = "hand grade <date>: <why>"' in block
        empty = out.partition('=== empty')[2].partition(
            '=== from-transcript')[0]
        assert 'Answer:\n(empty answer)' in empty
        assert 'Prompt:\n(none recorded)' in empty      # no transcript
        assert 'Answer:\nt-answer' in out                # transcript answer

    def test_show_with_a_judge_prints_packets_then_the_table(
            self, tmp_path, capsys, monkeypatch) -> None:
        run = self._setup(tmp_path, monkeypatch)
        code = cli_main(['grade', run.run_id, '--show', '--rubric',
                         'A|sonnet', '--out', str(tmp_path), '--labels',
                         str(self.labels_file)])
        assert code == 0
        out = capsys.readouterr().out
        assert out.index('=== graded') < out.index(f'run {run.run_id} (')
        assert 'agreement with hand: A|sonnet 1/1 (100%)' in out

    def test_run_without_rubric_cases_says_so(self, tmp_path, capsys,
                                              monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        run = Run(run_id='norubric0000', ts=runs.now_ts(), model='A|m',
                  git_sha='', cases=[_case(tmp_path, 'norubric0000', 'p',
                                           rubric_text='')])
        runs.save(run, tmp_path)
        monkeypatch.setattr(grading, 'resolve_judges',
                            lambda specs, adapters=None:
                            [(s, FakeJudge('{"score": 2}')) for s in specs])
        assert cli_main(['grade', 'norubric0000', '--rubric', 'A|m',
                         '--out', str(tmp_path)]) == 0
        assert 'no case carries a rubric' in capsys.readouterr().out
