"""Tests for the decision seam (guru.domain.decisions), shadow mode."""
import threading
import time

import pytest

from guru import config, session
from guru.domain import decisions, ledger


class FakeJudge:
    """Judge answering every noul with a fixed yes-probability."""
    name = 'fake'

    def __init__(self, p_yes: float = 0.9, fail: bool = False) -> None:
        self.p_yes, self.fail, self.calls = p_yes, fail, []

    def ask(self, questions: list) -> list:
        self.calls.append(questions)
        if self.fail:
            raise RuntimeError('judge exploded')
        dist = {'yes': self.p_yes, 'no': round(1 - self.p_yes, 4)}
        return [decisions.Answer(chosen=self.p_yes >= 0.5, dist=dist,
                                 confidence=abs(self.p_yes - 0.5) * 2,
                                 judge=self.name, ms=3) for _ in questions]


def _noul(qid='q1') -> decisions.Question:
    return decisions.Question(id=qid, kind=decisions.NOUL,
                              instructions='Is it a stall?', state='Let me…',
                              hypothesis='This reply is a stall.')


class TestShadow:
    """shadow(): judge on a daemon worker, one decisions row per question."""

    @pytest.fixture(autouse=True)
    def _repo(self, fake_repo) -> None:
        decisions.clear_judges()
        self.repo = fake_repo

    def test_worker_is_a_daemon_thread(self, monkeypatch) -> None:
        self._arm(monkeypatch, FakeJudge())
        decisions.shadow('stall', [_noul()], heuristic=True)
        decisions.flush()
        thread = decisions._shadow_worker._thread
        assert thread is not None and thread.daemon is True
        assert thread.is_alive()
        [r] = self._rows()
        assert isinstance(r['queued_ms'], int) and r['queued_ms'] >= 0

    def test_flush_with_timeout_never_raises(self, monkeypatch) -> None:
        decisions.flush(timeout=0.5)

    def _arm(self, monkeypatch, judge, mode='shadow'):
        monkeypatch.setattr(config, 'DECISIONS_MODE', mode)
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        decisions.set_judge('stall', judge)

    def _rows(self):
        decisions.flush()
        ledger.flush()
        return [r for s, r in self.repo.rows if s == 'decisions']

    def test_off_mode_never_calls_judge_or_writes(self, monkeypatch):
        j = FakeJudge()
        self._arm(monkeypatch, j, mode='off')
        decisions.shadow('stall', [_noul()], heuristic=True)
        assert self._rows() == [] and j.calls == []

    def test_unregistered_point_is_a_noop(self, monkeypatch):
        self._arm(monkeypatch, FakeJudge())
        decisions.shadow('panel', [_noul()], heuristic=False)
        assert self._rows() == []

    def test_logs_one_row_per_question_with_keys(self, monkeypatch):
        self._arm(monkeypatch, FakeJudge(p_yes=0.9))
        monkeypatch.setattr(session, 'agent_id', 'agent3')
        monkeypatch.setattr(session, 'task_id', 't7')
        monkeypatch.setattr(session, 'turn_id', 'u2')
        decisions.shadow('stall', [_noul('a'), _noul('b')], heuristic=False)
        rows = self._rows()
        assert [r['question'] for r in rows] == ['a', 'b']
        r = rows[0]
        assert r['point'] == 'stall' and r['mode'] == 'shadow'
        assert r['judge'] == 'fake' and r['chosen'] is True
        assert r['heuristic'] is False and r['agree'] is False
        assert r['dist'] == {'yes': 0.9, 'no': 0.1}
        assert r['agent'] == 'agent3' and r['task_id'] == 't7'
        assert r['turn_id'] == 'u2' and r['run_id'] == ledger.RUN_ID
        assert r['input_head'].startswith('Let me') and r['input_sha']

    def test_judge_failure_is_logged_not_raised(self, monkeypatch):
        self._arm(monkeypatch, FakeJudge(fail=True))
        decisions.shadow('stall', [_noul()], heuristic=True)
        rows = self._rows()
        assert len(rows) == 1 and 'judge exploded' in rows[0]['error']
        assert rows[0]['chosen'] is None

    def test_keys_captured_on_caller_thread(self, monkeypatch):
        gate = threading.Event()

        class BlockingJudge(FakeJudge):
            def ask(self, questions):
                gate.wait(5)
                return super().ask(questions)
        self._arm(monkeypatch, BlockingJudge())
        monkeypatch.setattr(session, 'turn_id', 'original')
        decisions.shadow('stall', [_noul()], heuristic=True)
        monkeypatch.setattr(session, 'turn_id', 'later')
        gate.set()
        rows = self._rows()
        assert len(rows) == 1 and rows[0]['turn_id'] == 'original'

    def test_shadow_rows_preview_the_configured_threshold(self, monkeypatch):
        self._arm(monkeypatch, FakeJudge(p_yes=0.55))
        monkeypatch.setattr(config, 'DECISIONS_THRESHOLDS', {'stall': 0.6})
        decisions.shadow('stall', [_noul()], heuristic=False)
        [r] = self._rows()
        assert r['chosen'] is False and r['threshold'] == 0.6
        assert r['agree'] is True and r['dist']['yes'] == 0.55

    def test_full_queue_drops_with_one_warning(self, monkeypatch, caplog):
        gate = threading.Event()

        class BlockingJudge(FakeJudge):
            def ask(self, questions):
                gate.wait(5)
                return super().ask(questions)
        self._arm(monkeypatch, BlockingJudge())
        monkeypatch.setattr(decisions, 'QUEUE_MAX', 2)
        worker = decisions._Worker('test-shadow')
        monkeypatch.setattr(decisions, '_shadow_worker', worker)
        try:
            with caplog.at_level('WARNING', logger='guru'):
                for _ in range(6):
                    decisions.shadow('stall', [_noul()], heuristic=True)
            drops = [r for r in caplog.records if 'dropp' in r.getMessage()]
            assert len(drops) == 1
        finally:
            gate.set()
            worker.flush(2)

    def test_short_answer_list_is_padded_with_error(self, monkeypatch):
        class ShortJudge(FakeJudge):
            def ask(self, questions):
                return super().ask(questions)[:1]
        self._arm(monkeypatch, ShortJudge())
        decisions.shadow('stall', [_noul('a'), _noul('b'), _noul('c')],
                         heuristic=True)
        rows = self._rows()
        assert [r['question'] for r in rows] == ['a', 'b', 'c']
        assert rows[0]['chosen'] is True
        assert rows[1]['chosen'] is None and rows[2]['chosen'] is None
        assert all('1 answers for 3 questions' in r['error'] for r in rows)

    def test_non_answer_object_is_logged_not_raised(self, monkeypatch):
        class WeirdJudge(FakeJudge):
            def ask(self, questions):
                return ['not an answer' for _ in questions]
        self._arm(monkeypatch, WeirdJudge())
        decisions.shadow('stall', [_noul()], heuristic=True)
        rows = self._rows()
        assert len(rows) == 1 and rows[0]['chosen'] is None
        assert rows[0]['agree'] is None and 'non-Answer' in rows[0]['error']
        assert rows[0]['judge'] == 'fake'


class TestDecide:
    """decide(): the judge answers synchronously for active points, bounded
    by the timeout; everything else falls back to the heuristic."""

    @pytest.fixture(autouse=True)
    def _repo(self, fake_repo, monkeypatch) -> None:
        decisions.clear_judges()
        decisions.reset_breakers()
        self.repo = fake_repo
        monkeypatch.setattr(config, 'DECISIONS_MODE', 'active')
        monkeypatch.setattr(config, 'DECISIONS_ACTIVE', {'stall': True})
        monkeypatch.setattr(config, 'DECISIONS_THRESHOLDS', {})
        monkeypatch.setattr(config, 'DECISIONS_TIMEOUT_MS', 500)
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)

    def _rows(self):
        decisions.flush()
        ledger.flush()
        return [r for s, r in self.repo.rows if s == 'decisions']

    def test_judge_decides_and_row_says_so(self, monkeypatch):
        decisions.set_judge('stall', FakeJudge(p_yes=0.9))
        monkeypatch.setattr(session, 'turn_id', 'u9')
        assert decisions.decide('stall', _noul(), heuristic=False) is True
        [r] = self._rows()
        assert r['mode'] == 'active' and r['used'] == 'judge'
        assert r['fallback_reason'] == '' and r['chosen'] is True
        assert r['heuristic'] is False and r['agree'] is False
        assert r['turn_id'] == 'u9' and r['dist'] == {'yes': 0.9, 'no': 0.1}
        assert isinstance(r['queued_ms'], int) and r['ms'] == 3

    def test_active_false_in_active_mode_is_shadow(self, monkeypatch):
        monkeypatch.setattr(config, 'DECISIONS_ACTIVE', {'stall': False})
        decisions.set_judge('stall', FakeJudge(p_yes=0.9))
        assert decisions.decide('stall', _noul(), heuristic=False) is False
        [r] = self._rows()
        assert r['mode'] == 'shadow' and r['fallback_reason'] == 'not_active'

    def test_noul_without_dist_yes_uses_chosen(self):
        class NoDist(FakeJudge):
            def ask(self, questions):
                return [decisions.Answer(chosen=True, dist={}, confidence=0.5,
                                         judge=self.name, ms=1)
                        for _ in questions]
        decisions.set_judge('stall', NoDist())
        assert decisions.decide('stall', _noul(), heuristic=False) is True
        [r] = self._rows()
        assert r['used'] == 'judge' and r['chosen'] is True

    def test_queued_shadow_batch_does_not_delay_active(self, monkeypatch):
        gate = threading.Event()

        class SlowShadow(FakeJudge):
            def ask(self, questions):
                gate.wait(5)
                return super().ask(questions)
        decisions.set_judge('panel', SlowShadow())
        decisions.set_judge('stall', FakeJudge(p_yes=0.9))
        try:
            decisions.shadow('panel', [_noul('a'), _noul('b')])
            t0 = time.monotonic()
            assert decisions.decide('stall', _noul(), heuristic=False) is True
            assert time.monotonic() - t0 < 0.4
            ledger.flush()
            rows = [r for s, r in self.repo.rows if s == 'decisions']
            assert [r['point'] for r in rows] == ['stall']
            assert rows[0]['used'] == 'judge'
        finally:
            gate.set()
            decisions.flush()

    def test_breaker_opens_after_consecutive_timeouts_and_closes(
            self, monkeypatch, caplog):
        gate = threading.Event()

        class Slow(FakeJudge):
            def ask(self, questions):
                gate.wait(5)
                return super().ask(questions)
        judge = Slow(p_yes=0.9)
        decisions.set_judge('stall', judge)
        monkeypatch.setattr(config, 'DECISIONS_TIMEOUT_MS', 20)
        monkeypatch.setattr(config, 'DECISIONS_BREAKER_TIMEOUTS', 3)
        monkeypatch.setattr(config, 'DECISIONS_BREAKER_COOLDOWN_S', 0.2)
        try:
            with caplog.at_level('WARNING', logger='guru'):
                for _ in range(5):
                    assert decisions.decide('stall', _noul(),
                                            heuristic=False) is False
            ledger.flush()
            rows = [r for s, r in self.repo.rows if s == 'decisions']
            assert [r['fallback_reason'] for r in rows] == [
                'timeout', 'timeout', 'timeout', 'breaker', 'breaker']
            assert len(judge.calls) <= 3
            opened = [r for r in caplog.records if 'breaker' in r.getMessage()]
            assert len(opened) == 1
            gate.set()                      # judge answers again
            decisions.flush()
            time.sleep(0.25)                # cooldown over
            assert decisions.decide('stall', _noul(), heuristic=False) is True
            ledger.flush()
            rows = [r for s, r in self.repo.rows if s == 'decisions']
            assert rows[-1]['used'] == 'judge'
        finally:
            gate.set()
            decisions.flush()

    def test_success_resets_the_timeout_count(self, monkeypatch):
        monkeypatch.setattr(config, 'DECISIONS_BREAKER_TIMEOUTS', 2)
        decisions._breaker_note('stall', timed_out=True)
        decisions._breaker_note('stall', timed_out=False)
        decisions._breaker_note('stall', timed_out=True)
        assert not decisions._breaker_open('stall')
        decisions._breaker_note('stall', timed_out=True)
        assert decisions._breaker_open('stall')

    def test_threshold_is_respected(self, monkeypatch):
        decisions.set_judge('stall', FakeJudge(p_yes=0.55))
        monkeypatch.setattr(config, 'DECISIONS_THRESHOLDS', {'stall': 0.6})
        assert decisions.decide('stall', _noul(), heuristic=True) is False
        [r] = self._rows()
        assert r['chosen'] is False and r['used'] == 'judge'
        assert r['threshold'] == 0.6

    def test_default_threshold_is_half(self):
        decisions.set_judge('stall', FakeJudge(p_yes=0.5))
        assert decisions.decide('stall', _noul(), heuristic=False) is True

    def test_timeout_falls_back_to_heuristic(self, monkeypatch):
        gate = threading.Event()

        class SlowJudge(FakeJudge):
            def ask(self, questions):
                gate.wait(5)
                return super().ask(questions)
        monkeypatch.setattr(config, 'DECISIONS_TIMEOUT_MS', 50)
        decisions.set_judge('stall', SlowJudge(p_yes=0.9))
        try:
            assert decisions.decide('stall', _noul(), heuristic=False) is False
            ledger.flush()
            [r] = [r for s, r in self.repo.rows if s == 'decisions']
            assert r['used'] == 'heuristic'
            assert r['fallback_reason'] == 'timeout'
            assert r['mode'] == 'active' and r['chosen'] is None
        finally:
            gate.set()
            decisions.flush()

    def test_judge_error_falls_back_to_heuristic(self):
        decisions.set_judge('stall', FakeJudge(fail=True))
        assert decisions.decide('stall', _noul(), heuristic=True) is True
        [r] = self._rows()
        assert r['used'] == 'heuristic' and r['fallback_reason'] == 'error'
        assert 'judge exploded' in r['error']

    def test_undecided_answer_is_an_error_fallback(self):
        class Undecided(FakeJudge):
            def ask(self, questions):
                return [decisions.Answer(chosen=None, dist={}, confidence=0.0,
                                         judge=self.name, ms=1)
                        for _ in questions]
        decisions.set_judge('stall', Undecided())
        assert decisions.decide('stall', _noul(), heuristic=True) is True
        [r] = self._rows()
        assert r['used'] == 'heuristic' and r['fallback_reason'] == 'error'

    def test_no_judge_falls_back_with_reason(self):
        assert decisions.decide('stall', _noul(), heuristic=True) is True
        [r] = self._rows()
        assert r['used'] == 'heuristic' and r['fallback_reason'] == 'no_judge'
        assert r['mode'] == 'active'

    def test_point_not_active_stays_shadow(self, monkeypatch):
        judge = FakeJudge(p_yes=0.9)
        decisions.set_judge('panel', judge)
        assert decisions.decide('panel', _noul(), heuristic=False) is False
        [r] = self._rows()
        assert r['mode'] == 'shadow' and r['used'] == 'heuristic'
        assert r['fallback_reason'] == 'not_active' and r['chosen'] is True
        assert judge.calls

    def test_shadow_mode_returns_heuristic_and_shadows(self, monkeypatch):
        monkeypatch.setattr(config, 'DECISIONS_MODE', 'shadow')
        decisions.set_judge('stall', FakeJudge(p_yes=0.9))
        assert decisions.decide('stall', _noul(), heuristic=False) is False
        [r] = self._rows()
        assert r['mode'] == 'shadow' and r['used'] == 'heuristic'
        assert r['fallback_reason'] == '' and r['chosen'] is True

    def test_off_mode_returns_heuristic_silently(self, monkeypatch):
        monkeypatch.setattr(config, 'DECISIONS_MODE', 'off')
        j = FakeJudge(p_yes=0.9)
        decisions.set_judge('stall', j)
        assert decisions.decide('stall', _noul(), heuristic=False) is False
        assert self._rows() == [] and j.calls == []

    def test_never_raises(self, monkeypatch):
        monkeypatch.setattr(decisions, '_run_with_timeout',
                            lambda *a, **k: 1 / 0)
        decisions.set_judge('stall', FakeJudge())
        assert decisions.decide('stall', _noul(), heuristic=True) is True

    def test_shadow_rows_in_active_mode_are_marked_shadow(self):
        decisions.set_judge('panel', FakeJudge())
        decisions.shadow('panel', [_noul()], heuristic=None)
        [r] = self._rows()
        assert r['mode'] == 'shadow' and r['used'] == 'heuristic'
        assert r['fallback_reason'] == 'not_active'


class TestRunWithTimeout:
    """_run_with_timeout(): a bounded wait on the daemon worker."""

    def test_returns_result_in_time(self):
        assert decisions._run_with_timeout(lambda: 42, 1.0) == (42, None,
                                                                False)

    def test_reports_exception(self):
        def boom():
            raise ValueError('nope')
        value, error, timed_out = decisions._run_with_timeout(boom, 1.0)
        assert value is None and isinstance(error, ValueError)
        assert timed_out is False

    def test_times_out(self):
        gate = threading.Event()
        try:
            value, error, timed_out = decisions._run_with_timeout(
                lambda: gate.wait(5), 0.05)
            assert value is None and error is None and timed_out is True
        finally:
            gate.set()
            decisions.flush()


class TestQuestionBuilders:
    """The questions guru's decision points ask."""
    def test_stall_question(self) -> None:
        q = decisions.stall_question("I'll read the files:")
        assert q.kind == decisions.NOUL and q.id == 'stall'
        assert list(q.options) == ['yes', 'no'] and q.hypothesis

    def test_panel_questions(self) -> None:
        qs = decisions.panel_questions('review the login endpoint')
        assert [q.id for q in qs] == ['needs_security', 'needs_architect',
                                      'needs_sre']
        assert all('review the login endpoint' in q.state for q in qs)

    def test_injection_question_truncates(self) -> None:
        q = decisions.injection_question('x' * 10000, 'https://a.b/c')
        assert q.id == 'injection' and len(q.state) <= 4100

    def test_noul_requires_yes_no_options(self) -> None:
        with pytest.raises(ValueError):
            decisions.Question(id='x', kind=decisions.NOUL, instructions='?',
                               state='s', options={'a': 'A', 'b': 'B'})
        decisions.Question(id='x', kind=decisions.CHOICE, instructions='?',
                           state='s', options={'a': 'A', 'b': 'B'})
