"""The shared turn loop (guru.adapters.turn): a text reply ends the turn,
duplicates and cut-off calls do not run, the stall monitor replaces the
round cap, changed files are verified before an answer, and the lead's
answer to work goes through the answer check."""
from guru import session, ui
from guru.adapters import turn
from guru.domain import claims, conversation, ledger, tools


class Scripted:
    """Drive ``run_loop`` with scripted rounds; ``run_tools`` threads a tool
    message per call the way an adapter does (a duplicate does not run)."""

    def __init__(self, monkeypatch, rounds, *, lead=True, task_id='',
                 request='build the thing', results=None) -> None:
        monkeypatch.setattr(ui, 'note_thinking', lambda: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        monkeypatch.setattr(ui.console, 'print', lambda *a, **k: None)
        self.rendered: list = []
        monkeypatch.setattr(turn, '_render_answer', self.rendered.append)
        monkeypatch.setattr(session, 'messages', [
            {'role': 'system', 'content': 's'},
            {'role': 'user', 'content': request}])
        monkeypatch.setattr(session, 'cancel_requested', False)
        monkeypatch.setattr(session, 'task_id', task_id)
        monkeypatch.setattr(session, 'can_spawn', lead and not task_id)
        monkeypatch.setattr(session, 'struggle',
                            {k: 0 for k in session.STRUGGLE_KEYS})
        self.rounds = iter(rounds)
        self.results = results or (lambda name, args: f'{name} ran')
        self.ran: list = []
        self.user_msgs: list = []

    def _step(self):
        item = next(self.rounds)
        text, calls = item if isinstance(item, tuple) else (item, [])
        msg: dict = {'role': 'assistant', 'content': text}
        if calls:
            msg['tool_calls'] = [{'id': f'c{i}', 'function': {
                'name': n, 'arguments': a}} for i, (n, a) in
                enumerate(calls)]
        session.messages.append(msg)
        return text, [(n, a, f'c{i}') for i, (n, a) in enumerate(calls)]

    def _execute(self, name, args):
        self.ran.append(name)
        return self.results(name, args)

    def _run_tools(self, pending) -> None:
        for name, args, _ref, dup in pending:
            content = turn.tool_result(name, args, dup,
                                       execute=self._execute)
            session.messages.append({'role': 'tool', 'tool_name': name,
                                     'tool_args': args, 'content': content})

    def _add_user(self, text) -> None:
        self.user_msgs.append(text)
        session.messages.append({'role': 'user', 'content': text})

    def run(self) -> str:
        return turn._drive(self._step, self._run_tools, self._add_user, [])


def _write(path='a.py'):
    return ('write_file', {'path': path, 'content': 'x'})


def _wrote(name, args):
    if name in ('write_file', 'edit_file'):
        return f"Wrote 1 bytes to {args['path']}. (sha:abc123)"
    return f'{name} ran {args}'


class TestEnding:
    def test_text_reply_is_the_answer(self, monkeypatch, fake_repo) -> None:
        s = Scripted(monkeypatch, ['hello'])
        assert s.run() == 'hello'
        assert s.rendered == ['hello']
        assert session.messages[-1] == {'role': 'assistant',
                                        'content': 'hello'}

    def test_tools_then_answer(self, monkeypatch, fake_repo) -> None:
        s = Scripted(monkeypatch, [
            ('', [('read_file', {'path': 'a.py'})]), 'done'])
        assert s.run() == 'done'
        assert s.ran == ['read_file']

    def test_empty_reply_is_a_violation(self, monkeypatch, fake_repo):
        s = Scripted(monkeypatch, [''])
        assert s.run() == '(no answer produced)'
        assert session.struggle['protocol_violation'] == 1

    def test_waiting_turn_ends_without_answer(self, monkeypatch,
                                              fake_repo) -> None:
        def results(name, args):
            session.turn_waiting = True
            return 'Waiting for agent2'
        monkeypatch.setattr(session, 'turn_waiting', False)
        s = Scripted(monkeypatch, [('', [('join', {'targets': 'agent2'})])],
                     results=results)
        assert s.run() == ''
        assert s.rendered == []


class TestCalls:
    def test_duplicate_does_not_run(self, monkeypatch, fake_repo) -> None:
        call = ('read_file', {'path': 'a.py'})
        s = Scripted(monkeypatch, [('', [call]), ('', [call]), 'ok'])
        s.run()
        assert s.ran == ['read_file']
        assert session.messages[-2]['content'].startswith('Already called')

    def test_failed_call_may_be_retried(self, monkeypatch, fake_repo):
        call = ('read_file', {'path': 'a.py'})
        outcomes = iter(['Tool error: busy', 'text'])
        s = Scripted(monkeypatch, [('', [call]), ('', [call]), 'ok'],
                     results=lambda n, a: next(outcomes))
        s.run()
        assert s.ran == ['read_file', 'read_file']

    def test_cut_reply_runs_nothing(self, monkeypatch, fake_repo) -> None:
        s = Scripted(monkeypatch, [('', [_write()]), 'ok'])
        real = s._step

        def step():
            out = real()
            if out[1]:
                session.output_cut = True
            return out
        s._step = step                                  # type: ignore
        s.run()
        assert s.ran == []
        assert turn.OUTPUT_CUT_REFUSAL in session.messages[3]['content']


class TestStall:
    def _quiet_rounds(self, n):
        # the same result every round (a grep with a new pattern that
        # finds the same nothing): no progress
        return [('', [('search_code', {'pattern': f'p{i}'})])
                for i in range(n)]

    def test_worker_warned_then_stopped_with_handoff(
            self, monkeypatch, fake_repo) -> None:
        n = 1 + turn.STALL_ROUNDS + turn.STALL_GRACE + 5
        s = Scripted(monkeypatch, self._quiet_rounds(n), task_id='t1',
                     results=lambda name, args: 'no matches')
        answer = s.run()
        assert answer.startswith('(stalled:')
        assert session.stalled is True
        # the first round is progress (new content); then the quiet streak
        assert len(s.ran) == 1 + turn.STALL_ROUNDS + turn.STALL_GRACE
        warned = [m for m in session.messages if m.get('role') == 'tool'
                  and '[guru]' in m['content']]
        assert len(warned) == 1
        assert session.struggle['stall_warnings'] == 1
        assert session.struggle['stalls'] == 1

    def test_lead_stops_with_its_last_text(self, monkeypatch, fake_repo):
        rounds = [('looking', [('search_code', {'pattern': f'p{i}'})])
                  for i in range(60)]
        s = Scripted(monkeypatch, rounds,
                     results=lambda name, args: 'no matches')
        answer = s.run()
        assert answer.startswith('looking\n\n(guru stopped the turn')

    def test_progress_means_no_cap(self, monkeypatch, fake_repo) -> None:
        rounds = [('', [('read_file', {'path': f'f{i}.py'})])
                  for i in range(100)] + ['done']
        s = Scripted(monkeypatch, rounds,
                     results=lambda name, args: f"text of {args['path']}")
        assert s.run() == 'done'
        assert len(s.ran) == 100

    def test_timings_are_not_progress(self, monkeypatch, fake_repo) -> None:
        count = iter(range(1000))
        n = 1 + turn.STALL_ROUNDS + turn.STALL_GRACE + 5
        s = Scripted(monkeypatch, [
            ('', [('run_tests', {'target': f't{i}'})]) for i in range(n)],
            task_id='t1',
            results=lambda name, a: f'1 failed in {next(count)}.2s')
        assert s.run().startswith('(stalled:')

    def test_progress_after_warning_resets(self, monkeypatch,
                                           fake_repo) -> None:
        m = turn.Monitor()
        for _ in range(turn.STALL_ROUNDS):
            m.saw([])
        assert m.note().startswith('[guru]')
        m.saw([{'role': 'tool', 'tool_name': 'read_file',
                'tool_args': {'path': 'x'}, 'content': 'new'}])
        assert (m.quiet, m.warned, m.stalled()) == (0, False, False)


class TestVerify:
    def test_untested_change_sent_back_once(self, monkeypatch,
                                            fake_repo) -> None:
        monkeypatch.setattr(tools, 'is_enabled', lambda name: False)
        s = Scripted(monkeypatch, [('', [_write()]), 'done', 'done again'],
                     task_id='t1', results=_wrote)
        assert s.run() == 'done again'
        assert len(s.user_msgs) == 1
        assert s.user_msgs[0].startswith(turn.VERIFY_REFUSAL)
        assert 'tests' in s.user_msgs[0] and 'lint' not in s.user_msgs[0]
        assert conversation.is_nudge(s.user_msgs[0])

    def test_tested_change_delivered(self, monkeypatch, fake_repo) -> None:
        monkeypatch.setattr(tools, 'is_enabled', lambda name: False)
        s = Scripted(monkeypatch, [
            ('', [_write()]), ('', [('run_tests', {'target': ''})]), 'done'],
            task_id='t1', results=_wrote)
        assert s.run() == 'done'
        assert s.user_msgs == []

    def test_lint_required_when_enabled(self, monkeypatch, fake_repo):
        monkeypatch.setattr(tools, 'is_enabled', lambda name: True)
        s = Scripted(monkeypatch, [
            ('', [_write()]), ('', [('run_tests', {'target': ''})]), 'done',
            'again'], task_id='t1', results=_wrote)
        s.run()
        assert 'lint' in s.user_msgs[0] and 'tests' not in s.user_msgs[0]


class _Checker:
    def __init__(self, problems) -> None:
        self.problems = problems
        self.calls = 0

    def check(self, request, answer):
        self.calls += 1
        return self.problems


class TestAnswerCheck:
    def test_lead_answer_to_work_sent_back_once(self, monkeypatch,
                                                fake_repo) -> None:
        checker = _Checker(['the store is not installed'])
        monkeypatch.setattr(claims, '_checker', checker)
        s = Scripted(monkeypatch, [
            ('', [('spawn', {'task': 'x'})]), 'all done', 'not wired: X'])
        assert s.run() == 'not wired: X'
        assert checker.calls == 1
        assert s.user_msgs[0].startswith(claims.PREFIX)

    def test_no_work_no_check(self, monkeypatch, fake_repo) -> None:
        checker = _Checker(['x'])
        monkeypatch.setattr(claims, '_checker', checker)
        s = Scripted(monkeypatch, [
            ('', [('read_file', {'path': 'a'})]), 'it does X'])
        assert s.run() == 'it does X'
        assert checker.calls == 0

    def test_worker_never_checked(self, monkeypatch, fake_repo) -> None:
        checker = _Checker(['x'])
        monkeypatch.setattr(claims, '_checker', checker)
        monkeypatch.setattr(tools, 'is_enabled', lambda name: False)
        s = Scripted(monkeypatch, [
            ('', [_write()]), ('', [('run_tests', {'target': ''})]),
            'done'], task_id='t1', results=_wrote)
        assert s.run() == 'done'
        assert checker.calls == 0


class TestChangedPaths:
    def test_apply_work_rows(self) -> None:
        content = ('Applied patch to your sandbox copy:\n'
                   'guru/a.py | +3 -1\nguru/b.py | +0 -2 deleted\n'
                   '2 files, +3 -3')
        assert turn.changed_paths('apply_work', {}, content) == [
            'guru/a.py', 'guru/b.py']

    def test_refused_write_changed_nothing(self) -> None:
        assert turn.changed_paths('write_file', {'path': 'a'},
                                  'Refused: no') == []


def test_turn_record_written(monkeypatch, fake_repo) -> None:
    s = Scripted(monkeypatch, ['hi'])
    turn.run_loop(step=s._step, run_tools=s._run_tools,
                  add_user=s._add_user)
    ledger.flush()
    rows = fake_repo.stream('turns')
    assert len(rows) == 1 and rows[0]['controller_executed'] is False


class TestReviewFixes:
    def test_sandbox_scripts_are_progress(self, monkeypatch,
                                          fake_repo) -> None:
        rounds = [('', [('sandbox_python', {'code': f'edit {i}'})])
                  for i in range(40)] + ['done']
        s = Scripted(monkeypatch, rounds, task_id='t1',
                     results=lambda name, args: 'exit 0 in 0.31s')
        assert s.run() == 'done'
        assert len(s.ran) == 40

    def test_tests_rerun_after_an_edit(self, monkeypatch, fake_repo):
        monkeypatch.setattr(tools, 'is_enabled', lambda name: False)
        test = ('run_tests', {'target': ''})
        s = Scripted(monkeypatch, [('', [test]), ('', [_write()]),
                                   ('', [test]), 'done'],
                     task_id='t1', results=_wrote)
        assert s.run() == 'done'
        assert s.ran == ['run_tests', 'write_file', 'run_tests']
        assert s.user_msgs == []

    def test_refused_test_run_is_not_testing(self, monkeypatch,
                                             fake_repo) -> None:
        monkeypatch.setattr(tools, 'is_enabled', lambda name: False)

        def results(name, args):
            if name == 'run_tests':
                return 'Refused: not now'
            return _wrote(name, args)
        s = Scripted(monkeypatch, [('', [_write()]),
                                   ('', [('run_tests', {'target': ''})]),
                                   'done', 'again'],
                     task_id='t1', results=results)
        assert s.run() == 'again'
        assert s.user_msgs[0].startswith(turn.VERIFY_REFUSAL)

    def test_provider_error_after_work_hands_back(self, monkeypatch,
                                                  fake_repo) -> None:
        s = Scripted(monkeypatch, [('', [_write()])], task_id='t1',
                     results=_wrote)
        real = s._step
        calls = iter([real, lambda: None])
        s._step = lambda: next(calls)()               # type: ignore
        monkeypatch.setattr(session, 'last_error', 'context too long')
        answer = s.run()
        assert answer.startswith('(stopped: the provider failed')
        assert 'a.py' in answer and session.stalled is True

    def test_provider_error_before_work_has_no_answer(self, monkeypatch,
                                                      fake_repo) -> None:
        s = Scripted(monkeypatch, [], task_id='t1')
        s._step = lambda: None                        # type: ignore
        assert s.run() == ''

    def test_apply_work_paths_are_stripped(self) -> None:
        from guru.domain import gate
        diff = ('diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n'
                '+++ b/pkg/a.py\n@@ -1 +1 @@\n-x\n+y\n'
                'diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n'
                '@@ -1 +1 @@\n-x\n+y\n')
        content = 'Applied patch to your sandbox copy:\n' + gate.stat_text(
            diff)
        assert turn.changed_paths('apply_work', {}, content) == [
            'pkg/a.py', 'b.py']
