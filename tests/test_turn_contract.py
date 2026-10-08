"""The turn contract in the shared loop (guru.adapters.turn): forced tool
calls, ``final_answer``, the controller's ``plan`` handling and the
protocol-violation bookkeeping."""
import math

from guru import session, ui
from guru.adapters import turn
from guru.adapters.base import FORCE_ANY, FORCE_PLAN
from guru.domain import conversation, ledger, plan, tools


class Forcing:
    """An adapter that forces every round (Anthropic without thinking,
    LiteLLM)."""
    name = 'forcing'

    def forces(self, tool: str) -> bool:
        return True


class Lenient:
    """An adapter that cannot force (Ollama)."""
    name = 'lenient'

    def forces(self, tool: str) -> bool:
        return False


class Scripted:
    """Drive ``run_loop`` with scripted rounds; ``run_tools`` threads a
    tool message per call the way an adapter does — honouring the
    duplicate flag (the call does not run; ``turn.DUPLICATE_RESULT`` is
    the result, as in the three real adapters), executing ``plan``
    through the tool layer (a scripted handler) and echoing every other
    tool."""

    def __init__(self, monkeypatch, rounds, *, adapter=None,
                 controller=False, request='review the login code',
                 messages=None, handler=None, kind='',
                 task_id='') -> None:
        monkeypatch.setattr(ui, 'note_thinking', lambda: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        monkeypatch.setattr(ui.console, 'print', lambda *a, **k: None)
        self.rendered: list = []
        monkeypatch.setattr(turn, '_render_answer', self.rendered.append)
        monkeypatch.setattr(session, 'messages', messages if messages
                            is not None else [
                                {'role': 'system', 'content': 's'},
                                {'role': 'user', 'content': request}])
        monkeypatch.setattr(session, 'cancel_requested', False)
        monkeypatch.setattr(session, 'task_id', task_id)
        monkeypatch.setattr(session, 'can_spawn', controller)
        monkeypatch.setattr(session, 'controller', controller)
        monkeypatch.setattr(session, 'task_kind', kind)
        monkeypatch.setattr(session, 'adapter', adapter)
        monkeypatch.setattr(session, 'struggle',
                            {k: 0 for k in session.STRUGGLE_KEYS})
        self.steps = iter(rounds)
        self.user_msgs: list = []
        self.ran: list = []
        self.skipped: list = []
        self.handler_calls: list = []
        self.handler = handler
        tools.set_plan_handler(self._handle)

    def _handle(self, args: dict) -> str:
        self.handler_calls.append(args)
        return self.handler(args) if self.handler else plan.ANSWER_ACK

    def _step(self):
        return next(self.steps)

    def _execute(self, name: str, args: dict) -> str:
        self.ran.append(name)
        if name in ('plan', 'final_answer'):
            return tools.execute_tool(name, args)
        if name in ('write_file', 'edit_file'):       # a real success line
            return f"Wrote 1 bytes to {args.get('path')}. (sha:abc123)"
        return f'{name} ran'

    def _run_tools(self, pending) -> None:
        for name, args, _ref, dup in pending:
            if dup:
                self.skipped.append(name)
            content = turn.tool_result(name, args, dup,
                                       execute=self._execute)
            session.messages.append({'role': 'tool', 'tool_name': name,
                                     'tool_args': args, 'content': content})

    def script(self, rounds) -> None:
        """Replace the scripted rounds (callables that append the
        assistant message and return ``(text, calls)``)."""
        it = iter(rounds)
        self._step = lambda: next(it)()               # type: ignore

    def _add_user(self, text: str) -> None:
        self.user_msgs.append(text)
        session.messages.append({'role': 'user', 'content': text})

    def run(self, fake_repo=None):
        try:
            turn.run_loop(step=self._step, run_tools=self._run_tools,
                          add_user=self._add_user, nudge=False)
        finally:
            tools.set_plan_handler(None)
        if fake_repo is not None:
            ledger.flush()
            return fake_repo.stream('turns')
        return None


def _assistant(text: str, calls=None) -> dict:
    """Append what an adapter's ``step`` appends, then return the round."""
    msg: dict = {'role': 'assistant', 'content': text}
    if calls:
        msg['tool_calls'] = [{'id': ref, 'function': {'name': n,
                                                      'arguments': a}}
                             for n, a, ref in calls]
    session.messages.append(msg)
    return (text, list(calls or []))


class TestForcedTool:
    def test_controller_forces_plan_others_any(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'controller', True)
        assert turn.forced_tool() == FORCE_PLAN == 'plan'
        monkeypatch.setattr(session, 'controller', False)
        assert turn.forced_tool() == FORCE_ANY == 'any'

    def test_forcing_asks_the_adapter(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'adapter', Forcing())
        assert turn._forcing(FORCE_ANY) is True
        monkeypatch.setattr(session, 'adapter', Lenient())
        assert turn._forcing(FORCE_ANY) is False
        monkeypatch.setattr(session, 'adapter', None)
        assert turn._forcing(FORCE_ANY) is False


class TestWorkerText:
    """A text-only reply: the answer on a lenient adapter; one re-prompt
    then the answer with ``protocol_violation`` on a forcing one."""

    def test_lenient_text_is_the_answer(self, monkeypatch, fake_repo):
        s = Scripted(monkeypatch, [], adapter=Lenient())
        s.steps = iter([lambda: _assistant('The bug is in parse.')])
        s._step = lambda: next(s.steps)()          # type: ignore[assignment]
        [row] = s.run(fake_repo)
        assert s.user_msgs == [] and s.rendered == ['The bug is in parse.']
        assert row['struggle']['protocol_violation'] == 0
        assert session.messages[-1] == {'role': 'assistant',
                                        'content': 'The bug is in parse.'}

    def test_forcing_reprompts_once_then_accepts(self, monkeypatch,
                                                 fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing())
        replies = iter(['Let me look.', 'Here is the answer.'])
        s._step = lambda: _assistant(next(replies))   # type: ignore
        [row] = s.run(fake_repo)
        assert s.user_msgs == [plan.REPROMPT_TEXT]
        assert s.rendered == ['Here is the answer.']
        assert row['struggle']['protocol_violation'] == 1
        assert row['struggle']['stall_nudges'] == 0

    def test_reprompt_then_tool_call_is_clean(self, monkeypatch, fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing())
        rounds = iter([
            lambda: _assistant('Let me look.'),
            lambda: _assistant('', [('final_answer', {'text': 'Done.'},
                                     'r1')])])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert s.user_msgs == [plan.REPROMPT_TEXT]
        assert row['struggle']['protocol_violation'] == 0
        assert s.rendered == ['Done.']

    def test_empty_reply_reprompts_on_any_adapter(self, monkeypatch,
                                                  fake_repo):
        s = Scripted(monkeypatch, [], adapter=Lenient())
        replies = iter(['', ''])
        s._step = lambda: _assistant(next(replies))   # type: ignore
        [row] = s.run(fake_repo)
        assert s.user_msgs == [plan.REPROMPT_TEXT]
        assert s.rendered == ['']
        assert row['struggle']['protocol_violation'] == 1

    def test_request_skips_the_reprompt(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'messages', [
            {'role': 'user', 'content': 'real request'},
            {'role': 'assistant', 'content': 'Let me…'},
            {'role': 'user', 'content': plan.REPROMPT_TEXT},
            {'role': 'user', 'content': plan.reask_text(['x'])}])
        assert turn._turn_request() == 'real request'
        assert turn._turn_start() == 0


class TestFinalAnswer:
    def test_lone_final_answer_collapses_into_text(self, monkeypatch,
                                                   fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing())
        rounds = iter([
            lambda: _assistant('', [('read_file', {'path': 'a.py'}, 'r1')]),
            lambda: _assistant('', [('final_answer', {'text': 'It works.'},
                                     'r2')])])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert s.ran == ['read_file', 'final_answer']
        assert s.rendered == ['It works.']
        assert row['tools_used'] == ['read_file', 'final_answer']
        # The final_answer round is one assistant text message now; the
        # read_file round is kept.
        tail = session.messages[-3:]
        assert tail[0]['role'] == 'assistant' and tail[0]['tool_calls']
        assert tail[1]['role'] == 'tool' and tail[1]['tool_name'] == \
            'read_file'
        assert tail[2] == {'role': 'assistant', 'content': 'It works.'}

    def test_final_answer_with_other_calls_is_appended(self, monkeypatch,
                                                       fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing())
        rounds = iter([lambda: _assistant('', [
            ('read_file', {'path': 'a.py'}, 'r1'),
            ('final_answer', {'text': 'Both.'}, 'r2')])])
        s._step = lambda: next(rounds)()              # type: ignore
        s.run(fake_repo)
        assert s.ran == ['read_file', 'final_answer']
        assert session.messages[-1] == {'role': 'assistant',
                                        'content': 'Both.'}
        assert session.messages[-2]['tool_name'] == 'final_answer'

    def test_misnamed_field_still_answers(self, monkeypatch, fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing())
        rounds = iter([lambda: _assistant('', [
            ('final_answer', {'answer': 'Named wrong.'}, 'r1')])])
        s._step = lambda: next(rounds)()              # type: ignore
        s.run(fake_repo)
        assert s.rendered == ['Named wrong.']

    def test_final_answer_readers_see_the_text(self, monkeypatch,
                                               fake_repo):
        """orchestrator.final_answer / bench._final_answer read the last
        assistant text: after the collapse that is the answer."""
        from guru import bench
        from guru.agents import Agent
        s = Scripted(monkeypatch, [], adapter=Forcing())
        rounds = iter([lambda: _assistant('', [
            ('final_answer', {'text': 'Findings: none.'}, 'r1')])])
        s._step = lambda: next(rounds)()              # type: ignore
        s.run(fake_repo)
        agent = Agent(id='a', title='a')
        agent.state.messages = list(session.messages)
        assert bench._final_answer(agent) == 'Findings: none.'
        assert bench._tool_names(agent) == []


class TestControllerPlan:
    """A controller's ``plan`` tool call, as the handler answers it."""

    def _plan(self, args, ref='p1'):
        return lambda: _assistant('', [('plan', args, ref)])

    def test_answer_outcome_is_the_reply_and_runs_nothing(
            self, monkeypatch, fake_repo):
        args = {'outcome': 'answer', 'answer': 'I can review code.'}
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True,
                     request='hi, what can you do?')
        rounds = iter([self._plan(args)])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert s.handler_calls == [args]
        assert s.rendered == ['I can review code.']
        assert row['tools_used'] == ['plan']
        assert row['tasks_spawned'] == 0
        assert row['controller_executed'] is False
        assert row['struggle']['protocol_violation'] == 0
        assert session.messages[-1] == {'role': 'assistant',
                                        'content': 'I can review code.'}
        assert session.messages[-2]['role'] == 'user'    # collapsed round

    def test_delegate_ends_the_turn_waiting(self, monkeypatch, fake_repo):
        args = {'outcome': 'delegate', 'tasks': [
            {'goal': 'review auth', 'kind': 'review',
             'complexity': 'hard'}]}

        def handler(a):
            session.turn_waiting = True
            return plan.delegated_text(['agent1'], plan.parse(a)[0].tasks)
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True,
                     handler=handler)
        rounds = iter([self._plan(args)])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert s.rendered == []
        assert row['tasks_spawned'] == 1 and row['tools_used'] == ['plan']
        assert row['controller_executed'] is False
        assert session.messages[-1]['tool_name'] == 'plan'   # kept native

    def test_reask_once_then_accepted(self, monkeypatch, fake_repo):
        bad = {'outcome': 'delegate', 'tasks': []}
        good = {'outcome': 'answer', 'answer': 'Fine.'}
        verdicts = iter([plan.reask_text(['needs a task']), plan.ANSWER_ACK])
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True,
                     handler=lambda a: next(verdicts))
        rounds = iter([self._plan(bad, 'p1'), self._plan(good, 'p2')])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert len(s.handler_calls) == 2
        assert s.rendered == ['Fine.']
        assert row['struggle']['protocol_violation'] == 0
        assert s.user_msgs == []          # the re-ask was the tool result

    def test_second_rejection_falls_back_with_violation(self, monkeypatch,
                                                        fake_repo):
        bad = {'outcome': 'delegate', 'tasks': [{'goal': 'look around'}],
               'answer': ''}
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True,
                     handler=lambda a: plan.reask_text(['task 1: bad']))
        rounds = iter([self._plan(bad, 'p1'), self._plan(bad, 'p2'),
                       self._plan(bad, 'p3')])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert len(s.handler_calls) == 2               # never a third round
        assert row['struggle']['protocol_violation'] == 1
        [text] = s.rendered
        assert text.startswith("(guru could not run the controller's plan:"
                               " task 1: bad.)")
        assert 'look around' in text
        assert session.messages[-1] == {'role': 'assistant',
                                        'content': text}

    def test_second_rejection_keeps_the_model_text(self, monkeypatch,
                                                   fake_repo):
        bad = {'outcome': 'later'}
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True,
                     handler=lambda a: plan.reask_text(['no outcome']))
        rounds = iter([self._plan(bad, 'p1'),
                       lambda: _assistant('I would rather chat.',
                                          [('plan', bad, 'p2')])])
        s._step = lambda: next(rounds)()              # type: ignore
        s.run(fake_repo)
        assert s.rendered == ['I would rather chat.']

    def test_refusal_keeps_the_turn_going(self, monkeypatch, fake_repo):
        args = {'outcome': 'delegate', 'tasks': [{'goal': 'g'}]}
        answers = iter([plan.refused_text(['refused: no rung']),
                        plan.ANSWER_ACK])
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True,
                     handler=lambda a: next(answers))
        rounds = iter([self._plan(args, 'p1'),
                       self._plan({'outcome': 'answer', 'answer': 'Sorry.'},
                                  'p2')])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert s.rendered == ['Sorry.']
        assert row['struggle']['protocol_violation'] == 0
        assert row['tasks_spawned'] == 0

    def test_foreign_tool_flips_controller_executed(self, monkeypatch,
                                                    fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True)
        rounds = iter([
            lambda: _assistant('', [('read_file', {'path': 'x'}, 'r1')]),
            self._plan({'outcome': 'answer', 'answer': 'ok'})])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert row['controller_executed'] is True

    def test_long_answer_without_delegation_flips(self, monkeypatch,
                                                  fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True)
        rounds = iter([self._plan({'outcome': 'answer',
                                   'answer': 'x' * 601})])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert row['controller_executed'] is True

    def test_mailbox_synthesis_never_flips(self, monkeypatch, fake_repo):
        messages = [
            {'role': 'user', 'content': 'review auth for security'},
            {'role': 'assistant', 'content': ''},
            {'role': 'user', 'content': '[joined results]\n— agent1: A1'}]
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True,
                     messages=messages)
        rounds = iter([self._plan({'outcome': 'answer',
                                   'answer': 'x' * 601})])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert row['controller_executed'] is False
        assert row['request'] == 'review auth for security'


class TestDuplicates:
    """Duplicate suppression: same name and arguments as an earlier call
    in the turn is answered unrun; never ``plan``/``final_answer``."""

    def test_repeated_read_is_answered_unrun(self, monkeypatch, fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing())
        s.script([
            lambda: _assistant('', [('read_file', {'path': 'a.py'}, 'r1')]),
            lambda: _assistant('', [('read_file', {'path': 'a.py'}, 'r2')]),
            lambda: _assistant('', [('final_answer', {'text': 'ok'},
                                     'r3')])])
        s.run(fake_repo)
        assert s.ran == ['read_file', 'final_answer']
        assert s.skipped == ['read_file']
        dup = [m for m in session.messages if m.get('role') == 'tool'][1]
        assert dup['content'].startswith(turn.DUPLICATE_RESULT.format(
            name='read_file'))

    def test_identical_plan_after_coverage_reask_is_accepted(
            self, monkeypatch, fake_repo):
        """The documented path: the handler re-asks once for coverage,
        the model sends the very same plan again and the handler (not
        the duplicate filter) accepts it."""
        args = {'outcome': 'delegate', 'tasks': [
            {'goal': 'review app/ for injection', 'kind': 'review',
             'complexity': 'hard'}]}
        verdicts = iter([plan.reask_text(["no task goal covers 'tests'"])])

        def handler(a):
            nxt = next(verdicts, None)
            if nxt is not None:
                return nxt
            session.turn_waiting = True
            return plan.delegated_text(['agent1'], plan.parse(a)[0].tasks)
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True,
                     handler=handler)
        s.script([lambda: _assistant('', [('plan', args, 'p1')]),
                  lambda: _assistant('', [('plan', args, 'p2')])])
        [row] = s.run(fake_repo)
        assert len(s.handler_calls) == 2 and s.skipped == []
        assert row['tasks_spawned'] == 1
        assert row['struggle']['protocol_violation'] == 0

    def test_identical_final_answer_is_never_a_duplicate(self, monkeypatch,
                                                         fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing())
        s.script([lambda: _assistant('', [('final_answer', {'text': 'A'},
                                           'r1')])])
        s.run(fake_repo)
        assert s.skipped == [] and s.rendered == ['A']


class TestRoundCap:
    """The per-turn round cap: the only exit while the model keeps
    calling tools or a controller keeps planning without an accepted
    plan."""

    def test_controller_refused_forever_ends_at_the_cap(self, monkeypatch,
                                                        fake_repo):
        args = {'outcome': 'delegate', 'tasks': [
            {'goal': 'look at auth', 'kind': 'review',
             'complexity': 'hard'}]}
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True,
                     handler=lambda a: plan.refused_text(['no rung']))
        s.script([lambda: _assistant('', [('plan', args, f'p{i}')])
                  for i in range(100)])
        [row] = s.run(fake_repo)
        assert len(s.handler_calls) == turn._MAX_ROUNDS == 12
        assert row['tools_used'].count('plan') == 12
        assert row['struggle']['protocol_violation'] == 1
        [text] = s.rendered
        assert text.startswith("(guru could not run the controller's plan:"
                               " 12 plan rounds without an accepted plan.)")
        assert 'look at auth' in text
        assert session.messages[-1] == {'role': 'assistant',
                                        'content': text}

    def test_controller_cap_prefers_its_own_text(self, monkeypatch,
                                                 fake_repo):
        args = {'outcome': 'delegate', 'tasks': [{'goal': 'g'}]}
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True,
                     handler=lambda a: plan.refused_text(['no rung']))
        s.script([lambda: _assistant('Trying again.', [('plan', args, 'p')])
                  for _ in range(100)])
        s.run(fake_repo)
        assert s.rendered == ['Trying again.']

    def test_worker_tool_rounds_end_at_the_wider_cap(self, monkeypatch,
                                                     fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing())
        s.script([lambda i=i: _assistant('', [('read_file',
                                               {'path': f'{i}.py'}, 'r')])
                  for i in range(100)])
        [row] = s.run(fake_repo)
        assert len(s.ran) == turn._MAX_TOOL_ROUNDS == 40
        assert row['struggle']['protocol_violation'] == 1
        # The main agent keeps the plain cap line (no worker handoff).
        assert s.rendered == [turn._CAPPED_TEXT.format(n=40)]
        assert session.capped is False

    def test_worker_cap_keeps_the_last_text(self, monkeypatch, fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing())
        texts = iter(['first', 'Looking at 3.py'] + [''] * 100)
        s.script([lambda i=i: _assistant(next(texts),
                                         [('read_file', {'path': f'{i}.py'},
                                           'r')])
                  for i in range(100)])
        s.run(fake_repo)
        assert s.rendered == ['Looking at 3.py']

    def test_under_the_cap_nothing_changes(self, monkeypatch, fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing())
        s.script([lambda i=i: _assistant('', [('read_file',
                                               {'path': f'{i}.py'}, 'r')])
                  for i in range(39)]
                 + [lambda: _assistant('', [('final_answer',
                                             {'text': 'done'}, 'f')])])
        [row] = s.run(fake_repo)
        assert s.rendered == ['done']
        assert row['struggle']['protocol_violation'] == 0
        assert session.capped is False


class TestRoundBudget:
    """The budget a worker (a sub-agent executing a task) sees: a footer on
    each round's first tool result, one checkpoint for a writing kind that
    changed nothing by half the cap, one last call HANDOFF_ROUNDS before
    the cap, and a code-built handoff when it still hits the cap."""

    def _worker(self, monkeypatch, kind=''):
        return Scripted(monkeypatch, [], adapter=Forcing(), kind=kind,
                        task_id='t1')

    def _reads(self, s, n: int, then=None) -> None:
        rounds = [lambda i=i: _assistant(
            '', [('read_file', {'path': f'{i}.py'}, 'r')])
            for i in range(n)]
        s.script(rounds + (then or []))

    def _results(self) -> list:
        return [m['content'] for m in session.messages
                if m.get('role') == 'tool']

    @staticmethod
    def _final():
        return lambda: _assistant('', [('final_answer', {'text': 'ok'}, 'f')])

    def test_every_round_carries_the_footer(self, monkeypatch) -> None:
        s = self._worker(monkeypatch)
        self._reads(s, 2, [self._final()])
        s.run()
        # The lone final_answer round collapses into the answer text.
        first, second = self._results()
        assert first.endswith('[guru] round 1/40 · files changed: 0')
        assert second.endswith('[guru] round 2/40 · files changed: 0')

    def test_the_main_agent_sees_no_footer(self, monkeypatch,
                                           fake_repo) -> None:
        s = Scripted(monkeypatch, [], adapter=Forcing())
        self._reads(s, 2, [self._final()])
        s.run(fake_repo)
        assert all('[guru] round' not in r for r in self._results())

    def test_footer_on_the_first_result_counts_finished_writes(
            self, monkeypatch) -> None:
        s = self._worker(monkeypatch)
        s.script([lambda: _assistant('', [
            ('read_file', {'path': 'a.py'}, 'r1'),
            ('write_file', {'path': 'b.py', 'content': 'x'}, 'w1')]),
            lambda: _assistant('', [('read_file', {'path': 'c.py'}, 'r2')]),
            lambda: _assistant('', [('run_tests', {}, 't')]),
            self._final()])
        s.run()
        read, write, read2, _tests = self._results()
        assert read.endswith('[guru] round 1/40 · files changed: 0')
        assert '[guru]' not in write
        assert read2.endswith('[guru] round 2/40 · files changed: 1')

    def test_a_failed_write_is_not_a_change(self, monkeypatch) -> None:
        s = self._worker(monkeypatch, kind='build')
        s._execute = lambda n, a: (                    # type: ignore
            "'old' text not found in a.py; nothing changed."
            if n == 'edit_file' else f'{n} ran')
        s.script([lambda: _assistant('', [('edit_file', {
            'path': 'a.py', 'old': 'x', 'new': 'y', 'sha': 's'}, 'e')])]
            + [lambda i=i: _assistant('', [('read_file',
                                            {'path': f'{i}.py'}, 'r')])
               for i in range(25)] + [self._final()])
        s.run()
        assert sum(turn.CHECKPOINT_TEXT in r for r in self._results()) == 1

    def test_checkpoint_for_a_writing_kind(self, monkeypatch) -> None:
        s = self._worker(monkeypatch, kind='build')
        self._reads(s, 25, [self._final()])
        s.run()
        results = self._results()
        at = math.ceil(turn._MAX_TOOL_ROUNDS * turn.CHECKPOINT_AT)  # 12
        assert all(turn.CHECKPOINT_TEXT not in r for r in results[:at - 1])
        assert turn.CHECKPOINT_TEXT in results[at - 1]
        assert sum(turn.CHECKPOINT_TEXT in r for r in results) == 1
        assert session.struggle['budget_nudges'] == 2   # + reads closed

    def test_no_checkpoint_once_a_file_changed(self, monkeypatch) -> None:
        s = self._worker(monkeypatch, kind='build')
        s.script([lambda: _assistant('', [('write_file', {
            'path': 'a.py', 'content': 'x'}, 'w')])]
            + [lambda i=i: _assistant('', [('read_file',
                                            {'path': f'{i}.py'}, 'r')])
               for i in range(25)]
            + [lambda: _assistant('', [('run_tests', {}, 't')]),
               self._final()])
        s.run()
        assert all(turn.CHECKPOINT_TEXT not in r for r in self._results())

    def test_no_checkpoint_while_editing_a_sandbox_copy(
            self, monkeypatch) -> None:
        s = self._worker(monkeypatch, kind='build')
        s.script([lambda: _assistant('', [('sandbox_run', {
            'command': 'sed -i s/a/b/ x.py'}, 's')])]
            + [lambda i=i: _assistant('', [('read_file',
                                            {'path': f'{i}.py'}, 'r')])
               for i in range(25)] + [self._final()])
        s.run()
        assert all(turn.CHECKPOINT_TEXT not in r for r in self._results())

    def test_no_checkpoint_for_a_reading_kind(self, monkeypatch) -> None:
        s = self._worker(monkeypatch, kind='explain')
        self._reads(s, 25, [self._final()])
        s.run()
        assert all(turn.CHECKPOINT_TEXT not in r for r in self._results())

    def test_last_call_then_handoff_at_the_cap(self, monkeypatch) -> None:
        s = self._worker(monkeypatch, kind='explain')
        texts = iter(['first', 'Looking at 3.py'] + [''] * 100)
        s.script([lambda i=i: _assistant(next(texts), [
            ('read_file', {'path': f'{i}.py'}, 'r')]) for i in range(100)])
        s.run()
        results = self._results()
        assert turn.LAST_CALL_TEXT.format(left=2) in results[37]   # 38/40
        assert sum('round(s) left' in r for r in results) == 1
        assert session.struggle['budget_nudges'] == 1
        assert session.capped is True
        [text] = s.rendered
        assert text.startswith('(capped: guru ended this task after 40'
                               ' rounds')
        assert 'Files changed: none' in text
        assert 'Files read: 0.py, 1.py' in text and '(+28 more)' in text
        assert text.endswith('Last note: Looking at 3.py')

    def test_spent_budget_refuses_all_but_final_answer(
            self, monkeypatch) -> None:
        s = self._worker(monkeypatch)
        self._reads(s, 39, [lambda: _assistant('', [
            ('final_answer', {'text': 'handoff after refusal'}, 'f')])])
        s.run()
        results = self._results()
        assert turn.BUDGET_REFUSAL not in results[37]       # round 38 ran
        assert results[38].startswith(turn.BUDGET_REFUSAL)  # round 39 not
        assert s.ran.count('read_file') == 38
        assert s.rendered == ['handoff after refusal']
        assert session.capped is False

    def test_main_agent_is_never_refused(self, monkeypatch,
                                         fake_repo) -> None:
        s = Scripted(monkeypatch, [], adapter=Forcing())
        self._reads(s, 40)
        s.run(fake_repo)
        assert all(turn.BUDGET_REFUSAL not in r for r in self._results())
        assert s.ran.count('read_file') == 40

    def test_reads_close_for_an_idle_writer(self, monkeypatch) -> None:
        s = self._worker(monkeypatch, kind='build')
        self._reads(s, 33, [self._final()])
        s.run()
        results = self._results()
        stop = int(turn._MAX_TOOL_ROUNDS * turn.READ_STOP_AT)   # round 20
        assert not any(r.startswith(turn.READ_REFUSAL)
                       for r in results[:stop - 1])
        assert results[stop - 1].startswith(turn.READ_REFUSAL)
        assert f'Reading is closed: {stop}/40' in results[stop - 1]
        assert sum('Reading is closed' in r for r in results) == 1
        assert s.ran.count('read_file') == stop - 1
        assert session.struggle['budget_nudges'] == 2   # checkpoint + stop

    def test_a_write_reopens_reading(self, monkeypatch) -> None:
        s = self._worker(monkeypatch, kind='build')
        s.script([lambda i=i: _assistant('', [('read_file',
                                               {'path': f'{i}.py'}, 'r')])
                  for i in range(31)]
                 + [lambda: _assistant('', [('write_file', {
                     'path': 'a.py', 'content': 'x'}, 'w')])]
                 + [lambda: _assistant('', [('read_file',
                                             {'path': 'z.py'}, 'r')]),
                    lambda: _assistant('', [('run_tests', {}, 't')]),
                    self._final()])
        s.run()
        assert not self._results()[-2].startswith(turn.READ_REFUSAL)

    def test_reads_stay_open_for_a_reading_kind(self, monkeypatch) -> None:
        s = self._worker(monkeypatch, kind='explain')
        self._reads(s, 35, [self._final()])
        s.run()
        assert all(not r.startswith(turn.READ_REFUSAL)
                   for r in self._results())

    def test_last_call_heeded_ends_normally(self, monkeypatch) -> None:
        s = self._worker(monkeypatch)
        self._reads(s, 38, [lambda: _assistant('', [
            ('final_answer', {'text': 'handoff: read 38 files'}, 'f')])])
        s.run()
        assert s.rendered == ['handoff: read 38 files']
        assert session.capped is False

    def test_controller_rounds_carry_no_footer(self, monkeypatch,
                                               fake_repo) -> None:
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True)
        s.script([lambda: _assistant('', [('plan', {
            'outcome': 'answer', 'answer': 'hi'}, 'p')])])
        s.run(fake_repo)
        assert all('[guru] round' not in r for r in self._results())


class TestChangedPaths:
    """Only a write that succeeded counts as a change (from its result)."""

    def test_writes(self) -> None:
        ok = 'Wrote 6 bytes to /p/a.py. (sha:9e26bf369911)'
        assert turn.changed_paths('write_file', {'path': 'a.py'}, ok) == [
            'a.py']
        assert turn.changed_paths('edit_file', {'path': 'a.py'},
                                  "'old' text not found; nothing changed.") \
            == []
        assert turn.changed_paths('write_file', {'path': 'a.py'},
                                  'Refused: read-only mode') == []

    def test_patch_and_submit_report_their_targets(self) -> None:
        applied = ('Applied patch:\n/p/a.py: 1 hunk(s) applied (sha:1)\n'
                   '/p/b.py: 2 hunk(s) applied (sha:2)')
        for name in ('apply_patch', 'sandbox_submit'):
            assert turn.changed_paths(name, {}, applied) == [
                '/p/a.py', '/p/b.py']
        assert turn.changed_paths(
            'apply_patch', {}, 'Patch rejected: no headers') == []

    def test_delete(self) -> None:
        assert turn.changed_paths('delete_file', {'path': 'a.py'},
                                  'Deleted /p/a.py.') == ['a.py']
        assert turn.changed_paths('delete_file', {'path': 'a.py'},
                                  'No such file: /p/a.py') == []

    def test_other_tools(self) -> None:
        assert turn.changed_paths('read_file', {'path': 'a.py'},
                                  '(sha:1)') == []


class TestTwoPlansInOneRound:
    def test_second_plan_is_refused_unrun_first_verdict_kept(
            self, monkeypatch, fake_repo):
        first = {'outcome': 'answer', 'answer': 'From the first.'}
        second = {'outcome': 'answer', 'answer': 'From the second.'}
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True)
        s.script([lambda: _assistant('', [('plan', first, 'p1'),
                                          ('plan', second, 'p2')])])
        [row] = s.run(fake_repo)
        assert s.handler_calls == [first]
        assert s.skipped == ['plan']
        assert s.rendered == ['From the first.']
        assert row['tools_used'] == ['plan', 'plan']

    def test_second_plan_refusal_does_not_hide_a_delegation(
            self, monkeypatch, fake_repo):
        first = {'outcome': 'delegate', 'tasks': [
            {'goal': 'review auth', 'kind': 'review',
             'complexity': 'hard'}]}

        def handler(a):
            session.turn_waiting = True
            return plan.delegated_text(['agent1'], plan.parse(a)[0].tasks)
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True,
                     handler=handler)
        s.script([lambda: _assistant('', [
            ('plan', first, 'p1'),
            ('plan', {'outcome': 'answer', 'answer': 'x'}, 'p2')])])
        [row] = s.run(fake_repo)
        assert len(s.handler_calls) == 1 and row['tasks_spawned'] == 1
        assert s.rendered == []


class TestAnswerWithoutText:
    """``answer`` is never rejected: an empty ``answer`` field takes the
    round's own text; only when both are empty does the loop re-prompt
    (once), then fall back with ``protocol_violation``."""

    def test_empty_answer_takes_the_round_text(self, monkeypatch, fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True)
        s.script([lambda: _assistant('Here is my reply.',
                                     [('plan', {'outcome': 'answer'}, 'p')])])
        [row] = s.run(fake_repo)
        assert s.handler_calls == [{'outcome': 'answer'}]
        assert s.rendered == ['Here is my reply.']
        assert row['struggle']['protocol_violation'] == 0
        assert session.messages[-1] == {'role': 'assistant',
                                        'content': 'Here is my reply.'}

    def test_both_empty_reprompts_once_then_falls_back(self, monkeypatch,
                                                       fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True)
        s.script([lambda: _assistant('', [('plan', {'outcome': 'answer'},
                                           'p1')]),
                  lambda: _assistant('', [('plan', {'outcome': 'answer',
                                                    'answer': ' '}, 'p2')]),
                  lambda: _assistant('', [('plan', {'outcome': 'answer'},
                                           'p3')])])
        [row] = s.run(fake_repo)
        assert len(s.handler_calls) == 2               # never a third round
        assert s.user_msgs == [plan.PLAN_REPROMPT_TEXT]
        assert row['struggle']['protocol_violation'] == 1
        assert s.rendered == ["(guru could not run the controller's plan:"
                              " outcome answer without text.)"]

    def test_reprompt_then_text_answers(self, monkeypatch, fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True)
        s.script([lambda: _assistant('', [('plan', {'outcome': 'answer'},
                                           'p1')]),
                  lambda: _assistant('', [('plan', {'outcome': 'answer',
                                                    'answer': 'Now.'},
                                           'p2')])])
        [row] = s.run(fake_repo)
        assert s.rendered == ['Now.']
        assert row['struggle']['protocol_violation'] == 0

    def test_text_path_uses_the_prose_around_the_json(self, monkeypatch,
                                                      fake_repo):
        text = 'Glad to help.\n{"outcome": "answer", "answer": ""}'
        s = Scripted(monkeypatch, [], adapter=Lenient(), controller=True)
        s.script([lambda: _assistant(text)])
        [row] = s.run(fake_repo)
        assert s.rendered == ['Glad to help.']
        assert row['struggle']['protocol_violation'] == 0
        assert session.messages[-1]['content'] == 'Glad to help.'

    def test_text_path_bare_empty_plan_reprompts(self, monkeypatch,
                                                 fake_repo):
        s = Scripted(monkeypatch, [], adapter=Lenient(), controller=True)
        s.script([lambda: _assistant('{"outcome": "answer"}'),
                  lambda: _assistant('Right: hello.')])
        [row] = s.run(fake_repo)
        assert s.user_msgs == [plan.PLAN_REPROMPT_TEXT]
        assert s.rendered == ['Right: hello.']
        assert row['struggle']['protocol_violation'] == 0


class TestControllerTextPath:
    """A controller on an adapter that cannot force: its text is parsed
    for the plan object; plain text is the answer."""

    def test_json_plan_in_text_answers(self, monkeypatch, fake_repo):
        text = 'Here you go:\n{"outcome": "answer", "answer": "Hello!"}'
        s = Scripted(monkeypatch, [], adapter=Lenient(), controller=True)
        rounds = iter([lambda: _assistant(text)])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert s.handler_calls == [{'outcome': 'answer', 'answer': 'Hello!'}]
        assert s.rendered == ['Hello!']
        assert row['tools_used'] == ['plan']
        # The JSON reply became the answer text in the history.
        assert session.messages[-1] == {'role': 'assistant',
                                        'content': 'Hello!'}

    def test_json_plan_in_text_delegates(self, monkeypatch, fake_repo):
        args = {'outcome': 'delegate', 'tasks': [
            {'goal': 'review auth', 'kind': 'review',
             'complexity': 'standard'}]}

        def handler(a):
            session.turn_waiting = True
            return plan.delegated_text(['agent1'], plan.parse(a)[0].tasks)
        s = Scripted(monkeypatch, [], adapter=Lenient(), controller=True,
                     handler=handler)
        import json
        rounds = iter([lambda: _assistant(json.dumps(args))])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert row['tasks_spawned'] == 1 and s.rendered == []

    def test_reask_goes_back_as_a_user_message(self, monkeypatch,
                                               fake_repo):
        import json
        bad = {'outcome': 'delegate', 'tasks': []}
        verdicts = iter([plan.reask_text(['needs a task']), plan.ANSWER_ACK])
        s = Scripted(monkeypatch, [], adapter=Lenient(), controller=True,
                     handler=lambda a: next(verdicts))
        rounds = iter([
            lambda: _assistant(json.dumps(bad)),
            lambda: _assistant(json.dumps({'outcome': 'answer',
                                           'answer': 'ok'}))])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert s.user_msgs == [plan.reask_text(['needs a task'])]
        assert s.rendered == ['ok']
        assert row['struggle']['protocol_violation'] == 0

    def test_second_text_rejection_falls_back(self, monkeypatch, fake_repo):
        import json
        bad = {'outcome': 'delegate', 'tasks': []}
        s = Scripted(monkeypatch, [], adapter=Lenient(), controller=True,
                     handler=lambda a: plan.reask_text(['needs a task']))
        rounds = iter([lambda: _assistant(json.dumps(bad))] * 3)
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert len(s.handler_calls) == 2
        assert row['struggle']['protocol_violation'] == 1
        assert s.rendered[0].startswith("(guru could not run")

    def test_plain_text_is_the_answer(self, monkeypatch, fake_repo):
        s = Scripted(monkeypatch, [], adapter=Lenient(), controller=True)
        rounds = iter([lambda: _assistant('Hello, I coordinate work.')])
        s._step = lambda: next(rounds)()              # type: ignore
        [row] = s.run(fake_repo)
        assert s.handler_calls == []
        assert s.rendered == ['Hello, I coordinate work.']
        assert row['struggle']['protocol_violation'] == 0

    def test_forcing_controller_text_reprompts_with_plan_text(
            self, monkeypatch, fake_repo):
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True)
        replies = iter(['I will delegate.', 'Fine, here.'])
        s._step = lambda: _assistant(next(replies))   # type: ignore
        [row] = s.run(fake_repo)
        assert s.user_msgs == [plan.PLAN_REPROMPT_TEXT]
        assert s.rendered == ['Fine, here.']
        assert row['struggle']['protocol_violation'] == 1


class TestSettle:
    def test_text_path_replaces_provider_message_content(self) -> None:
        import ollama
        msg = ollama.Message(role='assistant', content='{"outcome": "x"}')
        session.messages = [{'role': 'user', 'content': 'q'}, msg]
        turn._settle('Hello', collapse=False)
        assert session.messages[-1] is msg and msg.content == 'Hello'

    def test_collapse_needs_the_round_shape(self) -> None:
        session.messages = [{'role': 'user', 'content': 'q'},
                            {'role': 'assistant', 'content': 'said'}]
        turn._settle('Hello', collapse=True)       # no tool message: no pop
        assert session.messages == [{'role': 'user', 'content': 'q'},
                                    {'role': 'assistant', 'content': 'Hello'}]
        session.messages = [{'role': 'user', 'content': 'q'},
                            {'role': 'tool', 'tool_name': 'x',
                             'content': 'r'}]
        turn._settle('Hello', collapse=False)
        assert session.messages[-1] == {'role': 'assistant',
                                        'content': 'Hello'}


def test_controller_hint_numbers_match_the_code() -> None:
    """The hint states the worker budget and the deliverable limit; keep
    them in step with the constants that enforce them."""
    from guru import config
    hint = config.CONTROLLER_HINT
    assert f'budget of {turn._MAX_TOOL_ROUNDS} tool rounds' in hint
    assert f'at most {plan.MAX_DELIVERABLES} files' in hint


class TestVerifyBeforeFinal:
    """A worker that changed files and ran no tests since has its first
    final_answer sent back once; a tested change goes straight through."""

    def _worker(self, monkeypatch):
        return Scripted(monkeypatch, [], adapter=Forcing(), kind='build',
                        task_id='t1')

    @staticmethod
    def _write():
        return lambda: _assistant('', [('write_file', {
            'path': 'a.py', 'content': 'x'}, 'w')])

    @staticmethod
    def _final(text='done'):
        return lambda: _assistant('', [('final_answer', {'text': text},
                                        'f')])

    def test_untested_change_is_sent_back_once(self, monkeypatch) -> None:
        s = self._worker(monkeypatch)
        s.script([self._write(), self._final('first'),
                  lambda: _assistant('', [('run_tests', {}, 't')]),
                  self._final('tested')])
        s.run()
        assert s.rendered == ['tested']
        sent_back = [m['content'] for m in session.messages
                     if m.get('tool_name') == 'final_answer']
        assert sent_back[0].startswith(turn.VERIFY_REFUSAL)

    def test_only_once(self, monkeypatch) -> None:
        s = self._worker(monkeypatch)
        s.script([self._write(), self._final('first'),
                  self._final('second')])
        s.run()
        assert s.rendered == ['second']

    def test_tested_change_goes_through(self, monkeypatch) -> None:
        s = self._worker(monkeypatch)
        s.script([self._write(),
                  lambda: _assistant('', [('run_tests', {}, 't')]),
                  self._final('ok')])
        s.run()
        assert s.rendered == ['ok']

    def test_tests_in_the_same_round_count(self, monkeypatch) -> None:
        s = self._worker(monkeypatch)
        s.script([self._write(), lambda: _assistant('', [
            ('run_tests', {}, 't'), ('final_answer', {'text': 'ok'}, 'f')])])
        s.run()
        assert s.rendered == ['ok']

    def test_no_change_no_check(self, monkeypatch) -> None:
        s = self._worker(monkeypatch)
        s.script([self._final('nothing to do')])
        s.run()
        assert s.rendered == ['nothing to do']

    def test_main_agent_is_not_sent_back(self, monkeypatch,
                                         fake_repo) -> None:
        s = Scripted(monkeypatch, [], adapter=Forcing())
        s.script([self._write(), self._final('ok')])
        s.run(fake_repo)
        assert s.rendered == ['ok']


class TestAnswerCheckInTheLoop:
    """A controller answer sent back by the answer check: one more round,
    and the human request stays the request (tool and text paths)."""

    REQUEST = 'add a usage store and wire it in'

    def _handler(self):
        from guru.domain import claims
        verdicts = iter([claims.problems_text(['cli.py not changed']),
                         plan.ANSWER_ACK])
        return lambda args: next(verdicts)

    def test_tool_path(self, monkeypatch, fake_repo) -> None:
        from guru.domain import claims
        s = Scripted(monkeypatch, [], adapter=Forcing(), controller=True,
                     request=self.REQUEST, handler=self._handler())
        s.script([lambda: _assistant('', [('plan', {
            'outcome': 'answer', 'answer': 'Wired in.'}, 'p1')]),
            lambda: _assistant('', [('plan', {
                'outcome': 'answer', 'answer': 'Not wired: cli.py.'},
                'p2')])])
        s.run(fake_repo)
        assert s.rendered == ['Not wired: cli.py.']
        assert conversation.request_in(session.messages) == self.REQUEST
        assert claims.already_checked(session.messages)

    def test_text_path(self, monkeypatch, fake_repo) -> None:
        from guru.domain import claims
        s = Scripted(monkeypatch, [], adapter=Lenient(), controller=True,
                     request=self.REQUEST, handler=self._handler())
        s.script([lambda: _assistant(
            '{"outcome": "answer", "answer": "Wired in."}'),
            lambda: _assistant(
                '{"outcome": "answer", "answer": "Not wired: cli.py."}')])
        s.run(fake_repo)
        assert s.rendered == ['Not wired: cli.py.']
        # The problems came back as a user message: a loop nudge, never
        # the request, and it marks the request as checked.
        sent = [m for m in session.messages if m.get('role') == 'user'
                and m['content'].startswith(claims.PREFIX)]
        assert len(sent) == 1
        assert conversation.is_nudge(sent[0]['content'])
        assert conversation.request_in(session.messages) == self.REQUEST
        assert conversation.request_start(session.messages) == 1
        assert claims.already_checked(session.messages)
