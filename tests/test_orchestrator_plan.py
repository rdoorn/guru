"""The controller's ``plan`` handler (Orchestrator.do_plan): answer runs
no worker, delegate spawns exactly the plan through spawn_panel, review
tasks take the review ladder, re-asks, refusals, and the mailbox
synthesis round trip through the real turn loop."""
import asyncio

import pytest

from guru import bench, config, session
from guru.adapters import turn
from guru.adapters.base import Adapter
from guru.domain import conversation, ledger, plan, tools
from tests.test_orchestrator import (_FAKE_ENV, _FakeLoop, _tiers_orch,
                                     labels_isolated)

__all__ = ['labels_isolated']

_REQUEST = 'Review this repository for correctness and security issues.'
_TWO = {'outcome': 'delegate', 'tasks': [
    {'goal': 'review app/ for bugs and logic errors', 'kind': 'review',
     'complexity': 'hard', 'files': ['app/']},
    {'goal': 'review app/ for injection, authz and secrets',
     'kind': 'review', 'complexity': 'standard',
     'role': 'security-engineer', 'skill': 'code-review'}]}


def _orch(monkeypatch, request=_REQUEST, messages=None):
    """A bare orchestrator (routing inert) whose main agent is a
    controller mid-turn on ``request``; children are made but never
    launched."""
    from guru.orchestrator import Orchestrator
    monkeypatch.setattr(ledger, 'environment', lambda: dict(_FAKE_ENV))
    o = Orchestrator()
    o.loop = _FakeLoop()
    o.launch = lambda a: None                        # type: ignore
    main = o.manager.active
    main.busy = True
    main.state.controller = True
    main.state.can_spawn = True
    main.state.model = 'm'
    main.state.turn_id = 'T1'
    main.state.messages = messages if messages is not None else [
        {'role': 'system', 'content': 's'},
        {'role': 'user', 'content': request}]
    return o, main


class TestDoPlan:
    def test_answer_runs_no_worker(self, monkeypatch, fake_repo) -> None:
        o, main = _orch(monkeypatch)
        out = o.do_plan(main.state, {'outcome': 'answer',
                                     'answer': 'Nothing to do.'})
        assert out == plan.ANSWER_ACK
        assert len(o.manager.agents) == 1 and o.barriers == {}
        assert main.state.turn_waiting is False
        ledger.flush()
        assert fake_repo.stream('tasks') == []

    def test_answer_ignores_concern_coverage(self, monkeypatch, fake_repo):
        o, main = _orch(monkeypatch)
        assert o.do_plan(main.state, {'outcome': 'answer',
                                      'answer': 'Which file?'}) == \
            plan.ANSWER_ACK

    def test_delegate_spawns_exactly_the_plan(self, monkeypatch,
                                              fake_repo) -> None:
        o, main = _orch(monkeypatch)
        out = o.do_plan(main.state, _TWO)
        assert out.startswith(plan.DELEGATED_PREFIX)
        assert 'agent1 (review/hard), agent2 (review/standard)' in out
        children = o.manager.agents[1:]
        assert [c.title for c in children] == ['agent1', 'agent2']
        assert children[0].task == 'review app/ for bugs and logic errors' \
            '\nFiles: app/'
        assert children[1].task == \
            'review app/ for injection, authz and secrets'
        assert children[1].state.active_role == 'security-engineer'
        assert children[1].state.active_skill == 'code-review'
        assert children[0].state.active_role is None
        # join semantics: barrier over both, the turn ends here
        assert o.barriers[main]['remaining'] == {'agent1', 'agent2'}
        assert o.barriers[main]['synthesis'] == ''
        assert main.state.turn_waiting is True
        ledger.flush()
        rows = fake_repo.stream('tasks')
        assert [(r['kind'], r['complexity']) for r in rows] == [
            ('review', 'hard'), ('review', 'standard')]
        assert {r['status'] for r in rows} == {'running'}
        assert {r['parent'] for r in rows} == {'main'}
        assert {r['turn_id'] for r in rows} == {'T1'}

    def test_workers_are_not_controllers(self, monkeypatch, fake_repo):
        o, main = _orch(monkeypatch)
        o.do_plan(main.state, _TWO)
        for c in o.manager.agents[1:]:
            assert c.state.controller is False and c.state.can_spawn is False
            assert tools.final_answer in c.state.active_tools
            assert tools.plan not in c.state.active_tools
            assert config.CONTROLLER_HINT not in c.state.messages[0]['content']

    def test_malformed_plan_is_re_asked_every_time(self, monkeypatch,
                                                   fake_repo) -> None:
        o, main = _orch(monkeypatch)
        for _ in range(3):
            out = o.do_plan(main.state, {'outcome': 'delegate', 'tasks': []})
            assert plan.is_reask(out)
            assert 'at least one task' in out
            main.state.messages.append(
                {'role': 'tool', 'tool_name': 'plan', 'content': out})
        assert len(o.manager.agents) == 1
        assert main.state.turn_waiting is False

    def test_unknown_labels_are_re_asked(self, monkeypatch, fake_repo):
        o, main = _orch(monkeypatch)
        out = o.do_plan(main.state, {'outcome': 'delegate', 'tasks': [
            {'goal': 'g', 'kind': 'bugfix', 'complexity': 'standard'}]})
        assert plan.is_reask(out) and "unknown kind 'bugfix'" in out

    def test_missing_concern_re_asked_once_then_runs(self, monkeypatch,
                                                     fake_repo) -> None:
        one = {'outcome': 'delegate', 'tasks': [
            {'goal': 'review app/ for injection and authz',
             'kind': 'review', 'complexity': 'hard'}]}
        o, main = _orch(monkeypatch)
        out = o.do_plan(main.state, one)
        assert plan.is_reask(out) and "'correctness'" in out
        assert len(o.manager.agents) == 1
        # The re-ask is the plan tool's result in the conversation; the
        # second plan with the same gap runs.
        main.state.messages.append(
            {'role': 'tool', 'tool_name': 'plan', 'content': out})
        out = o.do_plan(main.state, one)
        assert out.startswith(plan.DELEGATED_PREFIX)
        assert len(o.manager.agents) == 2

    def test_coverage_reads_the_full_request(self, monkeypatch, fake_repo):
        """A concern named after the ledger's 1000-char request cap still
        counts: the handler evaluates the whole request."""
        request = ('Review this repository for correctness' + ' details' * 200
                   + ' and security.')
        assert len(request) > conversation.REQUEST_CHARS
        assert 'security' not in conversation.request_in(
            [{'role': 'user', 'content': request}])
        o, main = _orch(monkeypatch, request=request)
        one = {'outcome': 'delegate', 'tasks': [_TWO['tasks'][0]]}
        out = o.do_plan(main.state, one)
        assert plan.is_reask(out) and "'security'" in out

    def test_mailbox_turn_skips_coverage(self, monkeypatch, fake_repo):
        one = {'outcome': 'delegate', 'tasks': [
            {'goal': 're-check the injection finding in app/upload.py',
             'kind': 'review', 'complexity': 'standard'}]}
        o, main = _orch(monkeypatch, messages=[
            {'role': 'user', 'content': _REQUEST},
            {'role': 'assistant', 'content': ''},
            {'role': 'user', 'content': '[joined results]\n— agent1: A1'}])
        out = o.do_plan(main.state, one)
        assert out.startswith(plan.DELEGATED_PREFIX)

    def _delegated_history(self, rounds: int) -> list:
        msgs: list = [{'role': 'system', 'content': 's'},
                      {'role': 'user', 'content': _REQUEST}]
        for i in range(rounds):
            msgs += [{'role': 'assistant', 'content': ''},
                     {'role': 'tool', 'tool_name': 'plan',
                      'content': plan.delegated_text(
                          [f'agent{i}'], plan.parse(_TWO)[0].tasks[:1])},
                     {'role': 'user',
                      'content': f'[joined results]\n— agent{i}: partial'}]
        return msgs

    def test_delegate_refused_at_the_cap_answer_still_fine(
            self, monkeypatch, fake_repo) -> None:
        msgs = self._delegated_history(plan.MAX_DELEGATE_ROUNDS)
        o, main = _orch(monkeypatch, messages=msgs)
        out = o.do_plan(main.state, _TWO)
        assert out == plan.delegate_cap_text(3)
        assert len(o.manager.agents) == 1              # nothing spawned
        assert main.state.turn_waiting is False
        # The controller answers from what it has: always accepted.
        assert o.do_plan(main.state, {'outcome': 'answer',
                                      'answer': 'Done so far: ...'}) == \
            plan.ANSWER_ACK

    def test_delegate_under_the_cap_runs(self, monkeypatch, fake_repo):
        msgs = self._delegated_history(plan.MAX_DELEGATE_ROUNDS - 1)
        o, main = _orch(monkeypatch, messages=msgs)
        out = o.do_plan(main.state, _TWO)
        assert out.startswith(plan.DELEGATED_PREFIX)
        assert len(o.manager.agents) == 3

    def test_a_new_request_resets_the_count(self, monkeypatch, fake_repo):
        msgs = self._delegated_history(plan.MAX_DELEGATE_ROUNDS)
        msgs += [{'role': 'assistant', 'content': 'Here is the summary.'},
                 {'role': 'user', 'content': _REQUEST + ' Again, please.'}]
        o, main = _orch(monkeypatch, messages=msgs)
        assert o.do_plan(main.state, _TWO).startswith(plan.DELEGATED_PREFIX)

    def test_second_plan_in_one_round_is_ignored(self, monkeypatch,
                                                 fake_repo) -> None:
        o, main = _orch(monkeypatch)
        o.do_plan(main.state, _TWO)
        out = o.do_plan(main.state, _TWO)
        assert 'ignored' in out and len(o.manager.agents) == 3

    def test_handlers_installed_and_cleared(self, monkeypatch) -> None:
        o, _ = _orch(monkeypatch)
        o.install_handlers()
        try:
            assert tools._plan_handler == o.plan
        finally:
            o.clear_handlers()
        assert tools._plan_handler is None


class TestPlanRouting:
    """Plan tasks route on their own labels: a review task on the review
    ladder of the default block, everything else on the default."""

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, labels_isolated):
        monkeypatch.setattr(config, 'DECISIONS_MODE', 'off')

    def _controller(self, o, main, request=_REQUEST):
        o.loop = _FakeLoop()
        o.launch = lambda a: None                    # type: ignore
        main.state.controller = True
        main.state.messages = [{'role': 'user', 'content': request}]
        return main

    def test_review_tasks_take_the_review_ladder(self, fake_repo) -> None:
        o, main = _tiers_orch()
        self._controller(o, main)
        out = o.do_plan(main.state, _TWO)
        assert out.startswith(plan.DELEGATED_PREFIX)
        ledger.flush()
        rows = fake_repo.stream('tasks')
        assert [r['route']['ladder'] for r in rows] == ['review', 'review']
        assert [r['model'] for r in rows] == ['aws/claude-5-5-opus',
                                              'aws/claude-5-sonnet']

    def test_other_kinds_take_the_default_ladder(self, fake_repo) -> None:
        o, main = _tiers_orch()
        self._controller(o, main, request='where is X defined?')
        o.do_plan(main.state, {'outcome': 'delegate', 'tasks': [
            {'goal': 'find where X is defined', 'kind': 'explain',
             'complexity': 'trivial'}]})
        ledger.flush()
        [row] = fake_repo.stream('tasks')
        assert row['route']['ladder'] == 'default'
        assert row['model'] == 'aws/claude-4-5-haiku'

    def test_all_refused_returns_refusal_and_turn_goes_on(self, fake_repo):
        o, main = _tiers_orch()
        self._controller(o, main)
        o._routing_settings.mode = 'local-only'      # only remote rungs
        out = o.do_plan(main.state, _TWO)
        assert out.startswith(plan.REFUSED_PREFIX)
        assert 'outcome answer' in out
        assert main.state.turn_waiting is False
        assert len(o.manager.agents) == 1
        ledger.flush()
        assert {r['status'] for r in fake_repo.stream('tasks')} == {'refused'}


class _Scripted(Adapter):
    """A fake adapter driving the real turn loop: the controller delegates
    on its first turn and answers on the mailbox turn, every worker
    answers with ``final_answer``. Forces every round (like LiteLLM)."""
    name = 'scripted'

    def __init__(self) -> None:
        self.rounds: list = []

    def available(self):
        return True

    def list_models(self):
        return []

    def activate(self, m):
        pass

    def summarise(self, t):
        return 's'

    def forces(self, tool: str) -> bool:
        return True

    def _calls(self):
        st = session.current()
        if st.task_id:
            return [('final_answer', {'text': f'{st.agent_id} says hi'},
                     'f1')]
        if conversation.mailbox_turn(st.messages):
            return [('plan', {'outcome': 'answer',
                              'answer': 'synth: both say hi'}, 'p2')]
        return [('plan', _TWO, 'p1')]

    def run_turn(self) -> None:
        def step():
            calls = self._calls()
            self.rounds.append((session.agent_id, calls[0][0]))
            session.messages.append({'role': 'assistant', 'content': '',
                                     'tool_calls': [
                                         {'id': ref, 'function': {
                                             'name': n, 'arguments': a}}
                                         for n, a, ref in calls]})
            return ('', calls)

        def run_tools(pending):
            for name, args, ref, _dup in pending:
                content = tools.execute_tool(name, args)
                session.messages.append(
                    {'role': 'tool', 'tool_name': name, 'tool_args': args,
                     'tool_call_id': ref, 'content': content})

        def add_user(text):
            session.messages.append({'role': 'user', 'content': text})
        turn.run_loop(step=step, run_tools=run_tools, add_user=add_user)


class TestMailboxSynthesisViaPlan:
    """End to end on the real loop: plan delegate -> workers answer with
    final_answer -> joined delivery -> plan answer; the readers of the
    final answer (orchestrator, bench) see the texts."""

    def test_round_trip(self, monkeypatch, fake_repo) -> None:
        monkeypatch.setattr(ledger, 'environment', lambda: dict(_FAKE_ENV))
        monkeypatch.setattr(config, 'DECISIONS_MODE', 'off')
        from guru.repositories.settings import RoutingSettings
        adapter = _Scripted()
        base = session.SessionState()
        base.adapter = adapter
        base.model = 'fake'
        run = bench.BenchRun(base, routing=RoutingSettings(controller=True))
        agents = asyncio.run(run.run(_REQUEST, timeout=20))
        assert run.worker_errors == []
        assert [a.title for a in agents] == ['main', 'agent1', 'agent2']
        main, a1, a2 = agents
        # The workers' final_answer texts are their answers.
        assert run.final_answer(a1) == 'agent1 says hi'
        assert bench._final_answer(a2) == 'agent2 says hi'
        assert bench._tool_names(a1) == []        # collapsed into text
        # The controller: delegate round kept natively, joined delivery,
        # then the answer plan collapsed into text.
        assert bench._final_answer(main) == 'synth: both say hi'
        assert bench._tool_names(main) == ['plan']
        joined = [m for m in main.state.messages
                  if conversation.msg_role(m) == 'user'
                  and conversation.is_mailbox(conversation.msg_content(m))]
        assert len(joined) == 1
        assert 'agent1 says hi' in joined[0]['content']
        assert 'agent2 says hi' in joined[0]['content']
        assert adapter.rounds == [('main', 'plan'), ('agent1', 'final_answer'),
                                  ('agent2', 'final_answer'),
                                  ('main', 'plan')] or sorted(
            adapter.rounds[1:3]) == [('agent1', 'final_answer'),
                                     ('agent2', 'final_answer')]
        ledger.flush()
        turns = fake_repo.stream('turns')
        assert len(turns) == 2
        assert turns[0]['tasks_spawned'] == 2
        assert turns[0]['tools_used'] == ['plan']
        assert turns[1]['tasks_spawned'] == 0
        assert turns[1]['controller_executed'] is False
        assert turns[1]['request'] == _REQUEST
        assert all(t['struggle']['protocol_violation'] == 0 for t in turns)
        tasks = fake_repo.stream('tasks')
        assert [t['status'] for t in tasks] == ['running', 'running', 'done',
                                                'done']
        assert {t['answer_len'] for t in tasks if t['status'] == 'done'} \
            == {len('agent1 says hi')}
        assert all(t['struggle']['protocol_violation'] == 0
                   for t in tasks if t['status'] == 'done')
