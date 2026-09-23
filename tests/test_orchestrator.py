"""Tests for the shared spawn/check/join mailbox (guru.orchestrator)."""
import os

import pytest

from guru import config, session
from guru.domain import conversation, ledger

_REAL_ENVIRONMENT = ledger.environment
_FAKE_ENV = {'git_sha': 'abc', 'git_dirty': False, 'cwd': '/w',
             'guru_version': '0.1.0', 'config_hash': 'h', 'judge_models': {}}


class TestOrchestrator:
    """The shared spawn/check/join mailbox (guru.orchestrator.Orchestrator)."""

    def _agent(self, title, parent=None, busy=False, answer='done'):
        from guru.agents import Agent
        a = Agent(id=title, title=title)
        a.parent = parent
        a.busy = busy
        a.task = f'task-{title}'
        a.state.messages = [{'role': 'assistant', 'content': answer}]
        return a

    def test_do_check_lists_children(self) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        o.manager.agents += [
            self._agent('agent1', parent=main, busy=True),
            self._agent('agent2', parent=main, busy=False)]
        out = o.do_check(main.state, 'all')
        assert 'agent1: running' in out and 'agent2: done' in out

    def test_do_join_all_done_delivers_immediately(self) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        main.busy = True             # busy -> deliver only queues (no launch)
        o.manager.agents.append(
            self._agent('agent1', parent=main, busy=False, answer='A1'))
        msg = o.do_join(main.state, ['agent1'])
        assert 'resuming' in msg.lower()
        assert any('A1' in p for p in main.queue)

    def test_report_barrier_waits_then_delivers_joined(self) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        main.busy = True             # busy -> deliver only queues (no launch)
        c1 = self._agent('agent1', parent=main, busy=True, answer='A1')
        c2 = self._agent('agent2', parent=main, busy=True, answer='A2')
        o.manager.agents += [c1, c2]
        o.barriers[main] = {'remaining': {'agent1', 'agent2'}, 'results': {}}
        c1.busy = False
        o.report(c1)
        assert main in o.barriers and main.queue == []   # still waiting
        c2.busy = False
        o.report(c2)
        assert main not in o.barriers                     # barrier resolved
        joined = main.queue[-1]
        assert 'A1' in joined and 'A2' in joined \
            and 'joined results' in joined

    def test_barrier_synthesis_prefixes_joined_payload(self) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        main.busy = True
        c1 = self._agent('agent1', parent=main, busy=False, answer='A1')
        o.manager.agents.append(c1)
        o.barriers[main] = {'remaining': {'agent1'}, 'results': {},
                            'synthesis': 'SYNTH-LEAD'}
        o.report(c1)
        assert main.queue[-1].startswith('SYNTH-LEAD')
        assert 'A1' in main.queue[-1]

    def test_do_join_waiting_sets_turn_waiting(self) -> None:
        """A join that opens a barrier flags the caller's turn as waiting
        so the turn loop ends the turn instead of polling."""
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        o.manager.agents.append(self._agent('agent1', parent=main, busy=True))
        assert main.state.turn_waiting is False
        msg = o.do_join(main.state, ['agent1'])
        assert 'Waiting for agent1' in msg
        assert main.state.turn_waiting is True
        assert main in o.barriers

    def test_do_join_all_done_does_not_wait(self) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        main.busy = True
        o.manager.agents.append(self._agent('agent1', parent=main))
        o.do_join(main.state, ['agent1'])
        assert main.state.turn_waiting is False

    def test_do_join_unknown_targets_do_not_wait(self) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        o.manager.agents.append(self._agent('agent1', parent=main, busy=True))
        o.do_join(main.state, ['nope'])
        assert main.state.turn_waiting is False

    def test_do_check_third_all_running_poll_ends_the_turn(self) -> None:
        """Polling running sub-agents: the first poll is plain, the second
        tells the model to join, the third ends the turn like a join."""
        from guru import orchestrator
        o = orchestrator.Orchestrator()
        main = o.manager.active
        o.manager.agents += [
            self._agent('agent1', parent=main, busy=True),
            self._agent('agent2', parent=main, busy=True)]
        first = o.do_check(main.state, 'all')
        assert 'agent1: running' in first and 'join' not in first
        assert main.state.check_polls == 1
        assert main.state.turn_waiting is False
        second = o.do_check(main.state, 'all')
        assert 'agent1: running' in second and 'join' in second
        assert main.state.check_polls == 2
        assert main.state.turn_waiting is False
        third = o.do_check(main.state, 'all')
        assert third == orchestrator.CHECK_WAIT_TEXT
        assert 'still running' in third and 'resumed' in third
        assert main.state.turn_waiting is True
        # The running children are joined so their results arrive together.
        assert o.barriers[main]['remaining'] == {'agent1', 'agent2'}

    def test_do_check_counter_resets_when_a_child_is_done(self) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        c1 = self._agent('agent1', parent=main, busy=True)
        o.manager.agents += [c1, self._agent('agent2', parent=main,
                                             busy=True)]
        o.do_check(main.state, 'all')
        o.do_check(main.state, 'all')
        c1.busy = False
        out = o.do_check(main.state, 'all')
        assert 'agent1: done' in out and 'still running' not in out
        assert main.state.check_polls == 0
        assert main.state.turn_waiting is False
        c1.busy = True
        o.do_check(main.state, 'all')
        o.do_check(main.state, 'all')
        assert main.state.turn_waiting is False       # count restarted

    def test_do_check_named_running_target_counts_too(self) -> None:
        from guru import orchestrator
        o = orchestrator.Orchestrator()
        main = o.manager.active
        o.manager.agents += [
            self._agent('agent1', parent=main, busy=True),
            self._agent('agent2', parent=main, busy=False)]
        assert 'running' in o.do_check(main.state, 'agent1')
        assert 'running' in o.do_check(main.state, 'agent1')
        assert o.do_check(main.state, 'agent1') == \
            orchestrator.CHECK_WAIT_TEXT
        assert main.state.turn_waiting is True
        assert o.barriers[main]['remaining'] == {'agent1'}   # not agent2

    def test_do_check_waiting_merges_into_an_open_barrier(self) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        o.manager.agents += [
            self._agent('agent1', parent=main, busy=True),
            self._agent('agent2', parent=main, busy=True)]
        o.barriers[main] = {'remaining': {'agent1'}, 'results': {}}
        for _ in range(3):
            o.do_check(main.state, 'agent2')
        assert o.barriers[main]['remaining'] == {'agent1', 'agent2'}

    def test_do_check_done_target_never_waits(self) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        o.manager.agents.append(self._agent('agent1', parent=main,
                                            answer='A1'))
        for _ in range(4):
            out = o.do_check(main.state, 'agent1')
        assert 'A1' in out and main.state.turn_waiting is False

    def test_spawn_panel_runs_children_and_synthesises(self) -> None:
        import asyncio
        from guru.adapters.base import Adapter
        from guru.orchestrator import Orchestrator

        class FakeAdapter(Adapter):
            name = 'fake'
            def available(self): return True
            def list_models(self): return []
            def activate(self, m): pass
            def summarise(self, t): return 's'

            def run_turn(self):
                st = session.current()
                st.messages.append(
                    {'role': 'assistant', 'content': 'ok:' + (
                        st.active_role or 'main')})

        async def drive():
            o = Orchestrator()
            o.loop = asyncio.get_running_loop()
            main = o.manager.active
            main.state.adapter = FakeAdapter()
            main.state.model = 'fake'
            o.attach_console(main)
            tasks = [('review X for correctness', 'developer', 'code-review'),
                     ('review X for security', 'security-engineer',
                      'code-review')]
            o.spawn_panel(main, tasks, synthesis='SYNTH')
            for _ in range(300):
                await asyncio.sleep(0.02)
                if not any(a.busy or a.queue for a in o.manager.agents):
                    break
            return o

        o = asyncio.run(drive())
        assert len(o.manager.agents) == 3            # main + 2 panel agents
        main = o.manager.agents[0]
        # main ran a synthesis turn triggered by the joined delivery…
        assert any(conversation.msg_content(m) == 'ok:main'
                   for m in main.state.messages)
        # …and the joined delivery carried the synthesis lead-in
        assert any(isinstance(m, dict) and 'SYNTH' in (m.get('content') or '')
                   for m in main.state.messages)


class TestTaskRecords:
    """spawn/spawn_panel write a running TaskRecord; on_done closes it."""

    @pytest.fixture(autouse=True)
    def _constant_environment(self, monkeypatch) -> None:
        """No git subprocesses per spawn; one test restores the real one."""
        monkeypatch.setattr(ledger, 'environment', lambda: dict(_FAKE_ENV))

    def _tasks(self, repo):
        from guru.domain import ledger
        ledger.flush()
        return repo.stream('tasks')

    def _child(self, o, main, **kw):
        child = o._make_child(main, task='review auth', role='developer',
                              skill='code-review', **kw)
        child.queue.clear()          # work() pops the task before on_done
        child.started = 0.0
        return child

    def test_spawn_writes_running_then_done(
            self, monkeypatch, fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        repo = fake_repo
        o = Orchestrator()
        main = o.manager.active
        main.busy = True
        main.state.model = 'qwen3:14b'
        main.state.turn_id = 'T1'
        child = self._child(o, main)
        assert child.state.task_id == child.task_rec.task_id
        assert child.state.agent_id == child.id
        assert child.state.turn_id == 'T1' and child.task_rec.turn_id == 'T1'
        assert child.task == 'review auth' and child.parent is main
        assert child.state.active_role == 'developer'
        child.state.messages.append(
            {'role': 'tool', 'tool_name': 'read_file', 'content': 'x'})
        child.state.messages.append({'role': 'assistant', 'content': 'A1'})
        child.state.session_in, child.state.session_out = 40, 6
        child.busy = False
        o.on_done(child)
        tasks = self._tasks(repo)
        assert [t['status'] for t in tasks] == ['running', 'done']
        assert tasks[0]['task_id'] == tasks[1]['task_id']
        assert tasks[1]['answer_len'] == 2 and tasks[1]['parent'] == 'main'
        assert tasks[1]['role'] == 'developer'
        assert tasks[1]['skill'] == 'code-review'
        assert tasks[1]['model'] == 'qwen3:14b'
        assert tasks[1]['tools_used'] == ['read_file']
        assert tasks[1]['tokens_in'] == 40 and tasks[1]['tokens_out'] == 6
        assert tasks[1]['seconds'] > 0 and tasks[1]['cost_usd'] == 0.0
        assert tasks[1]['calls'] == 0 and tasks[1]['cost_known'] is True
        assert 'A1' in main.queue[-1]           # result still delivered
        assert child.task_rec is None           # closed exactly once

    def test_finish_row_carries_accumulators(
            self, monkeypatch, fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        main.busy = True
        child = self._child(o, main)
        child.state.messages.append({'role': 'assistant', 'content': 'A1'})
        child.state.call_count = 3
        child.state.cost_usd = 0.02
        child.state.struggle['tool_errors'] = 2
        o.on_done(child)
        done = self._tasks(fake_repo)[1]
        assert done['calls'] == 3 and done['cost_usd'] == 0.02
        assert done['cost_known'] is True
        assert done['struggle']['tool_errors'] == 2
        assert done['struggle']['refusals'] == 0

    def test_finish_row_cost_none_when_unknown(
            self, monkeypatch, fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        main.busy = True
        child = self._child(o, main)
        child.state.cost_usd = 0.02
        child.state.cost_known = False
        o.on_done(child)
        done = self._tasks(fake_repo)[1]
        assert done['cost_usd'] is None and done['cost_known'] is False

    def test_rows_carry_environment_prompt_and_tools(
            self, monkeypatch, fake_repo) -> None:
        import hashlib
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        main.busy = True
        child = self._child(o, main)
        prompt = child.state.messages[0]['content']
        expected = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]
        assert child.task_rec.prompt_sha == expected
        assert child.task_rec.tools_active == \
            sorted(child.state.active_tool_names)
        child.state.messages.append({
            'role': 'assistant', 'content': '', 'tool_calls': [{'function': {
                'name': 'read_file', 'arguments': {'path': 'x.py'}}}]})
        child.state.messages.append(
            {'role': 'tool', 'tool_name': 'read_file', 'content': 'd'})
        child.state.messages.append({'role': 'assistant', 'content': 'A1'})
        o.on_done(child)
        running, done = self._tasks(fake_repo)
        assert running['env']['git_sha'] == 'abc' and done['env'] == \
            running['env']
        assert running['prompt_sha'] == expected == done['prompt_sha']
        assert 'read_file' in done['tools_active']
        assert done['tools_active'] == sorted(child.state.active_tool_names)
        assert running['transcript_path'] == ''
        assert done['transcript_path'] == \
            fake_repo.transcript_path(done['task_id'])
        ledger.flush()                      # transcript write is off-thread
        saved = fake_repo.transcripts[done['task_id']]
        assert saved[0]['role'] == 'system' and saved[-1] == {
            'role': 'assistant', 'content': 'A1'}
        assert saved[-3]['tool_calls'] == [
            {'name': 'read_file', 'args': {'path': 'x.py'}}]
        assert done['tools_used'] == ['read_file']

    def test_real_environment_snapshot(self, monkeypatch, fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        monkeypatch.setattr(ledger, 'environment', _REAL_ENVIRONMENT)
        o = Orchestrator()
        child = self._child(o, o.manager.active)
        env = child.task_rec.env
        assert set(env) == {'git_sha', 'git_dirty', 'cwd', 'guru_version',
                            'config_hash', 'judge_models'}
        assert env['cwd'] == os.getcwd() and env['guru_version']
        assert isinstance(env['judge_models'], dict)

    def test_provider_error_without_answer_is_error(
            self, monkeypatch, fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        main.busy = True
        child = self._child(o, main)
        child.state.struggle['provider_errors'] = 1
        o.on_done(child)
        done = self._tasks(fake_repo)[1]
        assert done['status'] == 'error' and done['answer_len'] == 0
        assert done['struggle']['provider_errors'] == 1

    def test_provider_error_with_answer_stays_done(
            self, monkeypatch, fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        main.busy = True
        child = self._child(o, main)
        child.state.struggle['provider_errors'] = 1     # retried and won
        child.state.messages.append({'role': 'assistant', 'content': 'A1'})
        o.on_done(child)
        assert self._tasks(fake_repo)[1]['status'] == 'done'

    def test_cancel_wins_over_provider_error(
            self, monkeypatch, fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        main.busy = True
        child = self._child(o, main)
        child.state.struggle['provider_errors'] = 1
        child.state.cancel_requested = True
        o.on_done(child)
        assert self._tasks(fake_repo)[1]['status'] == 'cancelled'

    def test_second_on_done_writes_no_second_finish_row(
            self, monkeypatch, fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        main.busy = True
        child = self._child(o, main)
        child.state.messages.append({'role': 'assistant', 'content': 'A1'})
        o.on_done(child)
        o.on_done(child)
        assert [t['status'] for t in self._tasks(fake_repo)] == \
            ['running', 'done']

    def test_worker_error_writes_error_status(
            self, monkeypatch, fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        repo = fake_repo
        o = Orchestrator()
        main = o.manager.active
        main.busy = True
        child = self._child(o, main)
        child.status = 'error'       # what work() sets before on_done
        o.on_done(child)
        tasks = self._tasks(repo)
        assert [t['status'] for t in tasks] == ['running', 'error']
        assert tasks[1]['answer_len'] == 0      # '(no answer produced)'
        assert child.status == 'idle'

    def test_cancelled_turn_writes_cancelled_status(
            self, monkeypatch, fake_repo):
        from guru.orchestrator import Orchestrator
        repo = fake_repo
        o = Orchestrator()
        main = o.manager.active
        main.busy = True
        child = self._child(o, main)
        child.state.cancel_requested = True
        o.on_done(child)
        tasks = self._tasks(repo)
        assert [t['status'] for t in tasks] == ['running', 'cancelled']

    def test_panel_index_offsets_titles(self, monkeypatch, fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        repo = fake_repo
        o = Orchestrator()
        main = o.manager.active
        main.state.turn_id = 'T9'
        a = self._child(o, main, index=0)
        b = self._child(o, main, index=1)
        assert (a.title, b.title) == ('agent1', 'agent2')
        assert self._tasks(repo)[0]['turn_id'] == 'T9'
        assert a.task_rec.task_id != b.task_rec.task_id
        assert len(self._tasks(repo)) == 2

    def test_main_agent_done_writes_no_task_row(
            self, monkeypatch, fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        repo = fake_repo
        o = Orchestrator()
        main = o.manager.active
        o.on_done(main)
        assert self._tasks(repo) == []

    def test_work_marks_error_status_for_on_done(
            self, monkeypatch, fake_repo) -> None:
        import asyncio
        from guru.adapters.base import Adapter
        from guru.orchestrator import Orchestrator

        class Boom(Adapter):
            name = 'boom'
            def available(self): return True
            def list_models(self): return []
            def activate(self, m): pass
            def summarise(self, t): return 's'

            def run_turn(self):
                raise RuntimeError('kaboom')

        repo = fake_repo
        seen: list = []

        class Watching(Orchestrator):
            def on_done(self, agent) -> None:
                seen.append(agent.status)
                super().on_done(agent)

        async def drive():
            o = Watching()
            o.loop = asyncio.get_running_loop()
            main = o.manager.active
            main.state.adapter = Boom()
            main.busy = True
            child = o._make_child(main, task='t')
            o.attach_console(child)
            o.manager.agents.append(child)
            o.launch(child)
            for _ in range(200):
                await asyncio.sleep(0.01)
                if not child.busy:
                    break
            return child

        child = asyncio.run(drive())
        assert seen == ['error'] and child.status == 'idle'
        assert child.started > 0
        assert [t['status'] for t in self._tasks(repo)] == \
            ['running', 'error']


# --- routing (Task 4.4), controller (4.5) and the local retry (4.6) ---------

def _fake_adapter(name: str, remote: bool):
    """A minimal Adapter subclass with the given ``remote`` flag."""
    from guru.adapters.base import Adapter

    class Fake(Adapter):
        def available(self): return True
        def list_models(self): return []
        def activate(self, m): pass
        def summarise(self, t): return 's'
        def run_turn(self): pass
    a = Fake()
    a.name = name
    a.remote = remote
    return a


class MarkerScanner:
    """Finds every occurrence of ``SECRET`` (kind ``marker``)."""

    def scan(self, text: str) -> list:
        from guru.domain.policy import Finding
        out, start = [], 0
        while True:
            i = text.find('SECRET', start)
            if i < 0:
                return out
            out.append(Finding('marker', i, i + 6, 'SECRET'))
            start = i + 6


class TestRouting:
    """_make_child scans, resolves, confirms and applies a Route."""

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        from guru.domain import policy, spend
        monkeypatch.setattr(ledger, 'environment', lambda: dict(_FAKE_ENV))
        spend.reset()
        spend.set_spend_asker(None)
        policy.set_scanner(None)
        yield
        spend.reset()
        spend.set_spend_asker(None)
        policy.set_scanner(None)

    def _registry(self):
        from guru.repositories.adapters import AdapterRegistry
        return AdapterRegistry([_fake_adapter('Local', False),
                                _fake_adapter('Remote', True)])

    def _settings(self, **kw):
        from guru.repositories.settings import RoutingSettings, RungSpec
        kw.setdefault('ladders', {'default': [
            RungSpec('Local', 'qwen3:14b', 'standard', default=True),
            RungSpec('Remote', 'claude-sonnet-5', 'hard')]})
        kw.setdefault('spend_confirm', 'auto')
        return RoutingSettings(**kw)

    def _orch(self, registry=None, settings=None):
        from guru.orchestrator import Orchestrator
        registry = self._registry() if registry is None else registry
        o = Orchestrator(registry=registry,
                         routing=settings or self._settings())
        main = o.manager.active
        main.busy = True
        main.state.adapter = registry.get('Local')
        main.state.model = 'qwen3:32b'
        return o, main

    def test_standard_task_takes_the_lowest_rung(self, fake_repo) -> None:
        o, main = self._orch()
        child = o._make_child(main, 'explain X', kind='explain')
        assert child.state.adapter.name == 'Local'
        assert child.state.model == 'qwen3:14b'
        assert child.task_rec.route['rung_index'] == 0
        assert child.task_rec.kind == 'explain'
        assert child.task_rec.complexity == 'standard'

    def test_hard_task_goes_remote_in_auto_mode(self, fake_repo) -> None:
        o, main = self._orch()
        child = o._make_child(main, 'fix it', kind='debug', complexity='hard')
        assert child.state.adapter.name == 'Remote'
        assert child.state.model == 'claude-sonnet-5'
        assert child.task_rec.confirmation == 'granted'
        assert child.task_rec.adapter == 'Remote'
        assert child.task_rec.model == 'claude-sonnet-5'

    def test_local_only_never_picks_remote(self, fake_repo) -> None:
        o, main = self._orch(settings=self._settings(mode='local-only'))
        child = o._make_child(main, 'fix it', complexity='hard')
        assert child.state.adapter.name == 'Local'
        assert any(r.startswith('mode:local-only') for r in
                   child.task_rec.reason)

    def test_scan_finding_forces_local(self, fake_repo) -> None:
        from guru.domain import policy
        policy.set_scanner(MarkerScanner())
        o, main = self._orch()
        child = o._make_child(main, 'rotate SECRET now', complexity='hard')
        assert child.state.adapter.name == 'Local'
        assert child.task_rec.findings == 1
        assert any(r.startswith('scan:') for r in child.task_rec.reason)

    def test_scan_off_ignores_findings(self, fake_repo) -> None:
        from guru.domain import policy
        policy.set_scanner(MarkerScanner())
        o, main = self._orch(settings=self._settings(secret_scan=False))
        child = o._make_child(main, 'rotate SECRET now', complexity='hard')
        assert child.state.adapter.name == 'Remote'
        assert child.task_rec.findings == 0

    def test_ask_mode_asks_once_then_remembers(self, fake_repo) -> None:
        from guru.domain import spend
        calls: list = []
        spend.set_spend_asker(lambda q: calls.append(q) or True)
        o, main = self._orch(settings=self._settings(spend_confirm='ask'))
        a = o._make_child(main, 't1', complexity='hard')
        b = o._make_child(main, 't2', complexity='hard', index=1)
        assert len(calls) == 1
        assert a.state.adapter.name == b.state.adapter.name == 'Remote'
        assert a.task_rec.confirmation == 'granted'
        assert 'needs_confirmation' not in a.task_rec.reason

    def test_ask_mode_never_asks_for_a_local_pick(self, fake_repo) -> None:
        from guru.domain import spend
        calls: list = []
        spend.set_spend_asker(lambda q: calls.append(q) or True)
        o, main = self._orch(settings=self._settings(spend_confirm='ask'))
        child = o._make_child(main, 't1', complexity='trivial')
        assert calls == [] and child.task_rec.confirmation == 'pending'

    def test_declined_falls_back_to_local(self, fake_repo) -> None:
        from guru.domain import spend
        spend.set_spend_asker(lambda q: False)
        o, main = self._orch(settings=self._settings(spend_confirm='ask'))
        child = o._make_child(main, 't1', complexity='hard')
        assert child.state.adapter.name == 'Local'
        assert child.task_rec.confirmation == 'declined'
        assert any(r.startswith('confirmation:declined')
                   for r in child.task_rec.reason)

    def test_never_mode_strips_nothing_but_records_never(
            self, fake_repo) -> None:
        o, main = self._orch(settings=self._settings(spend_confirm='never'))
        child = o._make_child(main, 't1', complexity='hard')
        assert child.state.adapter.name == 'Remote'
        assert child.task_rec.confirmation == 'never'

    def test_route_and_reason_recorded_on_task_row(self, fake_repo) -> None:
        o, main = self._orch(settings=self._settings(mode='local-only'))
        child = o._make_child(main, 'fix it', kind='debug',
                              complexity='hard')
        child.queue.clear()
        child.started = 0.0
        reason = list(child.task_rec.reason)
        child.state.messages.append({'role': 'assistant', 'content': 'A'})
        o.on_done(child)
        ledger.flush()
        running, done = fake_repo.stream('tasks')
        for row in (running, done):
            assert row['route']['adapter'] == 'Local'
            assert row['route']['kind'] == 'debug'
            assert row['route']['complexity'] == 'hard'
            assert row['reason'] == reason
            assert row['findings'] == 0 and row['confirmation'] == 'granted'
            assert row['retry_of'] == ''
        assert reason and reason == running['route']['reason']

    def test_refused_route_writes_row_and_spawns_nothing(
            self, fake_repo) -> None:
        from guru.domain import policy
        policy.set_scanner(MarkerScanner())
        o, main = self._orch(settings=self._settings(mode='remote-only'))
        main.state.adapter = self._registry().get('Remote')
        child = o._make_child(main, 'use SECRET', complexity='hard')
        assert child is None
        ledger.flush()
        rows = fake_repo.stream('tasks')
        assert [r['status'] for r in rows] == ['refused']
        assert rows[0]['route']['refused'] is True
        assert rows[0]['findings'] == 1
        assert any(r.startswith('refused') for r in rows[0]['reason'])
        assert len(o.manager.agents) == 1

    def test_spawn_tool_reports_refusal(self, fake_repo) -> None:
        from guru.domain import policy
        policy.set_scanner(MarkerScanner())
        o, main = self._orch(settings=self._settings(mode='remote-only'))
        token = session.use(main.state)
        try:
            out = o.spawn('use SECRET', complexity='hard')
        finally:
            session.reset(token)
        assert 'Could not spawn' in out and 'scan:1 finding' in out
        assert 'refused:' in out
        assert len(o.manager.agents) == 1

    def test_no_registry_is_inert(self, fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator(routing=self._settings(mode='local-only'))
        main = o.manager.active
        main.busy = True
        main.state.adapter = _fake_adapter('Remote', True)
        main.state.model = 'claude-opus-5'
        child = o._make_child(main, 't', complexity='hard')
        assert child.state.adapter is main.state.adapter
        assert child.state.model == 'claude-opus-5'
        assert child.task_rec.route is None
        assert child.task_rec.reason == ['routing:disabled (no registry)']
        assert child.task_rec.adapter == 'Remote'

    def test_invalid_settings_fall_back_to_defaults(self, monkeypatch):
        from guru.orchestrator import Orchestrator
        from guru.repositories import settings as rs
        monkeypatch.setattr(rs, 'load_routing',
                            lambda: (_ for _ in ()).throw(ValueError('bad')))
        o = Orchestrator()
        cfg = o._routing()
        assert cfg.mode == 'local-and-remote' and cfg.ladders == {}

    def test_local_main_fallback_when_ladder_empties(self, fake_repo):
        from guru.domain import routing
        from guru.repositories.settings import RungSpec
        settings = self._settings(
            mode='local-only',
            ladders={'default': [RungSpec('Remote', 'claude', 'hard')]})
        o, main = self._orch(settings=settings)
        child = o._make_child(main, 't')
        assert child.state.adapter.name == 'Local'
        assert child.state.model == 'qwen3:32b'          # the parent's
        assert routing.LOCAL_MAIN_TAKEN in child.task_rec.reason

    def test_unknown_route_adapter_keeps_parent(self, fake_repo) -> None:
        from guru.domain import routing
        o, main = self._orch()
        monkeypatch_ladders = {'default': routing.Ladder([
            routing.Rung('Ghost', 'g1', 'hard', remote=False)])}
        o._ladders = monkeypatch_ladders
        child = o._make_child(main, 't')
        assert child.state.adapter.name == 'Local'
        assert child.state.model == 'qwen3:32b'
        assert any(r.startswith('adapter:Ghost unknown')
                   for r in child.task_rec.reason)

    def test_labels_are_normalised(self, fake_repo) -> None:
        o, main = self._orch()
        child = o._make_child(main, 't', kind='Weird', complexity='HARD')
        assert child.task_rec.kind == 'other'
        assert child.task_rec.complexity == 'hard'

    def test_spawn_panel_skips_refused_children(self, fake_repo) -> None:
        import asyncio
        from guru.domain import policy
        policy.set_scanner(MarkerScanner())
        o, main = self._orch(settings=self._settings(mode='remote-only'))
        main.state.adapter = self._registry().get('Remote')

        async def drive():
            o.loop = asyncio.get_running_loop()
            titles = o.spawn_panel(
                main, [('review SECRET', 'developer', 'code-review')])
            await asyncio.sleep(0.02)
            return titles
        assert asyncio.run(drive()) == []
        assert main not in o.barriers


class TestPreApprovedParent:
    """Decision A: with defaults (no ladders) a REMOTE parent keeps running
    its own model — no question, no refusal, findings or not."""

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        from guru.domain import policy, spend
        monkeypatch.setattr(ledger, 'environment', lambda: dict(_FAKE_ENV))
        spend.reset()
        spend.set_spend_asker(None)
        policy.set_scanner(None)
        yield
        spend.reset()
        spend.set_spend_asker(None)
        policy.set_scanner(None)

    def _orch(self):
        from guru.orchestrator import Orchestrator
        from guru.repositories.adapters import AdapterRegistry
        from guru.repositories.settings import RoutingSettings
        registry = AdapterRegistry([_fake_adapter('Remote', True)])
        o = Orchestrator(registry=registry, routing=RoutingSettings())
        main = o.manager.active
        main.busy = True
        main.state.adapter = registry.get('Remote')
        main.state.model = 'claude-opus-5'
        return o, main

    def test_defaults_keep_remote_parent_without_asking(self, fake_repo):
        from guru.domain import spend
        calls: list = []
        spend.set_spend_asker(lambda q: calls.append(q) or False)
        o, main = self._orch()
        child = o._make_child(main, 'fix it', complexity='hard')
        assert child is not None
        assert child.state.adapter.name == 'Remote'
        assert child.state.model == 'claude-opus-5'
        assert calls == []
        assert child.task_rec.confirmation == 'pending'
        assert child.task_rec.reason == [
            'fallback:local_main (pre-approved)']
        ledger.flush()
        assert [r['status'] for r in fake_repo.stream('tasks')] == \
            ['running']

    def test_defaults_with_finding_still_keep_remote_parent(
            self, fake_repo):
        from guru.domain import policy, spend
        calls: list = []
        spend.set_spend_asker(lambda q: calls.append(q) or False)
        policy.set_scanner(MarkerScanner())
        o, main = self._orch()
        child = o._make_child(main, 'rotate SECRET', complexity='hard')
        assert child is not None
        assert child.state.adapter.name == 'Remote'
        assert child.task_rec.findings == 1
        assert calls == []
        ledger.flush()
        assert all(r['status'] != 'refused'
                   for r in fake_repo.stream('tasks'))


class TestLoopThreadSafety:
    """Critical 1: the spend asker never runs on the loop thread; the
    /review path pre-confirms in a worker."""

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        from guru.domain import spend
        monkeypatch.setattr(ledger, 'environment', lambda: dict(_FAKE_ENV))
        spend.reset()
        yield
        spend.reset()
        spend.set_spend_asker(None)

    def _orch(self):
        from guru.orchestrator import Orchestrator
        from guru.repositories.adapters import AdapterRegistry
        from guru.repositories.settings import RoutingSettings, RungSpec
        registry = AdapterRegistry([_fake_adapter('Local', False),
                                    _fake_adapter('Remote', True)])
        settings = RoutingSettings(
            spend_confirm='ask', present=True,
            ladders={'default': [RungSpec('Remote', 'claude', 'hard')]})
        o = Orchestrator(registry=registry, routing=settings)
        main = o.manager.active
        main.state.adapter = registry.get('Local')
        main.state.model = 'qwen3:32b'
        return o, main

    def test_spawn_panel_on_loop_defers_instead_of_asking(self, fake_repo):
        import asyncio
        import threading
        from guru.domain import spend
        threads: list = []
        spend.set_spend_asker(
            lambda q: threads.append(threading.get_ident()) or True)
        o, main = self._orch()

        async def drive():
            o.loop = asyncio.get_running_loop()
            main.busy = True
            titles = o.spawn_panel(
                main, [('review X', 'developer', 'code-review')])
            await asyncio.sleep(0.02)
            return titles
        titles = asyncio.run(drive())
        assert titles == ['agent1']
        assert threads == []                       # never asked on the loop
        child = o.manager.agents[1]
        assert child.state.adapter.name == 'Local'     # pre-approved parent
        ledger.flush()
        row = fake_repo.stream('tasks')[0]
        assert 'confirmation:deferred (loop thread)' in row['reason']
        assert row['confirmation'] == 'pending'
        assert spend.status('ask') == 'pending'        # run still unanswered

    def test_preconfirm_runs_asker_off_loop_then_panel_goes_remote(
            self, fake_repo):
        import asyncio
        import threading
        from guru.domain import spend
        threads: list = []
        spend.set_spend_asker(
            lambda q: threads.append(threading.get_ident()) or True)
        o, main = self._orch()

        async def drive():
            loop = asyncio.get_running_loop()
            o.loop = loop
            main.busy = True
            await loop.run_in_executor(None, o.preconfirm_spend)
            o.spawn_panel(main, [('review X', 'developer', 'code-review')])
            await asyncio.sleep(0.02)
            return threading.get_ident()
        loop_thread = asyncio.run(drive())
        assert len(threads) == 1 and threads[0] != loop_thread
        assert spend.status('ask') == 'granted'
        child = o.manager.agents[1]
        assert child.state.adapter.name == 'Remote'
        ledger.flush()
        assert fake_repo.stream('tasks')[0]['confirmation'] == 'granted'

    def test_preconfirm_is_a_noop_without_remote_rungs(self) -> None:
        from guru.domain import spend
        from guru.orchestrator import Orchestrator
        from guru.repositories.adapters import AdapterRegistry
        from guru.repositories.settings import RoutingSettings, RungSpec
        calls: list = []
        spend.set_spend_asker(lambda q: calls.append(q) or True)
        registry = AdapterRegistry([_fake_adapter('Local', False)])
        o = Orchestrator(registry=registry, routing=RoutingSettings(
            spend_confirm='ask',
            ladders={'default': [RungSpec('Local', 'q', 'hard')]}))
        o.preconfirm_spend()
        assert calls == [] and spend.status('ask') == 'pending'

    def test_preconfirm_is_a_noop_in_auto_mode(self) -> None:
        from guru.domain import spend
        calls: list = []
        spend.set_spend_asker(lambda q: calls.append(q) or True)
        o, _ = self._orch()
        o._routing_settings.spend_confirm = 'auto'
        o.preconfirm_spend()
        assert calls == []


class TestControllerConfigure:
    """configure(controller=True) installs the controller hint + tool set."""

    def _orch(self):
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        base = session.SessionState()
        base.model = 'm'
        return o, base

    def test_controller_gets_hint_flag_and_tools(self) -> None:
        from guru.agents import Agent
        from guru.domain import tools
        o, base = self._orch()
        agent = Agent(id='main', title='main')
        o.configure(agent, base, can_spawn=True, controller=True)
        st = agent.state
        assert st.controller is True and st.can_spawn is True
        assert config.CONTROLLER_HINT in st.messages[0]['content']
        assert config.DELEGATION_HINT not in st.messages[0]['content']
        assert st.active_tools == [tools.spawn, tools.check, tools.join,
                                   tools.use_skill]
        assert st.active_tool_names == set()

    def test_non_controller_unchanged(self) -> None:
        from guru.agents import Agent
        from guru.domain import tools
        o, base = self._orch()
        agent = Agent(id='main', title='main')
        o.configure(agent, base, can_spawn=True)
        st = agent.state
        assert st.controller is False
        assert config.DELEGATION_HINT in st.messages[0]['content']
        assert tools.search_tools in st.active_tools

    def test_controller_requires_can_spawn(self) -> None:
        from guru.agents import Agent
        o, base = self._orch()
        agent = Agent(id='a', title='a')
        o.configure(agent, base, can_spawn=False, controller=True)
        assert agent.state.controller is False
        assert config.CONTROLLER_HINT not in agent.state.messages[0]['content']

    def test_controller_prompt_names_cwd_and_repo_rule(
            self, tmp_path, monkeypatch) -> None:
        """The rendered controller system prompt carries the working
        directory and the 'never ask which repository' rule."""
        from guru.agents import Agent
        monkeypatch.chdir(tmp_path)
        o, base = self._orch()
        base.git_branch = 'feat/routing'
        agent = Agent(id='main', title='main')
        o.configure(agent, base, can_spawn=True, controller=True)
        token = session.use(agent.state)
        try:
            conversation.refresh_system_context()
        finally:
            session.reset(token)
        body = agent.state.messages[0]['content']
        assert str(tmp_path.resolve()) in body
        assert 'feat/routing' in body
        assert 'Never ask which repository' in body


class TestLocalRetry:
    """A remote child that fails without an answer is respawned once
    locally (design §5)."""

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        from guru.domain import spend
        monkeypatch.setattr(ledger, 'environment', lambda: dict(_FAKE_ENV))
        spend.reset()
        yield
        spend.reset()

    def _orch(self, mode='local-and-remote', local_rung=True):
        from guru.orchestrator import Orchestrator
        from guru.repositories.adapters import AdapterRegistry
        from guru.repositories.settings import RoutingSettings, RungSpec
        registry = AdapterRegistry([_fake_adapter('Local', False),
                                    _fake_adapter('Remote', True)])
        rungs = [RungSpec('Remote', 'claude-sonnet-5', 'hard')]
        if local_rung:
            rungs.insert(
                0, RungSpec('Local', 'qwen3:14b', 'standard', default=True))
        settings = RoutingSettings(
            mode=mode, spend_confirm='auto', ladders={'default': rungs})
        o = Orchestrator(registry=registry, routing=settings)
        launched: list = []
        o.launch = launched.append                # type: ignore[assignment]
        main = o.manager.active
        main.busy = True
        main.state.adapter = registry.get('Local')
        main.state.model = 'qwen3:32b'
        main.state.turn_id = 'T1'
        return o, main, launched

    def _failed_remote_child(self, o, main):
        child = o._make_child(main, 'fix it', role='developer',
                              kind='debug', complexity='hard')
        assert child.state.adapter.name == 'Remote'
        o.manager.agents.append(child)
        child.queue.clear()
        child.started = 0.0
        child.state.last_error = 'overloaded'
        child.state.struggle['provider_errors'] = 1
        return child

    def test_remote_failure_respawns_locally_once(self, fake_repo) -> None:
        o, main, launched = self._orch()
        child = self._failed_remote_child(o, main)
        o.on_done(child)
        ledger.flush()
        rows = fake_repo.stream('tasks')
        # the original closes (fell_back) before the retry's running row
        assert [r['status'] for r in rows] == [
            'running', 'fell_back', 'running']
        assert rows[1]['task_id'] == rows[0]['task_id']
        retry = launched[0]
        assert retry in o.manager.agents and retry.parent is main
        assert retry.task == 'fix it'
        assert retry.state.adapter.name == 'Local'
        assert retry.state.model == 'qwen3:14b'
        assert retry.task_rec.retry_of == rows[0]['task_id']
        assert retry.task_rec.role == 'developer'
        assert retry.task_rec.kind == 'debug'
        assert retry.task_rec.complexity == 'hard'
        assert retry.task_rec.reason[0] == 'retry:local after remote failure'
        assert rows[2]['retry_of'] == rows[0]['task_id']
        assert main.queue == []                    # nothing reported yet

    def test_retry_failure_is_error_and_reported(self, fake_repo) -> None:
        o, main, launched = self._orch()
        child = self._failed_remote_child(o, main)
        o.on_done(child)
        retry = launched[0]
        retry.queue.clear()
        retry.started = 0.0
        retry.busy = False
        retry.state.last_error = 'still failing'
        retry.state.struggle['provider_errors'] = 1
        o.on_done(retry)
        ledger.flush()
        rows = fake_repo.stream('tasks')
        assert [r['status'] for r in rows] == \
            ['running', 'fell_back', 'running', 'error']
        assert len(launched) == 1                  # no second retry
        assert main.queue and 'no answer produced' in main.queue[-1]

    def test_retry_takes_the_failed_seat_in_a_barrier(self, fake_repo):
        o, main, launched = self._orch()
        child = self._failed_remote_child(o, main)
        o.barriers[main] = {'remaining': {child.title}, 'results': {}}
        o.on_done(child)
        retry = launched[0]
        assert o.barriers[main]['remaining'] == {retry.title}

    def test_local_failure_is_plain_error(self, fake_repo) -> None:
        o, main, launched = self._orch()
        child = o._make_child(main, 'explain', complexity='trivial')
        assert child.state.adapter.name == 'Local'
        o.manager.agents.append(child)
        child.queue.clear()
        child.started = 0.0
        child.state.last_error = 'boom'
        child.state.struggle['provider_errors'] = 1
        o.on_done(child)
        ledger.flush()
        assert [r['status'] for r in fake_repo.stream('tasks')] == \
            ['running', 'error']
        assert launched == [] and main.queue

    def test_remote_failure_with_answer_is_done(self, fake_repo) -> None:
        o, main, launched = self._orch()
        child = self._failed_remote_child(o, main)
        child.state.messages.append({'role': 'assistant', 'content': 'A'})
        o.on_done(child)
        ledger.flush()
        assert [r['status'] for r in fake_repo.stream('tasks')] == \
            ['running', 'done']
        assert launched == []

    def test_cancelled_remote_failure_is_not_retried(self, fake_repo):
        o, main, launched = self._orch()
        child = self._failed_remote_child(o, main)
        child.state.cancel_requested = True
        o.on_done(child)
        ledger.flush()
        assert [r['status'] for r in fake_repo.stream('tasks')] == \
            ['running', 'cancelled']
        assert launched == []

    def test_no_local_rung_means_error(self, fake_repo) -> None:
        o, main, launched = self._orch(local_rung=False)
        main.state.adapter = o.registry.get('Remote')      # parent is remote
        child = self._failed_remote_child(o, main)
        o.on_done(child)
        ledger.flush()
        statuses = [r['status'] for r in fake_repo.stream('tasks')]
        # the local-only retry resolve is refused (no local rung, remote
        # parent), so the original closes as an error (written first) and
        # is reported; the refused retry attempt is recorded after it
        assert statuses == ['running', 'error', 'refused']
        assert launched == [] and main.queue
