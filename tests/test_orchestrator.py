"""Tests for the shared spawn/check/join mailbox (guru.orchestrator)."""
from guru import session
from guru.domain import conversation


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
