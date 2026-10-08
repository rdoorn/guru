"""Tests for the multi-viewport agent model (guru.agents)."""


class TestAgentManager:
    """Tests for the multi-viewport agent model."""

    def test_starts_with_main(self) -> None:
        from guru.agents import AgentManager
        m = AgentManager()
        assert m.active.title == 'main'
        assert m.tabs() == [(True, 'main')]

    def test_add_and_switch(self) -> None:
        from guru.agents import AgentManager
        m = AgentManager()
        m.add('research')
        assert [t for _, t in m.tabs()] == ['main', 'research']
        m.switch(1)
        assert m.active.title == 'research'
        m.switch(1)                     # wraps
        assert m.active.title == 'main'
        m.switch(-1)
        assert m.active.title == 'research'

    def test_append_and_text(self) -> None:
        from guru.agents import Agent
        a = Agent(id='x')
        a.append('one')
        a.append('two')
        assert a.text == 'one\ntwo'

    def test_agents_are_hashable_by_identity(self) -> None:
        from guru.agents import Agent
        a, b = Agent(id='x'), Agent(id='x')
        # Usable as dict keys (join barriers) and distinct despite same id.
        d = {a: 1, b: 2}
        assert len(d) == 2 and d[a] == 1 and a != b


def test_session_state_has_ledger_keys() -> None:
    from guru.session import SessionState
    st = SessionState()
    assert st.agent_id == 'main' and st.task_id == '' and st.turn_id == ''


class TestRetire:
    """Finished sub-agents leave the tab bar (archived, still known to
    the orchestrator); titles are never reused."""

    def _manager(self, n: int = 3):
        from guru.agents import Agent, AgentManager
        m = AgentManager()
        for _ in range(n):
            t = m.next_title()
            m.agents.append(Agent(id=t, title=t))
        return m

    def test_titles_count_up_and_survive_archiving(self) -> None:
        m = self._manager(2)
        assert [a.title for a in m.agents] == ['main', 'agent1', 'agent2']
        m.retire(m.agents[1], keep=False)
        assert m.next_title() == 'agent3'

    def test_done_child_is_archived_at_once(self) -> None:
        m = self._manager()
        a1 = m.agents[1]
        m.retire(a1, keep=False)
        assert a1 not in m.agents and a1 in m.archived
        assert a1 in m.all_agents()
        assert [t for _, t in m.tabs()] == ['main', 'agent2', 'agent3']

    def test_kept_child_stays_until_viewed_and_left(self) -> None:
        m = self._manager()
        a1 = m.agents[1]
        m.retire(a1, keep=True)
        assert a1 in m.agents
        m.select(1)                          # view it
        assert a1 in m.agents
        m.select(2)                          # leave it
        assert a1 in m.archived and m.active.title == 'agent2'

    def test_a_kept_child_never_viewed_stays(self) -> None:
        m = self._manager()
        a1 = m.agents[1]
        m.retire(a1, keep=True)
        m.select(2)
        m.select(3)
        assert a1 in m.agents

    def test_the_viewed_tab_is_not_pulled_away(self) -> None:
        m = self._manager()
        m.select(2)
        a2 = m.active
        m.retire(a2, keep=False)             # finished while on screen
        assert m.active is a2
        m.select(1)
        assert a2 in m.archived and m.active.title == 'agent1'

    def test_archiving_before_the_active_tab_keeps_it_active(self) -> None:
        m = self._manager()
        m.select(3)
        m.retire(m.agents[1], keep=False)
        assert m.active.title == 'agent3'

    def test_main_is_never_archived(self) -> None:
        m = self._manager()
        m.retire(m.agents[0], keep=False)
        assert m.agents[0].title == 'main'

    def test_busy_agent_is_not_archived_on_leave(self) -> None:
        m = self._manager()
        a1 = m.agents[1]
        m.select(1)
        m.retire(a1, keep=False)
        a1.busy = True                       # the user typed into it
        m.select(2)
        assert a1 in m.agents
