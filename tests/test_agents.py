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
