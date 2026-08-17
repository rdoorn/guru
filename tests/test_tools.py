"""Tests for the tool registry, activation, and delegation tools."""
from guru import config, session, ui
from guru.domain import tools


class TestMatchTools:
    """Tests for tools._match_tools ranking."""

    def test_github_query_ranks_github_tool_first(self) -> None:
        ranked = tools._match_tools('get latest github release version')
        assert ranked[0] == 'fetch_github_releases'

    def test_fetch_query_ranks_web_fetch_first(self) -> None:
        assert tools._match_tools('fetch a webpage url')[0] == 'web_fetch'


class TestActiveSpecs:
    """Tests for tools.active_specs."""

    def test_search_tools_always_present(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'active_tool_names', set())
        specs = tools.active_specs()
        assert [s['name'] for s in specs] == ['search_tools', 'use_skill']

    def test_activated_tool_included(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'active_tool_names', {'web_fetch'})
        names = [s['name'] for s in tools.active_specs()]
        assert 'search_tools' in names and 'web_fetch' in names


class TestSpecsFor:
    """tools.specs_for builds tool specs without session routing."""

    def test_search_tools_always(self) -> None:
        assert [s['name'] for s in tools.specs_for(set(), False)] == [
            'search_tools', 'use_skill']

    def test_can_spawn_and_activated(self) -> None:
        names = {s['name'] for s in tools.specs_for({'web_fetch'}, True)}
        assert {'search_tools', 'spawn', 'check', 'join', 'web_fetch'} <= names


class TestToolSizeFormat:
    """Human byte/token formatting for the per-tool result size line."""

    def test_fmt_bytes(self) -> None:
        assert ui._fmt_bytes(80) == '80b'
        assert ui._fmt_bytes(1536) == '1.5k'
        assert ui._fmt_bytes(51200) == '50k'
        assert ui._fmt_bytes(5 * 1024 * 1024) == '5.0M'

    def test_fmt_size_includes_tokens(self) -> None:
        assert ui._fmt_size(40) == '40b · ~10 tok'
        assert '~2.0k tok' in ui._fmt_size(8000)   # 8000/4 = 2000 tok

    def test_execute_tool_reports_size_and_returns_result(
            self, monkeypatch) -> None:
        sizes = []
        monkeypatch.setattr(ui, 'note_tool_result', sizes.append)
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        tools.set_spawn_handler(lambda t, r, s: 'RESULT-9')
        try:
            out = tools.execute_tool('spawn', {'task': 't'})
        finally:
            tools.set_spawn_handler(None)
        assert out == 'RESULT-9'
        assert sizes == [len('RESULT-9')]


class TestSpawnTool:
    """Tests for the spawn delegation tool and its handler injection."""

    def test_spawn_without_handler_reports_repl(self) -> None:
        tools.set_spawn_handler(None)
        out = tools.spawn('do something')
        assert 'not available in this mode' in out

    def test_spawn_with_handler_delegates(self) -> None:
        seen: list = []
        tools.set_spawn_handler(
            lambda t, r, s: seen.append(t) or f'ok:{t}')
        try:
            assert tools.spawn('research topic') == 'ok:research topic'
            assert seen == ['research topic']
        finally:
            tools.set_spawn_handler(None)

    def test_execute_tool_routes_spawn(self) -> None:
        seen: list = []
        tools.set_spawn_handler(lambda t, r, s: seen.append(t) or 'done')
        try:
            assert tools.execute_tool('spawn', {'task': 'go'}) == 'done'
            assert seen == ['go']
        finally:
            tools.set_spawn_handler(None)

    def test_delegation_specs_gated_by_can_spawn(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'active_tool_names', set())
        monkeypatch.setattr(session, 'can_spawn', False)
        names = [s['name'] for s in tools.active_specs()]
        assert not ({'spawn', 'check', 'join'} & set(names))
        monkeypatch.setattr(session, 'can_spawn', True)
        names = [s['name'] for s in tools.active_specs()]
        assert {'spawn', 'check', 'join'} <= set(names)

    def test_reset_active_tools_honours_can_spawn(self) -> None:
        capable = session.SessionState()
        capable.can_spawn = True
        token = session.use(capable)
        try:
            tools.reset_active_tools()
            for fn in (tools.spawn, tools.check, tools.join,
                       tools.search_tools):
                assert fn in capable.active_tools
        finally:
            session.reset(token)
        plain = session.SessionState()
        token = session.use(plain)
        try:
            tools.reset_active_tools()
            assert tools.spawn not in plain.active_tools
            assert tools.check not in plain.active_tools
            assert tools.join not in plain.active_tools
        finally:
            session.reset(token)


class TestCollectTools:
    """Tests for the non-blocking check/join tools and their injection."""

    def test_check_without_handler_reports_repl(self) -> None:
        tools.set_check_handler(None)
        assert '--classic' in tools.check('all')

    def test_join_without_handler_reports_repl(self) -> None:
        tools.set_join_handler(None)
        assert '--classic' in tools.join('agent2')

    def test_check_and_join_delegate_and_route(self) -> None:
        seen: list = []
        tools.set_check_handler(lambda t: seen.append(('check', t)) or 'c')
        tools.set_join_handler(lambda t: seen.append(('join', t)) or 'j')
        try:
            assert tools.execute_tool('check', {'target': 'all'}) == 'c'
            assert tools.execute_tool('join', {'targets': 'a b'}) == 'j'
            assert seen == [('check', 'all'), ('join', 'a b')]
        finally:
            tools.set_check_handler(None)
            tools.set_join_handler(None)


class TestRetainPolicy:
    def test_web_tools_summarize(self) -> None:
        assert tools.retain_policy('web_search') == 'summarize'
        assert tools.retain_policy('web_fetch') == 'summarize'

    def test_read_file_outline(self) -> None:
        assert tools.retain_policy('read_file') == 'outline'

    def test_local_tools_keep(self) -> None:
        for name in ('search_code', 'list_dir', 'list_tree',
                     'write_file', 'edit_file', 'delete_file'):
            assert tools.retain_policy(name) == 'keep'

    def test_unknown_keeps(self) -> None:
        assert tools.retain_policy('nope') == 'keep'
        assert tools.retain_policy('') == 'keep'


class TestInitialTools:
    """The pre-activated core toolset lets weak models skip search_tools."""

    def test_initial_tools_includes_core(self, monkeypatch) -> None:
        monkeypatch.setattr(
            config, 'PREACTIVATE_TOOLS', ['read_file', 'search_code'])
        base, names = tools.initial_tools(can_spawn=False)
        assert names == {'read_file', 'search_code'}
        assert tools.search_tools in base and tools.use_skill in base
        assert tools.spawn not in base
        assert tools.TOOL_REGISTRY['read_file']['fn'] in base

    def test_initial_tools_spawn_and_empty_core(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'PREACTIVATE_TOOLS', [])
        base, names = tools.initial_tools(can_spawn=True)
        assert tools.spawn in base and names == set()

    def test_flat_activates_entire_registry(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'FLAT_TOOLS', True)
        monkeypatch.setattr(config, 'PREACTIVATE_TOOLS', [])
        base, names = tools.initial_tools(can_spawn=False)
        assert names == set(tools.TOOL_REGISTRY)     # every registry tool
        assert len(names) > len(['read_file', 'search_code'])
