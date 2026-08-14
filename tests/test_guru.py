"""Unit tests for guru's pure helper functions across the package."""
import json
from pathlib import Path

from guru import cli, config, session, ui
from guru.adapters import anthropic as anth
from guru.adapters import litellm as lite
from guru.adapters.ollama import OllamaAdapter
from guru.domain import conversation, files, tools


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


class TestHumanCtx:
    """Tests for cli._human_ctx formatting."""

    def test_kilo(self) -> None:
        assert cli._human_ctx(65536) == '64k'
        assert cli._human_ctx(4096) == '4k'

    def test_mega(self) -> None:
        assert cli._human_ctx(1024 * 1024) == '1M'


class TestDomainOf:
    """Tests for config.domain_of."""

    def test_strips_scheme_and_lowercases(self) -> None:
        assert config.domain_of('https://Example.COM/path') == 'example.com'

    def test_strips_port(self) -> None:
        assert config.domain_of('http://example.com:8443/x') == 'example.com'

    def test_bare_host_without_scheme(self) -> None:
        assert config.domain_of('example.com/path') == 'example.com'

    def test_subdomain_preserved(self) -> None:
        assert config.domain_of('https://api.github.com') == 'api.github.com'


class TestProjectMemoryDir:
    """The project memory dir lives under ./.guru/memory."""

    def test_points_into_project_guru_dir(self) -> None:
        assert config.PROJECT_MEMORY_DIR == Path.cwd() / '.guru' / 'memory'


class TestBuildSystemPrompt:
    """Tests for config.build_system_prompt assembly."""

    def test_appends_global_and_local(self, tmp_path, monkeypatch) -> None:
        global_md = tmp_path / 'GURU.md'
        local_md = tmp_path / '.GURU.md'
        global_md.write_text('GLOBAL RULES', encoding='utf-8')
        local_md.write_text('LOCAL RULES', encoding='utf-8')
        monkeypatch.setattr(config, 'GURU_MD_PATH', global_md)
        monkeypatch.setattr(config, 'PROJECT_GURU_MD', local_md)

        prompt = config.build_system_prompt()

        assert config.SYSTEM_PROMPT.strip() in prompt
        assert 'GLOBAL RULES' in prompt
        assert 'LOCAL RULES' in prompt
        assert prompt.index('GLOBAL RULES') < prompt.index('LOCAL RULES')

    def test_missing_files_are_skipped(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'GURU_MD_PATH', tmp_path / 'nope.md')
        monkeypatch.setattr(config, 'PROJECT_GURU_MD', tmp_path / 'nope2.md')
        assert config.build_system_prompt() == config.SYSTEM_PROMPT.strip()


class TestMessageToDict:
    """Tests for conversation.message_to_dict normalisation."""

    def test_plain_dict_user_message(self) -> None:
        msg = {'role': 'user', 'content': 'hello'}
        assert conversation.message_to_dict(msg) == {
            'role': 'user', 'content': 'hello'}

    def test_tool_message_keeps_tool_name(self) -> None:
        msg = {'role': 'tool', 'tool_name': 'web_fetch', 'content': 'x'}
        assert conversation.message_to_dict(msg)['tool_name'] == 'web_fetch'

    def test_none_content_becomes_empty_string(self) -> None:
        msg = {'role': 'assistant', 'content': None}
        assert conversation.message_to_dict(msg)['content'] == ''

    def test_pydantic_like_object_via_model_dump(self) -> None:
        class FakeMessage:
            def model_dump(self) -> dict:
                return {'role': 'assistant', 'content': 'hi'}

        assert conversation.message_to_dict(FakeMessage()) == {
            'role': 'assistant', 'content': 'hi'}


class TestFirstUserMessage:
    """Tests for conversation._first_user_message."""

    def test_returns_first_user_content(self, tmp_path) -> None:
        path = tmp_path / 'c.memory'
        path.write_text(json.dumps([
            {'role': 'assistant', 'content': 'ignored'},
            {'role': 'user', 'content': 'the question'},
        ]), encoding='utf-8')
        assert conversation._first_user_message(path) == 'the question'

    def test_truncates_long_message(self, tmp_path) -> None:
        path = tmp_path / 'c.memory'
        path.write_text(
            json.dumps([{'role': 'user', 'content': 'x' * 100}]),
            encoding='utf-8')
        result = conversation._first_user_message(path)
        assert result.endswith('…')
        assert len(result) == 61

    def test_unreadable_file(self, tmp_path) -> None:
        assert conversation._first_user_message(
            tmp_path / 'missing.memory') == '(unreadable)'


class TestMatchTools:
    """Tests for tools._match_tools ranking."""

    def test_github_query_ranks_github_tool_first(self) -> None:
        ranked = tools._match_tools('get latest github release version')
        assert ranked[0] == 'fetch_github_releases'

    def test_fetch_query_ranks_web_fetch_first(self) -> None:
        assert tools._match_tools('fetch a webpage url')[0] == 'web_fetch'


class _FakeInfo:
    def __init__(self, modelinfo: dict, parameters: str) -> None:
        self.modelinfo = modelinfo
        self.parameters = parameters


class TestResolveContextWindow:
    """Tests for OllamaAdapter._resolve_context_window."""

    def _adapter(self) -> OllamaAdapter:
        return OllamaAdapter()

    def test_uses_modelfile_num_ctx_capped_at_ceiling(
            self, monkeypatch) -> None:
        info = _FakeInfo(
            {'general.architecture': 'qwen3', 'qwen3.context_length': 40960},
            'num_ctx 32768\ntemperature 0.6')
        monkeypatch.setattr(
            'guru.adapters.ollama.ollama.show', lambda m: info)
        monkeypatch.setattr(session, 'num_ctx_override', 0)
        assert self._adapter()._resolve_context_window('m') == (32768, 40960)

    def test_defaults_when_modelfile_has_no_num_ctx(self, monkeypatch) -> None:
        info = _FakeInfo(
            {'general.architecture': 'llama', 'llama.context_length': 8192},
            'temperature 0.7')
        monkeypatch.setattr(
            'guru.adapters.ollama.ollama.show', lambda m: info)
        monkeypatch.setattr(session, 'num_ctx_override', 0)
        num_ctx, ceiling = self._adapter()._resolve_context_window('m')
        assert num_ctx == config.DEFAULT_NUM_CTX
        assert ceiling == 8192

    def test_cli_override_wins_but_is_capped(self, monkeypatch) -> None:
        info = _FakeInfo(
            {'general.architecture': 'qwen3', 'qwen3.context_length': 40960},
            'num_ctx 8192')
        monkeypatch.setattr(
            'guru.adapters.ollama.ollama.show', lambda m: info)
        monkeypatch.setattr(session, 'num_ctx_override', 100000)
        num_ctx, _ = self._adapter()._resolve_context_window('m')
        assert num_ctx == 40960

    def test_show_failure_falls_back(self, monkeypatch) -> None:
        def _boom(m: str) -> None:
            raise RuntimeError('no server')

        monkeypatch.setattr('guru.adapters.ollama.ollama.show', _boom)
        monkeypatch.setattr(session, 'num_ctx_override', 0)
        num_ctx, ceiling = self._adapter()._resolve_context_window('m')
        assert num_ctx == config.DEFAULT_NUM_CTX
        assert ceiling == 0


class TestGroupMessages:
    """Tests for conversation.group_messages turn-grouping."""

    def test_splits_at_each_user_message(self) -> None:
        msgs = [
            {'role': 'user', 'content': 'q1'},
            {'role': 'assistant', 'content': 'a1'},
            {'role': 'tool', 'content': 't1'},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 'a2'},
        ]
        groups = conversation.group_messages(msgs)
        assert len(groups) == 2
        assert len(groups[0]) == 3
        assert len(groups[1]) == 2


class TestEstimateTokens:
    """Tests for conversation.estimate_tokens."""

    def test_four_chars_per_token(self) -> None:
        assert conversation.estimate_tokens(
            [{'role': 'user', 'content': 'x' * 40}]) == 10


class TestCompactMessages:
    """Tests for conversation.compact_messages eviction path."""

    def test_evicts_old_tool_output_without_summary(
            self, monkeypatch) -> None:
        class _Adapter:
            def summarise(self, transcript: str) -> str:
                raise AssertionError('summary should not be called')

        monkeypatch.setattr(session, 'adapter', _Adapter())
        monkeypatch.setattr(config, 'KEEP_RECENT_GROUPS', 1)
        monkeypatch.setattr(session, 'num_ctx', 100)
        monkeypatch.setattr(session, 'messages', [
            {'role': 'system', 'content': 'SYS'},
            {'role': 'user', 'content': 'q1'},
            {'role': 'assistant', 'content': 'a1'},
            {'role': 'tool', 'tool_name': 'web_fetch', 'content': 'X' * 2000},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 'a2'},
        ])
        conversation.compact_messages(force=False)

        result = session.messages
        assert result[0]['content'] == 'SYS'
        tool_msg = next(m for m in result if m.get('role') == 'tool')
        assert 'evicted' in tool_msg['content']
        assert result[-1]['content'] == 'a2'


class TestStatusParts:
    """Tests for ui._status_parts colour thresholds and segments."""

    def test_green_yellow_red(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'num_ctx', 1000)
        monkeypatch.setattr(session, 'model', 'demo:latest')

        monkeypatch.setattr(session, 'ctx_used', 100)
        assert ui._status_parts()[3] == 'green'

        monkeypatch.setattr(session, 'ctx_used', 780)
        left, ctx_segment, right, colour = ui._status_parts()
        assert colour == 'yellow'
        assert '78%' in ctx_segment

        monkeypatch.setattr(session, 'ctx_used', 900)
        assert ui._status_parts()[3] == 'red'

    def test_segments_include_expected_fields(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'num_ctx', 1000)
        monkeypatch.setattr(session, 'ctx_used', 100)
        monkeypatch.setattr(session, 'model', 'demo:latest')
        monkeypatch.setattr(session, 'model_size', '32B')
        monkeypatch.setattr(session, 'session_in', 1234)
        monkeypatch.setattr(session, 'session_out', 56)
        monkeypatch.setattr(session, 'git_branch', 'main')

        left, ctx_segment, right, _ = ui._status_parts()
        assert 'demo' in left and '32B' in left
        assert '🧠' in ctx_segment
        assert '1234' in right and '56' in right and 'main' in right


class TestFormatBytes:
    """Tests for ui.format_bytes."""

    def test_gigabytes(self) -> None:
        assert ui.format_bytes(8_200_000_000) == '7.6 GB'

    def test_megabytes(self) -> None:
        assert ui.format_bytes(5_000_000) == '4.8 MB'


class TestActiveSpecs:
    """Tests for tools.active_specs."""

    def test_search_tools_always_present(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'active_tool_names', set())
        specs = tools.active_specs()
        assert [s['name'] for s in specs] == ['search_tools']

    def test_activated_tool_included(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'active_tool_names', {'web_fetch'})
        names = [s['name'] for s in tools.active_specs()]
        assert 'search_tools' in names and 'web_fetch' in names


class TestFileTools:
    """Tests for the filesystem tools and the directory allow-list gate."""

    def _only(self, monkeypatch, *dirs) -> None:
        """Make ALLOWED_DIRS contain exactly ``dirs`` for this test."""
        monkeypatch.setattr(
            config, 'ALLOWED_DIRS',
            {str(Path(d).resolve()) for d in dirs})

    def test_list_dir_shows_perms_and_size(self, monkeypatch) -> None:
        self._only(monkeypatch, Path.cwd())
        out = files.list_dir('guru')
        assert 'session.py' in out and '0644' in out

    def test_read_file_range_is_line_numbered(self, monkeypatch) -> None:
        self._only(monkeypatch, Path.cwd())
        out = files.read_file('guru/session.py', '1-3')
        assert 'lines 1-3 of' in out
        assert '\n     1\t' in out

    def test_read_file_bad_range(self, monkeypatch) -> None:
        self._only(monkeypatch, Path.cwd())
        assert 'Invalid line range' in files.read_file('guru/session.py', 'x')

    def test_read_file_caps_large_files(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        big = tmp_path / 'big.txt'
        big.write_text('\n'.join(str(i) for i in range(1, 501)) + '\n')
        out = files.read_file(str(big))
        assert 'lines 1-400 of 500' in out
        assert 'showing first 400 of 500' in out

    def test_read_file_refuses_binary(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        b = tmp_path / 'b.bin'
        b.write_bytes(b'\x00\x01\x02data')
        assert 'binary' in files.read_file(str(b))

    def test_list_tree_skips_noise_dirs(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / '.git').mkdir()
        (tmp_path / '.git' / 'INSIDE_GIT').write_text('y')
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'a.py').write_text('z')
        out = files.list_tree(str(tmp_path), '3')
        assert '(skipped)' in out
        assert 'INSIDE_GIT' not in out   # noise dir not descended
        assert 'a.py' in out             # normal dir descended

    def test_cwd_is_not_auto_allowed(self, monkeypatch) -> None:
        # A fresh project: nothing pre-allowed, so even cwd must be approved.
        self._only(monkeypatch)                       # empty allow-list
        files.set_path_asker(lambda d: False)
        try:
            assert 'denied' in files.list_dir('.').lower()
        finally:
            files.set_path_asker(None)

    def test_gate_denies_outside(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch)
        files.set_path_asker(lambda d: False)
        try:
            assert 'denied' in files.list_dir(str(tmp_path)).lower()
        finally:
            files.set_path_asker(None)

    def test_gate_approves_and_persists(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch)
        saved: list = []
        monkeypatch.setattr(config, 'persist_dir', saved.append)
        files.set_path_asker(lambda d: True)
        (tmp_path / 'f.txt').write_text('hello\n')
        try:
            out = files.list_dir(str(tmp_path))
            resolved = str(tmp_path.resolve())
            assert 'f.txt' in out
            assert resolved in config.ALLOWED_DIRS
            assert saved == [resolved]
        finally:
            files.set_path_asker(None)

    def test_parse_range(self) -> None:
        assert files._parse_range('', 500) == (1, 400)
        assert files._parse_range('10-20', 500) == (10, 20)
        assert files._parse_range('bad', 10) == (None, None)
        assert files._parse_range('5-3', 10) == (None, None)
        assert files._parse_range('1-9999', 50) == (1, 50)


class TestAccessPromptDefaults:
    """Access prompts default to allow on Enter but deny on any error."""

    def test_path_enter_allows(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: '')
        assert files._ask_path('/x') is True

    def test_path_explicit_no_denies(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: 'no')
        assert files._ask_path('/x') is False

    def test_path_error_denies(self, monkeypatch) -> None:
        def boom(*a):
            raise EOFError
        monkeypatch.setattr('builtins.input', boom)
        assert files._ask_path('/x') is False

    def test_domain_enter_allows(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: '  ')
        assert tools._ask_domain('x.com') is True

    def test_domain_no_denies(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: 'n')
        assert tools._ask_domain('x.com') is False

    def test_domain_error_denies(self, monkeypatch) -> None:
        def boom(*a):
            raise KeyboardInterrupt
        monkeypatch.setattr('builtins.input', boom)
        assert tools._ask_domain('x.com') is False


class TestOptionalToolParams:
    """Optional params are excluded from adapter 'required' schemas."""

    def test_anthropic_marks_optional_not_required(self) -> None:
        spec = {'name': 'read_file', 'description': 'd',
                'parameters': {'path': 'p', 'lines': 'l'},
                'optional': ['lines']}
        defn = anth.tool_defs([spec])[0]
        req = defn['input_schema']['required']
        assert req == ['path']

    def test_litellm_marks_optional_not_required(self) -> None:
        spec = {'name': 'read_file', 'description': 'd',
                'parameters': {'path': 'p', 'lines': 'l'},
                'optional': ['lines']}
        defn = lite.openai_tool_defs([spec])[0]
        req = defn['function']['parameters']['required']
        assert req == ['path']


class TestSpawnTool:
    """Tests for the spawn delegation tool and its handler injection."""

    def test_spawn_without_handler_reports_repl(self) -> None:
        tools.set_spawn_handler(None)
        out = tools.spawn('do something')
        assert '--classic' in out

    def test_spawn_with_handler_delegates(self) -> None:
        seen: list = []
        tools.set_spawn_handler(lambda t: seen.append(t) or f'ok:{t}')
        try:
            assert tools.spawn('research topic') == 'ok:research topic'
            assert seen == ['research topic']
        finally:
            tools.set_spawn_handler(None)

    def test_execute_tool_routes_spawn(self) -> None:
        seen: list = []
        tools.set_spawn_handler(lambda t: seen.append(t) or 'done')
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


class TestSessionRouting:
    """Tests for the per-context SessionState routing (parallel isolation)."""

    def test_use_and_reset(self) -> None:
        st = session.SessionState()
        token = session.use(st)
        session.model = 'bound-model'
        session.session_in += 7
        assert st.model == 'bound-model'
        assert st.session_in == 7
        session.reset(token)
        assert session.model != 'bound-model'

    def test_threads_are_isolated(self) -> None:
        import threading

        states = [session.SessionState() for _ in range(3)]
        errors: list = []
        start = threading.Barrier(len(states))

        def worker(idx: int) -> None:
            token = session.use(states[idx])
            try:
                start.wait()
                for _ in range(1000):
                    session.session_in += 1
                    session.model = f'model-{idx}'
                if (session.session_in != 1000
                        or session.model != f'model-{idx}'):
                    errors.append(idx)
            finally:
                session.reset(token)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(len(states))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert [s.session_in for s in states] == [1000, 1000, 1000]
        assert [s.model for s in states] == ['model-0', 'model-1', 'model-2']


class TestAdapterConfigRoundTrip:
    """Tests for config.save_adapter_configs / load_adapter_configs."""

    def test_round_trip_preserves_enable_and_fields(
            self, tmp_path, monkeypatch) -> None:
        path = tmp_path / 'adapters.toml'
        monkeypatch.setattr(config, 'ADAPTERS_PATH', path)
        monkeypatch.setattr(config, 'GURU_HOME', tmp_path)
        configs = [
            {'name': 'Ollama', 'type': 'ollama',
             'url': 'http://localhost:11434', 'enable': True},
            {'name': 'Anthropic Enterprise', 'type': 'anthropic',
             'auth': 'oauth', 'profile': 'guru', 'enable': False,
             'thinking': True},
        ]
        config.save_adapter_configs(configs)
        loaded = config.load_adapter_configs()
        assert loaded[0]['name'] == 'Ollama'
        assert loaded[0]['enable'] is True
        assert loaded[1]['enable'] is False
        assert loaded[1]['auth'] == 'oauth'
        assert loaded[1]['profile'] == 'guru'
        assert loaded[1]['thinking'] is True


class TestSettings:
    """Tests for config.load_settings / save_settings."""

    def test_round_trip(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / '.guru' / 'settings.json'
        monkeypatch.setattr(config, 'PROJECT_SETTINGS_PATH', path)
        assert config.load_settings() == {}
        config.save_settings({'adapter': 'Claude Code', 'model': 'x'})
        loaded = config.load_settings()
        assert loaded['adapter'] == 'Claude Code'
        assert loaded['model'] == 'x'


class TestAnthropicTranslation:
    """Tests for the pure Anthropic translation helpers."""

    def test_system_merged_and_history_flattened(self) -> None:
        messages = [
            {'role': 'system', 'content': 'BASE'},
            {'role': 'system', 'content': 'SUMMARY'},
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': '',
             'tool_calls': [{'function': {'name': 'web_search',
                                          'arguments': {'query': 'x'}}}]},
            {'role': 'tool', 'tool_name': 'web_search', 'content': 'results'},
        ]
        system, native = anth.to_anthropic_messages(messages)
        assert system == 'BASE\n\nSUMMARY'
        assert native[0] == {'role': 'user', 'content': 'hello'}
        assert native[1] == {'role': 'assistant', 'content': '(used tools)'}
        assert native[2]['role'] == 'user'
        assert 'web_search result' in native[2]['content']

    def test_tool_defs_schema(self) -> None:
        specs = [{
            'name': 'web_fetch',
            'description': 'fetch a page',
            'parameters': {'url': 'the url'},
        }]
        defs = anth.tool_defs(specs)
        assert defs[0]['name'] == 'web_fetch'
        schema = defs[0]['input_schema']
        assert schema['properties']['url']['type'] == 'string'
        assert schema['required'] == ['url']

    def test_neutral_assistant_with_and_without_tools(self) -> None:
        plain = anth.neutral_assistant('hi', [])
        assert plain == {'role': 'assistant', 'content': 'hi'}
        with_tools = anth.neutral_assistant(
            '', [('web_search', {'query': 'x'})])
        assert with_tools['tool_calls'][0]['function']['name'] == 'web_search'


class TestLiteLLMTranslation:
    """Tests for the pure LiteLLM/OpenAI translation helpers."""

    def test_messages_flattened(self) -> None:
        messages = [
            {'role': 'system', 'content': 'SYS'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': '',
             'tool_calls': [{'function': {'name': 'web_search',
                                          'arguments': {'query': 'x'}}}]},
            {'role': 'tool', 'tool_name': 'web_search', 'content': 'results'},
        ]
        out = lite.to_openai_messages(messages)
        assert out[0] == {'role': 'system', 'content': 'SYS'}
        assert out[1] == {'role': 'user', 'content': 'hi'}
        assert out[2] == {'role': 'assistant', 'content': '(used tools)'}
        assert out[3]['role'] == 'user'
        assert 'web_search result' in out[3]['content']

    def test_tool_defs_openai_shape(self) -> None:
        specs = [{
            'name': 'web_fetch',
            'description': 'fetch a page',
            'parameters': {'url': 'the url'},
        }]
        defs = lite.openai_tool_defs(specs)
        assert defs[0]['type'] == 'function'
        fn = defs[0]['function']
        assert fn['name'] == 'web_fetch'
        assert fn['parameters']['properties']['url']['type'] == 'string'
        assert fn['parameters']['required'] == ['url']

    def test_base_url_trailing_slash_stripped(self) -> None:
        a = lite.LiteLLMAdapter(base_url='https://proxy/v1/')
        assert a.base_url == 'https://proxy/v1'

    def test_inline_api_key_used_when_env_unset(self, monkeypatch) -> None:
        monkeypatch.delenv('MY_LLM_KEY', raising=False)
        a = lite.LiteLLMAdapter(
            base_url='https://p/v1', api_key_env='MY_LLM_KEY',
            api_key='sk-123')
        assert a._key() == 'sk-123'

    def test_env_key_overrides_inline(self, monkeypatch) -> None:
        monkeypatch.setenv('MY_LLM_KEY', 'sk-env')
        a = lite.LiteLLMAdapter(
            base_url='https://p/v1', api_key_env='MY_LLM_KEY',
            api_key='sk-123')
        assert a._key() == 'sk-env'
