"""Tests for the provider adapters and the shared tool-calling turn loop."""
from types import SimpleNamespace

from guru import config, session, ui
from guru.adapters import anthropic as anth
from guru.adapters import litellm as lite
from guru.adapters.ollama import OllamaAdapter
from guru.domain import conversation, files


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


class TestActNudge:
    """Nudge weak models that announce an action but call no tool."""

    def test_looks_like_preamble(self) -> None:
        from guru.adapters.ollama import _looks_like_preamble as p
        assert p("Let me read the files.") is True
        assert p("I'll inspect the code next.") is True
        assert p("Here is my plan:") is True
        assert p("The code is clean and well tested.") is False
        assert p("Let me " + "x" * 700) is False   # too long -> real answer

    def test_run_turn_nudges_then_answers(self, monkeypatch) -> None:
        import ollama
        from guru.adapters.ollama import OllamaAdapter
        a = OllamaAdapter()
        msgs = [{'role': 'system', 'content': 's'},
                {'role': 'user', 'content': 'inspect the code'}]
        monkeypatch.setattr(session, 'messages', msgs)
        monkeypatch.setattr(session, 'cancel_requested', False)
        monkeypatch.setattr(a, '_fit_after_load', lambda: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        seq = [
            ollama.Message(role='assistant',
                           content='Let me read the files.', tool_calls=None),
            ollama.Message(role='assistant',
                           content='The code is well structured overall.',
                           tool_calls=None),
        ]
        calls = {'n': 0}

        def fake_collect():
            m = seq[calls['n']]
            calls['n'] += 1
            return m
        monkeypatch.setattr(a, '_collect_response', fake_collect)
        a.run_turn()
        assert calls['n'] == 2      # looped again after the nudge
        assert any(m.get('role') == 'user'
                   and 'do it now' in (m.get('content') or '').lower()
                   for m in session.messages)


class TestStreamingCancel:
    """The streamed turn accumulates chunks and aborts on cancel_requested."""

    def _adapter(self) -> OllamaAdapter:
        return OllamaAdapter()

    def _chunk(self, content='', tool_calls=None, pin=0, ein=0):
        return SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
            prompt_eval_count=pin, eval_count=ein)

    def _bind(self, monkeypatch, a) -> None:
        monkeypatch.setattr(a, '_supports_thinking', lambda m: False)
        monkeypatch.setattr(session, 'model', 'm')
        monkeypatch.setattr(session, 'messages', [])
        monkeypatch.setattr(session, 'active_tools', [])
        monkeypatch.setattr(session, 'session_in', 0)
        monkeypatch.setattr(session, 'session_out', 0)

    def test_accumulates_content_and_counts(self, monkeypatch) -> None:
        a = self._adapter()
        self._bind(monkeypatch, a)
        monkeypatch.setattr(session, 'cancel_requested', False)

        def fake(*args, **kw):
            yield self._chunk('Hel')
            yield self._chunk('lo', pin=12, ein=5)
        monkeypatch.setattr('guru.adapters.ollama.ollama.chat', fake)
        msg = a._collect_response()
        assert msg.content == 'Hello' and not msg.tool_calls
        assert session.session_in == 12 and session.session_out == 5

    def test_returns_none_and_closes_on_cancel(self, monkeypatch) -> None:
        a = self._adapter()
        self._bind(monkeypatch, a)
        monkeypatch.setattr(session, 'cancel_requested', True)
        closed = {'v': False}

        def fake(*args, **kw):
            try:
                while True:
                    yield self._chunk('x')
            finally:
                closed['v'] = True
        monkeypatch.setattr('guru.adapters.ollama.ollama.chat', fake)
        assert a._collect_response() is None
        assert closed['v'] is True        # stream closed -> generation aborted


class TestTurnLoop:
    """The shared, provider-agnostic tool-calling loop (guru.adapters.turn).

    Every adapter drives its turn through run_loop, so these lock the nudge,
    duplicate-suppression, and cancel behaviour that all providers now share.
    """

    def _quiet(self, monkeypatch) -> None:
        from guru.adapters import turn
        monkeypatch.setattr(ui, 'note_thinking', lambda: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        monkeypatch.setattr(turn, '_render_answer', lambda c: None)
        monkeypatch.setattr(session, 'messages', [])
        monkeypatch.setattr(session, 'cancel_requested', False)

    def test_nudges_stalled_then_answers(self, monkeypatch) -> None:
        from guru.adapters import turn
        self._quiet(monkeypatch)
        seq = iter([("Let me look into it.", []),
                    ("It is well tested.", [])])
        nudges: list = []
        turn.run_loop(step=lambda: next(seq),
                      run_tools=lambda p: None,
                      add_user=lambda t: nudges.append(t))
        assert len(nudges) == 1 and 'do it now' in nudges[0].lower()

    def test_runs_tools_then_answers(self, monkeypatch) -> None:
        from guru.adapters import turn
        self._quiet(monkeypatch)
        seq = iter([("", [("read_file", {"path": "x"}, "r1")]),
                    ("done", [])])
        ran: list = []
        turn.run_loop(step=lambda: next(seq),
                      run_tools=lambda p: ran.extend(p),
                      add_user=lambda t: None)
        assert ran == [("read_file", {"path": "x"}, "r1", False)]

    def test_marks_duplicate_calls(self, monkeypatch) -> None:
        from guru.adapters import turn
        self._quiet(monkeypatch)
        call = ("read_file", {"path": "x"}, "r")
        seq = iter([("", [call]), ("", [call]), ("done", [])])
        seen: list = []
        turn.run_loop(step=lambda: next(seq),
                      run_tools=lambda p: seen.append(p[0][3]),
                      add_user=lambda t: None)
        assert seen == [False, True]     # 2nd identical call flagged duplicate

    def test_stops_on_cancel_without_raising(self, monkeypatch) -> None:
        from guru.adapters import turn
        self._quiet(monkeypatch)

        def step():
            session.cancel_requested = True
            return None
        turn.run_loop(step=step, run_tools=lambda p: None,
                      add_user=lambda t: None)   # returns, no exception

    def _reads(self, n):
        return [{'role': 'tool', 'tool_name': 'read_file', 'content': 'x'}
                for _ in range(n)]

    def test_delegation_nudges_broad_task(self, monkeypatch) -> None:
        from guru.adapters import turn
        self._quiet(monkeypatch)
        monkeypatch.setattr(session, 'can_spawn', True)
        monkeypatch.setattr(config, 'DELEGATION_NUDGE_MIN_READS', 3)
        monkeypatch.setattr(session, 'messages', self._reads(3))
        seq = iter([("Here is my full assessment of the code.", []),
                    ("Consolidated report.", [])])
        nudges: list = []
        turn.run_loop(step=lambda: next(seq), run_tools=lambda p: None,
                      add_user=lambda t: nudges.append(t))
        assert len(nudges) == 1 and 'decompose' in nudges[0].lower()

    def test_no_delegation_nudge_for_subagent(self, monkeypatch) -> None:
        from guru.adapters import turn
        self._quiet(monkeypatch)
        monkeypatch.setattr(session, 'can_spawn', False)      # a sub-agent
        monkeypatch.setattr(config, 'DELEGATION_NUDGE_MIN_READS', 3)
        monkeypatch.setattr(session, 'messages', self._reads(3))
        seq = iter([("An answer.", [])])
        nudges: list = []
        turn.run_loop(step=lambda: next(seq), run_tools=lambda p: None,
                      add_user=lambda t: nudges.append(t))
        assert nudges == []

    def test_no_delegation_nudge_when_already_spawned(
            self, monkeypatch) -> None:
        from guru.adapters import turn
        self._quiet(monkeypatch)
        monkeypatch.setattr(session, 'can_spawn', True)
        monkeypatch.setattr(config, 'DELEGATION_NUDGE_MIN_READS', 3)
        msgs = self._reads(3) + [
            {'role': 'tool', 'tool_name': 'spawn', 'content': 'ok'}]
        monkeypatch.setattr(session, 'messages', msgs)
        seq = iter([("An answer after delegating.", [])])
        nudges: list = []
        turn.run_loop(step=lambda: next(seq), run_tools=lambda p: None,
                      add_user=lambda t: nudges.append(t))
        assert nudges == []


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


class TestSamplingOptions:
    """Sampling respects modelfile defaults; overrides come from settings."""

    def _adapter(self) -> OllamaAdapter:
        return OllamaAdapter()

    def test_empty_by_default(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'SAMPLING', {})
        monkeypatch.setattr(config, 'SAMPLING_PER_MODEL', {})
        assert self._adapter()._sampling_options('qwen3:14b') == {}

    def test_global_and_per_model_merge(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'SAMPLING', {'temperature': 0.7})
        monkeypatch.setattr(
            config, 'SAMPLING_PER_MODEL',
            {'qwen3:14b': {'temperature': 0.6, 'top_p': 0.95}})
        a = self._adapter()
        assert a._sampling_options('devstral')['temperature'] == 0.7
        opts = a._sampling_options('qwen3:14b')
        assert opts['temperature'] == 0.6 and opts['top_p'] == 0.95


class TestRunTurnIntegration:
    """End-to-end: a real adapter's run_turn drives the shared loop, runs a
    REAL tool via execute_tool, threads the result, and renders a final
    answer. Only the network round (_collect_response) is mocked."""

    def test_ollama_calls_tool_then_answers(
            self, tmp_path, monkeypatch) -> None:
        import ollama
        (tmp_path / 'hello.txt').write_text('hi', encoding='utf-8')
        # allow reading the temp dir; run non-interactively
        monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
        monkeypatch.setattr(files, 'set_path_asker', lambda fn: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        monkeypatch.setattr(ui, 'note_thinking', lambda: None)
        monkeypatch.setattr(session, 'model', 'm')
        monkeypatch.setattr(session, 'num_ctx', 4096)
        monkeypatch.setattr(session, 'cancel_requested', False)
        monkeypatch.setattr(session, 'active_tool_names', {'list_dir'})
        monkeypatch.setattr(session, 'messages', [
            {'role': 'user', 'content': 'list the directory'}])

        a = OllamaAdapter()
        monkeypatch.setattr(a, '_fit_after_load', lambda: None)
        # Two rounds: a tool call, then a final answer.
        seq = [
            ollama.Message(role='assistant', content='', tool_calls=[{
                'function': {'name': 'list_dir',
                             'arguments': {'path': str(tmp_path)}}}]),
            ollama.Message(role='assistant',
                           content='The directory has one file.',
                           tool_calls=None),
        ]
        it = iter(seq)
        monkeypatch.setattr(a, '_collect_response', lambda: next(it))

        a.run_turn()

        tool_msgs = [m for m in session.messages
                     if isinstance(m, dict) and m.get('role') == 'tool']
        assert tool_msgs and tool_msgs[0]['tool_name'] == 'list_dir'
        assert 'hello.txt' in tool_msgs[0]['content']
        finals = [m for m in session.messages
                  if conversation.msg_role(m) == 'assistant'
                  and 'one file' in conversation.msg_content(m)]
        assert finals            # a real final answer was rendered
