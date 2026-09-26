"""Tests for the provider adapters and the shared tool-calling turn loop."""
from types import SimpleNamespace

import pytest

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

    Every adapter drives its turn through run_loop, so these lock the
    delegation nudge, duplicate-suppression, and cancel behaviour that all
    providers share (the turn contract has its own tests in
    tests/test_turn_contract.py).
    """

    def _quiet(self, monkeypatch) -> None:
        from guru.adapters import turn
        monkeypatch.setattr(ui, 'note_thinking', lambda: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        monkeypatch.setattr(turn, '_render_answer', lambda c: None)
        monkeypatch.setattr(session, 'messages', [])
        monkeypatch.setattr(session, 'cancel_requested', False)

    def test_text_reply_is_the_answer_without_forcing(self, monkeypatch):
        """No adapter that forces tool calls is bound: a text reply is the
        answer at once (no act nudge, no re-prompt)."""
        from guru.adapters import turn
        self._quiet(monkeypatch)
        monkeypatch.setattr(session, 'adapter', None)
        seq = iter([("Let me look into it.", [])])
        nudges: list = []
        turn.run_loop(step=lambda: next(seq),
                      run_tools=lambda p: None,
                      add_user=lambda t: nudges.append(t))
        assert nudges == []
        assert session.messages[-1] == {
            'role': 'assistant', 'content': 'Let me look into it.'}

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

    def _reads(self, n, paths=None, request='review the whole service'):
        """A user request followed by ``n`` read_file tool messages; each
        carries its ``path`` argument (distinct by default)."""
        paths = paths or [f'app/mod{i}.py' for i in range(n)]
        return [{'role': 'user', 'content': request}] + [
            {'role': 'tool', 'tool_name': 'read_file', 'content': 'x',
             'tool_args': {'path': paths[i % len(paths)]}}
            for i in range(n)]

    def _nudges(self, monkeypatch, messages, controller=False) -> list:
        from guru.adapters import turn
        self._quiet(monkeypatch)
        monkeypatch.setattr(session, 'can_spawn', True)
        monkeypatch.setattr(session, 'controller', controller)
        monkeypatch.setattr(config, 'DELEGATION_NUDGE_MIN_READS', 3)
        monkeypatch.setattr(session, 'messages', messages)
        seq = iter([("Here is my full assessment of the code.", []),
                    ("Consolidated report.", [])])
        nudges: list = []
        turn.run_loop(step=lambda: next(seq), run_tools=lambda p: None,
                      add_user=lambda t: nudges.append(t))
        return nudges

    def test_delegation_nudges_broad_task(self, monkeypatch) -> None:
        nudges = self._nudges(monkeypatch, self._reads(3))
        assert len(nudges) == 1 and 'decompose' in nudges[0].lower()

    def test_reads_of_the_same_file_count_once(self, monkeypatch) -> None:
        # Three reads, one distinct path: not a broad task.
        msgs = self._reads(3, paths=['app/one.py'])
        assert self._nudges(monkeypatch, msgs) == []
        # Two distinct paths read five times: still under the threshold.
        msgs = self._reads(5, paths=['a.py', 'b.py'])
        assert self._nudges(monkeypatch, msgs) == []

    def test_reads_without_args_do_not_count(self, monkeypatch) -> None:
        msgs = [{'role': 'user', 'content': 'review the service'}] + [
            {'role': 'tool', 'tool_name': 'read_file', 'content': 'x'}
            for _ in range(4)]
        assert self._nudges(monkeypatch, msgs) == []

    def test_search_code_paths_count_as_reads(self, monkeypatch) -> None:
        msgs = [{'role': 'user', 'content': 'review the service'}] + [
            {'role': 'tool', 'tool_name': 'search_code', 'content': 'x',
             'tool_args': {'pattern': 'p', 'path': d}}
            for d in ('app', 'tests', 'docs')]
        assert len(self._nudges(monkeypatch, msgs)) == 1

    def test_single_target_edit_request_is_not_nudged(
            self, monkeypatch) -> None:
        msgs = self._reads(
            4, request='Fix the failing test; the bug is in wordcount.py')
        assert self._nudges(monkeypatch, msgs) == []

    def test_edit_request_over_several_files_is_nudged(
            self, monkeypatch) -> None:
        msgs = self._reads(
            4, request='update app.py, models.py and views.py for the API')
        assert len(self._nudges(monkeypatch, msgs)) == 1

    def test_controller_is_never_nudged(self, monkeypatch) -> None:
        assert self._nudges(monkeypatch, self._reads(4),
                            controller=True) == []

    @pytest.mark.parametrize('request_text, single', [
        ('fix the failing test in wordcount.py', True),
        ('Rename count_words to word_count', True),
        ('please patch setup.toml', True),
        ('Update README.md.', True),
        ('change a.py and b.py to use the new API', False),
        ('review this repository for security issues', False),
        ('explain how the rollback procedure works', False),
        ('', False),
    ])
    def test_single_target_request(self, request_text, single) -> None:
        from guru.adapters import turn
        assert turn._single_target_request(request_text) is single

    def test_waiting_flag_ends_turn_after_tool_round(
            self, monkeypatch) -> None:
        """A join that opened a barrier sets ``session.turn_waiting`` from
        inside run_tools; the loop then ends the turn without another
        model round and renders no answer."""
        from guru.adapters import turn
        self._quiet(monkeypatch)
        rendered: list = []
        monkeypatch.setattr(turn, '_render_answer',
                            lambda c: rendered.append(c))
        printed: list = []
        monkeypatch.setattr(turn.ui.console, 'print',
                            lambda *a, **k: printed.append(str(a[0])))
        steps = {'n': 0}

        def step():
            steps['n'] += 1
            return ("", [("join", {"targets": "agent1"}, "r1")])

        def run_tools(pending):
            session.turn_waiting = True

        turn.run_loop(step=step, run_tools=run_tools,
                      add_user=lambda t: None)
        assert steps['n'] == 1                  # no second round
        assert rendered == []
        assert any('waiting for sub-agents' in p for p in printed)

    def test_waiting_flag_reset_at_turn_start(self, monkeypatch) -> None:
        from guru.adapters import turn
        self._quiet(monkeypatch)
        monkeypatch.setattr(session, 'turn_waiting', True)
        monkeypatch.setattr(session, 'check_polls', 2)
        seq = iter([("", [("read_file", {"path": "x"}, "r1")]),
                    ("done", [])])
        turn.run_loop(step=lambda: next(seq), run_tools=lambda p: None,
                      add_user=lambda t: None)
        assert session.turn_waiting is False and session.check_polls == 0

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
        assert tool_msgs[0]['tool_args'] == {'path': str(tmp_path)}
        assert 'hello.txt' in tool_msgs[0]['content']
        finals = [m for m in session.messages
                  if conversation.msg_role(m) == 'assistant'
                  and 'one file' in conversation.msg_content(m)]
        assert finals            # a real final answer was rendered


class TestCallRecords:
    """Every provider call emits exactly one ledger CallRecord."""

    def _arm(self, monkeypatch, fake_repo):
        from guru.adapters import turn
        monkeypatch.setattr(ui, 'note_thinking', lambda: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        monkeypatch.setattr(turn, '_render_answer', lambda c: None)
        monkeypatch.setattr(session, 'model', 'm')
        monkeypatch.setattr(session, 'num_ctx', 4096)
        monkeypatch.setattr(session, 'cancel_requested', False)
        monkeypatch.setattr(session, 'session_in', 0)
        monkeypatch.setattr(session, 'session_out', 0)
        monkeypatch.setattr(session, 'messages', [
            {'role': 'user', 'content': 'q'}])
        return fake_repo

    def _calls(self, repo):
        from guru.domain import ledger
        ledger.flush()
        return repo.stream('calls')

    # --- anthropic -----------------------------------------------------------

    def _anthropic(self, monkeypatch, resp):
        a = anth.AnthropicAdapter(thinking=False)
        client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kw: resp))
        monkeypatch.setattr(a, '_client', lambda: client)
        return a

    def _anthropic_resp(self, text='hi', **usage):
        return SimpleNamespace(
            usage=SimpleNamespace(**usage), stop_reason='end_turn',
            content=[SimpleNamespace(type='text', text=text)])

    def test_anthropic_step_records_cache_tokens(
            self, monkeypatch, fake_repo) -> None:
        repo = self._arm(monkeypatch, fake_repo)
        a = self._anthropic(monkeypatch, self._anthropic_resp(
            input_tokens=100, output_tokens=20, cache_read_input_tokens=30,
            cache_creation_input_tokens=10))
        a.run_turn()
        [row] = self._calls(repo)
        assert row['adapter'] == 'Anthropic' and row['tokens_in'] == 100
        assert row['tokens_out'] == 20 and row['phase'] == 'step'
        assert row['cache_read'] == 30 and row['cache_write'] == 10
        assert row['cost_source'] in ('table', 'unknown')
        assert row['seconds'] >= 0 and row['cost_source'] != 'local'
        assert row['model'] == 'm'

    def test_anthropic_summarise_records_phase(
            self, monkeypatch, fake_repo) -> None:
        repo = self._arm(monkeypatch, fake_repo)
        a = self._anthropic(monkeypatch, self._anthropic_resp(
            text='sum', input_tokens=8, output_tokens=2))
        assert a.summarise('long transcript') == 'sum'
        [row] = self._calls(repo)
        assert row['phase'] == 'summarise' and row['tokens_in'] == 8

    # --- ollama --------------------------------------------------------------

    def _ollama_chunk(self, content='', pin=0, ein=0, **durations):
        return SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None),
            prompt_eval_count=pin, eval_count=ein, **durations)

    def test_ollama_step_is_local_and_has_timing(
            self, monkeypatch, fake_repo) -> None:
        repo = self._arm(monkeypatch, fake_repo)
        a = OllamaAdapter()
        monkeypatch.setattr(a, '_supports_thinking', lambda m: False)
        monkeypatch.setattr(session, 'active_tools', [])

        def fake(*args, **kw):
            yield self._ollama_chunk('Hel')
            yield self._ollama_chunk('lo', pin=50, ein=10, load_duration=1e9,
                                     prompt_eval_duration=2e8,
                                     eval_duration=5e8)
        monkeypatch.setattr('guru.adapters.ollama.ollama.chat', fake)
        msg = a._collect_response()
        assert msg.content == 'Hello'
        [row] = self._calls(repo)
        assert row['adapter'] == 'Ollama' and row['phase'] == 'step'
        assert row['tokens_in'] == 50 and row['tokens_out'] == 10
        assert row['cost_usd'] == 0.0 and row['cost_source'] == 'local'
        assert row['load_s'] == 1.0 and row['prefill_s'] == 0.2
        assert row['generate_s'] == 0.5

    def test_ollama_cancelled_stream_records_nothing(
            self, monkeypatch, fake_repo):
        repo = self._arm(monkeypatch, fake_repo)
        a = OllamaAdapter()
        monkeypatch.setattr(a, '_supports_thinking', lambda m: False)
        monkeypatch.setattr(session, 'active_tools', [])
        monkeypatch.setattr(session, 'cancel_requested', True)

        def fake(*args, **kw):
            while True:
                yield self._ollama_chunk('x')
        monkeypatch.setattr('guru.adapters.ollama.ollama.chat', fake)
        assert a._collect_response() is None
        assert self._calls(repo) == []

    def test_ollama_summarise_records_phase(
            self, monkeypatch, fake_repo) -> None:
        repo = self._arm(monkeypatch, fake_repo)
        a = OllamaAdapter()
        resp = self._ollama_chunk('sum', pin=30, ein=4, load_duration=0,
                                  prompt_eval_duration=1e8,
                                  eval_duration=3e8)
        monkeypatch.setattr('guru.adapters.ollama.ollama.chat',
                            lambda *a, **k: resp)
        assert a.summarise('long transcript') == 'sum'
        [row] = self._calls(repo)
        assert row['phase'] == 'summarise' and row['cost_source'] == 'local'
        assert row['tokens_in'] == 30 and row['tokens_out'] == 4
        assert row['load_s'] is None and row['prefill_s'] == 0.1
        assert row['generate_s'] == 0.3

    # --- litellm -------------------------------------------------------------

    def _litellm(self, monkeypatch, resp, headers=None):
        a = lite.LiteLLMAdapter(base_url='http://proxy')
        monkeypatch.setattr(
            a, '_client', lambda: _fake_openai_client(resp, headers))
        return a

    def _litellm_resp(self, text='hi', **usage):
        return SimpleNamespace(
            usage=SimpleNamespace(**usage),
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason='stop')])

    def test_litellm_complete_sends_the_adapter_ceiling(
            self, monkeypatch, fake_repo) -> None:
        # The proxy enforces a thinking budget: a 300-token cap is a 400
        # error, so complete() always asks for _MAX_TOKENS at least.
        self._arm(monkeypatch, fake_repo)
        seen: dict = {}
        resp = self._litellm_resp(text='{"score": 2}', prompt_tokens=3,
                                  completion_tokens=2)
        a = lite.LiteLLMAdapter(base_url='http://proxy')
        monkeypatch.setattr(a, '_client', lambda: _fake_openai_client(
            resp, create=lambda **kw: seen.update(kw)))
        assert a.complete('grade this', max_tokens=300) == '{"score": 2}'
        assert seen['max_tokens'] == lite._MAX_TOKENS

    def test_litellm_step_prefers_cost_header(
            self, monkeypatch, fake_repo) -> None:
        repo = self._arm(monkeypatch, fake_repo)
        a = self._litellm(monkeypatch, self._litellm_resp(
            prompt_tokens=10, completion_tokens=5),
            headers={'x-litellm-response-cost': '0.002'})
        a.run_turn()
        [row] = self._calls(repo)
        assert row['adapter'] == 'LiteLLM' and row['phase'] == 'step'
        assert row['tokens_in'] == 10 and row['tokens_out'] == 5
        assert row['cost_usd'] == 0.002 and row['cost_source'] == 'header'

    def test_litellm_step_without_cost_header_uses_table(
            self, monkeypatch, fake_repo) -> None:
        repo = self._arm(monkeypatch, fake_repo)
        a = self._litellm(monkeypatch, self._litellm_resp(
            prompt_tokens=10, completion_tokens=5))
        a.run_turn()
        [row] = self._calls(repo)
        assert row['cost_source'] in ('table', 'unknown')

    def test_litellm_step_ignores_non_numeric_cost_header(
            self, monkeypatch, fake_repo) -> None:
        repo = self._arm(monkeypatch, fake_repo)
        a = self._litellm(monkeypatch, self._litellm_resp(
            prompt_tokens=10, completion_tokens=5),
            headers={'x-litellm-response-cost': 'n/a'})
        a.run_turn()
        [row] = self._calls(repo)
        assert row['cost_source'] in ('table', 'unknown')

    def test_litellm_summarise_records_phase(
            self, monkeypatch, fake_repo) -> None:
        repo = self._arm(monkeypatch, fake_repo)
        a = self._litellm(monkeypatch, self._litellm_resp(
            text='sum', prompt_tokens=8, completion_tokens=2),
            headers={'x-litellm-response-cost': '0.001'})
        assert a.summarise('long transcript') == 'sum'
        [row] = self._calls(repo)
        assert row['phase'] == 'summarise' and row['cost_usd'] == 0.001
        assert row['cost_source'] == 'header'


class TestPromptCaching:
    """cache_control markers (system prompt, last tool, last message of the
    conversation) on both remote adapters, the ``cache`` switch, and
    LiteLLM cache-usage parsing."""

    SPECS = [{'name': 'a', 'description': 'A', 'parameters': {'x': 'X'}},
             {'name': 'b', 'description': 'B', 'parameters': {}}]

    # --- pure helpers: anthropic --------------------------------------------

    def test_anthropic_system_blocks(self) -> None:
        assert anth.system_blocks('', True) is None
        assert anth.system_blocks('S', False) == 'S'
        assert anth.system_blocks('S', True) == [
            {'type': 'text', 'text': 'S',
             'cache_control': {'type': 'ephemeral'}}]

    def test_anthropic_cached_tools_marks_last_only(self) -> None:
        defs = anth.tool_defs(self.SPECS)
        out = anth.cached_tools(defs, True)
        assert 'cache_control' not in out[0]
        assert out[1]['cache_control'] == {'type': 'ephemeral'}
        assert 'cache_control' not in defs[1]          # copy, not in place
        assert anth.cached_tools(defs, False) is defs
        assert anth.cached_tools([], True) == []

    def test_anthropic_cached_messages_marks_last_user_text(self) -> None:
        msgs = [{'role': 'user', 'content': 'q1'},
                {'role': 'assistant', 'content': 'a1'},
                {'role': 'user', 'content': 'q2'}]
        out = anth.cached_messages(msgs, True)
        assert out[:2] == msgs[:2]
        assert out[2] == {'role': 'user', 'content': [
            {'type': 'text', 'text': 'q2',
             'cache_control': {'type': 'ephemeral'}}]}
        assert msgs[2]['content'] == 'q2'                # untouched
        assert anth.cached_messages(msgs, False) is msgs
        assert anth.cached_messages([], True) == []

    def test_anthropic_cached_messages_marks_last_tool_result(self) -> None:
        results = [{'type': 'tool_result', 'tool_use_id': 'a', 'content': 'x'},
                   {'type': 'tool_result', 'tool_use_id': 'b', 'content': 'y'}]
        msgs = [{'role': 'user', 'content': 'q'},
                {'role': 'assistant', 'content': []},
                {'role': 'user', 'content': results}]
        out = anth.cached_messages(msgs, True)
        assert 'cache_control' not in out[2]['content'][0]
        assert out[2]['content'][1] == {
            'type': 'tool_result', 'tool_use_id': 'b', 'content': 'y',
            'cache_control': {'type': 'ephemeral'}}
        assert 'cache_control' not in results[1]         # copy, not in place

    def test_anthropic_cached_messages_skips_unmarkable_tail(self) -> None:
        ends_assistant = [{'role': 'user', 'content': 'q'},
                          {'role': 'assistant', 'content': 'a'}]
        assert anth.cached_messages(ends_assistant, True) is ends_assistant
        empty_text = [{'role': 'user', 'content': ''}]
        assert anth.cached_messages(empty_text, True) is empty_text
        empty_blocks = [{'role': 'user', 'content': []}]
        assert anth.cached_messages(empty_blocks, True) is empty_blocks

    def test_anthropic_cached_messages_never_marks_an_empty_block(self):
        # The API rejects cache_control on an empty block: an empty tail
        # (text '' or a tool_result with no content) is skipped and the
        # block before it carries the marker instead.
        results = [{'type': 'tool_result', 'tool_use_id': 'a', 'content': 'x'},
                   {'type': 'tool_result', 'tool_use_id': 'b', 'content': ''}]
        msgs = [{'role': 'user', 'content': results}]
        out = anth.cached_messages(msgs, True)
        assert out[0]['content'][0]['cache_control'] == {'type': 'ephemeral'}
        assert out[0]['content'][1] == results[1]      # untouched, unmarked
        assert 'cache_control' not in results[0]       # copy, not in place
        tail_text = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'q'}, {'type': 'text', 'text': ''}]}]
        out = anth.cached_messages(tail_text, True)
        assert out[0]['content'][0] == {
            'type': 'text', 'text': 'q',
            'cache_control': {'type': 'ephemeral'}}
        assert 'cache_control' not in out[0]['content'][1]
        # tool_result content may be a block list: [] is empty too, and a
        # missing content key is empty; a non-text block (image) counts.
        mixed = [{'role': 'user', 'content': [
            {'type': 'image', 'source': {}},
            {'type': 'tool_result', 'tool_use_id': 'c', 'content': []},
            {'type': 'tool_result', 'tool_use_id': 'd'}]}]
        out = anth.cached_messages(mixed, True)
        assert out[0]['content'][0]['cache_control'] == {'type': 'ephemeral'}
        assert all('cache_control' not in b for b in out[0]['content'][1:])
        # Nothing with content: nothing is marked, the list is unchanged.
        all_empty = [{'role': 'user', 'content': [
            {'type': 'text', 'text': ''},
            {'type': 'tool_result', 'tool_use_id': 'e', 'content': ''}]}]
        assert anth.cached_messages(all_empty, True) is all_empty

    # --- pure helpers: litellm ----------------------------------------------

    def test_litellm_cached_messages_marks_last_system_part(self) -> None:
        msgs = [{'role': 'system', 'content': 'BASE'},
                {'role': 'system', 'content': 'SUMMARY'},
                {'role': 'user', 'content': 'q'}]
        out = lite.cached_messages(msgs, True)
        assert out[0]['content'] == [{'type': 'text', 'text': 'BASE'}]
        assert out[1]['content'] == [{
            'type': 'text', 'text': 'SUMMARY',
            'cache_control': {'type': 'ephemeral'}}]
        assert out[2] == {'role': 'user', 'content': [{
            'type': 'text', 'text': 'q',
            'cache_control': {'type': 'ephemeral'}}]}
        assert msgs[1]['content'] == 'SUMMARY'          # untouched
        assert msgs[2]['content'] == 'q'
        assert lite.cached_messages(msgs, False) is msgs
        no_marks = [{'role': 'user', 'content': 'q'},
                    {'role': 'assistant', 'content': 'a'}]
        assert lite.cached_messages(no_marks, True) is no_marks

    def test_litellm_cached_messages_marks_last_tool_result(self) -> None:
        # What the SBP proxy forwards to Anthropic as a marked tool_result
        # (probe 2026-09-25: cache reads on the whole history).
        msgs = [{'role': 'user', 'content': 'q'},
                {'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': 'c1', 'type': 'function',
                     'function': {'name': 'a', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'c1', 'content': 'result'}]
        out = lite.cached_messages(msgs, True)
        assert out[0] == msgs[0] and out[1] is msgs[1]
        assert out[2] == {'role': 'tool', 'tool_call_id': 'c1', 'content': [
            {'type': 'text', 'text': 'result',
             'cache_control': {'type': 'ephemeral'}}]}
        assert msgs[2]['content'] == 'result'
        # Only the last message carries the breakpoint; an empty tail or a
        # non-string content is left alone (the system marker still goes on).
        tail_empty = [{'role': 'system', 'content': 'S'},
                      {'role': 'tool', 'tool_call_id': 'c1', 'content': ''}]
        out = lite.cached_messages(tail_empty, True)
        assert out[0]['content'][0]['cache_control'] == {'type': 'ephemeral'}
        assert out[1] is tail_empty[1]
        assert lite.cached_messages([], True) == []

    def test_litellm_cached_tools_marks_last_only(self) -> None:
        defs = lite.openai_tool_defs(self.SPECS)
        out = lite.cached_tools(defs, True)
        assert 'cache_control' not in out[0]
        assert out[1]['cache_control'] == {'type': 'ephemeral'}
        assert out[1]['type'] == 'function'
        assert lite.cached_tools(None, True) is None

    def test_litellm_usage_anthropic_style_fields(self) -> None:
        # What the SBP proxy returns for aws/claude-* (probe 2026-09-25):
        # prompt_tokens counts the cached tokens too.
        u = lite.usage_from(SimpleNamespace(
            prompt_tokens=6746, completion_tokens=47,
            cache_read_input_tokens=0, cache_creation_input_tokens=6386,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0)))
        assert (u.input_tokens, u.output_tokens) == (360, 47)
        assert (u.cache_read_tokens, u.cache_write_tokens) == (0, 6386)

    def test_litellm_usage_openai_style_details(self) -> None:
        u = lite.usage_from({'prompt_tokens': 100, 'completion_tokens': 5,
                             'prompt_tokens_details': {'cached_tokens': 60}})
        assert u.input_tokens == 40 and u.cache_read_tokens == 60
        assert u.cache_write_tokens == 0

    def test_litellm_usage_plain_and_missing(self) -> None:
        u = lite.usage_from(SimpleNamespace(prompt_tokens=10,
                                            completion_tokens=5))
        assert (u.input_tokens, u.output_tokens) == (10, 5)
        assert u.cache_read_tokens == 0 and u.cache_write_tokens == 0
        assert lite.usage_from(None) == lite.pricing.Usage()
        odd = lite.usage_from(SimpleNamespace(prompt_tokens='x',
                                              completion_tokens=None,
                                              cache_read_input_tokens=True))
        assert odd == lite.pricing.Usage()

    # --- wiring: what the request carries ----------------------------------

    def _arm(self, monkeypatch):
        from guru.adapters import turn
        monkeypatch.setattr(ui, 'note_thinking', lambda: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        monkeypatch.setattr(turn, '_render_answer', lambda c: None)
        monkeypatch.setattr(session, 'model', 'm')
        monkeypatch.setattr(session, 'cancel_requested', False)
        monkeypatch.setattr(session, 'session_in', 0)
        monkeypatch.setattr(session, 'session_out', 0)
        monkeypatch.setattr(session, 'messages', [
            {'role': 'system', 'content': 'SYS'},
            {'role': 'user', 'content': 'q'}])
        monkeypatch.setattr('guru.domain.tools.active_specs',
                            lambda: self.SPECS)

    def _anthropic_kwargs(self, monkeypatch, cache: bool) -> dict:
        self._arm(monkeypatch)
        seen: dict = {}
        resp = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason='end_turn',
            content=[SimpleNamespace(type='text', text='ok')])

        def create(**kw):
            seen.update(kw)
            return resp
        a = anth.AnthropicAdapter(thinking=False, cache=cache)
        monkeypatch.setattr(a, '_client', lambda: SimpleNamespace(
            messages=SimpleNamespace(create=create)))
        a.run_turn()
        return seen

    def test_anthropic_step_sends_markers(self, monkeypatch) -> None:
        kw = self._anthropic_kwargs(monkeypatch, cache=True)
        assert kw['system'][0]['cache_control'] == {'type': 'ephemeral'}
        assert kw['system'][0]['text'] == 'SYS'
        assert kw['tools'][-1]['cache_control'] == {'type': 'ephemeral'}
        assert 'cache_control' not in kw['tools'][0]
        assert kw['messages'] == [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'q',
             'cache_control': {'type': 'ephemeral'}}]}]

    def test_anthropic_cache_off(self, monkeypatch) -> None:
        kw = self._anthropic_kwargs(monkeypatch, cache=False)
        assert kw['system'] == 'SYS'
        assert 'cache_control' not in str(kw['tools'])
        # The native list itself (no copy); the reply was appended after.
        assert kw['messages'][0] == {'role': 'user', 'content': 'q'}

    def _litellm_kwargs(self, monkeypatch, cache: bool, fake_repo) -> tuple:
        self._arm(monkeypatch)
        seen: dict = {}
        resp = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=6746, completion_tokens=47,
                                  cache_read_input_tokens=6386,
                                  cache_creation_input_tokens=0),
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='ok', tool_calls=None),
                finish_reason='stop')])
        a = lite.LiteLLMAdapter(base_url='http://proxy', cache=cache)
        monkeypatch.setattr(a, '_client', lambda: _fake_openai_client(
            resp, create=lambda **kw: seen.update(kw)))
        a.run_turn()
        from guru.domain import ledger
        ledger.flush()
        return seen, fake_repo.stream('calls')

    def test_litellm_step_sends_markers_and_records_cache(
            self, monkeypatch, fake_repo) -> None:
        kw, rows = self._litellm_kwargs(monkeypatch, True, fake_repo)
        system = kw['messages'][0]
        assert system['role'] == 'system'
        assert system['content'][0]['cache_control'] == {'type': 'ephemeral'}
        assert kw['tools'][-1]['cache_control'] == {'type': 'ephemeral'}
        assert 'cache_control' not in kw['tools'][0]
        assert kw['messages'][1] == {'role': 'user', 'content': [
            {'type': 'text', 'text': 'q',
             'cache_control': {'type': 'ephemeral'}}]}
        [row] = rows
        assert row['tokens_in'] == 360 and row['cache_read'] == 6386
        assert row['cache_write'] == 0 and row['tokens_out'] == 47

    def test_litellm_cache_off(self, monkeypatch, fake_repo) -> None:
        kw, _ = self._litellm_kwargs(monkeypatch, False, fake_repo)
        assert kw['messages'][0] == {'role': 'system', 'content': 'SYS'}
        assert kw['messages'][1] == {'role': 'user', 'content': 'q'}
        assert 'cache_control' not in str(kw['tools'])

    def test_cache_defaults_on_and_config_switch(self) -> None:
        assert anth.AnthropicAdapter().cache is True
        assert lite.LiteLLMAdapter().cache is True
        import guru.cli as cli
        assert cli._instantiate({'type': 'anthropic', 'cache': False}).cache \
            is False
        assert cli._instantiate({'type': 'litellm', 'cache': False}).cache \
            is False
        assert cli._instantiate({'type': 'litellm'}).cache is True


def _fake_openai_client(resp=None, headers=None, create=None):
    """Fake ``openai.OpenAI`` exposing only what the LiteLLM adapter calls:
    ``chat.completions.with_raw_response.create`` -> raw with ``.parse()``
    and dict-like ``.headers``. ``create`` overrides the call (for raising)."""
    def _create(**kw):
        if create is not None:
            create(**kw)
        return SimpleNamespace(parse=lambda: resp, headers=headers or {})
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=_create))))


class TestStruggleCounters(TestCallRecords):
    """Provider exceptions and refusals feed session.struggle/last_error."""

    def _fresh(self, monkeypatch, fake_repo):
        return self._arm(monkeypatch, fake_repo)   # counters: conftest

    def _raising(self, exc):
        def create(**kw):
            raise exc
        return create

    def test_anthropic_step_error(self, monkeypatch, fake_repo) -> None:
        repo = self._fresh(monkeypatch, fake_repo)
        a = anth.AnthropicAdapter(thinking=False)
        client = SimpleNamespace(messages=SimpleNamespace(
            create=self._raising(RuntimeError('overloaded 529'))))
        monkeypatch.setattr(a, '_client', lambda: client)
        a.run_turn()                                 # printed, not raised
        assert session.struggle['provider_errors'] == 1
        assert 'overloaded 529' in session.last_error
        assert len(session.last_error) <= 200
        assert self._calls(repo) == []

    def test_anthropic_summarise_error(self, monkeypatch, fake_repo) -> None:
        self._fresh(monkeypatch, fake_repo)
        a = anth.AnthropicAdapter(thinking=False)
        client = SimpleNamespace(messages=SimpleNamespace(
            create=self._raising(RuntimeError('boom'))))
        monkeypatch.setattr(a, '_client', lambda: client)
        assert a.summarise('t').startswith('(summary failed')
        assert session.struggle['provider_errors'] == 1
        assert 'boom' in session.last_error

    def test_anthropic_refusal_counted(self, monkeypatch, fake_repo) -> None:
        repo = self._fresh(monkeypatch, fake_repo)
        resp = self._anthropic_resp(text='I cannot help with that.',
                                    input_tokens=5, output_tokens=5)
        resp.stop_reason = 'refusal'
        a = self._anthropic(monkeypatch, resp)
        a.run_turn()
        assert session.struggle['refusals'] == 1
        assert session.struggle['provider_errors'] == 0
        assert len(self._calls(repo)) == 1

    def test_anthropic_end_turn_not_a_refusal(
            self, monkeypatch, fake_repo) -> None:
        self._fresh(monkeypatch, fake_repo)
        a = self._anthropic(monkeypatch, self._anthropic_resp(
            input_tokens=5, output_tokens=5))
        a.run_turn()
        assert session.struggle['refusals'] == 0

    def test_litellm_step_error(self, monkeypatch, fake_repo) -> None:
        self._fresh(monkeypatch, fake_repo)
        a = lite.LiteLLMAdapter(base_url='http://proxy')
        client = _fake_openai_client(
            create=self._raising(ConnectionError('refused')))
        monkeypatch.setattr(a, '_client', lambda: client)
        a.run_turn()
        assert session.struggle['provider_errors'] == 1
        assert 'refused' in session.last_error

    def test_litellm_summarise_error(self, monkeypatch, fake_repo) -> None:
        self._fresh(monkeypatch, fake_repo)
        a = lite.LiteLLMAdapter(base_url='http://proxy')
        client = _fake_openai_client(
            create=self._raising(ConnectionError('refused')))
        monkeypatch.setattr(a, '_client', lambda: client)
        assert a.summarise('t').startswith('(summary failed')
        assert session.struggle['provider_errors'] == 1

    def test_ollama_summarise_error(self, monkeypatch, fake_repo) -> None:
        self._fresh(monkeypatch, fake_repo)
        a = OllamaAdapter()

        def boom(**kw):
            raise ConnectionError('ollama down')
        monkeypatch.setattr('guru.adapters.ollama.ollama.chat', boom)
        assert a.summarise('t') == ''
        assert session.struggle['provider_errors'] == 1
        assert 'ollama down' in session.last_error

    def test_ollama_step_error(self, monkeypatch, fake_repo) -> None:
        self._fresh(monkeypatch, fake_repo)
        a = OllamaAdapter()
        monkeypatch.setattr(a, '_fit_after_load', lambda: None)

        def boom():
            raise ConnectionError('ollama down')
        monkeypatch.setattr(a, '_collect_response', boom)
        a.run_turn()                                 # printed, not raised
        assert session.struggle['provider_errors'] == 1
        assert 'ollama down' in session.last_error
        assert session.cancel_requested is False


class TestControllerExecuted:
    """controller_executed on the TurnRecord (Task 4.5)."""

    def _run(self, monkeypatch, fake_repo, seq, controller=True):
        from guru.adapters import turn
        monkeypatch.setattr(ui, 'note_thinking', lambda: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        monkeypatch.setattr(turn, '_render_answer', lambda c: None)
        monkeypatch.setattr(session, 'messages', [])
        monkeypatch.setattr(session, 'cancel_requested', False)
        monkeypatch.setattr(session, 'task_id', '')
        monkeypatch.setattr(session, 'can_spawn', True)
        monkeypatch.setattr(session, 'controller', controller)
        monkeypatch.setattr(config, 'DELEGATION_NUDGE_MIN_READS', 0)
        it = iter(seq)
        turn.run_loop(step=lambda: next(it), run_tools=lambda p: None,
                      add_user=lambda t: None)
        from guru.domain import ledger
        ledger.flush()
        rows = fake_repo.stream('turns')
        assert len(rows) == 1
        return rows[0]

    def test_foreign_tool_call_flips_flag(self, monkeypatch, fake_repo):
        row = self._run(monkeypatch, fake_repo, [
            ("", [("read_file", {"path": "x"}, "r1")]), ("short.", [])])
        assert row['controller_executed'] is True

    def test_long_answer_without_spawn_flips_flag(self, monkeypatch,
                                                  fake_repo):
        row = self._run(monkeypatch, fake_repo, [("x" * 601, [])])
        assert row['controller_executed'] is True

    def test_spawn_is_a_foreign_tool_for_a_controller(
            self, monkeypatch, fake_repo):
        """A controller has ``plan`` only; calling spawn/join is doing the
        coordination by hand and flips the flag."""
        row = self._run(monkeypatch, fake_repo, [
            ("", [("spawn", {"task": "t"}, "r1")]),
            ("", [("join", {"targets": "agent1"}, "r2")]),
            ("short.", [])])
        assert row['controller_executed'] is True

    def test_delegating_plan_counts_tasks_and_is_fine(self, monkeypatch,
                                                      fake_repo):
        from guru.adapters import turn
        from guru.domain import plan
        monkeypatch.setattr(ui, 'note_thinking', lambda: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        monkeypatch.setattr(turn, '_render_answer', lambda c: None)
        monkeypatch.setattr(session, 'messages', [
            {'role': 'user', 'content': 'review auth for security'}])
        monkeypatch.setattr(session, 'cancel_requested', False)
        monkeypatch.setattr(session, 'task_id', '')
        monkeypatch.setattr(session, 'can_spawn', True)
        monkeypatch.setattr(session, 'controller', True)
        args = {'outcome': 'delegate', 'tasks': [
            {'goal': 'review auth', 'kind': 'review',
             'complexity': 'standard'},
            {'goal': 'review tests', 'kind': 'review',
             'complexity': 'trivial'}]}
        it = iter([("", [("plan", args, "r1")])])

        def run_tools(pending):
            session.messages.append({
                'role': 'tool', 'tool_name': 'plan',
                'content': plan.delegated_text(
                    ['agent1', 'agent2'], plan.parse(args)[0].tasks)})
            session.turn_waiting = True
        turn.run_loop(step=lambda: next(it), run_tools=run_tools,
                      add_user=lambda t: None)
        from guru.domain import ledger
        ledger.flush()
        [row] = fake_repo.stream('turns')
        assert row['controller_executed'] is False
        assert row['tasks_spawned'] == 2
        assert row['tools_used'] == ['plan']

    def test_short_conversational_answer_is_fine(self, monkeypatch,
                                                 fake_repo):
        row = self._run(monkeypatch, fake_repo, [("Hello there.", [])])
        assert row['controller_executed'] is False

    def test_mailbox_delivery_turn_never_flips(self, monkeypatch, fake_repo):
        from guru.adapters import turn
        for prefix in ('[joined results]\n- agent1: ...',
                       '[result from agent1 · task: t]\nA1'):
            monkeypatch.setattr(session, 'messages', [])
            row = None

            def step(it=iter([("x" * 601, [])])):
                return next(it)
            monkeypatch.setattr(ui, 'note_thinking', lambda: None)
            monkeypatch.setattr(ui, 'status_draw', lambda: None)
            monkeypatch.setattr(turn, '_render_answer', lambda c: None)
            monkeypatch.setattr(session, 'task_id', '')
            monkeypatch.setattr(session, 'can_spawn', True)
            monkeypatch.setattr(session, 'controller', True)
            monkeypatch.setattr(config, 'DELEGATION_NUDGE_MIN_READS', 0)
            session.messages.append({'role': 'user', 'content': prefix})
            turn.run_loop(step=step, run_tools=lambda p: None,
                          add_user=lambda t: None)
            from guru.domain import ledger
            ledger.flush()
            row = fake_repo.stream('turns')[-1]
            assert row['controller_executed'] is False, prefix

    def test_mailbox_turn_records_the_human_request(self, monkeypatch,
                                                    fake_repo) -> None:
        """A synthesis turn is still recognised as one (never flips) when
        a human request precedes the delivery, and the TurnRecord names
        that request, not the delivery."""
        from guru.adapters import turn
        monkeypatch.setattr(session, 'messages', [
            {'role': 'user', 'content': 'review auth for security'},
            {'role': 'assistant', 'content': 'spawned'},
            {'role': 'user', 'content': '[joined results]\n- agent1: A1'}])
        monkeypatch.setattr(ui, 'note_thinking', lambda: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        monkeypatch.setattr(turn, '_render_answer', lambda c: None)
        monkeypatch.setattr(session, 'task_id', '')
        monkeypatch.setattr(session, 'can_spawn', True)
        monkeypatch.setattr(session, 'controller', True)
        monkeypatch.setattr(config, 'DELEGATION_NUDGE_MIN_READS', 0)
        assert turn._mailbox_turn() is True
        assert turn.turn_request() == 'review auth for security'
        it = iter([("x" * 601, [])])
        turn.run_loop(step=lambda: next(it), run_tools=lambda p: None,
                      add_user=lambda t: None)
        from guru.domain import ledger
        ledger.flush()
        row = fake_repo.stream('turns')[-1]
        assert row['controller_executed'] is False
        assert row['request'] == 'review auth for security'
        assert turn.request_in is conversation.request_in

    def test_turn_start_clears_last_error(self, monkeypatch, fake_repo):
        monkeypatch.setattr(session, 'last_error', 'old failure')
        self._run(monkeypatch, fake_repo, [("ok.", [])])
        assert session.last_error == ''

    def test_non_controller_never_flips(self, monkeypatch, fake_repo):
        row = self._run(monkeypatch, fake_repo, [
            ("", [("read_file", {"path": "x"}, "r1")]), ("x" * 601, [])],
            controller=False)
        assert row['controller_executed'] is False


class TestRequestDump:
    """``GURU_DUMP_REQUESTS=<dir>`` writes each outgoing request's kwargs
    as ``<ts>-<adapter>-<n>.json``; the API key lives in the SDK client and
    is never part of them."""

    def test_unset_writes_nothing(self, tmp_path, monkeypatch) -> None:
        from guru.adapters import base
        monkeypatch.delenv(base.DUMP_ENV, raising=False)
        assert base.dump_request('X', {'model': 'm'}) is None
        assert list(tmp_path.iterdir()) == []

    def test_writes_kwargs_as_json(self, tmp_path, monkeypatch) -> None:
        import json
        import re
        from guru.adapters import base
        monkeypatch.setenv(base.DUMP_ENV, str(tmp_path / 'dumps'))
        kwargs = {'model': 'm', 'messages': [{'role': 'user', 'content': 'q'}],
                  'tools': None}
        path = base.dump_request('SBP Litellm', kwargs)
        assert path is not None and path.parent == tmp_path / 'dumps'
        assert re.fullmatch(r'\d{8}T\d{9}Z-SBP_Litellm-\d+\.json', path.name)
        assert json.loads(path.read_text(encoding='utf-8')) == kwargs
        second = base.dump_request('SBP Litellm', kwargs)
        assert second is not None and second != path
        n1 = int(path.stem.rsplit('-', 1)[1])
        n2 = int(second.stem.rsplit('-', 1)[1])
        assert n2 == n1 + 1                           # the request sequence

    def test_sdk_objects_are_serialised(self, tmp_path, monkeypatch):
        import json
        from guru.adapters import base
        monkeypatch.setenv(base.DUMP_ENV, str(tmp_path))

        class Block:
            def model_dump(self):
                return {'type': 'text', 'text': 'hi'}
        path = base.dump_request('Anthropic', {'messages': [
            {'role': 'assistant', 'content': [Block()]}]})
        assert path is not None
        data = json.loads(path.read_text(encoding='utf-8'))
        assert data['messages'][0]['content'] == [
            {'type': 'text', 'text': 'hi'}]

    def test_write_failure_is_swallowed(self, tmp_path, monkeypatch) -> None:
        from guru.adapters import base
        blocker = tmp_path / 'file'
        blocker.write_text('x', encoding='utf-8')
        monkeypatch.setenv(base.DUMP_ENV, str(blocker / 'sub'))
        assert base.dump_request('X', {'model': 'm'}) is None

    def _quiet(self, monkeypatch) -> None:
        from guru.adapters import turn
        monkeypatch.setattr(ui, 'note_thinking', lambda: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        monkeypatch.setattr(turn, '_render_answer', lambda c: None)
        monkeypatch.setattr(session, 'model', 'm')
        monkeypatch.setattr(session, 'cancel_requested', False)
        monkeypatch.setattr(session, 'session_in', 0)
        monkeypatch.setattr(session, 'session_out', 0)
        monkeypatch.setattr(session, 'messages', [
            {'role': 'system', 'content': 'SYS'},
            {'role': 'user', 'content': 'q'}])
        monkeypatch.setattr('guru.domain.tools.active_specs', lambda: [])

    def test_litellm_turn_dumps_without_the_key(
            self, tmp_path, monkeypatch) -> None:
        import json
        from guru.adapters import base
        self._quiet(monkeypatch)
        monkeypatch.setenv(base.DUMP_ENV, str(tmp_path))
        resp = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='ok', tool_calls=None),
                finish_reason='stop')])
        a = lite.LiteLLMAdapter(name='SBP Litellm', base_url='http://p',
                                api_key='sk-very-secret-key')
        monkeypatch.setattr(a, '_client', lambda: _fake_openai_client(resp))
        a.run_turn()
        [path] = list(tmp_path.iterdir())
        assert '-SBP_Litellm-' in path.name
        text = path.read_text(encoding='utf-8')
        assert 'sk-very-secret-key' not in text
        data = json.loads(text)
        assert data['model'] == 'm' and data['messages'][-1]['content'] == [
            {'type': 'text', 'text': 'q',
             'cache_control': {'type': 'ephemeral'}}]

    def test_anthropic_turn_dumps_without_the_key(
            self, tmp_path, monkeypatch) -> None:
        import json
        from guru.adapters import base
        self._quiet(monkeypatch)
        monkeypatch.setenv(base.DUMP_ENV, str(tmp_path))
        resp = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason='end_turn',
            content=[SimpleNamespace(type='text', text='ok')])
        a = anth.AnthropicAdapter(name='Claude Code', thinking=False,
                                  api_key='sk-ant-very-secret')
        client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kw: resp))
        monkeypatch.setattr(a, '_client', lambda: client)
        a.run_turn()
        [path] = list(tmp_path.iterdir())
        assert '-Claude_Code-' in path.name
        text = path.read_text(encoding='utf-8')
        assert 'sk-ant-very-secret' not in text
        assert json.loads(text)['system'][0]['text'] == 'SYS'


class TestNativeRoundRebuild:
    """A past tool round with provider ids is sent in the native shape the
    in-flight turn used, so the request prefix is byte-identical across
    turns (the cache-miss the loop-1 triage saw on every controller
    mailbox turn: the flattened '(used tools)' / '[tool X result]' text
    differed from the previous request's tool_calls / tool messages)."""

    ROUND = [
        {'role': 'system', 'content': 'SYS'},
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call_1', 'raw_arguments': '{"path":"a.py"}',
             'function': {'name': 'read_file',
                          'arguments': {'path': 'a.py'}}},
            {'id': 'call_2',
             'function': {'name': 'list_dir', 'arguments': {'path': '.'}}}]},
        {'role': 'tool', 'tool_name': 'read_file', 'tool_call_id': 'call_1',
         'tool_args': {'path': 'a.py'}, 'content': 'A'},
        {'role': 'tool', 'tool_name': 'list_dir', 'tool_call_id': 'call_2',
         'tool_args': {'path': '.'}, 'content': 'B'},
        {'role': 'user', 'content': 'next'},
    ]

    def test_openai_round_rebuilt_natively(self) -> None:
        out = lite.to_openai_messages(self.ROUND)
        assert out[2] == {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call_1', 'type': 'function',
             'function': {'name': 'read_file',
                          'arguments': '{"path":"a.py"}'}},   # raw text kept
            {'id': 'call_2', 'type': 'function',
             'function': {'name': 'list_dir',
                          'arguments': '{"path": "."}'}}]}     # re-serialised
        assert out[3] == {'role': 'tool', 'tool_call_id': 'call_1',
                          'content': 'A'}
        assert out[4] == {'role': 'tool', 'tool_call_id': 'call_2',
                          'content': 'B'}
        assert out[5] == {'role': 'user', 'content': 'next'}
        assert len(out) == 6

    def test_openai_text_with_calls_keeps_the_text(self) -> None:
        head = dict(self.ROUND[2], content='Let me look.')
        msgs = [head] + self.ROUND[3:5]
        out = lite.to_openai_messages(msgs)
        assert out[0]['content'] == 'Let me look.'
        assert [t['id'] for t in out[0]['tool_calls']] == ['call_1', 'call_2']

    def test_openai_round_without_matching_results_flattens(self) -> None:
        # a result missing (or answering another id) — no native round,
        # the API would reject an unanswered tool call.
        msgs = self.ROUND[:4] + [self.ROUND[5]]
        out = lite.to_openai_messages(msgs)
        assert out[2] == {'role': 'assistant', 'content': '(used tools)'}
        assert out[3]['role'] == 'user' and out[3]['content'].startswith(
            '[tool read_file result]')
        swapped = self.ROUND[:3] + [self.ROUND[4], self.ROUND[3]]
        out = lite.to_openai_messages(swapped)
        assert out[2]['content'] == '(used tools)'

    def test_openai_round_without_ids_flattens(self) -> None:
        # an Ollama history / an older transcript: no ids anywhere.
        msgs = [
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'function': {'name': 'read_file',
                              'arguments': {'path': 'a.py'}}}]},
            {'role': 'tool', 'tool_name': 'read_file', 'content': 'A'}]
        out = lite.to_openai_messages(msgs)
        assert out[0] == {'role': 'assistant', 'content': '(used tools)'}
        assert out[1]['role'] == 'user'

    def test_anthropic_round_rebuilt_natively(self) -> None:
        system, out = anth.to_anthropic_messages(self.ROUND)
        assert system == 'SYS'
        assert out[1] == {'role': 'assistant', 'content': [
            {'type': 'tool_use', 'id': 'call_1', 'name': 'read_file',
             'input': {'path': 'a.py'}},
            {'type': 'tool_use', 'id': 'call_2', 'name': 'list_dir',
             'input': {'path': '.'}}]}
        assert out[2] == {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'call_1', 'content': 'A'},
            {'type': 'tool_result', 'tool_use_id': 'call_2', 'content': 'B'}]}
        assert out[3] == {'role': 'user', 'content': 'next'}
        with_text = [dict(self.ROUND[2], content='Looking.')] + self.ROUND[3:5]
        _, out = anth.to_anthropic_messages(with_text)
        assert out[0]['content'][0] == {'type': 'text', 'text': 'Looking.'}
        assert out[0]['content'][1]['type'] == 'tool_use'

    def test_anthropic_round_without_ids_flattens(self) -> None:
        msgs = [self.ROUND[1], {**self.ROUND[2], 'tool_calls': [
            {'function': {'name': 'read_file',
                          'arguments': {'path': 'a.py'}}}]},
            {'role': 'tool', 'tool_name': 'read_file', 'content': 'A'}]
        _, out = anth.to_anthropic_messages(msgs)
        assert out[1] == {'role': 'assistant', 'content': '(used tools)'}
        assert out[2]['content'].startswith('[tool read_file result]')

    def test_neutral_assistant_keeps_ids(self) -> None:
        msg = lite.neutral_assistant('', [
            ('read_file', {'path': 'a'}, 'call_1', '{"path":"a"}'),
            ('list_dir', {'path': '.'})])
        assert msg['tool_calls'][0] == {
            'id': 'call_1', 'raw_arguments': '{"path":"a"}',
            'function': {'name': 'read_file', 'arguments': {'path': 'a'}}}
        assert msg['tool_calls'][1] == {
            'function': {'name': 'list_dir', 'arguments': {'path': '.'}}}
        msg = anth.neutral_assistant('', [('read_file', {'path': 'a'}, 'tu1')])
        assert msg['tool_calls'][0]['id'] == 'tu1'

    # --- a whole turn, then the next turn's translation ----------------------

    def _arm(self, monkeypatch) -> None:
        from guru.adapters import turn
        from guru.domain import tools
        monkeypatch.setattr(ui, 'note_thinking', lambda: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        monkeypatch.setattr(turn, '_render_answer', lambda c: None)
        monkeypatch.setattr(session, 'model', 'm')
        monkeypatch.setattr(session, 'cancel_requested', False)
        monkeypatch.setattr(session, 'session_in', 0)
        monkeypatch.setattr(session, 'session_out', 0)
        monkeypatch.setattr(session, 'messages', [
            {'role': 'system', 'content': 'SYS'},
            {'role': 'user', 'content': 'q'}])
        monkeypatch.setattr(tools, 'active_specs', lambda: [
            {'name': 'spawn', 'description': 'd',
             'parameters': {'task': 't'}}])
        monkeypatch.setattr(tools, 'execute_tool',
                            lambda name, args: f'ran {name}')

    @staticmethod
    def _plain(m: dict) -> dict:
        """A message with a one-text-part content list read as its text
        (the cache marker's shape, equivalent for the API)."""
        c = m.get('content')
        if isinstance(c, list) and len(c) == 1 and c[0].get('type') == 'text':
            return {**m, 'content': c[0]['text']}
        return m

    def test_litellm_next_turn_prefix_matches_the_last_request(
            self, monkeypatch) -> None:
        self._arm(monkeypatch)
        seen: list = []
        tool_call = SimpleNamespace(
            id='tooluse_1', function=SimpleNamespace(
                name='spawn', arguments='{"task":"look at  x"}'))
        responses = iter([
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=None,
                                            tool_calls=[tool_call]),
                    finish_reason='tool_calls')]),
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='done', tool_calls=None),
                    finish_reason='stop')])])

        def _create(**kw):
            seen.append(kw)
            return SimpleNamespace(parse=lambda: next(responses), headers={})
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(with_raw_response=SimpleNamespace(
                create=_create))))
        a = lite.LiteLLMAdapter(base_url='http://p')
        monkeypatch.setattr(a, '_client', lambda: client)
        a.run_turn()
        assert len(seen) == 2
        tool_msg = session.messages[3]
        assert tool_msg['role'] == 'tool' and tool_msg['tool_call_id'] == \
            'tooluse_1' and tool_msg['tool_args'] == {'task': 'look at  x'}
        assert session.messages[2]['tool_calls'][0]['id'] == 'tooluse_1'
        assert session.messages[2]['tool_calls'][0]['raw_arguments'] == \
            '{"task":"look at  x"}'
        # Next turn: the user speaks again; what the adapter would send.
        session.messages.append({'role': 'user', 'content': 'and then?'})
        again = lite.cached_messages(
            lite.to_openai_messages(session.messages), True)
        last = seen[-1]['messages']
        assert [self._plain(m) for m in again[:len(last)]] == \
            [self._plain(m) for m in last]
        assert again[-1]['content'][0]['cache_control'] == {
            'type': 'ephemeral'}

    def test_anthropic_next_turn_prefix_matches_the_last_request(
            self, monkeypatch) -> None:
        self._arm(monkeypatch)
        seen: list = []
        use = SimpleNamespace(type='tool_use', id='tu_1', name='spawn',
                              input={'task': 'x'})
        responses = iter([
            SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason='tool_use', content=[use]),
            SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason='end_turn',
                content=[SimpleNamespace(type='text', text='done')])])

        def _create(**kw):
            seen.append(kw)
            return next(responses)
        a = anth.AnthropicAdapter(thinking=False)
        monkeypatch.setattr(a, '_client', lambda: SimpleNamespace(
            messages=SimpleNamespace(create=_create)))
        a.run_turn()
        assert session.messages[3]['tool_call_id'] == 'tu_1'
        assert session.messages[2]['tool_calls'][0]['id'] == 'tu_1'
        session.messages.append({'role': 'user', 'content': 'and then?'})
        _, again = anth.to_anthropic_messages(session.messages)
        # The in-flight native history held the SDK's block objects; the
        # rebuilt round is the same blocks as dicts.
        last = seen[-1]['messages']
        assert again[1]['content'] == [
            {'type': 'tool_use', 'id': 'tu_1', 'name': 'spawn',
             'input': {'task': 'x'}}]
        assert last[1]['content'] == [use]
        assert [self._plain(m) for m in again[2:3]] == [
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu_1',
                 'content': 'ran spawn'}]}]
        assert [self._plain(m) for m in last[2:3]] == [
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu_1',
                 'content': 'ran spawn',
                 'cache_control': {'type': 'ephemeral'}}]}]
        assert again[3] == {'role': 'assistant', 'content': 'done'}
        assert again[4] == {'role': 'user', 'content': 'and then?'}


class TestRequestIn:
    """``turn.request_in`` finds the turn's request in any agent's
    history (the orchestrator reads a parent's for the panel judge)."""

    def test_skips_nudges_and_non_user_messages(self) -> None:
        from guru.adapters import turn
        msgs = [{'role': 'system', 'content': 's'},
                {'role': 'user', 'content': 'review the auth service'},
                {'role': 'assistant', 'content': 'Let me…'},
                {'role': 'user', 'content': turn._NUDGE_TEXT},
                {'role': 'tool', 'tool_name': 'spawn', 'content': 'ok'}]
        assert turn.request_in(msgs) == 'review the auth service'
        assert turn.request_in([{'role': 'system', 'content': 's'}]) == ''
        assert turn.request_in([]) == ''


class TestThinkingLastRound:
    """With extended thinking on, the Anthropic API requires the *last*
    assistant message to start with a thinking block when it carries
    ``tool_use``; previous turns' thinking is not kept, so the rebuilt
    round that would be that last message is flattened to text instead,
    and every earlier round is still rebuilt from its ids."""

    ROUND = [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'toolu_1',
             'function': {'name': 'join', 'arguments': {'targets': 'a'}}}]},
        {'role': 'tool', 'tool_name': 'join', 'tool_call_id': 'toolu_1',
         'content': 'Waiting for agent1'},
    ]
    DELIVERY = {'role': 'user', 'content': '[joined results]\nA1'}

    def test_last_round_flattened_with_thinking(self) -> None:
        # a turn that ended on a join, resumed by the mailbox delivery
        msgs = self.ROUND + [self.DELIVERY]
        _, out = anth.to_anthropic_messages(msgs, thinking=True)
        assert out[1] == {'role': 'assistant', 'content': '(used tools)'}
        assert out[2] == {'role': 'user',
                          'content': '[tool join result]\nWaiting for agent1'}
        assert out[3] == self.DELIVERY
        assert anth.native_round(msgs, 1, thinking=True) is None

    def test_last_round_rebuilt_without_thinking(self) -> None:
        msgs = self.ROUND + [self.DELIVERY]
        _, out = anth.to_anthropic_messages(msgs)
        assert out[1]['content'] == [
            {'type': 'tool_use', 'id': 'toolu_1', 'name': 'join',
             'input': {'targets': 'a'}}]
        assert out[2]['content'][0]['type'] == 'tool_result'
        assert anth.native_round(msgs, 1, thinking=False) is not None
        assert anth.native_round(msgs, 1) is not None       # the default

    def test_earlier_round_rebuilt_with_thinking(self) -> None:
        # an assistant text follows the round: it is not the last assistant
        # message, so the API accepts it without its thinking blocks
        msgs = self.ROUND + [
            {'role': 'assistant', 'content': 'done'},
            {'role': 'user', 'content': 'and then?'}]
        _, out = anth.to_anthropic_messages(msgs, thinking=True)
        assert out[1]['content'][0]['type'] == 'tool_use'
        assert out[2]['content'][0]['type'] == 'tool_result'
        assert out[3] == {'role': 'assistant', 'content': 'done'}
        # two rounds, the second last: the first rebuilt, the second flat
        second = [
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'toolu_2',
                 'function': {'name': 'check', 'arguments': {}}}]},
            {'role': 'tool', 'tool_name': 'check', 'tool_call_id': 'toolu_2',
             'content': 'running'}]
        msgs = self.ROUND + second + [self.DELIVERY]
        _, out = anth.to_anthropic_messages(msgs, thinking=True)
        assert out[1]['content'][0]['type'] == 'tool_use'
        assert out[3] == {'role': 'assistant', 'content': '(used tools)'}
        assert out[4]['content'].startswith('[tool check result]')

    def test_run_turn_passes_the_adapter_thinking_flag(self, monkeypatch):
        seen: list = []
        monkeypatch.setattr(anth, 'to_anthropic_messages',
                            lambda msgs, thinking=False: seen.append(
                                thinking) or ('', []))
        monkeypatch.setattr(anth.tools, 'active_specs', lambda: [])
        monkeypatch.setattr(anth.turn, 'run_loop', lambda **kw: None)
        for flag in (True, False):
            a = anth.AnthropicAdapter(thinking=flag)
            monkeypatch.setattr(a, '_client', lambda: object())
            a.run_turn()
        assert seen == [True, False]

    def test_mixed_provider_ids_rebuild_on_either_side(self) -> None:
        """Ids are opaque: an Anthropic ``toolu_`` round replays through
        the LiteLLM translation and an OpenAI ``call_`` round through the
        Anthropic one, both rebuilt (not crashed, not flattened)."""
        anthropic_history = self.ROUND + [
            {'role': 'assistant', 'content': 'done'},
            {'role': 'user', 'content': 'more'}]
        out = lite.to_openai_messages(anthropic_history)
        assert out[1]['tool_calls'][0]['id'] == 'toolu_1'
        assert out[2] == {'role': 'tool', 'tool_call_id': 'toolu_1',
                          'content': 'Waiting for agent1'}
        native = lite.native_round(anthropic_history, 1)
        assert native is not None and native[1] == 3
        openai_history = [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call_abc', 'raw_arguments': '{"path": "a.py"}',
                 'function': {'name': 'read_file',
                              'arguments': {'path': 'a.py'}}}]},
            {'role': 'tool', 'tool_name': 'read_file',
             'tool_call_id': 'call_abc', 'content': 'A'},
            {'role': 'assistant', 'content': 'done'},
            {'role': 'user', 'content': 'more'}]
        for thinking in (False, True):
            _, out = anth.to_anthropic_messages(openai_history,
                                                thinking=thinking)
            assert out[1]['content'] == [
                {'type': 'tool_use', 'id': 'call_abc', 'name': 'read_file',
                 'input': {'path': 'a.py'}}]
            assert out[2]['content'] == [
                {'type': 'tool_result', 'tool_use_id': 'call_abc',
                 'content': 'A'}]
            assert out[3:] == [{'role': 'assistant', 'content': 'done'},
                               {'role': 'user', 'content': 'more'}]
