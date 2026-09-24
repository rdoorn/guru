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


class TestOverReadGuard:
    """turn._drive: a delegation-capable non-controller that reads
    ``config.OVER_READ_LIMIT`` distinct paths in one turn without spawning
    is nudged to delegate at once (once per turn), and the struggle
    counter ``over_read`` records it."""

    def _quiet(self, monkeypatch, messages=None, *, can_spawn=True,
               controller=False, limit=8) -> None:
        from guru.adapters import turn
        monkeypatch.setattr(ui, 'note_thinking', lambda: None)
        monkeypatch.setattr(ui, 'status_draw', lambda: None)
        monkeypatch.setattr(turn, '_render_answer', lambda c: None)
        monkeypatch.setattr(ui.console, 'print', lambda *a, **k: None)
        monkeypatch.setattr(session, 'messages', messages if messages
                            is not None else [
                                {'role': 'user',
                                 'content': 'review the whole service'}])
        monkeypatch.setattr(session, 'cancel_requested', False)
        monkeypatch.setattr(session, 'can_spawn', can_spawn)
        monkeypatch.setattr(session, 'controller', controller)
        monkeypatch.setattr(session, 'struggle', {'over_read': 0})
        monkeypatch.setattr(config, 'OVER_READ_LIMIT', limit)
        monkeypatch.setattr(config, 'DELEGATION_NUDGE_MIN_READS', 0)

    def _run(self, rounds: list) -> list:
        """Drive scripted ``rounds`` (each a list of read paths, or a
        ``'spawn'`` marker); every tool round threads tool messages into
        ``session.messages`` as an adapter would. Returns the nudges."""
        from guru.adapters import turn
        seq = iter(rounds + [("Done.", [])])

        def step():
            item = next(seq)
            if isinstance(item, tuple):
                return item
            calls = []
            for i, path in enumerate(item):
                if path == 'spawn':
                    calls.append(('spawn', {'task': 't'}, f'r{i}'))
                else:
                    calls.append(('read_file', {'path': path}, f'r{i}'))
            return ("", calls)

        def run_tools(pending):
            for name, args, _ref, _dup in pending:
                session.messages.append(
                    {'role': 'tool', 'tool_name': name, 'content': 'x',
                     'tool_args': dict(args)})
        nudges: list = []
        turn.run_loop(step=step, run_tools=run_tools,
                      add_user=lambda t: nudges.append(t))
        return nudges

    def _paths(self, n, start=0):
        return [f'app/m{i}.py' for i in range(start, start + n)]

    def test_nudges_at_the_limit_and_continues(self, monkeypatch) -> None:
        self._quiet(monkeypatch)
        nudges = self._run([self._paths(5), self._paths(3, 5),
                            self._paths(2, 8)])
        assert len(nudges) == 1
        assert nudges[0].startswith(
            'You have read 8 files without delegating. ')
        assert 'decompose' in nudges[0].lower()
        assert session.struggle['over_read'] == 1

    def test_under_the_limit_is_silent(self, monkeypatch) -> None:
        self._quiet(monkeypatch)
        assert self._run([self._paths(7)]) == []
        assert session.struggle['over_read'] == 0

    def test_only_once_per_turn(self, monkeypatch) -> None:
        self._quiet(monkeypatch)
        nudges = self._run([self._paths(8), self._paths(8, 8),
                            self._paths(8, 16)])
        assert len(nudges) == 1

    def test_same_path_counts_once(self, monkeypatch) -> None:
        self._quiet(monkeypatch)
        assert self._run([['a.py'] * 5, ['a.py', 'b.py'] * 4]) == []

    def test_controller_and_subagent_are_never_nudged(self, monkeypatch):
        self._quiet(monkeypatch, controller=True)
        assert self._run([self._paths(9)]) == []
        self._quiet(monkeypatch, can_spawn=False)
        assert self._run([self._paths(9)]) == []

    def test_spawn_in_the_turn_disarms_it(self, monkeypatch) -> None:
        self._quiet(monkeypatch)
        assert self._run([['spawn'], self._paths(9)]) == []

    def test_reads_from_earlier_turns_do_not_count(self, monkeypatch):
        earlier = [{'role': 'user', 'content': 'first question'}] + [
            {'role': 'tool', 'tool_name': 'read_file', 'content': 'x',
             'tool_args': {'path': p}} for p in self._paths(6, 100)] + [
            {'role': 'assistant', 'content': 'answered'},
            {'role': 'user', 'content': 'review the whole service'}]
        self._quiet(monkeypatch, messages=list(earlier))
        assert self._run([self._paths(5)]) == []         # 11 overall
        # a new turn: the 5 (and the 6) above are history
        session.messages += [
            {'role': 'assistant', 'content': 'answered again'},
            {'role': 'user', 'content': 'and the tests?'}]
        assert self._run([self._paths(3, 5)]) == []      # 14 overall, 3 now
        session.messages += [
            {'role': 'assistant', 'content': 'answered once more'},
            {'role': 'user', 'content': 'review it all'}]
        assert len(self._run([self._paths(8, 200)])) == 1

    def test_zero_limit_disables(self, monkeypatch) -> None:
        self._quiet(monkeypatch, limit=0)
        assert self._run([self._paths(12)]) == []

    def test_nudge_is_not_mistaken_for_the_request(self, monkeypatch):
        from guru.adapters import turn
        self._quiet(monkeypatch)
        seen: list = []
        monkeypatch.setattr(turn, '_render_answer', lambda c: None)
        orig_add = seen.append

        def add_user(t):
            session.messages.append({'role': 'user', 'content': t})
            orig_add(t)
        seq = iter([("", [('read_file', {'path': p}, p)
                          for p in self._paths(8)]), ("Done.", [])])

        def run_tools(pending):
            for name, args, _ref, _dup in pending:
                session.messages.append(
                    {'role': 'tool', 'tool_name': name, 'content': 'x',
                     'tool_args': dict(args)})
        turn.run_loop(step=lambda: next(seq), run_tools=run_tools,
                      add_user=add_user)
        assert len(seen) == 1
        assert turn._turn_request() == 'review the whole service'

    def test_over_read_is_a_struggle_key(self) -> None:
        assert 'over_read' in session.STRUGGLE_KEYS


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

    def test_long_answer_after_spawn_is_fine(self, monkeypatch, fake_repo):
        row = self._run(monkeypatch, fake_repo, [
            ("", [("spawn", {"task": "t"}, "r1")]),
            ("", [("join", {"targets": "agent1"}, "r2")]),
            ("x" * 601, [])])
        assert row['controller_executed'] is False
        assert row['tasks_spawned'] == 1

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

    def test_turn_start_clears_last_error(self, monkeypatch, fake_repo):
        monkeypatch.setattr(session, 'last_error', 'old failure')
        self._run(monkeypatch, fake_repo, [("ok.", [])])
        assert session.last_error == ''

    def test_non_controller_never_flips(self, monkeypatch, fake_repo):
        row = self._run(monkeypatch, fake_repo, [
            ("", [("read_file", {"path": "x"}, "r1")]), ("x" * 601, [])],
            controller=False)
        assert row['controller_executed'] is False
