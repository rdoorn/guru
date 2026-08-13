"""Unit tests for guru's pure helper functions across the package."""
import json
from pathlib import Path

from guru import config, session, ui
from guru.adapters import anthropic as anth
from guru.adapters.ollama import OllamaAdapter
from guru.domain import conversation, tools


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
