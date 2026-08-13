"""Unit tests for guru's pure helper functions."""
import json
from pathlib import Path

import guru


class TestDomainOf:
    """Tests for _domain_of."""

    def test_strips_scheme_and_lowercases(self) -> None:
        assert guru._domain_of('https://Example.COM/path') == 'example.com'

    def test_strips_port(self) -> None:
        assert guru._domain_of('http://example.com:8443/x') == 'example.com'

    def test_bare_host_without_scheme(self) -> None:
        assert guru._domain_of('example.com/path') == 'example.com'

    def test_subdomain_preserved(self) -> None:
        assert guru._domain_of('https://api.github.com') == 'api.github.com'


class TestProjectMemoryDir:
    """Tests for _project_memory_dir location."""

    def test_points_into_project_guru_dir(self) -> None:
        result = guru._project_memory_dir()
        assert result == Path.cwd() / '.guru' / 'memory'


class TestBuildSystemPrompt:
    """Tests for _build_system_prompt assembly."""

    def test_appends_global_and_local(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        global_md = tmp_path / 'GURU.md'
        local_md = tmp_path / '.GURU.md'
        global_md.write_text('GLOBAL RULES', encoding='utf-8')
        local_md.write_text('LOCAL RULES', encoding='utf-8')
        monkeypatch.setattr(guru, 'GURU_MD_PATH', global_md)
        monkeypatch.setattr(guru, 'PROJECT_GURU_MD', local_md)

        prompt = guru._build_system_prompt()

        assert guru.SYSTEM_PROMPT.strip() in prompt
        assert 'GLOBAL RULES' in prompt
        assert 'LOCAL RULES' in prompt
        # Built-in prompt comes first, then global, then local.
        assert prompt.index('GLOBAL RULES') < prompt.index('LOCAL RULES')

    def test_missing_files_are_skipped(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(guru, 'GURU_MD_PATH', tmp_path / 'nope.md')
        monkeypatch.setattr(guru, 'PROJECT_GURU_MD', tmp_path / 'nope2.md')

        prompt = guru._build_system_prompt()

        assert prompt == guru.SYSTEM_PROMPT.strip()


class TestMessageToDict:
    """Tests for _message_to_dict normalisation."""

    def test_plain_dict_user_message(self) -> None:
        msg = {'role': 'user', 'content': 'hello'}
        assert guru._message_to_dict(msg) == {
            'role': 'user', 'content': 'hello'
        }

    def test_tool_message_keeps_tool_name(self) -> None:
        msg = {'role': 'tool', 'tool_name': 'web_fetch', 'content': 'x'}
        out = guru._message_to_dict(msg)
        assert out['tool_name'] == 'web_fetch'

    def test_none_content_becomes_empty_string(self) -> None:
        msg = {'role': 'assistant', 'content': None}
        assert guru._message_to_dict(msg)['content'] == ''

    def test_pydantic_like_object_via_model_dump(self) -> None:
        class FakeMessage:
            def model_dump(self) -> dict:
                return {'role': 'assistant', 'content': 'hi'}

        assert guru._message_to_dict(FakeMessage()) == {
            'role': 'assistant', 'content': 'hi'
        }


class TestFirstUserMessage:
    """Tests for _first_user_message title extraction."""

    def test_returns_first_user_content(self, tmp_path: Path) -> None:
        path = tmp_path / 'c.memory'
        path.write_text(
            json.dumps([
                {'role': 'assistant', 'content': 'ignored'},
                {'role': 'user', 'content': 'the question'},
            ]),
            encoding='utf-8',
        )
        assert guru._first_user_message(path) == 'the question'

    def test_truncates_long_message(self, tmp_path: Path) -> None:
        path = tmp_path / 'c.memory'
        long_text = 'x' * 100
        path.write_text(
            json.dumps([{'role': 'user', 'content': long_text}]),
            encoding='utf-8',
        )
        result = guru._first_user_message(path)
        assert result.endswith('…')
        assert len(result) == 61  # 60 chars + ellipsis

    def test_unreadable_file(self, tmp_path: Path) -> None:
        path = tmp_path / 'missing.memory'
        assert guru._first_user_message(path) == '(unreadable)'


class TestMatchTools:
    """Tests for the pre-existing _match_tools ranking."""

    def test_github_query_ranks_github_tool_first(self) -> None:
        ranked = guru._match_tools('get latest github release version')
        assert ranked[0] == 'fetch_github_releases'

    def test_fetch_query_ranks_web_fetch_first(self) -> None:
        ranked = guru._match_tools('fetch a webpage url')
        assert ranked[0] == 'web_fetch'


class _FakeInfo:
    def __init__(self, modelinfo: dict, parameters: str) -> None:
        self.modelinfo = modelinfo
        self.parameters = parameters


class TestResolveContextWindow:
    """Tests for _resolve_context_window."""

    def test_uses_modelfile_num_ctx_capped_at_ceiling(
        self, monkeypatch
    ) -> None:
        info = _FakeInfo(
            {'general.architecture': 'qwen3', 'qwen3.context_length': 40960},
            'num_ctx 32768\ntemperature 0.6',
        )
        monkeypatch.setattr(guru.ollama, 'show', lambda m: info)
        monkeypatch.setattr(guru._args, 'num_ctx', 0)
        num_ctx, ceiling = guru._resolve_context_window('m')
        assert num_ctx == 32768
        assert ceiling == 40960

    def test_defaults_when_modelfile_has_no_num_ctx(self, monkeypatch) -> None:
        info = _FakeInfo(
            {'general.architecture': 'llama', 'llama.context_length': 8192},
            'temperature 0.7',
        )
        monkeypatch.setattr(guru.ollama, 'show', lambda m: info)
        monkeypatch.setattr(guru._args, 'num_ctx', 0)
        num_ctx, ceiling = guru._resolve_context_window('m')
        assert num_ctx == guru.DEFAULT_NUM_CTX
        assert ceiling == 8192

    def test_cli_override_wins_but_is_capped(self, monkeypatch) -> None:
        info = _FakeInfo(
            {'general.architecture': 'qwen3', 'qwen3.context_length': 40960},
            'num_ctx 8192',
        )
        monkeypatch.setattr(guru.ollama, 'show', lambda m: info)
        monkeypatch.setattr(guru._args, 'num_ctx', 100000)
        num_ctx, _ = guru._resolve_context_window('m')
        assert num_ctx == 40960  # capped at ceiling

    def test_show_failure_falls_back(self, monkeypatch) -> None:
        def _boom(m: str) -> None:
            raise RuntimeError('no server')

        monkeypatch.setattr(guru.ollama, 'show', _boom)
        monkeypatch.setattr(guru._args, 'num_ctx', 0)
        num_ctx, ceiling = guru._resolve_context_window('m')
        assert num_ctx == guru.DEFAULT_NUM_CTX
        assert ceiling == 0


class TestGroupMessages:
    """Tests for _group_messages turn-grouping."""

    def test_splits_at_each_user_message(self) -> None:
        msgs = [
            {'role': 'user', 'content': 'q1'},
            {'role': 'assistant', 'content': 'a1'},
            {'role': 'tool', 'content': 't1'},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 'a2'},
        ]
        groups = guru._group_messages(msgs)
        assert len(groups) == 2
        assert len(groups[0]) == 3
        assert len(groups[1]) == 2


class TestEstimateTokens:
    """Tests for _estimate_tokens."""

    def test_four_chars_per_token(self) -> None:
        msgs = [{'role': 'user', 'content': 'x' * 40}]
        assert guru._estimate_tokens(msgs) == 10


class TestCompactMessages:
    """Tests for _compact_messages eviction path (no model call)."""

    def test_evicts_old_tool_output_without_summary(self, monkeypatch) -> None:
        def _no_chat(*a, **k) -> None:
            raise AssertionError('summary should not be called')

        monkeypatch.setattr(guru.ollama, 'chat', _no_chat)
        monkeypatch.setattr(guru, 'KEEP_RECENT_GROUPS', 1)
        monkeypatch.setattr(guru, '_NUM_CTX', 100)  # limit = 85 tokens
        convo = [
            {'role': 'system', 'content': 'SYS'},
            {'role': 'user', 'content': 'q1'},
            {'role': 'assistant', 'content': 'a1'},
            {'role': 'tool', 'tool_name': 'web_fetch', 'content': 'X' * 2000},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 'a2'},
        ]
        monkeypatch.setattr(guru, 'messages', convo)
        guru._compact_messages(force=False)

        result = guru.messages
        assert result[0]['content'] == 'SYS'
        tool_msg = next(m for m in result if m.get('role') == 'tool')
        assert 'evicted' in tool_msg['content']
        # Recent group preserved verbatim.
        assert result[-1]['content'] == 'a2'


class TestStatusParts:
    """Tests for _status_parts colour thresholds and segments."""

    def test_green_yellow_red(self, monkeypatch) -> None:
        monkeypatch.setattr(guru, '_NUM_CTX', 1000)
        monkeypatch.setattr(guru, 'MODEL', 'demo:latest')

        monkeypatch.setattr(guru, '_ctx_used', 100)
        assert guru._status_parts()[3] == 'green'

        monkeypatch.setattr(guru, '_ctx_used', 780)
        left, ctx_segment, right, colour = guru._status_parts()
        assert colour == 'yellow'
        assert '78%' in ctx_segment

        monkeypatch.setattr(guru, '_ctx_used', 900)
        assert guru._status_parts()[3] == 'red'

    def test_segments_include_expected_fields(self, monkeypatch) -> None:
        monkeypatch.setattr(guru, '_NUM_CTX', 1000)
        monkeypatch.setattr(guru, '_ctx_used', 100)
        monkeypatch.setattr(guru, 'MODEL', 'demo:latest')
        monkeypatch.setattr(guru, '_MODEL_SIZE', '32B')
        monkeypatch.setattr(guru, '_session_in', 1234)
        monkeypatch.setattr(guru, '_session_out', 56)
        monkeypatch.setattr(guru, '_git_branch_value', 'main')

        left, ctx_segment, right, _ = guru._status_parts()
        assert 'demo' in left and '32B' in left
        assert '🧠' in ctx_segment
        assert '1234' in right and '56' in right and 'main' in right
