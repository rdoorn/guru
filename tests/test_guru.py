"""Unit tests for guru's pure helper functions."""
import json
import os
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
    """Tests for _project_memory_dir path encoding."""

    def test_encodes_cwd_with_dashes(self) -> None:
        result = guru._project_memory_dir()
        expected_name = str(Path.cwd()).replace(os.sep, '-')
        assert result.name == expected_name
        assert result.parent == guru.GURU_HOME


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
        monkeypatch.setattr(guru, 'LOCAL_GURU_MD', local_md)

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
        monkeypatch.setattr(guru, 'LOCAL_GURU_MD', tmp_path / 'nope2.md')

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
