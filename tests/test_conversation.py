"""Tests for guru.domain.conversation (grouping, retention, context)."""
import json
from pathlib import Path

from guru import config, session, skills
from guru.domain import conversation, files


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


class TestContextBreakdown:
    """conversation.context_breakdown categorises resident context tokens."""

    def test_categorises_and_totals(self) -> None:
        msgs = [
            {'role': 'system', 'content': 'x' * 40},    # 10 tokens
            {'role': 'user', 'content': 'y' * 20},      # 5 tokens
            {'role': 'assistant', 'content': 'z' * 8},  # 2 tokens
        ]
        bd = conversation.context_breakdown(msgs, set(), False)
        assert bd['sys'] == 10 and bd['in'] == 5 and bd['out'] == 2
        assert bd['tools'] > 0        # search_tools schema always present
        assert bd['total'] == (bd['sys'] + bd['in'] + bd['out']
                               + bd['toolout'] + bd['tools'])

    def test_can_spawn_adds_schema_tokens(self) -> None:
        base = conversation.context_breakdown([], set(), False)['tools']
        more = conversation.context_breakdown([], set(), True)['tools']
        assert more > base            # spawn/check/join schemas add tokens


class TestApplyRetention:
    def _msgs(self):
        return [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'find the k8s setting'},
            {'role': 'assistant', 'content': ''},        # text-less step
            {'role': 'tool', 'tool_name': 'web_fetch',
             'content': 'X' * 9000},                      # large -> summarize
            {'role': 'tool', 'tool_name': 'search_code',
             'content': 'a.py:1: hit'},                   # keep
            {'role': 'assistant', 'content': 'the answer'},
        ]

    def test_large_web_summarized_small_kept(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'WEB_SUMMARIZE_OVER_CHARS', 6000)

        class _Ad:
            def summarise(self, t):
                return 'GIST'
        monkeypatch.setattr(session, 'adapter', _Ad())
        msgs = self._msgs()
        conversation.apply_retention(msgs)
        assert all(not (m.get('role') == 'assistant'
                        and not m.get('content')) for m in msgs)
        web = next(m for m in msgs if m.get('tool_name') == 'web_fetch')
        assert 'GIST' in web['content'] and len(web['content']) < 9000
        grep = next(m for m in msgs if m.get('tool_name') == 'search_code')
        assert grep['content'] == 'a.py:1: hit'

    def test_small_web_not_summarized(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'WEB_SUMMARIZE_OVER_CHARS', 6000)

        class _Ad:
            def summarise(self, t):
                raise AssertionError('should not summarise small output')
        monkeypatch.setattr(session, 'adapter', _Ad())
        msgs = [
            {'role': 'user', 'content': 'q'},
            {'role': 'tool', 'tool_name': 'web_fetch', 'content': 'short'},
        ]
        conversation.apply_retention(msgs)
        assert msgs[-1]['content'] == 'short'

    def test_large_read_file_outlined(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'OUTLINE_FILE_OVER_CHARS', 50)
        src = "def foo():\n    return 1\n" * 10
        header = "/tmp/m.py (lines 1-20 of 20, sha:abc):"
        body = "\n".join(f"{i:>6}\t{ln}"
                         for i, ln in enumerate(src.splitlines(), 1))
        msgs = [
            {'role': 'user', 'content': 'q'},
            {'role': 'tool', 'tool_name': 'read_file',
             'content': f"{header}\n{body}"},
        ]
        conversation.apply_retention(msgs)
        assert '[outline]' in msgs[-1]['content']
        assert 'return 1' not in msgs[-1]['content']


class TestFileShaLedger:
    """The file->sha ledger + its rendering into the system prompt."""

    def _write_allowed(self, monkeypatch, *dirs):
        monkeypatch.setattr(
            config, 'ALLOWED_WRITE_DIRS',
            {str(Path(d).resolve()) for d in dirs})
        monkeypatch.setattr(config, 'persist_write_dir', lambda d: None)

    def test_read_write_edit_populate_ledger(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(session, 'file_shas', {})
        self._write_allowed(monkeypatch, tmp_path)
        monkeypatch.setattr(
            config, 'ALLOWED_READ_DIRS', {str(tmp_path.resolve())})
        p = tmp_path / 'c.py'
        key = str(p.resolve())
        files.write_file(str(p), 'hello\n')
        assert session.file_shas[key] == files._sha('hello\n')
        files.read_file(str(p))
        assert session.file_shas[key] == files._sha('hello\n')
        files.edit_file(str(p), 'hello', 'bye', session.file_shas[key])
        assert session.file_shas[key] == files._sha('bye\n')

    def test_delete_forgets_sha(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(session, 'file_shas', {})
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'd.txt'
        files.write_file(str(p), 'bye')
        assert str(p.resolve()) in session.file_shas
        files.delete_file(str(p))
        assert str(p.resolve()) not in session.file_shas

    def test_ledger_caps_and_drops_oldest(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'file_shas', {})
        cap = files._SHA_LEDGER_CAP
        for i in range(cap + 3):
            files._remember_sha(Path(f'/tmp/f{i}'), f'sha{i}')
        assert len(session.file_shas) == cap
        assert '/tmp/f0' not in session.file_shas
        assert session.file_shas[f'/tmp/f{cap + 2}'] == f'sha{cap + 2}'

    def test_remember_moves_key_to_newest(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'file_shas', {})
        files._remember_sha(Path('/tmp/a'), '1')
        files._remember_sha(Path('/tmp/b'), '2')
        files._remember_sha(Path('/tmp/a'), '3')     # re-touch a
        assert list(session.file_shas) == ['/tmp/b', '/tmp/a']
        assert session.file_shas['/tmp/a'] == '3'

    def test_refresh_adds_block_and_is_idempotent(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'file_shas', {'/tmp/x.txt': 'abc123'})
        monkeypatch.setattr(
            session, 'messages',
            [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        body = session.messages[0]['content']
        assert body.startswith('BASE')
        assert '[open files]' in body and 'abc123' in body
        conversation.refresh_system_context()             # again
        assert session.messages[0]['content'].count('[open files]') == 1

    def test_refresh_strips_block_when_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'file_shas', {'/tmp/x.txt': 'abc123'})
        monkeypatch.setattr(
            session, 'messages',
            [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        session.file_shas.clear()
        conversation.refresh_system_context()
        assert session.messages[0]['content'] == 'BASE'

    def test_refresh_shows_relative_path(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        key = str((tmp_path / 'sub' / 'f.py').resolve())
        monkeypatch.setattr(session, 'file_shas', {key: 'deadbeef'})
        monkeypatch.setattr(
            session, 'messages',
            [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        body = session.messages[0]['content']
        assert 'sub/f.py (sha:deadbeef)' in body

    def test_refresh_noop_without_system_message(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'file_shas', {'/tmp/x': 'y'})
        monkeypatch.setattr(session, 'messages', [])
        conversation.refresh_system_context()             # must not raise
        assert session.messages == []


class TestSystemContext:
    """Catalog + role + skill overlays rendered onto messages[0]."""

    def _reg(self):
        return {
            'developer': skills.SkillEntry(
                'developer', 'role', 'dev', '', 'BE A DEV'),
            'code-review': skills.SkillEntry(
                'code-review', 'skill', 'review', '', 'REVIEW METHOD'),
        }

    def test_catalog_rendered_when_registry_present(
            self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(session, 'active_role', None)
        monkeypatch.setattr(session, 'active_skill', None)
        monkeypatch.setattr(
            session, 'messages', [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        body = session.messages[0]['content']
        assert body.startswith('BASE')
        assert 'Available specialists' in body and 'developer' in body

    def test_role_and_skill_bodies_rendered(self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(session, 'active_role', 'developer')
        monkeypatch.setattr(session, 'active_skill', 'code-review')
        monkeypatch.setattr(
            session, 'messages', [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        body = session.messages[0]['content']
        assert '[role: developer]' in body and 'BE A DEV' in body
        assert '[skill: code-review]' in body and 'REVIEW METHOD' in body

    def test_idempotent_across_calls(self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(session, 'active_role', 'developer')
        monkeypatch.setattr(session, 'active_skill', None)
        monkeypatch.setattr(
            session, 'messages', [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        conversation.refresh_system_context()
        body = session.messages[0]['content']
        assert body.count('[role: developer]') == 1
        assert body.startswith('BASE')

    def test_unknown_active_name_ignored(self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(session, 'active_role', 'nonexistent')
        monkeypatch.setattr(session, 'active_skill', None)
        monkeypatch.setattr(
            session, 'messages', [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()      # must not raise
        assert '[role:' not in session.messages[0]['content']


class TestFocusedSummary:
    def test_recent_question_scans_backwards(self) -> None:
        msgs = [
            {'role': 'system', 'content': 's'},
            {'role': 'user', 'content': 'first'},
            {'role': 'assistant', 'content': 'a'},
            {'role': 'user', 'content': 'the question'},
            {'role': 'tool', 'tool_name': 'web_fetch', 'content': 'x'},
        ]
        assert conversation._recent_question(msgs, 4) == 'the question'

    def test_summarize_relevant_uses_adapter(self, monkeypatch) -> None:
        seen = {}

        class _Ad:
            def summarise(self, transcript):
                seen['t'] = transcript
                return 'RELEVANT BITS'
        monkeypatch.setattr(session, 'adapter', _Ad())
        out = conversation._summarize_relevant(
            'k8s setting?', 'huge content', 'web_fetch')
        assert 'RELEVANT BITS' in out
        assert 'web_fetch summary' in out and 'k8s setting?' in out
        assert 'huge content' in seen['t'] and 'k8s setting?' in seen['t']

    def test_summarize_relevant_falls_back_on_error(
            self, monkeypatch) -> None:
        class _Ad:
            def summarise(self, transcript):
                raise RuntimeError('down')
        monkeypatch.setattr(session, 'adapter', _Ad())
        out = conversation._summarize_relevant('q', 'A' * 100, 'web_fetch')
        assert 'truncated' in out and out.count('A') > 0


class TestOutlineCode:
    def _read_output(self, path, source):
        lines = source.splitlines()
        header = f"{path} (lines 1-{len(lines)} of {len(lines)}, sha:abc):"
        body = "\n".join(f"{i:>6}\t{ln}" for i, ln in enumerate(lines, 1))
        return f"{header}\n{body}"

    def test_outline_python_keeps_signatures_drops_bodies(self) -> None:
        src = (
            "import os\n"
            "\n"
            "def foo(a, b=2) -> int:\n"
            "    '''Add things.'''\n"
            "    return a + b\n"
            "\n"
            "class C:\n"
            "    def method(self, x):\n"
            "        '''Do it.'''\n"
            "        return x\n")
        out = conversation._outline_code(
            self._read_output('/tmp/m.py', src))
        assert 'def foo(a, b=2) -> int:' in out
        assert 'Add things.' in out
        assert 'class C' in out
        assert 'def method(self, x):' in out
        assert 'return a + b' not in out      # body dropped
        assert '/tmp/m.py' in out             # header kept

    def test_outline_non_python_truncates(self) -> None:
        src = "\n".join(f"line {i}" for i in range(200))
        out = conversation._outline_code(
            self._read_output('/tmp/notes.txt', src))
        assert '/tmp/notes.txt' in out
        assert len(out) < len(src)            # shrunk

    def test_outline_unparseable_python_falls_back(self) -> None:
        src = "def broken(:\n    pass\nimport sys\n"
        out = conversation._outline_code(
            self._read_output('/tmp/b.py', src))
        # regex fallback keeps def/import lines even when AST fails
        assert 'import sys' in out or 'def broken' in out
