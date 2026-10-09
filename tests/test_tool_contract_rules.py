"""Structural round, Package C items 2 and 3: argument validation from the
registry schema (one corrective line, nothing runs) and the per-task-kind
tool policy (a review task is read-only at the tool layer)."""
import pytest

from guru import config, session, ui
from guru.domain import files, routing, toolpolicy, tools


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
    monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
    monkeypatch.setattr(session, 'task_kind', '')
    yield
    tools.set_policy(None)


class TestSpecs:
    """Every callable tool has a spec the validator can read."""

    def test_every_registry_tool_has_an_example(self) -> None:
        for name, info in tools.TOOL_REGISTRY.items():
            spec = tools.tool_spec(name)
            assert spec['name'] == name
            assert 'example' in info, name
            for key in info['example']:
                assert key in info['parameters'], (name, key)
            for key in info.get('types', {}):
                assert key in info['parameters'], (name, key)
                assert info['types'][key] in tools._TYPE_NAMES
            for key in info.get('enum', {}):
                assert key in info['parameters'], (name, key)

    def test_always_on_tools_have_specs(self) -> None:
        for name in toolpolicy.ALWAYS_ON_TOOLS:
            spec = tools.tool_spec(name)
            assert spec and spec['parameters'], name
        assert toolpolicy.ALWAYS_ON_TOOLS == {
            'search_tools', 'use_skill', 'spawn', 'check', 'join',
            'apply_work'}

    def test_always_on_tools_validate_like_the_rest(self) -> None:
        assert tools.validate_arguments('apply_work', {'worker': 'a2'}) == \
            ({'worker': 'a2'}, '')
        _args, error = tools.validate_arguments('apply_work', {'task': 'a2'})
        assert error.startswith(tools.INVALID_ARGS_PREFIX)
        assert 'apply_work(worker: str)' in error
        for gone in ('plan', 'final_answer'):
            assert tools.tool_spec(gone) == {}

    def test_always_on_tools_are_never_policy_gated(self) -> None:
        toolpolicy.set_policy(toolpolicy.ToolsPolicy(enabled={'read_file'}))
        try:
            assert toolpolicy.is_enabled('apply_work')
            assert toolpolicy.is_enabled('spawn')
            assert not toolpolicy.is_enabled('edit_file')
        finally:
            toolpolicy.set_policy(None)
        assert tools.tool_spec('spawn')['enum']['kind'] == list(routing.KINDS)
        assert tools.tool_spec('nope') == {}

    def test_signature_and_example_text(self) -> None:
        spec = tools.tool_spec('read_file')
        assert tools.signature_text(spec) == \
            'read_file(path: str, lines?: str)'
        assert tools.example_text(spec) == \
            "read_file(path='src/app.py', lines='40-80')"
        spec = tools.tool_spec('sandbox_run')
        assert tools.signature_text(spec) == \
            'sandbox_run(argv: list, detail?: bool)'
        assert 'argv=["pytest", "-q"]' in tools.example_text(spec)


class TestValidateArguments:
    def _err(self, name, args) -> str:
        clean, error = tools.validate_arguments(name, args)
        assert clean is args
        assert error.startswith(tools.INVALID_ARGS_PREFIX)
        assert '\n' not in error
        assert f'Expected {name}(' in error and 'e.g. ' in error
        return error

    def test_valid_arguments_pass_through(self) -> None:
        clean, error = tools.validate_arguments(
            'read_file', {'path': 'a.py', 'lines': '1-5'})
        assert (clean, error) == ({'path': 'a.py', 'lines': '1-5'}, '')

    def test_missing_required(self) -> None:
        assert "missing required 'path'" in self._err('read_file', {})
        err = self._err('edit_file', {'path': 'a', 'old': 'x'})
        assert "'new', 'sha'" in err

    def test_unknown_parameter(self) -> None:
        err = self._err('read_file', {'path': 'a', 'file': 'b'})
        assert "no parameter 'file' (accepted: path, lines)" in err

    def test_not_an_object(self) -> None:
        err = self._err('read_file', 'a.py')
        assert 'takes an object of named parameters, got str' in err

    def test_int_is_coerced_or_refused(self) -> None:
        clean, _ = tools.validate_arguments('list_tree', {'depth': '3'})
        assert clean == {'depth': 3}
        clean, _ = tools.validate_arguments('run_tests', {'maxfail': 2})
        assert clean == {'maxfail': 2}
        err = self._err('list_tree', {'depth': 'deep'})
        assert "'depth' must be an integer, got 'deep'" in err
        assert "'depth' must be an integer" in self._err(
            'list_tree', {'depth': True})

    def test_bool_is_coerced_or_refused(self) -> None:
        for word, value in (('true', True), ('False', False), ('', False),
                            (True, True)):
            clean, _ = tools.validate_arguments('git_diff', {'detail': word})
            assert clean == {'detail': value}, word
        assert "'detail' must be true or false" in self._err(
            'git_diff', {'detail': 'maybe'})

    def test_list_accepts_list_or_json_list_only(self) -> None:
        clean, _ = tools.validate_arguments(
            'sandbox_run', {'argv': ['pytest', '-q']})
        assert clean == {'argv': ['pytest', '-q']}
        clean, _ = tools.validate_arguments(
            'sandbox_run', {'argv': '["python", "-m", "pytest"]'})
        assert clean == {'argv': ['python', '-m', 'pytest']}
        err = self._err('sandbox_run', {'argv': 'pytest -q'})
        assert 'list of strings (not a command string)' in err
        assert 'JSON list' in self._err('sandbox_run', {'argv': '[oops'})

    def test_enum_is_case_insensitive_and_canonicalised(self) -> None:
        clean, _ = tools.validate_arguments(
            'find_symbol', {'name': 'f', 'kind': 'DEF'})
        assert clean == {'name': 'f', 'kind': 'def'}
        err = self._err('find_symbol', {'name': 'f', 'kind': 'class'})
        assert "'kind' must be one of 'def', 'ref', got 'class'" in err
        clean, _ = tools.validate_arguments(
            'spawn', {'task': 't', 'kind': 'Review', 'complexity': 'HARD'})
        assert clean == {'task': 't', 'kind': 'review', 'complexity': 'hard'}
        assert "'kind' must be one of 'debug'" in self._err(
            'spawn', {'task': 't', 'kind': 'weird'})

    def test_none_for_optional_is_dropped(self) -> None:
        clean, error = tools.validate_arguments(
            'read_file', {'path': 'a', 'lines': None})
        assert (clean, error) == ({'path': 'a'}, '')
        assert "'path' must be a string, got None" in self._err(
            'read_file', {'path': None})

    def test_scalars_become_strings(self) -> None:
        clean, _ = tools.validate_arguments('read_file', {'path': 12})
        assert clean == {'path': '12'}
        assert "'path' must be a string" in self._err(
            'read_file', {'path': ['a']})

    def test_unknown_tool_is_left_to_execute_tool(self) -> None:
        assert tools.validate_arguments('nope', {'x': 1}) == ({'x': 1}, '')


class TestExecuteToolValidation:
    """The validator runs inside execute_tool: an invalid call returns the
    corrective line, runs nothing, counts as a tool error and audits."""

    def test_invalid_call_does_not_run(self, monkeypatch, fake_repo) -> None:
        calls: list = []
        monkeypatch.setitem(tools.TOOL_REGISTRY, 'read_file', {
            **tools.TOOL_REGISTRY['read_file'],
            'fn': lambda **kw: calls.append(kw) or 'read'})
        out = tools.execute_tool('read_file', {'file': 'a.py'})
        assert out.startswith(tools.INVALID_ARGS_PREFIX)
        assert calls == []
        assert session.struggle['tool_errors'] == 1
        from guru.domain import ledger
        ledger.flush()
        row = fake_repo.stream('tool_events')[-1]
        assert row['ok'] is False and row['denied'] == ''

    def test_coerced_arguments_reach_the_tool(self, monkeypatch) -> None:
        seen: dict = {}
        monkeypatch.setitem(tools.TOOL_REGISTRY, 'list_tree', {
            **tools.TOOL_REGISTRY['list_tree'],
            'fn': lambda **kw: seen.update(kw) or 'tree'})
        assert tools.execute_tool('list_tree', {'depth': '2'}) == 'tree'
        assert seen == {'depth': 2}

    def test_policy_refusal_wins_over_schema(self) -> None:
        tools.set_policy(tools.ToolsPolicy(disabled={'read_file'}))
        out = tools.execute_tool('read_file', {})
        assert out == "Tool 'read_file' is disabled by .guru/tools.toml"

    def test_spawn_label_error_is_corrective(self, monkeypatch) -> None:
        monkeypatch.setattr(tools, '_spawn_handler',
                            lambda *a: 'spawned')
        out = tools.execute_tool('spawn', {'task': 't', 'kind': 'weird'})
        assert out.startswith(tools.INVALID_ARGS_PREFIX)
        assert 'e.g. spawn(' in out
        assert tools.execute_tool(
            'spawn', {'task': 't', 'kind': 'Review'}) == 'spawned'


class TestForKind:
    def test_review_hides_every_write_tool(self) -> None:
        hidden = toolpolicy.for_kind('review')
        assert hidden == toolpolicy.WRITE_TOOLS
        assert {'write_file', 'edit_file', 'apply_patch', 'delete_file',
                'sandbox_run', 'sandbox_python', 'sandbox_submit',
                'request_dependency'} == set(hidden)
        assert 'sandbox_diff' not in hidden and 'code_health' not in hidden
        assert 'read_file' not in hidden

    def test_other_kinds_hide_nothing(self) -> None:
        for kind in routing.KINDS:
            if kind != 'review':
                assert toolpolicy.for_kind(kind) == frozenset(), kind
        assert toolpolicy.for_kind('') == frozenset()
        assert toolpolicy.for_kind(None) == frozenset()
        assert toolpolicy.for_kind(' REVIEW ') == toolpolicy.WRITE_TOOLS

    def test_refusal_text(self) -> None:
        text = toolpolicy.kind_refusal('edit_file', 'review')
        assert text.startswith('Refused: edit_file')
        assert 'review task (read-only)' in text


class TestKindAtTheToolLayer:
    """A review task neither sees nor may call a write tool."""

    def test_initial_tools_take_the_kind(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'FLAT_TOOLS', True)
        monkeypatch.setattr(tools, '_sandbox_available', lambda: False)
        _base, names = tools.initial_tools(can_spawn=False, kind='review')
        assert 'read_file' in names and 'outline' in names
        assert not names & toolpolicy.WRITE_TOOLS
        _base, names = tools.initial_tools(can_spawn=False, kind='build')
        assert 'edit_file' in names

    def test_specs_follow_the_kind(self, monkeypatch) -> None:
        monkeypatch.setattr(tools, '_sandbox_available', lambda: False)
        active = {'read_file', 'edit_file', 'write_file'}
        names = {s['name']
                 for s in tools.specs_for(active, False, kind='review')}
        assert 'read_file' in names and 'edit_file' not in names
        names = {s['name'] for s in tools.specs_for(active, False)}
        assert 'edit_file' in names

    def test_session_kind_drives_advertising(self, monkeypatch) -> None:
        monkeypatch.setattr(tools, '_sandbox_available', lambda: False)
        monkeypatch.setattr(session, 'task_kind', 'review')
        assert 'edit_file' not in tools._advertised()
        assert 'edit_file' not in tools._match_tools('edit a file')
        monkeypatch.setattr(session, 'task_kind', 'debug')
        assert 'edit_file' in tools._advertised()

    def test_execute_tool_refuses_for_a_review_task(self, tmp_path,
                                                    monkeypatch,
                                                    fake_repo) -> None:
        monkeypatch.setattr(tools, '_sandbox_available', lambda: False)
        monkeypatch.setattr(session, 'task_kind', 'review')
        monkeypatch.setattr(config, 'MODE', config.MODE_AUTO)
        monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', {str(tmp_path)})
        monkeypatch.setattr(files, '_show_change', lambda block: None)
        target = tmp_path / 'a.txt'
        out = tools.execute_tool('write_file',
                                 {'path': str(target), 'content': 'x'})
        assert out.startswith('Refused: write_file is not available to a'
                              ' review task')
        assert not target.exists()
        from guru.domain import ledger
        ledger.flush()
        assert fake_repo.stream('tool_events')[-1]['denied'] == 'kind'
        # Reads still work for the reviewer.
        target.write_text('hello\n')
        assert 'hello' in tools.execute_tool('read_file',
                                             {'path': str(target)})

    def test_session_state_has_task_kind(self) -> None:
        assert session.SessionState().task_kind == ''
