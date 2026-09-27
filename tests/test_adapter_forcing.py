"""Tool forcing per adapter (the turn contract): Anthropic ``tool_choice
any`` (never together with extended thinking), LiteLLM ``required`` with
the one-retry fallback, Ollama never (spec-driven tool schemas instead),
and the shared schema builder that sends the plan's nested schema."""
import itertools
from types import SimpleNamespace

import pytest

from guru import session, ui
from guru.adapters import anthropic as anth
from guru.adapters import base
from guru.adapters import litellm as lite
from guru.adapters.ollama import OllamaAdapter
from guru.domain import plan, tools

_PLAN_SPEC = tools._PLAN_SPEC
_READ_SPEC = {'name': 'read_file', 'description': 'd',
              'parameters': {'path': 'p', 'lines': 'l'},
              'optional': ['lines']}


class TestParametersSchema:
    def test_string_parameters(self) -> None:
        assert base.parameters_schema(_READ_SPEC) == {
            'type': 'object',
            'properties': {'path': {'type': 'string', 'description': 'p'},
                           'lines': {'type': 'string', 'description': 'l'}},
            'required': ['path']}

    def test_own_schema_wins(self) -> None:
        assert base.parameters_schema(_PLAN_SPEC) is plan.SCHEMA

    def test_openai_tool_defs_shared(self) -> None:
        defs = base.openai_tool_defs([_READ_SPEC, _PLAN_SPEC])
        assert defs[0]['type'] == 'function'
        assert defs[0]['function']['parameters']['required'] == ['path']
        assert defs[1]['function']['name'] == 'plan'
        assert defs[1]['function']['parameters'] is plan.SCHEMA
        assert lite.openai_tool_defs is base.openai_tool_defs

    def test_anthropic_tool_defs_send_the_plan_schema(self) -> None:
        defs = anth.tool_defs([_READ_SPEC, _PLAN_SPEC])
        assert defs[0]['input_schema']['required'] == ['path']
        assert defs[1] == {'name': 'plan',
                           'description': _PLAN_SPEC['description'],
                           'input_schema': plan.SCHEMA}

    def test_base_adapter_never_forces(self) -> None:
        class Plain(base.Adapter):
            def available(self): return True
            def list_models(self): return []
            def activate(self, m): pass
            def run_turn(self): pass
            def summarise(self, t): return ''
        assert Plain().forces(base.FORCE_PLAN) is False
        assert Plain().forces(base.FORCE_ANY) is False


def _arm(monkeypatch, controller: bool, specs=None) -> None:
    from guru.adapters import turn
    monkeypatch.setattr(ui, 'note_thinking', lambda: None)
    monkeypatch.setattr(ui, 'status_draw', lambda: None)
    monkeypatch.setattr(ui.console, 'print', lambda *a, **k: None)
    monkeypatch.setattr(turn, '_render_answer', lambda c: None)
    monkeypatch.setattr(session, 'model', 'm')
    monkeypatch.setattr(session, 'num_ctx', 4096)
    monkeypatch.setattr(session, 'cancel_requested', False)
    monkeypatch.setattr(session, 'session_in', 0)
    monkeypatch.setattr(session, 'session_out', 0)
    monkeypatch.setattr(session, 'controller', controller)
    monkeypatch.setattr(session, 'can_spawn', controller)
    monkeypatch.setattr(session, 'task_id', '')
    monkeypatch.setattr(session, 'messages', [
        {'role': 'system', 'content': 'SYS'},
        {'role': 'user', 'content': 'q'}])
    monkeypatch.setattr('guru.domain.tools.active_specs',
                        lambda: specs if specs is not None else [
                            _PLAN_SPEC if controller else _READ_SPEC])


class TestAnthropicForcing:
    def test_forces_plan_always_any_only_without_thinking(self) -> None:
        a = anth.AnthropicAdapter(thinking=True)
        assert a.forces(base.FORCE_PLAN) is True
        assert a.forces(base.FORCE_ANY) is False
        a = anth.AnthropicAdapter(thinking=False)
        assert a.forces(base.FORCE_PLAN) is True
        assert a.forces(base.FORCE_ANY) is True
        a._force_ok = False
        assert a.forces(base.FORCE_PLAN) is False

    def _run(self, monkeypatch, *, thinking: bool, controller: bool,
             responses=None):
        """Run one turn; returns the kwargs of every ``messages.create``."""
        _arm(monkeypatch, controller)
        monkeypatch.setattr(session, 'adapter', None)
        seen: list = []
        answer = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason='end_turn',
            content=[SimpleNamespace(type='text', text='ok')])
        script = itertools.chain(responses or [], itertools.repeat(answer))

        def create(**kw):
            seen.append(kw)
            item = next(script)
            if isinstance(item, Exception):
                raise item
            return item
        a = anth.AnthropicAdapter(thinking=thinking, cache=False)
        monkeypatch.setattr(session, 'adapter', a)
        monkeypatch.setattr(a, '_client', lambda: SimpleNamespace(
            messages=SimpleNamespace(create=create)))
        a.run_turn()
        return seen, a

    def test_controller_round_forced_without_thinking(self, monkeypatch):
        seen, _ = self._run(monkeypatch, thinking=True, controller=True)
        kw = seen[0]
        assert kw['tool_choice'] == {'type': 'any'}
        assert 'thinking' not in kw
        assert kw['tools'][0]['input_schema'] is plan.SCHEMA

    def test_thinking_worker_round_not_forced(self, monkeypatch) -> None:
        seen, _ = self._run(monkeypatch, thinking=True, controller=False)
        kw = seen[0]
        assert 'tool_choice' not in kw
        assert kw['thinking'] == {'type': 'adaptive', 'display': 'summarized'}
        # the text reply is the answer at once (no re-prompt)
        assert len(seen) == 1
        assert session.messages[-1] == {'role': 'assistant', 'content': 'ok'}

    def test_worker_round_forced_when_thinking_is_off(self, monkeypatch):
        seen, _ = self._run(monkeypatch, thinking=False, controller=False)
        assert seen[0]['tool_choice'] == {'type': 'any'}
        assert 'thinking' not in seen[0]
        # forcing adapter + text reply: one re-prompt, then accepted
        assert len(seen) == 2
        # (cache off sends the live native list itself, so look for the
        # re-prompt rather than at its tail after the turn)
        assert {'role': 'user', 'content': plan.REPROMPT_TEXT} in \
            seen[1]['messages']
        assert session.struggle['protocol_violation'] == 1

    def test_tool_choice_error_retries_unforced_and_disables(
            self, monkeypatch, fake_repo) -> None:
        boom = RuntimeError("400: thinking may not be enabled when "
                            "tool_choice forces tool use")
        seen, a = self._run(monkeypatch, thinking=False, controller=False,
                            responses=[boom])
        assert len(seen) == 2
        assert 'tool_choice' in seen[0] and 'tool_choice' not in seen[1]
        assert a._force_ok is False and a.forces(base.FORCE_PLAN) is False
        assert session.struggle['provider_errors'] == 0
        assert session.struggle['protocol_violation'] == 0   # not forcing now

    def test_other_error_is_not_retried(self, monkeypatch, fake_repo):
        seen, a = self._run(monkeypatch, thinking=False, controller=False,
                            responses=[RuntimeError('rate limited')])
        assert len(seen) == 1 and a._force_ok is True
        assert session.struggle['provider_errors'] == 1

    def test_no_tools_means_no_tool_choice(self, monkeypatch) -> None:
        _arm(monkeypatch, controller=False, specs=[])
        seen: list = []
        answer = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason='end_turn',
            content=[SimpleNamespace(type='text', text='ok')])
        a = anth.AnthropicAdapter(thinking=False, cache=False)
        monkeypatch.setattr(session, 'adapter', a)
        monkeypatch.setattr(a, '_client', lambda: SimpleNamespace(
            messages=SimpleNamespace(
                create=lambda **kw: seen.append(kw) or answer)))
        a.run_turn()
        assert 'tool_choice' not in seen[0]

    @pytest.mark.parametrize('text, is_it', [
        ('tool_choice: any is not allowed with thinking', True),
        ('Invalid tool_choice', True), ('overloaded', False)])
    def test_is_tool_choice_error(self, text, is_it) -> None:
        assert anth.is_tool_choice_error(RuntimeError(text)) is is_it
        assert lite.is_tool_choice_error(RuntimeError(text)) is is_it


class TestLiteLLMForcing:
    def _run(self, monkeypatch, *, controller: bool, responses=None):
        _arm(monkeypatch, controller)
        seen: list = []
        answer = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='ok', tool_calls=None),
                finish_reason='stop')])
        script = itertools.chain(responses or [], itertools.repeat(answer))

        def create(**kw):
            seen.append(kw)
            item = next(script)
            if isinstance(item, Exception):
                raise item
            return SimpleNamespace(parse=lambda: item, headers={})
        a = lite.LiteLLMAdapter(base_url='http://proxy', cache=False)
        monkeypatch.setattr(session, 'adapter', a)
        monkeypatch.setattr(a, '_client', lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=create)))))
        a.run_turn()
        return seen, a

    def test_forces_every_round(self) -> None:
        a = lite.LiteLLMAdapter(base_url='http://proxy')
        assert a.forces(base.FORCE_PLAN) and a.forces(base.FORCE_ANY)
        a._force_ok = False
        assert not a.forces(base.FORCE_ANY)

    def test_controller_and_worker_send_required(self, monkeypatch, fake_repo):
        seen, _ = self._run(monkeypatch, controller=True)
        assert seen[0]['tool_choice'] == 'required'
        assert seen[0]['tools'][0]['function']['parameters'] is plan.SCHEMA
        seen, _ = self._run(monkeypatch, controller=False)
        assert seen[0]['tool_choice'] == 'required'
        # a text reply on a forcing adapter: one re-prompt, then accepted
        assert len(seen) == 2
        assert {'role': 'user', 'content': plan.REPROMPT_TEXT} in \
            seen[1]['messages']

    def test_tool_choice_error_retries_unforced_and_disables(
            self, monkeypatch, fake_repo) -> None:
        boom = RuntimeError('litellm.BadRequestError: tool_choice required'
                            ' is not supported with thinking')
        seen, a = self._run(monkeypatch, controller=False,
                            responses=[boom])
        assert [('tool_choice' in kw) for kw in seen] == [True, False]
        assert a._force_ok is False
        assert session.struggle['provider_errors'] == 0
        assert session.messages[-1] == {'role': 'assistant', 'content': 'ok'}

    def test_other_error_is_not_retried(self, monkeypatch, fake_repo):
        seen, a = self._run(monkeypatch, controller=False,
                            responses=[RuntimeError('proxy down')])
        assert len(seen) == 1 and a._force_ok is True
        assert session.struggle['provider_errors'] == 1


class TestOllamaNeverForces:
    def test_forces_false(self) -> None:
        a = OllamaAdapter()
        assert a.forces(base.FORCE_PLAN) is False
        assert a.forces(base.FORCE_ANY) is False

    def test_tools_are_the_spec_schemas(self, monkeypatch) -> None:
        """The daemon gets the same schemas the other adapters send (the
        nested plan schema included), not introspected callables."""
        _arm(monkeypatch, controller=True)
        a = OllamaAdapter()
        monkeypatch.setattr(a, '_supports_thinking', lambda m: False)
        monkeypatch.setattr(session, 'active_tools', [tools.plan])
        seen: dict = {}

        def fake_chat(**kw):
            seen.update(kw)
            yield SimpleNamespace(
                message=SimpleNamespace(content='x', tool_calls=None),
                prompt_eval_count=1, eval_count=1)
        monkeypatch.setattr('guru.adapters.ollama.ollama.chat', fake_chat)
        a._collect_response()
        assert seen['tools'] == base.openai_tool_defs([_PLAN_SPEC])
        assert seen['tools'][0]['function']['parameters'] is plan.SCHEMA

    def test_controller_json_text_is_the_plan(self, monkeypatch) -> None:
        """Ollama cannot force: the controller's JSON text runs as the
        plan (guru.domain.plan.from_text)."""
        import ollama
        _arm(monkeypatch, controller=True)
        a = OllamaAdapter()
        monkeypatch.setattr(session, 'adapter', a)
        monkeypatch.setattr(a, '_fit_after_load', lambda: None)
        monkeypatch.setattr(ui, 'note_tool', lambda *x: None)
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        seen: list = []
        tools.set_plan_handler(lambda args: seen.append(args) or
                               plan.ANSWER_ACK)
        try:
            monkeypatch.setattr(a, '_collect_response', lambda: ollama.Message(
                role='assistant',
                content='{"outcome": "answer", "answer": "Hi there"}'))
            a.run_turn()
        finally:
            tools.set_plan_handler(None)
        assert seen == [{'outcome': 'answer', 'answer': 'Hi there'}]
        last = session.messages[-1]
        assert last.role == 'assistant' and last.content == 'Hi there'
