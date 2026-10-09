"""Thinking without forcing per adapter: LiteLLM sends ``reasoning_effort``
(the lead's or a worker's) and never ``tool_choice``, retries once without
reasoning when a model rejects it, and keeps the thinking blocks of a
tool round; Anthropic sends adaptive thinking and never ``tool_choice``;
every adapter takes a text reply as the answer at once."""
import itertools
from types import SimpleNamespace

import pytest

from guru import config, session, ui
from guru.adapters import anthropic as anth
from guru.adapters import base
from guru.adapters import litellm as lite
from guru.adapters import turn

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

    def test_openai_tool_defs_shared(self) -> None:
        defs = base.openai_tool_defs([_READ_SPEC])
        assert defs[0]['type'] == 'function'
        assert defs[0]['function']['parameters']['required'] == ['path']
        assert lite.openai_tool_defs is base.openai_tool_defs

    def test_base_adapter_has_no_forcing(self) -> None:
        assert not hasattr(base.Adapter, 'forces')
        assert not hasattr(base, 'FORCE_PLAN')


def _arm(monkeypatch, task_id: str = '') -> None:
    monkeypatch.setattr(ui, 'note_thinking', lambda: None)
    monkeypatch.setattr(ui, 'status_draw', lambda: None)
    monkeypatch.setattr(ui.console, 'print', lambda *a, **k: None)
    monkeypatch.setattr(turn, '_render_answer', lambda c: None)
    monkeypatch.setattr(session, 'model', 'm')
    monkeypatch.setattr(session, 'num_ctx', 4096)
    monkeypatch.setattr(session, 'cancel_requested', False)
    monkeypatch.setattr(session, 'session_in', 0)
    monkeypatch.setattr(session, 'session_out', 0)
    monkeypatch.setattr(session, 'can_spawn', not task_id)
    monkeypatch.setattr(session, 'task_id', task_id)
    monkeypatch.setattr(session, 'messages', [
        {'role': 'system', 'content': 'SYS'},
        {'role': 'user', 'content': 'q'}])
    monkeypatch.setattr('guru.domain.tools.active_specs',
                        lambda: [_READ_SPEC])


class TestReasoningEffort:
    def test_lead_and_worker_defaults(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'task_id', '')
        assert turn.reasoning_effort() == config.THINKING_LEAD == 'high'
        monkeypatch.setattr(session, 'task_id', 't1')
        assert turn.reasoning_effort() == config.THINKING_WORKER == 'medium'

    @pytest.mark.parametrize('raw, lead', [
        ('low', 'low'), ('OFF', ''), ('', ''), ('extreme', 'high')])
    def test_settings(self, monkeypatch, raw, lead) -> None:
        monkeypatch.setattr(config, 'THINKING_LEAD', 'high')
        monkeypatch.setattr(config, 'THINKING_WORKER', 'medium')
        config._apply_thinking({'lead': raw})
        assert config.THINKING_LEAD == lead
        assert config.THINKING_WORKER == 'medium'


def _lite_answer(text='ok', tool_calls=None, thinking=None):
    msg = SimpleNamespace(content=text, tool_calls=tool_calls)
    if thinking is not None:
        msg.thinking_blocks = thinking
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        choices=[SimpleNamespace(message=msg, finish_reason='stop')])


class TestLiteLLM:
    def _run(self, monkeypatch, *, task_id='', responses=None,
             adapter=None):
        _arm(monkeypatch, task_id)
        seen: list = []
        script = itertools.chain(responses or [],
                                 itertools.repeat(_lite_answer()))

        def create(**kw):
            seen.append(kw)
            item = next(script)
            if isinstance(item, Exception):
                raise item
            return SimpleNamespace(parse=lambda: item, headers={})
        a = adapter or lite.LiteLLMAdapter(base_url='http://proxy',
                                           cache=False)
        monkeypatch.setattr(session, 'adapter', a)
        monkeypatch.setattr(a, '_client', lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=create)))))
        monkeypatch.setattr('guru.domain.tools.execute_tool',
                            lambda name, args: 'file text')
        a.run_turn()
        return seen, a

    def test_lead_sends_high_effort_and_no_tool_choice(
            self, monkeypatch, fake_repo) -> None:
        seen, _ = self._run(monkeypatch)
        assert seen[0]['reasoning_effort'] == 'high'
        assert 'tool_choice' not in seen[0]
        assert seen[0]['max_tokens'] == lite.TURN_MAX_TOKENS
        # the text reply is the answer at once
        assert len(seen) == 1
        assert session.messages[-1] == {'role': 'assistant', 'content': 'ok'}

    def test_worker_sends_medium(self, monkeypatch, fake_repo) -> None:
        seen, _ = self._run(monkeypatch, task_id='t1')
        assert seen[0]['reasoning_effort'] == 'medium'

    def test_effort_off_sends_none(self, monkeypatch, fake_repo) -> None:
        monkeypatch.setattr(config, 'THINKING_LEAD', '')
        seen, _ = self._run(monkeypatch)
        assert 'reasoning_effort' not in seen[0]

    def test_rejected_reasoning_retried_once_and_remembered(
            self, monkeypatch, fake_repo) -> None:
        boom = RuntimeError('400: model does not support reasoning_effort')
        seen, a = self._run(monkeypatch, responses=[boom])
        assert [('reasoning_effort' in kw) for kw in seen] == [True, False]
        assert a._no_reasoning == {'m'}
        assert session.struggle['provider_errors'] == 0
        seen, _ = self._run(monkeypatch, adapter=a)
        assert 'reasoning_effort' not in seen[0]

    def test_other_error_is_not_retried(self, monkeypatch, fake_repo):
        seen, a = self._run(monkeypatch,
                            responses=[RuntimeError('proxy down')])
        assert len(seen) == 1 and not a._no_reasoning
        assert session.struggle['provider_errors'] == 1

    def test_thinking_blocks_kept_on_a_tool_round(
            self, monkeypatch, fake_repo) -> None:
        block = {'type': 'thinking', 'thinking': 'hm', 'signature': 's'}
        call = SimpleNamespace(id='c1', function=SimpleNamespace(
            name='read_file', arguments='{"path": "a.py"}'))
        seen, _ = self._run(monkeypatch, responses=[
            _lite_answer('', [call], [block])])
        # the next round carries the assistant tool round with its thinking
        assistant = next(m for m in seen[1]['messages']
                         if m.get('role') == 'assistant')
        assert assistant['thinking_blocks'] == [block]
        stored = next(m for m in session.messages
                      if m.get('role') == 'assistant' and m.get('tool_calls'))
        assert stored['thinking_blocks'] == [block]
        # and the next turn rebuilds the round natively with it
        native = lite.to_openai_messages(session.messages)
        rebuilt = next(m for m in native if m.get('tool_calls'))
        assert rebuilt['thinking_blocks'] == [block]

    @pytest.mark.parametrize('text, is_it', [
        ('reasoning_effort is not supported', True),
        ('litellm.UnsupportedParamsError: openai does not support'
         ' parameters: [reasoning_effort]', True),
        ('`max_tokens` must be greater than `thinking.budget_tokens`', False),
        ('Expected `thinking` or `redacted_thinking`, but found `tool_use`',
         False),
        ('overloaded', False)])
    def test_is_reasoning_error(self, text, is_it) -> None:
        assert lite.is_reasoning_error(RuntimeError(text)) is is_it

    def test_history_shape_error_keeps_thinking(self, monkeypatch,
                                                fake_repo) -> None:
        boom = RuntimeError('Expected `thinking` or `redacted_thinking`,'
                            ' but found `tool_use`')
        seen, a = self._run(monkeypatch, responses=[boom])
        assert len(seen) == 1 and not a._no_reasoning
        assert session.struggle['provider_errors'] == 1

    def test_small_output_model_falls_back_once(self, monkeypatch,
                                                fake_repo) -> None:
        boom = RuntimeError('400: max_tokens is too large: 32000. This'
                            ' model supports at most 16384 completion'
                            ' tokens')
        seen, a = self._run(monkeypatch, responses=[boom])
        assert [kw['max_tokens'] for kw in seen] == [
            lite.TURN_MAX_TOKENS, lite._MAX_TOKENS]
        assert a._small_output == {'m'}
        seen, _ = self._run(monkeypatch, adapter=a)
        assert seen[0]['max_tokens'] == lite._MAX_TOKENS


class TestAnthropic:
    def _run(self, monkeypatch, *, thinking: bool):
        _arm(monkeypatch)
        seen: list = []
        answer = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason='end_turn',
            content=[SimpleNamespace(type='text', text='ok')])

        def create(**kw):
            seen.append(kw)
            return answer
        a = anth.AnthropicAdapter(thinking=thinking, cache=False)
        monkeypatch.setattr(session, 'adapter', a)
        monkeypatch.setattr(a, '_client', lambda: SimpleNamespace(
            messages=SimpleNamespace(create=create)))
        a.run_turn()
        return seen

    def test_thinking_and_no_tool_choice(self, monkeypatch, fake_repo):
        seen = self._run(monkeypatch, thinking=True)
        assert 'tool_choice' not in seen[0]
        assert seen[0]['thinking'] == anth.THINKING
        assert len(seen) == 1
        assert session.messages[-1] == {'role': 'assistant', 'content': 'ok'}

    def test_thinking_off(self, monkeypatch, fake_repo) -> None:
        seen = self._run(monkeypatch, thinking=False)
        assert 'thinking' not in seen[0] and 'tool_choice' not in seen[0]

    def test_effort_off_turns_thinking_off(self, monkeypatch, fake_repo):
        monkeypatch.setattr(config, 'THINKING_LEAD', '')
        seen = self._run(monkeypatch, thinking=True)
        assert 'thinking' not in seen[0]
