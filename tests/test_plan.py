"""The typed controller plan (guru.domain.plan): schema, parsing,
validation and the texts the loop and the handler exchange."""
import json

import pytest

from guru.domain import plan, routing


def _task(goal='review auth for security', kind='review',
          complexity='standard', **extra) -> dict:
    return {'goal': goal, 'kind': kind, 'complexity': complexity, **extra}


class TestSchema:
    def test_schema_matches_the_vocabularies(self) -> None:
        props = plan.SCHEMA['properties']
        assert props['outcome']['enum'] == ['answer', 'delegate']
        assert plan.SCHEMA['required'] == ['outcome']
        task = props['tasks']['items']
        assert task is plan.TASK_SCHEMA
        assert task['properties']['kind']['enum'] == list(routing.KINDS)
        assert task['properties']['complexity']['enum'] == \
            list(routing.COMPLEXITY)
        assert task['required'] == ['goal', 'kind', 'complexity']
        assert task['properties']['files']['items'] == {'type': 'string'}

    def test_schema_is_json(self) -> None:
        json.dumps(plan.SCHEMA)

    def test_label_descriptions_carry_the_shared_rubric(self) -> None:
        text = plan.TASK_SCHEMA['properties']['complexity']['description']
        for tier, desc in routing.COMPLEXITY_DESCRIPTIONS.items():
            assert f'{tier} = {desc}' in text
        text = plan.TASK_SCHEMA['properties']['kind']['description']
        for kind, desc in routing.KIND_DESCRIPTIONS.items():
            assert f'{kind} = {desc}' in text


class TestParse:
    def test_answer(self) -> None:
        p, errors = plan.parse({'outcome': 'answer', 'answer': 'Hi.'})
        assert errors == [] and p == plan.Plan('answer', answer='Hi.')

    def test_answer_ignores_tasks(self) -> None:
        p, errors = plan.parse({'outcome': 'answer', 'answer': 'Hi.',
                                'tasks': [_task()]})
        assert errors == [] and p.tasks == []

    def test_answer_without_text_is_malformed(self) -> None:
        for args in ({'outcome': 'answer'},
                     {'outcome': 'answer', 'answer': '  '},
                     {'outcome': 'answer', 'answer': 3}):
            p, errors = plan.parse(args)
            assert p is not None and p.outcome == 'answer'
            assert errors == ['outcome answer needs the reply text in answer']

    @pytest.mark.parametrize('args', [
        {}, {'outcome': 'plan'}, {'outcome': None}, 'answer', None, [],
        {'outcome': 'Answer '}])
    def test_unknown_outcome(self, args) -> None:
        p, errors = plan.parse(args)
        if args == {'outcome': 'Answer '}:      # case/space normalised
            assert p is not None and p.outcome == 'answer'
        else:
            assert p is None and len(errors) == 1

    def test_delegate(self) -> None:
        p, errors = plan.parse({'outcome': 'Delegate', 'tasks': [
            _task(files=['a.py', ' b.py '], role='developer',
                  skill='code-review'),
            {'goal': 'fix it', 'kind': 'DEBUG', 'complexity': 'Hard'}]})
        assert errors == []
        assert p.outcome == 'delegate' and len(p.tasks) == 2
        assert p.tasks[0] == plan.Task('review auth for security', 'review',
                                       'standard', ['a.py', 'b.py'],
                                       'developer', 'code-review')
        assert (p.tasks[1].kind, p.tasks[1].complexity) == ('debug', 'hard')

    def test_delegate_needs_a_task(self) -> None:
        for tasks in (None, [], 'x', {}):
            p, errors = plan.parse({'outcome': 'delegate', 'tasks': tasks})
            assert p is not None and p.tasks == []
            assert errors == ['outcome delegate needs at least one task']

    def test_task_field_errors(self) -> None:
        p, errors = plan.parse({'outcome': 'delegate', 'tasks': [
            {'goal': '', 'kind': 'review', 'complexity': 'standard'},
            'not an object',
            _task(files='a.py'), _task(role=['x'])]})
        assert p is not None and len(p.tasks) == 3
        assert errors == [
            'task 1: goal must be a non-empty string',
            'task 2: must be an object with goal, kind and complexity',
            'task 3: files must be a list of strings',
            'task 4: role must be a string']

    def test_missing_labels_default(self) -> None:
        p, errors = plan.parse({'outcome': 'delegate',
                                'tasks': [{'goal': 'look'}]})
        assert errors == []
        assert (p.tasks[0].kind, p.tasks[0].complexity) == ('other',
                                                            'standard')

    def test_json_encoded_lists_are_decoded(self) -> None:
        tasks = json.dumps([_task(files=json.dumps(['a.py']))])
        p, errors = plan.parse({'outcome': 'delegate', 'tasks': tasks})
        assert errors == [] and p.tasks[0].files == ['a.py']


class TestValidate:
    def test_unknown_labels_are_hard_errors(self) -> None:
        p, _ = plan.parse({'outcome': 'delegate', 'tasks': [
            _task(kind='bugfix', complexity='medium')]})
        assert plan.schema_errors(p) == [
            "task 1: unknown kind 'bugfix'; one of " + ', '.join(
                routing.KINDS),
            "task 1: unknown complexity 'medium'; one of trivial, standard,"
            " hard"]
        assert plan.validate('r', p) == plan.schema_errors(p)

    def test_answer_is_never_rejected(self) -> None:
        p, _ = plan.parse({'outcome': 'answer', 'answer': 'Hi.'})
        request = 'review this for correctness, security and performance'
        assert plan.validate(request, p) == []
        v = plan.evaluate(request, {'outcome': 'answer', 'answer': 'Hi.'})
        assert v.ok and v.plan.answer == 'Hi.'

    @pytest.mark.parametrize('request_text, named', [
        ('Review this repository for correctness and security issues.',
         ['correctness', 'security']),
        ('Is there a path traversal risk in the upload handler?',
         ['security']),
        ('fix the failing test in wordcount.py', ['correctness', 'tests']),
        ('hi, what can you do?', []),
        ('check error handling and the README', ['reliability', 'docs']),
        ('is the design performant?', ['design']),   # no fuzzy matching
        ('secure-by-default settings', []),          # hyphenated compound
    ])
    def test_concerns_in(self, request_text, named) -> None:
        assert plan.concerns_in(request_text) == named

    def test_missing_concern_is_soft(self) -> None:
        request = 'Review this repository for correctness and security.'
        args = {'outcome': 'delegate', 'tasks': [
            _task(goal='review app/ for injection and authz', kind='review',
                  complexity='hard')]}
        v = plan.evaluate(request, args)
        assert v.plan is not None and v.errors == []
        assert v.missing == ['correctness'] and not v.ok
        [msg] = v.messages
        assert msg.startswith("no task goal covers the concern 'correctness'")
        assert plan.validate(request, v.plan) == [msg]

    def test_every_named_concern_covered_is_ok(self) -> None:
        request = 'Review this repository for correctness and security.'
        args = {'outcome': 'delegate', 'tasks': [
            _task(goal='review the code for bugs and logic errors'),
            _task(goal='review the code for injection and secrets')]}
        assert plan.evaluate(request, args).ok

    def test_single_concern_never_re_asks(self) -> None:
        request = 'Is there a path traversal risk in the upload handler?'
        args = {'outcome': 'delegate', 'tasks': [
            _task(goal='inspect app/upload.py save_upload')]}
        assert plan.evaluate(request, args).ok

    def test_followup_skips_coverage(self) -> None:
        request = 'Review this repository for correctness and security.'
        args = {'outcome': 'delegate', 'tasks': [
            _task(goal='re-check the injection finding')]}
        assert not plan.evaluate(request, args).ok
        assert plan.evaluate(request, args, followup=True).ok
        p, _ = plan.parse(args)
        assert plan.validate(request, p, followup=True) == []

    def test_hard_errors_suppress_coverage(self) -> None:
        request = 'Review this repository for correctness and security.'
        args = {'outcome': 'delegate', 'tasks': [_task(kind='nope')]}
        v = plan.evaluate(request, args)
        assert v.errors and v.missing == []

    def test_malformed_verdict(self) -> None:
        v = plan.evaluate('r', {'outcome': 'later'})
        assert v.plan is None and not v.ok
        assert v.messages == ["outcome must be 'answer' or 'delegate'"]


class TestFromText:
    def test_bare_and_fenced_json(self) -> None:
        obj = {'outcome': 'answer', 'answer': 'Hi.'}
        assert plan.from_text(json.dumps(obj)) == obj
        assert plan.from_text('Here:\n```json\n' + json.dumps(obj)
                              + '\n```\nthanks') == obj
        assert plan.from_text('prefix {"x": 1} then ' + json.dumps(obj)) \
            == obj

    def test_nested_plan(self) -> None:
        obj = {'outcome': 'delegate', 'tasks': [_task(files=['a.py'])]}
        assert plan.from_text('I will do this:\n' + json.dumps(obj)) == obj

    @pytest.mark.parametrize('text', [
        '', 'no json here', '{"not": "a plan"}', '{broken'])
    def test_none_without_a_plan_object(self, text) -> None:
        assert plan.from_text(text) is None

    def test_plan_object_inside_a_list_still_counts(self) -> None:
        assert plan.from_text('[{"outcome": "answer"}]') == {
            'outcome': 'answer'}


class TestTexts:
    def test_reask_round_trip(self) -> None:
        text = plan.reask_text(['a', 'b'])
        assert plan.is_reask(text) and plan.is_reprompt(text)
        assert text.startswith(plan.REASK_PREFIX)
        assert text.endswith(plan.REASK_SUFFIX)
        assert plan.reask_problems(text) == 'a; b'

    def test_reprompts_are_recognised(self) -> None:
        assert plan.is_reprompt(plan.REPROMPT_TEXT)
        assert plan.is_reprompt(plan.PLAN_REPROMPT_TEXT)
        assert not plan.is_reprompt('Plan: read the files')

    def test_reasks_in_counts_tool_results_and_user_messages(self) -> None:
        reask = plan.reask_text(['x'])
        msgs = [
            {'role': 'user', 'content': 'request'},
            {'role': 'tool', 'tool_name': 'plan', 'content': reask},
            {'role': 'tool', 'tool_name': 'read_file', 'content': reask},
            {'role': 'user', 'content': reask},
            {'role': 'tool', 'tool_name': 'plan', 'content': plan.ANSWER_ACK},
            {'role': 'assistant', 'content': reask},
            object()]
        assert plan.reasks_in(msgs) == 2

    def test_delegated_and_refused_texts(self) -> None:
        tasks = [plan.Task('a', 'review', 'hard'), plan.Task('b')]
        text = plan.delegated_text(['agent1', 'agent2'], tasks)
        assert text.startswith(plan.DELEGATED_PREFIX)
        assert 'agent1 (review/hard), agent2 (other/standard)' in text
        assert '[joined results]' in text and 'plan' in text
        text = plan.refused_text(['refused: no rung'])
        assert text.startswith(plan.REFUSED_PREFIX)
        assert 'refused: no rung' in text and 'outcome answer' in text
        assert 'no model is allowed' in plan.refused_text([])

    def test_fallback_text_prefers_the_model_text(self) -> None:
        args = {'outcome': 'delegate', 'answer': 'field',
                'tasks': [_task(goal='g1'), 'junk', {'goal': ''}]}
        assert plan.fallback_text('  said  ', args, ['e']) == 'said'
        assert plan.fallback_text('', args, ['e']) == 'field'
        del args['answer']
        text = plan.fallback_text('', args, ['e1', 'e2'])
        assert text.startswith("(guru could not run the controller's plan:"
                               " e1; e2.)")
        assert text.endswith('Proposed tasks:\n- g1')
        assert plan.fallback_text('', None, ['e']) == \
            "(guru could not run the controller's plan: e.)"

    def test_final_text(self) -> None:
        assert plan.final_text({'text': ' done '}) == 'done'
        assert plan.final_text({'answer': 'done'}) == 'done'
        assert plan.final_text({'n': 3}) == ''
        assert plan.final_text(None) == ''


class TestTaskText:
    def test_goal_files_and_hook(self) -> None:
        t = plan.Task('review auth', 'review', 'hard', files=['a.py', 'b/'])
        assert plan.task_text(t) == 'review auth\nFiles: a.py, b/'
        assert plan.task_text(plan.Task('look')) == 'look'

    def test_brief_hook_is_identity_for_now(self) -> None:
        assert plan.brief_hook('x') == 'x'
