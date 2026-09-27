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

    def test_descriptions_carry_the_controller_contract(self) -> None:
        """The controller has no other tool and no prose hint saying so:
        the contract lives in the plan tool's own descriptions."""
        from guru import config
        from guru.domain import tools
        sentence = ('You have no other tools. Anything that needs a file, a'
                    ' command, a package, a test or the sandbox must be'
                    ' delegated; answer is for replies that need no work.')
        assert config.PLAN_CONTRACT_SENTENCE == sentence
        assert sentence in config.PLAN_TOOL_DESCRIPTION
        assert sentence in tools.tool_spec('plan')['description']
        outcome = plan.SCHEMA['properties']['outcome']['description']
        assert 'You have no other tools' in outcome
        assert 'must be delegated' in outcome

    def test_schema_is_json(self) -> None:
        json.dumps(plan.SCHEMA)

    def test_label_descriptions_carry_the_shared_rubric(self) -> None:
        text = plan.TASK_SCHEMA['properties']['complexity']['description']
        for tier, desc in routing.COMPLEXITY_DESCRIPTIONS.items():
            assert f'{tier} = {desc}' in text
        text = plan.TASK_SCHEMA['properties']['kind']['description']
        for kind, desc in routing.KIND_DESCRIPTIONS.items():
            assert f'{kind} = {desc}' in text


class TestDelegateCap:
    def _history(self, rounds: int, text_path: bool = False) -> list:
        msgs: list = [{'role': 'user', 'content': 'fix the sandbox tool'}]
        for i in range(rounds):
            if text_path:
                msgs.append({'role': 'assistant',
                             'content': '{"outcome": "delegate"}'})
            else:
                msgs.append({'role': 'assistant', 'content': ''})
                msgs.append({'role': 'tool', 'tool_name': 'plan',
                             'content': plan.delegated_text(
                                 [f'agent{i}'], [plan.Task('g')])})
            msgs.append({'role': 'user',
                         'content': f'[joined results]\n— agent{i}: done'})
        return msgs

    def test_delegations_in_counts_tool_results_or_deliveries(self) -> None:
        assert plan.delegations_in([]) == 0
        assert plan.delegations_in(self._history(3)) == 3
        assert plan.delegations_in(self._history(2, text_path=True)) == 2
        # A refusal or an answer ack is not a delegation.
        msgs = [{'role': 'tool', 'tool_name': 'plan',
                 'content': plan.refused_text(['x'])},
                {'role': 'tool', 'tool_name': 'plan',
                 'content': plan.ANSWER_ACK},
                {'role': 'tool', 'tool_name': 'read_file',
                 'content': plan.DELEGATED_PREFIX + 'not a plan'}]
        assert plan.delegations_in(msgs) == 0

    def test_cap_text_is_a_refusal_that_says_answer(self) -> None:
        text = plan.delegate_cap_text(3)
        assert text.startswith(plan.REFUSED_PREFIX)
        assert 'delegated 3 times' in text and 'outcome answer' in text
        assert not plan.is_reask(text)
        assert plan.MAX_DELEGATE_ROUNDS == 3


class TestParse:
    def test_answer(self) -> None:
        p, errors = plan.parse({'outcome': 'answer', 'answer': 'Hi.'})
        assert errors == [] and p == plan.Plan('answer', answer='Hi.')

    def test_answer_ignores_tasks(self) -> None:
        p, errors = plan.parse({'outcome': 'answer', 'answer': 'Hi.',
                                'tasks': [_task()]})
        assert errors == [] and p.tasks == []

    def test_answer_without_text_is_never_rejected(self) -> None:
        # The loop takes the round's own text instead (turn._answer_text).
        for args in ({'outcome': 'answer'},
                     {'outcome': 'answer', 'answer': '  '},
                     {'outcome': 'answer', 'answer': 3}):
            p, errors = plan.parse(args)
            assert p == plan.Plan('answer', answer='') and errors == []
            assert plan.evaluate('review for correctness and security',
                                 args).ok

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
        ('fix the failing test in wordcount.py', []),   # verbs/artefacts
        ('hi, what can you do?', []),
        ('check error handling and the README', ['reliability', 'docs']),
        ('is the design performant?', ['design']),   # no fuzzy matching
        ('secure-by-default settings', []),          # hyphenated compound
        ('run the fast tests and fix the failure', ['tests']),
        ('Add docstrings and comments to auth.py', ['docs']),
        ('does the test suite cover the retry path?', ['tests']),
    ])
    def test_concerns_in(self, request_text, named) -> None:
        assert plan.concerns_in(request_text) == named

    @pytest.mark.parametrize('term', [
        'fix', 'fast', 'test', 'comments', 'structure', 'failure', 'auth',
        'secret', 'retry', 'timeout'])
    def test_ambiguous_triggers_are_out(self, term) -> None:
        assert plan.concerns_in(term) == []
        assert not any(term in terms for terms in plan.CONCERNS.values())

    @pytest.mark.parametrize('request_text, named', [
        ('run the fast tests and fix the failure', []),
        ('Add docstrings and comments to auth.py', []),
        ('Review this repository for correctness and security',
         ['correctness', 'security']),
        ('check correctness, security & performance of app/',
         ['correctness', 'security', 'performance']),
        ('Is the design sound? Also review the docs.', ['design', 'docs']),
        ('add docstrings to the security module', []),      # no coordinator
        ('review the docs of the tests directory', []),     # no coordinator
        ('security of the design', []),
        ('review app/ for bugs and for more bugs', []),     # one concern
        ('', []),
    ])
    def test_coverage_concerns_need_two_and_a_coordinator(
            self, request_text, named) -> None:
        assert plan.coverage_concerns(request_text) == named

    def test_coverage_skips_a_request_without_a_coordinator(self) -> None:
        # Two vocabulary words, one task: no re-ask for the "missing" one.
        request = 'add docstrings to the security module'
        args = {'outcome': 'delegate', 'tasks': [
            _task(goal='write docstrings in app/security.py')]}
        assert plan.evaluate(request, args).ok

    def test_missing_concern_is_soft(self) -> None:
        request = 'Review this repository for correctness and security.'
        args = {'outcome': 'delegate', 'tasks': [
            _task(goal='review app/ for injection and authz', kind='review',
                  complexity='hard')]}
        v = plan.evaluate(request, args)
        assert v.plan is not None and v.errors == []
        assert v.missing == ['correctness'] and not v.ok and v.soft
        # One task for two coordinated concerns is also undersplit.
        assert v.undersplit == ['correctness', 'security']
        msg, split = v.messages
        assert msg.startswith("no task goal covers the concern 'correctness'")
        assert 'one task per concern' in split
        assert plan.validate(request, v.plan) == [msg, split]

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


class TestProseAround:
    def test_bare_json_is_removed(self) -> None:
        text = 'Glad to help.\n{"outcome": "answer", "answer": ""} bye'
        assert plan.prose_around(text) == 'Glad to help.\n  bye'

    def test_fence_is_removed_whole(self) -> None:
        text = 'Sure.\n```json\n{"outcome": "answer"}\n```\n'
        assert plan.prose_around(text) == 'Sure.'

    def test_no_plan_object_is_the_text(self) -> None:
        assert plan.prose_around('  hello {"x": 1} ') == 'hello {"x": 1}'
        assert plan.prose_around('') == ''

    def test_only_json_is_empty(self) -> None:
        assert plan.prose_around('{"outcome": "answer"}') == ''


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


class TestUndersplit:
    """One worker per coordinated concern is enforced in code."""

    REQ = 'Review this repository for correctness and security'

    def _plan(self, goals):
        return plan.Plan('delegate', tasks=[
            plan.Task(goal=g, kind='review', complexity='standard')
            for g in goals])

    def test_one_task_naming_both_concerns_is_undersplit(self) -> None:
        p = self._plan(['review for correctness and security'])
        assert plan.missing_concerns(self.REQ, p) == []
        assert plan.undersplit_concerns(self.REQ, p) == [
            'correctness', 'security']
        v = plan.Verdict(p, [], [], plan.undersplit_concerns(self.REQ, p))
        assert not v.ok and v.soft
        assert 'spawn one task per concern' in v.messages[0]

    def test_two_tasks_are_fine(self) -> None:
        p = self._plan(['review correctness', 'review security'])
        assert plan.undersplit_concerns(self.REQ, p) == []
        assert plan.evaluate(self.REQ, {
            'outcome': 'delegate', 'tasks': [
                {'goal': 'review correctness', 'kind': 'review',
                 'complexity': 'standard'},
                {'goal': 'review security', 'kind': 'review',
                 'complexity': 'standard'}]}).ok

    def test_single_concern_or_answer_never_undersplit(self) -> None:
        one = self._plan(['fix the failing test'])
        assert plan.undersplit_concerns('fix the failing test', one) == []
        assert plan.undersplit_concerns(
            self.REQ, plan.Plan('answer', answer='hi')) == []

    def test_validate_lists_it_once(self) -> None:
        p = self._plan(['review for correctness and security'])
        msgs = plan.validate(self.REQ, p)
        assert len(msgs) == 1 and 'one task per concern' in msgs[0]
        assert plan.validate(self.REQ, p, followup=True) == []
