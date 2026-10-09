"""How the usage store is fed: topics begin on user turns and carry to
workers, call rows carry the topic, the topic labeler routes to the
cheapest rung, and the CLI starts (or points at) the dashboard."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from guru import config, session
from guru.domain import ledger, pricing
from tests.test_orchestrator import _FAKE_ENV
from guru.adapters import turn
from tests.test_turn_loop import Scripted


def _run(s: Scripted) -> None:
    """One turn through ``run_loop`` (where a topic begins)."""
    turn.run_loop(step=s._step, run_tools=s._run_tools,
                  add_user=s._add_user)


class TestTopicsInTheTurnLoop:
    def test_a_user_turn_begins_a_topic(self, monkeypatch,
                                        fake_repo) -> None:
        s = Scripted(monkeypatch, ['ok'], request='fix the login redirect')
        monkeypatch.setattr(session, 'topic_id', '')
        _run(s)
        ledger.flush()
        [row] = fake_repo.stream('topics')
        assert row['request'] == 'fix the login redirect'
        assert session.topic_id == row['topic_id'] == session.turn_id

    def test_a_mailbox_turn_keeps_the_topic(self, monkeypatch,
                                            fake_repo) -> None:
        s = Scripted(monkeypatch, ['ok'])
        monkeypatch.setattr(session, 'messages', [
            {'role': 'system', 'content': 's'},
            {'role': 'user', 'content': 'build it'},
            {'role': 'assistant', 'content': ''},
            {'role': 'user', 'content': '[joined results]\n— a1: ok'}])
        monkeypatch.setattr(session, 'topic_id', 'first-turn')
        _run(s)
        ledger.flush()
        assert fake_repo.stream('topics') == []
        assert session.topic_id == 'first-turn'
        assert session.turn_id != 'first-turn'

    def test_a_worker_turn_begins_no_topic(self, monkeypatch,
                                           fake_repo) -> None:
        s = Scripted(monkeypatch, ['ok'], task_id='k1')
        monkeypatch.setattr(session, 'topic_id', 'parent-topic')
        _run(s)
        ledger.flush()
        assert fake_repo.stream('topics') == []
        assert session.topic_id == 'parent-topic'


class TestTopicOnRows:
    def test_calls_carry_the_topic(self, monkeypatch, fake_repo) -> None:
        monkeypatch.setattr(session, 'topic_id', 'topic-7')
        ledger.record_call(adapter='A', model='m', usage=pricing.Usage(1, 1),
                           seconds=0.1, phase='step')
        ledger.flush()
        assert fake_repo.stream('calls')[0]['topic_id'] == 'topic-7'

    def test_children_inherit_the_topic(self, monkeypatch,
                                        fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        monkeypatch.setattr(ledger, 'environment', lambda: dict(_FAKE_ENV))
        o = Orchestrator()
        main = o.manager.active
        main.state.turn_id, main.state.topic_id = 'T2', 'T1'
        child = o._make_child(main, task='review auth')
        assert child.state.topic_id == 'T1'
        assert child.task_rec.topic_id == 'T1'
        ledger.flush()
        assert fake_repo.stream('tasks')[0]['topic_id'] == 'T1'


class TestRoutedTopicLabeler:
    def _patch(self, monkeypatch, picked, reply='{"topic": "login fix"}'):
        from guru.judges import llm
        seen: dict = {}

        class Adapter:
            remote = True

            def complete(self, prompt, max_tokens=0, model=''):
                seen.update(prompt=prompt, model=model)
                return reply

        def routed(text, adapter, model, kind, complexity, fallback=True):
            seen.update(kind=kind, complexity=complexity, fallback=fallback)
            return (SimpleNamespace(adapter=Adapter(), model='haiku')
                    if picked else None)
        monkeypatch.setattr(llm, 'routed_reviewer', routed)
        return seen

    def test_cheapest_rung_no_fallback(self, monkeypatch) -> None:
        from guru.judges.topic import RoutedTopicLabeler
        seen = self._patch(monkeypatch, picked=True)
        assert RoutedTopicLabeler().label('fix the login') == 'login fix'
        assert (seen['kind'], seen['complexity'], seen['fallback']) == \
            ('explain', 'trivial', False)
        assert seen['model'] == 'haiku' and 'fix the login' in seen['prompt']

    def test_no_rung_no_label(self, monkeypatch) -> None:
        from guru.judges.topic import RoutedTopicLabeler
        self._patch(monkeypatch, picked=False)
        assert RoutedTopicLabeler().label('x') is None

    def test_a_local_rung_does_not_label(self, monkeypatch) -> None:
        from guru.judges import llm
        from guru.judges.topic import RoutedTopicLabeler

        class Local:
            remote = False

            def complete(self, *a, **k):
                raise AssertionError('labelled on a local model')
        monkeypatch.setattr(llm, 'routed_reviewer',
                            lambda *a, **k: SimpleNamespace(adapter=Local(),
                                                            model='qwen3'))
        assert RoutedTopicLabeler().label('x') is None

    def test_no_main_model_last_resort(self, monkeypatch) -> None:
        """routed_reviewer without fallback never offers routing the
        session model: an emptied ladder yields None."""
        from guru.domain import routing
        from guru.judges import llm
        from guru.repositories.settings import RoutingSettings
        seen: dict = {}

        class Registry:
            def get(self, name):
                return None

            def is_remote(self, name):
                return False

        def resolve(*a, local_main=None, **k):
            seen['local_main'] = local_main
            return SimpleNamespace(refused=True, needs_confirmation=False,
                                   reason=['emptied'])
        monkeypatch.setattr(llm, '_registry', Registry())
        monkeypatch.setattr(llm, '_routing', RoutingSettings())
        monkeypatch.setattr(llm.routing_settings, 'ladders_from_settings',
                            lambda cfg, reg: {'default': ['rung']})
        monkeypatch.setattr(routing, 'resolve', resolve)
        main = SimpleNamespace(name='Ollama')
        assert llm.routed_reviewer('t', main, 'gpt-oss', 'explain',
                                   'trivial', fallback=False) is None
        assert seen['local_main'] is None
        llm.routed_reviewer('t', main, 'gpt-oss', 'review', 'standard')
        assert seen['local_main'] is not None      # the gate keeps it

    def test_garbage_reply_no_label(self, monkeypatch) -> None:
        from guru.judges.topic import RoutedTopicLabeler
        self._patch(monkeypatch, picked=True, reply='sure thing')
        assert RoutedTopicLabeler().label('x') is None


class TestCliDashboard:
    def test_off_without_store_or_setting(self, monkeypatch) -> None:
        from guru import cli
        monkeypatch.setattr(cli, 'USAGE', None)
        assert cli._start_dashboard() == 'off'
        monkeypatch.setattr(cli, 'USAGE', object())
        monkeypatch.setattr(config, 'DASHBOARD_ENABLED', False)
        assert cli._start_dashboard() == 'off'

    def test_starts_and_reports(self, monkeypatch, tmp_path) -> None:
        from guru import cli
        from guru.repositories.usage_sqlite import SqliteUsage
        from tests.test_dashboard import _free_port
        monkeypatch.setattr(cli, 'USAGE', SqliteUsage(tmp_path / 'u.db'))
        monkeypatch.setattr(config, 'DASHBOARD_ENABLED', True)
        monkeypatch.setattr(config, 'DASHBOARD_PORT', _free_port())
        try:
            status = cli._start_dashboard()
            assert status.startswith('serving at http://127.0.0.1:')
        finally:
            cli.DASHBOARD.stop()
            monkeypatch.setattr(cli, 'DASHBOARD', None)


@pytest.mark.parametrize('text, port, labels, enabled', [
    ('[dashboard]\nport = 8100\ntopic_labels = false\n', 8100, False, True),
    ('[dashboard]\nport = 80\nenabled = false\n', 7340, True, False),
])
def test_dashboard_settings(tmp_path, monkeypatch, text, port, labels,
                            enabled) -> None:
    path = tmp_path / 'settings.toml'
    path.write_text(text)
    monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', path)
    for name in ('DASHBOARD_PORT', 'TOPIC_LABELS', 'DASHBOARD_ENABLED',
                 'USAGE_DB'):
        monkeypatch.setattr(config, name, getattr(config, name))
    config._apply_settings()
    assert (config.DASHBOARD_PORT, config.TOPIC_LABELS,
            config.DASHBOARD_ENABLED, config.USAGE_DB) == \
        (port, labels, enabled, True)


def test_rubric_judge_calls_reach_the_store_as_eval(monkeypatch,
                                                    tmp_path) -> None:
    from guru.evals import runner
    from guru.repositories.usage_sqlite import SqliteUsage
    monkeypatch.setattr(runner, '_EVAL_USAGE', None)
    with runner._usage_only():
        ledger.record_call(adapter='A', model='judge', seconds=0.1,
                           usage=pricing.Usage(10, 1), phase='complete')
    store = SqliteUsage(config.USAGE_DB_PATH)
    assert store.totals(None, 'eval')['calls'] == 1
    assert store.totals(None, 'cli')['calls'] == 0
    assert ledger.repository() is None or not isinstance(
        ledger.repository(), SqliteUsage)


class TestTopicSeams:
    def test_a_retry_keeps_the_original_request(self, monkeypatch,
                                                fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        monkeypatch.setattr(ledger, 'environment', lambda: dict(_FAKE_ENV))
        o = Orchestrator()
        main = o.manager.active
        main.state.turn_id, main.state.topic_id = 'T1', 'T1'
        child = o._make_child(main, task='review auth')
        rec = child.task_rec
        main.state.turn_id, main.state.topic_id = 'T2', 'T2'   # moved on
        plan = o._plan_child(main, child.task, rec.kind, rec.complexity)
        retry = o._retry_child(child, rec, plan)
        assert (retry.state.turn_id, retry.state.topic_id) == ('T1', 'T1')
        assert retry.task_rec.topic_id == 'T1'

    def test_review_panel_gets_its_own_topic(self, monkeypatch,
                                             fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        o = Orchestrator()
        main = o.manager.active
        main.state.turn_id, main.state.topic_id = 'old', 'old'
        o.begin_request(main, '/review guru/auth.py')
        ledger.flush()
        [row] = fake_repo.stream('topics')
        assert row['request'] == '/review guru/auth.py'
        assert main.state.topic_id == row['topic_id'] != 'old'
