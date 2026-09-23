"""Tests for the eval runner endpoint and CLI (guru.evals.runner/__main__).

``BenchRun.run`` is monkeypatched to return canned agents, so nothing here
needs Ollama or a network; the fixture pytest runs really do run.
"""
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from guru import bench, config, session
from guru.adapters import turn
from guru.adapters.base import Adapter
from guru.agents import Agent
from guru.domain import files, ledger, tools
from guru.evals import cases, runner, runs
from guru.evals.__main__ import main as cli_main
from guru.evals.cases import Case, Expect
from guru.repositories.jsonl_ledger import JsonlLedger
from guru.repositories.settings import RoutingSettings

FIXTURES = Path(__file__).resolve().parents[1] / 'evals' / 'fixtures'


def _git(copy: Path, *args: str) -> str:
    return subprocess.run(['git', *args], cwd=copy, capture_output=True,
                          text=True, check=True).stdout


def _agent(title: str, messages: list, role=None, parent=None) -> Agent:
    a = Agent(id=title, title=title)
    a.state.messages = messages
    a.state.active_role = role
    a.state.model = 'fake-model'
    a.parent = parent
    return a


def _canned_agents() -> list:
    main = _agent('main', [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': 'the prompt'},
        {'role': 'tool', 'tool_name': 'search_code', 'content': 'hits'},
        {'role': 'user', 'content': turn._NUDGE_TEXT},
        {'role': 'tool', 'tool_name': 'spawn', 'content': 'ok'},
        {'role': 'assistant', 'content': 'Found path traversal in upload.py'},
    ])
    child = _agent('agent1', [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': 'task'},
        {'role': 'tool', 'tool_name': 'read_file', 'content': 'text'},
        {'role': 'user', 'content': turn._NUDGE_TEXT},
        {'role': 'assistant', 'content': 'child answer'},
    ], role='security-engineer', parent=main)
    return [main, child]


def _case(name='review', fixture='flaskish', **expect) -> Case:
    return Case(name=name, fixture=fixture, prompt='the prompt',
                timeout_s=100, expect=Expect(**expect))


class FakeAdapter(Adapter):
    name = 'Fake'

    def __init__(self) -> None:
        self.activated: list = []

    def available(self):
        return True

    def list_models(self):
        return []

    def activate(self, m):
        self.activated.append(m)
        # Like the Ollama adapter: a pinned --num-ctx skips the auto-fit
        # and loads at that size; otherwise pretend the GPU fit chose 40k.
        self.override_seen = session.num_ctx_override
        session.model = m
        session.num_ctx = session.num_ctx_override or 40960

    def summarise(self, t):
        return 's'

    def run_turn(self):
        pass


@pytest.fixture
def canned(monkeypatch):
    """Make ``BenchRun.run`` return canned agents; record the call."""
    seen: dict = {}

    async def fake_run(self, prompt, timeout=None):
        seen['prompt'] = prompt
        seen['timeout'] = timeout
        seen['base'] = self.base
        seen['cwd'] = os.getcwd()
        seen['has_upload'] = os.path.isfile('app/upload.py')
        seen['mode'] = config.MODE
        seen['ledger_enabled'] = config.LEDGER_ENABLED
        seen['persisters'] = (config.persist_read_dir,
                              config.persist_write_dir,
                              config.persist_domain)
        seen['read'] = set(config.ALLOWED_READ_DIRS)
        seen['write'] = set(config.ALLOWED_WRITE_DIRS)
        seen['repo'] = ledger.repository()
        seen['askers'] = (tools._domain_asker, files._path_asker)
        return seen.get('agents', _canned_agents())

    monkeypatch.setattr(bench.BenchRun, 'run', fake_run)
    return seen


def _base() -> session.SessionState:
    st = session.SessionState()
    st.adapter = FakeAdapter()
    st.model = 'base-model'
    return st


class TestPrepareFixture:
    def test_copies_and_inits_git(self, tmp_path: Path) -> None:
        copy = runner.prepare_fixture('cli-tool', tmp_path)
        assert copy == tmp_path / 'cli-tool'
        assert (copy / 'wordcount.py').is_file()
        assert (copy / 'tests' / 'test_wordcount.py').is_file()
        assert (copy / '.git').is_dir()
        assert _git(copy, 'status', '--porcelain') == ''
        assert _git(copy, 'rev-list', '--count', 'HEAD').strip() == '1'
        assert runner.files_changed(copy) == []

    def test_caches_are_not_copied(self, tmp_path: Path) -> None:
        copy = runner.prepare_fixture('flaskish', tmp_path)
        assert not list(copy.rglob('__pycache__'))

    def test_unknown_fixture_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='nope'):
            runner.prepare_fixture('nope', tmp_path)

    def test_custom_fixtures_dir(self, tmp_path: Path) -> None:
        src = tmp_path / 'fx' / 'mini'
        src.mkdir(parents=True)
        (src / 'a.txt').write_text('a')
        copy = runner.prepare_fixture('mini', tmp_path / 'work',
                                      fixtures_dir=tmp_path / 'fx')
        assert (copy / 'a.txt').read_text() == 'a'


class TestFilesChanged:
    def test_detects_modified_and_untracked(self, tmp_path: Path) -> None:
        copy = runner.prepare_fixture('cli-tool', tmp_path)
        (copy / 'wordcount.py').write_text('changed\n')
        (copy / 'new.txt').write_text('new\n')
        (copy / 'sub').mkdir()
        (copy / 'sub' / 'deep.txt').write_text('x\n')
        assert runner.files_changed(copy) == [
            'new.txt', 'sub/deep.txt', 'wordcount.py']

    def test_ignores_pycache(self, tmp_path: Path) -> None:
        copy = runner.prepare_fixture('cli-tool', tmp_path)
        (copy / '__pycache__').mkdir()
        (copy / '__pycache__' / 'x.pyc').write_bytes(b'\x00')
        assert runner.files_changed(copy) == []


class TestFixtureTests:
    def test_cli_tool_fails_and_flaskish_passes(self, tmp_path: Path):
        assert runner.fixture_tests_pass(
            runner.prepare_fixture('cli-tool', tmp_path)) is False
        assert runner.fixture_tests_pass(
            runner.prepare_fixture('flaskish', tmp_path)) is True


class TestRunCase:
    def test_observed_and_result(self, tmp_path: Path, canned) -> None:
        case = _case(tools_used_any=['search_code'], spawned_min=1,
                     roles_include=['security-engineer'],
                     stall_nudges_max=2,
                     answer_contains=['path traversal'], files_changed=[])
        base = _base()
        res = runner.run_case(case, base, [base.adapter], tmp_path / 'out')
        assert isinstance(res, runs.CaseResult)
        assert res.case == 'review'
        assert res.passed is True
        assert [c['name'] for c in res.checks] == [
            'tools_used_any', 'spawned_min', 'roles_include',
            'stall_nudges_max', 'answer_contains', 'files_changed']
        assert all(c['passed'] for c in res.checks)
        obs = res.observed
        assert obs['answer'] == 'Found path traversal in upload.py'
        assert obs['tools_used'] == ['search_code', 'spawn', 'read_file']
        assert obs['spawned'] == 1
        assert obs['roles'] == ['security-engineer']
        assert obs['stall_nudges'] == 2
        assert obs['files_changed'] == []
        assert obs['fixture_tests_pass'] is None
        assert obs['timed_out'] is False
        assert obs['error'] == ''
        assert obs['seconds'] >= 0
        assert res.rubric == ''
        assert res.cost_usd is None
        # the run saw the prompt, timeout and base state
        assert canned['prompt'] == 'the prompt'
        assert canned['timeout'] == 100
        assert canned['base'] is base

    def test_environment_during_run(self, tmp_path: Path, canned,
                                    monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        case = _case()
        case.mode = config.MODE_AUTO
        base = _base()
        runner.run_case(case, base, [base.adapter], tmp_path / 'out')
        cwd = Path(canned['cwd']).resolve()
        assert cwd.name == 'flaskish'
        assert canned['has_upload'] is True
        assert not cwd.exists()            # temp copy cleaned up
        assert canned['mode'] == config.MODE_AUTO
        assert str(cwd) in canned['read']
        assert str(cwd) in canned['write']
        assert isinstance(canned['repo'], JsonlLedger)
        assert canned['repo'].dir == tmp_path / 'out' / 'ledger'
        ask_domain, ask_path = canned['askers']
        assert ask_domain('web?') is False and ask_path('path?') is False
        assert canned['ledger_enabled'] is True
        for fn in canned['persisters']:
            assert fn is runner._no_persist

    def test_restores_everything(self, tmp_path: Path, canned,
                                 monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_READ_ONLY)
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        read_before = set(config.ALLOWED_READ_DIRS)
        write_before = set(config.ALLOWED_WRITE_DIRS)
        domains_before = set(config.ALLOWED_DOMAINS)
        persisters_before = (config.persist_read_dir,
                             config.persist_write_dir, config.persist_domain)
        cwd_before = os.getcwd()
        prev_repo = object()
        ledger.set_repository(prev_repo)   # type: ignore[arg-type]
        tools.set_domain_asker(None)
        files.set_path_asker(None)
        try:
            case = _case()
            case.mode = config.MODE_AUTO
            base = _base()
            runner.run_case(case, base, [base.adapter], tmp_path / 'out')
            assert os.getcwd() == cwd_before
            assert config.MODE == config.MODE_READ_ONLY
            assert set(config.ALLOWED_READ_DIRS) == read_before
            assert set(config.ALLOWED_WRITE_DIRS) == write_before
            assert set(config.ALLOWED_DOMAINS) == domains_before
            assert (config.persist_read_dir, config.persist_write_dir,
                    config.persist_domain) == persisters_before
            assert config.LEDGER_ENABLED is False
            assert ledger.repository() is prev_repo
            assert tools._domain_asker is None
            assert files._path_asker is None
        finally:
            ledger.set_repository(None)

    def test_writes_transcript_and_ledger_dir(self, tmp_path: Path,
                                              canned) -> None:
        out = tmp_path / 'out'
        base = _base()
        res = runner.run_case(_case(), base, [base.adapter], out)
        tpath = out / 'transcripts' / 'review.json.gz'
        assert tpath.is_file()
        assert res.transcript_path == str(tpath)
        with gzip.open(tpath, 'rt', encoding='utf-8') as fh:
            data = json.load(fh)
        assert [a['title'] for a in data] == ['main', 'agent1']
        assert data[0]['messages'][-1]['content'] == \
            'Found path traversal in upload.py'
        assert data[1]['messages'][2]['tool_name'] == 'read_file'
        assert (out / 'ledger').is_dir()

    def test_cost_from_ledger_calls(self, tmp_path: Path, canned,
                                    monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        out = tmp_path / 'out'

        async def run_and_record(self, prompt, timeout=None):
            ledger.submit('calls', {'cost_usd': 0.25})
            ledger.submit('calls', {'cost_usd': 0.5})
            return _canned_agents()

        monkeypatch.setattr(bench.BenchRun, 'run', run_and_record)
        base = _base()
        res = runner.run_case(_case(), base, [base.adapter], out)
        assert res.cost_usd == 0.75
        # a second case in the same out dir only counts its own rows
        res2 = runner.run_case(_case(name='second'), base, [base.adapter],
                               out)
        assert res2.cost_usd == 0.75

    def test_cost_none_when_a_row_is_unknown(self, tmp_path: Path,
                                             monkeypatch) -> None:
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)

        async def run_and_record(self, prompt, timeout=None):
            ledger.submit('calls', {'cost_usd': 0.25})
            ledger.submit('calls', {'cost_usd': None})
            return _canned_agents()

        monkeypatch.setattr(bench.BenchRun, 'run', run_and_record)
        base = _base()
        res = runner.run_case(_case(), base, [base.adapter], tmp_path)
        assert res.cost_usd is None

    def test_files_changed_detected(self, tmp_path: Path,
                                    monkeypatch) -> None:
        async def edit_then_answer(self, prompt, timeout=None):
            Path('wordcount.py').write_text('def count_words(t): return 0\n')
            return [_agent('main', [
                {'role': 'assistant', 'content': 'fixed'}])]

        monkeypatch.setattr(bench.BenchRun, 'run', edit_then_answer)
        case = _case(fixture='cli-tool', files_changed=['wordcount.py'],
                     files_unchanged=['tests/test_wordcount.py'])
        base = _base()
        res = runner.run_case(case, base, [base.adapter], tmp_path)
        assert res.observed['files_changed'] == ['wordcount.py']
        assert res.passed is True

    def test_fixture_tests_run_when_configured(self, tmp_path: Path,
                                               canned) -> None:
        base = _base()
        bad = runner.run_case(
            _case(fixture='cli-tool', fixture_tests_pass=True),
            base, [base.adapter], tmp_path)
        assert bad.observed['fixture_tests_pass'] is False
        assert bad.passed is False
        assert bad.checks[0]['detail'] == 'fixture tests failed, expected pass'
        good = runner.run_case(
            _case(fixture='flaskish', fixture_tests_pass=True),
            base, [base.adapter], tmp_path)
        assert good.observed['fixture_tests_pass'] is True
        assert good.passed is True

    def test_timeout_mirrors_bench(self, tmp_path: Path, canned,
                                   monkeypatch) -> None:
        # Patch the runner's own ``time`` reference (not the time module):
        # asyncio and the ledger thread also call time.monotonic, so a
        # global patch would leak into them. The first tick is the run
        # start, every later call (end of run, worker drain) sees 150 s.
        from types import SimpleNamespace
        ticks = [0.0, 150.0]
        monkeypatch.setattr(runner, 'time', SimpleNamespace(
            monotonic=lambda: ticks.pop(0) if len(ticks) > 1 else ticks[0],
            sleep=lambda s: None))
        base = _base()
        res = runner.run_case(_case(answer_contains=['path']), base,
                              [base.adapter], tmp_path)
        assert res.observed['timed_out'] is True
        assert res.observed['seconds'] == 150.0
        assert res.passed is False
        assert res.checks[0]['detail'] == 'timeout'

    def test_run_error_is_captured(self, tmp_path: Path, monkeypatch,
                                   ) -> None:
        async def boom(self, prompt, timeout=None):
            raise RuntimeError('adapter exploded')

        monkeypatch.setattr(bench.BenchRun, 'run', boom)
        base = _base()
        res = runner.run_case(_case(answer_contains=['x']), base,
                              [base.adapter], tmp_path)
        assert res.passed is False
        assert res.observed['error'] == 'adapter exploded'
        assert res.observed['answer'] == ''
        assert res.checks[0]['detail'] == 'error: adapter exploded'
        assert os.getcwd() != str(tmp_path)
        assert (tmp_path / 'transcripts' / 'review.json.gz').is_file()

    def test_empty_answer_is_an_error(self, tmp_path: Path, canned) -> None:
        canned['agents'] = [_agent('main', [
            {'role': 'user', 'content': 'p'},
            {'role': 'tool', 'tool_name': 'read_file', 'content': 'x'}])]
        base = _base()
        res = runner.run_case(_case(tools_used_any=['read_file']), base,
                              [base.adapter], tmp_path)
        assert res.observed['error'] == 'empty answer'
        assert res.passed is False

    def test_rubric_carried(self, tmp_path: Path, canned) -> None:
        base = _base()
        res = runner.run_case(_case(rubric='Names both bugs.'), base,
                              [base.adapter], tmp_path)
        assert res.rubric == 'Names both bugs.'
        assert res.passed is True and res.checks == []

    def test_specific_model_uses_named_adapter(self, tmp_path: Path,
                                               canned) -> None:
        other = FakeAdapter()
        other.name = 'Other'
        base = _base()
        case = _case()
        case.model = 'Other|big-model'
        res = runner.run_case(case, base, [base.adapter, other], tmp_path)
        assert res.passed is True
        used = canned['base']
        assert used is not base
        assert used.adapter is other
        assert used.model == 'big-model'
        assert other.activated == ['big-model']
        assert base.model == 'base-model'

    def test_unknown_adapter_is_an_error(self, tmp_path: Path,
                                         canned) -> None:
        base = _base()
        case = _case()
        case.model = 'Missing|m'
        res = runner.run_case(case, base, [base.adapter], tmp_path)
        assert res.passed is False
        assert 'Missing' in res.observed['error']


class TestResolveBase:
    def test_default_prefers_cli_default_model(self, monkeypatch) -> None:
        from types import SimpleNamespace
        from guru import cli
        ollama = FakeAdapter()
        monkeypatch.setattr(bench, '_adapter_for',
                            lambda name, built: ollama if name is None
                            else None)
        ollama.list_models = lambda: [                # type: ignore
            SimpleNamespace(model_id='a'),
            SimpleNamespace(model_id=cli.DEFAULT_MODEL)]
        st, spec = runner.resolve_base(None, [ollama])
        assert st.adapter is ollama
        assert st.model == cli.DEFAULT_MODEL
        assert spec == f'Fake|{cli.DEFAULT_MODEL}'

    def test_default_falls_back_to_first_listed(self, monkeypatch) -> None:
        from types import SimpleNamespace
        ollama = FakeAdapter()
        monkeypatch.setattr(bench, '_adapter_for',
                            lambda name, built: ollama)
        ollama.list_models = lambda: [                # type: ignore
            SimpleNamespace(model_id='only-one')]
        st, _ = runner.resolve_base('default', [ollama])
        assert st.model == 'only-one'

    def test_default_when_listing_fails(self, monkeypatch) -> None:
        from guru import cli
        ollama = FakeAdapter()
        monkeypatch.setattr(bench, '_adapter_for',
                            lambda name, built: ollama)

        def fail():
            raise OSError('down')
        ollama.list_models = fail                     # type: ignore
        st, _ = runner.resolve_base(None, [ollama])
        assert st.model == cli.DEFAULT_MODEL

    def test_explicit_spec(self) -> None:
        a = FakeAdapter()
        st, spec = runner.resolve_base('Fake|m1', [a])
        assert st.adapter is a and st.model == 'm1' and spec == 'Fake|m1'
        assert a.activated == ['m1']

    def test_bad_spec(self) -> None:
        with pytest.raises(ValueError, match='Adapter|model'):
            runner.resolve_base('no-bar', [FakeAdapter()])
        with pytest.raises(ValueError, match='Nope'):
            runner.resolve_base('Nope|m', [FakeAdapter()])

    def test_activate_does_not_persist_model_ctx(self, tmp_path: Path,
                                                 monkeypatch) -> None:
        """The pin (or the fit) must not land in ~/.guru/model_ctx.json."""
        monkeypatch.setattr(config, 'MODEL_CTX_PATH',
                            tmp_path / 'model_ctx.json')
        saver_before = config.save_model_ctx

        class Persisting(FakeAdapter):
            def activate(self, m):
                super().activate(m)
                config.save_model_ctx(m, session.num_ctx)   # as Ollama does

        a = Persisting()
        st, _ = runner.resolve_base('Fake|m1', [a], num_ctx=8192)
        assert st.num_ctx == 8192
        assert not (tmp_path / 'model_ctx.json').exists()
        assert config.save_model_ctx is saver_before
        # the real saver still works afterwards
        config.save_model_ctx('m1', 4096)
        assert config.load_model_ctx() == {'m1': 4096}

    def test_activate_restores_saver_on_error(self) -> None:
        saver_before = config.save_model_ctx

        class Boom(FakeAdapter):
            def activate(self, m):
                raise RuntimeError('down')

        with pytest.raises(RuntimeError, match='down'):
            runner.resolve_base('Fake|m1', [Boom()])
        assert config.save_model_ctx is saver_before

    def test_num_ctx_pin_is_set_before_activate(self) -> None:
        a = FakeAdapter()
        st, _ = runner.resolve_base('Fake|m1', [a], num_ctx=8192)
        assert a.override_seen == 8192
        assert st.num_ctx_override == 8192
        assert st.num_ctx == 8192

    def test_num_ctx_zero_leaves_auto_fit(self) -> None:
        a = FakeAdapter()
        st, _ = runner.resolve_base('Fake|m1', [a])
        assert a.override_seen == 0
        assert st.num_ctx_override == 0
        assert st.num_ctx == 40960


class TestRunSuite:
    def test_runs_cases_saves_and_appends(self, tmp_path: Path, canned,
                                          monkeypatch) -> None:
        base = _base()
        out = tmp_path / 'runs'
        suite = [_case(name='a', answer_contains=['path traversal']),
                 _case(name='b', answer_contains=['nope'])]
        run = runner.run_suite(suite, 'Fake|base-model', out,
                               base_state=base, adapters=[base.adapter],
                               note='first', trajectory_dir=tmp_path)
        assert isinstance(run, runs.Run)
        assert run.model == 'Fake|base-model'
        assert [c.case for c in run.cases] == ['a', 'b']
        assert [c.passed for c in run.cases] == [True, False]
        assert len(run.git_sha) in (40, 64)     # sha1 or sha256 repos
        saved = list(out.glob('*.json'))
        assert len(saved) == 1
        assert runs.load(saved[0]) == run
        assert (out / run.run_id / 'transcripts' / 'a.json.gz').is_file()
        assert (out / run.run_id / 'transcripts' / 'b.json.gz').is_file()
        assert not (out / runs.TRAJECTORY_FILE).exists()
        traj = (tmp_path / runs.TRAJECTORY_FILE).read_text()
        assert run.run_id in traj and '| 1/2 |' in traj and 'first' in traj

    def test_default_trajectory_is_evals_dir(self) -> None:
        assert runner.DEFAULT_TRAJECTORY_DIR == cases.REPO_ROOT / 'evals'
        import inspect
        sig = inspect.signature(runner.run_suite)
        assert sig.parameters['trajectory_dir'].default == \
            runner.DEFAULT_TRAJECTORY_DIR

    def test_resolves_base_when_not_given(self, tmp_path: Path, canned,
                                          monkeypatch) -> None:
        a = FakeAdapter()
        monkeypatch.setattr(bench, '_build_adapters', lambda: [a])
        run = runner.run_suite([_case(name='a')], 'Fake|m', tmp_path,
                               trajectory_dir=tmp_path)
        assert run.model == 'Fake|m'
        assert canned['base'].adapter is a
        assert a.activated == ['m']
        assert run.num_ctx == 40960            # what the adapter resolved

    def test_num_ctx_pin_recorded_and_in_trajectory(self, tmp_path: Path,
                                                    canned,
                                                    monkeypatch) -> None:
        a = FakeAdapter()
        monkeypatch.setattr(bench, '_build_adapters', lambda: [a])
        run = runner.run_suite([_case(name='a')], 'Fake|m', tmp_path,
                               trajectory_dir=tmp_path, num_ctx=8192)
        assert run.num_ctx == 8192
        assert canned['base'].num_ctx_override == 8192
        traj = (tmp_path / runs.TRAJECTORY_FILE).read_text()
        assert '| Fake\\|m@8k |' in traj
        assert runs.load(next(tmp_path.glob('*.json'))).num_ctx == 8192

    def test_case_model_inherits_num_ctx_pin(self, tmp_path: Path,
                                             canned) -> None:
        other = FakeAdapter()
        other.name = 'Other'
        base = _base()
        base.num_ctx_override = 8192
        case = _case()
        case.model = 'Other|big-model'
        runner.run_case(case, base, [base.adapter, other], tmp_path)
        assert other.override_seen == 8192
        assert canned['base'].num_ctx == 8192

    def test_loads_skill_catalog_once(self, tmp_path: Path, canned,
                                      monkeypatch) -> None:
        from guru import skills
        monkeypatch.setattr(config, 'GURU_SKILLS_DIR', tmp_path / 'skills')
        monkeypatch.setattr(skills, 'REGISTRY', {})
        base = _base()
        runner.run_suite([_case(name='a')], 'Fake|base-model', tmp_path,
                         base_state=base, adapters=[base.adapter],
                         trajectory_dir=tmp_path)
        assert skills.get('code-review') is not None
        assert skills.get('security-engineer') is not None

    def test_given_base_state_records_its_context(self, tmp_path: Path,
                                                  canned) -> None:
        base = _base()
        base.num_ctx = 16384
        run = runner.run_suite([_case(name='a')], 'Fake|base-model',
                               tmp_path, base_state=base,
                               adapters=[base.adapter],
                               trajectory_dir=tmp_path)
        assert run.num_ctx == 16384

    def test_git_sha_empty_on_failure(self, monkeypatch) -> None:
        def fail(*a, **k):
            raise OSError('no git')
        monkeypatch.setattr(runner.subprocess, 'run', fail)
        assert runner.git_sha() == ''


class TestCli:
    def test_list_prints_case_names(self, capsys) -> None:
        assert cli_main(['list']) == 0
        out = capsys.readouterr().out
        names = [c.name for c in cases.load_cases(cases.CASES_DIR)]
        assert names
        for n in names:
            assert n in out
        assert 'flaskish' in out

    def test_list_custom_dir(self, tmp_path: Path, capsys) -> None:
        (tmp_path / 'x.toml').write_text(
            'name = "x-case"\nfixture = "docs-only"\nprompt = "hi"\n')
        assert cli_main(['list', '--cases-dir', str(tmp_path)]) == 0
        assert 'x-case' in capsys.readouterr().out

    def test_run_exit_code_and_table(self, tmp_path: Path, capsys,
                                     monkeypatch) -> None:
        cdir = tmp_path / 'cases'
        cdir.mkdir()
        (cdir / 'a.toml').write_text(
            'name = "a"\nfixture = "docs-only"\nprompt = "hi"\n')
        (cdir / 'b.toml').write_text(
            'name = "b"\nfixture = "docs-only"\nprompt = "hi"\n')

        def fake_suite(suite, model_spec, out_root, base_state=None,
                       adapters=None, note='', on_result=None, num_ctx=0,
                       routing=None, routing_name='', allow_spend=False,
                       decisions=None):
            assert [c.name for c in suite] == ['a']
            assert decisions is None
            assert callable(on_result)
            assert model_spec == 'Fake|m'
            assert note == 'n1'
            assert num_ctx == 4096
            assert routing is None and routing_name == ''
            assert allow_spend is False
            r = runs.Run(run_id='rid', ts=runs.now_ts(), model=model_spec,
                         git_sha='', num_ctx=num_ctx, cases=[runs.CaseResult(
                             case='a', passed=False,
                             checks=[{'name': 'answer_contains',
                                      'passed': False,
                                      'detail': "answer lacks ['x']"}],
                             observed={'seconds': 2.5}, rubric='',
                             transcript_path='t', cost_usd=0.1)])
            runs.save(r, out_root)
            return r

        monkeypatch.setattr(runner, 'run_suite', fake_suite)
        code = cli_main(['run', '--cases', 'a', '--cases-dir', str(cdir),
                         '--model', 'Fake|m', '--out', str(tmp_path / 'r'),
                         '--note', 'n1', '--num-ctx', '4096'])
        assert code == 1
        out = capsys.readouterr().out
        assert 'FAIL' in out and 'answer_contains' in out and '2.5' in out
        assert '0/1' in out
        assert 'model Fake|m@4k' in out

    def test_run_tags_filter(self, tmp_path: Path, capsys,
                             monkeypatch) -> None:
        cdir = tmp_path / 'cases'
        cdir.mkdir()
        (cdir / 'a.toml').write_text(
            'name = "a"\nfixture = "docs-only"\nprompt = "hi"\n'
            'tags = ["fast"]\n')
        (cdir / 'b.toml').write_text(
            'name = "b"\nfixture = "docs-only"\nprompt = "hi"\n'
            'tags = ["edit"]\n')
        (cdir / 'c.toml').write_text(
            'name = "c"\nfixture = "docs-only"\nprompt = "hi"\n')
        seen: dict = {}

        def fake_suite(suite, model_spec, out_root, **kw):
            seen['names'] = [c.name for c in suite]
            return runs.Run(run_id='rid', ts=runs.now_ts(), model='m',
                            git_sha='', cases=[])

        monkeypatch.setattr(runner, 'run_suite', fake_suite)
        assert cli_main(['run', '--tags', 'fast, edit', '--cases-dir',
                         str(cdir), '--out', str(tmp_path)]) == 0
        assert seen['names'] == ['a', 'b']
        assert cli_main(['run', '--tags', 'fast', '--cases', 'a,c',
                         '--cases-dir', str(cdir),
                         '--out', str(tmp_path)]) == 0
        assert seen['names'] == ['a']
        assert cli_main(['run', '--tags', 'nope', '--cases-dir', str(cdir),
                         '--out', str(tmp_path)]) == 2
        assert 'nope' in capsys.readouterr().err

    def test_list_tags_filter(self, tmp_path: Path, capsys) -> None:
        (tmp_path / 'a.toml').write_text(
            'name = "a-case"\nfixture = "docs-only"\nprompt = "hi"\n'
            'tags = ["fast"]\n')
        (tmp_path / 'b.toml').write_text(
            'name = "b-case"\nfixture = "docs-only"\nprompt = "hi"\n')
        assert cli_main(['list', '--tags', 'fast',
                         '--cases-dir', str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert 'a-case' in out and 'b-case' not in out

    def test_run_defaults_come_from_settings(self, tmp_path: Path,
                                             monkeypatch) -> None:
        monkeypatch.setattr(config, 'EVALS_MODEL', 'Fake|from-settings')
        monkeypatch.setattr(config, 'EVALS_NUM_CTX', 2048)
        seen: dict = {}

        def fake_suite(suite, model_spec, out_root, **kw):
            seen['model'] = model_spec
            seen['num_ctx'] = kw.get('num_ctx')
            return runs.Run(run_id='rid', ts=runs.now_ts(), model='m',
                            git_sha='', cases=[])

        monkeypatch.setattr(runner, 'run_suite', fake_suite)
        assert cli_main(['run', '--out', str(tmp_path)]) == 0
        assert seen == {'model': 'Fake|from-settings', 'num_ctx': 2048}
        # flags win over settings; 0 means auto-fit
        assert cli_main(['run', '--out', str(tmp_path), '--model', 'Fake|x',
                         '--num-ctx', '0']) == 0
        assert seen == {'model': 'Fake|x', 'num_ctx': 0}

    def test_run_without_settings_uses_guru_default(self, tmp_path: Path,
                                                    monkeypatch) -> None:
        monkeypatch.setattr(config, 'EVALS_MODEL', '')
        seen: dict = {}

        def fake_suite(suite, model_spec, out_root, **kw):
            seen['model'] = model_spec
            return runs.Run(run_id='rid', ts=runs.now_ts(), model='m',
                            git_sha='', cases=[])

        monkeypatch.setattr(runner, 'run_suite', fake_suite)
        assert cli_main(['run', '--out', str(tmp_path)]) == 0
        assert seen['model'] is None

    def test_negative_num_ctx_is_usage_error(self, tmp_path: Path,
                                             capsys) -> None:
        assert cli_main(['run', '--out', str(tmp_path),
                         '--num-ctx', '-1']) == 2
        assert 'num-ctx' in capsys.readouterr().err

    def test_timeout_row_shows_observed_and_progress_flags_it(
            self, tmp_path: Path, capsys, monkeypatch) -> None:
        def fake_suite(suite, model_spec, out_root, on_result=None, **kw):
            res = runs.CaseResult(
                case='fix', passed=False,
                checks=[{'name': 'files_changed', 'passed': False,
                         'detail': 'timeout'}],
                observed={'seconds': 315.9, 'timed_out': True,
                          'tools_used': ['search_tools', 'read_file',
                                         'read_file', 'edit_file', 'spawn',
                                         'spawn'],
                          'spawned': 2, 'files_changed': ['wordcount.py']},
                rubric='', transcript_path='t', cost_usd=None)
            if on_result is not None:
                on_result(res)
            return runs.Run(run_id='rid', ts=runs.now_ts(), model='m',
                            git_sha='', cases=[res])

        monkeypatch.setattr(runner, 'run_suite', fake_suite)
        assert cli_main(['run', '--out', str(tmp_path)]) == 1
        out = capsys.readouterr().out
        assert '[evals] fix: FAIL (315.9s, timed out)' in out
        assert 'files_changed' in out
        assert ('timed out: tools=search_tools, read_file(2), edit_file, '
                'spawn(2); spawned=2; files_changed=wordcount.py') in out

    def test_timeout_row_without_tools(self) -> None:
        from guru.evals import __main__ as cli
        res = runs.CaseResult(
            case='x', passed=False, checks=[],
            observed={'seconds': 1.0, 'timed_out': True, 'tools_used': [],
                      'spawned': 0, 'files_changed': []},
            rubric='r', transcript_path='t', cost_usd=None)
        row = cli._row(res)
        assert row[-1] == ('timed out: tools=-; spawned=0; files_changed=-'
                           '; rubric: grade by hand')

    def test_run_passing_exits_zero(self, tmp_path: Path, capsys,
                                    monkeypatch) -> None:
        def fake_suite(suite, model_spec, out_root, **kw):
            return runs.Run(run_id='rid', ts=runs.now_ts(), model='m',
                            git_sha='', cases=[runs.CaseResult(
                                case=c.name, passed=True, checks=[],
                                observed={'seconds': 1.0}, rubric='r',
                                transcript_path='t', cost_usd=None)
                                for c in suite])

        monkeypatch.setattr(runner, 'run_suite', fake_suite)
        assert cli_main(['run', '--out', str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert 'PASS' in out and 'rubric' in out

    def test_run_unknown_case_is_usage_error(self, tmp_path: Path,
                                             capsys) -> None:
        assert cli_main(['run', '--cases', 'no-such-case',
                         '--out', str(tmp_path)]) == 2
        assert 'no-such-case' in capsys.readouterr().err

    def test_compare(self, tmp_path: Path, capsys) -> None:
        def res(name, ok, secs, cost):
            return runs.CaseResult(case=name, passed=ok, checks=[],
                                   observed={'seconds': secs}, rubric='',
                                   transcript_path='', cost_usd=cost)
        old = runs.Run('old1', '2026-09-23T10:00:00+00:00', 'm', '',
                       [res('a', False, 10, 1.0), res('b', True, 5, None)])
        new = runs.Run('new1', '2026-09-24T10:00:00+00:00', 'm', '',
                       [res('a', True, 8, 0.5), res('b', False, 6, None),
                        res('c', True, 1, None)])
        po = runs.save(old, tmp_path)
        pn = runs.save(new, tmp_path)
        assert cli_main(['compare', str(po), str(pn)]) == 0
        out = capsys.readouterr().out
        assert 'newly passing' in out and 'a' in out
        assert 'newly failing' in out and 'b' in out
        assert 'added' in out and 'c' in out
        assert '-2.0' in out and '-0.5' in out
        assert '50%' in out and '67%' in out

    def test_compare_bad_file(self, tmp_path: Path, capsys) -> None:
        p = tmp_path / 'x.json'
        p.write_text('{}')
        assert cli_main(['compare', str(p), str(p)]) == 2
        assert 'x.json' in capsys.readouterr().err

    def test_module_entry_point(self) -> None:
        proc = subprocess.run(
            [sys.executable, '-m', 'guru.evals', 'list'],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1])
        assert proc.returncode == 0
        assert 'greet' in proc.stdout


class TestEndToEnd:
    """The real BenchRun + orchestrator with a fake adapter that calls the
    real file tools inside the fixture copy (no model, no network)."""

    class ToolAdapter(FakeAdapter):
        def run_turn(self):
            st = session.current()
            out = tools.execute_tool('write_file', {
                'path': 'wordcount.py',
                'content': 'def count_words(text):\n'
                           '    return len(text.split())\n'})
            st.messages.append({'role': 'tool', 'tool_name': 'write_file',
                                'content': out})
            st.messages.append({'role': 'assistant',
                                'content': f'wrote: {out[:40]}'})

    def _run(self, tmp_path: Path, mode: str) -> runs.CaseResult:
        base = session.SessionState()
        base.adapter = self.ToolAdapter()
        base.model = 'fake'
        case = _case(fixture='cli-tool', files_changed=['wordcount.py'],
                     fixture_tests_pass=True)
        case.mode = mode
        try:
            return runner.run_case(case, base, [base.adapter], tmp_path)
        finally:
            # BenchRun.run clears its handlers itself; reset again here so
            # a failure inside the run can never leak them into other tests.
            tools.set_spawn_handler(None)
            tools.set_check_handler(None)
            tools.set_join_handler(None)

    def test_write_in_auto_mode_changes_the_copy(self, tmp_path: Path):
        res = self._run(tmp_path, config.MODE_AUTO)
        assert res.observed['error'] == ''
        assert res.observed['tools_used'] == ['write_file']
        assert res.observed['files_changed'] == ['wordcount.py']
        assert res.observed['fixture_tests_pass'] is True
        assert res.passed is True
        # the fixture itself is untouched
        assert 'split(\' \')' in (FIXTURES / 'cli-tool' /
                                  'wordcount.py').read_text()

    def test_read_only_mode_refuses_the_write(self, tmp_path: Path):
        res = self._run(tmp_path, config.MODE_READ_ONLY)
        assert res.observed['files_changed'] == []
        assert res.observed['fixture_tests_pass'] is False
        assert res.passed is False

    def test_auto_mode_contains_escalations(self, tmp_path: Path,
                                            monkeypatch) -> None:
        """C1: an auto case writing outside the copy is refused: the sandbox
        turns ``config.AUTO_GRANT`` off so the deny asker is consulted; the
        developer's allow-list files stay untouched, the in-memory lists are
        reset afterwards and the knob is restored."""
        allow = tmp_path / 'guru-home'
        allow.mkdir()
        monkeypatch.setattr(config, 'READ_DIRS_ALLOW_PATH',
                            allow / 'read.txt')
        monkeypatch.setattr(config, 'WRITE_DIRS_ALLOW_PATH',
                            allow / 'write.txt')
        monkeypatch.setattr(config, 'DOMAINS_ALLOW_PATH',
                            allow / 'domains.txt')
        outside = tmp_path / 'outside.txt'
        write_before = set(config.ALLOWED_WRITE_DIRS)
        assert config.AUTO_GRANT is True
        seen: dict = {}

        class Escalating(FakeAdapter):
            def run_turn(self):
                st = session.current()
                seen['auto_grant'] = config.AUTO_GRANT
                out = tools.execute_tool('write_file', {
                    'path': str(outside), 'content': 'leak\n'})
                st.messages.append({'role': 'tool',
                                    'tool_name': 'write_file',
                                    'content': out})
                st.messages.append({'role': 'assistant', 'content': out})

        base = session.SessionState()
        base.adapter = Escalating()
        base.model = 'fake'
        case = _case(fixture='docs-only', files_changed=[])
        case.mode = config.MODE_AUTO
        try:
            res = runner.run_case(case, base, [base.adapter],
                                  tmp_path / 'out')
        finally:
            tools.set_spawn_handler(None)
            tools.set_check_handler(None)
            tools.set_join_handler(None)
        assert res.observed['error'] == ''
        assert seen['auto_grant'] is False
        assert not outside.exists()                  # refused
        assert res.passed is True                    # files_changed == []
        assert not list(allow.iterdir())             # nothing persisted
        assert set(config.ALLOWED_WRITE_DIRS) == write_before
        assert str(tmp_path.resolve()) not in config.ALLOWED_WRITE_DIRS
        assert config.AUTO_GRANT is True


class TestGuards:
    def test_leaked_workers_keep_deny_askers(self, tmp_path: Path, canned,
                                             monkeypatch) -> None:
        """I1: a worker still busy after the run -> error, askers stay."""
        agents = _canned_agents()
        agents[1].busy = True
        canned['agents'] = agents
        monkeypatch.setattr(runner, 'WORKER_DRAIN_S', 0.3)
        monkeypatch.setattr(runner, '_WORKER_POLL_S', 0.05)
        tools.set_domain_asker(None)
        files.set_path_asker(None)
        mode_before, cwd_before = config.MODE, os.getcwd()
        base = _base()
        try:
            res = runner.run_case(_case(answer_contains=['path']), base,
                                  [base.adapter], tmp_path)
            assert res.observed['error'] == \
                'workers still running after timeout'
            assert res.passed is False
            assert tools._domain_asker is runner._deny
            assert files._path_asker is runner._deny
            # everything else is still restored
            assert config.MODE == mode_before and os.getcwd() == cwd_before
        finally:
            tools.set_domain_asker(None)
            files.set_path_asker(None)

    def test_prepare_failure_has_zero_seconds(self, tmp_path: Path,
                                              canned) -> None:
        """I2: the clock only covers the model run."""
        base = _base()
        res = runner.run_case(_case(fixture='no-such-fixture'), base,
                              [base.adapter], tmp_path)
        assert res.observed['seconds'] == 0.0
        assert 'no-such-fixture' in res.observed['error']
        assert res.observed['timed_out'] is False

    def test_refuses_running_event_loop(self, tmp_path: Path) -> None:
        """I3: synchronous entry points guard against a live loop."""
        import asyncio
        base = _base()

        async def inside():
            with pytest.raises(RuntimeError, match='running event loop'):
                runner.run_case(_case(), base, [base.adapter], tmp_path)
            with pytest.raises(RuntimeError, match='running event loop'):
                runner.run_suite([_case()], 'Fake|m', tmp_path,
                                 base_state=base, adapters=[base.adapter],
                                 trajectory_dir=tmp_path)

        asyncio.run(inside())
        assert not list(tmp_path.iterdir())


ROUTING_TOML = '''
[routing]
mode = "local-and-remote"
controller = true
complexity_router = true
spend_confirm = "auto"
secret_scan = true

[[routing.ladder]]
adapter = "Fake"
model = "small"
max_complexity = "standard"
default = true

[[routing.ladder]]
adapter = "Remote"
model = "aws/claude-5-sonnet"
max_complexity = "hard"
'''


class TestRoutingFile:
    """``runner.load_routing_file``: a ``[routing]`` table as in
    settings.toml."""

    def test_parses_table(self, tmp_path: Path) -> None:
        p = tmp_path / 'exp.toml'
        p.write_text(ROUTING_TOML)
        rs = runner.load_routing_file(p)
        assert rs.present is True and rs.controller is True
        assert rs.spend_confirm == 'auto'
        assert [r.model for r in rs.ladders['default']] == [
            'small', 'aws/claude-5-sonnet']

    @pytest.mark.parametrize('text, match', [
        ('mode = "x"\n', r'no \[routing\] table'),
        ('[routing]\nmode = "bogus"\n', 'mode'),
        ('[routing]\nnope = 1\n', 'unknown keys'),
        ('[routing\n', 'invalid TOML'),
    ])
    def test_invalid_raises_value_error(self, tmp_path: Path, text: str,
                                        match: str) -> None:
        p = tmp_path / 'bad.toml'
        p.write_text(text)
        with pytest.raises(ValueError, match=match):
            runner.load_routing_file(p)

    def test_missing_file_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='no-such'):
            runner.load_routing_file(tmp_path / 'no-such.toml')


DECISIONS_TOML = """
[decisions]
mode = "shadow"

[decisions.points]
panel = "encoder"
injection = "injection"
"""


class TestDecisionsFile:
    """``runner.load_decisions_file``: the optional ``[decisions]`` table
    of an experiment file."""

    def test_parses_table(self, tmp_path: Path) -> None:
        p = tmp_path / 'exp.toml'
        p.write_text(ROUTING_TOML + DECISIONS_TOML)
        ds = runner.load_decisions_file(p)
        assert ds is not None
        assert ds.mode == 'shadow'
        assert ds.points == {'panel': 'encoder', 'injection': 'injection'}
        assert ds.active == {} and ds.thresholds == {}
        # the same file still parses as a routing file
        assert runner.load_routing_file(p).controller is True

    def test_no_table_is_none(self, tmp_path: Path) -> None:
        p = tmp_path / 'exp.toml'
        p.write_text(ROUTING_TOML)
        assert runner.load_decisions_file(p) is None

    @pytest.mark.parametrize('text, match', [
        ('[decisions]\nmode = "bogus"\n', 'mode'),
        ('[decisions]\nmode = "shadow"\nnope = 1\n', 'unknown keys'),
        ('[decisions\n', 'invalid TOML'),
    ])
    def test_invalid_raises_value_error(self, tmp_path: Path, text: str,
                                        match: str) -> None:
        p = tmp_path / 'bad.toml'
        p.write_text(text)
        with pytest.raises(ValueError, match=match):
            runner.load_decisions_file(p)

    def test_missing_file_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='no-such'):
            runner.load_decisions_file(tmp_path / 'no-such.toml')


@pytest.fixture
def routed(monkeypatch):
    """Like ``canned`` but records the routing wiring, the spend asker, the
    scanner state and lets a test write ledger ``tasks`` rows."""
    from guru.domain import policy, spend
    seen: dict = {}

    async def fake_run(self, prompt, timeout=None):
        seen['registry'] = self.registry
        seen['routing'] = self._routing_settings
        seen['controller'] = self.controller
        seen['spend_asker'] = spend._asker
        seen['spend_granted'] = spend._asker(spend.QUESTION) \
            if spend._asker is not None else None
        seen['secret_scan'] = config.SECRET_SCAN
        seen['scanner'] = policy.scanner()
        for row in seen.get('tasks', []):
            ledger.repository().append('tasks', row)
        return _canned_agents()

    monkeypatch.setattr(bench.BenchRun, 'run', fake_run)
    return seen


def _routing(**kw) -> RoutingSettings:
    return RoutingSettings(present=True, **kw)


class TestRunCaseRouting:
    def test_inert_without_routing(self, tmp_path: Path, routed) -> None:
        from guru.domain import spend
        spend.set_spend_asker(None)
        base = _base()
        runner.run_case(_case(), base, [base.adapter], tmp_path)
        assert routed['registry'] is None
        assert routed['routing'] is None
        assert routed['controller'] is False
        assert routed['spend_asker'] is runner._deny
        assert routed['spend_granted'] is False
        assert spend._asker is None                  # restored

    def test_registry_routing_and_controller(self, tmp_path: Path,
                                             routed) -> None:
        from guru.repositories.adapters import AdapterRegistry
        base = _base()
        settings = _routing(controller=True)
        routing = runner.Routing(settings, AdapterRegistry([base.adapter]),
                                 name='exp')
        res = runner.run_case(_case(), base, [base.adapter], tmp_path,
                              routing=routing)
        assert routed['registry'] is routing.registry
        assert routed['registry'].get('Fake') is base.adapter
        assert routed['routing'] is settings
        assert routed['controller'] is True
        assert res.routes == []

    def test_allow_spend_grants_and_restores(self, tmp_path: Path,
                                             routed) -> None:
        from guru.domain import spend
        sentinel = object()
        spend.set_spend_asker(sentinel)      # type: ignore[arg-type]
        try:
            base = _base()
            runner.run_case(_case(), base, [base.adapter], tmp_path,
                            allow_spend=True)
            assert routed['spend_granted'] is True
            assert spend._asker is sentinel
        finally:
            spend.set_spend_asker(None)

    def test_routes_are_distinct_task_adapter_models(self, tmp_path: Path,
                                                     routed) -> None:
        routed['tasks'] = [
            {'task_id': 't1', 'adapter': 'Fake', 'model': 'small',
             'status': 'running'},
            {'task_id': 't1', 'adapter': 'Fake', 'model': 'small',
             'status': 'done'},
            {'task_id': 't2', 'adapter': 'Remote',
             'model': 'aws/claude-5-sonnet', 'status': 'running'},
            {'task_id': 't3', 'adapter': '', 'model': '',
             'status': 'refused'},
            {'task_id': 't4', 'adapter': 'Fake', 'model': 'small',
             'status': 'running'},
        ]
        base = _base()
        res = runner.run_case(_case(), base, [base.adapter], tmp_path)
        assert res.routes == ['Fake|small', 'Remote|aws/claude-5-sonnet']
        # only this case's rows count
        routed['tasks'] = []
        res2 = runner.run_case(_case(name='second'), base, [base.adapter],
                               tmp_path)
        assert res2.routes == []


class TestRunSuiteRouting:
    def test_records_routing_and_builds_registry(self, tmp_path: Path,
                                                 routed) -> None:
        from guru.repositories.adapters import AdapterRegistry
        base = _base()
        settings = _routing(controller=True, secret_scan=False)
        run = runner.run_suite([_case(name='a')], 'Fake|base-model',
                               tmp_path, base_state=base,
                               adapters=[base.adapter],
                               trajectory_dir=tmp_path, routing=settings,
                               routing_name='exp-b')
        assert run.routing == 'exp-b' and run.controller is True
        assert isinstance(routed['registry'], AdapterRegistry)
        assert routed['registry'].get('Fake') is base.adapter
        assert routed['routing'] is settings
        assert run.model_label() == 'Fake|base-model+routed:exp-b+controller'
        assert runs.load(next(tmp_path.glob('*.json'))).routing == 'exp-b'
        traj = (tmp_path / runs.TRAJECTORY_FILE).read_text()
        assert '+routed:exp-b+controller' in traj

    def test_without_routing_records_defaults(self, tmp_path: Path,
                                              routed) -> None:
        base = _base()
        run = runner.run_suite([_case(name='a')], 'Fake|base-model',
                               tmp_path, base_state=base,
                               adapters=[base.adapter],
                               trajectory_dir=tmp_path)
        assert run.routing == '' and run.controller is False
        assert routed['registry'] is None

    @pytest.mark.parametrize('scan', [True, False])
    def test_secret_scan_follows_routing_and_is_restored(
            self, tmp_path: Path, routed, monkeypatch, scan: bool) -> None:
        from guru.domain import policy
        monkeypatch.setattr(config, 'SECRET_SCAN', not scan)
        before = object()
        policy.set_scanner(before)          # type: ignore[arg-type]
        try:
            base = _base()
            runner.run_suite([_case(name='a')], 'Fake|base-model', tmp_path,
                             base_state=base, adapters=[base.adapter],
                             trajectory_dir=tmp_path,
                             routing=_routing(secret_scan=scan))
            assert routed['secret_scan'] is scan
            assert (routed['scanner'] is not None) is scan
            assert routed['scanner'] is not before
            assert config.SECRET_SCAN is (not scan)
            assert policy.scanner() is before
        finally:
            policy.set_scanner(None)

    def test_without_routing_scanner_untouched(self, tmp_path: Path,
                                               routed, monkeypatch) -> None:
        from guru.domain import policy
        monkeypatch.setattr(config, 'SECRET_SCAN', False)
        policy.set_scanner(None)
        base = _base()
        runner.run_suite([_case(name='a')], 'Fake|base-model', tmp_path,
                         base_state=base, adapters=[base.adapter],
                         trajectory_dir=tmp_path)
        assert routed['secret_scan'] is False and routed['scanner'] is None


@pytest.fixture
def judged(monkeypatch):
    """Like ``canned`` but records the decision seam as the run saw it."""
    from guru.domain import decisions as seam
    seen: dict = {}

    async def fake_run(self, prompt, timeout=None):
        seen['mode'] = config.DECISIONS_MODE
        seen['points'] = dict(config.DECISIONS_POINTS)
        seen['active'] = dict(config.DECISIONS_ACTIVE)
        seen['thresholds'] = dict(config.DECISIONS_THRESHOLDS)
        seen['judges'] = dict(seam._judges)
        return _canned_agents()

    monkeypatch.setattr(bench.BenchRun, 'run', fake_run)
    return seen


def _decisions(**kw):
    from guru.repositories.settings import DecisionsSettings
    return DecisionsSettings(**kw)


class TestRunSuiteJudges:
    """``run_suite(decisions=...)`` installs the experiment's judges for the
    run and restores the process afterwards."""

    @pytest.fixture(autouse=True)
    def _off(self, monkeypatch):
        from guru.domain import decisions as seam
        monkeypatch.setattr(config, 'DECISIONS_MODE', 'off')
        monkeypatch.setattr(config, 'DECISIONS_POINTS', {'stall': 'ollama'})
        monkeypatch.setattr(config, 'DECISIONS_ACTIVE', {})
        monkeypatch.setattr(config, 'DECISIONS_THRESHOLDS', {})
        seam.clear_judges()
        yield
        seam.clear_judges()

    def _run(self, tmp_path, decisions):
        base = _base()
        return runner.run_suite([_case(name='a')], 'Fake|base-model',
                                tmp_path, base_state=base,
                                adapters=[base.adapter],
                                trajectory_dir=tmp_path,
                                decisions=decisions)

    def test_judges_installed_for_the_run_and_cleared(
            self, tmp_path: Path, judged, monkeypatch) -> None:
        from guru.domain import decisions as seam
        from guru.judges import encoder
        monkeypatch.setattr(encoder, 'available', lambda: True)
        ds = _decisions(mode='shadow',
                        points={'panel': 'encoder',
                                'injection': 'injection'},
                        thresholds={'panel': 0.6})
        run = self._run(tmp_path, ds)
        assert judged['mode'] == 'shadow'
        assert judged['points'] == ds.points
        assert judged['active'] == {}
        assert judged['thresholds'] == {'panel': 0.6}
        assert set(judged['judges']) == {'panel', 'injection'}
        assert judged['judges']['panel'].name.startswith('encoder:')
        assert run.judges == [
            'panel=' + judged['judges']['panel'].name,
            'injection=' + judged['judges']['injection'].name]
        # restored
        assert config.DECISIONS_MODE == 'off'
        assert config.DECISIONS_POINTS == {'stall': 'ollama'}
        assert config.DECISIONS_THRESHOLDS == {}
        assert seam._judges == {}
        # persisted and reloadable
        assert runs.load(next(tmp_path.glob('*.json'))).judges == run.judges

    def test_unavailable_judges_leave_the_list_empty(
            self, tmp_path: Path, judged, monkeypatch) -> None:
        from guru.judges import encoder
        monkeypatch.setattr(encoder, 'available', lambda: False)
        run = self._run(tmp_path, _decisions(
            mode='shadow', points={'panel': 'encoder'}))
        assert judged['mode'] == 'shadow' and judged['judges'] == {}
        assert run.judges == []

    def test_restored_when_a_case_raises(self, tmp_path: Path,
                                         monkeypatch) -> None:
        from guru.domain import decisions as seam
        from guru.judges import encoder
        monkeypatch.setattr(encoder, 'available', lambda: True)

        def boom(*a, **k):
            raise RuntimeError('boom')
        monkeypatch.setattr(runner, 'run_case', boom)
        with pytest.raises(RuntimeError):
            self._run(tmp_path, _decisions(mode='shadow',
                                           points={'panel': 'encoder'}))
        assert config.DECISIONS_MODE == 'off' and seam._judges == {}

    def test_without_decisions_seam_untouched(self, tmp_path: Path,
                                              judged, monkeypatch) -> None:
        from guru.domain import decisions as seam
        marker = object()
        seam.set_judge('stall', marker)      # type: ignore[arg-type]
        run = self._run(tmp_path, None)
        assert judged['mode'] == 'off'
        assert judged['judges'] == {'stall': marker}
        assert seam._judges == {'stall': marker}     # not cleared
        assert run.judges == []


class TestCliRouting:
    def _cases_dir(self, tmp_path: Path) -> Path:
        cdir = tmp_path / 'cases'
        cdir.mkdir()
        (cdir / 'a.toml').write_text(
            'name = "a"\nfixture = "docs-only"\nprompt = "hi"\n')
        return cdir

    def test_routing_file_is_parsed_and_passed(self, tmp_path: Path, capsys,
                                               monkeypatch) -> None:
        from guru.repositories import settings as rs
        cdir = self._cases_dir(tmp_path)
        rfile = tmp_path / 'exp-b.toml'
        rfile.write_text(ROUTING_TOML)
        seen: dict = {}

        def fake_suite(suite, model_spec, out_root, **kw):
            seen.update(kw)
            r = runs.Run(run_id='rid', ts=runs.now_ts(), model='Fake|m',
                         git_sha='', routing=kw['routing_name'],
                         controller=kw['routing'].controller,
                         cases=[runs.CaseResult(
                             case='a', passed=True, checks=[],
                             observed={'seconds': 1.0}, rubric='',
                             transcript_path='t', cost_usd=0.0,
                             routes=['Fake|small', 'Remote|big'])])
            runs.save(r, out_root)
            return r

        monkeypatch.setattr(runner, 'run_suite', fake_suite)
        code = cli_main(['run', '--cases-dir', str(cdir),
                         '--out', str(tmp_path / 'r'), '--model', 'Fake|m',
                         '--routing', str(rfile)])
        assert code == 0
        assert isinstance(seen['routing'], rs.RoutingSettings)
        assert seen['routing'].controller is True
        assert seen['routing_name'] == 'exp-b'
        assert seen['allow_spend'] is False
        out = capsys.readouterr().out
        assert 'routes: Fake|small, Remote|big' in out
        assert 'model Fake|m+routed:exp-b+controller' in out
        assert 'routing exp-b (controller)' in out

    def test_decisions_table_is_parsed_and_judges_printed(
            self, tmp_path: Path, capsys, monkeypatch) -> None:
        from guru.repositories import settings as rs
        cdir = self._cases_dir(tmp_path)
        rfile = tmp_path / 'exp-j.toml'
        rfile.write_text(ROUTING_TOML + DECISIONS_TOML)
        seen: dict = {}

        def fake_suite(suite, model_spec, out_root, **kw):
            seen.update(kw)
            r = runs.Run(run_id='rid', ts=runs.now_ts(), model='Fake|m',
                         git_sha='', routing=kw['routing_name'],
                         controller=True, cases=[],
                         judges=['panel=encoder:x', 'injection=injection:y'])
            runs.save(r, out_root)
            return r

        monkeypatch.setattr(runner, 'run_suite', fake_suite)
        code = cli_main(['run', '--cases-dir', str(cdir),
                         '--out', str(tmp_path / 'r'), '--model', 'Fake|m',
                         '--routing', str(rfile)])
        assert code == 0
        assert isinstance(seen['decisions'], rs.DecisionsSettings)
        assert seen['decisions'].mode == 'shadow'
        assert seen['decisions'].points == {'panel': 'encoder',
                                            'injection': 'injection'}
        out = capsys.readouterr().out
        assert 'judges panel=encoder:x, injection=injection:y' in out

    def test_bad_decisions_table_is_usage_error(self, tmp_path: Path,
                                                capsys) -> None:
        cdir = self._cases_dir(tmp_path)
        bad = tmp_path / 'bad.toml'
        bad.write_text(ROUTING_TOML + '[decisions]\nmode = "bogus"\n')
        assert cli_main(['run', '--cases-dir', str(cdir),
                         '--out', str(tmp_path),
                         '--routing', str(bad)]) == 2
        assert 'mode' in capsys.readouterr().err

    def test_allow_spend_flag(self, tmp_path: Path, monkeypatch) -> None:
        cdir = self._cases_dir(tmp_path)
        seen: dict = {}

        def fake_suite(suite, model_spec, out_root, **kw):
            seen.update(kw)
            return runs.Run(run_id='rid', ts=runs.now_ts(), model='m',
                            git_sha='', cases=[])

        monkeypatch.setattr(runner, 'run_suite', fake_suite)
        assert cli_main(['run', '--cases-dir', str(cdir),
                         '--out', str(tmp_path), '--allow-spend']) == 0
        assert seen['allow_spend'] is True
        assert seen['routing'] is None and seen['routing_name'] == ''
        assert seen['decisions'] is None

    def test_missing_or_invalid_routing_file_is_usage_error(
            self, tmp_path: Path, capsys) -> None:
        cdir = self._cases_dir(tmp_path)
        assert cli_main(['run', '--cases-dir', str(cdir),
                         '--out', str(tmp_path),
                         '--routing', str(tmp_path / 'nope.toml')]) == 2
        assert 'nope.toml' in capsys.readouterr().err
        bad = tmp_path / 'bad.toml'
        bad.write_text('[routing]\nmode = "bogus"\n')
        assert cli_main(['run', '--cases-dir', str(cdir),
                         '--out', str(tmp_path),
                         '--routing', str(bad)]) == 2
        assert 'mode' in capsys.readouterr().err

    def test_routed_run_without_spawns_shows_dash(self) -> None:
        from guru.evals import __main__ as cli
        res = runs.CaseResult(case='x', passed=True, checks=[],
                              observed={'seconds': 1.0}, rubric='',
                              transcript_path='t', cost_usd=None)
        assert cli._row(res, routed=True)[-1] == 'routes: -'
        assert cli._row(res, routed=False)[-1] == ''
