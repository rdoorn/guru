"""Structural round, Package C item 5: ``bench/tool_contract.py`` -- the
task list covers the registry, the runner records call success, schema
errors, retries, tokens and seconds, and the report is written per slug."""
import json
import sys
from pathlib import Path

import pytest

from guru import config, session
from guru.domain import tools

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import tool_contract as tc                        # noqa: E402


class ScriptedDriver(tc.Driver):
    """Replays scripted steps; records what came back."""
    forcing = True

    def __init__(self, steps) -> None:
        super().__init__(adapter=None, model='fake')
        self.steps = list(steps)
        self.history: list = []

    def start(self, system, user) -> None:
        self.history = [('system', system), ('user', user)]

    def step(self, spec, force) -> tc.Step:
        self.history.append(('step', spec['name'], force))
        return self.steps.pop(0)

    def tool_result(self, call_id, name, content) -> None:
        self.history.append(('tool_result', name, content))

    def user(self, text) -> None:
        self.history.append(('user', text))


@pytest.fixture
def copy(tmp_path, monkeypatch) -> Path:
    from guru import ui
    monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
    monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
    monkeypatch.setattr(session, 'controller', False)
    monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', set())
    monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
    monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
    return tc._fresh_copy(tmp_path)


class TestTasks:
    def test_one_task_per_non_sandbox_tool(self) -> None:
        covered = {t.tool for t in tc.TASKS}
        expected = (set(tools.TOOL_REGISTRY) - set(tools.SANDBOX_TOOLS)
                    | tools.ALWAYS_ON_TOOLS)
        assert covered == expected
        assert len(covered) == len(tc.TASKS)          # no duplicates
        web = {t.tool for t in tc.TASKS if t.web}
        assert web == {'web_search', 'web_fetch', 'fetch_github_releases'}

    def test_setups_fill_their_prompts(self, copy) -> None:
        by = {t.tool: t for t in tc.TASKS}
        fills = by['edit_file'].setup(copy)
        assert len(fills['sha']) == 12
        assert '{sha}' not in by['edit_file'].prompt.format(**fills)
        diff = by['apply_patch'].setup(copy)['diff']
        assert diff.startswith('--- a/wordcount.py') and '+' in diff
        by['delete_file'].setup(copy)
        assert (copy / 'scratch.txt').is_file()

    def test_slug_and_resolve(self) -> None:
        assert tc.slug('SBP Litellm|aws/claude-4-5-haiku') == \
            'sbp-litellm-aws-claude-4-5-haiku'
        assert tc.slug('Ollama|qwen3:4b') == 'ollama-qwen3-4b'

        class A:
            name = 'SBP Litellm'
        adapter, model = tc.resolve('sbp litellm|m1', [A()])
        assert isinstance(adapter, A) and model == 'm1'
        with pytest.raises(SystemExit):
            tc.resolve('Nope|m', [A()])


class TestRunTask:
    def _task(self, tool: str) -> tc.Task:
        return next(t for t in tc.TASKS if t.tool == tool)

    def test_clean_call_is_ok(self, copy) -> None:
        driver = ScriptedDriver([tc.Step('', [
            ('read_file', {'path': 'wordcount.py', 'lines': '1-12'}, 'c1')],
            100, 20, 0.001)])
        out = tc.run_task(self._task('read_file'), driver, copy)
        assert out.called and out.right_tool and out.ok
        assert (out.schema_errors, out.retries, out.attempts) == (0, 0, 1)
        assert (out.tokens_in, out.tokens_out, out.cost_usd) == (100, 20,
                                                                 0.001)
        assert out.seconds >= 0 and 'lines 1-12' in out.result_head
        assert driver.history[2] == ('step', 'read_file', True)
        assert Path.cwd() == copy

    def test_schema_error_then_fix_counts_a_retry(self, copy) -> None:
        driver = ScriptedDriver([
            tc.Step('', [('read_file', {'file': 'wordcount.py'}, 'c1')],
                    50, 5),
            tc.Step('', [('read_file', {'path': 'wordcount.py',
                                        'lines': '1-12'}, 'c2')], 60, 6)])
        out = tc.run_task(self._task('read_file'), driver, copy)
        assert out.ok and out.schema_errors == 1 and out.retries == 1
        assert out.attempts == 2 and out.tokens_in == 110
        kind, name, content = driver.history[3]
        assert kind == 'tool_result' and name == 'read_file'
        assert content.startswith(tools.INVALID_ARGS_PREFIX)

    def test_no_call_is_nudged_then_reported(self, copy) -> None:
        driver = ScriptedDriver([tc.Step('I would read it', [], 10, 1)] * 3)
        out = tc.run_task(self._task('read_file'), driver, copy)
        assert not out.called and not out.ok
        assert out.retries == 2 and out.attempts == 3
        assert out.note == 'no tool call'
        assert sum(1 for h in driver.history if h[0] == 'user') == 3

    def test_wrong_tool_is_noted(self, copy) -> None:
        driver = ScriptedDriver([
            tc.Step('', [('list_dir', {}, 'c1')], 10, 1),
            tc.Step('', [('list_dir', {}, 'c2')], 10, 1),
            tc.Step('', [('list_dir', {}, 'c3')], 10, 1)])
        out = tc.run_task(self._task('read_file'), driver, copy)
        assert out.called and not out.right_tool and not out.ok
        assert out.note == 'called list_dir' and out.retries == 3

    def test_persistent_schema_error_is_reported(self, copy) -> None:
        bad = tc.Step('', [('read_file', {'file': 'x'}, 'c')], 10, 1)
        out = tc.run_task(self._task('read_file'), ScriptedDriver([bad] * 3),
                          copy)
        assert out.schema_errors == 3 and out.retries == 2 and not out.ok
        assert out.result_head.startswith(tools.INVALID_ARGS_PREFIX)

    def test_provider_error_is_the_note(self, copy) -> None:
        class Boom(ScriptedDriver):
            def step(self, spec, force):
                raise RuntimeError('proxy down')
        out = tc.run_task(self._task('read_file'), Boom([]), copy)
        assert not out.called and out.note == 'RuntimeError: proxy down'

    def test_check_decides_ok(self, copy) -> None:
        # The call runs, but the file does not hold the fix -> not ok.
        driver = ScriptedDriver([tc.Step('', [
            ('edit_file', {'path': 'wordcount.py', 'old': 'sys',
                           'new': 'sys', 'sha': 'stale'}, 'c1')], 1, 1)])
        out = tc.run_task(self._task('edit_file'), driver, copy)
        assert out.called and out.right_tool and not out.ok

    def test_spawn_runs_against_fake_handlers(self, copy) -> None:
        tc._install_fake_handlers()
        try:
            driver = ScriptedDriver([tc.Step('', [
                ('spawn', {'task': 'review', 'kind': 'review',
                           'complexity': 'standard'}, 'c1')], 1, 1)])
            out = tc.run_task(self._task('spawn'), driver, copy)
            assert out.ok and out.result_head.startswith('spawned agent2')
        finally:
            tools.set_spawn_handler(None)
            tools.set_check_handler(None)
            tools.set_join_handler(None)


class TestReport:
    def test_render_and_summary(self) -> None:
        ran = tc.ToolResult('read_file', called=True, right_tool=True, ok=True,
                            tokens_in=100, tokens_out=10, seconds=1.5,
                            cost_usd=0.01)
        bad = tc.ToolResult('lint', called=True, right_tool=True,
                            schema_errors=2, retries=2, tokens_in=50,
                            tokens_out=5, seconds=2.0, note='x')
        skipped = tc.ToolResult('sandbox_run', 'skipped', tc.SKIP_SANDBOX)
        report = {'model': 'A|m', 'forcing': True, 'summary': {
            'tools': 2, 'ok': 1, 'called': 2, 'schema_errors': 2,
            'retries': 2, 'tokens_in': 150, 'tokens_out': 15,
            'seconds': 3.5, 'cost_usd': 0.01},
            'tools': {r.tool: r.__dict__ for r in (ran, bad, skipped)}}
        text = tc.render(report)
        assert text.splitlines()[0] == 'tool contract: A|m (forcing: yes)'
        assert 'read_file               yes yes      0     0     100' in text
        assert 'skipped: needs a provisioned sandbox image' in text
        assert text.splitlines()[-1] == (
            'summary: 1/2 ok, 2 called, 2 schema errors, 2 retries, 150+15'
            ' tokens, 3.5s $0.0100')

    def test_main_writes_the_slug_file(self, tmp_path, monkeypatch) -> None:
        fake_report = {'model': 'A|m', 'adapter': 'A', 'model_id': 'm',
                       'forcing': False, 'date': 'now', 'summary': {
                           'tools': 0, 'ok': 0, 'called': 0,
                           'schema_errors': 0, 'retries': 0, 'tokens_in': 0,
                           'tokens_out': 0, 'seconds': 0.0,
                           'cost_usd': None}, 'tools': {}}
        monkeypatch.setattr(tc, 'run_all', lambda *a, **k: fake_report)
        monkeypatch.setattr(tc, '_quiet', lambda: None)
        import guru.bench
        monkeypatch.setattr(guru.bench, 'build_adapters', lambda: [])
        assert tc.main(['--model', 'A|m', '--out', str(tmp_path)]) == 0
        written = json.loads((tmp_path / 'a-m.json').read_text())
        assert written == fake_report

    def test_committed_results_have_the_shape(self) -> None:
        results = Path(__file__).resolve().parents[1] / 'evals' / 'models'
        for path in results.glob('*.json'):
            data = json.loads(path.read_text())
            assert {'model', 'forcing', 'summary', 'tools'} <= set(data)
            assert path.stem == tc.slug(data['model'])
            for name, row in data['tools'].items():
                assert row['tool'] == name
                assert row['status'] in ('run', 'skipped')
