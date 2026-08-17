"""Tests for guru.bench and guru.bench_plot."""
import json
from types import SimpleNamespace

from guru import bench, bench_plot, config, session
from guru.domain import files, tools


class TestBenchModels:
    def test_load_models_parses_adapter_and_filters(
            self, tmp_path) -> None:
        p = tmp_path / 'models.txt'
        p.write_text(
            "# a comment\nqwen3:14b\n\n  devstral:24b \n"
            "litellm|aws/claude-4-8-opus\n", encoding='utf-8')
        assert bench.load_models(p) == [
            (None, 'qwen3:14b'),
            (None, 'devstral:24b'),
            ('litellm', 'aws/claude-4-8-opus'),
        ]

    def test_human_size_units_are_explicit(self) -> None:
        assert files._human(0) == '0B'
        assert files._human(112) == '112B'
        assert files._human(1536) == '1.5KB'
        assert files._human(80 * 1024) == '80.0KB'
        assert files._human(5 * 1024 * 1024) == '5.0MB'

    def test_sort_models_groups_by_basename(self) -> None:
        models = [
            (None, 'gpt-oss-20b-32k:latest'),
            (None, 'qwen3:14b'),
            ('litellm', 'aws/claude-4-8-opus'),
            (None, 'hf.co/unsloth/Qwen3-14B-128K-GGUF:Q4_K_M'),
            (None, 'batiai/qwen3.6-27b:q3'),
        ]
        ordered = [m for _, m in bench.sort_models(models)]
        # all qwen* adjacent; sorted by the name after the last '/'
        assert ordered == [
            'aws/claude-4-8-opus',                      # claude-...
            'gpt-oss-20b-32k:latest',                   # gpt-oss-...
            'hf.co/unsloth/Qwen3-14B-128K-GGUF:Q4_K_M',  # qwen3-14b-128k
            'batiai/qwen3.6-27b:q3',                    # qwen3.6-...
            'qwen3:14b',                                # qwen3:14b
        ]


class TestBenchMetrics:
    def _agent(self, title, model, tool_names, tin, tout,
               role=None, skill=None):
        from guru.agents import Agent
        a = Agent(id=title, title=title)
        a.state.model = model
        a.state.session_in = tin
        a.state.session_out = tout
        a.state.active_role = role
        a.state.active_skill = skill
        a.state.messages = [{'role': 'user', 'content': 'q'}]
        for n in tool_names:
            a.state.messages.append(
                {'role': 'tool', 'tool_name': n, 'content': 'x'})
        a.state.messages.append(
            {'role': 'assistant', 'content': 'ANSWER'})
        return a

    def test_collect_metrics_aggregates(self) -> None:
        main = self._agent('main', 'qwen3:14b',
                           ['search_tools', 'read_file'], 100, 40)
        sub = self._agent('agent1', 'qwen3:14b', ['read_file'], 20, 10,
                          role='security-engineer', skill='code-review')
        rec = bench.collect_metrics(
            'qwen3:14b', num_ctx=40960, ceiling=40960, seconds=4.0,
            agents=[main, sub])
        assert rec['tokens_in'] == 120 and rec['tokens_out'] == 50
        assert rec['tokens_per_sec'] == 12.5      # 50/4.0
        assert rec['tool_count'] == 3
        assert sorted(rec['tools_called']) == [
            'read_file', 'read_file', 'search_tools']
        assert rec['agents_used'] == 2
        assert rec['agents'][1]['role'] == 'security-engineer'
        assert rec['result'] == 'ANSWER'
        assert rec['accuracy'] is None and rec['error'] is None

    def test_collect_metrics_empty_agents(self) -> None:
        rec = bench.collect_metrics('m', 4096, 4096, 0.0, [],
                                    error='boom')
        assert rec['agents_used'] == 0 and rec['result'] == ''
        assert rec['error'] == 'boom' and rec['tokens_per_sec'] == 0.0

    def test_blank_answer_is_flagged_empty(self) -> None:
        a = self._agent('main', 'm', ['read_file'], 10, 5)
        a.state.messages[-1]['content'] = '   '     # blank final answer
        rec = bench.collect_metrics('m', 4096, 4096, 1.0, [a])
        assert rec['error'] == 'empty answer' and rec['tool_count'] == 1

    def test_caller_error_wins_over_empty_flag(self) -> None:
        a = self._agent('main', 'm', [], 10, 5)
        a.state.messages[-1]['content'] = ''        # also blank
        rec = bench.collect_metrics('m', 4096, 4096, 1.0, [a],
                                    error='timeout after 600s')
        assert rec['error'] == 'timeout after 600s'

    def test_serialize_transcript(self) -> None:
        from guru.agents import Agent
        a = Agent(id='main', title='main')
        a.state.model = 'm'
        a.state.messages = [
            {'role': 'user', 'content': 'q'},
            SimpleNamespace(role='assistant', content='thinking', tool_calls=[
                SimpleNamespace(function=SimpleNamespace(name='read_file'))]),
            {'role': 'tool', 'tool_name': 'read_file', 'content': 'data'},
            {'role': 'assistant', 'content': 'ANSWER'},
        ]
        out = bench.serialize_transcript([a])
        assert out[0]['title'] == 'main' and out[0]['model'] == 'm'
        msgs = out[0]['messages']
        assert msgs[0] == {'role': 'user', 'content': 'q'}
        assert msgs[1]['tool_calls'] == ['read_file']
        assert msgs[2]['tool_name'] == 'read_file'
        assert msgs[3]['content'] == 'ANSWER'


class TestBenchOrchestrator:
    def test_spawn_runs_child_and_counts_agents(self, monkeypatch) -> None:
        import asyncio
        from guru.adapters.base import Adapter

        class FakeAdapter(Adapter):
            name = 'fake'
            def available(self): return True
            def list_models(self): return []
            def activate(self, m): pass
            def summarise(self, t): return 's'

            def run_turn(self):
                st = session.current()
                spawned = any(
                    isinstance(m, dict) and m.get('tool_name') == 'spawn'
                    for m in st.messages)
                if st.can_spawn and not spawned:
                    tools.spawn('sub task', role='', skill='')
                    st.messages.append(
                        {'role': 'tool', 'tool_name': 'spawn',
                         'content': 'ok'})
                st.session_out += 5
                st.messages.append(
                    {'role': 'assistant', 'content': 'done'})

        base = session.SessionState()
        base.adapter = FakeAdapter()
        base.model = 'fake'
        base.num_ctx = 4096
        try:
            agents = asyncio.run(bench.run_once(base))
            titles = [a.title for a in agents]
            assert 'main' in titles and len(agents) >= 2
        finally:
            tools.set_spawn_handler(None)
            tools.set_check_handler(None)
            tools.set_join_handler(None)

    def test_abort_flags_cancel_on_stalled_agents(self) -> None:
        import asyncio
        from guru.agents import Agent

        b = bench._Bench(session.SessionState())
        stuck = Agent(id='main', title='main')
        stuck.busy = True                 # never winds down on its own
        b.manager.agents = [stuck]

        async def drive() -> None:
            b.loop = asyncio.get_running_loop()
            await b._abort(grace=0.2)     # gives up after the grace window
        asyncio.run(drive())
        assert stuck.state.cancel_requested is True


class TestBenchRunner:
    def test_writes_results_json(self, tmp_path, monkeypatch) -> None:
        from guru.agents import Agent

        class FakeAdapter:
            name = 'fake'

            def activate(self, m):
                session.current().model = m
                session.current().num_ctx = 4096
                session.current().ctx_ceiling = 4096

        monkeypatch.setattr(bench, '_build_adapters',
                            lambda: [FakeAdapter()])
        monkeypatch.setattr(bench, '_adapter_for',
                            lambda name, built: built[0])

        def fake_run_once(base):
            a = Agent(id='main', title='main')
            a.state.model = base.model
            a.state.session_out = 10
            a.state.messages = [{'role': 'assistant', 'content': 'A'}]
            return [a]
        monkeypatch.setattr(bench, 'run_once', fake_run_once)
        # make asyncio.run a passthrough (run_once is a plain function here)
        monkeypatch.setattr(bench.asyncio, 'run', lambda coro: coro)

        path = bench.run_benchmark([(None, 'm1')], out_dir=tmp_path)
        data = json.loads(path.read_text(encoding='utf-8'))
        assert len(data) == 1
        assert data[0]['model'] == 'm1' and data[0]['result'] == 'A'
        assert data[0]['tokens_out'] == 10
        # a per-run transcript file is written alongside the results
        tfile = next(tmp_path.glob('transcript-*.json'))
        tdata = json.loads(tfile.read_text(encoding='utf-8'))
        assert tdata[0]['model'] == 'm1'
        assert tdata[0]['agents'][0]['messages'][0]['content'] == 'A'

    def test_records_timeout_when_over_limit(
            self, tmp_path, monkeypatch) -> None:
        from guru.agents import Agent

        class FakeAdapter:
            name = 'fake'

            def activate(self, m):
                session.current().model = m
                session.current().num_ctx = 4096
                session.current().ctx_ceiling = 4096

        monkeypatch.setattr(bench, '_build_adapters',
                            lambda: [FakeAdapter()])
        monkeypatch.setattr(bench, '_adapter_for',
                            lambda name, built: built[0])
        monkeypatch.setattr(config, 'BENCH_MODEL_TIMEOUT', 600)

        def slow_run_once(base):
            a = Agent(id='main', title='main')
            a.state.model = base.model
            a.state.messages = [{'role': 'assistant', 'content': 'partial'}]
            return [a]
        monkeypatch.setattr(bench, 'run_once', slow_run_once)
        monkeypatch.setattr(bench.asyncio, 'run', lambda coro: coro)
        # Simulate a run whose wall-clock overran the limit (t0=0, end=700).
        times = iter([0.0, 700.0])
        monkeypatch.setattr(bench.time, 'monotonic', lambda: next(times))

        path = bench.run_benchmark([(None, 'm1')], out_dir=tmp_path)
        data = json.loads(path.read_text(encoding='utf-8'))
        assert 'timeout' in (data[0]['error'] or '')
        # partial metrics are still recorded, not discarded
        assert data[0]['result'] == 'partial'

    def test_partial_results_saved_on_interrupt(
            self, tmp_path, monkeypatch) -> None:
        from guru.agents import Agent

        class FakeAdapter:
            name = 'fake'

            def activate(self, m):
                session.current().model = m
                session.current().num_ctx = 4096
                session.current().ctx_ceiling = 4096

        monkeypatch.setattr(bench, '_build_adapters',
                            lambda: [FakeAdapter()])
        monkeypatch.setattr(bench, '_adapter_for',
                            lambda name, built: built[0])
        calls = {'n': 0}

        def flaky_run_once(base):
            calls['n'] += 1
            if calls['n'] == 2:            # cancel during the 2nd model
                raise KeyboardInterrupt
            a = Agent(id='main', title='main')
            a.state.model = base.model
            a.state.messages = [{'role': 'assistant', 'content': 'A'}]
            return [a]
        monkeypatch.setattr(bench, 'run_once', flaky_run_once)
        monkeypatch.setattr(bench.asyncio, 'run', lambda coro: coro)

        path = bench.run_benchmark(
            [(None, 'm1'), (None, 'm2')], out_dir=tmp_path)
        data = json.loads(path.read_text(encoding='utf-8'))
        # the first model's result survived the cancel
        assert len(data) == 1 and data[0]['model'] == 'm1'

    def test_auto_denies_escalation_and_restores(
            self, tmp_path, monkeypatch) -> None:
        from guru.agents import Agent
        from guru.domain import files as F

        class FakeAdapter:
            name = 'fake'

            def activate(self, m):
                session.current().num_ctx = 4096
                session.current().ctx_ceiling = 4096

        monkeypatch.setattr(bench, '_build_adapters',
                            lambda: [FakeAdapter()])
        monkeypatch.setattr(bench, '_adapter_for',
                            lambda name, built: built[0])
        seen = {}

        def run_once(base):
            # during the run, any permission prompt must be auto-denied
            seen['path'] = F._path_asker('Allow WRITE?')
            seen['domain'] = tools._domain_asker('Allow web?')
            a = Agent(id='main', title='main')
            a.state.messages = [{'role': 'assistant', 'content': 'x'}]
            return [a]
        monkeypatch.setattr(bench, 'run_once', run_once)
        monkeypatch.setattr(bench.asyncio, 'run', lambda coro: coro)

        bench.run_benchmark([(None, 'm1')], out_dir=tmp_path)
        assert seen['path'] is False and seen['domain'] is False
        # askers are restored (not left denying) after the run
        assert tools._domain_asker is None and F._path_asker is None


class TestBenchPlot:
    def test_points_skips_null_accuracy(self) -> None:
        records = [
            {'model': 'a', 'num_ctx': 4096, 'tokens_per_sec': 20.0,
             'seconds': 5.0, 'accuracy': 80},
            {'model': 'b', 'num_ctx': 8192, 'tokens_per_sec': 10.0,
             'seconds': 9.0, 'accuracy': None},
        ]
        assert bench_plot.points(records, x_key='tokens_per_sec') == [
            (20.0, 80, 'a (4096)')]

    def test_render_writes_two_pngs(self, tmp_path) -> None:
        recs = [{'model': 'a', 'num_ctx': 4096, 'tokens_per_sec': 20.0,
                 'seconds': 5.0, 'accuracy': 80}]
        rp = tmp_path / 'results-x.json'
        rp.write_text(json.dumps(recs), encoding='utf-8')
        paths = bench_plot.render(rp, out_dir=tmp_path)
        assert len(paths) == 2
        for p in paths:
            assert p.exists() and p.stat().st_size > 0
