# Coding-Model Benchmark (Stage 1) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Headlessly run several Ollama coding models through guru on one fixed prompt, capture cost/behavior metrics + the full result, then plot speed vs accuracy (accuracy scored in a separate review step).

**Architecture:** `guru/bench.py` drives each model through guru's real turn loop (`adapter.run_turn`) inside a compact headless asyncio orchestrator that mirrors the TUI mailbox (spawn/check/join) with no-op UI, so sub-agents actually run and are measured. Results go to `bench/results-<ts>.json` (`accuracy: null`); Claude fills accuracy; `guru/bench_plot.py` renders two scatter PNGs.

**Tech Stack:** Python 3.11+, asyncio, rich (quiet Console), matplotlib (new dep), pytest, flake8 (79-col, PEP8/257/484).

**Project conventions (override skill defaults):**
- **Commit ONCE at the end** (final task); test-first cadence within tasks.
- `.venv`; `flake8 guru tests` clean + `pytest` green before commit.
- Add deps with `uv`, never pip.
- No emojis.
- Design doc `docs/plans/2026-08-15-coding-benchmark-design.md` already exists; fold into the final commit.

The fixed prompt (define once as `guru/bench.py::PROMPT`):
`"i want you to inspect current code in this repository, and tell me something about code quality"`

---

### Task 1: models config + matplotlib dep

**Files:**
- Create: `bench/models.txt`
- Create: `guru/bench.py` (start it — `load_models` only)
- Modify: `pyproject.toml` (add matplotlib)
- Test: `tests/test_guru.py` (new class `TestBenchModels`)

**Step 1: add dep** — `uv add matplotlib` (adds to `pyproject.toml`). Confirm it imports: `.venv/bin/python -c "import matplotlib"`.

**Step 2: create `bench/models.txt`:**
```
# One model per line, optionally adapter-qualified as 'adapter|model'.
# No prefix = Ollama. '#' comments and blanks ignored. Runs in order.
qwen3:14b
devstral-small-2:24b
gpt-oss-20b-32k:latest
batiai/qwen3.6-27b:q3
# Cloud baseline via the configured litellm adapter (much faster; we want the
# magnitude). Requires the litellm adapter set up in ~/.guru/adapters.toml.
litellm|aws/claude-4-8-opus
```

`load_models` returns `[(adapter_name_or_None, model), ...]` — split each line
on the first `|`; no `|` → `(None, line)` (Ollama). Adjust the Task-1 tests to
assert the tuple form (e.g. `('litellm', 'aws/claude-4-8-opus')` and
`(None, 'qwen3:14b')`).

**Step 3: failing test** (append to `tests/test_guru.py`; add `from guru import bench` to imports):
```python
class TestBenchModels:
    def test_load_models_parses_and_filters(
            self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'models.txt'
        p.write_text("# a comment\nqwen3:14b\n\n  devstral:24b \n",
                     encoding='utf-8')
        assert bench.load_models(p) == ['qwen3:14b', 'devstral:24b']

    def test_load_models_missing_returns_empty(self, tmp_path) -> None:
        assert bench.load_models(tmp_path / 'nope.txt') == []
```

**Step 4: run** → FAIL (`guru.bench` missing).

**Step 5: implement** `guru/bench.py` (start the module):
```python
"""Headless benchmark: run coding models through guru on one prompt and
record cost/behavior metrics for a speed-vs-accuracy comparison."""
from pathlib import Path

PROMPT = ("i want you to inspect current code in this repository, and tell me"
          " something about code quality")

BENCH_DIR = Path('bench')
MODELS_FILE = BENCH_DIR / 'models.txt'


def load_models(path: Path = MODELS_FILE) -> list:
    """Read model ids from a file (one per line; '#' comments/blanks ignored)."""
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        s = ln.split('#', 1)[0].strip()
        if s:
            out.append(s)
    return out
```

**Step 6: run** `tests/test_guru.py::TestBenchModels` → PASS; `flake8 guru bench tests` clean.

---

### Task 2: metrics collection from a finished run

**Files:** Modify `guru/bench.py`; Test `tests/test_guru.py::TestBenchMetrics`.

Metrics are gathered from the agents after their turns (do NOT call
`after_turn`, so `tool` messages remain for counting).

**Step 1: failing tests:**
```python
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
        from guru import bench
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
        assert rec['result'] == 'ANSWER'          # main's last answer
        assert rec['accuracy'] is None and rec['error'] is None
```

**Step 2: run** → FAIL.

**Step 3: implement** in `guru/bench.py`:
```python
from guru.domain import conversation


def _final_answer(agent) -> str:
    for m in reversed(agent.state.messages):
        if conversation.msg_role(m) == 'assistant' \
                and conversation.msg_content(m).strip():
            return conversation.msg_content(m).strip()
    return ''


def _tool_names(agent) -> list:
    names = []
    for m in agent.state.messages:
        if conversation.msg_role(m) == 'tool':
            names.append(m.get('tool_name', '') if isinstance(m, dict)
                         else '')
    return names


def collect_metrics(model, num_ctx, ceiling, seconds, agents,
                    error=None) -> dict:
    """Aggregate a finished run (main + sub-agents) into a metrics record."""
    tools_called = []
    for a in agents:
        tools_called += _tool_names(a)
    tokens_in = sum(a.state.session_in for a in agents)
    tokens_out = sum(a.state.session_out for a in agents)
    tps = round(tokens_out / seconds, 1) if seconds > 0 else 0.0
    return {
        'model': model, 'num_ctx': num_ctx, 'ctx_ceiling': ceiling,
        'seconds': round(seconds, 1),
        'tokens_in': tokens_in, 'tokens_out': tokens_out,
        'tokens_per_sec': tps,
        'tools_called': tools_called, 'tool_count': len(tools_called),
        'agents_used': len(agents),
        'agents': [{'title': a.title, 'model': a.state.model,
                    'role': a.state.active_role,
                    'skill': a.state.active_skill} for a in agents],
        'spawn_calls': sum(1 for n in tools_called if n == 'spawn'),
        'result': _final_answer(agents[0]) if agents else '',
        'accuracy': None, 'error': error,
    }
```

**Step 4: run** → PASS; flake8 clean.

---

### Task 3: headless orchestrator (spawn/check/join)

**Files:** Modify `guru/bench.py`; Test `tests/test_guru.py::TestBenchOrchestrator`.

Build a compact asyncio coordinator that mirrors the TUI's mailbox. Use
`guru/tui.py` as the reference for `_spawn_agent`, `_deliver`, `_report`,
`_on_done`, `_check` (`_do_check`), `_join` (`_do_join`) and the `barriers`
dict — copy that logic, replacing per-agent rich consoles with a quiet
`Console(file=io.StringIO())` and dropping all UI calls (`_invalidate`,
status/tab rendering). Key references already confirmed:
- `AgentManager()` starts with a `main` Agent; `Agent` has `id/title/queue/
  busy/state/console/parent/task` (`guru/agents.py`).
- Configure an agent exactly like `tui._configure`: `messages=[system]`
  (+ `config.DELEGATION_HINT` when `can_spawn`), `active_tools=[search_tools,
  use_skill]` (+ `spawn/check/join` when `can_spawn`), copy `model/adapter/
  num_ctx/ctx_ceiling/model_size` from a base state, set `active_role/skill`.
- A worker runs like `tui._work`: `session.use(agent.state)` +
  `ui.use_console(agent.console)`, then for each queued message append the
  user turn, `conversation.refresh_system_context()`, `session.adapter.
  run_turn()` (do NOT call `after_turn`), and on finish `call_soon_threadsafe`
  an `_on_done` that relaunches on remaining queue or reports to the parent.
- Install handlers with `tools.set_spawn_handler`, `set_check_handler`,
  `set_join_handler` (signatures: `spawn(task, role, skill)`,
  `check(target)`, `join(targets)`), and restore them (set to None) at the end.

Expose one entry point:
```python
async def run_once(base_state) -> list:
    """Run PROMPT through a fresh main agent (+ any sub-agents it spawns) using
    base_state's adapter/model/ctx. Returns the list of all agents that ran
    (main first), after all queues drain and no agent is busy."""
```
Loop until `not any(a.busy or a.queue for a in manager.agents)`, then return
`manager.agents`.

**Step 1: failing test** (fake adapter whose turn spawns a child on the first
call, answers on the second — proving real sub-agent execution):
```python
class TestBenchOrchestrator:
    def test_spawn_runs_child_and_counts_agents(self, monkeypatch) -> None:
        import asyncio
        from guru import bench, session, tools
        from guru.adapters.base import Adapter

        class FakeAdapter(Adapter):
            name = 'fake'
            def available(self): return True
            def list_models(self): return []
            def activate(self, m): pass
            def summarise(self, t): return 's'

            def run_turn(self):
                st = session.current()
                # main spawns once; everyone answers.
                if st.can_spawn and not any(
                        m.get('tool_name') == 'spawn'
                        for m in st.messages if isinstance(m, dict)):
                    tools.spawn('sub task', role='', skill='')
                    st.messages.append(
                        {'role': 'tool', 'tool_name': 'spawn',
                         'content': 'ok'})
                st.session_out += 5
                st.messages.append(
                    {'role': 'assistant', 'content': 'done'})

        base = session.SessionState()
        base.adapter = FakeAdapter()
        base.model = 'fake'; base.num_ctx = 4096

        agents = asyncio.run(bench.run_once(base))
        try:
            titles = [a.title for a in agents]
            assert 'main' in titles and len(agents) >= 2   # spawned a child
        finally:
            tools.set_spawn_handler(None)
            tools.set_check_handler(None)
            tools.set_join_handler(None)
```

**Step 2: run** → FAIL. **Step 3: implement** `run_once` + the coordinator
(mirror tui as above). **Step 4: run** → PASS; full suite green; flake8 clean;
`.venv/bin/python -c "import guru.bench"` OK.

Note: keep the coordinator self-contained in `bench.py`; do not modify
`guru/tui.py`.

---

### Task 4: the per-model runner

**Files:** Modify `guru/bench.py`; Test `tests/test_guru.py::TestBenchRunner`.

**Adapter resolution (multi-provider).** The runner must serve both Ollama and
cloud (litellm) models. Reuse guru's own adapter construction: call
`cli._build_adapters()` (or `config.load_adapter_configs()` + the adapter
classes) once to get the configured adapters keyed by type/name. A small
`_adapter_for(adapter_name)` returns the `OllamaAdapter` when `adapter_name` is
None, else the configured adapter whose type/name matches (e.g. `litellm`).
Note: only Ollama has the GPU auto-fit; for litellm, `activate` sets its own
`num_ctx` (or leave guru's default) — `collect_metrics` just records whatever
`base.num_ctx/ctx_ceiling` end up as. The headless orchestrator is
adapter-agnostic (it only calls `session.adapter.run_turn`), and spawn is known
to work on Opus/litellm, so multi-agent runs are measured there too.

**Step 1: implement** `run_benchmark(models, out_dir=BENCH_DIR)` where `models`
is the `[(adapter_name, model), ...]` list from `load_models`:
- For each `(adapter_name, model)`: resolve the adapter via `_adapter_for`, a
  fresh `SessionState` as the base, `session.use(base)`,
  `session.adapter = adapter`, `adapter.activate(model)` (fills
  `base.model/num_ctx/ctx_ceiling`), wrapped in try/except → on error record
  `collect_metrics(..., agents=[], error=str(e))` (guard `agents=[]` in
  collect_metrics: `result=''`).
- Time `asyncio.run(run_once(base))`; `collect_metrics(model, base.num_ctx,
  base.ctx_ceiling, seconds, agents)`.
- Print a one-line progress note per model via `print()` (this is a CLI
  script, plain stdout is fine).
- Write all records to `out_dir/results-<stamp>.json` — the caller passes the
  stamp (tests pass a fixed one) since `bench.py` must not call `Date.now`
  itself is irrelevant here (normal Python `time`/`datetime` is fine in guru;
  only Workflow scripts forbid it). Use `datetime.now()` for the filename.
- Add `if __name__ == '__main__': run_benchmark(load_models())`.

Make `collect_metrics` tolerate `agents=[]` (already returns `result=''`,
`agents_used=0`).

**Step 2: test** the JSON writing with a monkeypatched `run_once` + a fake
adapter so no real model is needed:
```python
class TestBenchRunner:
    def test_writes_results_json(self, tmp_path, monkeypatch) -> None:
        import asyncio
        from guru import bench, session
        from guru.agents import Agent

        def fake_run_once(base):
            a = Agent(id='main', title='main')
            a.state.model = base.model
            a.state.session_out = 10
            a.state.messages = [{'role': 'assistant', 'content': 'A'}]
            return [a]
        monkeypatch.setattr(bench, 'run_once',
                            lambda base: fake_run_once(base))

        class FakeAdapter:
            name = 'fake'
            def activate(self, m):
                session.current().model = m
                session.current().num_ctx = 4096
                session.current().ctx_ceiling = 4096
        monkeypatch.setattr(bench, 'OllamaAdapter', lambda: FakeAdapter())
        # run_once is sync-mocked; make asyncio.run a passthrough
        monkeypatch.setattr(bench.asyncio, 'run', lambda coro: coro)

        path = bench.run_benchmark(['m1'], out_dir=tmp_path)
        import json
        data = json.loads(path.read_text())
        assert data[0]['model'] == 'm1' and data[0]['result'] == 'A'
```
(Adjust the async/mocking shape as needed so the test drives `run_benchmark`
without a real Ollama; the key assertion is a results JSON with the record.)

**Step 3:** run → PASS; flake8 clean.

---

### Task 5: the plot

**Files:** Create `guru/bench_plot.py`; Test `tests/test_guru.py::TestBenchPlot`.

**Step 1: failing test** for the pure data-prep:
```python
class TestBenchPlot:
    def test_points_skips_null_accuracy(self) -> None:
        from guru import bench_plot
        records = [
            {'model': 'a', 'num_ctx': 4096, 'tokens_per_sec': 20.0,
             'seconds': 5.0, 'accuracy': 80},
            {'model': 'b', 'num_ctx': 8192, 'tokens_per_sec': 10.0,
             'seconds': 9.0, 'accuracy': None},
        ]
        pts = bench_plot.points(records, x_key='tokens_per_sec')
        assert pts == [(20.0, 80, 'a (4096)')]
```

**Step 2: implement** `guru/bench_plot.py`:
```python
"""Render benchmark results as speed-vs-accuracy scatter plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')                       # headless backend
import matplotlib.pyplot as plt             # noqa: E402


def points(records: list, x_key: str) -> list:
    """[(x, accuracy, 'model (ctx)')] for records with a numeric accuracy."""
    out = []
    for r in records:
        acc = r.get('accuracy')
        if acc is None:
            continue
        out.append((r.get(x_key, 0), acc,
                    f"{r['model']} ({r.get('num_ctx', '?')})"))
    return out


def _scatter(records, x_key, xlabel, path) -> None:
    pts = points(records, x_key)
    fig, ax = plt.subplots(figsize=(9, 6))
    for x, y, label in pts:
        ax.scatter(x, y)
        ax.annotate(label, (x, y), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('accuracy (0-100)')
    ax.set_ylim(0, 100)
    ax.set_title('guru coding-model benchmark')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render(results_path: Path, out_dir: Path = Path('bench')) -> list:
    records = json.loads(Path(results_path).read_text(encoding='utf-8'))
    a = out_dir / 'speed_tokens.png'
    b = out_dir / 'speed_walltime.png'
    _scatter(records, 'tokens_per_sec', 'speed (tokens/sec)', a)
    _scatter(records, 'seconds', 'wall-time (seconds, lower=faster)', b)
    return [a, b]


if __name__ == '__main__':
    import sys
    render(Path(sys.argv[1]))
```

**Step 3:** run `TestBenchPlot` → PASS. Then a smoke: build a tiny results
list with accuracy set, call `render` to a tmp dir, assert both PNGs exist and
are non-empty (add that as a second test). flake8 clean (note the `# noqa:
E402` on the pyplot import after `matplotlib.use`).

---

### Task 6: manual run + final verify + commit

**Step 1 (manual, needs Ollama + models):**
- `.venv/bin/python -m guru.bench` → writes `bench/results-<ts>.json`.
- Paste the results file to Claude; Claude scores each `result` 0-100 vs the
  real repo and writes the `accuracy` fields back.
- `.venv/bin/python -m guru.bench_plot bench/results-<ts>.json` → two PNGs.

**Step 2:** `flake8 guru bench tests` clean; `pytest -q` green.

**Step 3: single commit** (incl. design doc, `bench/`, `guru/bench.py`,
`guru/bench_plot.py`, `pyproject.toml`):
```bash
git add -A
git commit -m "feat: headless coding-model benchmark with speed/accuracy plots"
```

**Step 4:** remind user to `ssh-add` + push; and that Task-6 step-1 (real run
+ accuracy scoring) is done together interactively after the code lands.

---

## Notes for the executor
- Do NOT modify `guru/tui.py`; the orchestrator is a self-contained mirror in
  `bench.py`.
- Do NOT call `conversation.after_turn()` in the bench worker — retention would
  strip the `tool` messages we count.
- Restore `tools.set_spawn_handler/check/join` to `None` after each run so
  benchmark state can't leak into other tests.
- `bench/*.json` and `bench/*.png` are artifacts — add `bench/results-*.json`
  and `bench/*.png` to `.gitignore` (keep `bench/models.txt`).
- matplotlib uses the `Agg` backend (no display needed).
