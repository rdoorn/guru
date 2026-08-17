"""Headless benchmark: run coding models through guru on one prompt and record
cost/behavior metrics for a speed-vs-accuracy comparison."""
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from guru import config, session
from guru.adapters.anthropic import AnthropicAdapter
from guru.adapters.litellm import LiteLLMAdapter
from guru.adapters.ollama import OllamaAdapter
from guru.domain import conversation, files, tools
from guru.orchestrator import Orchestrator

PROMPT = ("i want you to inspect current code in this repository, and tell me"
          " something about code quality")

BENCH_DIR = Path('bench')
MODELS_FILE = BENCH_DIR / 'models.txt'


def load_models(path: Path = MODELS_FILE) -> list:
    """Parse the model list. Each non-comment line is '[adapter|]model'; the
    adapter prefix is optional (None = the default Ollama adapter). Returns a
    list of (adapter_name_or_None, model) tuples in file order."""
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        s = ln.split('#', 1)[0].strip()
        if not s:
            continue
        if '|' in s:
            adapter, _, model = s.partition('|')
            out.append((adapter.strip() or None, model.strip()))
        else:
            out.append((None, s))
    return out


def sort_models(models: list) -> list:
    """Order (adapter, model) tuples by the name after the last '/', so related
    models line up regardless of their path prefix (e.g. all the qwen* group
    together whether they're 'qwen3:14b' or 'hf.co/unsloth/Qwen3-14B-...')."""
    return sorted(models, key=lambda am: am[1].rsplit('/', 1)[-1].lower())


def _final_answer(agent) -> str:
    for m in reversed(agent.state.messages):
        if conversation.msg_role(m) == 'assistant' \
                and conversation.msg_content(m).strip():
            return conversation.msg_content(m).strip()
    return ''


def _tool_names(agent) -> list:
    out = []
    for m in agent.state.messages:
        if conversation.msg_role(m) == 'tool':
            out.append(m.get('tool_name', '') if isinstance(m, dict) else '')
    return out


def collect_metrics(model, num_ctx, ceiling, seconds, agents,
                    error=None) -> dict:
    """Aggregate a finished run (main + sub-agents) into a metrics record."""
    tools_called = []
    for a in agents:
        tools_called += _tool_names(a)
    tokens_in = sum(a.state.session_in for a in agents)
    tokens_out = sum(a.state.session_out for a in agents)
    tps = round(tokens_out / seconds, 1) if seconds > 0 else 0.0
    result = _final_answer(agents[0]) if agents else ''
    # A blank final answer is a run that explored (or errored) but never
    # synthesised — flag it so it is visibly distinct from a real result and
    # accuracy scoring can skip it. A caller-supplied error (timeout) wins.
    if error is None and not result.strip():
        error = 'empty answer'
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
        'result': result,
        'accuracy': None, 'error': error,
    }


def serialize_transcript(agents) -> list:
    """Flatten agents' message histories to JSON-safe records for debugging a
    run (why an answer was empty, which tools ran in what order). Normalises
    both dict messages and provider Message objects to role/content, keeping
    tool names and the names of any tool calls."""
    out = []
    for a in agents:
        msgs = []
        for m in a.state.messages:
            rec: dict = {'role': conversation.msg_role(m),
                         'content': conversation.msg_content(m)}
            if isinstance(m, dict) and m.get('tool_name'):
                rec['tool_name'] = m['tool_name']
            tcs = (m.get('tool_calls') if isinstance(m, dict)
                   else getattr(m, 'tool_calls', None))
            if tcs:
                names = []
                for tc in tcs:
                    fn = (tc.get('function') if isinstance(tc, dict)
                          else getattr(tc, 'function', None))
                    names.append(
                        (fn.get('name') if isinstance(fn, dict)
                         else getattr(fn, 'name', '')) if fn else '')
                rec['tool_calls'] = names
            msgs.append(rec)
        out.append(
            {'title': a.title, 'model': a.state.model, 'messages': msgs})
    return out


class _Bench(Orchestrator):
    """Headless coordinator: the shared spawn/check/join orchestrator with the
    default quiet console and no per-turn retention/timing, so sub-agents
    actually run and their raw behaviour can be measured."""

    def __init__(self, base) -> None:
        super().__init__()
        self.base = base

    async def _abort(self, grace: float = 30.0) -> None:
        """Cooperatively stop a stalled run: flag cancel on every agent (the
        adapters check it — Ollama aborts mid-stream) and wait briefly for the
        worker threads to wind down. Threads can't be force-killed, so this is
        best-effort; mid-stream cancel usually stops generation quickly."""
        assert self.loop is not None
        end = self.loop.time() + grace
        while any(a.busy for a in self.manager.agents):
            for a in self.manager.agents:
                a.state.cancel_requested = True
            if self.loop.time() >= end:
                break
            await asyncio.sleep(0.05)

    async def run(self, prompt, timeout=None) -> list:
        self.loop = asyncio.get_running_loop()
        assert self.loop is not None
        self.install_handlers()
        deadline = (self.loop.time() + timeout) if timeout and timeout > 0 \
            else None
        try:
            main = self.manager.active
            self.configure(main, self.base, can_spawn=True)
            main.queue.append(prompt)
            self.launch(main)
            while any(a.busy or a.queue for a in self.manager.agents):
                await asyncio.sleep(0.05)
                if deadline is not None and self.loop.time() >= deadline:
                    await self._abort()
                    break
        finally:
            self.clear_handlers()
        return list(self.manager.agents)


async def run_once(base_state) -> list:
    """Run PROMPT through a fresh main agent (+ any sub-agents it spawns) on
    base_state's adapter/model/ctx. Returns all agents that ran (main first).

    A per-model wall-clock ceiling (config.BENCH_MODEL_TIMEOUT) cooperatively
    cancels a stalled run so one slow model can't hang the suite."""
    return await _Bench(base_state).run(
        PROMPT, timeout=config.BENCH_MODEL_TIMEOUT)


_ADAPTER_CLASS = {
    'ollama': OllamaAdapter,
    'litellm': LiteLLMAdapter,
    'anthropic': AnthropicAdapter,
}


def _build_adapters() -> list:
    """The configured adapter instances (reuses guru's own construction)."""
    from guru import cli
    return cli._build_adapters()


def _adapter_for(name, built):
    """Resolve a models.txt adapter prefix to a built adapter instance. None
    -> the Ollama adapter; a name like 'litellm' -> that adapter class; else
    match by display name. Returns None if unresolved."""
    if not name:
        for a in built:
            if isinstance(a, OllamaAdapter):
                return a
        return OllamaAdapter()
    cls = _ADAPTER_CLASS.get(name.lower())
    if cls is not None:
        for a in built:
            if isinstance(a, cls):
                return a
    for a in built:
        if getattr(a, 'name', '').lower() == name.lower():
            return a
    return None


def run_benchmark(models, out_dir=BENCH_DIR):
    """Run each (adapter_name, model) through guru on PROMPT, collect metrics,
    and write bench/results-<timestamp>.json plus a companion
    transcript-<timestamp>.json (full message histories, for debugging why an
    answer was empty). Both are rewritten after every model, so a Ctrl+C
    mid-run keeps what was gathered. Returns the results file path."""
    built = _build_adapters()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    path = out_dir / f'results-{stamp}.json'
    tpath = out_dir / f'transcript-{stamp}.json'
    records = []
    transcripts = []

    def _flush() -> None:
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False),
                        encoding='utf-8')
        tpath.write_text(
            json.dumps(transcripts, indent=2, ensure_ascii=False),
            encoding='utf-8')

    # Sandbox: allow reading the repo (the task needs it) but AUTO-DENY every
    # escalation (writes, new dirs, web) — an unattended run must never sit on
    # or grant a permission prompt, and a model asking for more rights is
    # denied (which is itself part of what we measure).
    config.ALLOWED_READ_DIRS.add(str(Path.cwd().resolve()))
    tools.set_domain_asker(lambda q: False)
    files.set_path_asker(lambda q: False)

    try:
        for adapter_name, model in sort_models(models):
            adapter = _adapter_for(adapter_name, built)
            base = session.SessionState()
            base.adapter = adapter
            token = session.use(base)
            try:
                if adapter is None:
                    raise RuntimeError(f"no adapter for '{adapter_name}'")
                session.adapter = adapter
                adapter.activate(model)
                t0 = time.monotonic()
                agents = asyncio.run(run_once(base))
                secs = time.monotonic() - t0
                limit = config.BENCH_MODEL_TIMEOUT
                err = (f"timeout after {int(secs)}s (limit {limit}s)"
                       if limit and secs >= limit else None)
                rec = collect_metrics(model, base.num_ctx, base.ctx_ceiling,
                                      secs, agents, error=err)
                transcripts.append(
                    {'model': model, 'agents': serialize_transcript(agents)})
            except Exception as e:                       # noqa: BLE001
                rec = collect_metrics(model, base.num_ctx or 0,
                                      base.ctx_ceiling or 0, 0.0, [],
                                      error=str(e))
            finally:
                session.reset(token)
            print(f"[bench] {model}: {rec.get('error') or 'ok'}"
                  f" ({rec['seconds']}s, {rec['tool_count']} tools,"
                  f" {rec['agents_used']} agents)")
            records.append(rec)
            _flush()          # persist after each model (survives Ctrl+C)
    except KeyboardInterrupt:
        print(f"[bench] interrupted — {len(records)} result(s) saved to"
              f" {path}")
    finally:
        tools.set_domain_asker(None)
        files.set_path_asker(None)
    return path


if __name__ == '__main__':
    run_benchmark(load_models())
