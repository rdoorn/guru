"""Headless benchmark: run coding models through guru on one prompt and record
cost/behavior metrics for a speed-vs-accuracy comparison."""
import asyncio
import io
import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console

from guru import config, session, ui
from guru.adapters.anthropic import AnthropicAdapter
from guru.adapters.litellm import LiteLLMAdapter
from guru.adapters.ollama import OllamaAdapter
from guru.agents import Agent, AgentManager
from guru.domain import conversation, files, tools

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


def _quiet_console() -> Console:
    """A console that discards output (headless benchmark, no TUI)."""
    return Console(file=io.StringIO(), force_terminal=False)


class _Bench:
    """Compact headless coordinator: mirrors the TUI spawn/check/join mailbox
    with no UI, so sub-agents actually run and can be measured."""

    def __init__(self, base) -> None:
        self.base = base
        self.manager = AgentManager()
        self.loop = None
        self.barriers: dict = {}

    def _agent_for_state(self, st):
        return next(
            (a for a in self.manager.agents if a.state is st), None)

    def _configure(self, agent, can_spawn, role=None, skill=None) -> None:
        st = agent.state
        st.messages = [{'role': 'system',
                        'content': config.build_system_prompt()}]
        if can_spawn:
            st.messages[0]['content'] += "\n\n" + config.DELEGATION_HINT
        st.active_tools, st.active_tool_names = tools.initial_tools(can_spawn)
        st.active_role = role or None
        st.active_skill = skill or None
        st.can_spawn = can_spawn
        st.model = self.base.model
        st.adapter = self.base.adapter
        st.num_ctx = self.base.num_ctx
        st.ctx_ceiling = self.base.ctx_ceiling
        st.model_size = self.base.model_size
        agent.console = _quiet_console()

    def _work(self, agent) -> None:
        tok = session.use(agent.state)
        ct = ui.use_console(agent.console)
        try:
            while agent.queue:
                msg = agent.queue.pop(0)
                agent.state.messages.append(
                    {'role': 'user', 'content': msg})
                conversation.refresh_system_context()
                session.adapter.run_turn()
        except Exception:                                # noqa: BLE001
            pass
        finally:
            ui.reset_console(ct)
            session.reset(tok)
            self.loop.call_soon_threadsafe(self._on_done, agent)

    def _launch(self, agent) -> None:
        agent.busy = True
        self.loop.run_in_executor(None, self._work, agent)

    def _final_answer(self, agent) -> str:
        for m in reversed(agent.state.messages):
            if conversation.msg_role(m) == 'assistant' \
                    and conversation.msg_content(m).strip():
                return conversation.msg_content(m).strip()
        return ''

    def _deliver(self, parent, payload) -> None:
        parent.queue.append(payload)
        if not parent.busy:
            self._launch(parent)

    def _report(self, child) -> None:
        parent = child.parent
        if parent is None:
            return
        answer = self._final_answer(child)
        bar = self.barriers.get(parent)
        if bar is not None and child.title in bar['remaining']:
            bar['remaining'].discard(child.title)
            bar['results'][child.title] = (child.task, answer)
            if not bar['remaining']:
                del self.barriers[parent]
                combined = "\n".join(
                    f"[{t}] {tk}\n{ans}"
                    for t, (tk, ans) in bar['results'].items())
                self._deliver(parent, "[joined]\n" + combined)
        else:
            self._deliver(parent, f"[result from {child.title}]\n{answer}")

    def _on_done(self, agent) -> None:
        agent.busy = False
        if agent.queue:
            self._launch(agent)
            return
        if agent.parent is not None:
            self._report(agent)

    def _spawn(self, task, role='', skill='') -> str:
        parent = self._agent_for_state(session.current())
        title = f"agent{len(self.manager.agents)}"
        child = Agent(id=title, title=title)
        self._configure(child, can_spawn=False, role=role, skill=skill)
        child.task = task
        child.queue.append(task)

        def _add_start() -> None:
            child.parent = parent
            self.manager.agents.append(child)
            self._launch(child)

        self.loop.call_soon_threadsafe(_add_start)
        return f"Spawned {title}; its result is delivered when it finishes."

    def _check(self, target) -> str:
        caller = self._agent_for_state(session.current())
        kids = [a for a in self.manager.agents if a.parent is caller]
        if not kids:
            return "You have no sub-agents."
        return "Sub-agents:\n" + "\n".join(
            f"{a.title}: {'running' if a.busy else 'done'}" for a in kids)

    def _join(self, targets) -> str:
        caller = self._agent_for_state(session.current())
        kids = {a.title: a for a in self.manager.agents
                if a.parent is caller}
        names = [t for t in targets.replace(',', ' ').split() if t]
        chosen = [kids[n] for n in names if n in kids]
        if not chosen:
            return "None of those are your sub-agents."
        remaining = {a.title for a in chosen if a.busy}
        results = {a.title: (a.task, self._final_answer(a))
                   for a in chosen if not a.busy}
        if remaining:
            self.barriers[caller] = {'remaining': remaining,
                                     'results': results}
            return "Waiting for: " + ", ".join(sorted(remaining))
        return "[joined]\n" + "\n".join(
            f"[{t}] {tk}\n{ans}" for t, (tk, ans) in results.items())

    async def run(self, prompt) -> list:
        self.loop = asyncio.get_running_loop()
        tools.set_spawn_handler(self._spawn)
        tools.set_check_handler(self._check)
        tools.set_join_handler(self._join)
        try:
            main = self.manager.active
            self._configure(main, can_spawn=True)
            main.queue.append(prompt)
            self._launch(main)
            while any(a.busy or a.queue for a in self.manager.agents):
                await asyncio.sleep(0.05)
        finally:
            tools.set_spawn_handler(None)
            tools.set_check_handler(None)
            tools.set_join_handler(None)
        return list(self.manager.agents)


async def run_once(base_state) -> list:
    """Run PROMPT through a fresh main agent (+ any sub-agents it spawns) on
    base_state's adapter/model/ctx. Returns all agents that ran (main
    first)."""
    return await _Bench(base_state).run(PROMPT)


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
    and write bench/results-<timestamp>.json. The file is rewritten after every
    model, so a Ctrl+C mid-run keeps the results gathered so far. Returns the
    file path."""
    built = _build_adapters()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    path = out_dir / f'results-{stamp}.json'
    records = []

    def _flush() -> None:
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False),
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
                rec = collect_metrics(model, base.num_ctx, base.ctx_ceiling,
                                      secs, agents)
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
