"""Full-screen multi-viewport TUI (Phase B — parallel turns + delegation).

Each agent viewport owns its own conversation, tools, model, counters, and
output console. Submitting runs that agent's turn in a background thread; the
prompt stays live and you can queue more or switch viewports
(Shift+Left/Right). Ctrl+N spawns a new agent (inherits the current model). A
single Ctrl+C cancels the active agent's running turn (cooperatively); a
double Ctrl+C exits.

Turns run in parallel: each agent binds its own ``session.state`` and console
via contextvars for the duration of its turn, so the adapters — which read and
write ``session`` — operate on that agent's state without a shared lock. Local
Ollama turns against the same model still serialize inside the daemon; remote
adapters (Anthropic/LiteLLM) and mixed models parallelise fully.

Delegation is a mailbox: a spawned sub-agent's final answer is delivered back
into its parent's queue when it finishes, auto-waking the parent to synthesise
— the parent never blocks, so it stays free to take new tasks or spawn more.
``check`` inspects sub-agents without blocking; ``join`` resumes the parent
once a named group all finish. All queue/busy/launch/deliver transitions run
on the event-loop thread (workers only run the blocking turn), so there are no
races on ``busy`` or on the agent list.

Layout, top to bottom: output pane · rule · prompt · rule · status · tabs.
"""
import asyncio
import shutil
import threading
from pathlib import Path

from prompt_toolkit import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import HorizontalLine, TextArea
from rich.console import Console

from guru import config, session, ui
from guru.agents import Agent, AgentManager
from guru.domain import conversation, files, tools

_CTX_COLOUR = {'green': 'ansigreen', 'yellow': 'ansiyellow', 'red': 'ansired'}
_CHROME_ROWS = 5   # 2 rules + prompt + status + tabs


class _BufferWriter:
    """File-like sink that routes rich output into an agent's buffer."""

    def __init__(self) -> None:
        self.target = None
        self.refresh = lambda: None
        self._partial = ''

    def write(self, text: str) -> None:
        if self.target is None:
            return
        self._partial += text
        while '\n' in self._partial:
            line, self._partial = self._partial.split('\n', 1)
            self.target.append(line)
        self.refresh()

    def flush(self) -> None:
        if self.target is not None and self._partial:
            self.target.append(self._partial)
            self._partial = ''
            self.refresh()


def _status_from(st) -> tuple:
    """Build the status tuple (left, ctx, right, colour) from state."""
    total = st.num_ctx or 1
    used = st.ctx_used or 0
    pct = min(1.0, used / total)
    filled = round(pct * 10)
    bar = '█' * filled + '░' * (10 - filled)
    if pct >= config.COMPACT_AT:
        colour = 'red'
    elif pct >= 0.70:
        colour = 'yellow'
    else:
        colour = 'green'
    model = (st.model or '?').split(':')[0]
    left = f"🤖 {model} | 💪 {st.model_size or '?'} | "
    ctx = f"🧠 {int(pct * 100)}% {bar}"
    # Composition of the resident context in tokens (rough, ~4 chars/token):
    # sys=system prompt+GURU.md, tl=tool schemas, in=prompts, out=responses,
    # res=lingering tool output (usually 0 after pruning).
    bd = conversation.context_breakdown(
        st.messages, st.active_tool_names, st.can_spawn)

    def _kt(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    comp = (f"📊 sys:{_kt(bd['sys'])} tl:{_kt(bd['tools'])}"
            f" in:{_kt(bd['in'])} out:{_kt(bd['out'])}")
    if bd['toolout']:
        comp += f" res:{_kt(bd['toolout'])}"
    right = (
        f" · {comp}"
        f" | ↓ {st.session_in} | ↑ {st.session_out}"
        f" | 📁 {Path.cwd().name} | 🌿 {st.git_branch or 'none'}"
    )
    return left, ctx, right, colour


def run() -> None:
    ui.refresh_git_branch()
    manager = AgentManager()
    # The main agent adopts the startup session cli already populated (model,
    # adapter, system prompt, context accounting) and may delegate via spawn.
    main = manager.active
    main.state = session.current()
    main.state.can_spawn = True
    if (main.state.messages
            and main.state.messages[0].get('role') == 'system'):
        main.state.messages[0]['content'] += "\n\n" + config.DELEGATION_HINT
    for fn in (tools.spawn, tools.check, tools.join):
        if fn not in main.state.active_tools:
            main.state.active_tools.append(fn)
    main.append("guru — Ctrl+N new agent · Shift+Left/Right switch"
                " · Ctrl+C cancel · double Ctrl+C exit")

    state = {'loop': None, 'prompt': None, 'closing': False}
    cols = shutil.get_terminal_size((100, 30)).columns
    barriers: dict = {}   # parent Agent -> {'remaining': set, 'results': dict}
    # One serialized permission prompt at a time: every agent's access
    # question funnels through this lock and is shown in the [main] viewport
    # on the loop thread, so sub-agents never pop competing prompts.
    ask_lock = threading.Lock()

    def _open_prompt(req: dict) -> None:
        """Show a permission question in the [main] viewport (loop thread)."""
        state['prompt'] = req
        manager.active_index = 0          # bring the main viewport forward
        main_agent = manager.agents[0]
        main_agent.append("")
        main_agent.append(
            f"[access] Allow access to '{req['target']}' ?"
            "  Y = allow · N = deny  (Enter = allow)")
        _invalidate()

    def _resolve_prompt(result: bool) -> None:
        """Answer the pending permission question (loop thread)."""
        req = state['prompt']
        if req is None:
            return
        state['prompt'] = None
        manager.agents[0].append(
            f"[access] → {'allowed' if result else 'denied'}.")
        req['result'] = result
        req['event'].set()
        _invalidate()

    def _domain_asker(target: str) -> bool:
        # Ask in the [main] viewport, on the loop thread, one question at a
        # time (ask_lock). The worker blocks here until answered. Default is
        # allow (Enter); any failure to get an answer — including shutdown —
        # denies, so a broken prompt can never silently grant access.
        with ask_lock:
            if state['closing']:
                return False
            event = threading.Event()
            req = {'target': target, 'event': event, 'result': False}
            state['loop'].call_soon_threadsafe(_open_prompt, req)
            while not event.wait(0.25):
                if state['closing']:
                    return False
            return req['result']

    tools.set_domain_asker(_domain_asker)
    files.set_path_asker(_domain_asker)

    # --- per-agent output ----------------------------------------------------

    def _attach_console(agent) -> None:
        writer = _BufferWriter()
        writer.target = agent
        writer.refresh = _invalidate
        agent.console = Console(
            file=writer, force_terminal=True,
            color_system='standard', width=cols)

    # --- background execution ------------------------------------------------
    #
    # Workers only run the blocking turn; every queue/busy/launch/deliver/
    # barrier transition happens on the loop thread (via call_soon_threadsafe
    # or _on_loop), so there are no races on ``busy`` or the agent list.

    def _work(agent) -> None:
        # Bind this agent's state + console to the worker thread's context, so
        # the adapters (which touch ``session``) and tool output target this
        # agent only. No lock: other agents run concurrently on their own.
        token = session.use(agent.state)
        ctoken = ui.use_console(agent.console)
        try:
            while agent.queue:
                message = agent.queue.pop(0)
                agent.state.messages.append(
                    {'role': 'user', 'content': message})
                session.adapter.run_turn()
                # Prune tool output and compact if needed, so context does
                # not grow unbounded across turns (the TUI had no compaction).
                conversation.after_turn()
        except Exception as e:                           # noqa: BLE001
            agent.append(f"[error] {e}")
        finally:
            agent.console.file.flush()
            ui.reset_console(ctoken)
            session.reset(token)
            state['loop'].call_soon_threadsafe(_on_done, agent)

    def _launch(agent) -> None:
        """Mark busy and start a worker turn (loop thread only)."""
        agent.busy = True
        agent.status = 'thinking'
        state['loop'].run_in_executor(None, _work, agent)

    def _submit(text: str) -> None:
        agent = manager.active
        agent.append(f"> {text}")
        agent.queue.append(text)
        if not agent.busy:
            _launch(agent)

    def _agent_for_state(st):
        return next((a for a in manager.agents if a.state is st), None)

    def _final_answer(agent) -> str:
        for m in reversed(agent.state.messages):
            role = (m.get('role') if isinstance(m, dict)
                    else getattr(m, 'role', ''))
            content = (m.get('content') if isinstance(m, dict)
                       else getattr(m, 'content', '')) or ''
            if role == 'assistant' and content.strip():
                return content.strip()
        return '(no answer produced)'

    def _format_join(results: dict) -> str:
        parts = ["[joined results]"]
        for tid, (task, ans) in results.items():
            parts.append(f"\n— {tid} · task: {task}\n{ans}")
        return "\n".join(parts)

    def _deliver(parent, notice: str, payload: str) -> None:
        """Drop a result into a parent's mailbox and wake it (loop thread)."""
        parent.append(notice)
        parent.queue.append(payload)
        if not parent.busy:
            _launch(parent)
        _invalidate()

    def _report(child) -> None:
        """Deliver a finished child's result to its parent (loop thread)."""
        parent = child.parent
        answer = _final_answer(child)
        bar = barriers.get(parent)
        if bar is not None and child.title in bar['remaining']:
            bar['remaining'].discard(child.title)
            bar['results'][child.title] = (child.task, answer)
            if not bar['remaining']:
                del barriers[parent]
                _deliver(parent, "[inbox] join complete",
                         _format_join(bar['results']))
        else:
            _deliver(
                parent,
                f"[inbox] result from {child.title}",
                f"[result from {child.title} · task: {child.task}]\n{answer}")

    def _on_done(agent) -> None:
        """A worker finished (loop thread): relaunch, report, or go idle."""
        agent.busy = False
        agent.status = 'idle'
        if agent.queue:                 # more arrived during teardown
            _launch(agent)
            _invalidate()
            return
        if agent.parent is not None:
            _report(agent)
        _invalidate()

    def _on_loop(fn):
        """Run fn() on the loop thread; block the caller for its result."""
        done = threading.Event()
        box: dict = {}

        def _runner() -> None:
            try:
                box['result'] = fn()
            except Exception as e:                       # noqa: BLE001
                box['error'] = e
            finally:
                done.set()

        state['loop'].call_soon_threadsafe(_runner)
        done.wait()
        if 'error' in box:
            raise box['error']
        return box['result']

    def _configure(agent, base, can_spawn: bool) -> None:
        """Seed a child agent's state from ``base`` and attach its console."""
        st = agent.state
        st.messages = [
            {'role': 'system', 'content': config.build_system_prompt()}]
        st.active_tools = [tools.search_tools]
        if can_spawn:
            st.messages[0]['content'] += "\n\n" + config.DELEGATION_HINT
            st.active_tools.extend([tools.spawn, tools.check, tools.join])
        st.active_tool_names = set()
        st.model = base.model
        st.adapter = base.adapter
        st.num_ctx = base.num_ctx
        st.ctx_ceiling = base.ctx_ceiling
        st.model_size = base.model_size
        st.git_branch = base.git_branch
        st.can_spawn = can_spawn
        _attach_console(agent)

    def _new_agent() -> None:
        base = manager.active.state
        agent = manager.add(f"agent{len(manager.agents)}")
        _configure(agent, base, can_spawn=True)
        agent.append(f"[{agent.title}] new agent · model {agent.state.model}")
        manager.active_index = len(manager.agents) - 1

    def _spawn_agent(task: str) -> str:
        """Spawn handler (worker thread): delegate a task in parallel.

        Sub-agents can't spawn further agents. The manager mutation, parent
        link, and turn launch are scheduled on the loop thread to avoid racing
        the renderer and other orchestration.
        """
        base = session.current()   # the delegating agent's state
        title = f"agent{len(manager.agents)}"
        child = Agent(id=title, title=title)
        _configure(child, base, can_spawn=False)
        child.task = task
        child.append(f"[{title}] spawned · task: {task}")
        child.append(f"> {task}")
        child.queue.append(task)

        def _add_and_start() -> None:
            child.parent = _agent_for_state(base)
            manager.agents.append(child)
            _launch(child)
            _invalidate()

        state['loop'].call_soon_threadsafe(_add_and_start)
        return (
            f"Spawned {title} to work on this task in parallel. Its result"
            f" will be delivered back to you automatically when it finishes."
        )

    def _do_check(caller_state, target: str) -> str:
        """Report sub-agent status/results for the caller (loop thread)."""
        caller = _agent_for_state(caller_state)
        children = [a for a in manager.agents if a.parent is caller]
        if not children:
            return "You have no sub-agents."
        target = (target or 'all').strip()
        if target in ('all', '*', ''):
            lines = [
                f"{a.title}: {'running' if a.busy else 'done'}"
                for a in children]
            return "Sub-agents:\n" + "\n".join(lines)
        match = next((a for a in children if a.title == target), None)
        if match is None:
            names = ', '.join(a.title for a in children)
            return f"No sub-agent named '{target}'. Yours: {names}."
        if match.busy:
            return f"{match.title}: running (task: {match.task})"
        return (f"{match.title}: done\ntask: {match.task}\n"
                f"{_final_answer(match)}")

    def _do_join(caller_state, titles: list) -> str:
        """Register a join barrier for the caller (loop thread)."""
        caller = _agent_for_state(caller_state)
        children = {a.title: a for a in manager.agents if a.parent is caller}
        targets = [children[t] for t in titles if t in children]
        if not targets:
            have = ', '.join(children) or 'none'
            return f"None of those are your sub-agents. Yours: {have}."
        remaining: set = set()
        results: dict = {}
        for a in targets:
            if a.busy:
                remaining.add(a.title)
            else:
                results[a.title] = (a.task, _final_answer(a))
        if remaining:
            barriers[caller] = {'remaining': remaining, 'results': results}
            waiting = ', '.join(sorted(remaining))
            return (f"Waiting for {waiting} to finish; I'll be resumed"
                    f" automatically with their combined results.")
        _deliver(caller, "[inbox] join complete", _format_join(results))
        return "Those sub-agents already finished; resuming with results now."

    def _check(target: str) -> str:
        st = session.current()
        return _on_loop(lambda: _do_check(st, target))

    def _join(targets: str) -> str:
        st = session.current()
        titles = [t for t in targets.replace(',', ' ').split() if t]
        return _on_loop(lambda: _do_join(st, titles))

    tools.set_spawn_handler(_spawn_agent)
    tools.set_check_handler(_check)
    tools.set_join_handler(_join)

    # --- views ---------------------------------------------------------------

    def _output():
        rows = shutil.get_terminal_size((100, 30)).lines
        visible = max(1, rows - _CHROME_ROWS)
        return ANSI('\n'.join(manager.active.lines[-visible:]))

    output = Window(
        FormattedTextControl(_output),
        wrap_lines=True, height=Dimension(weight=1))

    def _tabs() -> FormattedText:
        parts: list = []
        for active, title in manager.tabs():
            parts.append(
                ('reverse' if active else 'ansibrightblack', f'[{title}]'))
            parts.append(('', ' '))
        return FormattedText(parts)

    tabline = Window(FormattedTextControl(_tabs), height=1)

    def _status() -> FormattedText:
        # Each agent's live state is mutated in place by its worker thread, so
        # reading the active agent's state here always reflects current values.
        left, ctx, right, colour = _status_from(manager.active.state)
        return FormattedText([
            ('ansibrightblack', left),
            (_CTX_COLOUR[colour], ctx),
            ('ansibrightblack', right),
        ])

    statusline = Window(FormattedTextControl(_status), height=1)

    input_area = TextArea(height=1, prompt='> ', multiline=False)

    def _accept(buff) -> bool:
        text = buff.text.strip()
        if text:
            _submit(text)
        return False

    input_area.accept_handler = _accept

    # --- keys ----------------------------------------------------------------

    kb = KeyBindings()

    @kb.add('c-d')
    def _quit(event) -> None:
        event.app.exit()

    @kb.add('c-c')
    def _ctrl_c(event) -> None:
        if ui.note_ctrl_c():
            event.app.exit()
        elif manager.active.busy:
            manager.active.state.cancel_requested = True
        else:
            event.current_buffer.text = ''

    # eager=True so these win over the input buffer's default bindings.
    @kb.add('c-n', eager=True)
    def _spawn(event) -> None:
        _new_agent()

    # Shift+Left/Right cycle viewports. Plain Ctrl+Left/Right can't be used:
    # macOS grabs Ctrl+Arrow for Mission Control, so they never reach the
    # terminal (Shift+Arrow sends a distinct CSI 1;2 sequence that does).
    @kb.add('s-right', eager=True)
    def _next(event) -> None:
        manager.switch(1)

    @kb.add('s-left', eager=True)
    def _prev(event) -> None:
        manager.switch(-1)

    # Permission prompt answers — active only while a question is pending, so
    # y/n type normally otherwise. Enter defaults to allow.
    _has_prompt = Condition(lambda: state['prompt'] is not None)

    @kb.add('y', filter=_has_prompt, eager=True)
    @kb.add('Y', filter=_has_prompt, eager=True)
    @kb.add('enter', filter=_has_prompt, eager=True)
    def _prompt_allow(event) -> None:
        _resolve_prompt(True)

    @kb.add('n', filter=_has_prompt, eager=True)
    @kb.add('N', filter=_has_prompt, eager=True)
    def _prompt_deny(event) -> None:
        _resolve_prompt(False)

    # --- run -----------------------------------------------------------------

    root = HSplit([
        output,
        HorizontalLine(),
        input_area,
        HorizontalLine(),
        statusline,
        tabline,
    ])
    app = Application(
        layout=Layout(root, focused_element=input_area),
        key_bindings=kb,
        full_screen=True,
        mouse_support=False,
    )

    def _invalidate() -> None:
        app.invalidate()   # thread-safe in prompt_toolkit

    _attach_console(manager.active)

    async def _amain() -> None:
        state['loop'] = asyncio.get_running_loop()
        try:
            await app.run_async()
        finally:
            # Unblock any worker still waiting on a permission answer (deny).
            state['closing'] = True
            req = state['prompt']
            if req is not None:
                state['prompt'] = None
                req['result'] = False
                req['event'].set()

    asyncio.run(_amain())
