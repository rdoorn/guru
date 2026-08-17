"""Shared sub-agent orchestrator: the spawn/check/join mailbox used by both the
interactive TUI and the headless benchmark.

An Orchestrator owns the AgentManager, the join barriers, and the asyncio loop,
and runs each agent's blocking turn in a worker thread, delivering a sub-agent
result back to its parent through a mailbox (the parent's queue). When a parent
``join``s a group, a barrier holds the results until the whole group finishes,
then delivers them together.

Front-ends subclass it and override a few hooks to supply their own surface:

* ``attach_console`` — the per-agent console (a buffered viewport vs. a quiet
  discard console),
* ``invalidate`` — signal the UI to redraw (no-op when headless),
* ``notice`` — post a mailbox line to a viewport (no-op when headless),
* ``post_turn`` — per-turn bookkeeping/footer (the TUI runs retention + prints
  timing; the benchmark deliberately skips both to measure raw behaviour),
* ``run_on_loop`` — how ``check``/``join`` reach the loop thread (the TUI hops;
  headless runs inline).

The mailbox/barrier logic lives here once and is exercised by the benchmark's
orchestrator tests.
"""
import asyncio
import io
import time
from typing import Optional

from rich.console import Console

from guru import config, log, session, ui
from guru.agents import Agent, AgentManager
from guru.domain import conversation, tools


class Orchestrator:
    """Owns the agents, join barriers, and the worker-thread turn loop."""

    def __init__(self, manager=None) -> None:
        self.manager = manager or AgentManager()
        self.barriers: dict = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    # --- hooks front-ends override -------------------------------------------

    def attach_console(self, agent) -> None:
        """Give ``agent`` a console. Default discards output (headless)."""
        agent.console = Console(file=io.StringIO(), force_terminal=False)

    def invalidate(self) -> None:
        """Ask the UI to redraw. No-op when headless."""

    def notice(self, agent, text: str) -> None:
        """Post a mailbox line to ``agent``'s viewport. No-op when headless."""

    def post_turn(self, agent, start: float) -> None:
        """Run after each turn (retention, a timing footer). No-op by default;
        ``start`` is the turn's ``time.monotonic()`` start."""

    def on_worker_error(self, agent, exc: Exception) -> None:
        """Handle an exception raised by a worker turn. Default logs it."""
        log.exc(f'orchestrator worker failed: {exc}')

    def run_on_loop(self, fn):
        """Run ``fn`` for the thread-sensitive check/join handlers. Default is
        inline; the TUI overrides to hop to the loop thread."""
        return fn()

    # --- shared helpers ------------------------------------------------------

    def agent_for_state(self, st):
        return next((a for a in self.manager.agents if a.state is st), None)

    def final_answer(self, agent) -> str:
        for m in reversed(agent.state.messages):
            if conversation.msg_role(m) == 'assistant' \
                    and conversation.msg_content(m).strip():
                return conversation.msg_content(m).strip()
        return '(no answer produced)'

    def configure(self, agent, base, can_spawn: bool,
                  role=None, skill=None) -> None:
        """Set up ``agent``'s fresh conversation + tools, inheriting the model
        and context from ``base``. Delegation-capable agents get the panel
        hint; sub-agents get a role/skill overlay."""
        st = agent.state
        st.messages = [
            {'role': 'system', 'content': config.build_system_prompt()}]
        if can_spawn:
            st.messages[0]['content'] += "\n\n" + config.DELEGATION_HINT
        st.active_tools, st.active_tool_names = tools.initial_tools(can_spawn)
        st.active_role = role or None
        st.active_skill = skill or None
        st.model = base.model
        st.adapter = base.adapter
        st.num_ctx = base.num_ctx
        st.ctx_ceiling = base.ctx_ceiling
        st.model_size = base.model_size
        st.git_branch = getattr(base, 'git_branch', None)
        st.can_spawn = can_spawn
        self.attach_console(agent)

    # --- worker thread -------------------------------------------------------

    def work(self, agent) -> None:
        """Drain ``agent``'s queue, running each user message as one turn with
        the agent's session + console bound. Runs in a background thread."""
        token = session.use(agent.state)
        ctoken = ui.use_console(agent.console)
        try:
            while agent.queue:
                message = agent.queue.pop(0)
                agent.state.messages.append(
                    {'role': 'user', 'content': message})
                conversation.refresh_system_context()
                start = time.monotonic()
                session.adapter.run_turn()
                self.post_turn(agent, start)
        except Exception as e:                           # noqa: BLE001
            self.on_worker_error(agent, e)
        finally:
            try:
                agent.console.file.flush()
            except Exception:                            # noqa: BLE001
                log.exc('console flush failed')
            ui.reset_console(ctoken)
            session.reset(token)
            assert self.loop is not None
            self.loop.call_soon_threadsafe(self.on_done, agent)

    def launch(self, agent) -> None:
        agent.busy = True
        agent.status = 'thinking'
        assert self.loop is not None
        self.loop.run_in_executor(None, self.work, agent)

    def submit(self, agent, text: str) -> None:
        """Queue a user message for ``agent`` and start it if idle."""
        self.notice(agent, f"> {text}")
        agent.queue.append(text)
        if not agent.busy:
            self.launch(agent)

    # --- mailbox / barriers --------------------------------------------------

    def _format_join(self, results: dict) -> str:
        parts = ["[joined results]"]
        for tid, (task, ans) in results.items():
            parts.append(f"\n— {tid} · task: {task}\n{ans}")
        return "\n".join(parts)

    def deliver(self, parent, notice: str, payload: str) -> None:
        """Post a result to ``parent``'s mailbox and resume it if idle."""
        self.notice(parent, notice)
        parent.queue.append(payload)
        if not parent.busy:
            self.launch(parent)
        self.invalidate()

    def report(self, child) -> None:
        """A finished ``child`` reports to its parent — into a pending join
        barrier if one is open, otherwise delivered on its own."""
        parent = child.parent
        if parent is None:
            return
        answer = self.final_answer(child)
        bar = self.barriers.get(parent)
        if bar is not None and child.title in bar['remaining']:
            bar['remaining'].discard(child.title)
            bar['results'][child.title] = (child.task, answer)
            if not bar['remaining']:
                del self.barriers[parent]
                self.deliver(parent, "[inbox] join complete",
                             self._format_join(bar['results']))
        else:
            self.deliver(
                parent,
                f"[inbox] result from {child.title}",
                f"[result from {child.title} · task: {child.task}]\n{answer}")

    def on_done(self, agent) -> None:
        agent.busy = False
        agent.status = 'idle'
        if agent.queue:
            self.launch(agent)
            self.invalidate()
            return
        if agent.parent is not None:
            self.report(agent)
        self.invalidate()

    # --- delegation handlers (installed via tools.set_*_handler) -------------

    def spawn(self, task: str, role: str = '', skill: str = '') -> str:
        base = session.current()
        parent = self.agent_for_state(base)
        title = f"agent{len(self.manager.agents)}"
        child = Agent(id=title, title=title)
        self.configure(child, base, can_spawn=False, role=role, skill=skill)
        child.task = task
        child.parent = parent
        self.notice(child, f"[{title}] spawned · task: {task}")
        child.queue.append(task)

        # Append to the agent list on the loop thread — never mutate it from a
        # worker thread while the loop may be iterating it.
        def _start() -> None:
            self.manager.agents.append(child)
            self.launch(child)
            self.invalidate()

        assert self.loop is not None
        self.loop.call_soon_threadsafe(_start)
        return (
            f"Spawned {title} to work on this task in parallel. Its result"
            f" will be delivered back to you automatically when it finishes.")

    def do_check(self, caller_state, target: str) -> str:
        caller = self.agent_for_state(caller_state)
        children = [a for a in self.manager.agents if a.parent is caller]
        if not children:
            return "You have no sub-agents."
        target = (target or 'all').strip()
        if target in ('all', '*', ''):
            lines = [f"{a.title}: {'running' if a.busy else 'done'}"
                     for a in children]
            return "Sub-agents:\n" + "\n".join(lines)
        match = next((a for a in children if a.title == target), None)
        if match is None:
            names = ', '.join(a.title for a in children)
            return f"No sub-agent named '{target}'. Yours: {names}."
        if match.busy:
            return f"{match.title}: running (task: {match.task})"
        return (f"{match.title}: done\ntask: {match.task}\n"
                f"{self.final_answer(match)}")

    def do_join(self, caller_state, titles: list) -> str:
        caller = self.agent_for_state(caller_state)
        children = {a.title: a for a in self.manager.agents
                    if a.parent is caller}
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
                results[a.title] = (a.task, self.final_answer(a))
        if remaining:
            self.barriers[caller] = {
                'remaining': remaining, 'results': results}
            waiting = ', '.join(sorted(remaining))
            return (f"Waiting for {waiting} to finish; I'll be resumed"
                    f" automatically with their combined results.")
        self.deliver(caller, "[inbox] join complete",
                     self._format_join(results))
        return "Those sub-agents already finished; resuming with results now."

    def check(self, target: str) -> str:
        st = session.current()
        return self.run_on_loop(lambda: self.do_check(st, target))

    def join(self, targets: str) -> str:
        st = session.current()
        titles = [t for t in targets.replace(',', ' ').split() if t]
        return self.run_on_loop(lambda: self.do_join(st, titles))

    def install_handlers(self) -> None:
        """Wire spawn/check/join so the tool layer routes to this instance."""
        tools.set_spawn_handler(self.spawn)
        tools.set_check_handler(self.check)
        tools.set_join_handler(self.join)

    def clear_handlers(self) -> None:
        tools.set_spawn_handler(None)
        tools.set_check_handler(None)
        tools.set_join_handler(None)
