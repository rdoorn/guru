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

The main agent is the LEAD (``config.LEAD_HINT``): it keeps the overview,
works itself where that is quicker, spawns workers for parts of the work
and reviews and integrates what they return. In a sandbox project a
worker's copy is kept when it finishes and its diff rides along with its
report; the lead merges it into its own copy (``apply_work``) and submits
the integrated change once through the gate.

Routing (design doc §2, §4, §5): ``spawn`` carries ``kind``/``complexity``
labels; ``_make_child`` scans the task text, resolves a ``Route`` over the
configured ladders, asks the once-per-run spend question when a remote pick
needs it, and configures the child with the route's adapter/model from the
registry. Without a registry routing is inert (the child keeps the parent's
adapter and model). A remote child that fails without an answer is respawned
once on the best local rung (``retry_of``); the original row is
``fell_back``.

Panel judge (item 6 of ``docs/plans/2026-09-25-top10-remaining.md``): when
the ``panel`` decision point is *active* and the lead spawns a
``review``-kind task without a security reviewer, the judge's
``needs_security`` verdict over the user's request and the task text (the
request first, so a lead that drops "security" from the task it
writes still triggers it) adds one ``security-engineer``
worker on the same task (once per parent turn; the row says
``origin = "panel"`` and its ``reason`` starts with ``origin:panel``).
Shadow mode changes nothing (the turn loop already shadows the panel
questions).
"""
import asyncio
import io
import re
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console

from guru import config, log, session, ui
from guru.agents import Agent, AgentManager
from guru.domain import brief as _brief
from guru.domain import (conversation, decisions, ledger, policy, routing,
                         spend, tools)
from guru.domain.brief import Brief
from guru.repositories import briefs
from guru.repositories import settings as routing_settings
from guru.repositories.settings import RoutingSettings

_NO_ANSWER = '(no answer produced)'
_INERT_REASON = 'routing:disabled (no registry)'

# Check-poll rule (design decision 10, cost bug: a lead polled
# `check` ~20 times after spawning). The Nth consecutive `check` that finds
# every checked sub-agent still running ends the turn like a `join`; the
# one before it tells the model to join.
CHECK_POLL_LIMIT = 3
CHECK_JOIN_HINT = (
    "All of them are still running. Do not poll again: call join with"
    " their names to be resumed when they finish.")
CHECK_WAIT_TEXT = (
    "All sub-agents are still running; this turn ends here and you will"
    " be resumed with their results.")
_DEFERRED_REASON = 'confirmation:deferred (loop thread)'
_RETRY_REASON = 'retry:local after remote failure'
# The panel judge's extra worker: role/skill/focus from the /review panel's
# security member, the task row's origin marker, and the reply suffix that
# tells the lead to join it.
PANEL_ORIGIN = 'origin:panel (needs_security)'
SECURITY_ROLE = 'security-engineer'
_SECURITY_MEMBER = next(m for m in config.REVIEW_PANEL
                        if m[0] == SECURITY_ROLE)
_SECURITY_TASK = (
    "{task}\n\nFocus on {focus}. Give concrete findings with file:line and"
    " a suggested fix; be specific.")
_SECURITY_SPAWNED = (
    " guru also spawned {title} ({role}) on the same task because the panel"
    " judge found it needs a security review; join it as well.")


def panel_text(messages: list, task: str) -> str:
    """What the panel judge reads for a spawned ``task``: the parent's
    request (:func:`guru.domain.conversation.request_in` over
    ``messages`` — the human request behind a mailbox delivery, capped)
    and then the task text, blank-line separated; the task alone when the
    request is empty or already the task."""
    request = conversation.request_in(messages)
    if not request or request == task:
        return task
    return request + "\n\n" + task


@dataclass
class _Plan:
    """Everything ``_make_child`` decided before building the agent: the
    normalised labels, the scan result, the resolved route (None when
    routing is inert) and the confirmation in force."""
    kind: str
    complexity: str
    findings: int
    route: Optional[routing.Route]
    confirmation: str
    reason: list = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.route is not None and self.route.refused


def _brief_block(task: str, project: Optional[Brief]) -> str:
    """The project brief slice for ``task`` as a system-context block
    (``brief.slice`` over ``project``: map, test command, the modules and
    symbols the task names), so a worker starts informed instead of
    exploring. Empty without a brief or on any failure (logged) — the
    brief is a saving, never a dependency."""
    if project is None:
        return ''
    try:
        text = _brief.slice(project, task)
    except Exception:                                    # noqa: BLE001
        log.exc('brief slice failed')
        return ''
    block = f"\n\n[project brief]\n{text}" if text.strip() else ''
    return block + _rules_block(project)


def _rules_block(project: Optional[Brief]) -> str:
    """The project's rules file (``brief.rules``) as a system-context
    block for the lead and every worker; empty without one."""
    if project is None:
        return ''
    try:
        text = _brief.rules(Path(project.root))
    except Exception:                                    # noqa: BLE001
        log.exc('project rules read failed')
        return ''
    return (f"\n\n[project rules — follow them]\n{text}"
            if text else '')


def _map_block(project: Optional[Brief]) -> str:
    """The lead's block: the map alone (``brief.render_map``) — the
    overview, not the outlines. Empty without a brief."""
    if project is None:
        return ''
    try:
        text = _brief.render_map(project)
    except Exception:                                    # noqa: BLE001
        log.exc('brief map failed')
        return ''
    block = f"\n\n[project map]\n{text}" if text.strip() else ''
    return block + _rules_block(project)


# A worker's sandbox diff in its report to the lead, capped.
DIFF_REPORT_CHARS = 30000
_TICKS_RE = re.compile(r'`{3,}')


def _sandbox_available() -> bool:
    """Whether the current project has a usable sandbox (lazy import;
    any failure means no)."""
    try:
        from guru.sandbox import verbs
        return bool(verbs.available())
    except Exception:                                    # noqa: BLE001
        log.exc('sandbox availability check failed')
        return False


def _diff_block(child) -> str:
    """The ``[sandbox diff]`` block for ``child``'s report: its copy's diff
    (capped at ``DIFF_REPORT_CHARS``) and how to merge it; empty without a
    sandbox change. Never raises."""
    task_id = getattr(child.state, 'task_id', '')
    if not task_id:
        return ''
    try:
        from guru.sandbox import verbs
        diff = verbs.worker_diff(task_id)
    except Exception:                                    # noqa: BLE001
        log.exc(f'worker diff of {task_id} unavailable')
        return ''
    if not diff.strip():
        return ''
    if len(diff) > DIFF_REPORT_CHARS:
        diff = (diff[:DIFF_REPORT_CHARS]
                + f'\n[... {len(diff) - DIFF_REPORT_CHARS} more chars]')
    # A fence longer than any backtick run in the diff, so file content
    # cannot close it and speak to the lead outside it.
    fence = '`' * max(3, 1 + max((len(m) for m in _TICKS_RE.findall(diff)),
                                 default=0))
    return (f'\n\n[sandbox diff of {child.title} — review it, then'
            f' apply_work("{child.title}") to merge it into your copy]\n'
            f'{fence}diff\n{diff}\n{fence}')


class Orchestrator:
    """Owns the agents, join barriers, and the worker-thread turn loop."""

    def __init__(self, manager=None, registry=None, routing=None) -> None:
        self.manager = manager or AgentManager()
        # AdapterRegistry (guru.repositories.adapters) used to turn a
        # resolved route's adapter name into the adapter object; None until
        # the CLI wires it (headless/tests run without one).
        self.registry = registry
        # RoutingSettings (guru.repositories.settings); None loads the
        # [routing] table lazily on first use.
        self._routing_settings = routing
        self._ladders: Optional[dict] = None
        # (parent title, turn_id) pairs that already have a security worker
        # (spawned by the lead or added by the panel judge).
        self._security_turns: set = set()
        self.barriers: dict = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        # Children ``spawn``/``spawn_panel`` made but whose registration on
        # the loop thread (``_start``: append to the agent list, launch) has
        # not run yet, by parent. Filled synchronously in the spawning
        # thread, drained by ``_start``, so a ``join``/``check`` in the same
        # tool round as the ``spawn`` (headless front-ends run it inline on
        # the worker thread) already sees the child as a running sub-agent
        # instead of "None of those are your sub-agents".
        self._pending: dict = {}
        self._pending_lock = threading.Lock()
        # The project brief by (root, HEAD): one build or load per HEAD,
        # not one per child (a five-worker plan reads the ~400 KB JSON once;
        # the lead's map comes from the same object).
        self._briefs: dict = {}

    def project_brief(self) -> Optional[Brief]:
        """The brief of the project in the current working directory at
        its checked-out HEAD (``brief.current`` over the JSON store),
        cached on this orchestrator per (root, HEAD); None when it cannot
        be built (logged) — the brief is a saving, never a dependency."""
        try:
            root = Path.cwd().resolve()
            head = _brief.head_sha(root)
            key = (str(root), head)
            if key not in self._briefs:
                self._briefs[key] = _brief.current(root, briefs, head=head)
            return self._briefs[key]
        except Exception:                                # noqa: BLE001
            log.exc('project brief unavailable')
            return None

    # --- routing -------------------------------------------------------------

    def _routing(self) -> RoutingSettings:
        """The RoutingSettings in force; an invalid table logs a warning
        and falls back to the defaults."""
        if self._routing_settings is None:
            try:
                self._routing_settings = routing_settings.load_routing()
            except ValueError as e:
                log.warning('routing: %s; using defaults', e)
                self._routing_settings = routing_settings.RoutingSettings()
        return self._routing_settings

    def ladders(self) -> dict:
        """Domain ladders for the registry (built once); ``{}`` without a
        registry."""
        if self._ladders is None:
            if self.registry is None:
                self._ladders = {}
            else:
                self._ladders = routing_settings.ladders_from_settings(
                    self._routing(), self.registry)
        return self._ladders

    def set_routing(self, settings: RoutingSettings) -> None:
        """Replace the RoutingSettings in force (``/routing on|off``) and
        rebuild the ladders on next use."""
        self._routing_settings = settings
        self._ladders = None

    def _local_main(self, parent) -> Optional[routing.Rung]:
        """The parent's adapter/model as a fallback rung, when the registry
        knows the adapter."""
        name = getattr(parent.state.adapter, 'name', '')
        if self.registry is None or not name or not parent.state.model:
            return None
        try:
            remote = self.registry.is_remote(name)
        except KeyError:
            return None
        return routing.Rung(name, parent.state.model, 'hard', remote=remote,
                            default=True)

    def _resolve(self, parent, kind: str, complexity: str, findings: int,
                 local_only: bool) -> tuple[Optional[routing.Route], str]:
        """Resolve a route for a child of ``parent``.

        Returns ``(route, confirmation)``; ``route`` is None when routing
        is inert (no registry). A remote pick that needs the spend
        confirmation asks once (``guru.domain.spend``) and re-resolves with
        the answer — unless this runs on the event-loop thread, where an
        interactive asker cannot block: the pick is then resolved as
        declined for this task only (the run's answer stays pending) and
        the deferral is recorded. ``local_only`` forces ``local-only`` mode
        (the retry).
        """
        if self.registry is None:
            return None, ''
        cfg = self._routing()
        mode = 'local-only' if local_only else cfg.mode
        local_main = self._local_main(parent)

        def _run(confirmation: str) -> routing.Route:
            return routing.resolve(
                kind, complexity, self.ladders(), mode=mode,
                scan_findings=findings, confirmation=confirmation,
                complexity_router=cfg.complexity_router,
                type_router=cfg.type_router, local_main=local_main)
        conf = spend.status(cfg.spend_confirm)
        route = _run(conf)
        if route.needs_confirmation:
            if spend.on_event_loop():
                log.warning('spend confirmation needed on the loop thread; '
                            'deferring (this task runs as if declined)')
                route = _run('declined')
                route.reason.append(_DEFERRED_REASON)
            else:
                conf = spend.confirmation(cfg.spend_confirm)
                route = _run(conf)
        if local_only:
            route.reason.insert(0, _RETRY_REASON)
        return route, conf

    def preconfirm_spend(self) -> None:
        """Ask the once-per-run spend question now if a later remote pick
        would need it (``ask`` mode, still pending, a remote rung in some
        ladder). Blocking: call it off the loop thread (the TUI runs it in
        an executor before ``spawn_panel``)."""
        cfg = self._routing()
        if self.registry is None or cfg.spend_confirm != 'ask' \
                or spend.status('ask') != 'pending':
            return
        if any(r.remote for ladder in self.ladders().values()
               for r in ladder.rungs):
            spend.confirmation('ask')

    def _apply_route(self, child, route: Optional[routing.Route],
                     reason: list) -> None:
        """Point ``child`` at the route's adapter/model; keeps the parent's
        (already configured) when the registry does not know the adapter."""
        if route is None:
            return
        assert self.registry is not None
        adapter = self.registry.get(route.adapter)
        if adapter is None:
            reason.append(
                f'adapter:{route.adapter} unknown to registry; keeping '
                f'parent adapter/model')
            return
        child.state.adapter = adapter
        child.state.model = route.model

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
        return next((a for a in self.manager.all_agents() if a.state is st),
                    None)

    def _register_pending(self, parent, children: list) -> None:
        """Record ``children`` as ``parent``'s sub-agents ahead of their
        registration on the loop thread; each counts as running (``busy``)
        from now on — ``launch`` sets the flag again when it runs."""
        for c in children:
            c.busy = True
        with self._pending_lock:
            self._pending.setdefault(parent, []).extend(children)

    def _drain_pending(self, parent, children: list) -> None:
        """Forget ``children`` once ``_start`` has appended them."""
        with self._pending_lock:
            left = [c for c in self._pending.get(parent, ())
                    if not any(c is done for done in children)]
            if left:
                self._pending[parent] = left
            else:
                self._pending.pop(parent, None)

    def children_of(self, parent) -> list:
        """``parent``'s sub-agents: the registered ones (agent-list order)
        followed by the pending ones not yet on the list."""
        out = [a for a in self.manager.all_agents() if a.parent is parent]
        with self._pending_lock:
            pending = list(self._pending.get(parent, ()))
        out.extend(c for c in pending if not any(c is a for a in out))
        return out

    def final_answer(self, agent) -> str:
        for m in reversed(agent.state.messages):
            if conversation.msg_role(m) == 'assistant' \
                    and conversation.msg_content(m).strip():
                return conversation.msg_content(m).strip()
        return _NO_ANSWER

    def configure(self, agent, base, can_spawn: bool,
                  role=None, skill=None, kind: str = '') -> None:
        """Set up ``agent``'s fresh conversation + tools, inheriting the model
        and context from ``base``. A delegation-capable agent is a lead: it
        gets the lead hint (plus the sandbox part in a sandbox project) and
        the project map; a sub-agent gets the worker hint and a role/skill
        overlay."""
        st = agent.state
        st.messages = [
            {'role': 'system', 'content': config.build_system_prompt()}]
        if can_spawn:
            hint = config.LEAD_HINT
            if _sandbox_available():
                hint += config.LEAD_SANDBOX_HINT
            st.messages[0]['content'] += "\n\n" + hint
            st.messages[0]['content'] += _map_block(self.project_brief())
        else:
            st.messages[0]['content'] += "\n\n" + config.WORKER_HINT
        st.active_tools, st.active_tool_names = tools.initial_tools(
            can_spawn, kind=kind or None)
        st.task_kind = kind or ''      # toolpolicy.for_kind at execute time
        st.active_role = role or None
        st.active_skill = skill or None
        st.model = base.model
        st.adapter = base.adapter
        st.num_ctx = base.num_ctx
        st.ctx_ceiling = base.ctx_ceiling
        st.model_size = base.model_size
        st.git_branch = getattr(base, 'git_branch', None)
        st.can_spawn = can_spawn
        st.agent_id = agent.id
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
            agent.status = 'error'       # on_done reads it for the task row
            self.on_worker_error(agent, e)
        finally:
            # The worker's sandbox diff for its report, read here on the
            # worker thread (git) rather than on the event loop.
            agent.report_diff = _diff_block(agent) if agent.parent else ''
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
        agent.started = time.monotonic()
        assert self.loop is not None
        self.loop.run_in_executor(None, self.work, agent)

    def _launch_all(self, children: list) -> None:
        """``launch`` each child; one whose launch raises is left on the
        list idle in ``error`` (logged through ``on_worker_error``) so
        the parent's ``join`` resolves with its (empty) answer instead of
        waiting forever on a phantom, and the others still start."""
        for c in children:
            try:
                self.launch(c)
            except Exception as exc:                     # noqa: BLE001
                c.busy = False
                c.status = 'error'
                self.on_worker_error(c, exc)

    def submit(self, agent, text: str) -> None:
        """Queue a user message for ``agent`` and start it if idle. A new
        request from the user drops the sandbox copies its finished workers
        left unapplied."""
        self.notice(agent, f"> {text}")
        if not agent.busy and not agent.queue:
            # Only when the agent has nothing left to read: a queued
            # report may still name a copy to apply.
            done = [c.state.task_id for c in self.children_of(agent)
                    if not c.busy and c.state.task_id]
            if done and self.loop is not None:
                self.loop.run_in_executor(None, self._cleanup_many, done)
            else:
                self._cleanup_many(done)
        agent.queue.append(text)
        if not agent.busy:
            self.launch(agent)

    # --- mailbox / barriers --------------------------------------------------

    @staticmethod
    def _outcome_tag(child) -> str:
        """' · stalled' (or error, ...) for a child whose task
        did not simply finish; '' for done — the header stays as it was."""
        outcome = getattr(child, 'outcome', '')
        return f' · {outcome}' if outcome and outcome != 'done' else ''

    def _format_join(self, results: dict) -> str:
        parts = ["[joined results]"]
        for tid, (task, ans, tag) in results.items():
            parts.append(f"\n— {tid}{tag} · task: {task}\n{ans}")
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
        answer = self.final_answer(child) + getattr(child, 'report_diff',
                                                    '')
        bar = self.barriers.get(parent)
        if bar is not None and child.title in bar['remaining']:
            bar['remaining'].discard(child.title)
            bar['results'][child.title] = (child.task, answer,
                                           self._outcome_tag(child))
            if not bar['remaining']:
                del self.barriers[parent]
                payload = self._format_join(bar['results'])
                if bar.get('synthesis'):
                    payload = bar['synthesis'] + "\n\n" + payload
                self.deliver(parent, "[inbox] join complete", payload)
        else:
            self.deliver(
                parent,
                f"[inbox] result from {child.title}",
                f"[result from {child.title}{self._outcome_tag(child)}"
                f" · task: {child.task}]\n{answer}")

    def on_done(self, agent) -> None:
        if agent.status == 'error':
            status = 'error'
        elif agent.state.stalled:
            status = 'stalled'       # the stall monitor ended it (a handoff)
        else:
            status = 'done'
        agent.busy = False
        agent.status = 'idle'
        if agent.queue:
            self.launch(agent)
            self.invalidate()
            return
        if agent.parent is not None:
            plan = self._retry_plan(agent)
            rec = agent.task_rec
            # The original's closing row goes first, then the retry's.
            can_retry = plan is not None and not plan.refused
            self._finish_task(agent, 'fell_back' if can_retry else status)
            retry = (self._retry_child(agent, rec, plan)
                     if plan is not None else None)
            if retry is not None:
                self._start_retry(agent, retry)
            else:
                self.report(agent)
            self.retire(agent)
        self.invalidate()

    def retire(self, agent) -> None:
        """A sub-agent's task is closed and reported. No-op here: the
        bench and the eval runner keep every agent for their metrics; the
        TUI takes finished agents off the tab bar."""

    def _should_retry(self, agent) -> bool:
        """Design §5: a remote child that hit a provider error and ended
        without an answer is respawned locally once — never a retry of a
        retry, never after a cancel, and only when routing is active."""
        rec = agent.task_rec
        st = agent.state
        return (rec is not None and not rec.retry_of
                and self.registry is not None
                and not st.cancel_requested
                and bool(st.struggle.get('provider_errors'))
                and self.final_answer(agent) == _NO_ANSWER
                and bool(getattr(st.adapter, 'remote', False)))

    def _retry_plan(self, agent) -> Optional[_Plan]:
        """The local-only plan for retrying a failed remote ``agent``, or
        None when no retry applies. Resolving before the original's row is
        closed lets ``on_done`` write ``fell_back`` only when a local rung
        exists."""
        if not self._should_retry(agent):
            return None
        rec = agent.task_rec
        return self._plan_child(agent.parent, agent.task, rec.kind,
                                rec.complexity, local_only=True)

    def _retry_child(self, agent, rec, plan: _Plan) -> Optional[Agent]:
        """Build the retry of ``agent`` (whose closed TaskRecord is
        ``rec``) from ``plan``; a refused plan writes its ``refused`` row
        and yields None."""
        child = self._make_child(
            agent.parent, agent.task, role=rec.role, skill=rec.skill,
            env=rec.env, retry_of=rec.task_id, plan=plan,
            request=(rec.turn_id, rec.topic_id))
        if child is None:
            log.warning('routing: no local rung for the retry of task %s',
                        rec.task_id)
        return child

    def _start_retry(self, failed, child) -> None:
        """Launch ``child`` in place of ``failed`` (on the loop thread): it
        takes the failed child's seat in an open join barrier."""
        bar = self.barriers.get(failed.parent)
        if bar is not None and failed.title in bar['remaining']:
            bar['remaining'].discard(failed.title)
            bar['remaining'].add(child.title)
        self.notice(failed, f"[{failed.title}] remote failure; retrying as "
                            f"{child.title} on a local model")
        self.manager.agents.append(child)
        self.launch(child)

    def _finish_task(self, agent, status: str) -> None:
        """Close a sub-agent's TaskRecord exactly once.

        The row carries the child's session accumulators (calls, cost,
        struggle) and its transcript path. A task that produced no answer
        after a provider error is an ``error``, not ``done`` (or
        ``fell_back`` when the caller respawns it locally).
        """
        if agent.parent is not None:
            # Whatever happened to its registration, a finished child is
            # not pending any more.
            self._drain_pending(agent.parent, [agent])
        if agent.task_rec is None:
            return
        st = agent.state
        answer = self.final_answer(agent)
        if answer == _NO_ANSWER:
            answer = ''
        if st.cancel_requested:
            status = 'cancelled'
        elif not answer and st.struggle.get('provider_errors') \
                and status != 'fell_back':
            status = 'error'
        transcript = ledger.save_transcript(
            agent.task_rec.task_id,
            [conversation.transcript_record(m) for m in st.messages])
        agent.outcome = status
        ledger.finish_task(
            agent.task_rec, status=status,
            seconds=time.monotonic() - agent.started,
            answer_len=len(answer),
            calls=st.call_count, tokens_in=st.session_in,
            tokens_out=st.session_out,
            cost_usd=st.cost_usd if st.cost_known else None,
            cost_known=st.cost_known, struggle=dict(st.struggle),
            transcript_path=transcript, tools_used=[
                m.get('tool_name') for m in st.messages
                if isinstance(m, dict) and m.get('role') == 'tool'])
        agent.task_rec = None
        # The sandbox copy stays until the lead applies it (apply_work) or
        # the user's next request (submit); a failed task's goes now.
        if status not in ('done', 'stalled'):
            self._cleanup_sandbox(st.task_id)

    def _cleanup_many(self, task_ids: list) -> None:
        for task_id in task_ids:
            self._cleanup_sandbox(task_id)

    @staticmethod
    def cleanup_copies() -> None:
        """Remove every sandbox working copy (exit). Never raises."""
        try:
            from guru.sandbox import verbs
            verbs.cleanup_all()
        except Exception:                            # noqa: BLE001
            log.exc('sandbox copy cleanup at exit failed')

    @staticmethod
    def _cleanup_sandbox(task_id: str) -> None:
        """Remove the sandbox working copies a finished task left behind
        (``guru.sandbox.verbs.cleanup_task``); imported lazily so the
        orchestrator does not load the runtime unless a task used it.
        Never raises."""
        try:
            from guru.sandbox import verbs
            verbs.cleanup_task(task_id)
        except Exception:                            # noqa: BLE001
            log.exc(f'sandbox copy cleanup failed for task {task_id}')

    # --- delegation handlers (installed via tools.set_*_handler) -------------

    def _plan_child(self, parent, task: str, kind: str, complexity: str,
                    local_only: bool = False) -> _Plan:
        """Decide how a child for ``task`` would run: normalise the labels,
        let the ``labels`` judge check them, scan the task text (findings
        force a local rung), resolve the route and gather the reasons.
        Pure apart from the spend question and the judge, which sees the
        normalised labels as its heuristics (skipped for the local retry:
        same task, same labels).

        Judge: in shadow (or with ``labels`` not active) both label
        questions are shadowed and the lead's labels route. With
        ``labels`` active the complexity question is a synchronous,
        margin-gated tie-breaker (:func:`decisions.decide_choice`): the
        judge's tier routes when it beats the runner-up by
        ``config.DECISIONS_LABELS_MARGIN``, recorded as
        ``labels:judge override standard->hard (0.57 vs 0.33)``; the kind
        question stays shadow (kind routes nothing while ``type_router``
        is off).
        """
        kind, complexity = routing.normalise_labels(kind, complexity)
        reason: list = []
        if not local_only:
            complexity = self._judged_complexity(task, kind, complexity,
                                                 reason)
        cfg = self._routing()
        findings = len(policy.scan(task)) if cfg.secret_scan else 0
        if findings:
            reason.append(f'scan:{findings} finding(s) in task text')
        route, confirmation = self._resolve(
            parent, kind, complexity, findings, local_only)
        if route is None:
            reason.append(_INERT_REASON)
        else:
            reason.extend(route.reason)
        return _Plan(kind, complexity, findings, route, confirmation, reason)

    @staticmethod
    def _judged_complexity(task: str, kind: str, complexity: str,
                           reason: list) -> str:
        """The complexity to route on after the ``labels`` judge: the
        lead's unless the point is active and the judge overrides
        it by margin (then noted in ``reason``). See :meth:`_plan_child`.
        """
        complexity_q, kind_q = decisions.label_questions(task)
        if not decisions.active('labels'):
            decisions.shadow('labels', [complexity_q, kind_q],
                             heuristics=[complexity, kind])
            return complexity
        verdict = decisions.decide_choice(
            'labels', complexity_q, heuristic=complexity,
            margin=config.DECISIONS_LABELS_MARGIN)
        decisions.shadow('labels', [kind_q], heuristic=kind)
        if verdict.overrode and isinstance(verdict.chosen, str):
            reason.append('labels:' + verdict.describe())
            return verdict.chosen
        return complexity

    def _make_child(self, parent, task: str, role: str = '', skill: str = '',
                    env: Optional[dict] = None,
                    kind: str = 'other', complexity: str = 'standard',
                    retry_of: str = '', local_only: bool = False,
                    plan: Optional[_Plan] = None,
                    refusal: Optional[list] = None,
                    origin: str = '',
                    request: Optional[tuple] = None) -> Optional[Agent]:
        """Create a configured, routed child agent for ``parent`` with a
        running TaskRecord.

        Not yet appended to the manager or launched; its title comes from
        ``AgentManager.next_title`` (never reused). A batch passes one
        shared environment snapshot via ``env``. The route comes from
        ``plan`` (else from ``_plan_child``) and is applied to the child;
        the outcome lands on the TaskRecord, as does ``origin`` (``'panel'``
        for the panel judge's worker). Returns None when the route is
        refused: a ``refused`` task row is written, the reasons are appended
        to ``refusal`` (when given) and no child exists.
        """
        if plan is None:
            plan = self._plan_child(parent, task, kind, complexity,
                                    local_only)
        # The request the task belongs to: the parent's current one, or
        # the original task's (``request``) for a retry, which may run after
        # the parent moved on to a new request.
        turn_id, topic_id = request or (parent.state.turn_id,
                                        parent.state.topic_id)
        route, reason = plan.route, plan.reason
        env = ledger.environment() if env is None else env
        if plan.refused:
            assert route is not None
            rec = ledger.new_task(
                task=task, parent=parent.title, role=role, skill=skill,
                kind=plan.kind, complexity=plan.complexity,
                turn_id=turn_id, topic_id=topic_id, model='', adapter='',
                env=env, route=route.as_dict(), reason=reason,
                findings=plan.findings, confirmation=plan.confirmation,
                retry_of=retry_of, origin=origin)
            rec.status = 'refused'
            ledger.record_task(rec)
            log.warning('routing refused task %s: %s', rec.task_id,
                        '; '.join(reason))
            if refusal is not None:
                refusal.extend(reason)
            return None
        title = self.manager.next_title()
        child = Agent(id=title, title=title)
        # The task's kind reaches the tool layer: toolpolicy.for_kind hides
        # the write tools for a review task (configure sets task_kind on
        # the child's own state and filters its initial tool list).
        self.configure(child, parent.state, can_spawn=False, role=role,
                       skill=skill, kind=kind)
        block = _brief_block(task, self.project_brief())
        if block:
            child.state.messages[0]['content'] += block
        self._apply_route(child, route, reason)
        child.task = task
        child.parent = parent
        # Join keys come from the parent, not from whichever session happens
        # to be bound on the calling thread (spawn_panel runs on the loop).
        rec = ledger.new_task(
            task=task, parent=parent.title, role=role, skill=skill,
            kind=plan.kind, complexity=plan.complexity,
            turn_id=turn_id, topic_id=topic_id, model=child.state.model,
            adapter=getattr(child.state.adapter, 'name', ''),
            env=env,
            prompt_sha=ledger.prompt_hash(child.state.messages[0]['content']),
            tools_active=sorted(child.state.active_tool_names),
            route=route.as_dict() if route is not None else None,
            reason=reason, findings=plan.findings,
            confirmation=plan.confirmation, retry_of=retry_of, origin=origin)
        child.task_rec = rec
        child.state.task_id = rec.task_id
        child.state.task_text = task
        child.state.turn_id = turn_id
        child.state.topic_id = topic_id                # same request
        ledger.record_task(rec)
        where = (f" · {rec.adapter}|{rec.model}" if route is not None
                 else '')
        self.notice(child, f"[{title}] spawned{where} · task: {task}")
        self.notice(child, f"> {task}")
        child.queue.append(task)
        return child

    def spawn(self, task: str, role: str = '', skill: str = '',
              kind: str = 'other', complexity: str = 'standard') -> str:
        base = session.current()
        parent = self.agent_for_state(base)
        refusal: list = []
        child = self._make_child(parent, task, role=role, skill=skill,
                                 kind=kind, complexity=complexity,
                                 refusal=refusal)
        if child is None:
            return (
                "Could not spawn a sub-agent for this task: no model is"
                f" allowed to run it ({'; '.join(refusal)}). Handle it"
                " yourself, or ask the user to adjust the routing settings.")
        title = child.title
        extra = self._panel_security_worker(parent, child)
        children = [child] if extra is None else [child, extra]

        # Append to the agent list on the loop thread — never mutate it from a
        # worker thread while the loop may be iterating it. Until then the
        # children are pending (a join/check in this tool round sees them).
        def _start() -> None:
            try:
                for c in children:
                    self.manager.agents.append(c)
                self._launch_all(children)
            finally:
                self._drain_pending(parent, children)
                self.invalidate()

        assert self.loop is not None
        self._register_pending(parent, children)
        self.loop.call_soon_threadsafe(_start)
        reply = (
            f"Spawned {title} to work on this task in parallel. Its result"
            f" will be delivered back to you automatically when it finishes.")
        if extra is not None:
            reply += _SECURITY_SPAWNED.format(title=extra.title,
                                              role=SECURITY_ROLE)
        return reply

    @staticmethod
    def _is_security(role: Optional[str], skill: Optional[str]) -> bool:
        return role == SECURITY_ROLE or 'security' in (skill or '')

    def _panel_security_worker(self, parent, child) -> Optional[Agent]:
        """The panel judge's extra worker for ``child`` (a spawned task), or
        None.

        Only with the ``panel`` point active, for a ``review``-kind task,
        when neither ``child`` nor any earlier child of ``parent`` in this
        turn is a security reviewer, and when the judge answers yes to
        ``needs_security`` over the user's request and the task text
        (:func:`panel_text`; heuristic: no; a timeout or error adds
        nothing). The worker takes the same task with the
        security focus of the /review panel, kind ``review`` at the
        child's complexity; its row carries ``origin = 'panel'`` and a
        ``reason`` opening with ``PANEL_ORIGIN``. At most one per parent
        turn.
        """
        rec = child.task_rec
        key = (parent.title, parent.state.turn_id)
        if self._is_security(child.state.active_role,
                             child.state.active_skill):
            self._security_turns.add(key)
            return None
        if (rec is None or rec.kind != 'review'
                or not decisions.active('panel')
                or key in self._security_turns
                or self._security_seen(parent)):
            return None
        needed = decisions.decide(
            'panel', decisions.security_question(
                panel_text(parent.state.messages, child.task)),
            heuristic=False)
        if not needed:
            return None
        task = _SECURITY_TASK.format(task=child.task,
                                     focus=_SECURITY_MEMBER[2])
        plan = self._plan_child(parent, task, 'review', rec.complexity)
        plan.reason.insert(0, PANEL_ORIGIN)
        extra = self._make_child(parent, task, role=SECURITY_ROLE,
                                 skill=_SECURITY_MEMBER[1],
                                 env=rec.env, plan=plan, origin='panel')
        if extra is not None:
            self._security_turns.add(key)
        return extra

    def _security_seen(self, parent) -> bool:
        """True when a security reviewer already ran for ``parent`` in the
        current turn (covers children the lead spawned before the
        panel point became active)."""
        turn = parent.state.turn_id
        for a in self.manager.all_agents():
            if (a.parent is parent and a.state.turn_id == turn
                    and self._is_security(a.state.active_role,
                                          a.state.active_skill)):
                self._security_turns.add((parent.title, turn))
                return True
        return False

    def begin_request(self, agent, request: str) -> None:
        """Open a new request on ``agent`` outside its turn loop (the
        ``/review`` panel): a fresh turn id and the request's topic, so the
        panel and its synthesis are not filed under the previous request.
        """
        from guru.domain import usage
        agent.state.turn_id = ledger.new_turn_id()
        token = session.use(agent.state)
        try:
            usage.begin_topic(request)
        finally:
            session.reset(token)

    def spawn_panel(self, parent, tasks, synthesis: str = '',
                    refusal: Optional[list] = None) -> list:
        """Deterministically spawn a group of sub-agents parented to
        ``parent`` and open a join barrier, so guru itself runs the
        multi-agent path of the ``/review`` panel. ``tasks`` items are
        ``(task, role, skill)`` triples (kind ``review``, default
        complexity). When all finish, their
        combined findings (with an optional ``synthesis`` lead-in) are
        delivered back to ``parent``. A task routing refuses is skipped,
        its reasons appended to ``refusal`` when given. Returns the child
        titles."""
        env = ledger.environment()      # one snapshot for the whole panel
        children: list = []
        for task, role, skill in tasks:
            child = self._make_child(parent, task, role=role, skill=skill,
                                     env=env, kind='review', refusal=refusal)
            if child is not None:
                children.append(child)
        titles = [c.title for c in children]
        if not children:
            return titles

        def _start() -> None:
            try:
                for c in children:
                    self.manager.agents.append(c)
                # Open the barrier BEFORE launching, so a fast child can't
                # report into a not-yet-existing barrier.
                self.barriers[parent] = {
                    'remaining': set(titles), 'results': {},
                    'synthesis': synthesis}
                self._launch_all(children)
            finally:
                self._drain_pending(parent, children)
                self.invalidate()

        assert self.loop is not None
        self._register_pending(parent, children)
        self.loop.call_soon_threadsafe(_start)
        return titles

    def do_check(self, caller_state, target: str) -> str:
        """Status of the caller's sub-agents (``target`` a title or 'all').

        Polling is counted: when every checked sub-agent is still running
        the caller's ``check_polls`` grows; the second such poll in a row
        tells the model to ``join``, the third ends the turn like a join
        (``turn_waiting``, barrier over the running children, see
        ``CHECK_POLL_LIMIT``). Any done sub-agent resets the count.
        """
        caller = self.agent_for_state(caller_state)
        children = self.children_of(caller)
        if not children:
            return "You have no sub-agents."
        target = (target or 'all').strip()
        if target in ('all', '*', ''):
            lines = [f"{a.title}: {'running' if a.busy else 'done'}"
                     for a in children]
            text = "Sub-agents:\n" + "\n".join(lines)
            return self._count_poll(caller, caller_state, children, text)
        match = next((a for a in children if a.title == target), None)
        if match is None:
            names = ', '.join(a.title for a in children)
            return f"No sub-agent named '{target}'. Yours: {names}."
        if match.busy:
            text = f"{match.title}: running (task: {match.task})"
            return self._count_poll(caller, caller_state, [match], text)
        caller_state.check_polls = 0
        return (f"{match.title}: done\ntask: {match.task}\n"
                f"{self.final_answer(match)}")

    def _count_poll(self, caller, caller_state, checked: list,
                    text: str) -> str:
        """Apply the check-poll rule to one ``check`` over ``checked``."""
        running = [a for a in checked if a.busy]
        if len(running) < len(checked):
            caller_state.check_polls = 0
            return text
        caller_state.check_polls += 1
        if caller_state.check_polls >= CHECK_POLL_LIMIT:
            self._open_barrier(caller, {a.title for a in running})
            caller_state.turn_waiting = True
            return CHECK_WAIT_TEXT
        if caller_state.check_polls >= CHECK_POLL_LIMIT - 1:
            return text + "\n" + CHECK_JOIN_HINT
        return text

    def _open_barrier(self, caller, titles: set) -> None:
        """Wait for ``titles`` on ``caller``'s join barrier, merging into an
        open one so a single joined payload delivers everything."""
        bar = self.barriers.get(caller)
        if bar is None:
            self.barriers[caller] = {'remaining': set(titles), 'results': {}}
        else:
            bar['remaining'].update(titles)

    def do_join(self, caller_state, titles: list) -> str:
        caller = self.agent_for_state(caller_state)
        children = {a.title: a for a in self.children_of(caller)}
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
                results[a.title] = (a.task, self.final_answer(a),
                                    self._outcome_tag(a))
        if remaining:
            self.barriers[caller] = {
                'remaining': remaining, 'results': results}
            # End the caller's turn after this tool round (turn loop);
            # the barrier resumes it with the combined results.
            caller_state.turn_waiting = True
            waiting = ', '.join(sorted(remaining))
            return (f"Waiting for {waiting} to finish; I'll be resumed"
                    f" automatically with their combined results.")
        self.deliver(caller, "[inbox] join complete",
                     self._format_join(results))
        return "Those sub-agents already finished; resuming with results now."

    def do_apply_work(self, caller_state, worker: str) -> str:
        """The lead's ``apply_work``: merge finished sub-agent ``worker``'s
        sandbox diff into the caller's own copy
        (``guru.sandbox.verbs.apply_work``)."""
        child = self._finished_child(caller_state, worker)
        if isinstance(child, str):
            return child
        from guru.sandbox import verbs
        return verbs.apply_work(child.state.task_id)

    def _finished_child(self, caller_state, worker: str):
        """The caller's finished sub-agent named ``worker``, or the text
        that says why there is none (read on the loop thread)."""
        caller = self.agent_for_state(caller_state)
        children = {a.title: a for a in self.children_of(caller)}
        child = children.get(str(worker or '').strip())
        if child is None:
            have = ', '.join(children) or 'none'
            return f"No sub-agent named '{worker}'. Yours: {have}."
        if child.busy:
            return f"{child.title} is still running; join it first."
        return child

    def apply_work(self, worker: str) -> str:
        st = session.current()
        child = self.run_on_loop(lambda: self._finished_child(st, worker))
        if isinstance(child, str):
            return child
        from guru.sandbox import verbs
        return verbs.apply_work(child.state.task_id)

    def check(self, target: str) -> str:
        st = session.current()
        return self.run_on_loop(lambda: self.do_check(st, target))

    def join(self, targets: str) -> str:
        st = session.current()
        titles = [t for t in targets.replace(',', ' ').split() if t]
        return self.run_on_loop(lambda: self.do_join(st, titles))

    def install_handlers(self) -> None:
        """Wire spawn/check/join/apply_work so the tool layer routes to
        this instance."""
        tools.set_spawn_handler(self.spawn)
        tools.set_check_handler(self.check)
        tools.set_join_handler(self.join)
        tools.set_apply_work_handler(self.apply_work)

    def clear_handlers(self) -> None:
        tools.set_spawn_handler(None)
        tools.set_check_handler(None)
        tools.set_join_handler(None)
        tools.set_apply_work_handler(None)
