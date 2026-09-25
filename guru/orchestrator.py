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

Routing (design doc §2, §4, §5): ``spawn`` carries ``kind``/``complexity``
labels; ``_make_child`` scans the task text, resolves a ``Route`` over the
configured ladders, asks the once-per-run spend question when a remote pick
needs it, and configures the child with the route's adapter/model from the
registry. Without a registry routing is inert (the child keeps the parent's
adapter and model). A remote child that fails without an answer is respawned
once on the best local rung (``retry_of``); the original row is
``fell_back``.

Panel judge (item 6 of ``docs/plans/2026-09-25-top10-remaining.md``): when
the ``panel`` decision point is *active* and a controller spawns a
``review``-kind task without a security reviewer, the judge's
``needs_security`` verdict over the task text adds one ``security-engineer``
worker on the same task (once per parent turn; the row says
``origin = "panel"`` and its ``reason`` starts with ``origin:panel``).
Shadow mode changes nothing (the turn loop already shadows the panel
questions).
"""
import asyncio
import io
import time
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console

from guru import config, log, session, ui
from guru.agents import Agent, AgentManager
from guru.domain import (conversation, decisions, ledger, policy, routing,
                         spend, tools)
from guru.repositories import settings as routing_settings
from guru.repositories.settings import RoutingSettings

_NO_ANSWER = '(no answer produced)'
_INERT_REASON = 'routing:disabled (no registry)'

# Check-poll rule (design decision 10, cost bug: a controller polled
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
# tells the controller to join it.
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
        # (spawned by the controller or added by the panel judge).
        self._security_turns: set = set()
        self.barriers: dict = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None

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
        return next((a for a in self.manager.agents if a.state is st), None)

    def final_answer(self, agent) -> str:
        for m in reversed(agent.state.messages):
            if conversation.msg_role(m) == 'assistant' \
                    and conversation.msg_content(m).strip():
                return conversation.msg_content(m).strip()
        return _NO_ANSWER

    def configure(self, agent, base, can_spawn: bool,
                  role=None, skill=None, controller: bool = False) -> None:
        """Set up ``agent``'s fresh conversation + tools, inheriting the model
        and context from ``base``. Delegation-capable agents get the panel
        hint; a controller (``[routing] controller``; implies ``can_spawn``)
        gets the controller hint and only spawn/check/join/use_skill;
        sub-agents get a role/skill overlay."""
        st = agent.state
        controller = bool(controller and can_spawn)
        st.messages = [
            {'role': 'system', 'content': config.build_system_prompt()}]
        if controller:
            st.messages[0]['content'] += "\n\n" + config.CONTROLLER_HINT
        elif can_spawn:
            st.messages[0]['content'] += "\n\n" + config.DELEGATION_HINT
        st.active_tools, st.active_tool_names = tools.initial_tools(
            can_spawn, controller)
        st.controller = controller
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
                payload = self._format_join(bar['results'])
                if bar.get('synthesis'):
                    payload = bar['synthesis'] + "\n\n" + payload
                self.deliver(parent, "[inbox] join complete", payload)
        else:
            self.deliver(
                parent,
                f"[inbox] result from {child.title}",
                f"[result from {child.title} · task: {child.task}]\n{answer}")

    def on_done(self, agent) -> None:
        status = 'error' if agent.status == 'error' else 'done'
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
        self.invalidate()

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
            env=rec.env, retry_of=rec.task_id, plan=plan)
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
        task_id = agent.task_rec.task_id
        agent.task_rec = None
        self._cleanup_sandbox(task_id)

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
        questions are shadowed and the controller's labels route. With
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
        controller's unless the point is active and the judge overrides
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
                    index: int = 0, env: Optional[dict] = None,
                    kind: str = 'other', complexity: str = 'standard',
                    retry_of: str = '', local_only: bool = False,
                    plan: Optional[_Plan] = None,
                    refusal: Optional[list] = None,
                    origin: str = '') -> Optional[Agent]:
        """Create a configured, routed child agent for ``parent`` with a
        running TaskRecord.

        Not yet appended to the manager or launched; ``index`` offsets the
        title when several children are made in one batch, which also passes
        one shared environment snapshot via ``env``. The route comes from
        ``plan`` (else from ``_plan_child``) and is applied to the child;
        the outcome lands on the TaskRecord, as does ``origin`` (``'panel'``
        for the panel judge's worker). Returns None when the route is
        refused: a ``refused`` task row is written, the reasons are appended
        to ``refusal`` (when given) and no child exists.
        """
        if plan is None:
            plan = self._plan_child(parent, task, kind, complexity,
                                    local_only)
        route, reason = plan.route, plan.reason
        env = ledger.environment() if env is None else env
        if plan.refused:
            assert route is not None
            rec = ledger.new_task(
                task=task, parent=parent.title, role=role, skill=skill,
                kind=plan.kind, complexity=plan.complexity,
                turn_id=parent.state.turn_id, model='', adapter='',
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
        title = f"agent{len(self.manager.agents) + index}"
        child = Agent(id=title, title=title)
        self.configure(child, parent.state, can_spawn=False, role=role,
                       skill=skill)
        self._apply_route(child, route, reason)
        child.task = task
        child.parent = parent
        # Join keys come from the parent, not from whichever session happens
        # to be bound on the calling thread (spawn_panel runs on the loop).
        rec = ledger.new_task(
            task=task, parent=parent.title, role=role, skill=skill,
            kind=plan.kind, complexity=plan.complexity,
            turn_id=parent.state.turn_id, model=child.state.model,
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
        child.state.turn_id = parent.state.turn_id
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
        # worker thread while the loop may be iterating it.
        def _start() -> None:
            for c in children:
                self.manager.agents.append(c)
                self.launch(c)
            self.invalidate()

        assert self.loop is not None
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
        ``needs_security`` over the task text (heuristic: no; a timeout or
        error adds nothing). The worker takes the same task with the
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
            'panel', decisions.security_question(child.task),
            heuristic=False)
        if not needed:
            return None
        task = _SECURITY_TASK.format(task=child.task,
                                     focus=_SECURITY_MEMBER[2])
        plan = self._plan_child(parent, task, 'review', rec.complexity)
        plan.reason.insert(0, PANEL_ORIGIN)
        extra = self._make_child(parent, task, role=SECURITY_ROLE,
                                 skill=_SECURITY_MEMBER[1], index=1,
                                 env=rec.env, plan=plan, origin='panel')
        if extra is not None:
            self._security_turns.add(key)
        return extra

    def _security_seen(self, parent) -> bool:
        """True when a security reviewer already ran for ``parent`` in the
        current turn (covers children the controller spawned before the
        panel point became active)."""
        turn = parent.state.turn_id
        for a in self.manager.agents:
            if (a.parent is parent and a.state.turn_id == turn
                    and self._is_security(a.state.active_role,
                                          a.state.active_skill)):
                self._security_turns.add((parent.title, turn))
                return True
        return False

    def spawn_panel(self, parent, tasks, synthesis: str = '') -> list:
        """Deterministically spawn a fixed panel of sub-agents parented to
        ``parent`` and open a join barrier, so guru itself runs the multi-agent
        path even for a model that would never delegate. ``tasks`` is a list of
        ``(task, role, skill)``; when all finish, their combined findings (with
        an optional ``synthesis`` lead-in) are delivered back to ``parent``.
        Returns the child titles."""
        env = ledger.environment()      # one snapshot for the whole panel
        children: list = []
        for task, role, skill in tasks:
            child = self._make_child(parent, task, role=role, skill=skill,
                                     index=len(children), env=env,
                                     kind='review')
            if child is not None:
                children.append(child)
        titles = [c.title for c in children]
        if not children:
            return titles

        def _start() -> None:
            for c in children:
                self.manager.agents.append(c)
            # Open the barrier BEFORE launching, so a fast child can't report
            # into a not-yet-existing barrier.
            self.barriers[parent] = {
                'remaining': set(titles), 'results': {},
                'synthesis': synthesis}
            for c in children:
                self.launch(c)
            self.invalidate()

        assert self.loop is not None
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
        children = [a for a in self.manager.agents if a.parent is caller]
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
            # End the caller's turn after this tool round (turn loop);
            # the barrier resumes it with the combined results.
            caller_state.turn_waiting = True
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
