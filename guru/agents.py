"""Agent and AgentManager — the multi-viewport model for the TUI.

Each Agent owns a viewport: a scrollback buffer plus (later) its own
conversation, adapter+model, and background task. The AgentManager tracks the
list of agents and which one is active (visible/receiving input).

This module is pure state — no UI, no I/O — so it is unit-testable and can be
built up before the full TUI wiring lands.
"""
import itertools
from dataclasses import dataclass, field
from typing import Any

from guru.session import SessionState


@dataclass(eq=False)
class Agent:
    """One agent viewport with its own output buffer and conversation state.

    ``eq=False`` keeps identity-based equality and hashing, so agents can be
    used as dict keys (e.g. the TUI's join barriers) and compared with ``is``.
    """
    id: str
    title: str = "main"
    lines: list = field(default_factory=list)
    status: str = "idle"        # idle | thinking | error
    queue: list = field(default_factory=list)   # pending user messages
    busy: bool = False          # a turn is running in the background

    # Fully independent runtime state (model, conversation, tools, token
    # counts, cancel flag). The TUI binds this via ``session.use(agent.state)``
    # for the duration of the agent's background turn, so turns run in parallel
    # without sharing mutable session globals.
    state: SessionState = field(default_factory=SessionState)
    # Per-agent rich Console writing into this viewport's buffer; bound via
    # ``ui.use_console(agent.console)`` while the turn runs.
    # Assigned a rich Console at runtime (see tui); Any so its dynamic
    # ``.print``/``.file`` surface type-checks without importing rich here.
    console: Any = None
    # Delegation: the agent that spawned this one (None for main and
    # user-created agents), and the task it was spawned to do. Used to deliver
    # results back to the parent's mailbox when the turn finishes.
    parent: object = None
    task: str = ""
    # Ledger bookkeeping for a spawned sub-agent: its running TaskRecord
    # (guru.domain.ledger.TaskRecord; Any keeps this module import-free) and
    # the monotonic time its turn was launched.
    task_rec: Any = None
    started: float = 0.0
    # How the task ended, as recorded on its closing TaskRecord (done,
    # stalled, error, cancelled, fell_back); '' until then.
    outcome: str = ''
    # The worker's sandbox diff block for its report (read on its own
    # thread when its turn ends; '' without a sandbox change).
    report_diff: str = ''
    # Finished and due to leave the tab bar once the user has looked at it
    # and switched away (AgentManager.retire / select).
    retire_pending: bool = False
    viewed: bool = False

    def append(self, text: str) -> None:
        """Append a line (or block) to the scrollback buffer."""
        self.lines.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class AgentManager:
    """The set of agents and the active (visible) one."""

    def __init__(self) -> None:
        self.agents: list = [Agent(id="main", title="main")]
        self.active_index: int = 0
        # Finished sub-agents off the tab bar: the orchestrator still finds
        # them (check/join, retries); their transcripts are in the ledger.
        self.archived: list = []
        # Atomic under the GIL: spawn/plan make children on worker threads.
        self._next = itertools.count(1)

    @property
    def active(self) -> Agent:
        return self.agents[self.active_index]

    def next_title(self) -> str:
        """``agentN`` with N counting up for the whole run: a title is
        never reused after its agent is archived (join barriers key on
        titles)."""
        return f"agent{next(self._next)}"

    def all_agents(self) -> list:
        """Every agent, on the tab bar or archived."""
        return self.agents + self.archived

    def add(self, title: str = '') -> Agent:
        """Create and append a new agent viewport (does not switch to it);
        the title defaults to its ``agentN`` id."""
        ident = self.next_title()
        agent = Agent(id=ident, title=title or ident)
        self.agents.append(agent)
        return agent

    def switch(self, step: int) -> None:
        """Move the active viewport by +1/-1 (wraps)."""
        self.select((self.active_index + step) % len(self.agents))

    def select(self, index: int) -> None:
        """Show tab ``index``; the tab left behind is archived when it
        was due to retire, has been viewed and is idle."""
        leaving = self.active
        self.active_index = index
        self.active.viewed = True
        if (leaving is not self.active and leaving.retire_pending
                and leaving.viewed and not leaving.busy):
            self._archive(leaving)

    def retire(self, agent: Agent, keep: bool) -> None:
        """A sub-agent finished: archive it now, or (``keep`` — it ended
        stalled or in error — or it is on screen) once the user
        has viewed it and switched away. ``main`` never retires."""
        if agent is self.agents[0] or agent not in self.agents:
            return
        if keep or agent is self.active:
            agent.retire_pending = True
            # Seen after it finished: only if it is on screen right now.
            agent.viewed = agent is self.active
            return
        self._archive(agent)

    def _archive(self, agent: Agent) -> None:
        active = self.active
        self.agents.remove(agent)
        self.archived.append(agent)
        if active is agent:
            active = self.agents[min(self.active_index,
                                     len(self.agents) - 1)]
        self.active_index = self.agents.index(active)

    def tabs(self) -> list:
        """Return [(is_active, title), ...] for the tab line."""
        return [
            (i == self.active_index, a.title)
            for i, a in enumerate(self.agents)
        ]
