"""Agent and AgentManager — the multi-viewport model for the TUI.

Each Agent owns a viewport: a scrollback buffer plus (later) its own
conversation, adapter+model, and background task. The AgentManager tracks the
list of agents and which one is active (visible/receiving input).

This module is pure state — no UI, no I/O — so it is unit-testable and can be
built up before the full TUI wiring lands.
"""
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

    @property
    def active(self) -> Agent:
        return self.agents[self.active_index]

    def add(self, title: str) -> Agent:
        """Create and append a new agent viewport (does not switch to it)."""
        agent = Agent(id=f"agent{len(self.agents)}", title=title)
        self.agents.append(agent)
        return agent

    def switch(self, step: int) -> None:
        """Move the active viewport by +1/-1 (wraps)."""
        self.active_index = (self.active_index + step) % len(self.agents)

    def tabs(self) -> list:
        """Return [(is_active, title), ...] for the tab line."""
        return [
            (i == self.active_index, a.title)
            for i, a in enumerate(self.agents)
        ]
