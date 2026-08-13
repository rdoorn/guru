"""Agent and AgentManager — the multi-viewport model for the TUI.

Each Agent owns a viewport: a scrollback buffer plus (later) its own
conversation, adapter+model, and background task. The AgentManager tracks the
list of agents and which one is active (visible/receiving input).

This module is pure state — no UI, no I/O — so it is unit-testable and can be
built up before the full TUI wiring lands.
"""
from dataclasses import dataclass, field


@dataclass
class Agent:
    """One agent viewport: an id, a short title, and an output buffer."""
    id: str
    title: str = "main"
    lines: list = field(default_factory=list)
    status: str = "idle"        # idle | thinking | error
    queue: list = field(default_factory=list)   # pending user messages
    busy: bool = False          # a turn is running in the background

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
