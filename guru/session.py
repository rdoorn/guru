"""Runtime state for a guru session, routed per context for parallelism.

All runtime state lives on a :class:`SessionState`. Modules read and mutate it
through ``from guru import session`` (``session.model``, ``session.messages``,
…) exactly as before — but those attribute accesses are routed to the
``SessionState`` bound to the *current context* via a ``ContextVar``.

Outside the TUI (the REPL and unit tests) nothing binds a state, so every
access falls through to a single default state and behaviour is unchanged.
The TUI binds each agent's own state for the duration of its background turn
(``session.use(agent.state)``), so several turns can run at once on different
threads without stomping each other's model, conversation, or token counts.
"""
from __future__ import annotations

import contextvars
import sys
import types
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:                     # avoid an import cycle at runtime
    from guru.adapters.base import Adapter


class SessionState:
    """All mutable runtime state for one conversation/agent."""

    def __init__(self) -> None:
        # Active provider adapter and its selected model.
        self.adapter: "Optional[Adapter]" = None
        self.model: str = ""
        # Context accounting for the active model.
        self.num_ctx: int = 0             # effective window (adapter-resolved)
        self.ctx_ceiling: int = 0         # architecture maximum, if known
        self.num_ctx_override: int = 0    # --num-ctx (0 = auto-detect)
        self.model_size: str = "?"        # parameter size, for the status bar
        self.ctx_used: int = 0            # latest measured occupancy
        self.session_in: int = 0          # cumulative input tokens
        self.session_out: int = 0         # cumulative output tokens
        # UI state.
        self.git_branch: Optional[str] = None
        # Conversation state. messages is the neutral message list (dicts).
        self.messages: list = []
        self.active_tools: list = []
        self.active_tool_names: set = set()
        # Cooperative cancellation: adapters check this between rounds.
        self.cancel_requested: bool = False
        # Whether this agent may delegate via the spawn tool (main + user-made
        # agents may; tool-spawned sub-agents may not, to avoid recursion).
        self.can_spawn: bool = False


_default = SessionState()
_current: "contextvars.ContextVar[SessionState]" = contextvars.ContextVar(
    "guru_session", default=_default)


def current() -> SessionState:
    """Return the SessionState bound to the current context."""
    return _current.get()


def use(state: SessionState):
    """Bind ``state`` to the current context; returns a reset token."""
    return _current.set(state)


def reset(token) -> None:
    """Restore the state bound before the matching :func:`use` call."""
    _current.reset(token)


class _SessionProxy(types.ModuleType):
    """Module stand-in whose attributes route to the current SessionState."""

    SessionState = SessionState
    current = staticmethod(current)
    use = staticmethod(use)
    reset = staticmethod(reset)

    def __getattr__(self, name: str):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(_current.get(), name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith('__'):
            super().__setattr__(name, value)
        else:
            setattr(_current.get(), name, value)


# Replace this module with the routing proxy so existing ``session.X`` access
# throughout the codebase transparently targets the current context's state.
sys.modules[__name__] = _SessionProxy(__name__, __doc__)
