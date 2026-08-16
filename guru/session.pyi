"""Type stub for the session module.

At runtime ``guru/session.py`` replaces itself with a ``_SessionProxy`` that
routes attribute access to the ``SessionState`` bound to the current context.
mypy cannot see through that swap, so this stub declares the module surface —
the routed ``SessionState`` fields plus the module functions — mirroring
``SessionState.__init__``.
"""
from typing import Any, Optional


class SessionState:
    adapter: Any
    model: str
    num_ctx: int
    ctx_ceiling: int
    num_ctx_override: int
    model_size: str
    ctx_used: int
    session_in: int
    session_out: int
    git_branch: Optional[str]
    messages: list
    active_tools: list
    active_tool_names: set
    file_shas: dict
    active_role: Optional[str]
    active_skill: Optional[str]
    cancel_requested: bool
    can_spawn: bool
    def __init__(self) -> None: ...


def current() -> SessionState: ...
def use(state: SessionState) -> Any: ...
def reset(token: Any) -> None: ...


# Module-level attributes routed to the current SessionState by the proxy.
adapter: Any
model: str
num_ctx: int
ctx_ceiling: int
num_ctx_override: int
model_size: str
ctx_used: int
session_in: int
session_out: int
git_branch: Optional[str]
messages: list
active_tools: list
active_tool_names: set
file_shas: dict
active_role: Optional[str]
active_skill: Optional[str]
cancel_requested: bool
can_spawn: bool
