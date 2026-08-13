"""Shared mutable runtime state for a guru session.

Modules read and mutate these attributes via ``from guru import session``.
Using module-level attributes (rather than scattered globals threaded through
``global`` declarations) gives every module one obvious place to find the
current model, adapter, conversation, and context accounting.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:                     # avoid an import cycle at runtime
    from guru.adapters.base import Adapter

# Active provider adapter and its selected model.
adapter: "Optional[Adapter]" = None
model: str = ""

# Context accounting for the active model.
num_ctx: int = 0                      # effective window (adapter-resolved)
ctx_ceiling: int = 0                  # architecture maximum, if known
num_ctx_override: int = 0             # --num-ctx (0 = auto-detect)
model_size: str = "?"                 # e.g. parameter size, for the status bar
ctx_used: int = 0                     # latest measured occupancy
session_in: int = 0                   # cumulative input tokens this session
session_out: int = 0                  # cumulative output tokens this session

# UI state.
git_branch: Optional[str] = None

# Conversation state. messages is the neutral message list (dicts).
messages: list = []
active_tools: list = []
active_tool_names: set = set()
