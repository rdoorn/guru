"""The project tool policy seam (``.guru/tools.toml``): the ``ToolsPolicy``
entity, the installed instance and the ``is_enabled`` predicate.

Split out of ``guru.domain.tools`` so the audited verbs (``quality``,
``gitread``) can read the runner and limits at import time without a
circular import — ``tools`` registers those verbs, so it must import them,
not the other way round. ``tools`` re-exports every name here, so
``tools.set_policy`` / ``tools.is_enabled`` / ``tools.active_policy`` keep
working for the CLI and the tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = ['ALWAYS_ON_TOOLS', 'ToolsPolicy', 'active_policy',
           'is_enabled', 'set_policy']

# Tools every agent has regardless of the project tool policy: discovery,
# method selection and the delegation mailbox are not registry tools.
ALWAYS_ON_TOOLS = frozenset(
    ('search_tools', 'use_skill', 'spawn', 'check', 'join'))


@dataclass
class ToolsPolicy:
    """A project's tool policy (loaded by
    ``guru.repositories.settings.load_tools_policy``).

    ``disabled`` always wins; a non-empty ``enabled`` set is an allowlist
    for registry tools. ``test_runner`` is ``pytest`` or ``unittest``;
    ``limits`` holds ``[tools.limits]`` overrides for ``procs.Limits``.
    The default (no file) enables everything.
    """
    enabled: set = field(default_factory=set)
    disabled: set = field(default_factory=set)
    test_runner: str = 'pytest'
    limits: dict = field(default_factory=dict)


_policy = ToolsPolicy()


def set_policy(pol: Optional[ToolsPolicy]) -> None:
    """Install the project tool policy (the CLI at startup); None resets
    to the default that enables everything."""
    global _policy
    _policy = pol if pol is not None else ToolsPolicy()


def active_policy() -> ToolsPolicy:
    """The installed project tool policy."""
    return _policy


def is_enabled(name: str) -> bool:
    """Whether the project policy lets ``name`` run: always-on tools are
    never gated; ``disabled`` wins; a non-empty ``enabled`` set allows only
    its members."""
    if name in ALWAYS_ON_TOOLS:
        return True
    if name in _policy.disabled:
        return False
    if _policy.enabled:
        return name in _policy.enabled
    return True
