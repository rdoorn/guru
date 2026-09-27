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

__all__ = ['ALWAYS_ON_TOOLS', 'KIND_HIDDEN_TOOLS', 'ToolsPolicy',
           'WRITE_TOOLS', 'active_policy', 'for_kind', 'is_enabled',
           'kind_refusal', 'set_policy']

# Tools every agent has regardless of the project tool policy: discovery,
# method selection, the delegation mailbox and the turn contract's own
# calls (``final_answer``, a controller's ``plan``) are not registry tools
# and are never gated.
ALWAYS_ON_TOOLS = frozenset(
    ('search_tools', 'use_skill', 'spawn', 'check', 'join', 'final_answer',
     'plan'))

# Every registry tool that changes files -- directly, or in the sandbox
# copy (``sandbox_run`` executes a command in the copy; ``sandbox_python``
# runs code there) and through the gate. The per-task-kind policy below
# hides them: a reviewer reads, it neither edits nor runs.
WRITE_TOOLS = frozenset((
    'write_file', 'edit_file', 'apply_patch', 'delete_file',
    'sandbox_run', 'sandbox_python', 'sandbox_submit',
    'request_dependency'))

# Per task kind (``guru.domain.routing.KINDS``): the registry tools a task
# of that kind neither sees nor may call (structural round, Package C
# item 3). A ``review`` task is read-only: a reviewer reports, it does not
# edit. Kinds absent here hide nothing.
KIND_HIDDEN_TOOLS: dict[str, frozenset] = {'review': WRITE_TOOLS}


def for_kind(kind: object) -> frozenset:
    """The tool names hidden from (and refused for) a task of ``kind``.

    Case-insensitive; an unknown or empty kind hides nothing. The
    orchestrator calls this when it configures a child (``initial_tools``
    takes the same ``kind``) and ``execute_tool`` calls it with the bound
    session's ``task_kind`` so a hidden tool named anyway is refused."""
    key = str(kind).strip().lower() if isinstance(kind, str) else ''
    return KIND_HIDDEN_TOOLS.get(key, frozenset())


def kind_refusal(name: str, kind: object) -> str:
    """The result text for a tool call ``for_kind`` refuses."""
    return (f"Refused: {name} is not available to a {str(kind).lower()}"
            " task (read-only): report the change you would make, with the"
            " file and lines, instead of making it.")


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
