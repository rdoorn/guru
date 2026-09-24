"""Adapter registry: the configured provider adapters by name.

Built by the CLI from ``adapters.toml`` and handed to the orchestrator so a
resolved ``Route`` (an adapter *name* plus a model) can be turned into the
adapter object that runs the sub-agent. ``is_remote`` exposes the adapter's
``remote`` flag for the repository layer that builds routing ladders.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Iterator, Optional

if TYPE_CHECKING:                    # type-only: keep the endpoint layer out
    from guru.adapters.base import Adapter


class AdapterRegistry:
    """Ordered name -> Adapter mapping; registering a name again replaces the
    earlier adapter in place (order preserved)."""

    def __init__(self, adapters: Iterable[Adapter] = ()) -> None:
        self._adapters: dict[str, Adapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: Adapter) -> None:
        """Add (or replace) ``adapter`` under its ``name``."""
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> Optional[Adapter]:
        """The adapter called ``name``, or None."""
        return self._adapters.get(name)

    def names(self) -> list[str]:
        """Registered adapter names in registration order."""
        return list(self._adapters)

    def is_remote(self, name: str) -> bool:
        """Whether adapter ``name`` sends content off-machine.

        Raises ``KeyError`` for an unknown name so a misconfigured ladder
        rung is caught at load time rather than defaulting silently.
        """
        return bool(self._adapters[name].remote)

    def __iter__(self) -> Iterator[Adapter]:
        return iter(self._adapters.values())

    def __len__(self) -> int:
        return len(self._adapters)


def registry_from(adapters: Iterable[Adapter]) -> AdapterRegistry:
    """The registry over ``adapters`` (all of them, enabled or not, so a
    ladder rung on a disabled adapter is reported rather than unknown).
    Shared by the CLI and the eval runner."""
    return AdapterRegistry(adapters)
