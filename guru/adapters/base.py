"""Provider adapter interface.

An adapter knows how to list a provider's models and run one user turn
(including the provider's tool-calling loop) against the shared session
state. Tool execution and gating stay in the domain layer — adapters call
``guru.domain.tools.execute_tool`` when a model requests a tool.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """A selectable model, grouped in /models under its adapter."""
    adapter: str
    model_id: str
    label: str
    context_window: int
    size: str = ""
    # Estimated RAM to run, in bytes (0 = N/A, e.g. a remote model).
    memory: int = 0


class Adapter(ABC):
    """Base class for provider adapters."""

    name: str = "adapter"

    @abstractmethod
    def available(self) -> bool:
        """Return True if the provider is configured and reachable."""

    @abstractmethod
    def list_models(self) -> list:
        """Return the provider's models as a list of ModelInfo."""

    @abstractmethod
    def activate(self, model_id: str) -> None:
        """Prepare the adapter for a newly selected model.

        Resolves the effective context window, model size, and any provider
        setup (e.g. ensuring a local daemon is running), writing results to
        ``guru.session``.
        """

    @abstractmethod
    def run_turn(self) -> None:
        """Run one user turn against ``guru.session`` to a final answer.

        Assumes the user's message is already appended to session.messages.
        Runs the provider's tool-calling loop, renders output through
        ``guru.ui``, and updates session context/token accounting.
        """

    @abstractmethod
    def summarise(self, transcript: str) -> str:
        """Return a concise summary of a conversation transcript."""
