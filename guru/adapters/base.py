"""Provider adapter interface.

An adapter knows how to list a provider's models and run one user turn
(including the provider's tool-calling loop) against the shared session
state. Tool execution and gating stay in the domain layer — adapters call
``guru.domain.tools.execute_tool`` when a model requests a tool.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

# System instruction for :meth:`Adapter.complete`: every single-shot
# completion guru asks for (the sandbox gate's review) is machine-read, so
# the model is told once, the same way on every provider, to emit JSON only.
JSON_ONLY = ('Answer with a single JSON object and nothing else: no prose,'
             ' no markdown fences, no explanation outside the object.')


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
    enabled: bool = True
    # Whether prompts leave the machine. Routing strips remote adapters in
    # local-only mode and whenever the secret scanner finds something; the
    # remote path also redacts tool output. Local providers set False.
    remote: bool = True

    def verify(self) -> tuple:
        """Check the adapter works, triggering auth if needed.

        Returns ``(ok, message)``. The default probes ``available()``;
        adapters override to run connectivity or login checks.
        """
        return (self.available(), "")

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

    def complete(self, prompt: str, max_tokens: int = 1024,
                 model: str = '') -> str:
        """One single-shot completion of ``prompt`` under the ``JSON_ONLY``
        system instruction; returns the model's text (the caller parses
        it). ``model`` overrides the session's model (the gate reviewer
        runs on a routed model, possibly off the session's thread), so an
        empty ``model`` should only be used from a bound session. No tool
        loop, nothing appended to the conversation; the call is recorded
        with ``phase='complete'``. Raises the provider's error: the
        decision seam turns it into an ``error`` row. Adapters that cannot
        complete raise ``NotImplementedError`` (the default)."""
        raise NotImplementedError(f'{self.name} has no complete()')
