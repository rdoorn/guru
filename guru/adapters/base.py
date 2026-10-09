"""Provider adapter interface.

An adapter knows how to list a provider's models and run one user turn
(including the provider's tool-calling loop) against the shared session
state. Tool execution and gating stay in the domain layer — adapters call
``guru.domain.tools.execute_tool`` when a model requests a tool.
"""
import itertools
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from guru.domain.startup import Location, host_location

# System instruction for :meth:`Adapter.complete`: every single-shot
# completion guru asks for (the sandbox gate's review) is machine-read, so
# the model is told once, the same way on every provider, to emit JSON only.
JSON_ONLY = ('Answer with a single JSON object and nothing else: no prose,'
             ' no markdown fences, no explanation outside the object.')


# Request dumping (debugging prompt caching and prefix stability): with
# ``GURU_DUMP_REQUESTS=<dir>`` every outgoing request's keyword arguments
# are written to ``<dir>/<ts>-<adapter>-<n>.json``. The kwargs are the
# request body only (model, messages, tools, system, ...); the API key is
# held by the SDK client and never appears in them.
DUMP_ENV = 'GURU_DUMP_REQUESTS'
_dump_counter = itertools.count(1)
_SLUG_RE = re.compile(r'[^A-Za-z0-9._-]+')


def _jsonable(value):
    """JSON fallback for SDK objects in a request (content blocks a
    previous response returned): ``model_dump()`` when available, else
    ``repr``."""
    dump = getattr(value, 'model_dump', None)
    if callable(dump):
        return dump()
    return repr(value)


def dump_request(adapter: str, kwargs: dict) -> Optional[Path]:
    """Write ``kwargs`` (one outgoing request) as JSON under the directory
    named by ``$GURU_DUMP_REQUESTS``; returns the path, or None when the
    variable is unset. The file is ``<ts>-<adapter>-<n>.json`` with ``ts``
    in UTC to the millisecond and ``n`` a per-process counter, so a
    directory listing is the request sequence. Never raises into the turn:
    a write failure is swallowed (the dump is a debugging aid)."""
    directory = os.environ.get(DUMP_ENV, '')
    if not directory:
        return None
    try:
        now = time.time()
        stamp = (time.strftime('%Y%m%dT%H%M%S', time.gmtime(now))
                 + f'{int(now * 1000) % 1000:03d}Z')
        slug = _SLUG_RE.sub('_', adapter).strip('_') or 'adapter'
        path = Path(directory) / f'{stamp}-{slug}-{next(_dump_counter)}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(kwargs, indent=1, ensure_ascii=False,
                                   default=_jsonable), encoding='utf-8')
        return path
    except Exception:                                    # noqa: BLE001
        return None


def parameters_schema(spec: dict) -> dict:
    """The JSON schema of a provider-neutral tool spec's input: the
    spec's own ``schema`` when it carries one (the ``plan`` tool's nested
    tasks), else one string property per ``parameters`` entry with the
    non-``optional`` ones required. Shared by every adapter so all three
    send the same shape."""
    schema = spec.get('schema')
    if isinstance(schema, dict):
        return schema
    params = spec.get('parameters', {})
    return {
        'type': 'object',
        'properties': {name: {'type': 'string', 'description': desc}
                       for name, desc in params.items()},
        'required': [k for k in params if k not in spec.get('optional', ())],
    }


def openai_tool_defs(specs: list) -> list:
    """Translate provider-neutral tool specs to OpenAI function-calling
    (the shape the LiteLLM proxy and the Ollama daemon both take)."""
    return [{'type': 'function',
             'function': {'name': spec['name'],
                          'description': spec['description'],
                          'parameters': parameters_schema(spec)}}
            for spec in specs]


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
    # Host shown when the adapter has no url/base_url of its own.
    default_host: str = ''

    def location(self) -> Location:
        """Where the models run: a loopback url is local, any other host
        remote; without a url the ``remote`` flag and ``default_host``."""
        url = getattr(self, 'url', None) or getattr(self, 'base_url', None)
        if url:
            return host_location(url, self.remote)
        return Location('remote' if self.remote else 'local',
                        self.default_host)

    def describe(self) -> str:
        """``Name (local, host)`` for the startup step list."""
        return f'{self.name} ({self.location()})'

    def placement(self) -> str:
        """Where the loaded model sits (e.g. 'GPU'); '' when unknown."""
        return ''

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
