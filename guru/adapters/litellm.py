"""LiteLLM (OpenAI-compatible) provider adapter.

Talks to a LiteLLM proxy over the standard OpenAI API: the ``openai`` SDK for
chat completions, and a raw ``/v1/models`` GET for listing (so LiteLLM's
``mode`` / ``max_input_tokens`` extras are visible). Full tool parity via
OpenAI function-calling.

Config (adapters.toml):

    [[adapter]]
    name = "SBP Litellm"
    type = "litellm"
    base_url = "https://proxy.example/v1"   # include /v1
    api_key_env = "SBP_LITELLM_KEY"          # env var holding the virtual key
    # models = ["azure/gpt-4.1", "anthropic/claude-..."]  # optional allowlist
    # cache = true                            # prompt-cache markers (default)

Prompt caching: with ``cache`` on, system messages are sent as content
parts and the last part and the last tool definition carry
``cache_control: {type: ephemeral}`` — the OpenAI-compatible shape a LiteLLM
proxy forwards to Anthropic and Bedrock. Cache usage comes back either as
Anthropic-style ``cache_read_input_tokens`` / ``cache_creation_input_tokens``
on ``usage`` or as ``prompt_tokens_details.cached_tokens``; both are read.
"""
import json
import math
import os
import time

import requests

from guru import log, session, ui
from guru.adapters import turn
from guru.adapters.base import JSON_ONLY, Adapter, ModelInfo
from guru.domain import ledger, pricing, tools

_MAX_TOKENS = 16384   # proxies may enforce a thinking budget above 8k
_DEFAULT_CONTEXT = 128000
CACHE_CONTROL = {'type': 'ephemeral'}
# LiteLLM `mode` values that are not chat models — hidden from /models.
_NON_CHAT_MODES = {
    'audio_transcription', 'audio_speech', 'embedding',
    'image_generation', 'moderation', 'rerank', 'completion',
}


# --- pure translation helpers (unit-tested) ----------------------------------

def to_openai_messages(messages: list) -> list:
    """Translate neutral messages to OpenAI chat messages.

    Historical tool calls/results are flattened to text — precise
    tool_call_id linking is only needed for the in-flight turn, which the
    adapter builds natively.
    """
    out: list = []
    for m in messages:
        role = m.get('role') if isinstance(m, dict) else getattr(m, 'role', '')
        content = (
            m.get('content') if isinstance(m, dict)
            else getattr(m, 'content', '')) or ''
        if role == 'system':
            if content:
                out.append({'role': 'system', 'content': content})
        elif role == 'tool':
            name = (
                m.get('tool_name', 'tool')
                if isinstance(m, dict) else 'tool')
            out.append({
                'role': 'user',
                'content': f"[tool {name} result]\n{content}",
            })
        elif role == 'assistant':
            tool_calls = m.get('tool_calls') if isinstance(m, dict) else None
            text = content or ('(used tools)' if tool_calls else '')
            out.append({'role': 'assistant', 'content': text})
        else:
            out.append({'role': 'user', 'content': content})
    return out


def openai_tool_defs(specs: list) -> list:
    """Translate provider-neutral tool specs to OpenAI function-calling."""
    defs = []
    for spec in specs:
        params = spec.get('parameters', {})
        properties = {
            name: {'type': 'string', 'description': desc}
            for name, desc in params.items()
        }
        defs.append({
            'type': 'function',
            'function': {
                'name': spec['name'],
                'description': spec['description'],
                'parameters': {
                    'type': 'object',
                    'properties': properties,
                    'required': [
                        k for k in params
                        if k not in spec.get('optional', ())],
                },
            },
        })
    return defs


def cached_messages(messages: list, cache: bool) -> list:
    """``messages`` with every system message's text as one content part
    and the cache marker on the last system message (copies); unchanged
    when ``cache`` is off or there is no system message."""
    if not cache:
        return messages
    last = max((i for i, m in enumerate(messages)
                if m.get('role') == 'system'), default=-1)
    if last < 0:
        return messages
    out = []
    for i, m in enumerate(messages):
        if m.get('role') != 'system' or not isinstance(m.get('content'), str):
            out.append(m)
            continue
        part: dict = {'type': 'text', 'text': m['content']}
        if i == last:
            part['cache_control'] = dict(CACHE_CONTROL)
        out.append({**m, 'content': [part]})
    return out


def cached_tools(defs, cache: bool):
    """``defs`` with the cache marker on the last tool (a copy), or ``defs``
    unchanged when ``cache`` is off or there are no tools."""
    if not cache or not defs:
        return defs
    out = [dict(d) for d in defs]
    out[-1]['cache_control'] = dict(CACHE_CONTROL)
    return out


def _attr(obj, name: str) -> int:
    """``int(obj.<name>)`` (or ``obj[name]`` for a dict), 0 when missing
    or not a number."""
    value = (obj.get(name) if isinstance(obj, dict)
             else getattr(obj, name, None))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def usage_from(usage) -> pricing.Usage:
    """Token counts of an OpenAI-compatible ``usage`` (object or dict).

    ``prompt_tokens`` is the whole prompt (LiteLLM counts cached tokens in
    it); cache reads come from Anthropic-style ``cache_read_input_tokens``
    or ``prompt_tokens_details.cached_tokens``, cache writes from
    ``cache_creation_input_tokens``. ``input_tokens`` is the uncached
    remainder so the price table charges each class once.
    """
    if usage is None:
        return pricing.Usage()
    prompt = _attr(usage, 'prompt_tokens')
    details = (usage.get('prompt_tokens_details') if isinstance(usage, dict)
               else getattr(usage, 'prompt_tokens_details', None))
    read = _attr(usage, 'cache_read_input_tokens')
    if not read and details is not None:
        read = _attr(details, 'cached_tokens')
    write = _attr(usage, 'cache_creation_input_tokens')
    return pricing.Usage(
        input_tokens=max(prompt - read - write, 0),
        output_tokens=_attr(usage, 'completion_tokens'),
        cache_read_tokens=read, cache_write_tokens=write)


def neutral_assistant(text: str, tool_calls: list) -> dict:
    """Build a neutral assistant message from text + [(name, input), ...]."""
    msg: dict = {'role': 'assistant', 'content': text}
    if tool_calls:
        msg['tool_calls'] = [
            {'function': {'name': name, 'arguments': args}}
            for name, args in tool_calls
        ]
    return msg


# --- adapter -----------------------------------------------------------------

class LiteLLMAdapter(Adapter):
    """OpenAI-compatible provider (e.g. a LiteLLM proxy)."""

    def __init__(self, name: str = "LiteLLM", base_url=None,
                 api_key_env=None, api_key=None, models=None,
                 cache: bool = True) -> None:
        self.name = name
        self.base_url = (base_url or '').rstrip('/')
        self.api_key_env = api_key_env
        self.api_key = api_key
        self.static_models = models or []
        self.cache = bool(cache)
        self._context_by_model: dict = {}

    def _key(self) -> str:
        """Resolve the key: env var → inline api_key → OPENAI_API_KEY."""
        return (
            os.environ.get(self.api_key_env or '')
            or self.api_key
            or os.environ.get('OPENAI_API_KEY')
            or ''
        )

    def _client(self):
        import openai
        return openai.OpenAI(base_url=self.base_url, api_key=self._key())

    # --- discovery -----------------------------------------------------------

    def available(self) -> bool:
        return bool(self.base_url)

    def verify(self) -> tuple:
        if not self.base_url:
            return (False, "no base_url configured")
        if not self._key():
            env = self.api_key_env or 'OPENAI_API_KEY'
            return (False, f"no API key (set ${env} or api_key in config)")
        try:
            r = requests.get(
                self.base_url + '/models',
                headers={'Authorization': f'Bearer {self._key()}'},
                timeout=15)
            r.raise_for_status()
            return (True, "reachable")
        except Exception as e:
            return (False, str(e))

    def list_models(self) -> list:
        if self.static_models:
            return [
                ModelInfo(self.name, m, m,
                          self._context_by_model.get(m, _DEFAULT_CONTEXT))
                for m in self.static_models
            ]
        try:
            r = requests.get(
                self.base_url + '/models',
                headers={'Authorization': f'Bearer {self._key()}'},
                timeout=15)
            r.raise_for_status()
            data = r.json().get('data', [])
        except Exception:
            log.exc('litellm models.list failed')
            return []
        infos = []
        for m in data:
            if m.get('mode') in _NON_CHAT_MODES:
                continue
            ctx = int(m.get('max_input_tokens') or 0) or _DEFAULT_CONTEXT
            self._context_by_model[m['id']] = ctx
            infos.append(ModelInfo(self.name, m['id'], m['id'], ctx))
        return sorted(infos, key=lambda i: i.model_id)

    def activate(self, model_id: str) -> None:
        session.model = model_id
        session.model_size = ''
        session.num_ctx = self._context_by_model.get(
            model_id, _DEFAULT_CONTEXT)
        session.ctx_ceiling = session.num_ctx

    # --- ledger --------------------------------------------------------------

    def _record_call(self, phase: str, resp, seconds: float,
                     cost_header=None, model: str = '') -> None:
        """Write one CallRecord. ``cost_header`` is the proxy's per-response
        cost (from :func:`_complete`); it wins over the price table;
        ``model`` names the model when it is not the session's. Never
        raises into the turn."""
        try:
            ledger.record_call(
                adapter=self.name, model=model or session.model,
                usage=usage_from(getattr(resp, 'usage', None)),
                seconds=seconds, phase=phase, cost_header=cost_header)
        except Exception:                                # noqa: BLE001
            log.exc('litellm call record failed')

    # --- turn loop -----------------------------------------------------------

    def run_turn(self) -> None:
        client = self._client()
        native = to_openai_messages(session.messages)
        oa_tools = cached_tools(openai_tool_defs(tools.active_specs()),
                                self.cache)

        def step():
            """One chat-completions round; returns (text, [(name, args, id)])
            or None on error (printed) — the shared loop handles cancel."""
            t0 = time.perf_counter()
            try:
                resp, cost = _complete(
                    client,
                    model=session.model,
                    messages=cached_messages(native, self.cache),
                    tools=oa_tools or None,
                    max_tokens=_MAX_TOKENS,
                )
            except Exception as e:
                _note_error(e)
                ui.console.print(f"[red]LiteLLM error: {e}[/red]")
                return None

            usage = getattr(resp, 'usage', None)
            if usage:
                session.session_in += getattr(usage, 'prompt_tokens', 0) or 0
                session.session_out += (
                    getattr(usage, 'completion_tokens', 0) or 0)
                session.ctx_used = (
                    getattr(usage, 'prompt_tokens', 0) or session.ctx_used)
            self._record_call(
                'step', resp, time.perf_counter() - t0, cost)

            msg = resp.choices[0].message
            text = msg.content or ''
            tool_calls = list(getattr(msg, 'tool_calls', None) or [])

            ui.debug(
                f"finish={resp.choices[0].finish_reason} text={text!r}"
                f" tools={[t.function.name for t in tool_calls]}")

            # Append the assistant turn to native history for id-linking.
            assistant: dict = {'role': 'assistant', 'content': text or None}
            if tool_calls:
                assistant['tool_calls'] = [
                    {
                        'id': tc.id,
                        'type': 'function',
                        'function': {
                            'name': tc.function.name,
                            'arguments': tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            native.append(assistant)

            calls = []
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or '{}')
                except json.JSONDecodeError:
                    args = {}
                calls.append((tc.function.name, args, tc.id))
            session.messages.append(neutral_assistant(
                text, [(name, args) for name, args, _ in calls]))
            return (text, calls)

        def run_tools(pending):
            for name, args, call_id, duplicate in pending:
                if duplicate:
                    ui.console.print(
                        f"[yellow]\\[SKIP][/yellow] duplicate: {name}({args})")
                    content = (f"Already called {name} with these arguments."
                               " Use the previous result.")
                else:
                    content = tools.execute_tool(name, args)
                native.append({
                    'role': 'tool', 'tool_call_id': call_id,
                    'content': content})
                session.messages.append({
                    'role': 'tool', 'tool_name': name, 'tool_args': args,
                    'content': content})

        def add_user(text):
            native.append({'role': 'user', 'content': text})
            session.messages.append({'role': 'user', 'content': text})

        turn.run_loop(step=step, run_tools=run_tools, add_user=add_user)

    # --- summarisation -------------------------------------------------------

    def summarise(self, transcript: str) -> str:
        try:
            t0 = time.perf_counter()
            resp, cost = _complete(
                self._client(),
                model=session.model,
                max_tokens=1024,
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            'Summarise the following conversation concisely.'
                            ' Keep facts, decisions, and any URLs or'
                            ' identifiers the user may refer to later. Output'
                            ' only the summary.'
                        ),
                    },
                    {'role': 'user', 'content': transcript},
                ],
            )
            self._record_call(
                'summarise', resp, time.perf_counter() - t0, cost)
            return (resp.choices[0].message.content or '').strip() \
                or '(summary unavailable)'
        except Exception as e:
            _note_error(e)
            return f'(summary failed: {e})'

    def complete(self, prompt: str, max_tokens: int = 1024,
                 model: str = '') -> str:
        t0 = time.perf_counter()
        resp, cost = _complete(
            self._client(), model=model or session.model,
            max_tokens=int(max_tokens),
            messages=[{'role': 'system', 'content': JSON_ONLY},
                      {'role': 'user', 'content': prompt}])
        self._record_call('complete', resp, time.perf_counter() - t0, cost,
                          model=model)
        return (resp.choices[0].message.content or '').strip()


_COST_HEADER = 'x-litellm-response-cost'


def _complete(client, **kwargs) -> tuple:
    """Run one chat completion; return ``(response, cost_or_None)``.

    Goes through ``with_raw_response`` so the proxy's HTTP headers are
    visible: a LiteLLM proxy reports the per-call price in
    ``x-litellm-response-cost``. The ``openai`` client never populates the
    litellm SDK's ``_hidden_params``, so the header is the only source.
    Exceptions from the call propagate to the caller unchanged.
    """
    raw = client.chat.completions.with_raw_response.create(**kwargs)
    return raw.parse(), _header_cost(getattr(raw, 'headers', None))


def _header_cost(headers):
    """Parse the cost header to a float; None when absent or malformed."""
    try:
        value = headers.get(_COST_HEADER) if headers is not None else None
        cost = float(value) if value not in (None, '') else None
        return cost if cost is not None and math.isfinite(cost) else None
    except (TypeError, ValueError, AttributeError):
        return None


def _note_error(e: Exception) -> None:
    """Count a provider failure on the bound session and keep its text."""
    ledger.bump('provider_errors')
    session.last_error = repr(e)[:200]
