"""Anthropic provider adapter.

One class, two configured auth modes:

- ``api_key`` — the ``anthropic`` SDK with an API key (+ optional ``base_url``)
  pointed at a local endpoint that speaks the Anthropic Messages API.
- ``oauth`` — an ``ant``-managed profile under ``~/.config/anthropic``; the SDK
  refreshes and re-stores tokens and adds the ``oauth-2025-04-20`` beta header.

Full tool parity: guru's tool directory is translated to Anthropic tool
schema and the model's ``tool_use`` requests run through the shared
``guru.domain.tools.execute_tool``.

Prompt caching (``cache = true`` in the adapter record, the default): the
system prompt is sent as one text block and it and the last tool
definition carry ``cache_control: {type: ephemeral}``, so the stable prefix
(tools, then system) is cached across the rounds of a turn and across
turns; the messages themselves stay uncached.
"""
import os
import pathlib
import shutil
import subprocess
import time
from typing import Optional, Union

from guru import log, session, ui
from guru.adapters import turn
from guru.adapters.base import JSON_ONLY, Adapter, ModelInfo
from guru.domain import ledger, pricing, tools

# Non-streaming per tool-call round (parity with the Ollama adapter). Kept at
# the SDK's non-streaming ceiling to avoid the large-output timeout guard.
_MAX_TOKENS = 16000
_DEFAULT_CONTEXT = 200000
CACHE_CONTROL = {'type': 'ephemeral'}


# --- pure translation helpers (unit-tested) ----------------------------------

def to_anthropic_messages(messages: list) -> tuple:
    """Translate neutral messages to (system_str, anthropic_messages).

    All ``system`` messages are merged into the top-level system string.
    Historical tool calls/results are flattened to plain text — precise
    tool_use/tool_result id-linking is only needed for the in-flight turn,
    which the adapter builds natively. This keeps cross-provider history
    (e.g. a chat started on Ollama) translatable without fabricated ids.
    """
    system_parts: list = []
    out: list = []
    for m in messages:
        role = m.get('role') if isinstance(m, dict) else getattr(m, 'role', '')
        content = (
            m.get('content') if isinstance(m, dict)
            else getattr(m, 'content', '')) or ''
        if role == 'system':
            if content:
                system_parts.append(content)
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
            text = content
            if tool_calls and not text:
                text = '(used tools)'
            out.append({'role': 'assistant', 'content': text})
        else:  # user
            out.append({'role': 'user', 'content': content})
    return "\n\n".join(system_parts), out


def tool_defs(specs: list) -> list:
    """Translate provider-neutral tool specs to Anthropic tool schema."""
    defs = []
    for spec in specs:
        params = spec.get('parameters', {})
        properties = {
            name: {'type': 'string', 'description': desc}
            for name, desc in params.items()
        }
        defs.append({
            'name': spec['name'],
            'description': spec['description'],
            'input_schema': {
                'type': 'object',
                'properties': properties,
                'required': [
                    k for k in params if k not in spec.get('optional', ())],
            },
        })
    return defs


def system_blocks(system: str,
                  cache: bool) -> Optional[Union[str, list]]:
    """The ``system`` request field: with ``cache`` a one-block list that
    carries the cache marker, else the plain string; None when empty."""
    if not system:
        return None
    if not cache:
        return system
    return [{'type': 'text', 'text': system,
             'cache_control': dict(CACHE_CONTROL)}]


def cached_tools(defs: list, cache: bool) -> list:
    """``defs`` with the cache marker on the last tool (a copy), or ``defs``
    unchanged when ``cache`` is off or there are no tools."""
    if not cache or not defs:
        return defs
    out = [dict(d) for d in defs]
    out[-1]['cache_control'] = dict(CACHE_CONTROL)
    return out


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

class AnthropicAdapter(Adapter):
    """Anthropic Messages API provider (api_key or oauth)."""

    def __init__(self, name: str = "Anthropic", auth: str = "api_key",
                 base_url=None, api_key_env=None, api_key=None, profile=None,
                 models=None, thinking: bool = True,
                 cache: bool = True) -> None:
        self.name = name
        self.auth = auth
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.api_key = api_key
        self.profile = profile
        self.static_models = models or []
        self.thinking = thinking
        self.cache = bool(cache)
        self._context_by_model: dict = {}

    # --- client construction -------------------------------------------------

    def _oauth_credentials_path(self) -> pathlib.Path:
        """Path to the SDK/ant OAuth credentials file for this profile."""
        base = (os.environ.get('ANTHROPIC_CONFIG_DIR')
                or os.path.expanduser('~/.config/anthropic'))
        profile = (self.profile or os.environ.get('ANTHROPIC_PROFILE')
                   or 'default')
        return pathlib.Path(base) / 'credentials' / f'{profile}.json'

    def _client(self):
        import anthropic
        if self.auth not in ('api_key', 'oauth'):
            raise RuntimeError(f"unknown auth mode '{self.auth}'")
        if self.auth == 'oauth':
            # The SDK reads the profile, refreshes tokens (persisting rotated
            # refresh tokens back to the credentials file), and adds the
            # oauth-2025-04-20 beta header itself.
            kwargs = {}
            if self.profile:
                kwargs['profile'] = self.profile
            if self.base_url:
                kwargs['base_url'] = self.base_url
            return anthropic.Anthropic(**kwargs)
        env = self.api_key_env or 'ANTHROPIC_API_KEY'
        key = os.environ.get(env) or self.api_key or 'local'
        kwargs = {'api_key': key}
        if self.base_url:
            kwargs['base_url'] = self.base_url
        return anthropic.Anthropic(**kwargs)

    # --- discovery -----------------------------------------------------------

    def available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except Exception:
            log.exc('anthropic SDK import failed')
            return False
        if self.auth not in ('api_key', 'oauth'):
            return False
        if self.auth == 'oauth':
            # A profile must have been created by a one-time `ant auth login`.
            return self._oauth_credentials_path().exists()
        return True

    def _run_ant_login(self) -> tuple:
        """Run the one-time browser OAuth login via the ant CLI."""
        if not shutil.which('ant'):
            return (False, "the `ant` CLI is not installed — run:"
                           " brew install anthropics/tap/ant")
        profile = self.profile or 'default'
        ui.console.print(
            f"[dim]Opening browser login: ant auth login"
            f" --profile {profile}…[/dim]"
        )
        try:
            subprocess.run(['ant', 'auth', 'login', '--profile', profile])
        except Exception as e:
            return (False, f"ant auth login failed: {e}")
        if self._oauth_credentials_path().exists():
            return (True, "logged in")
        return (False, "login did not produce credentials")

    def verify(self) -> tuple:
        try:
            import anthropic  # noqa: F401
        except Exception:
            return (False, "the anthropic SDK is not installed")
        if self.auth not in ('api_key', 'oauth'):
            return (False, f"unknown auth mode '{self.auth}'")
        if (self.auth == 'oauth'
                and not self._oauth_credentials_path().exists()):
            ok, msg = self._run_ant_login()
            if not ok:
                return (False, msg)
        if self.static_models:
            return (True, "configured")
        try:
            for _ in self._client().models.list():   # one network round-trip
                break
            return (True, "authenticated")
        except Exception as e:
            return (False, str(e))

    def list_models(self) -> list:
        if self.static_models:
            return [
                ModelInfo(self.name, mid, mid,
                          self._context_by_model.get(mid, _DEFAULT_CONTEXT))
                for mid in self.static_models
            ]
        try:
            client = self._client()
            out = []
            for m in client.models.list():
                ctx = getattr(m, 'max_input_tokens', 0) or _DEFAULT_CONTEXT
                self._context_by_model[m.id] = ctx
                out.append(ModelInfo(
                    adapter=self.name,
                    model_id=m.id,
                    label=getattr(m, 'display_name', None) or m.id,
                    context_window=ctx,
                ))
            return out
        except Exception:
            log.exc('anthropic models.list failed')
            return []

    def activate(self, model_id: str) -> None:
        session.model = model_id
        session.model_size = ''
        ctx = self._context_by_model.get(model_id)
        if ctx is None:
            ctx = self._retrieve_context(model_id)
        session.num_ctx = ctx
        session.ctx_ceiling = ctx

    def _retrieve_context(self, model_id: str) -> int:
        try:
            info = self._client().models.retrieve(model_id)
            return getattr(info, 'max_input_tokens', 0) or _DEFAULT_CONTEXT
        except Exception:
            log.exc('anthropic context retrieve failed')
            return _DEFAULT_CONTEXT

    # --- ledger --------------------------------------------------------------

    def _record_call(self, phase: str, resp, seconds: float,
                     model: str = '') -> None:
        """Write one CallRecord from a Messages API response's usage
        (``model`` when the call ran on another model than the session's).
        Never raises into the turn."""
        try:
            usage = getattr(resp, 'usage', None)
            ledger.record_call(
                adapter=self.name, model=model or session.model,
                usage=pricing.Usage(
                    input_tokens=getattr(usage, 'input_tokens', 0) or 0,
                    output_tokens=getattr(usage, 'output_tokens', 0) or 0,
                    cache_read_tokens=getattr(
                        usage, 'cache_read_input_tokens', 0) or 0,
                    cache_write_tokens=getattr(
                        usage, 'cache_creation_input_tokens', 0) or 0),
                seconds=seconds, phase=phase)
        except Exception:                                # noqa: BLE001
            log.exc('anthropic call record failed')

    # --- turn loop -----------------------------------------------------------

    def run_turn(self) -> None:
        try:
            client = self._client()
        except Exception as e:
            ui.console.print(f"[red]Anthropic auth error: {e}[/red]")
            return
        system, native = to_anthropic_messages(session.messages)
        anth_tools = cached_tools(tool_defs(tools.active_specs()), self.cache)
        system_field = system_blocks(system, self.cache)

        def step():
            """One Messages API round; returns (text, [(name, input, block)])
            or None on error (printed) — the shared loop handles cancel."""
            kwargs: dict = {
                'model': session.model,
                'max_tokens': _MAX_TOKENS,
                'messages': native,
                'tools': anth_tools,
            }
            if system_field is not None:
                kwargs['system'] = system_field
            if self.thinking:
                kwargs['thinking'] = {
                    'type': 'adaptive', 'display': 'summarized'}
            t0 = time.perf_counter()
            try:
                resp = client.messages.create(**kwargs)
            except Exception as e:
                _note_error(e)
                ui.console.print(f"[red]Anthropic error: {e}[/red]")
                return None
            if getattr(resp, 'stop_reason', None) == 'refusal':
                ledger.bump('refusals')

            usage = resp.usage
            session.session_in += getattr(usage, 'input_tokens', 0) or 0
            session.session_out += getattr(usage, 'output_tokens', 0) or 0
            session.ctx_used = (
                getattr(usage, 'input_tokens', 0) or session.ctx_used)
            self._record_call('step', resp, time.perf_counter() - t0)

            text_parts: list = []
            tool_uses: list = []
            for block in resp.content:
                if block.type == 'text':
                    text_parts.append(block.text)
                elif block.type == 'tool_use':
                    tool_uses.append(block)

            ui.debug(
                f"stop={resp.stop_reason} text={''.join(text_parts)!r}"
                f" tools={[b.name for b in tool_uses]}")

            # Native history keeps precise tool linking for this turn.
            native.append({'role': 'assistant', 'content': resp.content})
            session.messages.append(neutral_assistant(
                ''.join(text_parts),
                [(b.name, dict(b.input)) for b in tool_uses],
            ))
            calls = [(b.name, dict(b.input), b) for b in tool_uses]
            return (''.join(text_parts), calls)

        def run_tools(pending):
            # All tool_results for a round must go back in ONE user turn.
            results = []
            for name, args, block, duplicate in pending:
                if duplicate:
                    ui.console.print(
                        f"[yellow]\\[SKIP][/yellow] duplicate: {name}({args})")
                    content = (f"Already called {name} with these arguments."
                               " Use the previous result.")
                else:
                    content = tools.execute_tool(name, args)
                results.append({
                    'type': 'tool_result',
                    'tool_use_id': block.id,
                    'content': content,
                })
                session.messages.append({
                    'role': 'tool', 'tool_name': name, 'tool_args': args,
                    'content': content})
            native.append({'role': 'user', 'content': results})

        def add_user(text):
            native.append({'role': 'user', 'content': text})
            session.messages.append({'role': 'user', 'content': text})

        turn.run_loop(step=step, run_tools=run_tools, add_user=add_user)

    # --- summarisation -------------------------------------------------------

    def summarise(self, transcript: str) -> str:
        try:
            t0 = time.perf_counter()
            resp = self._client().messages.create(
                model=session.model,
                max_tokens=1024,
                system=(
                    'Summarise the following conversation concisely. Keep'
                    ' facts, decisions, and any URLs or identifiers the user'
                    ' may refer to later. Output only the summary.'
                ),
                messages=[{'role': 'user', 'content': transcript}],
            )
            self._record_call('summarise', resp, time.perf_counter() - t0)
            text = next(
                (b.text for b in resp.content if b.type == 'text'), '')
            return text.strip() or '(summary unavailable)'
        except Exception as e:
            _note_error(e)
            return f'(summary failed: {e})'

    def complete(self, prompt: str, max_tokens: int = 1024,
                 model: str = '') -> str:
        t0 = time.perf_counter()
        resp = self._client().messages.create(
            model=model or session.model, max_tokens=int(max_tokens),
            system=JSON_ONLY,
            messages=[{'role': 'user', 'content': prompt}])
        self._record_call('complete', resp, time.perf_counter() - t0,
                          model=model)
        return next((b.text for b in resp.content if b.type == 'text'),
                    '').strip()


def _note_error(e: Exception) -> None:
    """Count a provider failure on the bound session and keep its text."""
    ledger.bump('provider_errors')
    session.last_error = repr(e)[:200]
