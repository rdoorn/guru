"""Anthropic provider adapter.

One class, two configured auth modes:

- ``api_key`` — the ``anthropic`` SDK with an API key (+ optional ``base_url``)
  pointed at a local endpoint that speaks the Anthropic Messages API.
- ``oauth`` — an ``ant``-managed profile under ``~/.config/anthropic``; the SDK
  refreshes and re-stores tokens and adds the ``oauth-2025-04-20`` beta header.

Full tool parity: guru's tool directory is translated to Anthropic tool
schema and the model's ``tool_use`` requests run through the shared
``guru.domain.tools.execute_tool``.
"""
import os
import pathlib
import shutil
import subprocess

from rich.markdown import Markdown

from guru import session, ui
from guru.adapters.base import Adapter, ModelInfo
from guru.domain import tools

# Non-streaming per tool-call round (parity with the Ollama adapter). Kept at
# the SDK's non-streaming ceiling to avoid the large-output timeout guard.
_MAX_TOKENS = 16000
_DEFAULT_CONTEXT = 200000


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
                'required': list(params.keys()),
            },
        })
    return defs


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
                 base_url=None, api_key_env=None, profile=None,
                 models=None, thinking: bool = True) -> None:
        self.name = name
        self.auth = auth
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.profile = profile
        self.static_models = models or []
        self.thinking = thinking
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
        key = os.environ.get(env) or 'local'
        kwargs = {'api_key': key}
        if self.base_url:
            kwargs['base_url'] = self.base_url
        return anthropic.Anthropic(**kwargs)

    # --- discovery -----------------------------------------------------------

    def available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except Exception:
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
            return _DEFAULT_CONTEXT

    # --- turn loop -----------------------------------------------------------

    def run_turn(self) -> None:
        try:
            client = self._client()
        except Exception as e:
            ui.console.print(f"[red]Anthropic auth error: {e}[/red]")
            return
        system, native = to_anthropic_messages(session.messages)
        anth_tools = tool_defs(tools.active_specs())

        while True:
            kwargs: dict = {
                'model': session.model,
                'max_tokens': _MAX_TOKENS,
                'messages': native,
                'tools': anth_tools,
            }
            if system:
                kwargs['system'] = system
            if self.thinking:
                kwargs['thinking'] = {
                    'type': 'adaptive', 'display': 'summarized'}
            try:
                resp = client.messages.create(**kwargs)
            except Exception as e:
                ui.console.print(f"[red]Anthropic error: {e}[/red]")
                return

            usage = resp.usage
            session.session_in += getattr(usage, 'input_tokens', 0) or 0
            session.session_out += getattr(usage, 'output_tokens', 0) or 0
            session.ctx_used = (
                getattr(usage, 'input_tokens', 0) or session.ctx_used)
            ui.status_draw()

            text_parts: list = []
            thinking_parts: list = []
            tool_uses: list = []
            for block in resp.content:
                if block.type == 'thinking':
                    thinking_parts.append(getattr(block, 'thinking', '') or '')
                elif block.type == 'text':
                    text_parts.append(block.text)
                elif block.type == 'tool_use':
                    tool_uses.append(block)

            if any(thinking_parts):
                ui.console.print("\n[dim]\\[THINKING][/dim]")
                ui.console.print(
                    f"[dim italic]{' '.join(thinking_parts)}[/dim italic]")

            ui.console.print(
                f"[DEBUG] stop={resp.stop_reason}"
                f" text={''.join(text_parts)!r}"
                f" tools={[b.name for b in tool_uses]}",
                style="dim yellow", markup=False,
            )

            # Native history keeps precise tool linking for this turn.
            native.append({'role': 'assistant', 'content': resp.content})
            session.messages.append(neutral_assistant(
                ''.join(text_parts),
                [(b.name, dict(b.input)) for b in tool_uses],
            ))

            if resp.stop_reason != 'tool_use':
                final = ''.join(text_parts).strip()
                ui.console.print("\n[bold green]Guru>[/bold green]")
                ui.console.print(Markdown(final))
                ui.console.print()
                return

            results = []
            for block in tool_uses:
                result = tools.execute_tool(block.name, dict(block.input))
                results.append({
                    'type': 'tool_result',
                    'tool_use_id': block.id,
                    'content': result,
                })
                session.messages.append({
                    'role': 'tool',
                    'tool_name': block.name,
                    'content': result,
                })
            native.append({'role': 'user', 'content': results})

    # --- summarisation -------------------------------------------------------

    def summarise(self, transcript: str) -> str:
        try:
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
            text = next(
                (b.text for b in resp.content if b.type == 'text'), '')
            return text.strip() or '(summary unavailable)'
        except Exception as e:
            return f'(summary failed: {e})'
