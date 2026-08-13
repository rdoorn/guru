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
"""
import json
import os

import requests
from rich.markdown import Markdown

from guru import session, ui
from guru.adapters.base import Adapter, ModelInfo
from guru.domain import tools

_MAX_TOKENS = 4096
_DEFAULT_CONTEXT = 128000
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
                    'required': list(params.keys()),
                },
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

class LiteLLMAdapter(Adapter):
    """OpenAI-compatible provider (e.g. a LiteLLM proxy)."""

    def __init__(self, name: str = "LiteLLM", base_url=None,
                 api_key_env=None, api_key=None, models=None) -> None:
        self.name = name
        self.base_url = (base_url or '').rstrip('/')
        self.api_key_env = api_key_env
        self.api_key = api_key
        self.static_models = models or []
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

    # --- turn loop -----------------------------------------------------------

    def run_turn(self) -> None:
        client = self._client()
        native = to_openai_messages(session.messages)
        oa_tools = openai_tool_defs(tools.active_specs())

        while True:
            try:
                resp = client.chat.completions.create(
                    model=session.model,
                    messages=native,
                    tools=oa_tools or None,
                    max_tokens=_MAX_TOKENS,
                )
            except Exception as e:
                ui.console.print(f"[red]LiteLLM error: {e}[/red]")
                return

            usage = getattr(resp, 'usage', None)
            if usage:
                session.session_in += getattr(usage, 'prompt_tokens', 0) or 0
                session.session_out += (
                    getattr(usage, 'completion_tokens', 0) or 0)
                session.ctx_used = (
                    getattr(usage, 'prompt_tokens', 0) or session.ctx_used)
                ui.status_draw()

            msg = resp.choices[0].message
            text = msg.content or ''
            tool_calls = list(getattr(msg, 'tool_calls', None) or [])

            ui.console.print(
                f"[DEBUG] finish={resp.choices[0].finish_reason}"
                f" text={text!r}"
                f" tools={[t.function.name for t in tool_calls]}",
                style="dim yellow", markup=False,
            )

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

            parsed = []
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or '{}')
                except json.JSONDecodeError:
                    args = {}
                parsed.append((tc.id, tc.function.name, args))
            session.messages.append(neutral_assistant(
                text, [(name, args) for _, name, args in parsed]))

            if not tool_calls:
                ui.console.print("\n[bold green]answer>[/bold green]")
                ui.console.print(Markdown(text.strip()))
                ui.console.print()
                return

            for call_id, name, args in parsed:
                result = tools.execute_tool(name, args)
                native.append({
                    'role': 'tool',
                    'tool_call_id': call_id,
                    'content': result,
                })
                session.messages.append({
                    'role': 'tool',
                    'tool_name': name,
                    'content': result,
                })

    # --- summarisation -------------------------------------------------------

    def summarise(self, transcript: str) -> str:
        try:
            resp = self._client().chat.completions.create(
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
            return (resp.choices[0].message.content or '').strip() \
                or '(summary unavailable)'
        except Exception as e:
            return f'(summary failed: {e})'
