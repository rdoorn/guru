"""Ollama provider adapter.

Wraps the existing local-Ollama behaviour behind the Adapter interface:
model listing, context-window resolution, the tool-calling turn loop, and a
daemon-reachable check with an on-demand model pull (moved from start.sh).
"""
import subprocess

import ollama
from rich.markdown import Markdown

from guru import config, session, ui
from guru.adapters.base import Adapter, ModelInfo
from guru.domain import tools


class OllamaAdapter(Adapter):
    """Local models served by the Ollama daemon."""

    def __init__(self, name: str = "Ollama",
                 url: str = "http://localhost:11434") -> None:
        self.name = name
        self.url = url

    # --- discovery -----------------------------------------------------------

    def available(self) -> bool:
        try:
            ollama.list()
            return True
        except Exception:
            return False

    def list_models(self) -> list:
        try:
            models = ollama.list().models
        except Exception:
            return []
        infos = []
        for m in sorted(models, key=lambda x: x.model):
            num_ctx, _ = self._resolve_context_window(m.model)
            infos.append(ModelInfo(
                adapter=self.name,
                model_id=m.model,
                label=m.model,
                context_window=num_ctx,
                size=self._param_size(m.model),
                # On-disk weight size is a good proxy for the RAM needed to
                # run the model; used to flag models that won't fit memory.
                memory=int(getattr(m, 'size', 0) or 0),
            ))
        return infos

    def activate(self, model_id: str) -> None:
        self._ensure_daemon()
        self._ensure_model(model_id)
        session.model = model_id
        session.num_ctx, session.ctx_ceiling = (
            self._resolve_context_window(model_id))
        session.model_size = self._param_size(model_id)

    # --- context / metadata --------------------------------------------------

    def _resolve_context_window(self, model: str) -> tuple:
        """Return (effective num_ctx, architecture ceiling) for a model."""
        try:
            info = ollama.show(model)
        except Exception:
            return (session.num_ctx_override or config.DEFAULT_NUM_CTX, 0)

        modelinfo = info.modelinfo or {}
        arch = modelinfo.get('general.architecture', '')
        ceiling = int(modelinfo.get(f'{arch}.context_length', 0) or 0)

        modelfile_num_ctx = 0
        for line in (info.parameters or '').splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == 'num_ctx':
                try:
                    modelfile_num_ctx = int(parts[1])
                except ValueError:
                    pass

        num_ctx = (session.num_ctx_override or modelfile_num_ctx
                   or config.DEFAULT_NUM_CTX)
        if ceiling:
            num_ctx = min(num_ctx, ceiling)
        return (num_ctx, ceiling)

    def _param_size(self, model: str) -> str:
        try:
            info = ollama.show(model)
            return getattr(info.details, 'parameter_size', '') or '?'
        except Exception:
            return '?'

    def _ensure_daemon(self) -> None:
        """Best-effort check that the Ollama daemon is reachable."""
        if not self.available():
            ui.console.print(
                "[red]Ollama is not reachable. Launch the Ollama app and"
                " try again.[/red]"
            )

    def _ensure_model(self, model_id: str) -> None:
        """Pull the model if it is not already present locally."""
        try:
            present = {m.model for m in ollama.list().models}
        except Exception:
            return
        if model_id in present:
            return
        ui.console.print(f"[dim]Pulling {model_id} (one-time download)…[/dim]")
        try:
            subprocess.run(['ollama', 'pull', model_id], check=False)
        except Exception as e:
            ui.console.print(f"[red]Could not pull {model_id}: {e}[/red]")

    # --- turn loop -----------------------------------------------------------

    def run_turn(self) -> None:
        called: set = set()
        nudged = 0
        while True:
            # Non-streaming for tool-call rounds: more reliable tool-call
            # detection; streaming is reserved for the final text answer.
            response = ollama.chat(
                model=session.model,
                messages=session.messages,
                think=True,
                tools=session.active_tools,
                options={'num_ctx': session.num_ctx},
            )

            session.session_in += (
                getattr(response, 'prompt_eval_count', 0) or 0)
            session.session_out += (
                getattr(response, 'eval_count', 0) or 0)
            session.ctx_used = (
                getattr(response, 'prompt_eval_count', 0) or session.ctx_used)
            ui.status_draw()

            msg = response.message

            if msg.thinking:
                ui.console.print("\n[dim]\\[THINKING][/dim]")
                ui.console.print(f"[dim italic]{msg.thinking}[/dim italic]")

            ui.console.print(
                f"[DEBUG] content={msg.content!r}"
                f" tool_calls={msg.tool_calls}",
                style="dim yellow",
                markup=False,
            )

            session.messages.append(msg)

            if not msg.tool_calls:
                content = (msg.content or '').strip()
                if not content and nudged < 1:
                    nudged += 1
                    ui.console.print(
                        "[dim yellow]\\[NUDGE][/dim yellow]"
                        " empty response — retrying"
                    )
                    session.messages.append({
                        "role": "user",
                        "content": (
                            "Please continue — use search_tools"
                            " to find what you need, then call it."
                        ),
                    })
                    continue
                ui.console.print("\n[bold green]Guru>[/bold green]")
                ui.console.print(Markdown(content))
                ui.console.print()
                break

            for call in msg.tool_calls:
                name = call.function.name
                arguments = call.function.arguments

                call_key = (name, tuple(sorted(arguments.items())))
                if call_key in called:
                    ui.console.print(
                        f"[yellow]\\[SKIP][/yellow]"
                        f" duplicate: {name}({arguments})"
                    )
                    session.messages.append({
                        "role": "tool",
                        "tool_name": name,
                        "content": (
                            f"Already called {name} with these"
                            " arguments. Use the previous result."
                        ),
                    })
                    continue
                called.add(call_key)

                result = tools.execute_tool(name, arguments)
                session.messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": result,
                })

    # --- summarisation -------------------------------------------------------

    def summarise(self, transcript: str) -> str:
        resp = ollama.chat(
            model=session.model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'Summarise the following conversation concisely.'
                        ' Keep facts, decisions, and any URLs or identifiers'
                        ' the user may refer to later. Output only the'
                        ' summary.'
                    ),
                },
                {'role': 'user', 'content': transcript},
            ],
            think=False,
            options={'num_ctx': session.num_ctx},
        )
        return (resp.message.content or '').strip() or '(summary unavailable)'
