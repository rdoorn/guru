"""Ollama provider adapter.

Wraps the existing local-Ollama behaviour behind the Adapter interface:
model listing, context-window resolution, the tool-calling turn loop, and a
daemon-reachable check with an on-demand model pull (moved from start.sh).
"""
import platform
import subprocess

import ollama
from rich.markdown import Markdown

from guru import config, session, ui
from guru.adapters.base import Adapter, ModelInfo
from guru.domain import tools

# Smallest context to fall back to before giving up on fitting into memory.
_CTX_FLOOR = 2048


class OllamaAdapter(Adapter):
    """Local models served by the Ollama daemon."""

    def __init__(self, name: str = "Ollama",
                 url: str = "http://localhost:11434") -> None:
        self.name = name
        self.url = url
        self._fitted: set = set()   # models already fitted to memory
        self._thinks: dict = {}     # model -> supports thinking

    # --- discovery -----------------------------------------------------------

    def available(self) -> bool:
        try:
            ollama.list()
            return True
        except Exception:
            return False

    def verify(self) -> tuple:
        if self.available():
            return (True, "daemon reachable")
        return (False, "Ollama daemon not reachable — launch the Ollama app")

    def list_models(self) -> list:
        try:
            models = ollama.list().models
        except Exception:
            return []
        infos = []
        for m in sorted(models, key=lambda x: x.model):
            num_ctx, ceiling = self._resolve_context_window(m.model)
            infos.append(ModelInfo(
                adapter=self.name,
                model_id=m.model,
                label=m.model,
                # Report the model's architecture max so /models and /context
                # agree; the running context is shown in the status bar.
                context_window=ceiling or num_ctx,
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
        num_ctx, ceiling = self._resolve_context_window(model_id)
        session.ctx_ceiling = ceiling
        # First-selection default only: apply the stored per-model choice or a
        # fresh GPU auto-fit when the user gave no explicit --num-ctx. A manual
        # /context or --num-ctx always wins; the reported max is untouched.
        if not session.num_ctx_override:
            num_ctx = self._default_ctx(model_id, num_ctx, ceiling)
        session.num_ctx = num_ctx
        session.model_size = self._param_size(model_id)
        self._preload_and_fit()
        config.save_model_ctx(model_id, session.num_ctx)

    def _default_ctx(self, model: str, resolved: int, ceiling: int) -> int:
        """First-selection default context: a stored choice from a prior
        session wins; otherwise the measured GPU fit; otherwise the metadata
        estimate; otherwise the resolved default. Never exceeds the ceiling."""
        stored = config.load_model_ctx().get(model)
        if stored:
            return min(int(stored), ceiling or int(stored))
        ui.console.print(
            f"[dim]Fitting {model} context to the GPU (one-time)…[/dim]")
        ctx = self._calibrated_ctx(model, ceiling)
        if not ctx:
            ctx = self._max_gpu_ctx(model, ceiling)   # metadata fallback
        return ctx or resolved

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

    def _supports_thinking(self, model: str) -> bool:
        """Whether a model supports think mode (per its capabilities)."""
        if model not in self._thinks:
            try:
                caps = ollama.show(model).capabilities or []
                self._thinks[model] = 'thinking' in caps
            except Exception:
                self._thinks[model] = False
        return self._thinks[model]

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

    # --- GPU auto-fit (first-selection default) ------------------------------

    def _ps_stats(self) -> tuple:
        """(total footprint, bytes resident in VRAM) for the current model."""
        try:
            for m in ollama.ps().models:
                if m.model == session.model:
                    return (int(getattr(m, 'size', 0) or 0),
                            int(getattr(m, 'size_vram', 0) or 0))
        except Exception:                                # noqa: BLE001
            pass
        return (0, 0)

    def _measure_at(self, ctx: int) -> tuple:
        """Load the model at ``ctx`` and return its (size, size_vram)."""
        if not self._reload(ctx):
            return (0, 0)
        return self._ps_stats()

    def _calibrated_ctx(self, model: str, ceiling: int) -> int:
        """Largest context that stays 100% on GPU, measured from Ollama.

        Two probe loads give the footprint as a line in the context length
        (``size = weights + ctx * kv_per_token``); a spill reveals the real GPU
        budget (``size_vram``). This is accurate regardless of the KV cache
        type (f16/q8_0) because the per-token cost is measured, not assumed.
        Returns 0 if measurement is unavailable (fall back to the estimate).
        """
        lo = _CTX_FLOOR
        hi = min(ceiling or config.CTX_PROBE_HIGH, config.CTX_PROBE_HIGH)
        if hi <= lo:
            return 0
        size_lo, vram_lo = self._measure_at(lo)
        size_hi, vram_hi = self._measure_at(hi)
        if size_lo <= 0 or size_hi <= 0:
            return 0
        kv = (size_hi - size_lo) / (hi - lo)
        if kv <= 0:
            return min(hi, ceiling or hi)            # fit at both probes
        weights = size_lo - lo * kv
        self._report_kv_type(model, kv)
        spill = vram_hi if (vram_hi and vram_hi < size_hi) else (
            vram_lo if (vram_lo and vram_lo < size_lo) else 0)
        if spill:
            # A spill measured the real GPU budget directly.
            avail = spill * config.GPU_FIT_SAFETY - weights
        else:
            # Both probes fit — extend toward the ceiling using the platform
            # budget estimate against the MEASURED weights/kv (only the budget
            # is estimated now, not the per-token cost). Never drop below hi,
            # which is known to fit; the load-time safety net corrects any
            # over-estimate by scaling down on a real spill.
            total = self._total_gpu_bytes()
            if total <= 0:
                return min(hi, ceiling or hi)
            avail = (total * (1 - config.GPU_MEM_HEADROOM)
                     - weights - config.FIT_OVERHEAD_BYTES)
            if avail <= 0:
                return min(hi, ceiling or hi)
        ctx = int(avail // kv)
        ctx = (ctx // 1024) * 1024
        if not spill:
            ctx = max(ctx, hi)                       # hi is known to fit
        ctx = max(_CTX_FLOOR, ctx)
        if ceiling:
            ctx = min(ctx, ceiling)
        return ctx

    def _report_kv_type(self, model: str, measured_kv: float) -> None:
        """Log the KV cache type inferred from the measured per-token cost vs
        the f16 value from metadata — the only way to confirm it, since Ollama
        does not expose the setting over its API. Only claims a type when the
        ratio lands near a known quant level; some architectures (sliding
        window / MoE, e.g. gpt-oss) don't match the simple f16 formula, so it
        reports the measured cost without guessing rather than misleading."""
        meta = self._kv_bytes_per_token(model)
        if not meta:
            return
        ratio = measured_kv / meta
        kind = None
        if 0.85 <= ratio <= 1.2:
            kind = 'f16'
        elif 0.4 <= ratio <= 0.65:
            kind = 'q8_0'
        elif 0.18 <= ratio <= 0.32:
            kind = 'q4_0'
        kb = measured_kv / 1024
        if kind:
            ui.console.print(
                f"[dim]KV cache ~{kb:.0f} KB/token — {kind}"
                f" (f16 would be ~{meta / 1024:.0f} KB).[/dim]")
        else:
            ui.console.print(
                f"[dim]KV cache ~{kb:.1f} KB/token (measured; type"
                f" indeterminate for this model's attention).[/dim]")

    def _total_gpu_bytes(self) -> int:
        """Best-effort total GPU memory in bytes (0 if it can't be found)."""
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=3)
            if out.returncode == 0 and out.stdout.strip():
                mib = int(out.stdout.strip().splitlines()[0])
                return mib * 1024 * 1024
        except Exception:                                # noqa: BLE001
            pass
        # Apple Silicon: unified memory; the GPU working set is a fraction.
        if platform.system() == 'Darwin':
            try:
                out = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True, text=True, timeout=3)
                if out.returncode == 0 and out.stdout.strip():
                    ram = int(out.stdout.strip())
                    return int(ram * config.MAC_GPU_FRACTION)
            except Exception:                            # noqa: BLE001
                pass
        return 0

    def _kv_bytes_per_token(self, model: str) -> int:
        """KV-cache bytes per token from model metadata (0 if unknown)."""
        try:
            mi = ollama.show(model).modelinfo or {}
        except Exception:                                # noqa: BLE001
            return 0
        arch = mi.get('general.architecture', '')

        def _int(key: str) -> int:
            return int(mi.get(f'{arch}.{key}', 0) or 0)

        layers = _int('block_count')
        heads = _int('attention.head_count')
        kv_heads = _int('attention.head_count_kv') or heads
        emb = _int('embedding_length')
        head_dim = _int('attention.key_length') or (
            emb // heads if heads else 0)
        if not (layers and kv_heads and head_dim):
            return 0
        # 2 = one tensor each for keys and values.
        return int(2 * layers * kv_heads * head_dim * config.KV_CACHE_BYTES)

    def _weight_bytes(self, model: str) -> int:
        """On-disk weight size (proxy for VRAM weights); 0 if unknown."""
        try:
            for m in ollama.list().models:
                if m.model == model:
                    return int(getattr(m, 'size', 0) or 0)
        except Exception:                                # noqa: BLE001
            pass
        return 0

    def _max_gpu_ctx(self, model: str, ceiling: int) -> int:
        """Largest num_ctx whose weights + KV cache stay within the GPU budget
        (minus headroom). 0 when the budget or metadata is unavailable, so the
        caller can fall back to the plain default."""
        total = self._total_gpu_bytes()
        kv = self._kv_bytes_per_token(model)
        if total <= 0 or kv <= 0:
            return 0
        budget = total * (1 - config.GPU_MEM_HEADROOM)
        avail = budget - self._weight_bytes(model) - config.FIT_OVERHEAD_BYTES
        if avail <= 0:
            return 0            # weights barely fit — let the caller default
        ctx = int(avail // kv)
        ctx = (ctx // 1024) * 1024                      # multiple of 1024
        ctx = max(_CTX_FLOOR, ctx)
        if ceiling:
            ctx = min(ctx, ceiling)
        return ctx

    # --- fit context to memory ----------------------------------------------

    def mark_fitted(self) -> None:
        """Mark the current model's context as user-chosen (skips auto-fit)."""
        if session.model:
            self._fitted.add(session.model)

    def _reload(self, num_ctx: int) -> bool:
        """Load the model at a context (no generation); False on error."""
        try:
            # Empty-prompt generate loads the model into memory and returns.
            ollama.generate(
                model=session.model,
                prompt='',
                options={'num_ctx': num_ctx},
                keep_alive='30m',
            )
            return True
        except Exception:
            return False

    def _preload_and_fit(self) -> None:
        """Load the model now (so the first answer is fast) and fit context."""
        if not self.available():
            return
        ui.console.print(
            f"[dim]Loading {session.model} (context"
            f" {session.num_ctx:,})…[/dim]")
        if not self._reload(session.num_ctx):
            ui.console.print(
                f"[yellow]Could not preload {session.model}; it will load on"
                f" the first question.[/yellow]")
            return
        self._fit_after_load()
        ui.console.print(f"[dim]{session.model} ready.[/dim]")

    def _gpu_fits(self) -> bool:
        """True if the loaded model sits entirely in VRAM (no CPU spill)."""
        try:
            for m in ollama.ps().models:
                if m.model == session.model:
                    total = getattr(m, 'size', 0) or 0
                    vram = getattr(m, 'size_vram', 0) or 0
                    return total <= 0 or vram >= total
        except Exception:
            pass
        return True

    def _fit_after_load(self) -> None:
        """Once the model is loaded, scale context down if it spilled to CPU.

        Runs after a real turn has loaded the model. Ollama spills to CPU
        (slow) when weights + KV cache exceed VRAM; halve the context until it
        fits or we hit the floor, warning on each step. Once per model.
        """
        if session.model in self._fitted:
            return
        self._fitted.add(session.model)
        if self._gpu_fits():
            return
        ceiling = session.ctx_ceiling or session.num_ctx
        num_ctx = session.num_ctx
        while num_ctx > _CTX_FLOOR:
            smaller = max(_CTX_FLOOR, num_ctx // 2)
            ui.console.print(
                f"[yellow]{session.model} spilled to CPU at context"
                f" {num_ctx:,}; scaling down to {smaller:,}"
                f" (model max: {ceiling:,}).[/yellow]")
            session.num_ctx = smaller
            num_ctx = smaller
            if self._reload(smaller) and self._gpu_fits():
                return
        ui.console.print(
            f"[red]{session.model} still spills to CPU at the minimum context"
            f" {num_ctx:,} — too large for this machine's memory. Consider a"
            f" smaller model or quant.[/red]")

    # --- turn loop -----------------------------------------------------------

    def _collect_response(self):
        """Stream a chat response, checking cancel between chunks so a running
        generation can be interrupted (Ctrl+C sets cancel_requested). Returns
        the assembled assistant Message, or None if cancelled mid-stream. The
        stream is closed on cancel, which aborts generation server-side."""
        content: list = []
        tool_calls: list = []
        prompt_ct = eval_ct = 0
        stream = ollama.chat(
            model=session.model,
            messages=session.messages,
            think=self._supports_thinking(session.model),
            tools=session.active_tools,
            options={'num_ctx': session.num_ctx},
            stream=True,
        )
        for chunk in stream:
            if session.cancel_requested:
                try:
                    stream.close()
                except Exception:                        # noqa: BLE001
                    pass
                return None
            m = getattr(chunk, 'message', None)
            if m is not None:
                if getattr(m, 'content', None):
                    content.append(m.content)
                if getattr(m, 'tool_calls', None):
                    tool_calls.extend(m.tool_calls)
            prompt_ct = getattr(chunk, 'prompt_eval_count', 0) or prompt_ct
            eval_ct = getattr(chunk, 'eval_count', 0) or eval_ct
        session.session_in += prompt_ct
        session.session_out += eval_ct
        if prompt_ct:
            session.ctx_used = prompt_ct
        return ollama.Message(
            role='assistant',
            content=''.join(content),
            tool_calls=tool_calls or None,
        )

    def run_turn(self) -> None:
        session.cancel_requested = False
        called: set = set()
        nudged = 0
        while True:
            if session.cancel_requested:
                ui.console.print("[yellow]* cancelled[/yellow]")
                return
            ui.note_thinking()
            msg = self._collect_response()
            if msg is None:                  # cancelled mid-generation
                ui.console.print("[yellow]* cancelled[/yellow]")
                return
            # The model is now loaded — check for CPU spill and scale down.
            self._fit_after_load()
            ui.status_draw()

            ui.debug(
                f"content={msg.content!r} tool_calls={msg.tool_calls}")

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
                ui.console.print("\n[bold green]answer>[/bold green]")
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
