"""Command-line entry point: adapter wiring, prompt loop, slash commands."""
import argparse
import atexit
import signal

from guru import config, session, ui
from guru.adapters.anthropic import AnthropicAdapter
from guru.adapters.litellm import LiteLLMAdapter
from guru.adapters.ollama import OllamaAdapter
from guru.domain import conversation, tools

# Configured provider adapters and their raw config dicts, kept parallel so
# /adapters can persist enable flags back to ~/.guru/adapters.toml.
ADAPTERS: list = []
ADAPTER_CONFIGS: list = []

DEFAULT_MODEL = "qwen3-abliterated-32k:latest"


def _instantiate(cfg: dict):
    """Build one adapter from a config dict, or None for unknown types."""
    kind = cfg.get('type')
    name = cfg.get('name', kind or 'adapter')
    if kind == 'ollama':
        return OllamaAdapter(
            name=name, url=cfg.get('url', 'http://localhost:11434'))
    if kind == 'anthropic':
        return AnthropicAdapter(
            name=name,
            auth=cfg.get('auth', 'api_key'),
            base_url=cfg.get('base_url'),
            api_key_env=cfg.get('api_key_env'),
            api_key=cfg.get('api_key'),
            profile=cfg.get('profile'),
            models=cfg.get('models'),
            thinking=cfg.get('thinking', True),
        )
    if kind == 'litellm':
        return LiteLLMAdapter(
            name=name,
            base_url=cfg.get('base_url'),
            api_key_env=cfg.get('api_key_env'),
            api_key=cfg.get('api_key'),
            models=cfg.get('models'),
        )
    return None


def _build_adapters() -> list:
    """Instantiate adapters from config, carrying their enable flag."""
    global ADAPTER_CONFIGS
    ADAPTER_CONFIGS = config.load_adapter_configs() or [
        {'name': 'Ollama', 'type': 'ollama',
         'url': 'http://localhost:11434'}]
    built = []
    for cfg in ADAPTER_CONFIGS:
        adapter = _instantiate(cfg)
        if adapter is None:
            continue
        adapter.enabled = bool(cfg.get('enable', True))
        built.append(adapter)
    if not built:
        built.append(OllamaAdapter())
    return built


def _enabled_adapters() -> list:
    return [a for a in ADAPTERS if a.enabled] or ADAPTERS


def _select_model(adapter: object, model_id: str) -> None:
    """Make (adapter, model_id) the active provider + model and persist it."""
    session.adapter = adapter
    adapter.activate(model_id)
    config.save_settings({'adapter': adapter.name, 'model': model_id})


def _restore_last(explicit_model) -> bool:
    """Restore the last-used adapter + model and ensure it is logged in.

    Skipped when --model was passed explicitly. Returns True if a saved
    selection was restored.
    """
    if explicit_model:
        return False
    saved = config.load_settings()
    name, model_id = saved.get('adapter'), saved.get('model')
    if not name or not model_id:
        return False
    adapter = next(
        (a for a in ADAPTERS if a.name == name and a.enabled), None)
    if adapter is None:
        return False
    _select_model(adapter, model_id)
    ok, msg = adapter.verify()
    style = 'dim' if ok else 'yellow'
    ui.console.print(
        f"[{style}]{adapter.name}: {msg or ('ok' if ok else 'not ready')}"
        f"[/{style}]"
    )
    return True


def _startup_select(default_model: str) -> None:
    """Pick a sensible active adapter+model at startup.

    Prefer an enabled Ollama adapter with the --model argument (preserves the
    original default); otherwise use the first enabled adapter's first model.
    """
    enabled = _enabled_adapters()
    for adapter in enabled:
        if isinstance(adapter, OllamaAdapter):
            _select_model(adapter, default_model)
            return
    adapter = enabled[0]
    models = adapter.list_models()
    _select_model(
        adapter, models[0].model_id if models else default_model)


def _models_command() -> None:
    """Cross-adapter model selector, grouped by adapter.

    Rows show context window and, for local (Ollama) models, the estimated
    memory footprint — coloured red when it exceeds 80% of system memory.
    """
    total_mem = ui.total_memory_bytes()
    mem_limit = total_mem * 0.8 if total_mem else 0

    options: list = []
    selectable: list = []
    row_styles: list = []
    entries: list = []            # (adapter, ModelInfo) aligned with rows
    active_idx = -1

    def _add(text: str, sel: bool, entry, style: str = '') -> None:
        options.append(text)
        selectable.append(sel)
        row_styles.append(style)
        entries.append(entry)

    for adapter in ADAPTERS:
        if not adapter.enabled:
            continue
        _add(f"— {adapter.name} —", False, None)
        # Ensure each enabled adapter is logged in / reachable before listing.
        ok, msg = adapter.verify()
        if not ok:
            _add(f"  (unavailable: {msg[:48]})", False, None)
            continue
        infos = adapter.list_models()
        if not infos:
            _add("  (no models listed)", False, None)
            continue
        for info in infos:
            row = f"{info.label}  ({info.context_window:,} ctx"
            warn = False
            if info.memory:
                row += f" · {ui.format_bytes(info.memory)}"
                warn = bool(mem_limit and info.memory > mem_limit)
            row += ")"
            if (adapter is session.adapter
                    and info.model_id == session.model):
                active_idx = len(options)
            _add(row, True, (adapter, info),
                 'class:warn' if warn else '')

    if not any(selectable):
        ui.console.print("[yellow]No models available.[/yellow]")
        return

    idx = ui.pick(
        'Models  ↑/↓ navigate · Enter select · Esc cancel',
        options, active_idx, selectable, row_styles,
    )
    if idx is None or entries[idx] is None:
        return
    adapter, info = entries[idx]
    _select_model(adapter, info.model_id)
    ui.console.print(
        f"\n[green]Model:[/green] [bold]{session.model}[/bold]"
        f" [dim]({adapter.name} · context {session.num_ctx:,})[/dim]"
    )


def _adapters_command() -> None:
    """Enable/disable adapters, persist, and verify the enabled ones.

    Space toggles, Enter confirms. On confirm the enable flags are written to
    adapters.toml, adapters are rebuilt, and each enabled adapter is verified
    — which triggers the one-time OAuth login for enterprise adapters.
    """
    global ADAPTERS
    if not ADAPTER_CONFIGS:
        ui.console.print("[yellow]No adapters configured.[/yellow]")
        return
    labels = [
        f"{c.get('name', c.get('type', 'adapter'))} [{c.get('type')}]"
        for c in ADAPTER_CONFIGS
    ]
    states = [bool(c.get('enable', True)) for c in ADAPTER_CONFIGS]

    new_states = ui.pick_multi('Adapters', labels, states)
    if new_states is None:
        return

    for cfg, enabled in zip(ADAPTER_CONFIGS, new_states):
        cfg['enable'] = enabled
    config.save_adapter_configs(ADAPTER_CONFIGS)
    ui.console.print(f"[green]Saved[/green] {config.ADAPTERS_PATH}")

    ADAPTERS = _build_adapters()
    for adapter in ADAPTERS:
        if not adapter.enabled:
            continue
        ui.console.print(f"[dim]Verifying {adapter.name}…[/dim]")
        ok, msg = adapter.verify()
        mark, colour = ('✓', 'green') if ok else ('✗', 'red')
        ui.console.print(f"[{colour}]{mark} {adapter.name}[/{colour}] {msg}")

    # Ensure the active adapter is still enabled; otherwise re-select.
    active = {a.name for a in ADAPTERS if a.enabled}
    if session.adapter is None or session.adapter.name not in active:
        _startup_select(session.model or '')


def _handle_slash_search(query: str) -> None:
    """Directly invoke web_search and optionally web_fetch for testing."""
    if not tools.ensure_domain_allowed(config.SEARCH_BACKEND_DOMAIN):
        ui.console.print(
            f"[red]Denied[/red] access to '{config.SEARCH_BACKEND_DOMAIN}';"
            " cannot search."
        )
        return
    tools.web_search(query)

    from ddgs import DDGS
    raw = list(DDGS().text(query, max_results=10))
    scored = sorted(
        raw, key=lambda r: tools._relevance_score(query, r), reverse=True)
    relevant = (
        [r for r in scored if tools._relevance_score(query, r) > 0][:5]
        or scored[:3]
    )
    urls = [r.get('href') for r in relevant if r.get('href')]
    if not urls:
        return

    ui.console.print("\n[bold]Fetch one of these URLs?[/bold]")
    for i, url in enumerate(urls, 1):
        ui.console.print(f"  [cyan]{i}[/cyan] {url}")
    ui.console.print(
        "  [dim]Enter a number to fetch, or press Enter to skip[/dim]"
    )
    choice = input("> ").strip()
    if choice.isdigit():
        i = int(choice) - 1
        if 0 <= i < len(urls):
            content = tools.web_fetch(urls[i])
            ui.console.print(
                "\n[bold green]--- Page content (first 2000 chars) ---"
                "[/bold green]"
            )
            ui.console.print(content[:2000])
            ui.console.print("[bold green]--- End ---[/bold green]")


def _print_banner() -> None:
    ui.console.print(f"Using model: [bold]{session.model}[/bold]")
    ui.console.print("[bold]guru[/bold]")
    ui.console.print("Type [italic]'exit'[/italic] to quit.")
    ui.console.print(
        "Enter to submit · Shift+Enter for newline"
        " · pasted newlines are safe.\n"
    )
    ui.console.print(
        "[italic]/search[/italic] search · [italic]/models[/italic] model ·"
        " [italic]/adapters[/italic] providers · [italic]/save[/italic] save ·"
        " [italic]/resume[/italic] restore · [italic]/compact[/italic]"
        " shrink context.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="guru — local LLM agent")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--num-ctx", type=int, default=0,
        help="Override the context window (0 = auto-detect from the model)",
    )
    args, _ = parser.parse_known_args()

    session.num_ctx_override = args.num_ctx
    global ADAPTERS
    ADAPTERS = _build_adapters()
    # Restore the last-used adapter+model (and log in); else pick a default.
    if not _restore_last(args.model):
        _startup_select(args.model or DEFAULT_MODEL)

    session.messages = [
        {"role": "system", "content": config.build_system_prompt()}]
    tools.reset_active_tools()

    signal.signal(signal.SIGINT, ui.sigint_handler)
    signal.signal(signal.SIGWINCH, ui.sigwinch_handler)
    atexit.register(ui.reset_terminal)

    _print_banner()

    while True:
        ui.refresh_git_branch()
        try:
            question = ui.read_line()
        except KeyboardInterrupt:
            # modifyOtherKeys Ctrl+C arrives via prompt_toolkit, not SIGINT.
            if ui.note_ctrl_c():
                ui.console.print("\n[bold red]Exiting.[/bold red]")
                break
            continue
        except EOFError:
            break

        ui.reset_terminal()

        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        if question in ('/models', '/model'):
            _models_command()
            continue
        if question == '/adapters':
            _adapters_command()
            continue
        if question == '/save':
            conversation.save_conversation()
            continue
        if question == '/resume':
            conversation.resume_command()
            continue
        if question == '/compact':
            conversation.compact_messages(force=True)
            session.ctx_used = conversation.estimate_tokens(session.messages)
            ui.console.print(
                f"[green]Compacted[/green] · ~{session.ctx_used:,} tokens"
                f" of {session.num_ctx:,}."
            )
            continue
        if question.startswith("/search "):
            _handle_slash_search(question[8:].strip())
            continue

        ui.console.print()
        ui.console.print(f" {question} ", style="bold white on grey23")
        ui.console.print()

        checkpoint = len(session.messages)
        session.messages.append({"role": "user", "content": question})

        ui.status_enable()
        try:
            session.adapter.run_turn()
        except KeyboardInterrupt:
            ui.console.print(
                "\n[yellow]\\[CANCELLED][/yellow] Response cancelled."
            )
            del session.messages[checkpoint:]
            ui.reset_terminal()
        finally:
            ui.status_disable()

        # Proactive compaction once the turn's tool calls are all resolved.
        if session.num_ctx and (
                session.ctx_used > config.COMPACT_AT * session.num_ctx):
            ui.console.print(
                f"[dim]\\[COMPACT] context {session.ctx_used:,}"
                f"/{session.num_ctx:,} over"
                f" {int(config.COMPACT_AT * 100)}% — compacting[/dim]"
            )
            conversation.compact_messages()
            session.ctx_used = conversation.estimate_tokens(session.messages)


if __name__ == '__main__':
    main()
