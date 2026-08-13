"""Command-line entry point: adapter wiring, prompt loop, slash commands."""
import argparse
import atexit
import signal

from guru import config, session, ui
from guru.adapters.ollama import OllamaAdapter
from guru.domain import conversation, tools

# Configured provider adapters, built from ~/.guru/adapters.toml.
ADAPTERS: list = []


def _build_adapters() -> list:
    """Instantiate adapters from config. Unknown types are skipped."""
    built = []
    for cfg in config.load_adapter_configs():
        kind = cfg.get('type')
        name = cfg.get('name', kind or 'adapter')
        if kind == 'ollama':
            built.append(OllamaAdapter(
                name=name, url=cfg.get('url', 'http://localhost:11434')))
        elif kind == 'anthropic':
            # Added in Phase 2; skip for now so the app still starts.
            continue
    if not built:
        built.append(OllamaAdapter())
    return built


def _select_model(adapter: object, model_id: str) -> None:
    """Make (adapter, model_id) the active provider + model."""
    session.adapter = adapter
    adapter.activate(model_id)


def _models_command() -> None:
    """Cross-adapter model selector, grouped by adapter."""
    options: list = []
    selectable: list = []
    entries: list = []            # (adapter, ModelInfo) aligned with rows
    active_idx = -1

    for adapter in ADAPTERS:
        options.append(f"— {adapter.name} —")
        selectable.append(False)
        entries.append(None)
        if not adapter.available():
            options.append("  (unavailable)")
            selectable.append(False)
            entries.append(None)
            continue
        for info in adapter.list_models():
            row = f"{info.label}  ({info.context_window:,} ctx)"
            if (adapter is session.adapter
                    and info.model_id == session.model):
                active_idx = len(options)
            options.append(row)
            selectable.append(True)
            entries.append((adapter, info))

    if not any(selectable):
        ui.console.print("[yellow]No models available.[/yellow]")
        return

    idx = ui.pick(
        'Models  ↑/↓ navigate · Enter select · Esc cancel',
        options, active_idx, selectable,
    )
    if idx is None or entries[idx] is None:
        return
    adapter, info = entries[idx]
    _select_model(adapter, info.model_id)
    ui.console.print(
        f"\n[green]Model:[/green] [bold]{session.model}[/bold]"
        f" [dim]({adapter.name} · context {session.num_ctx:,})[/dim]"
    )


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
        "[italic]/search <query>[/italic] search · [italic]/models[/italic]"
        " model · [italic]/save[/italic] save · [italic]/resume[/italic]"
        " restore · [italic]/compact[/italic] shrink context.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="guru — local LLM agent")
    parser.add_argument("--model", default="qwen3-abliterated-32k:latest")
    parser.add_argument(
        "--num-ctx", type=int, default=0,
        help="Override the context window (0 = auto-detect from the model)",
    )
    args, _ = parser.parse_known_args()

    session.num_ctx_override = args.num_ctx
    global ADAPTERS
    ADAPTERS = _build_adapters()
    _select_model(ADAPTERS[0], args.model)

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
