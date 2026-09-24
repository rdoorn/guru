"""Command-line entry point: adapter wiring, model selection, slash commands.

The interactive UI lives in ``guru.tui`` (hybrid: main agent in the normal
buffer, sub-agents in a full-screen viewer). The slash-command helpers here are
reused by that UI.
"""
import argparse
from pathlib import Path
from typing import Optional

from guru import config, judges, session, ui
from guru.adapters.base import Adapter
from guru.adapters.anthropic import AnthropicAdapter
from guru.adapters.litellm import LiteLLMAdapter
from guru.adapters.ollama import OllamaAdapter
from guru.domain import ledger, policy, tools
from guru.repositories import settings as routing_settings
from guru.repositories.adapters import AdapterRegistry, registry_from
from guru.repositories.jsonl_ledger import JsonlLedger
from guru.scanners.secrets import load_project_scanner

# Configured provider adapters and their raw config dicts, kept parallel so
# /adapters can persist enable flags back to ~/.guru/adapters.toml.
ADAPTERS: list = []
ADAPTER_CONFIGS: list = []
# Name -> Adapter view of ADAPTERS for the routing layer (rebuilt with it).
REGISTRY = AdapterRegistry()

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


def build_registry(adapters: list) -> AdapterRegistry:
    """The AdapterRegistry over ``adapters`` (all of them, enabled or not,
    so a ladder rung on a disabled adapter is reported rather than unknown).
    """
    return registry_from(adapters)


def load_routing() -> routing_settings.RoutingSettings:
    """The validated ``[routing]`` table, wired into the process.

    An invalid table warns and yields the defaults. Only when a table is
    ``present`` does ``config.SECRET_SCAN`` mirror ``secret_scan`` and the
    project secret scanner get bound; without one both stay off, so guru
    behaves exactly as before the routing framework.
    """
    from guru import log
    try:
        routing = routing_settings.load_routing()
    except ValueError as e:
        log.warning('%s; using routing defaults', e)
        ui.console.print(f"[yellow]{e}; using routing defaults.[/yellow]")
        routing = routing_settings.RoutingSettings()
    scan = routing.present and routing.secret_scan
    config.SECRET_SCAN = scan
    policy.set_scanner(load_project_scanner() if scan else None)
    return routing


def load_tools_policy() -> tools.ToolsPolicy:
    """The project's ``.guru/tools.toml`` policy.

    An absent file is the default (everything enabled). A file that is
    present but invalid or unreadable FAILS CLOSED: every registry tool is
    disabled (the always-on tools stay) and a warning names the file, so a
    typo in a policy meant to restrict tools never widens them.
    """
    from guru import log
    try:
        return routing_settings.load_tools_policy()
    except ValueError as e:
        log.warning('%s; failing closed: every registry tool disabled', e)
        ui.console.print(
            f"[yellow]{e}; every registry tool is disabled until the file "
            "is fixed or removed.[/yellow]")
        return tools.ToolsPolicy(enabled=set(),
                                 disabled=set(tools.TOOL_REGISTRY))


def _enabled_adapters() -> list:
    return [a for a in ADAPTERS if a.enabled] or ADAPTERS


def _select_model(adapter: Adapter, model_id: str) -> None:
    """Make (adapter, model_id) the active provider + model and persist it."""
    session.adapter = adapter
    adapter.activate(model_id)
    config.save_settings({'adapter': adapter.name, 'model': model_id})


def _restore_last(explicit_model) -> bool:
    """Restore the last-used adapter + model, if it still exists.

    Skipped when --model was passed explicitly. Warns and returns False (so
    the caller falls back) if the adapter is gone/unavailable or the saved
    model no longer exists on it — never selects a stale model.
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
        ui.console.print(
            f"[yellow]Last adapter '{name}' is no longer configured;"
            f" selecting another model.[/yellow]")
        return False
    ok, msg = adapter.verify()
    if not ok:
        ui.console.print(
            f"[yellow]Last adapter '{name}' is unavailable ({msg});"
            f" selecting another model.[/yellow]")
        return False
    available = {m.model_id for m in adapter.list_models()}
    if model_id not in available:
        ui.console.print(
            f"[yellow]Last model '{model_id}' no longer exists on"
            f" '{name}'; selecting another model.[/yellow]")
        return False
    _select_model(adapter, model_id)
    return True


def _startup_select(explicit_model) -> None:
    """Pick an active adapter+model: honour --model, else first available.

    With no --model, picks the first available model of the first enabled
    adapter (Ollama first), preferring the default model when it's installed.
    Only an explicit --model is trusted verbatim (Ollama may pull it).
    """
    enabled = _enabled_adapters()
    if explicit_model:
        target = next(
            (a for a in enabled if isinstance(a, OllamaAdapter)), enabled[0])
        _select_model(target, explicit_model)
        return
    ordered = sorted(
        enabled, key=lambda a: 0 if isinstance(a, OllamaAdapter) else 1)
    for adapter in ordered:
        model_ids = [m.model_id for m in adapter.list_models()]
        if not model_ids:
            continue
        default_ok = (
            isinstance(adapter, OllamaAdapter)
            and DEFAULT_MODEL in model_ids)
        pick = DEFAULT_MODEL if default_ok else model_ids[0]
        _select_model(adapter, pick)
        return
    _select_model(enabled[0], DEFAULT_MODEL)


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
    global ADAPTERS, REGISTRY
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
    REGISTRY = build_registry(ADAPTERS)
    judges.set_registry(REGISTRY)
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


def _human_ctx(n: int) -> str:
    """Format a context size, e.g. 65536 -> '64k'."""
    if n % (1024 * 1024) == 0:
        return f"{n // (1024 * 1024)}M"
    if n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def _context_command() -> None:
    """Pick a context window in halves of the model's max, down to 4k."""
    ceiling = session.ctx_ceiling or session.num_ctx or 4096
    options: list = []
    n = ceiling
    while n >= 4096:
        options.append(n)
        if n == 4096:
            break
        n = max(4096, n // 2)
    labels = [f"{_human_ctx(v)}  ({v:,})" for v in options]
    active_idx = options.index(session.num_ctx) \
        if session.num_ctx in options else -1
    idx = ui.pick(
        'Context  ↑/↓ navigate · Enter select · Esc cancel',
        labels, active_idx)
    if idx is None:
        return
    session.num_ctx = options[idx]
    if isinstance(session.adapter, OllamaAdapter):
        # Respect the manual choice; don't let auto-fit override it.
        session.adapter.mark_fitted()
    config.save_model_ctx(session.model, session.num_ctx)
    ui.console.print(
        f"[green]Context[/green] set to {session.num_ctx:,}"
        f" (applies on the next turn)."
    )


def _label_command(label: str, note: str = '') -> None:
    """``/good [note]`` and ``/bad [note]``: label the last completed turn.

    Labels ``session.turn_id`` of the bound (main) session — the id of the
    turn that last ran, since ``run_loop`` assigns it at turn start and
    leaves it in place — plus every task row of this run with that
    ``turn_id`` when the repository can read rows back. Labeller ``user``.
    """
    repo = ledger.repository()
    if repo is None or not config.LEDGER_ENABLED:
        ui.console.print('[yellow]No ledger repository; nothing labelled.'
                         '[/yellow]')
        return
    turn_id = session.turn_id
    if not turn_id:
        ui.console.print('[yellow]No completed turn to label yet.[/yellow]')
        return
    targets = [turn_id]
    rows_fn = getattr(repo, 'rows', None)
    if callable(rows_fn):
        ledger.flush()               # finish rows are fire-and-forget
        seen: dict = {}
        for r in rows_fn('tasks', run_id=ledger.RUN_ID):
            if r.get('turn_id') == turn_id and r.get('task_id'):
                seen[r['task_id']] = True
        targets += list(seen)
    for target in targets:
        ledger.record_label(target, 'user', label, note)
    colour = 'green' if label == 'good' else 'red'
    ui.console.print(
        f"[{colour}]{label}[/{colour}] -> turn {turn_id}"
        f" (+{len(targets) - 1} tasks)"
        + (f": {note}" if note else ''))


def _format_run_summary(summary: dict) -> str:
    """Plain-text rendering of :func:`ledger.run_summary` for ``/ledger``."""
    def money(v: Optional[float]) -> str:
        return '$?' if v is None else f'${v:.4f}'

    lines = [f"run {summary['run_id']}", '', 'calls per model']
    if not summary['models']:
        lines.append('  (none)')
    for key, m in sorted(summary['models'].items(),
                         key=lambda kv: -kv[1]['calls']):
        lines.append(f"  {key:<40} {m['calls']:>4} calls"
                     f"  in {m['tokens_in']:>8}  out {m['tokens_out']:>8}"
                     f"  cache r/w {m['cache_read']}/{m['cache_write']}"
                     f"  {money(m['cost_usd'])}")
    lines += ['', 'tasks per model']
    if not summary['tasks']:
        lines.append('  (none)')
    for key, n in sorted(summary['tasks'].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {key:<40} {n:>4}")
    lines += ['', 'most expensive tasks']
    if not summary['top_tasks']:
        lines.append('  (none)')
    for t in summary['top_tasks']:
        who = t['role'] or t['kind'] or 'task'
        secs = '?' if t['seconds'] is None else f"{t['seconds']:.1f}s"
        lines.append(f"  {money(t['cost_usd']):>9} {secs:>7}  {who}"
                     f" [{t['adapter']}|{t['model']}] {t['status']}"
                     f"  {t['task']}")
    tot = summary['totals']
    lines += ['', f"total: {tot['calls']} calls, in {tot['tokens_in']},"
                  f" out {tot['tokens_out']}, cache r/w {tot['cache_read']}"
                  f"/{tot['cache_write']}, {tot['tasks']} tasks,"
                  f" {money(tot['cost_usd'])}"]
    return '\n'.join(lines)


def _ledger_command() -> None:
    """``/ledger``: print this run's spend from the installed repository."""
    repo = ledger.repository()
    rows_fn = getattr(repo, 'rows', None)
    if repo is None or not callable(rows_fn) or not config.LEDGER_ENABLED:
        ui.console.print('[yellow]No readable ledger repository.[/yellow]')
        return
    ledger.flush()                       # queued rows land before we read
    summary = ledger.run_summary(rows_fn('calls', run_id=ledger.RUN_ID),
                                 rows_fn('tasks', run_id=ledger.RUN_ID),
                                 ledger.RUN_ID)
    ui.console.print(_format_run_summary(summary), markup=False,
                     highlight=False)


_ARGS_COL = 44                       # width of the args column in /tools


def _args_head(args: object) -> str:
    """``k=v`` pairs of a tool_events ``args`` dict, cut to the column."""
    if not isinstance(args, dict):
        return str(args or '')[:_ARGS_COL]
    text = ' '.join(f"{k}={' '.join(str(v).split())}" for k, v in args.items())
    return text if len(text) <= _ARGS_COL else text[:_ARGS_COL - 1] + '…'


def _format_tool_events(rows: list) -> str:
    """Plain-text table of tool_events rows for ``/tools``: tool, args
    head, seconds, bytes shown/produced and the denial (if any)."""
    header = f"{'tool':<22} {'args':<{_ARGS_COL}} {'secs':>7}  " \
             f"{'shown/produced':>16}  denied"
    lines = [header]
    for r in rows:
        secs = r.get('seconds')
        ratio = f"{r.get('shown_bytes') or 0}/{r.get('produced_bytes') or 0}"
        lines.append(
            f"{str(r.get('tool') or '?'):<22} "
            f"{_args_head(r.get('args')):<{_ARGS_COL}} "
            f"{('?' if secs is None else f'{float(secs):.2f}'):>7}  "
            f"{ratio:>16}  {r.get('denied') or ''}".rstrip())
    return '\n'.join(lines)


def _tools_command() -> None:
    """``/tools``: print the last turn's tool calls from the audit stream."""
    repo = ledger.repository()
    rows_fn = getattr(repo, 'rows', None)
    if repo is None or not callable(rows_fn) or not config.LEDGER_ENABLED:
        ui.console.print('[yellow]No readable ledger repository.[/yellow]')
        return
    turn_id = session.turn_id
    ledger.flush()                       # queued rows land before we read
    rows = [r for r in rows_fn('tool_events', run_id=ledger.RUN_ID)
            if turn_id and r.get('turn_id') == turn_id]
    if not rows:
        ui.console.print('[dim]No tool calls in the last turn.[/dim]')
        return
    ui.console.print(_format_tool_events(rows), markup=False,
                     highlight=False)


def _sandbox_status(project: Optional[Path] = None) -> str:
    """Plain-text ``/sandbox status``: runtime availability, the project's
    sandbox settings/spec, and the recorded image (tag, digest, built_at,
    whether a build is needed). Never raises: each broken part is one
    line."""
    from guru.domain import sandbox as sb
    from guru.repositories import sandbox_images as images
    from guru.repositories.settings import load_sandbox
    from guru.sandbox import colima
    root = Path(project) if project is not None \
        else config.PROJECT_GURU_DIR.parent
    lines = ['sandbox status']
    lines.append('runtime: docker '
                 + ('available' if colima.available(root) else
                    'unavailable (docker info failed; is Colima running?)'))
    policy_path = config.SANDBOX_POLICY_PATH
    try:
        settings = load_sandbox()
    except ValueError as e:
        lines.append(f'settings: invalid: {e}')
        return '\n'.join(lines)
    lines.append(f"project file: {policy_path} "
                 f"{'present' if policy_path.is_file() else 'absent'}; "
                 f"enabled: {'yes' if settings.enabled else 'no'}")
    try:
        spec = sb.spec_from(root, settings)
    except ValueError as e:
        lines.append(f'spec: none ({e})')
        return '\n'.join(lines)
    lines.append(f'spec: {sb.spec_summary(spec)}')
    lines.append(f'base image: {spec.base_image}')
    try:
        dockerfile = sb.dockerfile_for(spec.project, spec.base_image)
    except ValueError as e:
        lines.append(f'dockerfile: cannot generate ({e})')
        return '\n'.join(lines)
    rec = images.load_record(spec)
    if rec is None:
        lines.append('image: not built (records in '
                     f'{images.record_dir(spec)})')
    else:
        lines.append(f'image: {rec.tag} digest {rec.digest} '
                     f'built {rec.built_at}')
    lines.append('needs build: '
                 + ('yes' if images.needs_build(spec, dockerfile) else 'no'))
    pending = images.pending_requests(spec)
    lines.append('pending dependency requests: '
                 + (', '.join(r.spec for r in pending) if pending
                    else 'none'))
    lines.extend(_sandbox_copies(spec))
    return '\n'.join(lines)


def _sandbox_copies(spec) -> list:
    """``task copies`` lines: the live per-task working copies of this
    process, then any other copy directory left under the work root."""
    from guru.repositories import sandbox_images as images
    from guru.sandbox import colima, verbs
    live = {path.resolve(): key for key, path in verbs.copies().items()
            if key[0] == str(spec.project)}
    rows = [f'  {path} (task {key[1]})' for path, key in live.items()]
    root = images.work_root(spec)
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if (child.resolve() not in live
                    and (child / colima.COPY_MARKER).is_file()):
                rows.append(f'  {child} (stale; safe to delete)')
    return ['task copies: ' + ('none' if not rows else '')] + rows


_GATE_ROWS = 10


def _sandbox_gate() -> str:
    """Plain-text ``/sandbox gate``: this run's last submit verdicts
    (``sandbox_events`` kind ``submit``) and the reviewer's ``decisions``
    rows for the ``gate`` point."""
    repo = ledger.repository()
    rows_fn = getattr(repo, 'rows', None)
    if repo is None or not callable(rows_fn) or not config.LEDGER_ENABLED:
        return 'sandbox gate: no readable ledger repository'
    ledger.flush()
    submits = [r for r in rows_fn('sandbox_events', run_id=ledger.RUN_ID)
               if r.get('kind') == 'submit'][-_GATE_ROWS:]
    reviews = [r for r in rows_fn('decisions', run_id=ledger.RUN_ID)
               if r.get('point') == 'gate'][-_GATE_ROWS:]
    lines = ['gate verdicts (last submits):']
    if not submits:
        lines.append('  none this run')
    for r in submits:
        argv = r.get('argv') or []
        intent = argv[1] if len(argv) > 1 else ''
        lines.append(f"  {r.get('ts', '?')} agent {r.get('agent', '?')}: "
                     f"{r.get('detail', '')}  [intent: {intent}]")
    lines.append('reviewer rows (decisions/gate):')
    if not reviews:
        lines.append('  none this run')
    for r in reviews:
        dist = r.get('dist') or {}
        answers = ', '.join(f'{k}={v}' for k, v in dist.items()
                            if k != 'notes')
        lines.append(f"  {r.get('ts', '?')} {r.get('judge', '?')}: "
                     f"used={r.get('used')} chosen={r.get('chosen')}"
                     + (f" fallback={r['fallback_reason']}"
                        if r.get('fallback_reason') else '')
                     + (f' ms={r["ms"]}' if r.get('ms') is not None else '')
                     + (f'  {answers}' if answers else '')
                     + (f"  error={r['error']}" if r.get('error') else ''))
    return '\n'.join(lines)


_SANDBOX_USAGE = ('usage: /sandbox status | provision [--force] | gate | deps '
                  '| deps request <name>[<constraint>] | deps apply <name>')


def _sandbox_provision(force: bool = False) -> str:
    """Plain-text ``/sandbox provision``: build the project's sandbox image
    through the provisioning proxy (or confirm the recorded one)."""
    from guru.repositories.settings import load_sandbox
    from guru.sandbox import provision
    root = config.PROJECT_GURU_DIR.parent
    try:
        settings = load_sandbox()
        rec = provision.provision(root, settings, force=force)
    except (ValueError, provision.ProvisionError) as e:
        return f'sandbox provision failed: {e}'
    return (f'sandbox image {rec.tag} digest {rec.digest} built '
            f'{rec.built_at} (lockfile {rec.lockfile_sha[:12]})')


def _split_requirement(text: str) -> tuple:
    """``('six', '>=1.16')`` from ``six>=1.16``: the name ends at the first
    version operator character."""
    spec = text.strip().strip('"\'')
    for i, ch in enumerate(spec):
        if ch in '=<>!~':
            return spec[:i], spec[i:]
    return spec, ''


def _sandbox_deps(args: str) -> str:
    """Plain-text ``/sandbox deps [request <spec> | apply <name>]``."""
    from guru.domain import sandbox as sb
    from guru.repositories import sandbox_images as images
    from guru.repositories.settings import load_sandbox
    from guru.sandbox import provision
    root = config.PROJECT_GURU_DIR.parent
    words = args.split(None, 1)
    verb = words[0] if words else ''
    rest = words[1].strip() if len(words) > 1 else ''
    try:
        settings = load_sandbox()
        spec = sb.spec_from(root, settings)
        if verb == '':
            pending = images.pending_requests(spec)
            if not pending:
                return 'sandbox deps: no pending dependency requests'
            return 'pending dependency requests:\n' + '\n'.join(
                f'  {r.spec}  (requested {r.requested_at}; apply with '
                f'/sandbox deps apply {r.name})' for r in pending)
        if verb == 'request' and rest:
            name, constraint = _split_requirement(rest)
            return provision.request_dependency(name, constraint,
                                                project=root,
                                                settings=settings)
        if verb == 'apply' and rest:
            from guru.domain import deps
            key = deps.normalise(rest)
            match = [r for r in images.pending_requests(spec)
                     if r.key == key]
            if not match:
                return f"sandbox deps: no pending request named '{rest}'"
            return provision.apply_dependency(root, match[0], settings)
    except (ValueError, OSError, RuntimeError,
            provision.ProvisionError) as e:
        # One line, never a traceback: bad settings/lockfile, an
        # unreadable store, a copy or docker failure.
        return f'sandbox deps: {verb or "list"} failed: {e}'
    return _SANDBOX_USAGE


def _sandbox_command(args: str = '') -> None:
    """``/sandbox status | provision [--force] | gate | deps …``."""
    words = (args or '').split()
    sub = words[0] if words else 'status'
    rest = ' '.join(words[1:])
    if sub == 'status' and not rest:
        text = _sandbox_status()
    elif sub == 'gate' and not rest:
        text = _sandbox_gate()
    elif sub == 'provision' and rest in ('', '--force'):
        text = _sandbox_provision(force=rest == '--force')
    elif sub == 'deps':
        text = _sandbox_deps(rest)
    else:
        text = f"Unknown /sandbox command '{args.strip()}'; {_SANDBOX_USAGE}"
    ui.console.print(text, markup=False, highlight=False)


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
    urls: list = [h for r in relevant if (h := r.get('href'))]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="guru — local LLM agent")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--num-ctx", type=int, default=0,
        help="Override the context window (0 = auto-detect from the model)",
    )
    parser.add_argument(
        "--reset-skills", action="store_true",
        help="Overwrite the baked-in default roles/skills on startup",
    )
    args, _ = parser.parse_known_args()

    from guru import log
    log.setup()

    from guru import skills
    skills.setup(reset=args.reset_skills)

    ledger.set_repository(JsonlLedger(config.LEDGER_DIR))
    tools.set_policy(load_tools_policy())

    session.num_ctx_override = args.num_ctx
    global ADAPTERS, REGISTRY
    ADAPTERS = _build_adapters()
    REGISTRY = build_registry(ADAPTERS)
    routing = load_routing()
    judges.set_registry(REGISTRY, routing)     # llm: judges, gate reviewer
    installed = judges.install()
    if installed:
        log.info('shadow judges: %s', installed)
    # Restore the last-used adapter+model (and log in); else pick a default.
    if not _restore_last(args.model):
        _startup_select(args.model or DEFAULT_MODEL)

    session.messages = [
        {"role": "system", "content": config.build_system_prompt()}]
    tools.reset_active_tools()

    from guru import tui
    tui.run(registry=REGISTRY, routing=routing)
    # Remember the context the (final) model ran at, so the next launch loads
    # it directly instead of recomputing the GPU fit.
    config.save_model_ctx(session.model, session.num_ctx)


if __name__ == '__main__':
    main()
