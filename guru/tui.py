"""Hybrid UI: the main agent lives in the normal terminal buffer, sub-agents
live in a full-screen (alternate-screen) viewer you toggle into.

- [main] is an async line prompt in the primary/normal screen buffer: prompts
  go here, main's output streams above the live prompt (via patch_stdout), the
  terminal scrolls back normally, and nothing is cleared on exit.
- [agent1…N] are shown in a full-screen Application (the alternate screen),
  entered with Shift+Right (or Ctrl+N to spawn+view); Shift+Left/Right cycle
  sub-agents and Shift+Left off the first one drops back to [main].

A small coordinator loop swaps between the two surfaces; entering/leaving the
full-screen app performs the terminal buffer switch automatically.

Execution is unchanged from before: each agent owns its own ``SessionState``
and console (bound per worker thread via contextvars), turns run in background
threads, and delegation is a mailbox (spawn/check/join). All queue/busy/launch/
deliver transitions happen on the event-loop thread.
"""
import asyncio
import shutil
import sys
import threading
import time
from pathlib import Path

from prompt_toolkit import Application, PromptSession
from prompt_toolkit.application import get_app, run_in_terminal
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.widgets import HorizontalLine, TextArea
from rich.console import Console

from guru import config, log, session, skills, ui
from guru.agents import Agent, AgentManager
from guru.domain import conversation, files, tools

_CTX_COLOUR = {'green': 'ansigreen', 'yellow': 'ansiyellow', 'red': 'ansired'}
_CHROME_ROWS = 5   # 2 rules + prompt + status + tabs

# Sentinels the main prompt's key bindings return to drive the coordinator.
_ENTER_TUI = object()
_NEW_AGENT = object()
_QUIT = object()


class _BufferWriter:
    """File-like sink that routes rich output into a sub-agent's buffer."""

    def __init__(self) -> None:
        self.target = None
        self.refresh = lambda: None
        self._partial = ''

    def write(self, text: str) -> None:
        if self.target is None:
            return
        self._partial += text
        while '\n' in self._partial:
            line, self._partial = self._partial.split('\n', 1)
            self.target.append(line)
        self.refresh()

    def flush(self) -> None:
        if self.target is not None and self._partial:
            self.target.append(self._partial)
            self._partial = ''
            self.refresh()


class _MainWriter:
    """Routes the main agent's output to the normal terminal buffer.

    Lines stream to stdout (above the live prompt via patch_stdout) while the
    [main] view is on screen; while the sub-agent TUI is showing they are held
    and flushed on return. Every line is also mirrored into the agent buffer.
    """

    def __init__(self, agent, state) -> None:
        self.agent = agent
        self.state = state
        self._partial = ''
        self._pending: list = []
        self._lock = threading.Lock()

    def _emit(self, line: str) -> None:
        self.agent.append(line)
        # While [main] is on screen, write to the terminal — patch_stdout
        # (active during the prompt) lifts it above the live input so main's
        # turn streams in sync. While the viewer (alt screen) is up, hold the
        # line and flush it on return.
        if self.state.get('view') == 'main':
            try:
                sys.stdout.write(line + '\n')
                sys.stdout.flush()
                return
            except Exception:                            # noqa: BLE001
                log.exc('stdout emit failed')
                pass
        self._pending.append(line)

    def write(self, text: str) -> None:
        with self._lock:
            self._partial += text
            while '\n' in self._partial:
                line, self._partial = self._partial.split('\n', 1)
                self._emit(line)

    def flush(self) -> None:
        with self._lock:
            if self._partial:
                self._emit(self._partial)
                self._partial = ''

    def drain(self) -> None:
        """Flush lines held while the TUI was showing (call in [main] view)."""
        with self._lock:
            for line in self._pending:
                try:
                    sys.stdout.write(line + '\n')
                except Exception:                        # noqa: BLE001
                    log.exc('drain write failed')
                    pass
            sys.stdout.flush()
            self._pending.clear()


def _app_cols() -> int:
    """Full render width from the active app (accurate under prompt_toolkit;
    shutil returns the fallback size while a prompt owns the terminal)."""
    try:
        return get_app().output.get_size().columns
    except Exception:                                    # noqa: BLE001
        log.exc('app cols read failed')
        return shutil.get_terminal_size((100, 30)).columns


def _status_from(st) -> tuple:
    """Build the status tuple (left, ctx, right, colour) from state."""
    total = st.num_ctx or 1
    used = st.ctx_used or 0
    pct = min(1.0, used / total)
    filled = round(pct * 10)
    bar = '█' * filled + '░' * (10 - filled)
    if pct >= config.COMPACT_AT:
        colour = 'red'
    elif pct >= 0.70:
        colour = 'yellow'
    else:
        colour = 'green'
    model = (st.model or '?').split(':')[0]
    mode = {config.MODE_READ_ONLY: 'read-only',
            config.MODE_ASK: 'ask', config.MODE_AUTO: 'auto'}.get(
        config.MODE, config.MODE)
    rs = st.active_role or 'general'
    if st.active_skill:
        rs += f"/{st.active_skill}"
    left = (f"🤖 {model} | 💪 {st.model_size or '?'} | 🔐 {mode}"
            f" | 🎭 {rs} | ")
    ctx = f"🧠 {int(pct * 100)}% {bar}"
    # Composition of the resident context in tokens (rough, ~4 chars/token).
    bd = conversation.context_breakdown(
        st.messages, st.active_tool_names, st.can_spawn)

    def _kt(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    comp = (f"📊 sys:{_kt(bd['sys'])} tl:{_kt(bd['tools'])}"
            f" in:{_kt(bd['in'])} out:{_kt(bd['out'])}")
    if bd['toolout']:
        comp += f" res:{_kt(bd['toolout'])}"
    right = (
        f" · {comp}"
        f" | ↓ {st.session_in} | ↑ {st.session_out}"
        f" | 📁 {Path.cwd().name} | 🌿 {st.git_branch or 'none'}"
    )
    return left, ctx, right, colour


def run() -> None:
    ui.refresh_git_branch()
    manager = AgentManager()
    main = manager.active
    main.state = session.current()
    main.state.can_spawn = True
    if (main.state.messages
            and main.state.messages[0].get('role') == 'system'):
        main.state.messages[0]['content'] += "\n\n" + config.DELEGATION_HINT
    for fn in (tools.spawn, tools.check, tools.join):
        if fn not in main.state.active_tools:
            main.state.active_tools.append(fn)

    state = {'loop': None, 'view': 'main', 'quit': False, 'closing': False}
    cols = shutil.get_terminal_size((100, 30)).columns
    barriers: dict = {}
    ask_lock = threading.Lock()

    main_writer = _MainWriter(main, state)
    main.console = Console(
        file=main_writer, force_terminal=True,
        color_system='256', width=cols)

    # --- permission asker (run_in_terminal; works in either view) -----------

    def _access_asker(question: str) -> bool:
        # question is the full, possibly multi-line prompt (write asks include
        # the exact operation). Shown via run_in_terminal so it works in both
        # the [main] prompt and the viewer; serialized so agents don't collide.
        def _ask() -> bool:
            # Drop modifyOtherKeys/raw mode for the prompt so Ctrl+C raises a
            # real KeyboardInterrupt (deny) and stray keys aren't re-encoded as
            # CSI-u text; restore the modes afterwards. Only an explicit yes
            # (Enter default, y, or yes) approves — anything else denies.
            ui.reset_terminal()
            try:
                ans = input(f"{question}\n[Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            finally:
                ui.enable_terminal_modes()
            return ans in ('', 'y', 'yes')

        async def _prompt() -> bool:
            return await run_in_terminal(_ask)

        with ask_lock:
            if state['closing']:
                return False
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    _prompt(), state['loop'])
                return bool(fut.result())
            except Exception:                            # noqa: BLE001
                log.exc('access asker prompt failed')
                return False

    tools.set_domain_asker(_access_asker)
    files.set_path_asker(_access_asker)

    # --- per-agent output (sub-agents use buffer consoles) ------------------

    def _attach_console(agent) -> None:
        writer = _BufferWriter()
        writer.target = agent
        writer.refresh = _invalidate
        agent.console = Console(
            file=writer, force_terminal=True,
            color_system='256', width=cols)

    # --- background execution (workers only run the blocking turn) ----------

    def _work(agent) -> None:
        token = session.use(agent.state)
        ctoken = ui.use_console(agent.console)
        try:
            while agent.queue:
                message = agent.queue.pop(0)
                agent.state.messages.append(
                    {'role': 'user', 'content': message})
                conversation.refresh_system_context()
                start = time.monotonic()
                session.adapter.run_turn()
                conversation.after_turn()
                elapsed = time.monotonic() - start
                verb = ('stopped after' if agent.state.cancel_requested
                        else 'answered in')
                agent.console.print(f"[dim]({verb} {elapsed:.1f}s)[/dim]")
        except Exception as e:                           # noqa: BLE001
            agent.append(f"[error] {e}")
        finally:
            agent.console.file.flush()
            ui.reset_console(ctoken)
            session.reset(token)
            state['loop'].call_soon_threadsafe(_on_done, agent)

    def _launch(agent) -> None:
        agent.busy = True
        agent.status = 'thinking'
        state['loop'].run_in_executor(None, _work, agent)

    def _submit(agent, text: str) -> None:
        agent.append(f"> {text}")
        agent.queue.append(text)
        if not agent.busy:
            _launch(agent)

    def _agent_for_state(st):
        return next((a for a in manager.agents if a.state is st), None)

    def _final_answer(agent) -> str:
        for m in reversed(agent.state.messages):
            role = (m.get('role') if isinstance(m, dict)
                    else getattr(m, 'role', ''))
            content = (m.get('content') if isinstance(m, dict)
                       else getattr(m, 'content', '')) or ''
            if role == 'assistant' and content.strip():
                return content.strip()
        return '(no answer produced)'

    def _format_join(results: dict) -> str:
        parts = ["[joined results]"]
        for tid, (task, ans) in results.items():
            parts.append(f"\n— {tid} · task: {task}\n{ans}")
        return "\n".join(parts)

    def _deliver(parent, notice: str, payload: str) -> None:
        parent.append(notice)
        parent.queue.append(payload)
        if not parent.busy:
            _launch(parent)
        _invalidate()

    def _report(child) -> None:
        parent = child.parent
        answer = _final_answer(child)
        bar = barriers.get(parent)
        if bar is not None and child.title in bar['remaining']:
            bar['remaining'].discard(child.title)
            bar['results'][child.title] = (child.task, answer)
            if not bar['remaining']:
                del barriers[parent]
                _deliver(parent, "[inbox] join complete",
                         _format_join(bar['results']))
        else:
            _deliver(
                parent,
                f"[inbox] result from {child.title}",
                f"[result from {child.title} · task: {child.task}]\n{answer}")

    def _on_done(agent) -> None:
        agent.busy = False
        agent.status = 'idle'
        if agent.queue:
            _launch(agent)
            _invalidate()
            return
        if agent.parent is not None:
            _report(agent)
        _invalidate()

    def _on_loop(fn):
        done = threading.Event()
        box: dict = {}

        def _runner() -> None:
            try:
                box['result'] = fn()
            except Exception as e:                       # noqa: BLE001
                box['error'] = e
            finally:
                done.set()

        state['loop'].call_soon_threadsafe(_runner)
        done.wait()
        if 'error' in box:
            raise box['error']
        return box['result']

    def _configure(agent, base, can_spawn: bool,
                   role=None, skill=None) -> None:
        st = agent.state
        st.messages = [
            {'role': 'system', 'content': config.build_system_prompt()}]
        if can_spawn:
            st.messages[0]['content'] += "\n\n" + config.DELEGATION_HINT
        st.active_tools, st.active_tool_names = tools.initial_tools(can_spawn)
        st.active_role = role or None
        st.active_skill = skill or None
        st.model = base.model
        st.adapter = base.adapter
        st.num_ctx = base.num_ctx
        st.ctx_ceiling = base.ctx_ceiling
        st.model_size = base.model_size
        st.git_branch = base.git_branch
        st.can_spawn = can_spawn
        _attach_console(agent)

    def _new_agent() -> None:
        base = manager.active.state
        agent = manager.add(f"agent{len(manager.agents)}")
        _configure(agent, base, can_spawn=True)
        agent.append(f"[{agent.title}] new agent · model {agent.state.model}")
        manager.active_index = len(manager.agents) - 1

    def _spawn_agent(task: str, role: str = '', skill: str = '') -> str:
        base = session.current()
        title = f"agent{len(manager.agents)}"
        child = Agent(id=title, title=title)
        _configure(child, base, can_spawn=False, role=role, skill=skill)
        child.task = task
        child.append(f"[{title}] spawned · task: {task}")
        child.append(f"> {task}")
        child.queue.append(task)

        def _add_and_start() -> None:
            child.parent = _agent_for_state(base)
            manager.agents.append(child)
            _launch(child)
            _invalidate()

        state['loop'].call_soon_threadsafe(_add_and_start)
        return (
            f"Spawned {title} to work on this task in parallel. Its result"
            f" will be delivered back to you automatically when it finishes."
        )

    def _do_check(caller_state, target: str) -> str:
        caller = _agent_for_state(caller_state)
        children = [a for a in manager.agents if a.parent is caller]
        if not children:
            return "You have no sub-agents."
        target = (target or 'all').strip()
        if target in ('all', '*', ''):
            lines = [
                f"{a.title}: {'running' if a.busy else 'done'}"
                for a in children]
            return "Sub-agents:\n" + "\n".join(lines)
        match = next((a for a in children if a.title == target), None)
        if match is None:
            names = ', '.join(a.title for a in children)
            return f"No sub-agent named '{target}'. Yours: {names}."
        if match.busy:
            return f"{match.title}: running (task: {match.task})"
        return (f"{match.title}: done\ntask: {match.task}\n"
                f"{_final_answer(match)}")

    def _do_join(caller_state, titles: list) -> str:
        caller = _agent_for_state(caller_state)
        children = {a.title: a for a in manager.agents if a.parent is caller}
        targets = [children[t] for t in titles if t in children]
        if not targets:
            have = ', '.join(children) or 'none'
            return f"None of those are your sub-agents. Yours: {have}."
        remaining: set = set()
        results: dict = {}
        for a in targets:
            if a.busy:
                remaining.add(a.title)
            else:
                results[a.title] = (a.task, _final_answer(a))
        if remaining:
            barriers[caller] = {'remaining': remaining, 'results': results}
            waiting = ', '.join(sorted(remaining))
            return (f"Waiting for {waiting} to finish; I'll be resumed"
                    f" automatically with their combined results.")
        _deliver(caller, "[inbox] join complete", _format_join(results))
        return "Those sub-agents already finished; resuming with results now."

    def _check(target: str) -> str:
        st = session.current()
        return _on_loop(lambda: _do_check(st, target))

    def _join(targets: str) -> str:
        st = session.current()
        titles = [t for t in targets.replace(',', ' ').split() if t]
        return _on_loop(lambda: _do_join(st, titles))

    tools.set_spawn_handler(_spawn_agent)
    tools.set_check_handler(_check)
    tools.set_join_handler(_join)

    # --- sub-agent full-screen viewer (alternate screen) --------------------

    def _output():
        rows = shutil.get_terminal_size((100, 30)).lines
        visible = max(1, rows - _CHROME_ROWS)
        return ANSI('\n'.join(manager.active.lines[-visible:]))

    output = Window(
        FormattedTextControl(_output),
        wrap_lines=True, height=Dimension(weight=1))

    def _tab_parts(highlight: int) -> list:
        parts: list = []
        for i, agent in enumerate(manager.agents):
            style = 'reverse' if i == highlight else 'ansibrightblack'
            parts.append((style, f'[{agent.title}]'))
            parts.append(('', ' '))
        return parts

    def _tabs() -> FormattedText:
        return FormattedText(_tab_parts(manager.active_index))

    tabline = Window(FormattedTextControl(_tabs), height=1)

    def _status() -> FormattedText:
        left, ctx, right, colour = _status_from(manager.active.state)
        return FormattedText([
            ('ansibrightblack', left),
            (_CTX_COLOUR[colour], ctx),
            ('ansibrightblack', right),
        ])

    statusline = Window(FormattedTextControl(_status), height=1)
    input_area = TextArea(height=1, prompt='> ', multiline=True)

    def _accept(buff) -> bool:
        text = buff.text.strip()
        if text:
            _submit(manager.active, text)
        return False

    input_area.accept_handler = _accept

    tui_kb = KeyBindings()

    @tui_kb.add('c-d')
    def _tui_quit(event) -> None:
        state['quit'] = True
        event.app.exit()

    @tui_kb.add('c-c')
    def _tui_ctrl_c(event) -> None:
        if ui.note_ctrl_c():
            state['quit'] = True
            event.app.exit()
        elif manager.active.busy:
            manager.active.state.cancel_requested = True
        else:
            event.current_buffer.text = ''

    @tui_kb.add('c-n', eager=True)
    def _tui_spawn(event) -> None:
        _new_agent()

    @tui_kb.add('s-right', eager=True)
    def _tui_next(event) -> None:
        if manager.active_index < len(manager.agents) - 1:
            manager.active_index += 1

    @tui_kb.add('s-left', eager=True)
    def _tui_prev(event) -> None:
        # Off the first sub-agent (index 1), drop back to the [main] view.
        if manager.active_index > 1:
            manager.active_index -= 1
        else:
            state['view'] = 'main'
            event.app.exit()

    @tui_kb.add('s-tab', eager=True)
    def _tui_cycle_mode(event) -> None:
        _cycle_mode()
        event.app.invalidate()

    # Multiline input: Shift+Enter (F13 via modifyOtherKeys) and Ctrl+J insert
    # a newline; Enter submits unless keys are still queued (a paste). Mirrors
    # the REPL's ui._kb; eager so it beats the TextArea's default Enter.
    @tui_kb.add('f13', eager=True)
    @tui_kb.add('c-j', eager=True)
    def _tui_newline(event) -> None:
        event.current_buffer.insert_text('\n')

    @tui_kb.add('enter', eager=True)
    def _tui_enter(event) -> None:
        if event.app.key_processor.input_queue:
            event.current_buffer.insert_text('\n')
        else:
            event.current_buffer.validate_and_handle()

    @tui_kb.add('f14', eager=True)
    @tui_kb.add('escape', 'enter', eager=True)
    def _tui_submit(event) -> None:
        event.current_buffer.validate_and_handle()

    root = HSplit([
        output,
        HorizontalLine(),
        input_area,
        HorizontalLine(),
        statusline,
        tabline,
    ])
    tui_app = Application(
        layout=Layout(root, focused_element=input_area),
        key_bindings=tui_kb,
        full_screen=True,
        mouse_support=False,
    )

    def _invalidate() -> None:
        if state['view'] == 'tui':
            tui_app.invalidate()

    # --- main prompt (normal buffer) ----------------------------------------

    main_kb = KeyBindings()

    @main_kb.add('c-d')
    def _m_quit(event) -> None:
        event.app.exit(result=_QUIT)

    @main_kb.add('c-c')
    def _m_ctrl_c(event) -> None:
        if ui.note_ctrl_c():
            event.app.exit(result=_QUIT)
        elif main.busy:
            main.state.cancel_requested = True
        else:
            event.current_buffer.text = ''

    @main_kb.add('c-n', eager=True)
    def _m_spawn(event) -> None:
        # erase_when_done so leaving for the viewer wipes the prompt instead of
        # depositing a leftover '> ' line in the scrollback (which stacked up
        # every round-trip). A real Enter-submit keeps erase_when_done False,
        # so submitted input stays visible.
        event.app.erase_when_done = True
        event.app.exit(result=_NEW_AGENT)

    @main_kb.add('s-right', eager=True)
    def _m_enter_tui(event) -> None:
        # Only leave the prompt if there's actually a sub-agent to view, so an
        # accidental Shift+Right doesn't discard what you're typing.
        if len(manager.agents) > 1:
            event.app.erase_when_done = True
            event.app.exit(result=_ENTER_TUI)

    @main_kb.add('s-tab', eager=True)
    def _m_cycle_mode(event) -> None:
        _cycle_mode()
        event.app.invalidate()   # refresh the mode shown in the status bar

    ps = PromptSession(
        history=FileHistory(str(config.GURU_HOME / 'history')),
        multiline=True,
        key_bindings=merge_key_bindings([ui._kb, main_kb]),
    )

    def _main_toolbar():
        # Match the TUI's bottom chrome: rule · status · tabs. The output
        # "pane" for [main] is the terminal scrollback above this bar. The
        # rule is one column short of full width: a line exactly the terminal
        # width auto-wraps the cursor and corrupts prompt_toolkit's redraw.
        columns = _app_cols()
        left, ctx, right, colour = _status_from(main.state)
        parts = [
            ('', '─' * (columns - 1) + '\n'),
            ('ansibrightblack', left),
            (_CTX_COLOUR[colour], ctx),
            ('ansibrightblack', right),
            ('', '\n'),
        ]
        parts.extend(_tab_parts(0))       # [main] highlighted in main view
        return FormattedText(parts)

    def _main_message():
        # A rule above the '> ' input frames the prompt (── / > / ── status),
        # for the TUI feel. One column short of full width to avoid the cursor
        # wrap that corrupts the redraw.
        return FormattedText([
            ('', '─' * (_app_cols() - 1) + '\n'),
            ('', '> '),
        ])

    async def _in_terminal(fn, *args) -> None:
        """Run a blocking, terminal-controlling command (picker/input) in a
        worker thread, awaited so the main prompt does not restart under it.
        No patch_stdout / prompt is active here, so the terminal is free."""
        await state['loop'].run_in_executor(None, lambda: fn(*args))

    def _cycle_mode() -> None:
        i = config.MODES.index(config.MODE)
        config.MODE = config.MODES[(i + 1) % len(config.MODES)]

    def _set_role(arg: str) -> None:
        arg = (arg or '').strip()
        if arg in ('', 'off', 'none', 'general-purpose'):
            main.state.active_role = None
            main.console.print("[green]Role[/green] -> general-purpose")
            return
        entry = skills.get(arg)
        if entry is None or entry.kind != skills.ROLE:
            avail = ', '.join(skills.names(skills.REGISTRY, skills.ROLE))
            main.console.print(
                f"[red]No role '{arg}'.[/red] Roles: {avail}")
            return
        main.state.active_role = arg
        main.console.print(f"[green]Role[/green] -> {arg}")

    def _set_skill(arg: str) -> None:
        arg = (arg or '').strip()
        if arg in ('', 'off', 'none'):
            main.state.active_skill = None
            main.console.print("[green]Skill[/green] -> none")
            return
        entry = skills.get(arg)
        if entry is None or entry.kind != skills.SKILL:
            avail = ', '.join(skills.names(skills.REGISTRY, skills.SKILL))
            main.console.print(
                f"[red]No skill '{arg}'.[/red] Skills: {avail}")
            return
        main.state.active_skill = arg
        main.console.print(f"[green]Skill[/green] -> {arg}")

    def _set_mode(arg: str) -> None:
        """Set the access mode by name (prefix match) or cycle if no arg."""
        if arg:
            match = next(
                (m for m in config.MODES if m.startswith(arg)), None)
            if match is None:
                main.console.print(
                    f"[red]Unknown mode '{arg}'.[/red] Modes: "
                    + ', '.join(config.MODES))
                return
            config.MODE = match
        else:
            _cycle_mode()
        main.console.print(f"[green]Mode:[/green] {config.MODE}")

    async def _handle_command(text: str) -> bool:
        """Run a slash command; return True if it was a command."""
        low = text.lower()
        if low in ('exit', 'quit'):
            state['quit'] = True
            return True
        if text == '/mode' or text.startswith('/mode '):
            _set_mode(text[6:].strip())
            return True
        if text == '/role' or text.startswith('/role '):
            _set_role(text[5:].strip())
            return True
        if text == '/skill' or text.startswith('/skill '):
            _set_skill(text[6:].strip())
            return True
        if text in ('/models', '/model'):
            import guru.cli as cli
            await _in_terminal(cli._models_command)
            return True
        if text == '/adapters':
            import guru.cli as cli
            await _in_terminal(cli._adapters_command)
            return True
        if text == '/context':
            import guru.cli as cli
            await _in_terminal(cli._context_command)
            return True
        if text == '/resume':
            await _in_terminal(conversation.resume_command)
            return True
        if text.startswith('/search '):
            import guru.cli as cli
            await _in_terminal(cli._handle_slash_search, text[8:].strip())
            return True
        if text == '/save':
            conversation.save_conversation()
            return True
        if text == '/compact':
            conversation.compact_messages(force=True)
            session.ctx_used = conversation.estimate_tokens(session.messages)
            main.console.print(
                f"[green]Compacted[/green] · ~{session.ctx_used:,} tokens.")
            return True
        return False

    async def _run_main() -> None:
        while state['view'] == 'main' and not state['quit']:
            # Flush output held during the last prompt / the viewer so it
            # lands above the fresh prompt. modifyOtherKeys is armed via
            # pre_run (after prompt_toolkit sets up the terminal and probes
            # the cursor), matching the classic REPL — arming it manually
            # beforehand corrupted the cursor-position detection.
            main_writer.drain()
            # Default: keep the line on submit (Enter). The sentinel exits
            # (Ctrl+N / Shift+Right) flip this to True to erase instead.
            ps.app.erase_when_done = False
            try:
                # raw=True so rich's ANSI colour codes (from main's console)
                # pass through instead of being shown literally (?[1;32m…).
                with patch_stdout(raw=True):
                    res = await ps.prompt_async(
                        _main_message, bottom_toolbar=_main_toolbar,
                        style=ui._TOOLBAR_STYLE,
                        pre_run=ui.enable_terminal_modes)
            except EOFError:
                state['quit'] = True
                return
            except KeyboardInterrupt:
                continue
            if res is _QUIT:
                state['quit'] = True
                return
            if res is _NEW_AGENT:
                _new_agent()
                state['view'] = 'tui'
                return
            if res is _ENTER_TUI:
                if len(manager.agents) > 1:
                    if manager.active_index < 1:
                        manager.active_index = 1
                    state['view'] = 'tui'
                    return
                main.console.print(
                    "[dim](no sub-agents yet — Ctrl+N to spawn one)[/dim]")
                continue
            text = (res or '').strip()
            if not text:
                continue
            if await _handle_command(text):
                continue
            _submit(main, text)

    def _greet() -> None:
        main.console.print(
            f"[bold]guru[/bold] · model [bold]{main.state.model}[/bold]")
        main.console.print(
            "Enter submits · Ctrl+N new agent · Shift+Right view agents"
            " · Shift+Tab cycle access mode · double Ctrl+C exit")
        main.console.print(
            "[dim]/mode /role /skill /models /context /adapters /save"
            " /resume /compact /search[/dim]\n")

    async def _amain() -> None:
        state['loop'] = asyncio.get_running_loop()
        _greet()
        try:
            while not state['quit']:
                if state['view'] == 'main':
                    await _run_main()
                elif len(manager.agents) <= 1:
                    state['view'] = 'main'
                else:
                    await tui_app.run_async(
                        pre_run=ui.enable_terminal_modes)
        finally:
            state['closing'] = True
            ui.reset_terminal()

    asyncio.run(_amain())
