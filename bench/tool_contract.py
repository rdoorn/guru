"""Tool contract benchmark: can a model call each guru tool correctly?

    .venv/bin/python bench/tool_contract.py --model 'Adapter|model' \\
        [--out evals/models] [--web] [--only read_file,edit_file]

One canned mini-task per tool on a fresh copy of the ``cli-tool`` fixture.
The model sees exactly one tool spec (built the way guru's adapters build
it) and is forced to call it where the provider supports forcing
(Anthropic ``tool_choice``, OpenAI-style ``tool_choice`` through a LiteLLM
proxy); Ollama cannot force, so the prompt asks. Per tool the run records
whether a call came, whether it was the right tool, how many arguments
were rejected by ``tools.validate_arguments`` (schema errors), how many
retries the corrective line bought, whether the executed call succeeded,
tokens in/out, USD when the proxy or the price table says, and seconds.

Writes ``evals/models/<slug>.json`` (``slug`` from the ``Adapter|model``
spec) and prints one table row per tool plus a summary; the eval matrix
reads the JSON for its ``contract`` column. Web tools run only with
``--web`` (network); the sandbox verbs need a provisioned image and are
listed as skipped; ``spawn``/``check``/``join`` run against recording
handlers.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root

from guru import config, session, ui                            # noqa: E402
from guru.adapters import anthropic as ant                      # noqa: E402
from guru.adapters import litellm as lite                       # noqa: E402
from guru.adapters.anthropic import AnthropicAdapter            # noqa: E402
from guru.adapters.litellm import LiteLLMAdapter                # noqa: E402
from guru.adapters.ollama import OllamaAdapter                  # noqa: E402
from guru.domain import files, pricing, procs, tools            # noqa: E402
from guru.evals import runs                                     # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / 'evals/fixtures/cli-tool'
OUT_DIR = Path('evals/models')
MAX_ATTEMPTS = 3
OLLAMA_NUM_CTX = 8192
SYSTEM = (
    "You are guru's worker on the project in the current directory. Do the"
    " task by calling the tool provided: exactly one call, with arguments"
    " that fit its schema (paths are relative to the project root). Do not"
    " answer in prose and do not ask questions.")
# A result that starts like this did not do the job.
FAIL_PREFIXES = ('Tool error:', 'Unknown tool:', tools.INVALID_ARGS_PREFIX,
                 'Refused', 'Denied', 'No such', 'Access to', 'Invalid',
                 'Not a directory', 'Cannot')
SKIP_SANDBOX = 'needs a provisioned sandbox image'


# --- tasks -------------------------------------------------------------------

@dataclass
class Task:
    """One tool's mini-task: the prompt, an optional setup on the fixture
    copy (returns text merged into the prompt) and an optional success
    check on (result, copy)."""
    tool: str
    prompt: str
    setup: Optional[Callable[[Path], dict]] = None
    check: Optional[Callable[[str, Path], bool]] = None
    web: bool = False


def _git(copy: Path, *args: str) -> procs.ProcResult:
    return procs.run(['git', '-C', str(copy), '-c', 'user.name=bench', '-c',
                      'user.email=bench@local', *args], copy,
                     procs.Limits(timeout_s=30))


def _setup_git(copy: Path) -> dict:
    _git(copy, 'init', '-q')
    _git(copy, 'add', '-A')
    _git(copy, 'commit', '-q', '-m', 'fixture')
    return {}


def _setup_git_dirty(copy: Path) -> dict:
    _setup_git(copy)
    target = copy / 'wordcount.py'
    target.write_text(target.read_text() + '\n# touched\n')
    return {}


def _setup_sha(copy: Path) -> dict:
    return {'sha': files.sha_of((copy / 'wordcount.py').read_text())}


def _setup_scratch(copy: Path) -> dict:
    (copy / 'scratch.txt').write_text('scratch\n')
    return {}


def _setup_patch(copy: Path) -> dict:
    old = (copy / 'wordcount.py').read_text()
    new = old.replace("text.split(' ')", 'text.split()')
    diff = ''.join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True),
        'a/wordcount.py', 'b/wordcount.py'))
    return {'diff': diff}


def _fixed(copy: Path) -> bool:
    return 'text.split()' in (copy / 'wordcount.py').read_text()


TASKS: list[Task] = [
    Task('search_tools', 'Find a tool that can fetch a web page by URL.',
         check=lambda r, c: 'web_fetch' in r),
    Task('use_skill', 'Adopt the code-review skill for this task.'),
    Task('spawn', 'Delegate a code review of wordcount.py to a sub-agent:'
         ' it is a review task of standard complexity; give it the'
         ' developer role and the code-review skill.',
         check=lambda r, c: r.startswith('spawned')),
    Task('check', 'Check the status of all your sub-agents.',
         check=lambda r, c: r.startswith('checked')),
    Task('join', 'Ask to be resumed when sub-agents agent2 and agent3 have'
         ' both finished.', check=lambda r, c: r.startswith('joined')),
    Task('list_dir', 'List the files in the project root directory.',
         check=lambda r, c: 'wordcount.py' in r),
    Task('list_tree', 'Show the project directory tree, two levels deep.',
         check=lambda r, c: 'tests/' in r),
    Task('read_file', 'Read lines 1 to 12 of wordcount.py.',
         check=lambda r, c: 'lines 1-12' in r),
    Task('search_code', "Search the project's Python files for the text"
         " 'count_words'.", check=lambda r, c: 'wordcount.py:' in r),
    Task('write_file', 'Create a file named notes.txt in the project root'
         ' containing the single line: hello',
         check=lambda r, c: (c / 'notes.txt').is_file()),
    Task('edit_file', "In wordcount.py replace the text text.split(' ')"
         ' with text.split() -- the current sha of wordcount.py is {sha}.',
         setup=_setup_sha, check=lambda r, c: _fixed(c)),
    Task('delete_file', 'Delete the file scratch.txt.', setup=_setup_scratch,
         check=lambda r, c: not (c / 'scratch.txt').exists()),
    Task('outline', 'Show the outline (functions and classes with their line'
         ' ranges) of wordcount.py.', check=lambda r, c: 'count_words' in r),
    Task('find_symbol', 'Find where the function count_words is defined and'
         ' where it is referenced.', check=lambda r, c: 'def: ' in r),
    Task('run_tests', "Run the project's test suite.",
         check=lambda r, c: 'failed' in r or 'passed' in r),
    Task('check_syntax', 'Check wordcount.py for syntax errors.',
         check=lambda r, c: r.startswith('ok')),
    Task('lint', 'Run the linters on wordcount.py.',
         check=lambda r, c: 'flake8' in r),
    Task('git_status', 'Show the git working tree status of the project.',
         setup=_setup_git),
    Task('git_diff', 'Show the unstaged git diff stat of the project.',
         setup=_setup_git_dirty, check=lambda r, c: 'wordcount.py' in r),
    Task('apply_patch', 'Apply this unified diff to the project:\n{diff}',
         setup=_setup_patch, check=lambda r, c: _fixed(c)),
    Task('web_search', 'Search the web for the latest Python release.',
         web=True),
    Task('web_fetch', 'Fetch the page https://example.com/ and read it.',
         web=True),
    Task('fetch_github_releases', 'What is the latest release of the GitHub'
         ' project kubernetes/kubernetes?', web=True),
]


# --- drivers -----------------------------------------------------------------

@dataclass
class Step:
    """One model round: its text, ``[(name, args, call_id)]`` and usage."""
    text: str
    calls: list
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: Optional[float] = None


def _forcing_rejected(e: Exception) -> bool:
    """Whether a provider error is about a forced tool choice (Anthropic
    refuses ``tool_choice`` other than ``auto`` with extended thinking)."""
    text = str(e).lower()
    return 'tool_choice' in text or 'tool choice' in text


class Driver:
    """A provider-specific single-tool conversation. Subclasses keep the
    native history; ``forcing`` says whether the provider can force the
    call."""
    forcing = False

    def __init__(self, adapter, model: str) -> None:
        self.adapter = adapter
        self.model = model

    def start(self, system: str, user: str) -> None:
        raise NotImplementedError

    def step(self, spec: dict, force: bool) -> Step:
        raise NotImplementedError

    def tool_result(self, call_id: str, name: str, content: str) -> None:
        raise NotImplementedError

    def user(self, text: str) -> None:
        raise NotImplementedError


class AnthropicDriver(Driver):
    forcing = True

    def start(self, system: str, user: str) -> None:
        self.system = system
        self.native: list = [{'role': 'user', 'content': user}]
        self.client = self.adapter._client()

    def step(self, spec: dict, force: bool) -> Step:
        kwargs: dict = {
            'model': self.model, 'max_tokens': ant._MAX_TOKENS,
            'system': self.system, 'messages': self.native,
            'tools': ant.tool_defs([spec])}
        if force and self.forcing:
            kwargs['tool_choice'] = {'type': 'tool', 'name': spec['name']}
        try:
            resp = self.adapter._create(self.client, **kwargs)
        except Exception as e:                       # noqa: BLE001
            if not _forcing_rejected(e) or 'tool_choice' not in kwargs:
                raise
            # The provider refuses a forced call (extended thinking);
            # fall back to asking, and say so in the report.
            self.forcing = False
            del kwargs['tool_choice']
            resp = self.adapter._create(self.client, **kwargs)
        self.native.append({'role': 'assistant', 'content': resp.content})
        text = ''.join(b.text for b in resp.content if b.type == 'text')
        calls = [(b.name, dict(b.input), b.id) for b in resp.content
                 if b.type == 'tool_use']
        u = resp.usage
        tokens_in = ((getattr(u, 'input_tokens', 0) or 0)
                     + (getattr(u, 'cache_read_input_tokens', 0) or 0)
                     + (getattr(u, 'cache_creation_input_tokens', 0) or 0))
        tokens_out = getattr(u, 'output_tokens', 0) or 0
        cost = pricing.cost_usd(self.model, pricing.Usage(
            input_tokens=getattr(u, 'input_tokens', 0) or 0,
            output_tokens=tokens_out))
        return Step(text, calls, tokens_in, tokens_out, cost)

    def tool_result(self, call_id: str, name: str, content: str) -> None:
        self.native.append({'role': 'user', 'content': [{
            'type': 'tool_result', 'tool_use_id': call_id,
            'content': content}]})

    def user(self, text: str) -> None:
        self.native.append({'role': 'user', 'content': text})


class OpenAIDriver(Driver):
    """LiteLLM proxy (OpenAI chat completions)."""
    forcing = True

    def start(self, system: str, user: str) -> None:
        self.native = [{'role': 'system', 'content': system},
                       {'role': 'user', 'content': user}]
        self.client = self.adapter._client()

    def step(self, spec: dict, force: bool) -> Step:
        kwargs: dict = {
            'model': self.model, 'messages': self.native,
            'tools': lite.openai_tool_defs([spec]),
            'max_tokens': lite._MAX_TOKENS}
        if force and self.forcing:
            kwargs['tool_choice'] = {
                'type': 'function', 'function': {'name': spec['name']}}
        try:
            resp, cost = lite._complete(self.client, dump_as='bench',
                                        **kwargs)
        except Exception as e:                       # noqa: BLE001
            if not _forcing_rejected(e) or 'tool_choice' not in kwargs:
                raise
            self.forcing = False
            del kwargs['tool_choice']
            resp, cost = lite._complete(self.client, dump_as='bench',
                                        **kwargs)
        msg = resp.choices[0].message
        text = msg.content or ''
        tool_calls = list(getattr(msg, 'tool_calls', None) or [])
        assistant: dict = {'role': 'assistant', 'content': text or None}
        if tool_calls:
            assistant['tool_calls'] = [
                {'id': tc.id, 'type': 'function',
                 'function': {'name': tc.function.name,
                              'arguments': tc.function.arguments}}
                for tc in tool_calls]
        self.native.append(assistant)
        calls = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or '{}')
            except json.JSONDecodeError:
                args = tc.function.arguments
            calls.append((tc.function.name, args, tc.id))
        usage = getattr(resp, 'usage', None)
        tokens_in = getattr(usage, 'prompt_tokens', 0) or 0
        tokens_out = getattr(usage, 'completion_tokens', 0) or 0
        if cost is None:
            cost = pricing.cost_usd(self.model, pricing.Usage(
                input_tokens=tokens_in, output_tokens=tokens_out))
        return Step(text, calls, tokens_in, tokens_out, cost)

    def tool_result(self, call_id: str, name: str, content: str) -> None:
        self.native.append({'role': 'tool', 'tool_call_id': call_id,
                            'content': content})

    def user(self, text: str) -> None:
        self.native.append({'role': 'user', 'content': text})


class OllamaDriver(Driver):
    forcing = False

    def start(self, system: str, user: str) -> None:
        self.native = [{'role': 'system', 'content': system},
                       {'role': 'user', 'content': user}]

    def step(self, spec: dict, force: bool) -> Step:
        import ollama
        resp = ollama.chat(
            model=self.model, messages=self.native,
            tools=lite.openai_tool_defs([spec]),
            think=self.adapter._supports_thinking(self.model),
            options={'num_ctx': OLLAMA_NUM_CTX})
        msg = resp.message
        self.native.append(msg)
        calls = [(c.function.name, dict(c.function.arguments or {}),
                  str(uuid.uuid4())[:8]) for c in (msg.tool_calls or [])]
        return Step(msg.content or '', calls,
                    getattr(resp, 'prompt_eval_count', 0) or 0,
                    getattr(resp, 'eval_count', 0) or 0, 0.0)

    def tool_result(self, call_id: str, name: str, content: str) -> None:
        self.native.append({'role': 'tool', 'tool_name': name,
                            'content': content})

    def user(self, text: str) -> None:
        self.native.append({'role': 'user', 'content': text})


_DRIVERS = ((AnthropicAdapter, AnthropicDriver),
            (LiteLLMAdapter, OpenAIDriver), (OllamaAdapter, OllamaDriver))


def driver_for(adapter, model: str) -> Driver:
    for cls, drv in _DRIVERS:
        if isinstance(adapter, cls):
            return drv(adapter, model)
    raise SystemExit(f'no driver for adapter {type(adapter).__name__}')


def resolve(spec: str, adapters: list) -> tuple:
    """``(adapter, model)`` for an ``Adapter|model`` spec (name match,
    case-insensitive; no adapter part = Ollama)."""
    name, sep, model = spec.partition('|')
    if not sep:
        name, model = '', name
    for a in adapters:
        if not name and isinstance(a, OllamaAdapter):
            return a, model
        if name and getattr(a, 'name', '').lower() == name.lower():
            return a, model
    raise SystemExit(f"no adapter named {name!r} in {config.ADAPTERS_PATH}"
                     f" (have: {', '.join(a.name for a in adapters)})")


# The result file stem: one implementation, shared with the matrix reader.
slug = runs.model_slug


# --- running -----------------------------------------------------------------

@dataclass
class ToolResult:
    tool: str
    status: str = 'run'        # run | skipped
    note: str = ''
    called: bool = False
    right_tool: bool = False
    schema_errors: int = 0
    retries: int = 0
    ok: bool = False
    attempts: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: Optional[float] = None
    seconds: float = 0.0
    arguments: dict = field(default_factory=dict)
    result_head: str = ''


def _quiet() -> None:
    """Silence the tool-side console output (the table is the report):
    the ``* tool arg`` notes, the size line and the write diff blocks
    (``files._show_change`` prints through the base console, which the
    proxy's ``quiet`` flag does not reach)."""
    ui.note_tool = lambda *a: None
    ui.note_tool_result = lambda n: None
    files._show_change = lambda diff: None
    try:
        ui._base_console.quiet = True
    except AttributeError:
        pass


# Lead-only tools not benchmarked here: apply_work needs a sandbox and a
# finished worker.
LOOP_TOOLS = frozenset(('apply_work',))


def _install_fake_handlers() -> None:
    tools.set_spawn_handler(
        lambda task, role, skill, kind, complexity:
        f'spawned agent2: kind={kind} complexity={complexity} role={role}'
        f' skill={skill}')
    tools.set_check_handler(lambda target: f'checked {target}: running')
    tools.set_join_handler(lambda targets: f'joined {targets}')


def _fresh_copy(base: Path) -> Path:
    copy = base / f'copy-{uuid.uuid4().hex[:6]}'
    shutil.copytree(FIXTURE, copy, ignore=shutil.ignore_patterns(
        '.venv', '__pycache__', '.pytest_cache', 'uv.lock'))
    return copy


def _bind(copy: Path) -> None:
    os.chdir(copy)
    config.ALLOWED_READ_DIRS = {str(copy.resolve())}
    config.ALLOWED_WRITE_DIRS = {str(copy.resolve())}
    config.MODE = config.MODE_AUTO
    st = session.current()
    st.file_shas = {}
    st.controller = False
    st.task_kind = ''


def _add_cost(total: Optional[float], part: Optional[float]
              ) -> Optional[float]:
    if part is None:
        return total
    return (total or 0.0) + part


def run_task(task: Task, driver: Driver, copy: Path) -> ToolResult:
    """Run one mini-task to a result: up to ``MAX_ATTEMPTS`` rounds, each
    retry carrying the corrective line back to the model."""
    out = ToolResult(task.tool)
    _bind(copy)
    fills = task.setup(copy) if task.setup else {}
    prompt = task.prompt.format(**fills)
    spec = tools.tool_spec(task.tool)
    driver.start(SYSTEM, prompt)
    started = time.monotonic()
    result = ''
    for attempt in range(1, MAX_ATTEMPTS + 1):
        out.attempts = attempt
        try:
            step = driver.step(spec, driver.forcing)
        except Exception as e:                       # noqa: BLE001
            out.note = f'{type(e).__name__}: {str(e)[:120]}'
            break
        out.tokens_in += step.tokens_in
        out.tokens_out += step.tokens_out
        out.cost_usd = _add_cost(out.cost_usd, step.cost_usd)
        if not step.calls:
            if attempt < MAX_ATTEMPTS:
                out.retries += 1
                driver.user(f'No tool was called. Call the {task.tool} tool'
                            ' now, with the arguments the task needs.')
                continue
            out.note = 'no tool call'
            break
        name, args, call_id = step.calls[0]
        out.called = True
        out.arguments = args if isinstance(args, dict) else {'_raw': args}
        if name != task.tool:
            out.note = f'called {name}'
            driver.tool_result(call_id, name, f'Unknown tool: {name}. The'
                               f' tool you have is {task.tool}.')
            out.retries += 1
            continue
        out.right_tool = True
        _clean, error = tools.validate_arguments(task.tool, args)
        if error:
            out.schema_errors += 1
            driver.tool_result(call_id, name, error)
            if attempt < MAX_ATTEMPTS:
                out.retries += 1
                continue
            result = error
            break
        result = tools.execute_tool(task.tool, args)
        out.ok = (not result.startswith(FAIL_PREFIXES)
                  and (task.check is None or bool(task.check(result, copy))))
        if not out.ok:
            out.note = 'call ran but did not do the job'
        break
    out.seconds = round(time.monotonic() - started, 2)
    out.result_head = _strip_copy(result, copy)[:160]
    return out


# What replaces the fixture copy's absolute path in ``result_head``: the
# temp path differs per run and per task and says nothing about the tool.
COPY_TOKEN = '<copy>'


def _strip_copy(text: str, copy: Path) -> str:
    """``text`` with every spelling of the copy's path (as given and
    resolved) replaced by ``COPY_TOKEN``."""
    for spelling in sorted({str(copy), str(copy.resolve())}, key=len,
                           reverse=True):
        text = text.replace(spelling, COPY_TOKEN)
    return text


def run_all(spec: str, adapters: list, *, web: bool = False,
            only: Optional[set] = None) -> dict:
    adapter, model = resolve(spec, adapters)
    driver = driver_for(adapter, model)
    _install_fake_handlers()
    results: list = []
    base = Path(tempfile.mkdtemp(prefix='guru-contract-'))
    cwd = Path.cwd()
    try:
        for task in TASKS:
            if only and task.tool not in only:
                continue
            if task.web and not web:
                results.append(ToolResult(task.tool, 'skipped',
                                          'network; pass --web'))
                continue
            copy = _fresh_copy(base)
            res = run_task(task, driver, copy)
            results.append(res)
            print(_row(res), flush=True)
        for name in tools.SANDBOX_TOOLS:
            if not only or name in only:
                results.append(ToolResult(name, 'skipped', SKIP_SANDBOX))
    finally:
        os.chdir(cwd)
        shutil.rmtree(base, ignore_errors=True)
    ran = [r for r in results if r.status == 'run']
    summary = {
        'tools': len(ran), 'called': sum(r.called for r in ran),
        'right_tool': sum(r.right_tool for r in ran),
        'ok': sum(r.ok for r in ran),
        'schema_errors': sum(r.schema_errors for r in ran),
        'retries': sum(r.retries for r in ran),
        'tokens_in': sum(r.tokens_in for r in ran),
        'tokens_out': sum(r.tokens_out for r in ran),
        'cost_usd': (round(sum(r.cost_usd or 0.0 for r in ran), 4)
                     if any(r.cost_usd is not None for r in ran) else None),
        'seconds': round(sum(r.seconds for r in ran), 1),
        'pass_rate': round(sum(r.ok for r in ran) / len(ran), 3)
        if ran else 0.0,
    }
    return {'model': spec, 'adapter': adapter.name, 'model_id': model,
            'forcing': driver.forcing,
            'date': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'summary': summary,
            'tools': {r.tool: asdict(r) for r in results}}


# --- output ------------------------------------------------------------------

_HEAD = (f"{'tool':<22} {'call':>4} {'ok':>3} {'schema':>6} {'retry':>5}"
         f" {'tok_in':>7} {'tok_out':>7} {'sec':>6}  note")


def _row(r: ToolResult) -> str:
    if r.status != 'run':
        return f"{r.tool:<22} {'-':>4} {'-':>3} {'-':>6} {'-':>5}" \
               f" {'-':>7} {'-':>7} {'-':>6}  skipped: {r.note}"
    return (f"{r.tool:<22} {'yes' if r.called else 'no':>4}"
            f" {'yes' if r.ok else 'no':>3} {r.schema_errors:>6}"
            f" {r.retries:>5} {r.tokens_in:>7} {r.tokens_out:>7}"
            f" {r.seconds:>6.1f}  {r.note}")


def render(report: dict) -> str:
    rows = [f"tool contract: {report['model']} (forcing:"
            f" {'yes' if report['forcing'] else 'no'})", _HEAD]
    for data in report['tools'].values():
        rows.append(_row(ToolResult(**data)))
    s = report['summary']
    cost = f" ${s['cost_usd']:.4f}" if s['cost_usd'] is not None else ''
    rows.append(f"summary: {s['ok']}/{s['tools']} ok, {s['called']} called,"
                f" {s['schema_errors']} schema errors, {s['retries']}"
                f" retries, {s['tokens_in']}+{s['tokens_out']} tokens,"
                f" {s['seconds']}s{cost}")
    return "\n".join(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', required=True,
                        help="'Adapter|model' as in adapters.toml")
    parser.add_argument('--out', type=Path, default=OUT_DIR,
                        help=f'results directory (default {OUT_DIR})')
    parser.add_argument('--web', action='store_true',
                        help='also run the web tools (network)')
    parser.add_argument('--only', default='',
                        help='comma-separated tool names to run')
    args = parser.parse_args(argv)
    from guru import bench
    _quiet()
    only = {t.strip() for t in args.only.split(',') if t.strip()} or None
    print(_HEAD, flush=True)
    report = run_all(args.model, bench.build_adapters(), web=args.web,
                     only=only)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f'{slug(args.model)}.json'
    path.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(render(report).splitlines()[-1])
    print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
