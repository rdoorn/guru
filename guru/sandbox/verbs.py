"""The sandbox verbs (design plan §1 tools, §2 steps 2-4, chunk S3):
``sandbox_run``, ``sandbox_python``, ``sandbox_diff``, ``code_health``,
``sandbox_submit`` and ``request_dependency``, plus the per-task working
copies they share.

Every verb is refused until the project's sandbox image exists
(``/sandbox provision``); ``guru.domain.tools`` registers thin wrappers
that import this module lazily and advertises the verbs only while
:func:`available` says so — and while it does, the direct write tools
(``write_file``, ``edit_file``, ``apply_patch``, ``delete_file``) are
neither advertised nor executable by an agent: in a sandbox-enabled
project the quality gate is the only write path, and ``apply_patch`` is
reached as a module function from :func:`sandbox_submit` (and from
provisioning) only. A run happens in the task's *copy* of the
project — one per ``(project, task)``, made on the first verb call with
:func:`guru.sandbox.colima.copy_excludes_for`, removed when the task ends
(``Orchestrator._finish_task`` → :func:`cleanup_task`) or when a submit
lands — inside a container with no network (``colima.run``). The model
sees a digest (exit code, first lines of output) with ``detail`` for the
last 4 KB; the full output goes to guru's log.

``sandbox_submit`` is the only way changes reach the real tree: the
copy's diff runs through the deterministic rules (``gate.rules`` with
the task's :class:`gate.Tally` of already-applied submits as ``prior``,
so the destructive thresholds hold over the whole task, not per submit)
and — unless the rules already found the change suspicious — the
code-health rules (``gate.health_flags_from`` over ``gate.health_deltas``
with a baseline reader over ``git show HEAD:<path>`` in the copy — the
copy's commit is the project as it was when the task started), then the
reviewer
(``decisions.decide_review`` on the configured ``gate`` judge, else
``judges.llm.default_reviewer``) with the user's request, the task, the
agent's intent, the change summary, the code-health rows and the diff,
and ``gate.decide`` gives the verdict. ``code_health`` shows a worker the
same rows before it submits. ``intended`` applies via
``patch.apply_patch`` (auto mode) or asks first (ask mode); ``unclear``
asks in every mode with the reviewer's reasons; ``suspicious`` refuses;
read-only mode reports the diff without consulting the reviewer. The
copy's lock is held from the diff through the review to the apply, so
what the reviewer saw is what lands. Each submit is a ``sandbox_events``
row. A copy whose baseline commit no longer matches the sha its marker
records (``colima.BaselineChanged``) is refused and discarded by every
verb that reads it: its diff can no longer be trusted.
"""
from __future__ import annotations

import itertools
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from guru import config, log, session
from guru.domain import decisions, gate, health, patch
from guru.domain import sandbox as sb
from guru.domain.decisions import Judge
from guru.repositories import sandbox_images as images
from guru.repositories.settings import load_sandbox
from guru.sandbox import colima, provision

NOT_PROVISIONED = 'Refused: sandbox not provisioned; run /sandbox provision'
COPY_GONE = ('Refused: sandbox copy no longer exists; run a sandbox verb to '
             'make a fresh one')
BASELINE_CHANGED = ('Refused: sandbox copy baseline changed; the copy was '
                    'discarded — run a sandbox verb to make a fresh one')
DIGEST_LINES = 30           # lines of stdout/stderr in a run digest
DETAIL_BYTES = 4096         # tail returned by detail
HEALTH_DIGEST_CHARS = 600   # cap on a code_health digest
SCRIPT_PREFIX = colima.SCRIPT_PREFIX
# First line of the question ``sandbox_submit`` puts to the approval asker;
# :func:`question_verdict` reads the state back (the eval runner's asker
# grants ``intended`` only).
SUBMIT_QUESTION = 'Sandbox submit — gate verdict {state}.'
_LOG_HEAD = 20_000
_FALSE = ('', 'false', '0', 'no', 'none')

# Lock order everywhere: a copy's own lock (``_key_lock``) first, then the
# table lock ``_lock`` for the dicts. ``_copy`` takes only ``_lock``.
_lock = threading.Lock()
_copies: dict[tuple[str, str], Path] = {}       # (project, task) -> copy
_copy_locks: dict[tuple[str, str], threading.RLock] = {}
_last_run: dict[tuple[str, str], colima.RunResult] = {}
# (project, task) -> the destructive counts of the submits already applied
# in that task; survives the copy's removal after an apply, cleared with
# the task (``cleanup_task``/``cleanup_all``).
_tally: dict[tuple[str, str], gate.Tally] = {}
_script_counter = itertools.count(1)


# --- project / spec / copy ---------------------------------------------------

def _project(project: Optional[Path]) -> Path:
    return (Path(project) if project is not None
            else config.PROJECT_GURU_DIR.parent).expanduser().resolve()


def spec_for(project: Optional[Path] = None) -> Optional[sb.SandboxSpec]:
    """The project's ``SandboxSpec`` under the current settings, or None
    when the settings are invalid or the project has no lockfile."""
    try:
        return sb.spec_from(_project(project), load_sandbox())
    except ValueError:
        return None


def available(project: Optional[Path] = None) -> bool:
    """True when ``project`` (default the current one) has a recorded
    sandbox image, so the verbs can run. Never raises."""
    try:
        spec = spec_for(project)
        return spec is not None and images.load_record(spec) is not None
    except Exception:                            # noqa: BLE001
        log.exc('sandbox: availability check failed')
        return False


def _ready(project: Optional[Path] = None
           ) -> tuple[Optional[sb.SandboxSpec], str]:
    """``(spec, '')`` when the verbs may run, else ``(None, refusal)``."""
    spec = spec_for(project)
    if spec is None or images.load_record(spec) is None:
        return None, NOT_PROVISIONED
    return spec, ''


def _task_key(spec: sb.SandboxSpec) -> tuple[str, str]:
    return (str(spec.project), session.task_id or session.agent_id or 'main')


def _key_lock(key: tuple[str, str]) -> threading.RLock:
    """The per-copy lock: held by a submit from diff to apply, and by a
    discard, so a copy cannot be removed or changed under a review."""
    with _lock:
        lock = _copy_locks.get(key)
        if lock is None:
            lock = _copy_locks[key] = threading.RLock()
        return lock


def _copy(spec: sb.SandboxSpec) -> Path:
    """The working copy for this task (made on first use)."""
    key = _task_key(spec)
    with _lock:
        copy = _copies.get(key)
        if copy is not None and copy.is_dir():
            return copy
        dest = (images.work_root(spec)
                / f'task-{sb.safe_name(key[1])}-{uuid.uuid4().hex[:6]}')
        copy = colima.prepare_copy(spec.project, dest,
                                   colima.copy_excludes_for(spec.project))
        _copies[key] = copy
        return copy


def _discard(key: tuple[str, str]) -> None:
    with _key_lock(key):
        with _lock:
            copy = _copies.pop(key, None)
            _last_run.pop(key, None)
        if copy is None:
            return
        try:
            colima.remove_copy(copy)
        except (ValueError, OSError):
            log.exc(f'sandbox: could not remove working copy {copy}')


def copies() -> dict[tuple[str, str], Path]:
    """The live working copies: ``(project, task) -> path``."""
    with _lock:
        return dict(_copies)


def task_tally(task_id: str, project: Optional[Path] = None) -> gate.Tally:
    """The destructive counts of the submits ``task_id`` has applied so
    far in ``project`` (default the current one)."""
    with _lock:
        return _tally.get((str(_project(project)), task_id), gate.Tally())


def _prior(key: tuple[str, str]) -> gate.Tally:
    with _lock:
        return _tally.get(key, gate.Tally())


def _record_applied(key: tuple[str, str], diff: str) -> gate.Tally:
    """Add ``diff``'s counts to the task's tally; returns the new total."""
    with _lock:
        total = _tally.get(key, gate.Tally()) + gate.tally(diff)
        _tally[key] = total
        return total


def cleanup_task(task_id: str) -> int:
    """Remove every working copy of ``task_id`` and forget its tally;
    returns how many copies. Waits for a submit in flight on that copy."""
    with _lock:
        keys = [k for k in _copies if k[1] == task_id]
        for k in [k for k in _tally if k[1] == task_id]:
            del _tally[k]
    for key in keys:
        _discard(key)
    return len(keys)


def cleanup_all() -> int:
    """Remove every working copy and every tally (exit, tests); returns
    how many copies."""
    with _lock:
        keys = list(_copies)
        _tally.clear()
    for key in keys:
        _discard(key)
    return len(keys)


# --- digests -----------------------------------------------------------------

def parse_argv(argv: object) -> list[str]:
    """A list of strings from ``argv``: a list as given, a JSON list in a
    string, else the string split on whitespace."""
    if isinstance(argv, (list, tuple)):
        return [str(a) for a in argv]
    text = str(argv or '').strip()
    if text.startswith('['):
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, list):
            return [str(a) for a in data]
    return text.split()


def _head(label: str, text: str) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []
    out = [f'--- {label} ---', *lines[:DIGEST_LINES]]
    if len(lines) > DIGEST_LINES:
        out.append(f'…truncated ({len(lines) - DIGEST_LINES} more lines; '
                   'pass detail=true for the last 4 KB)')
    return out


def digest(res: colima.RunResult) -> str:
    """The model-facing summary of a run: the exit line, then the first
    ``DIGEST_LINES`` of stdout and stderr."""
    if res.denied:
        return f'Refused: {res.denied}'
    head = f'exit {res.returncode} in {res.seconds:.1f}s'
    if res.timed_out:
        head += ' (timed out; the container was killed)'
    if res.truncated:
        head += ' [output capped]'
    lines = [head, *_head('stdout', res.stdout), *_head('stderr', res.stderr)]
    return '\n'.join(lines)


def tail(res: colima.RunResult, limit: int = DETAIL_BYTES) -> str:
    """The last ``limit`` bytes of a run's stdout+stderr."""
    if res.denied:
        return f'Refused: {res.denied}'
    text = res.stdout + (f'\n--- stderr ---\n{res.stderr}' if res.stderr
                         else '')
    if len(text) <= limit:
        return f'exit {res.returncode}\n{text}'
    return (f'exit {res.returncode}\n… ({len(text) - limit} earlier chars '
            f'omitted)\n{text[-limit:]}')


def _wants_detail(detail: object) -> bool:
    return str(detail or '').strip().lower() not in _FALSE


def _log_run(verb: str, res: colima.RunResult) -> None:
    log.info('%s: %s -> rc=%s %.1fs%s\n--- stdout ---\n%s\n--- stderr ---'
             '\n%s', verb, ' '.join(res.argv), res.returncode, res.seconds,
             ' timed out' if res.timed_out else '',
             res.stdout[:_LOG_HEAD], res.stderr[:_LOG_HEAD])


def _run(spec: sb.SandboxSpec, argv: list[str], verb: str,
         detail: object) -> str:
    copy = _copy(spec)
    res = colima.run(spec, argv, copy)
    _log_run(verb, res)
    _last_run[_task_key(spec)] = res
    return tail(res) if _wants_detail(detail) else digest(res)


# --- verbs -------------------------------------------------------------------

def sandbox_run(argv: object, detail: object = '',
                project: Optional[Path] = None) -> str:
    """Run ``argv`` (``argv[0]`` one of ``sandbox.RUNNERS``) in the task's
    copy inside the sandbox container; returns the digest, or the last 4
    KB when ``detail`` is set. An empty ``argv`` with ``detail`` returns
    the previous run's tail without running anything."""
    spec, refusal = _ready(project)
    if spec is None:
        return refusal
    args = parse_argv(argv)
    if not args and _wants_detail(detail):
        last = _last_run.get(_task_key(spec))
        return tail(last) if last is not None else 'No previous sandbox run.'
    denial = sb.check_argv(args)
    if denial:
        images.record_sandbox_event('run', args, 0.0, False, denial)
        return f'Refused: {denial}'
    return _run(spec, args, 'sandbox_run', detail)


def sandbox_python(code: str, detail: object = '',
                   project: Optional[Path] = None) -> str:
    """Write ``code`` to ``.guru-sandbox-<n>.py`` in the task's copy, run
    it with the sandbox's ``python`` and remove the file; same digest as
    :func:`sandbox_run`."""
    spec, refusal = _ready(project)
    if spec is None:
        return refusal
    text = str(code or '')
    if not text.strip():
        return 'Refused: sandbox_python needs code to run.'
    copy = _copy(spec)
    name = f'{SCRIPT_PREFIX}{next(_script_counter)}.py'
    script = copy / name
    script.write_text(text if text.endswith('\n') else text + '\n',
                      encoding='utf-8')
    try:
        return _run(spec, ['python', name], 'sandbox_python', detail)
    finally:
        try:
            script.unlink()
        except OSError:
            log.exc(f'sandbox: could not remove {script}')


def sandbox_diff(project: Optional[Path] = None) -> str:
    """Per-file ``+/-`` counts of the task's copy against the project (the
    files the sandbox changed), or a note that nothing changed."""
    spec, refusal = _ready(project)
    if spec is None:
        return refusal
    diff, refusal = _read_diff(spec, _copy(spec))
    if refusal:
        return refusal
    if not diff.strip():
        return 'The sandbox copy is unchanged.'
    return ('Sandbox changes (not yet applied; call sandbox_submit with '
            'your intent to apply them):\n' + gate.stat_text(diff))


def _read_diff(spec: sb.SandboxSpec, copy: Path) -> tuple[str, str]:
    """``(diff, '')`` from ``colima.diff``, or ``('', refusal)``: a copy
    whose baseline moved is discarded (``BASELINE_CHANGED``); a vanished
    copy is ``COPY_GONE``; any other git failure is reported."""
    try:
        return colima.diff(copy, spec.project), ''
    except colima.BaselineChanged as e:
        log.warning('sandbox: %s', e)
        _discard(_task_key(spec))
        return '', BASELINE_CHANGED
    except RuntimeError as e:
        if not copy.is_dir():
            with _lock:
                _copies.pop(_task_key(spec), None)
                _last_run.pop(_task_key(spec), None)
            return '', COPY_GONE
        return '', f'Refused: cannot read the sandbox diff: {e}'


def _baseline_reader(copy: Path, project: Path
                     ) -> Callable[[str], Optional[str]]:
    """A ``gate.BaselineReader`` over the copy's baseline commit."""
    def read(path: str) -> Optional[str]:
        return colima.show_baseline(copy, path, project)
    return read


def _clip(text: str, limit: int = HEALTH_DIGEST_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit('\n', 1)[0].rstrip()
    return f'{cut}\n… ({len(text) - len(cut)} more chars)'


def _file_health_text(rel: str, source: str) -> str:
    """Every function of one unchanged file with its metrics, the ones
    over a threshold first."""
    rows = health.file_health(source)
    if not rows:
        return f'{rel}: no functions (or the file does not parse).'
    over = [fh for fh in rows if fh.over()]
    lines = [f'{rel} is unchanged in the sandbox copy: {len(rows)} '
             f'function(s), {len(over)} over a threshold.']
    lines += [f'{fh.name}: {fh.describe()}'
              for fh in over + [fh for fh in rows if not fh.over()]]
    return '\n'.join(lines)


def code_health(path: str = '', project: Optional[Path] = None) -> str:
    """The code-health deltas of the task's copy against the project
    (``gate.health_deltas`` over the copy's diff): every changed or new
    function with its verdict and before→after metrics, degraded first.
    With ``path`` only that file — the diff is filtered to it before any
    baseline is read — or, when the copy did not change it, the file's
    current metrics (a path that resolves outside the copy, a symlink
    out of it included, is refused). Digest capped at
    ``HEALTH_DIGEST_CHARS``; nothing is applied or run."""
    spec, refusal = _ready(project)
    if spec is None:
        return refusal
    want = str(path or '').strip().replace('\\', '/')
    while want.startswith('./'):
        want = want[2:]
    if want and (Path(want).is_absolute() or '..' in Path(want).parts):
        return f'Refused: {want!r} is not a project-relative path.'
    copy = _copy(spec)
    diff, refusal = _read_diff(spec, copy)
    if refusal:
        return refusal
    deltas = gate.health_deltas(diff, _baseline_reader(copy, spec.project),
                                paths={want} if want else None)
    if want and not deltas and want not in {
            rel for rel, *_r in gate.stat(diff)}:
        return _unchanged_file_health(copy, want)
    if not deltas:
        what = want or 'the changed Python files'
        return (f'Code health: no function of {what} changed its metrics '
                'in the sandbox copy.')
    ordered = sorted(deltas, key=lambda item: (
        health.VERDICTS[::-1].index(item[1].verdict)))
    counts = {v: sum(1 for _rel, d in deltas if d.verdict == v)
              for v in health.VERDICTS}
    head = ('Code health of the sandbox changes: '
            + ', '.join(f'{counts[v]} {v}' for v in health.VERDICTS
                        if counts[v])
            + f' (thresholds: lines >{health.LINES_MAX}, complexity '
              f'>{health.COMPLEXITY_MAX}, nesting >{health.NESTING_MAX}, '
              f'args >{health.ARGS_MAX}).')
    return _clip('\n'.join([head] + [f'{rel}: {d.describe()}'
                                     for rel, d in ordered]))


def _unchanged_file_health(copy: Path, want: str) -> str:
    """The current metrics of ``want``, a file the copy did not change;
    refused when it resolves (through a symlink) outside the copy."""
    target = copy / want
    try:
        inside = target.resolve().is_relative_to(copy.resolve())
    except OSError:
        inside = False
    if not inside:
        return f'Refused: {want!r} resolves outside the sandbox copy.'
    if not target.is_file() or not want.endswith('.py'):
        return f'{want}: not a Python file in the sandbox copy.'
    try:
        source = target.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as e:
        return f'{want}: cannot read: {e}'
    return _clip(_file_health_text(want, source))


def _reviewer(diff: str) -> Optional[Judge]:
    """The configured ``gate`` judge, else the default LLM reviewer."""
    judge = decisions.judge_for(gate.GATE_POINT)
    if judge is not None:
        return judge
    from guru.judges import llm
    return llm.default_reviewer(diff, session.adapter, session.model)


def _user_request() -> str:
    from guru.adapters import turn
    return turn.turn_request()


def _verdict(diff: str, intent: str, project: Path,
             baseline: Optional[gate.BaselineReader] = None,
             prior: Optional[gate.Tally] = None) -> gate.Verdict:
    """Rules first (the deterministic ones with the task's ``prior``
    tally, then — only when nothing is already suspicious — the code-
    health ones over ``baseline``); the reviewer only when the rules
    found nothing that already settles the verdict."""
    flags = gate.rules(diff, project, prior=prior)
    deltas: list = []
    if baseline and not gate.has_suspicious(flags):
        deltas = gate.health_deltas(diff, baseline)
        flags += gate.health_flags_from(deltas)
    review = None
    if not gate.has_suspicious(flags):
        packet = gate.packet_text(_user_request(), session.task_text, intent,
                                  diff, health_block=gate.health_text(deltas))
        review = decisions.decide_review(gate.GATE_POINT,
                                         gate.review_question(packet),
                                         judge=_reviewer(diff))
    return gate.decide(flags, review)


def sandbox_submit(intent: str, project: Optional[Path] = None) -> str:
    """Run the quality gate over the task's copy and, per verdict and
    access mode, apply the diff to the real tree (``apply_patch``), ask
    the user first, or refuse. See the module docstring."""
    spec, refusal = _ready(project)
    if spec is None:
        return refusal
    what = str(intent or '').strip()
    if not what:
        return ('Refused: sandbox_submit needs your intent: one or two '
                'sentences on what the change does and why.')
    key = _task_key(spec)
    with _key_lock(key):
        with _lock:
            known = _copies.get(key)
        if known is not None and not known.is_dir():
            with _lock:
                _copies.pop(key, None)
                _last_run.pop(key, None)
            return COPY_GONE
        started = time.monotonic()
        copy = _copy(spec)
        diff, refusal = _read_diff(spec, copy)
        if refusal:
            return refusal
        if not diff.strip():
            return 'Nothing to submit: the sandbox copy is unchanged.'
        stat = gate.stat_text(diff)
        if config.MODE == config.MODE_READ_ONLY:
            # No reviewer call: nothing could be applied anyway, so the
            # diff would leave the machine for no decision.
            images.record_sandbox_event(
                'submit', ['submit', what[:80]], time.monotonic() - started,
                False, 'read-only: not applied')
            return (f'read-only: not applied. The sandbox copy differs from '
                    f'the project:\n{stat}\nChange the access mode to '
                    'submit through the gate.')
        verdict = _verdict(diff, what, spec.project,
                           _baseline_reader(copy, spec.project),
                           prior=_prior(key))
        return _settle(spec, key, what, diff, stat, verdict, started)


def _settle(spec: sb.SandboxSpec, key: tuple[str, str], what: str,
            diff: str, stat: str, verdict: gate.Verdict,
            started: float) -> str:
    """Apply, ask or refuse per ``verdict`` (the copy lock is held)."""
    reasons = '\n'.join(f'  - {r}' for r in verdict.reasons) or '  - (none)'
    gone = gate.deleted_paths(diff)
    images.record_sandbox_event(
        'submit', ['submit', what[:80]], time.monotonic() - started,
        verdict.state == gate.INTENDED,
        f"{verdict.state}: {'; '.join(verdict.reasons)}"
        + (f"; deletes: {', '.join(gone)}" if gone else ''))
    header = f'Gate verdict: {verdict.state}\n{reasons}\n{stat}'
    if gone:
        header += '\nDeleted or emptied files: ' + ', '.join(gone)
    if verdict.state == gate.SUSPICIOUS:
        return ('Refused: the quality gate found the change suspicious; '
                f'nothing was applied.\n{header}\nThe sandbox copy is kept; '
                'remove the flagged changes and submit again, or explain '
                'to the user.')
    silent = (verdict.state == gate.INTENDED
              and config.MODE == config.MODE_AUTO and config.AUTO_GRANT)
    if not silent:
        question = (SUBMIT_QUESTION.format(state=verdict.state) + '\n'
                    f'Intent: {what}\n{reasons}\n{stat}\n'
                    f'Apply this change to {spec.project}?')
        if not provision.ask(question):
            images.record_sandbox_event('apply', ['apply', what[:80]], 0.0,
                                        False, 'declined')
            return (f'Declined: the user did not approve the change '
                    f'({verdict.describe()}). Nothing was applied; the '
                    'sandbox copy is kept.')
    applied = patch.apply_patch(patch.rebase(diff, spec.project))
    ok = applied.startswith('Applied patch')
    detail = applied[:200]
    if ok:
        total = _record_applied(key, diff)
        detail = f'applied; task so far: {total.describe()}'
    images.record_sandbox_event('apply', ['apply', what[:80]],
                                time.monotonic() - started, ok, detail)
    if not ok:
        return f'{applied}\n{header}\nThe sandbox copy is kept.'
    _discard(key)
    return (f'{header}\n{applied}\nThe sandbox copy was removed; verify '
            'with run_tests on the real tree.')


def question_verdict(question: str) -> str:
    """The gate state a ``sandbox_submit`` approval question carries (its
    first line is ``SUBMIT_QUESTION``), or ``''`` for any other question
    (a dependency approval, a path prompt)."""
    head = (question or '').split('\n', 1)[0]
    for state in gate.STATES:
        if head == SUBMIT_QUESTION.format(state=state):
            return state
    return ''


def request_dependency(name: str, constraint: str = '',
                       project: Optional[Path] = None) -> str:
    """Record a dependency request (``provision.request_dependency``);
    nothing is installed until the user applies it."""
    spec, refusal = _ready(project)
    if spec is None:
        return refusal
    return provision.request_dependency(name, constraint,
                                        project=spec.project)
