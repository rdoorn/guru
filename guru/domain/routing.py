"""Routing policy: which adapter/model runs a sub-agent task.

Pure rules over a small vocabulary (design doc §4). A task carries two
labels, ``kind`` and ``complexity``; the settings carry one or more
*ladders* of rungs (cheapest first); ``resolve`` applies the working-mode,
secret-scan and spend-confirmation filters, picks a rung, and falls back in
a fixed order. Every filter that changed the outcome is recorded in
``Route.reason`` so the task record can explain itself verbatim.

No I/O and no knowledge of the adapter objects: a ``Rung`` carries its own
``remote`` flag, filled in by the repository layer from the registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

COMPLEXITY = ('trivial', 'standard', 'hard')
KINDS = ('debug', 'build', 'refactor', 'review', 'explain', 'docs', 'ops',
         'other')
MODES = ('local-only', 'local-and-remote', 'remote-only')
# What each label means: the controller hint (config.CONTROLLER_HINT) and
# the ``labels`` judge (decisions.label_questions) both read these, so the
# model that labels and the judge that checks it share one rubric. The
# examples after "e.g." are drawn from the real cases in
# evals/triage/2026-09-24-real-cases.md, where labelling was the weak spot.
COMPLEXITY_DESCRIPTIONS = {
    'trivial': 'greetings, one-line lookups (where is X defined, what does'
               ' this flag do), a single-file summary, a definition'
               ' (e.g. summarise one README section; find where a function'
               ' is defined and who calls it; explain a shell command)',
    'standard': 'read or inspect a few files, explain or fix one bug, a'
                ' one-file edit with its tests (e.g. explain how one function'
                ' or one module works; fix one failing test in one file; add'
                ' a small CLI flag plus a test)',
    'hard': 'multi-file refactors, an architecture or security review of a'
            ' whole codebase, subtle concurrency or data-race bugs (e.g.'
            ' review several modules for consistency, security or error'
            ' handling; explain an algorithm that spans multiple files with'
            ' its measurements and persistence; concurrency bugs;'
            ' architecture)',
}
KIND_DESCRIPTIONS = {
    'debug': 'find and fix a bug, a crash or a failing test',
    'build': 'implement a new feature, flag, command or component',
    'refactor': 'restructure or clean up existing code without changing'
                ' what it does',
    'review': 'review code for correctness, security or quality and report',
    'explain': 'explain how existing code or a system works',
    'docs': 'write or update documentation, READMEs or comments',
    'ops': 'deployment, configuration, infrastructure or operations',
    'other': 'anything that fits none of the other kinds',
}
assert tuple(COMPLEXITY_DESCRIPTIONS) == COMPLEXITY
assert tuple(KIND_DESCRIPTIONS) == KINDS
CONFIRMATIONS = ('granted', 'declined', 'pending', 'never')

DEFAULT_KIND = 'other'
DEFAULT_COMPLEXITY = 'standard'
DEFAULT_LADDER = 'default'
NEEDS_CONFIRMATION = 'needs_confirmation'
LOCAL_MAIN_TAKEN = 'fallback:local_main (pre-approved)'
LOCAL_MAIN_SKIPPED = 'fallback:local_main skipped (remote, local-only)'


@dataclass(frozen=True)
class Rung:
    """One step of a ladder: a model on an adapter, and the hardest task it
    should take. ``default`` marks the rung used when the complexity router
    is off (at most one per ladder)."""
    adapter: str
    model: str
    max_complexity: str
    remote: bool
    default: bool = False


@dataclass
class Ladder:
    """An ordered list of rungs, cheapest/lowest first."""
    rungs: list[Rung] = field(default_factory=list)


@dataclass
class Route:
    """The outcome of ``resolve``.

    ``rung_index`` indexes the *original* ladder named by ``ladder`` (None
    when the local main model or nothing was chosen). ``reason`` lists every
    filter/fallback that changed the outcome, in order. ``refused`` means no
    rung survived and no fallback applied; ``adapter``/``model`` are empty.
    """
    adapter: str
    model: str
    rung_index: Optional[int]
    reason: list[str] = field(default_factory=list)
    refused: bool = False
    ladder: str = ''
    kind: str = DEFAULT_KIND
    complexity: str = DEFAULT_COMPLEXITY

    @property
    def needs_confirmation(self) -> bool:
        """True when a spend confirmation is still pending for this pick."""
        return NEEDS_CONFIRMATION in self.reason

    def as_dict(self) -> dict:
        """JSON-ready record for the task ledger."""
        return {
            'adapter': self.adapter, 'model': self.model,
            'rung_index': self.rung_index, 'ladder': self.ladder,
            'kind': self.kind, 'complexity': self.complexity,
            'reason': list(self.reason), 'refused': self.refused,
        }


def normalise_labels(kind: object, complexity: object) -> tuple[str, str]:
    """Return ``(kind, complexity)`` from the vocabularies; unknown, empty or
    non-string values become ``('other', 'standard')`` component-wise."""
    k = str(kind).strip().lower() if isinstance(kind, str) else ''
    c = (str(complexity).strip().lower()
         if isinstance(complexity, str) else '')
    return (k if k in KINDS else DEFAULT_KIND,
            c if c in COMPLEXITY else DEFAULT_COMPLEXITY)


def _level(complexity: str) -> int:
    return COMPLEXITY.index(complexity)


def _plural(n: int, noun: str) -> str:
    return f'{n} {noun}' if n == 1 else f'{n} {noun}s'


def _strip(indexed: list[tuple[int, Rung]],
           remote: bool) -> tuple[list[tuple[int, Rung]], int]:
    """Drop rungs whose ``remote`` flag equals ``remote``; return the
    survivors and how many were dropped."""
    kept = [(i, r) for i, r in indexed if r.remote != remote]
    return kept, len(indexed) - len(kept)


def _pick(indexed: list[tuple[int, Rung]], complexity: str,
          complexity_router: bool, reason: list[str]) -> tuple[int, Rung]:
    """Choose ``(index, rung)`` from surviving ``indexed`` rungs (non-empty).

    Complexity router on: the lowest rung whose ``max_complexity`` is at
    least ``complexity``; when none is high enough the top surviving rung is
    used and the shortfall is recorded. Router off: the rung marked
    ``default``, else the first.
    """
    if complexity_router:
        for i, rung in indexed:
            if _level(rung.max_complexity) >= _level(complexity):
                return i, rung
        reason.append(
            f'complexity:{complexity} exceeds ladder; using top rung')
        return indexed[-1]
    for i, rung in indexed:
        if rung.default:
            return i, rung
    return indexed[0]


def _survivors(name: str, ladder: Ladder, mode: str, scan_findings: int,
               confirmation: str
               ) -> tuple[list[tuple[int, Rung]], list[str]]:
    """Apply the mode, scan and decline filters to ``ladder``; return the
    surviving ``(index, rung)`` pairs and one reason per filter that removed
    rungs, each tagged with the ladder name."""
    indexed = list(enumerate(ladder.rungs))
    notes: list[str] = []
    tag = f' (ladder {name})'
    if mode == 'local-only':
        indexed, n = _strip(indexed, remote=True)
        if n:
            notes.append(
                f'mode:local-only stripped {_plural(n, "remote rung")}{tag}')
    elif mode == 'remote-only':
        indexed, n = _strip(indexed, remote=False)
        if n:
            notes.append(
                f'mode:remote-only stripped {_plural(n, "local rung")}{tag}')
    if scan_findings > 0:
        indexed, n = _strip(indexed, remote=True)
        if n:
            notes.append(
                f'scan:{_plural(scan_findings, "finding")} forced local{tag}')
    if confirmation == 'declined':
        indexed, n = _strip(indexed, remote=True)
        if n:
            notes.append(f'confirmation:declined{tag}')
    return indexed, notes


def resolve(kind: str, complexity: str, ladders: dict, *, mode: str,
            scan_findings: int, confirmation: str, complexity_router: bool,
            type_router: bool, local_main: Optional[Rung]) -> Route:
    """Pick the adapter/model for a task (design doc §4).

    Ladder: ``ladders[kind]`` when ``type_router`` is on and such a ladder
    exists, else ``ladders['default']``. Filters: ``local-only`` strips
    remote rungs, ``remote-only`` strips local ones, any scan finding strips
    remote, a ``declined`` confirmation strips remote; ``pending`` computes
    the route as if granted and flags ``needs_confirmation`` when the pick
    is remote, so the caller can ask and re-resolve. Fallbacks when the
    chosen ladder is emptied: the default ladder (if a kind ladder was
    chosen), then ``local_main`` — the parent's own adapter/model, which is
    *pre-approved*: the parent already runs it and already saw the task
    text, so it never needs a spend confirmation and neither a scan finding
    nor a decline skips it; only ``remote-only`` (always) and an explicit
    ``local-only`` (when it is remote) do — then the first surviving rung of
    any ladder, then a refused route. Inputs are never mutated.
    """
    if mode not in MODES:
        raise ValueError(f'unknown routing mode {mode!r}; expected one of '
                         + ', '.join(MODES))
    if confirmation not in CONFIRMATIONS:
        raise ValueError(f'unknown confirmation {confirmation!r}; expected '
                         'one of ' + ', '.join(CONFIRMATIONS))
    kind, complexity = normalise_labels(kind, complexity)
    reason: list[str] = []

    def route(name: str, index: Optional[int], rung: Rung) -> Route:
        if confirmation == 'pending' and rung.remote:
            reason.append(NEEDS_CONFIRMATION)
        return Route(rung.adapter, rung.model, index, reason, ladder=name,
                     kind=kind, complexity=complexity)

    def filtered(name: str) -> tuple[list[tuple[int, Rung]], list[str]]:
        return _survivors(name, ladders[name], mode, scan_findings,
                          confirmation)

    # 1. The ladder for this task, then the default ladder as a fallback.
    candidates: list[str] = []
    if type_router and kind in ladders and kind != DEFAULT_LADDER:
        candidates.append(kind)
    if DEFAULT_LADDER in ladders:
        candidates.append(DEFAULT_LADDER)
    for pos, name in enumerate(candidates):
        indexed, notes = filtered(name)
        reason.extend(notes)
        if indexed:
            if pos > 0:
                reason.append(f'fallback:ladder {name}')
            index, rung = _pick(indexed, complexity, complexity_router,
                                reason)
            return route(name, index, rung)

    # 2. The main model as configured: pre-approved, so it bypasses the
    # scan/decline filters and never asks. Never in remote-only; a remote
    # main model is skipped only by an explicit local-only mode.
    if local_main is not None and mode != 'remote-only':
        if local_main.remote and mode == 'local-only':
            reason.append(LOCAL_MAIN_SKIPPED)
        else:
            reason.append(LOCAL_MAIN_TAKEN)
            return Route(local_main.adapter, local_main.model, None, reason,
                         ladder='', kind=kind, complexity=complexity)

    # 3. The first surviving rung of any remaining ladder. Only the chosen
    # ladder's filter notes are recorded (the emptied ones would be noise).
    for name in ladders:
        if name in candidates:
            continue
        indexed, notes = filtered(name)
        if indexed:
            reason.extend(notes)
            reason.append(f'fallback:ladder {name}')
            return route(name, indexed[0][0], indexed[0][1])

    reason.append(f'refused: no rung survives in mode {mode}'
                  + (f' with {_plural(scan_findings, "finding")}'
                     if scan_findings else '')
                  + (' after decline' if confirmation == 'declined' else ''))
    return Route('', '', None, reason, refused=True, kind=kind,
                 complexity=complexity)
