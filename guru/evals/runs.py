"""Run files, run-to-run comparison and the trajectory table.

Repository layer: JSON under ``evals/runs/<ts>-<run_id>.json``, the
gzipped transcripts under ``evals/runs/<run_id>/transcripts/`` and the
Markdown table ``evals/TRAJECTORY.md``. Nothing here runs a case; the
runner hands over a :class:`Run`. :func:`find_run` locates a run file by
id (``grade RUN_ID``) and :func:`load_transcript` reads a case transcript
back for offline grading.

Trajectory columns: ``ts | run_id | model | passed/total | mean seconds |
cost | note | tok/case | turns/case`` — the two metric columns were added
at the end (2026-09-26) so rows written before them still parse
(:func:`parse_trajectory` reads them with the metrics missing); an
existing file's header is upgraded in place when the next row is
appended.
"""
from __future__ import annotations

import gzip
import json
import math
import re
import statistics
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

TRAJECTORY_FILE = 'TRAJECTORY.md'
RUBRIC_MAX = 2                   # points per graded case
TRAJECTORY_COLUMNS = ('ts', 'run_id', 'model', 'passed/total',
                      'mean seconds', 'cost', 'note', 'tok/case',
                      'turns/case')
_OLD_TRAJECTORY_COLUMNS = TRAJECTORY_COLUMNS[:7]
_TRAJECTORY_INTRO = (
    '# Eval trajectory\n\n'
    'One row per recorded run (appended by `python -m guru.evals run`).\n\n')


def _header_lines(columns: tuple) -> tuple[str, str]:
    return ('| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns))


_TRAJECTORY_HEADER = (_TRAJECTORY_INTRO
                      + '\n'.join(_header_lines(TRAJECTORY_COLUMNS)) + '\n')
# Case metrics (``CaseResult.metrics``): the keys ``ledger.usage_metrics``
# writes; ``smells`` the three tool-usage smell counts the runner keeps.
SMELL_KEYS = ('whole_file_after_outline', 'repeated_calls', 'refused')


@dataclass
class CaseResult:
    """Outcome of one case: verdict, check rows, what was observed."""
    case: str
    passed: bool
    checks: list[dict[str, Any]]     # asdict(checks.CheckResult) rows
    observed: dict[str, Any]         # asdict(checks.Observed)
    rubric: str
    transcript_path: str
    cost_usd: Optional[float]
    # Distinct ``Adapter|model`` the case's sub-agent tasks ran on (from the
    # ledger ``tasks`` rows); empty when nothing was spawned or for run
    # files from before this field.
    routes: list[str] = field(default_factory=list)
    # The rubric judge's grade (0-2) and its one-line reason; None / '' when
    # the case has no rubric or no judge graded it. A failed grading keeps
    # ``rubric_score`` None and puts ``error: ...`` in ``rubric_reason``.
    rubric_score: Optional[int] = None
    rubric_reason: str = ''
    # Every sample score behind ``rubric_score`` when the judge was asked
    # more than once (``--samples N``; the score is their median); empty
    # for one sample and for run files from before this field.
    rubric_samples: list[int] = field(default_factory=list)
    # Model-agnostic cost of the case from its ledger rows
    # (``ledger.usage_metrics``: tokens in/out, cache read/write, their
    # total ``tokens``, ``tool_bytes`` shown, ``turns``, ``calls``) and the
    # tool-usage smell counts (``SMELL_KEYS``, from
    # ``ledger_report.tool_smells`` over the case's tool events). Empty for
    # run files from before these fields.
    metrics: dict[str, int] = field(default_factory=dict)
    smells: dict[str, int] = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        """Wall-clock seconds of the case (from ``observed``)."""
        return float(self.observed.get('seconds', 0.0))

    def metric(self, key: str) -> Optional[int]:
        """``metrics[key]`` as an int; None when the case has no metrics
        (an older run file)."""
        if not self.metrics or key not in self.metrics:
            return None
        return int(self.metrics[key])

    @property
    def tokens(self) -> Optional[int]:
        """Total tokens the case processed; None when unknown."""
        return self.metric('tokens')

    @property
    def turns(self) -> Optional[int]:
        """Agent-loop round trips of the case; None when unknown."""
        return self.metric('turns')

    def smell_total(self) -> int:
        """Sum of the recorded smell counts (0 without any)."""
        return sum(int(v) for v in self.smells.values())


@dataclass
class Run:
    """One suite run: which model, which commit, one result per case."""
    run_id: str
    ts: str                      # ISO-8601 UTC
    model: str
    git_sha: str
    cases: list[CaseResult]
    # Context window the suite's model was loaded at (0 = unknown; run
    # files from before this field load with 0).
    num_ctx: int = 0
    # Routing file stem the run used ('' = no routing, inert) and whether
    # the main agent ran as a controller; older run files load with the
    # defaults.
    routing: str = ''
    controller: bool = False
    # Judges the experiment file's [decisions] table installed for the run,
    # as ``point=judge name`` (empty: none configured or none available).
    judges: list[str] = field(default_factory=list)
    # The rubric judge's ``Adapter|model`` ('' = no grading this run) and
    # how many samples it gave per case (the median is recorded).
    rubric: str = ''
    rubric_samples: int = 1

    def model_label(self) -> str:
        """``Adapter|model`` plus ``@<ctx>`` when the context is known,
        ``+routed:<file>`` when a routing file was used and ``+controller``
        when the main agent was a controller."""
        label = ctx_label(self.num_ctx)
        out = f'{self.model}@{label}' if label else self.model
        if self.routing:
            out += f'+routed:{self.routing}'
        if self.controller:
            out += '+controller'
        return out

    def pass_rate(self) -> float:
        """Fraction of cases that passed; 0.0 for an empty run."""
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.passed) / len(self.cases)

    def total_cost(self) -> Optional[float]:
        """Sum of case costs; None unless every case has a known cost."""
        if not self.cases or any(c.cost_usd is None for c in self.cases):
            return None
        return sum(c.cost_usd for c in self.cases if c.cost_usd is not None)

    def mean_seconds(self) -> float:
        """Mean wall-clock seconds per case; 0.0 for an empty run."""
        if not self.cases:
            return 0.0
        return sum(c.seconds for c in self.cases) / len(self.cases)

    def rubric_total(self) -> Optional[tuple[int, int]]:
        """``(points, maximum)`` over the graded cases (two points per
        case); None when no case was graded."""
        graded = [c.rubric_score for c in self.cases
                  if c.rubric_score is not None]
        if not graded:
            return None
        return sum(graded), RUBRIC_MAX * len(graded)

    def total_metric(self, key: str) -> Optional[int]:
        """Sum of ``metrics[key]`` over the cases; None unless every case
        has it (an older run, or no cases)."""
        values = [c.metric(key) for c in self.cases]
        if not values or any(v is None for v in values):
            return None
        return sum(v for v in values if v is not None)

    def mean_metric(self, key: str) -> Optional[float]:
        """``total_metric(key)`` per case; None when unknown."""
        total = self.total_metric(key)
        return None if total is None else total / len(self.cases)

    def total_smells(self) -> int:
        """Sum of every case's smell counts."""
        return sum(c.smell_total() for c in self.cases)


def ctx_label(num_ctx: int) -> str:
    """``8192`` -> ``'8k'``, ``40960`` -> ``'40k'``, ``5000`` -> ``'5000'``,
    ``0`` -> ``''``."""
    if not num_ctx:
        return ''
    if num_ctx % 1024 == 0:
        return f'{num_ctx // 1024}k'
    return str(num_ctx)


def new_run_id() -> str:
    """A fresh 12-hex-digit run id."""
    return uuid.uuid4().hex[:12]


def now_ts() -> str:
    """Current UTC time as ISO-8601 with seconds precision."""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _stamp(ts: str) -> str:
    """``2026-09-23T10:00:00+00:00`` -> ``20260923T100000Z`` (UTC)."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime('%Y%m%dT%H%M%SZ')


def save(run: Run, directory: Path) -> Path:
    """Write ``run`` to ``directory/<stamp>-<run_id>.json``; return the path.

    The file carries ``passed``/``total`` summary keys next to the
    dataclass fields for humans and ``jq``; :func:`load` ignores them.
    Content that is not JSON-serialisable raises ``TypeError``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{_stamp(run.ts)}-{run.run_id}.json'
    data = asdict(run)
    data['passed'] = sum(1 for c in run.cases if c.passed)
    data['total'] = len(run.cases)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n',
                    encoding='utf-8')
    return path


def load(path: Path) -> Run:
    """Read a run file written by :func:`save`; ``ValueError`` if malformed."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        run_keys = {f.name for f in fields(Run)}
        case_keys = {f.name for f in fields(CaseResult)}
        kw = {k: v for k, v in data.items() if k in run_keys}
        kw['cases'] = [CaseResult(**{k: v for k, v in c.items()
                                     if k in case_keys})
                       for c in data['cases']]
        return Run(**kw)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise ValueError(f'{path}: not a run file: {e}') from e


def find_run(directory: Path, run_id: str) -> Path:
    """The run file ``<stamp>-<run_id>.json`` under ``directory`` (or
    ``run_id`` itself when it is a path to a run file); ``ValueError``
    when there is none, or more than one."""
    given = Path(run_id)
    if given.suffix == '.json' and given.is_file():
        return given
    hits = sorted(Path(directory).glob(f'*-{run_id}.json'))
    if not hits:
        raise ValueError(f'no run {run_id!r} under {directory}')
    if len(hits) > 1:
        raise ValueError(f'run id {run_id!r} is ambiguous under {directory}: '
                         + ', '.join(h.name for h in hits))
    return hits[0]


def model_slug(spec: str) -> str:
    """``'SBP Litellm|aws/claude-4-5-haiku'`` ->
    ``'sbp-litellm-aws-claude-4-5-haiku'``: the spec lowercased with every
    non-alphanumeric character replaced by ``-`` (the file stem
    ``bench/tool_contract.py`` writes under ``evals/models/``)."""
    return re.sub(r'[^a-z0-9]', '-', spec.lower())


def load_contract(models_dir: Path, spec: str) -> Optional[dict[str, Any]]:
    """The tool-contract record ``models_dir/<model_slug(spec)>.json``
    as a dict; None when there is no such file or it is not a JSON
    object (nothing here is required, the matrix shows ``-``)."""
    path = Path(models_dir) / f'{model_slug(spec)}.json'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_transcript(path: Path) -> list[dict[str, Any]]:
    """The agents of a ``<case>.json.gz`` transcript written by the runner
    (``[{'title', 'model', 'messages': [{'role', 'content', ...}]}]``);
    ``ValueError`` when the file is missing or not a transcript."""
    try:
        with gzip.open(Path(path), 'rt', encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, ValueError, EOFError) as e:
        raise ValueError(f'{path}: not a transcript: {e}') from e
    if not isinstance(data, list) or not all(
            isinstance(a, dict) and isinstance(a.get('messages'), list)
            for a in data):
        raise ValueError(f'{path}: not a transcript (expected a list of '
                         'agents with messages)')
    return data


def _main_messages(transcript: list[dict[str, Any]]) -> list[dict]:
    return [m for m in (transcript[0]['messages'] if transcript else [])
            if isinstance(m, dict)]


def transcript_prompt(transcript: list[dict[str, Any]]) -> str:
    """The user's prompt of a run: the main agent's first ``user``
    message; ``''`` when there is none."""
    for m in _main_messages(transcript):
        if m.get('role') == 'user':
            return str(m.get('content') or '')
    return ''


def transcript_answer(transcript: list[dict[str, Any]]) -> str:
    """The main agent's last non-empty ``assistant`` message (what the
    runner recorded as ``observed.answer``); ``''`` when there is none."""
    for m in reversed(_main_messages(transcript)):
        content = str(m.get('content') or '').strip()
        if m.get('role') == 'assistant' and content:
            return content
    return ''


def _delta(old: Optional[float], new: Optional[float]) -> Optional[float]:
    if old is None or new is None:
        return None
    return round(new - old, 6)


def compare(old: Run, new: Run) -> dict[str, Any]:
    """Diff two runs by case name.

    Returns ``newly_passing``, ``newly_failing``, ``still_failing``,
    ``still_passing``, ``added``, ``removed`` (sorted case names),
    ``pass_rate`` for both and per-case ``deltas`` of seconds and cost
    (``new - old``; cost None when either side is unknown).
    """
    o = {c.case: c for c in old.cases}
    n = {c.case: c for c in new.cases}
    both = sorted(set(o) & set(n))
    return {
        'old': old.run_id,
        'new': new.run_id,
        'newly_passing': [c for c in both if not o[c].passed and n[c].passed],
        'newly_failing': [c for c in both if o[c].passed and not n[c].passed],
        'still_failing': [c for c in both
                          if not o[c].passed and not n[c].passed],
        'still_passing': [c for c in both if o[c].passed and n[c].passed],
        'added': sorted(set(n) - set(o)),
        'removed': sorted(set(o) - set(n)),
        'pass_rate': {'old': old.pass_rate(), 'new': new.pass_rate()},
        'deltas': {c: {'seconds': _delta(o[c].seconds, n[c].seconds),
                       'cost_usd': _delta(o[c].cost_usd, n[c].cost_usd)}
                   for c in both},
    }


# --- repeated runs -----------------------------------------------------------

def _mean_spread(values: list[float]) -> tuple[float, float]:
    """``(mean, sample standard deviation)``; the spread is 0.0 for fewer
    than two values."""
    mean = statistics.fmean(values)
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, spread


def required_passes(repeats: int) -> int:
    """How often a case must pass over ``repeats`` runs: ``ceil(N/2)``
    (1 of 1, 2 of 3, 3 of 5)."""
    return math.ceil(max(int(repeats), 0) / 2)


def aggregate(run_list: list[Run]) -> dict[str, dict[str, Any]]:
    """Per case over several runs of the same selection.

    ``{case: {'passes', 'runs', 'seconds': (mean, spread), 'cost_usd':
    (mean, spread) | None, 'rubric_mean': float | None, 'tokens': (mean,
    spread) | None, 'turns': (mean, spread) | None, 'smells': int}}`` in
    the order the cases first appear. ``runs`` counts the runs the case
    appears in (a case missing from a run is not a failure, it is
    absent); the cost is None when any of its runs has no known cost, the
    token and turn pairs when any run lacks the metrics; ``rubric_mean``
    averages the graded runs only; ``smells`` sums the smell counts. The
    spread is the sample standard deviation (0.0 for a single run).
    """
    out: dict[str, dict[str, Any]] = {}
    per_case: dict[str, list[CaseResult]] = {}
    for run in run_list:
        for c in run.cases:
            per_case.setdefault(c.case, []).append(c)
    for name, results in per_case.items():
        costs = [c.cost_usd for c in results]
        graded = [float(c.rubric_score) for c in results
                  if c.rubric_score is not None]
        out[name] = {
            'passes': sum(1 for c in results if c.passed),
            'runs': len(results),
            'seconds': _mean_spread([c.seconds for c in results]),
            'cost_usd': (None if any(v is None for v in costs)
                         else _mean_spread([float(v) for v in costs
                                            if v is not None])),
            'rubric_mean': (statistics.fmean(graded) if graded else None),
            'tokens': _metric_spread(results, 'tokens'),
            'turns': _metric_spread(results, 'turns'),
            'smells': sum(c.smell_total() for c in results),
        }
    return out


def _metric_spread(results: list[CaseResult],
                   key: str) -> Optional[tuple[float, float]]:
    """``(mean, spread)`` of a metric over the results; None when any of
    them lacks it."""
    values = [c.metric(key) for c in results]
    if any(v is None for v in values):
        return None
    return _mean_spread([float(v) for v in values if v is not None])


def aggregate_ok(agg: dict[str, dict[str, Any]], repeats: int) -> bool:
    """True when every case passed at least :func:`required_passes` times
    (vacuously for an empty aggregate, as ``all`` over no cases)."""
    need = required_passes(repeats)
    return all(v['passes'] >= need for v in agg.values())


def _cell(text: str) -> str:
    return text.replace('|', '\\|').replace('\n', ' ')


def kilo(value: Optional[float]) -> str:
    """Tokens in thousands with one decimal (``12345`` -> ``'12.3'``);
    ``'-'`` for None."""
    return '-' if value is None else f'{value / 1000:.1f}'


def _upgrade_header(path: Path) -> None:
    """Rewrite the old seven-column header of an existing trajectory file
    to the current one (rows keep their cells; a renderer shows the old
    rows with empty metric columns)."""
    text = path.read_text(encoding='utf-8')
    old_head, old_rule = _header_lines(_OLD_TRAJECTORY_COLUMNS)
    new_head, new_rule = _header_lines(TRAJECTORY_COLUMNS)
    old = f'{old_head}\n{old_rule}\n'
    if old in text and new_head not in text:
        path.write_text(text.replace(old, f'{new_head}\n{new_rule}\n', 1),
                        encoding='utf-8')


def append_trajectory(run: Run, directory: Path, note: str = '') -> None:
    """Append one row for ``run`` to ``directory/TRAJECTORY.md``.

    The header is written once, when the file is created; a file with
    the pre-metrics header gets the current one first. The last two
    cells are the mean tokens per case in thousands and the mean turns
    per case (``-`` when the run has no metrics).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TRAJECTORY_FILE
    if path.is_file():
        _upgrade_header(path)
    cost = run.total_cost()
    turns = run.mean_metric('turns')
    fmt = ('| {ts} | {rid} | {model} | {p}/{t} | {secs:.1f} | {cost} '
           '| {note} | {tok} | {turns} |\n')
    row = fmt.format(
        ts=_cell(run.ts), rid=_cell(run.run_id),
        model=_cell(run.model_label()),
        p=sum(1 for c in run.cases if c.passed), t=len(run.cases),
        secs=run.mean_seconds(),
        cost='n/a' if cost is None else f'${cost:.2f}', note=_cell(note),
        tok=kilo(run.mean_metric('tokens')),
        turns='-' if turns is None else f'{turns:.1f}')
    with path.open('a', encoding='utf-8') as fh:
        if fh.tell() == 0:
            fh.write(_TRAJECTORY_HEADER)
        fh.write(row)


_UNESCAPED_PIPE = re.compile(r'(?<!\\)\|')


def _split_row(line: str) -> list[str]:
    """The cells of a Markdown table row (escaped pipes restored)."""
    inner = line.strip()
    if inner.startswith('|'):
        inner = inner[1:]
    if inner.endswith('|') and not inner.endswith('\\|'):
        inner = inner[:-1]
    return [c.strip().replace('\\|', '|')
            for c in _UNESCAPED_PIPE.split(inner)]


def parse_trajectory(path: Path) -> list[dict[str, str]]:
    """The data rows of a trajectory file as ``{column: cell}`` dicts
    keyed by :data:`TRAJECTORY_COLUMNS`.

    Rows written before the metric columns lack ``tok/case`` and
    ``turns/case`` (the keys are absent, not ``'-'``); header and rule
    lines are skipped. ``ValueError`` when the file cannot be read.
    """
    try:
        lines = Path(path).read_text(encoding='utf-8').splitlines()
    except OSError as e:
        raise ValueError(f'{path}: cannot read trajectory: {e}') from e
    out: list[dict[str, str]] = []
    for line in lines:
        if not line.startswith('|'):
            continue
        cells = _split_row(line)
        if not cells or cells[0] in ('ts', '') or set(cells[0]) <= {'-'}:
            continue
        out.append(dict(zip(TRAJECTORY_COLUMNS, cells)))
    return out
