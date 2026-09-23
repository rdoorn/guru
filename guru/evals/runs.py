"""Run files, run-to-run comparison and the trajectory table.

Repository layer: JSON under ``evals/runs/<ts>-<run_id>.json`` and the
Markdown table ``evals/TRAJECTORY.md``. Nothing here runs a case; the
runner hands over a :class:`Run`.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

TRAJECTORY_FILE = 'TRAJECTORY.md'
_TRAJECTORY_HEADER = (
    '# Eval trajectory\n\n'
    'One row per recorded run (appended by `python -m guru.evals run`).\n\n'
    '| ts | run_id | model | passed/total | mean seconds | cost | note |\n'
    '|---|---|---|---|---|---|---|\n')


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

    @property
    def seconds(self) -> float:
        """Wall-clock seconds of the case (from ``observed``)."""
        return float(self.observed.get('seconds', 0.0))


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

    def model_label(self) -> str:
        """``Adapter|model`` plus ``@<ctx>`` when the context is known."""
        label = ctx_label(self.num_ctx)
        return f'{self.model}@{label}' if label else self.model

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


def _cell(text: str) -> str:
    return text.replace('|', '\\|').replace('\n', ' ')


def append_trajectory(run: Run, directory: Path, note: str = '') -> None:
    """Append one row for ``run`` to ``directory/TRAJECTORY.md``.

    The header is written once, when the file is created.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TRAJECTORY_FILE
    cost = run.total_cost()
    fmt = ('| {ts} | {rid} | {model} | {p}/{t} | {secs:.1f} | {cost} '
           '| {note} |\n')
    row = fmt.format(
        ts=_cell(run.ts), rid=_cell(run.run_id),
        model=_cell(run.model_label()),
        p=sum(1 for c in run.cases if c.passed), t=len(run.cases),
        secs=run.mean_seconds(),
        cost='n/a' if cost is None else f'${cost:.2f}', note=_cell(note))
    with path.open('a', encoding='utf-8') as fh:
        if fh.tell() == 0:
            fh.write(_TRAJECTORY_HEADER)
        fh.write(row)
