"""Offline re-grading of a stored run (endpoint of ``guru.evals grade``).

No case is re-run: for every rubric case of a saved :class:`Run` the
answer comes from ``observed.answer`` in the run file (the transcript's
last main-agent assistant message when that field is empty), the user's
prompt from the transcript's first user message (the case file when the
transcript is gone), and the evidence block from the stored ``observed``
dict and cost — the same packet :func:`guru.evals.runner.grade_case`
sends. Each judge (``Adapter|model`` specs, resolved through the adapter
registry like the runner's judge) grades every case ``samples`` times
(:func:`guru.evals.rubric.grade_samples`; the recorded score is the
median, ties to the lower value, and a case is "stable" when all its
samples agree); the hand grade comes from ``evals/rubric-labels.toml``
(:mod:`guru.evals.labels`).

Hand grading: :func:`show_text` renders what a judge sees for one case
(prompt, rubric, evidence, answer) plus a ``[[label]]`` stub for
``evals/rubric-labels.toml``; ``python -m guru.evals grade RUN --show``
prints it per rubric case so a human can grade the same packet and grow
the hand-label set without a model call.

Everything graded is recorded in the run's own ledger directory
(``<out_root>/<run_id>/ledger``): one ``labels`` row per judge grade
(``target_id = <run_id>:<case>``, labeller ``rubric:<model>``, the label
the median and the note every sample's score and reason), one per
hand grade that applied (labeller ``hand``), and the judges' own ``calls``
rows, so the ledger report can score judges against the hand labels
later. The ledger is pointed at that directory for the duration and
restored afterwards.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

from guru import bench, config, judges, log
from guru.adapters.base import Adapter
from guru.domain import ledger
from guru.evals import cases, labels, rubric, runner, runs
from guru.evals.labels import HandLabel
from guru.evals.runs import CaseResult, Run
from guru.repositories.adapters import registry_from
from guru.repositories.jsonl_ledger import JsonlLedger
from guru.repositories.settings import RoutingSettings

LEDGER_DIR = 'ledger'          # under <out_root>/<run_id>/


@dataclass
class GradeRow:
    """One rubric case of the run: per judge spec its sampled grade (None
    when the judge failed; ``errors`` says why) and the hand grade, if
    any."""
    case: str
    grades: dict[str, Optional[rubric.SampledGrade]] = field(
        default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    hand: Optional[HandLabel] = None


@dataclass
class Regrade:
    """What :func:`regrade` produced: the rows, per spec ``(agreed,
    compared)`` of the median against the hand grades and ``(stable,
    graded)`` — the cases whose samples all agreed over the cases the
    judge graded — where the labels went, how many samples were asked
    for and what the judges' calls cost (None when unknown)."""
    run_id: str
    specs: list[str]
    rows: list[GradeRow]
    ledger_dir: Path
    agreement: dict[str, tuple[int, int]]
    stability: dict[str, tuple[int, int]] = field(default_factory=dict)
    samples: int = 1
    cost_usd: Optional[float] = None


def resolve_judges(specs: list[str], adapters: Optional[list[Adapter]] = None
                   ) -> list[tuple[str, rubric.LLMJudge]]:
    """``[(spec, judge)]`` for ``Adapter|model`` specs over the configured
    adapters (default: guru's, as the runner builds them); ``ValueError``
    for an empty list, a duplicate, or a spec that does not resolve. The
    adapter registry installed for the lookup is cleared again."""
    if not specs:
        raise ValueError('at least one --rubric SPEC is required')
    if len(set(specs)) != len(specs):
        raise ValueError('the same --rubric SPEC was given twice')
    if adapters is None:
        adapters = bench.build_adapters()
    judges.set_registry(registry_from(adapters), RoutingSettings())
    try:
        out: list[tuple[str, rubric.LLMJudge]] = []
        for spec in specs:
            judge = rubric.judge_from_spec(spec)
            if judge is None:
                raise ValueError(f'rubric judge {spec!r}: not '
                                 "'Adapter|model' or the adapter is not "
                                 'configured')
            out.append((spec, judge))
        return out
    finally:
        judges.set_registry(None)


@contextlib.contextmanager
def _ledger_at(repo: JsonlLedger) -> Iterator[None]:
    """Point the ledger at ``repo`` (enabled) and restore afterwards."""
    prev_repo, prev_enabled = ledger.repository(), config.LEDGER_ENABLED
    ledger.set_repository(repo)
    config.LEDGER_ENABLED = True
    try:
        yield
    finally:
        ledger.flush()
        ledger.set_repository(prev_repo)
        config.LEDGER_ENABLED = prev_enabled


def _transcript(res: CaseResult) -> list[dict]:
    """The case's transcript, or ``[]`` when it cannot be read."""
    try:
        return runs.load_transcript(Path(res.transcript_path))
    except ValueError as e:
        log.info('evals grade: %s', e)
        return []


def _case_prompt(name: str, cases_dir: Optional[Path]) -> str:
    """The prompt of the case file ``name`` (the fallback when the
    transcript is gone); ``''`` when the case cannot be loaded."""
    directory = Path(cases_dir) if cases_dir is not None else cases.CASES_DIR
    try:
        loaded = cases.load_cases(directory, names=[name])
    except ValueError:
        return ''
    return loaded[0].prompt if loaded else ''


def answer_and_prompt(res: CaseResult, cases_dir: Optional[Path] = None
                      ) -> tuple[str, str]:
    """``(answer, prompt)`` for a stored case: the answer from
    ``observed.answer``, else the transcript; the prompt from the
    transcript, else the case file, else ``''``."""
    answer = str(res.observed.get('answer') or '').strip()
    transcript: Optional[list[dict]] = None
    if not answer:
        transcript = _transcript(res)
        answer = runs.transcript_answer(transcript)
    if transcript is None:
        transcript = _transcript(res)
    prompt = runs.transcript_prompt(transcript).strip()
    if not prompt:
        prompt = _case_prompt(res.case, cases_dir)
    return answer, prompt


LABEL_STUB_NOTE = 'hand grade <date>: <why>'


def label_stub(case: str, run_id: str, score: Optional[int] = None,
               note: str = LABEL_STUB_NOTE) -> str:
    """A ``[[label]]`` table for ``evals/rubric-labels.toml`` pinned to
    ``run_id``; ``score`` is left as a ``0 | 1 | 2`` placeholder when
    None (the stub is valid TOML either way: the placeholder line is
    commented)."""
    lines = ['[[label]]', f'case = "{case}"', f'run = "{run_id}"']
    if score is None:
        lines.append('# score = 0 | 1 | 2   (uncomment and pick one)')
    else:
        lines.append(f'score = {int(score)}')
    lines.append(f'note = "{note}"')
    return '\n'.join(lines)


def show_text(run: Run, res: CaseResult, answer: str, prompt: str,
              hand: Optional[HandLabel] = None) -> str:
    """The packet for one rubric case as a human grader reads it: the
    prompt, the rubric, the evidence block (unfenced; the same lines the
    judge gets), the answer, the existing hand grade when there is one,
    and a :func:`label_stub` to append after grading."""
    head = f'=== {res.case} (run {run.run_id})'
    if hand is not None:
        head += (f' — hand grade {hand.score}'
                 + (f' for run {hand.run}' if hand.run != labels.ANY_RUN
                    else ' (any run)'))
    parts = [
        head,
        '', 'Prompt:', prompt.strip() or '(none recorded)',
        '', 'Rubric:', (res.rubric or '').strip() or '(empty rubric)',
        '', rubric.evidence(res.observed, res.cost_usd),
        '', 'Answer:', answer.strip() or '(empty answer)',
        '', 'Scale: 2 = meets the intent of every rubric point (an '
        'equivalent mechanism or identifier counts; what the evidence '
        'confirms is met without pasted output); 1 = one substantive '
        'point missing or wrong; 0 = wrong, unsupported or contradicted '
        'by the evidence. Do not penalise brevity, missing code listings '
        'or the spelling of identifiers.',
        '', f'Append to {labels.DEFAULT_LABELS_FILE.name} after grading:',
        label_stub(res.case, run.run_id),
    ]
    return '\n'.join(parts)


def _grade_one(res: CaseResult, answer: str, prompt: str,
               judge: rubric.Judge, samples: int = 1
               ) -> tuple[Optional[rubric.SampledGrade], str]:
    """One judge's ``samples`` grades of a stored case, or ``(None,
    error)`` when any sample failed; an empty answer scores 0 without a
    call, as the runner does."""
    if not answer:
        return runner.empty_answer_grade(samples), ''
    try:
        return rubric.grade_samples(
            prompt, res.rubric, answer, judge,
            evidence_text=rubric.evidence(res.observed, res.cost_usd),
            samples=samples), ''
    except Exception as e:                           # noqa: BLE001
        log.warning('evals grade: %s on %s failed: %s', judge.model,
                    res.case, e)
        return None, str(e) or type(e).__name__


def _score(grade: Optional[rubric.SampledGrade]) -> Optional[int]:
    return None if grade is None else grade.score


def stability(grades: Sequence[Optional[rubric.SampledGrade]]
              ) -> tuple[int, int]:
    """``(stable, graded)``: how many of the grades that exist have all
    their samples agreeing, over how many exist."""
    graded = [g for g in grades if g is not None]
    return sum(1 for g in graded if g.stable), len(graded)


def regrade(run: Run, out_root: Path,
            judge_list: Sequence[tuple[str, rubric.Judge]],
            hand: list[HandLabel], cases_dir: Optional[Path] = None,
            samples: int = 1) -> Regrade:
    """Grade every rubric case of ``run`` with each judge ``samples``
    times and record the grades (median; and the applicable hand grades)
    as ``labels`` rows in ``out_root/<run_id>/ledger``; see the module
    docstring. Never raises for a failing judge (the row's ``errors``
    says why); ``ValueError`` for ``samples < 1``."""
    if samples < 1:
        raise ValueError('samples must be at least 1')
    ledger_dir = Path(out_root) / run.run_id / LEDGER_DIR
    repo = JsonlLedger(ledger_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    calls_before = len(repo.rows('calls'))
    rows: list[GradeRow] = []
    with _ledger_at(repo):
        for res in run.cases:
            if not res.rubric:
                continue
            row = GradeRow(res.case,
                           hand=labels.hand_label(hand, res.case, run.run_id))
            target = f'{run.run_id}:{res.case}'
            answer, prompt = answer_and_prompt(res, cases_dir)
            for spec, judge in judge_list:
                grade, error = _grade_one(res, answer, prompt, judge,
                                          samples)
                row.grades[spec] = grade
                if grade is not None:
                    ledger.record_label(target, rubric.labeller(judge),
                                        str(grade.score), note=grade.note)
                else:
                    row.errors[spec] = error
            if row.hand is not None:
                ledger.record_label(target, labels.HAND_LABELLER,
                                    str(row.hand.score), note=row.hand.note)
            rows.append(row)
        ledger.flush()
    agreement = {
        spec: labels.agreement([
            (_score(r.grades.get(spec)),
             None if r.hand is None else r.hand.score) for r in rows])
        for spec, _ in judge_list}
    stable = {spec: stability([r.grades.get(spec) for r in rows])
              for spec, _ in judge_list}
    calls = repo.rows('calls')[calls_before:]
    cost = (None if not calls or any(c.get('cost_usd') is None
                                     for c in calls)
            else float(sum(c['cost_usd'] for c in calls)))
    return Regrade(run.run_id, [s for s, _ in judge_list], rows, ledger_dir,
                   agreement, stable, samples, cost)
