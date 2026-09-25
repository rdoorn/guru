"""Offline re-grading of a stored run (endpoint of ``guru.evals grade``).

No case is re-run: for every rubric case of a saved :class:`Run` the
answer comes from ``observed.answer`` in the run file (the transcript's
last main-agent assistant message when that field is empty), the user's
prompt from the transcript's first user message (the case file when the
transcript is gone), and the evidence block from the stored ``observed``
dict and cost — the same packet :func:`guru.evals.runner.grade_case`
sends. Each judge (``Adapter|model`` specs, resolved through the adapter
registry like the runner's judge) grades every case; the hand grade comes
from ``evals/rubric-labels.toml`` (:mod:`guru.evals.labels`).

Everything graded is recorded in the run's own ledger directory
(``<out_root>/<run_id>/ledger``): one ``labels`` row per judge grade
(``target_id = <run_id>:<case>``, labeller ``rubric:<model>``), one per
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
from guru.evals import cases, labels, rubric, runs
from guru.evals.labels import HandLabel
from guru.evals.runs import CaseResult, Run
from guru.repositories.adapters import registry_from
from guru.repositories.jsonl_ledger import JsonlLedger
from guru.repositories.settings import RoutingSettings

LEDGER_DIR = 'ledger'          # under <out_root>/<run_id>/


@dataclass
class GradeRow:
    """One rubric case of the run: per judge spec its grade (None when the
    judge failed; ``errors`` says why) and the hand grade, if any."""
    case: str
    grades: dict[str, Optional[rubric.Grade]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    hand: Optional[HandLabel] = None


@dataclass
class Regrade:
    """What :func:`regrade` produced: the rows, per spec ``(agreed,
    compared)`` against the hand grades, where the labels went and what
    the judges' calls cost (None when unknown)."""
    run_id: str
    specs: list[str]
    rows: list[GradeRow]
    ledger_dir: Path
    agreement: dict[str, tuple[int, int]]
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
        adapters = bench._build_adapters()
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


def _grade_one(res: CaseResult, answer: str, prompt: str,
               judge: rubric.Judge) -> tuple[Optional[rubric.Grade], str]:
    """One judge's grade of a stored case, or ``(None, error)``; an empty
    answer scores 0 without a call, as the runner does."""
    if not answer:
        return rubric.Grade(0, 'empty answer'), ''
    try:
        return rubric.grade(
            prompt, res.rubric, answer, judge,
            evidence_text=rubric.evidence(res.observed, res.cost_usd)), ''
    except Exception as e:                           # noqa: BLE001
        log.warning('evals grade: %s on %s failed: %s', judge.model,
                    res.case, e)
        return None, str(e) or type(e).__name__


def _score(grade: Optional[rubric.Grade]) -> Optional[int]:
    return None if grade is None else grade.score


def regrade(run: Run, out_root: Path,
            judge_list: Sequence[tuple[str, rubric.Judge]],
            hand: list[HandLabel], cases_dir: Optional[Path] = None
            ) -> Regrade:
    """Grade every rubric case of ``run`` with each judge and record the
    grades (and the applicable hand grades) as ``labels`` rows in
    ``out_root/<run_id>/ledger``; see the module docstring. Never raises
    for a failing judge (the row's ``errors`` says why)."""
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
                grade, error = _grade_one(res, answer, prompt, judge)
                row.grades[spec] = grade
                if grade is not None:
                    ledger.record_label(target, rubric.labeller(judge),
                                        str(grade.score), note=grade.reason)
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
    calls = repo.rows('calls')[calls_before:]
    cost = (None if not calls or any(c.get('cost_usd') is None
                                     for c in calls)
            else float(sum(c['cost_usd'] for c in calls)))
    return Regrade(run.run_id, [s for s, _ in judge_list], rows, ledger_dir,
                   agreement, cost)
