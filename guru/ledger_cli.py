"""``python -m guru.ledger_cli``: the review loop over the ledger.

    review --point stall [--n 50] [--dir DIR]   label decision rows y/n/s/q
    report [--point stall] [--dir DIR]           judge vs heuristic vs labels
    tasks [--unlabelled] [--n 20] [--dir DIR]    recent tasks for triage

``review`` shows the most recent unlabelled decision rows for a point
(input, judge P(yes) and verdict, heuristic) and writes one ``labels`` row
per answer through :func:`guru.domain.ledger.record_label`: ``target_id``
is :func:`guru.domain.ledger_report.decision_key`
(``<point>:<question>:<input_sha>``), labeller ``user``, label ``yes`` /
``no`` (the correct answer to the question), note
``point:<point>;question:<id>``. ``report`` renders
:func:`guru.domain.ledger_report.judge_metrics`. The aggregation is pure
(``guru.domain.ledger_report``); this module only loads, prompts and prints.
Design: docs/review-loop.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

from guru import config
from guru.domain import decisions, ledger, ledger_report
from guru.repositories.jsonl_ledger import JsonlLedger

Ask = Callable[[str], str]
Out = Callable[..., None]

_ANSWERS = {'y': 'yes', 'yes': 'yes', 'n': 'no', 'no': 'no',
            's': 'skip', '': 'skip', 'skip': 'skip', 'q': 'quit',
            'quit': 'quit'}
_PROMPT = '  correct answer? [y]es / [n]o / [s]kip / [q]uit: '


def _hints() -> dict:
    """``question id -> the affirmative hypothesis`` for guru's points."""
    qs = [decisions.stall_question(''), decisions.injection_question('')]
    qs += decisions.panel_questions('')
    return {q.id: q.hypothesis for q in qs}


def _fmt_p(row: dict) -> str:
    """The row's ``P(yes)`` with two decimals, or ``n/a``."""
    dist = row.get('dist')
    p = dist.get('yes') if isinstance(dist, dict) else None
    return 'n/a' if not isinstance(p, (int, float)) else f'{p:.2f}'


def _show_decision(row: dict, i: int, total: int, hints: dict,
                   out: Out) -> None:
    """Print one decision row for labelling."""
    out(f"[{i}/{total}] {row.get('point')}/{row.get('question')}  "
        f"{row.get('ts') or ''}  judge={row.get('judge')}  "
        f"P(yes)={_fmt_p(row)} chosen={row.get('chosen')}  "
        f"heuristic={row.get('heuristic')}  used={row.get('used') or '?'}"
        + (f" fallback={row['fallback_reason']}"
           if row.get('fallback_reason') else ''))
    hint = hints.get(str(row.get('question')))
    if hint:
        out(f'  question: {hint}')
    for line in str(row.get('input_head') or '').splitlines():
        out(f'  | {line}')


def _answer(ask: Ask, out: Out) -> str:
    """One of ``yes`` / ``no`` / ``skip`` / ``quit``; re-asks on anything
    else; EOF or Ctrl-C quits."""
    while True:
        try:
            raw = ask(_PROMPT)
        except (EOFError, KeyboardInterrupt):
            return 'quit'
        answer = _ANSWERS.get(raw.strip().lower())
        if answer is not None:
            return answer
        out('  answer y, n, s or q')


def cmd_review(directory: Path, point: str, n: int, *, ask: Ask,
               out: Out) -> int:
    """Interactive labelling of ``point``'s most recent decision rows."""
    repo = JsonlLedger(directory)
    queue = ledger_report.review_queue(repo.rows('decisions'),
                                       repo.rows('labels'), point=point, n=n)
    if not queue:
        out(f'nothing to review for point {point!r} in {directory}')
        return 0
    hints = _hints()
    labelled = skipped = 0
    previous, enabled = ledger.repository(), config.LEDGER_ENABLED
    ledger.set_repository(repo)
    config.LEDGER_ENABLED = True          # the user asked to label
    try:
        for i, row in enumerate(queue, 1):
            _show_decision(row, i, len(queue), hints, out)
            answer = _answer(ask, out)
            if answer == 'quit':
                break
            if answer == 'skip':
                skipped += 1
                continue
            ledger.record_label(ledger_report.decision_key(row), 'user',
                                answer, ledger_report.review_note(row))
            labelled += 1
    finally:
        ledger.flush()
        ledger.set_repository(previous)
        config.LEDGER_ENABLED = enabled
    out(f'labelled {labelled}, skipped {skipped} of {len(queue)} '
        f'({point})')
    return 0


def cmd_report(directory: Path, point: Optional[str], *, out: Out) -> int:
    """Per point and judge: agreement, precision/recall/F1 vs labels,
    suggested threshold."""
    repo = JsonlLedger(directory)
    metrics = ledger_report.judge_metrics(repo.rows('decisions'),
                                          repo.rows('labels'), point=point)
    out(ledger_report.render_metrics(metrics).rstrip('\n'))
    return 0


def _money(v: object) -> str:
    """``$0.1234`` or ``n/a`` for an unknown cost."""
    return 'n/a' if not isinstance(v, (int, float)) else f'${v:.4f}'


def _secs(v: object) -> str:
    """``12.3s`` or ``n/a``."""
    return 'n/a' if not isinstance(v, (int, float)) else f'{v:.1f}s'


def cmd_tasks(directory: Path, n: int, unlabelled: bool, *, out: Out) -> int:
    """Recent finished tasks (optionally only unlabelled ones) for triage."""
    repo = JsonlLedger(directory)
    labels = repo.rows('labels') if unlabelled else []
    rows = ledger_report.unlabelled_tasks(repo.rows('tasks'), labels, n=n)
    if not rows:
        out('no tasks' + (' without a label' if unlabelled else '')
            + f' in {directory}')
        return 0
    for t in rows:
        out(f"task {t['task_id']}  {t['status']}  {_secs(t['seconds'])}  "
            f"{_money(t['cost_usd'])}  {t['route']}  "
            f"{t['kind'] or '-'}/{t['complexity'] or '-'}  {t['ts']}")
        if t['reason']:
            out(f"  reason: {', '.join(str(r) for r in t['reason'])}")
        out(f"  task: {t['task'][:300]}")
        if t['transcript_path']:
            out(f"  transcript: {t['transcript_path']}")
        out('')
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m guru.ledger_cli',
        description='Review loop over the guru ledger: label decision rows, '
                    'score judges, triage tasks.')
    sub = p.add_subparsers(dest='cmd', required=True)

    def _dir(sp: argparse.ArgumentParser) -> None:
        sp.add_argument('--dir', type=Path, default=config.LEDGER_DIR,
                        help=f'ledger directory (default {config.LEDGER_DIR})')
    r = sub.add_parser('review', help='label decision rows y/n/s/q')
    r.add_argument('--point', required=True, help='decision point, e.g. stall')
    r.add_argument('--n', type=int, default=50,
                   help='most recent unlabelled rows to show (default 50)')
    _dir(r)
    rp = sub.add_parser('report', help='judge vs heuristic vs labels')
    rp.add_argument('--point', default=None, help='only this decision point')
    _dir(rp)
    t = sub.add_parser('tasks', help='recent tasks for triage')
    t.add_argument('--unlabelled', action='store_true',
                   help='skip tasks that already have a labels row')
    t.add_argument('--n', type=int, default=20,
                   help='how many tasks (default 20)')
    _dir(t)
    return p


def main(argv: Optional[list] = None, *, ask: Ask = input,
         out: Out = print) -> int:
    """Entry point; ``ask`` and ``out`` are injectable for tests."""
    args = _parser().parse_args(argv)
    if args.cmd == 'review':
        return cmd_review(args.dir, args.point, args.n, ask=ask, out=out)
    if args.cmd == 'report':
        return cmd_report(args.dir, args.point, out=out)
    return cmd_tasks(args.dir, args.n, args.unlabelled, out=out)


if __name__ == '__main__':
    raise SystemExit(main())
