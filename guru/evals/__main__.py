"""``python -m guru.evals``: run the suite, compare two runs, list cases.

Plain-text output (like ``guru.bench``); ``run`` exits 1 when any case
failed so it can gate a script. ``--repeat N`` runs the selection N times
(one run file each) and prints an aggregate; the gate is then that every
case passed at least ``ceil(N/2)`` times. ``--rubric 'Adapter|model'``
grades the rubric cases with a model (default with ``--allow-spend`` and
``--routing``: the routing file's cheapest rung; ``--rubric none`` turns
that off); a grade is reported, and fails the case only under
``--rubric-min N``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from guru import config
from guru.evals import cases, runner, runs
from guru.evals.runs import CaseResult, Run
from guru.repositories.settings import RoutingSettings

DEFAULT_OUT = cases.REPO_ROOT / 'evals' / 'runs'
RUBRIC_OFF = 'none'           # --rubric none: no grading, even with a default


def _fmt_cost(cost: Optional[float]) -> str:
    return 'n/a' if cost is None else f'${cost:.3f}'


def _table(headers: list, rows: list) -> str:
    """Fixed-width text table; the last column is left ragged."""
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows))
              if rows else len(str(h)) for i, h in enumerate(headers)]
    line = '  '.join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line.rstrip(), '  '.join('-' * w for w in widths)]
    for r in rows:
        out.append('  '.join(str(c).ljust(w)
                             for c, w in zip(r, widths)).rstrip())
    return '\n'.join(out)


def _counted(names: list) -> str:
    """``['a', 'b', 'b']`` -> ``'a, b(2)'`` (first-appearance order)."""
    counts: dict = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return ', '.join(n if c == 1 else f'{n}({c})' for n, c in counts.items())


def _timeout_detail(observed: dict) -> str:
    """What a timed-out case did, so triage does not need the run JSON."""
    tools = _counted(observed.get('tools_used') or []) or '-'
    changed = ', '.join(observed.get('files_changed') or []) or '-'
    return (f'timed out: tools={tools}; spawned={observed.get("spawned", 0)}'
            f'; files_changed={changed}')


def _row(res: CaseResult, routed: bool = False) -> list:
    """One table row: case, verdict, seconds, cost, failed checks/rubric.

    A timed-out case fails every check by design; its row shows the
    observed tools, sub-agents and changed files instead of the check list.
    For a routed run the detail also lists where the sub-agent tasks went
    (``routes: Adapter|model, ...``; ``-`` when nothing was spawned); a
    sandbox case lists its gate verdicts (``gate: intended``).
    """
    parts: list = []
    if res.observed.get('timed_out'):
        parts.append(_timeout_detail(res.observed))
    else:
        failed = [c['name'] for c in res.checks if not c['passed']]
        if failed:
            parts.append(', '.join(failed))
    if res.rubric:
        parts.append(_rubric_detail(res))
    if routed or res.routes:
        parts.append('routes: ' + (', '.join(res.routes) or '-'))
    verdicts = res.observed.get('gate_verdicts') or []
    if verdicts:
        parts.append('gate: ' + ', '.join(verdicts))
    return [res.case, 'PASS' if res.passed else 'FAIL',
            f'{res.seconds:.1f}', _fmt_cost(res.cost_usd), '; '.join(parts)]


def _rubric_detail(res: CaseResult) -> str:
    """``rubric: 2/2`` when graded, ``rubric: error`` when the judge
    failed, ``rubric: grade by hand`` when nothing graded it."""
    if res.rubric_score is not None:
        return f'rubric: {res.rubric_score}/{runs.RUBRIC_MAX}'
    if res.rubric_reason.startswith('error:'):
        return 'rubric: error'
    return 'rubric: grade by hand'


def _gate_counts(run: Run) -> list:
    """``[(verdict, count)]`` over every case's gate verdicts (sandbox
    cases), in verdict order; empty when no case submitted anything."""
    counts: dict = {}
    for c in run.cases:
        for v in c.observed.get('gate_verdicts') or []:
            counts[v] = counts.get(v, 0) + 1
    return [(v, counts[v]) for v in ('intended', 'unclear', 'suspicious')
            if v in counts]


def _print_run(run: Run, out_root: Path) -> None:
    routed = bool(run.routing)
    print(_table(['case', 'result', 'seconds', 'cost', 'detail'],
                 [_row(c, routed) for c in run.cases]))
    passed = sum(1 for c in run.cases if c.passed)
    summary = (f'\npassed {passed}/{len(run.cases)}'
               f' · mean {run.mean_seconds():.1f}s'
               f' · cost {_fmt_cost(run.total_cost())}'
               f' · model {run.model_label()}')
    if routed:
        summary += f' · routing {run.routing}'
        if run.controller:
            summary += ' (controller)'
    if run.judges:
        summary += ' · judges ' + ', '.join(run.judges)
    total = run.rubric_total()
    if total is not None:
        summary += f' · rubric {total[0]}/{total[1]}'
    counts = _gate_counts(run)
    if counts:
        summary += ' · gate ' + ' '.join(f'{k}={v}' for k, v in counts)
    print(summary)
    trajectory = runner.DEFAULT_TRAJECTORY_DIR / runs.TRAJECTORY_FILE
    print(f'run {run.run_id} saved under {out_root} '
          f'(transcripts: {out_root / run.run_id / "transcripts"}; '
          f'trajectory: {trajectory})')


def _csv(text: Optional[str]) -> Optional[list]:
    """``'a, b'`` -> ``['a', 'b']``; empty/None -> None (no filter)."""
    items = [n.strip() for n in (text or '').split(',') if n.strip()]
    return items or None


def _load_suite(args: argparse.Namespace) -> list:
    """The cases selected by ``--cases``/``--tags``; ``ValueError`` when a
    name or tag is unknown."""
    return cases.load_cases(Path(args.cases_dir), names=_csv(args.cases),
                            tags=_csv(args.tags))


def _pm(pair: Optional[tuple], fmt: str) -> str:
    """``mean ± spread`` with ``fmt`` applied to both; ``n/a`` for None."""
    if pair is None:
        return 'n/a'
    mean, spread = pair
    return f'{fmt.format(mean)} ± {fmt.format(spread)}'


def _print_aggregate(run_list: list, repeats: int) -> None:
    """The ``--repeat`` table: per case the pass count, cost and seconds
    as mean ± spread (sample standard deviation) and the mean rubric."""
    agg = runs.aggregate(run_list)
    rows = []
    for name, v in agg.items():
        mean_rubric = v['rubric_mean']
        rows.append([name, f"{v['passes']}/{v['runs']}",
                     _pm(v['cost_usd'], '${:.3f}'),
                     _pm(v['seconds'], '{:.1f}'),
                     '-' if mean_rubric is None
                     else f'{mean_rubric:.1f}/{runs.RUBRIC_MAX}'])
    need = runs.required_passes(repeats)
    weak = [n for n, v in agg.items() if v['passes'] < need]
    print(f'\naggregate over {len(run_list)} run(s):')
    print(_table(['case', 'passes', 'cost', 'seconds', 'rubric'], rows))
    print(f'gate: every case must pass at least {need}/{repeats}'
          + (f' — below: {", ".join(weak)}' if weak else ' — ok'))
    print('runs: ' + ', '.join(r.run_id for r in run_list))


def _resolve_rubric(args: argparse.Namespace,
                    routing: Optional[RoutingSettings]) -> str:
    """The rubric judge spec for the run: the flag, else (with
    ``--allow-spend`` and a routing file) the file's cheapest rung; ``''``
    for no grading (``--rubric none`` or no default)."""
    if args.rubric is not None:
        spec = args.rubric.strip()
        return '' if spec.lower() == RUBRIC_OFF else spec
    if args.allow_spend and routing is not None:
        return runner.default_rubric_spec(routing)
    return ''


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        suite = _load_suite(args)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    if not suite:
        print('error: no cases found', file=sys.stderr)
        return 2
    # Flags win; then [evals] in settings.toml; then guru's own defaults.
    model = args.model or config.EVALS_MODEL or None
    num_ctx = config.EVALS_NUM_CTX if args.num_ctx is None else args.num_ctx
    if num_ctx < 0:
        print('error: --num-ctx must be 0 (auto-fit) or positive',
              file=sys.stderr)
        return 2
    if args.repeat < 1:
        print('error: --repeat must be at least 1', file=sys.stderr)
        return 2
    if args.rubric_min is not None and \
            args.rubric_min not in range(runs.RUBRIC_MAX + 1):
        print(f'error: --rubric-min must be 0..{runs.RUBRIC_MAX}',
              file=sys.stderr)
        return 2
    out_root = Path(args.out)
    routing = None
    routing_name = ''
    decisions = None
    if args.routing:
        try:
            routing = runner.load_routing_file(Path(args.routing))
            decisions = runner.load_decisions_file(Path(args.routing))
        except ValueError as e:
            print(f'error: {e}', file=sys.stderr)
            return 2
        routing_name = Path(args.routing).stem
    rubric_spec = _resolve_rubric(args, routing)
    if args.rubric_min is not None and not rubric_spec:
        print('error: --rubric-min needs a rubric judge (--rubric, or '
              '--allow-spend with --routing)', file=sys.stderr)
        return 2

    def progress(res: CaseResult) -> None:
        flags = ', timed out' if res.observed.get('timed_out') else ''
        if res.observed.get('skipped'):
            flags += f', skipped: {res.observed.get("error", "")}'
        if res.rubric_score is not None:
            flags += f', rubric {res.rubric_score}/{runs.RUBRIC_MAX}'
        print(f'[evals] {res.case}: {"PASS" if res.passed else "FAIL"}'
              f' ({res.seconds:.1f}s{flags})', flush=True)

    run_list: list = []
    for i in range(1, args.repeat + 1):
        note = args.note
        if args.repeat > 1:
            note = f'{note} (repeat {i}/{args.repeat})' if note \
                else f'repeat {i}/{args.repeat}'
            print(f'[evals] repeat {i}/{args.repeat}', flush=True)
        try:
            run = runner.run_suite(suite, model, out_root, note=note,
                                   on_result=progress, num_ctx=num_ctx,
                                   routing=routing,
                                   routing_name=routing_name,
                                   allow_spend=args.allow_spend,
                                   decisions=decisions,
                                   rubric_spec=rubric_spec,
                                   rubric_min=args.rubric_min)
        except ValueError as e:
            print(f'error: {e}', file=sys.stderr)
            return 2
        _print_run(run, out_root)
        run_list.append(run)
    if args.repeat > 1:
        _print_aggregate(run_list, args.repeat)
    return 0 if runs.aggregate_ok(runs.aggregate(run_list),
                                  args.repeat) else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        old = runs.load(Path(args.old))
        new = runs.load(Path(args.new))
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    diff = runs.compare(old, new)
    rates = diff['pass_rate']
    print(f'old {old.run_id} ({old.ts}, {old.model}): {rates["old"]:.0%}')
    print(f'new {new.run_id} ({new.ts}, {new.model}): {rates["new"]:.0%}')
    for key in ('newly_passing', 'newly_failing', 'still_failing',
                'still_passing', 'added', 'removed'):
        print(f'{key.replace("_", " ")}: '
              f'{", ".join(diff[key]) or "-"}')
    rows = []
    for name, d in sorted(diff['deltas'].items()):
        secs = d['seconds']
        cost = d['cost_usd']
        rows.append([name, '-' if secs is None else f'{secs:+.1f}',
                     '-' if cost is None else f'{cost:+.2f}'])
    if rows:
        print()
        print(_table(['case', 'seconds delta', 'cost delta'], rows))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    try:
        suite = _load_suite(args)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    rows = [[c.name, c.fixture_label, c.mode, c.model, ','.join(c.tags) or '-',
             c.prompt if len(c.prompt) <= 50 else c.prompt[:47] + '...']
            for c in suite]
    print(_table(['case', 'fixture', 'mode', 'model', 'tags', 'prompt'],
                 rows))
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m guru.evals',
        description='Functional evaluation suite (see evals/README.md).')
    sub = p.add_subparsers(dest='cmd', required=True)

    run_p = sub.add_parser('run', help='run cases and record a run file')
    run_p.add_argument('--cases', default='',
                       help='comma-separated case names (default: all)')
    run_p.add_argument('--tags', default='',
                       help='comma-separated tags; cases having any of them '
                            '(combinable with --cases; e.g. --tags fast)')
    run_p.add_argument('--model', default=None,
                       help="'Adapter|model' (default: [evals] model in "
                            "settings.toml, else guru's default)")
    run_p.add_argument('--num-ctx', type=int, default=None,
                       help='context window to load the model at; 0 = GPU '
                            'auto-fit (default: [evals] num_ctx, else '
                            f'{config.EVALS_NUM_CTX})')
    run_p.add_argument('--out', default=str(DEFAULT_OUT),
                       help=f'run directory (default: {DEFAULT_OUT})')
    run_p.add_argument('--note', default='',
                       help='free text for the trajectory row')
    run_p.add_argument('--routing', default=None, metavar='FILE',
                       help='experiment TOML: a [routing] table (as in '
                            'settings.toml) to route sub-agents, plus an '
                            'optional [decisions] table whose judges are '
                            'installed for the run; see '
                            'evals/routing/README.md')
    run_p.add_argument('--allow-spend', action='store_true',
                       help='grant the remote-spend question for the run '
                            '(default: deny, so remote rungs are skipped) '
                            'and let sandbox cases apply an "intended" '
                            'submit (default: deny, nothing is applied)')
    run_p.add_argument('--rubric', default=None, metavar='SPEC',
                       help="'Adapter|model' that grades the rubric cases "
                            "0-2 (default: with --allow-spend and "
                            "--routing, the routing file's cheapest rung; "
                            f"'{RUBRIC_OFF}' turns grading off)")
    run_p.add_argument('--rubric-min', type=int, default=None, metavar='N',
                       help='fail a rubric case graded below N (default: '
                            'a grade is reported only)')
    run_p.add_argument('--repeat', type=int, default=1, metavar='N',
                       help='run the selection N times (one run file each) '
                            'and print an aggregate; exit 1 when a case '
                            'passed fewer than ceil(N/2) times (default 1)')
    run_p.add_argument('--cases-dir', default=str(cases.CASES_DIR),
                       help=argparse.SUPPRESS)
    run_p.set_defaults(func=_cmd_run)

    cmp_p = sub.add_parser('compare', help='diff two run files')
    cmp_p.add_argument('old')
    cmp_p.add_argument('new')
    cmp_p.set_defaults(func=_cmd_compare)

    list_p = sub.add_parser('list', help='list the cases')
    list_p.add_argument('--cases', default='',
                        help='comma-separated case names (default: all)')
    list_p.add_argument('--tags', default='',
                        help='comma-separated tags; cases having any of them')
    list_p.add_argument('--cases-dir', default=str(cases.CASES_DIR),
                        help=argparse.SUPPRESS)
    list_p.set_defaults(func=_cmd_list)
    return p


def main(argv: Optional[list] = None) -> int:
    """Entry point; returns the exit code."""
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':
    sys.exit(main())
