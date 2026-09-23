"""``python -m guru.evals``: run the suite, compare two runs, list cases.

Plain-text output (like ``guru.bench``); ``run`` exits 1 when any case
failed so it can gate a script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from guru import config
from guru.evals import cases, runner, runs
from guru.evals.runs import CaseResult, Run

DEFAULT_OUT = cases.REPO_ROOT / 'evals' / 'runs'


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
    (``routes: Adapter|model, ...``; ``-`` when nothing was spawned).
    """
    parts: list = []
    if res.observed.get('timed_out'):
        parts.append(_timeout_detail(res.observed))
    else:
        failed = [c['name'] for c in res.checks if not c['passed']]
        if failed:
            parts.append(', '.join(failed))
    if res.rubric:
        parts.append('rubric: grade by hand')
    if routed or res.routes:
        parts.append('routes: ' + (', '.join(res.routes) or '-'))
    return [res.case, 'PASS' if res.passed else 'FAIL',
            f'{res.seconds:.1f}', _fmt_cost(res.cost_usd), '; '.join(parts)]


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
    out_root = Path(args.out)
    routing = None
    routing_name = ''
    if args.routing:
        try:
            routing = runner.load_routing_file(Path(args.routing))
        except ValueError as e:
            print(f'error: {e}', file=sys.stderr)
            return 2
        routing_name = Path(args.routing).stem

    def progress(res: CaseResult) -> None:
        flags = ', timed out' if res.observed.get('timed_out') else ''
        print(f'[evals] {res.case}: {"PASS" if res.passed else "FAIL"}'
              f' ({res.seconds:.1f}s{flags})', flush=True)

    try:
        run = runner.run_suite(suite, model, out_root, note=args.note,
                               on_result=progress, num_ctx=num_ctx,
                               routing=routing, routing_name=routing_name,
                               allow_spend=args.allow_spend)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    _print_run(run, out_root)
    return 0 if all(c.passed for c in run.cases) else 1


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
    rows = [[c.name, c.fixture, c.mode, c.model, ','.join(c.tags) or '-',
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
                       help='TOML file with a [routing] table (as in '
                            'settings.toml) to route sub-agents; see '
                            'evals/routing/README.md')
    run_p.add_argument('--allow-spend', action='store_true',
                       help='grant the remote-spend question for the run '
                            '(default: deny, so remote rungs are skipped)')
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
