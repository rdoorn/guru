"""``python -m guru.evals``: run the suite, compare two runs, list cases,
re-grade a stored run, run a model matrix.

Plain-text output (like ``guru.bench``); ``run`` exits 1 when any case
failed so it can gate a script. The table carries the model-agnostic
metrics per case — ``tok`` (thousands of tokens processed), ``turns``
(agent-loop round trips) and ``smells`` (whole-file reads after an
outline, repeated identical calls, refused calls) — and the summary line
their totals. ``--repeat N`` runs the selection N times
(one run file each) and prints an aggregate; the gate is then that every
case passed at least ``ceil(N/2)`` times. ``matrix --models 'A|m1,B|m2'``
runs the selection once per model as the main model (the routing file,
when given, routes the sub-agents as usual) and prints one row per model:
passed, tokens, tool bytes, seconds, cost and the tool-contract score
from ``evals/models/<slug>.json`` when ``bench/tool_contract.py`` wrote
one; it compares and never gates (exit 0). ``--rubric 'Adapter|model'``
grades the rubric cases with a model (default whenever ``--allow-spend``
is given: the routing file's cheapest rung, else the cheapest Claude tier
of the first enabled remote adapter; ``--rubric none`` turns that off); a
grade is reported, and fails the case only under ``--rubric-min N``.
``grade RUN_ID --rubric SPEC [--rubric SPEC2]`` re-grades a stored run
offline (no case re-runs) with one column per judge next to the hand grade
from ``evals/rubric-labels.toml``; ``grade RUN_ID --show`` prints what the
judge sees per case (prompt, rubric,
evidence, answer) with a ``[[label]]`` stub, for grading by hand.
``--samples N`` (both commands) asks each judge N times per case; the
recorded score is the median (ties to the lower value), the cell reads
``2 (2,2,1)`` and the agreement line gains a stability share (cases whose
samples all agreed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from guru import config, log
from guru.evals import cases, grading, labels, runner, runs
from guru.evals.runs import CaseResult, Run
from guru.repositories import settings as routing_settings
from guru.repositories.settings import RoutingSettings

DEFAULT_OUT = cases.REPO_ROOT / 'evals' / 'runs'
DEFAULT_MODELS_DIR = cases.REPO_ROOT / 'evals' / 'models'   # tool contracts
RUBRIC_OFF = 'none'           # --rubric none: no grading, even with a default
_SMELL_LABELS = {'whole_file_after_outline': 'whole-file',
                 'repeated_calls': 'repeated', 'refused': 'refused'}


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


def _smells_text(smells: dict) -> str:
    """``whole-file 1, repeated 2`` (non-zero smells only); ``-`` for
    none."""
    parts = [f'{_SMELL_LABELS.get(k, k)} {v}' for k, v in smells.items()
             if v]
    return ', '.join(parts) or '-'


def _int_cell(value: Optional[int]) -> str:
    return '-' if value is None else str(value)


def _row(res: CaseResult, routed: bool = False) -> list:
    """One table row: case, verdict, seconds, tokens (thousands), turns,
    smells, cost, failed checks/rubric.

    A timed-out case fails every check by design; its row shows the
    observed tools, sub-agents and changed files instead of the check list.
    For a routed run the detail also lists where the sub-agent tasks went
    (``routes: Adapter|model, ...``; ``-`` when nothing was spawned); a
    sandbox case lists its gate verdicts (``gate: intended``). The metric
    cells read ``-`` for a result without metrics (an older run file).
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
            f'{res.seconds:.1f}', runs.kilo(res.tokens),
            _int_cell(res.turns), _smells_text(res.smells),
            _fmt_cost(res.cost_usd), '; '.join(parts)]


def _samples_text(sample_scores: list) -> str:
    """`` (2,2,1)`` after a median when there was more than one sample,
    else ``''``."""
    if len(sample_scores) < 2:
        return ''
    return ' (' + ','.join(str(s) for s in sample_scores) + ')'


def _rubric_detail(res: CaseResult) -> str:
    """``rubric: 2/2`` when graded (``rubric: 2/2 (2,2,1)`` with the
    samples behind a median), ``rubric: error`` when the judge failed,
    ``rubric: grade by hand`` when nothing graded it."""
    if res.rubric_score is not None:
        return (f'rubric: {res.rubric_score}/{runs.RUBRIC_MAX}'
                + _samples_text(res.rubric_samples))
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


TABLE_HEADERS = ['case', 'result', 'seconds', 'tok', 'turns', 'smells',
                 'cost', 'detail']


def _metrics_summary(run: Run) -> str:
    """`` · tok 45.2k (5.6k/case) · turns 31 · smells 2``; empty when the
    run has no metrics (an older run file)."""
    tokens, turns = run.total_metric('tokens'), run.total_metric('turns')
    if tokens is None or turns is None:
        return ''
    mean = run.mean_metric('tokens') or 0.0
    return (f' · tok {runs.kilo(tokens)}k ({runs.kilo(mean)}k/case)'
            f' · turns {turns} · smells {run.total_smells()}')


def _print_run(run: Run, out_root: Path) -> None:
    routed = bool(run.routing)
    print(_table(TABLE_HEADERS, [_row(c, routed) for c in run.cases]))
    passed = sum(1 for c in run.cases if c.passed)
    summary = (f'\npassed {passed}/{len(run.cases)}'
               f' · mean {run.mean_seconds():.1f}s'
               f'{_metrics_summary(run)}'
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
        if run.rubric_samples > 1:
            summary += f' (median of {run.rubric_samples} samples)'
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


def _pm_kilo(pair: Optional[tuple]) -> str:
    """``mean ± spread`` in thousands (one decimal); ``n/a`` for None."""
    if pair is None:
        return 'n/a'
    return f'{runs.kilo(pair[0])} ± {runs.kilo(pair[1])}'


def _print_aggregate(run_list: list, repeats: int) -> None:
    """The ``--repeat`` table: per case the pass count, cost, seconds,
    tokens (thousands) and turns as mean ± spread (sample standard
    deviation), the summed smells and the mean rubric."""
    agg = runs.aggregate(run_list)
    rows = []
    for name, v in agg.items():
        mean_rubric = v['rubric_mean']
        rows.append([name, f"{v['passes']}/{v['runs']}",
                     _pm(v['cost_usd'], '${:.3f}'),
                     _pm(v['seconds'], '{:.1f}'),
                     _pm_kilo(v['tokens']), _pm(v['turns'], '{:.1f}'),
                     v['smells'],
                     '-' if mean_rubric is None
                     else f'{mean_rubric:.1f}/{runs.RUBRIC_MAX}'])
    need = runs.required_passes(repeats)
    weak = [n for n, v in agg.items() if v['passes'] < need]
    print(f'\naggregate over {len(run_list)} run(s):')
    print(_table(['case', 'passes', 'cost', 'seconds', 'tok', 'turns',
                  'smells', 'rubric'], rows))
    print(f'gate: every case must pass at least {need}/{repeats}'
          + (f' — below: {", ".join(weak)}' if weak else ' — ok'))
    print('runs: ' + ', '.join(r.run_id for r in run_list))


def _resolve_rubric(args: argparse.Namespace,
                    routing: Optional[RoutingSettings]) -> str:
    """The rubric judge spec for the run: the flag, else — whenever the
    run may spend (``--allow-spend``) — the routing file's cheapest rung
    or, without one, the cheapest Claude tier of the first enabled remote
    adapter in ``adapters.toml``; ``''`` for no grading (``--rubric
    none``, no spend, or no remote adapter to grade with)."""
    if args.rubric is not None:
        spec = args.rubric.strip()
        return '' if spec.lower() == RUBRIC_OFF else spec
    if not args.allow_spend:
        return ''
    spec = runner.default_rubric_spec(routing) if routing is not None else ''
    return spec or routing_settings.cheapest_remote_spec()


def _progress(res: CaseResult) -> None:
    """The per-case line printed as a case finishes."""
    flags = ', timed out' if res.observed.get('timed_out') else ''
    if res.observed.get('skipped'):
        flags += f', skipped: {res.observed.get("error", "")}'
    if res.rubric_score is not None:
        flags += (f', rubric {res.rubric_score}/{runs.RUBRIC_MAX}'
                  + _samples_text(res.rubric_samples))
    print(f'[evals] {res.case}: {"PASS" if res.passed else "FAIL"}'
          f' ({res.seconds:.1f}s{flags})', flush=True)


def _load_routing(args: argparse.Namespace
                  ) -> tuple[Optional[RoutingSettings], str, Any]:
    """``(routing, routing_name, decisions)`` from ``--routing``; all
    empty without the flag. ``ValueError`` as the loaders raise."""
    if not args.routing:
        return None, '', None
    path = Path(args.routing)
    return (runner.load_routing_file(path), path.stem,
            runner.load_decisions_file(path))


def _num_ctx(args: argparse.Namespace) -> int:
    """The context pin: the flag, else ``[evals] num_ctx``."""
    return config.EVALS_NUM_CTX if args.num_ctx is None else args.num_ctx


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
    num_ctx = _num_ctx(args)
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
    if args.samples < 1:
        print('error: --samples must be at least 1', file=sys.stderr)
        return 2
    out_root = Path(args.out)
    try:
        routing, routing_name, decisions = _load_routing(args)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    rubric_spec = _resolve_rubric(args, routing)
    if args.rubric_min is not None and not rubric_spec:
        print('error: --rubric-min needs a rubric judge (--rubric, or '
              '--allow-spend with a remote adapter configured)',
              file=sys.stderr)
        return 2
    run_list: list = []
    for i in range(1, args.repeat + 1):
        note = args.note
        if args.repeat > 1:
            note = f'{note} (repeat {i}/{args.repeat})' if note \
                else f'repeat {i}/{args.repeat}'
            print(f'[evals] repeat {i}/{args.repeat}', flush=True)
        try:
            run = runner.run_suite(suite, model, out_root, note=note,
                                   on_result=_progress, num_ctx=num_ctx,
                                   routing=routing,
                                   routing_name=routing_name,
                                   allow_spend=args.allow_spend,
                                   decisions=decisions,
                                   rubric_spec=rubric_spec,
                                   rubric_min=args.rubric_min,
                                   rubric_samples=args.samples)
        except ValueError as e:
            print(f'error: {e}', file=sys.stderr)
            return 2
        _print_run(run, out_root)
        run_list.append(run)
    if args.repeat > 1:
        _print_aggregate(run_list, args.repeat)
    return 0 if runs.aggregate_ok(runs.aggregate(run_list),
                                  args.repeat) else 1


# --- matrix -----------------------------------------------------------------

MATRIX_HEADERS = ['model', 'passed', 'tok', 'tool kB', 'seconds', 'cost',
                  'contract']


def contract_cell(contract: Optional[dict]) -> str:
    """``ok/calls`` from a tool-contract record (``bench/tool_contract.py``
    writes ``{"model", "calls", "ok", "schema_errors", ...}``); ``-``
    without a record or when either key is missing or not a number."""
    if not contract:
        return '-'
    ok, calls = contract.get('ok'), contract.get('calls')
    if not isinstance(ok, (int, float)) or isinstance(ok, bool) or \
            not isinstance(calls, (int, float)) or isinstance(calls, bool):
        return '-'
    return f'{int(ok)}/{int(calls)}'


def _matrix_row(spec: str, run: Run, models_dir: Path) -> list:
    """One matrix row: the model, cases passed, tokens (thousands), tool
    bytes shown (kB), summed case seconds, cost, tool-contract score."""
    passed = sum(1 for c in run.cases if c.passed)
    seconds = sum(c.seconds for c in run.cases)
    tool_bytes = run.total_metric('tool_bytes')
    return [spec, f'{passed}/{len(run.cases)}',
            runs.kilo(run.total_metric('tokens')), runs.kilo(tool_bytes),
            f'{seconds:.1f}', _fmt_cost(run.total_cost()),
            contract_cell(runs.load_contract(models_dir, spec))]


def _cmd_matrix(args: argparse.Namespace) -> int:
    specs = _csv(args.models) or []
    bad = [s for s in specs if '|' not in s or not s.partition('|')[2]]
    if not specs or bad:
        print("error: --models takes 'Adapter|model' specs separated by "
              f"commas{': ' + ', '.join(repr(b) for b in bad) if bad else ''}",
              file=sys.stderr)
        return 2
    try:
        suite = _load_suite(args)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    if not suite:
        print('error: no cases found', file=sys.stderr)
        return 2
    num_ctx = _num_ctx(args)
    if num_ctx < 0:
        print('error: --num-ctx must be 0 (auto-fit) or positive',
              file=sys.stderr)
        return 2
    try:
        routing, routing_name, decisions = _load_routing(args)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    out_root = Path(args.out)
    rows: list = []
    run_list: list = []
    for spec in specs:
        note = f'matrix {spec}' + (f': {args.note}' if args.note else '')
        print(f'[evals] matrix: {spec}', flush=True)
        try:
            run = runner.run_suite(suite, spec, out_root, note=note,
                                   on_result=_progress, num_ctx=num_ctx,
                                   routing=routing,
                                   routing_name=routing_name,
                                   allow_spend=args.allow_spend,
                                   decisions=decisions)
        except ValueError as e:
            print(f'error: {e}', file=sys.stderr)
            return 2
        rows.append(_matrix_row(spec, run, Path(args.models_dir)))
        run_list.append(run)
    print()
    print(_table(MATRIX_HEADERS, rows))
    print(f'{len(suite)} case(s) per model; contract = ok/calls from '
          f'{args.models_dir}/<slug>.json (bench/tool_contract.py)')
    print('runs: ' + ', '.join(r.run_id for r in run_list)
          + f' (under {out_root})')
    return 0


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


def _judge_headers(specs: list) -> list:
    """Short column headers: the model part of each spec, the whole spec
    when two share a model."""
    models = [spec.partition('|')[2] or spec for spec in specs]
    return [m if models.count(m) == 1 else spec
            for m, spec in zip(models, specs)]


def _grade_cell(row: grading.GradeRow, spec: str) -> str:
    """``2`` (one sample) or ``2 (2,2,1)`` (median, then the samples);
    ``err`` when the judge failed, ``-`` when it did not grade."""
    grade = row.grades.get(spec)
    if grade is not None:
        return grade.cell()
    return 'err' if spec in row.errors else '-'


def _share(hits: int, total: int) -> str:
    """``2/3 (67%)``; no percentage when nothing was counted."""
    rate = f' ({hits / total:.0%})' if total else ''
    return f'{hits}/{total}{rate}'


def _print_regrade(result: grading.Regrade) -> None:
    headers = ['case', *_judge_headers(result.specs), 'hand']
    rows = [[r.case, *(_grade_cell(r, s) for s in result.specs),
             '-' if r.hand is None else str(r.hand.score)]
            for r in result.rows]
    print(_table(headers, rows))
    if result.samples > 1:
        print(f'cells: median of {result.samples} samples (ties to the '
              'lower value), then the samples in call order')
    for spec in result.specs:
        line = (f'agreement with hand: {spec} '
                f'{_share(*result.agreement[spec])}')
        if result.samples > 1:
            line += f' · stability {_share(*result.stability[spec])}'
        print(line)
    for r in result.rows:
        for spec, err in r.errors.items():
            print(f'error: {r.case} / {spec}: {err}')
    print(f'labels rows recorded under {result.ledger_dir}'
          f' · grading cost {_fmt_cost(result.cost_usd)}')


def _show_cases(run: Run, hand: list, cases_dir: Path) -> None:
    """``grade --show``: the judge's packet per rubric case, for a human
    grader (no model call)."""
    for i, res in enumerate(c for c in run.cases if c.rubric):
        answer, prompt = grading.answer_and_prompt(res, cases_dir)
        if i:
            print()
        print(grading.show_text(
            run, res, answer, prompt,
            hand=labels.hand_label(hand, res.case, run.run_id)))


def _cmd_grade(args: argparse.Namespace) -> int:
    specs = [s.strip() for s in args.rubric or []]
    if not specs and not args.show:
        print('error: grade needs --rubric SPEC (a judge) and/or --show '
              '(print the packet for grading by hand)', file=sys.stderr)
        return 2
    if args.samples < 1:
        print('error: --samples must be at least 1', file=sys.stderr)
        return 2
    try:
        run = runs.load(runs.find_run(Path(args.out), args.run_id))
        hand = labels.load_labels(Path(args.labels))
        judge_list = grading.resolve_judges(specs) if specs else []
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    if not any(c.rubric for c in run.cases):
        print(f'run {run.run_id}: no case carries a rubric; nothing to grade')
        return 0
    if args.show:
        _show_cases(run, hand, Path(args.cases_dir))
        if not judge_list:
            return 0
        print()
    result = grading.regrade(run, Path(args.out), judge_list, hand,
                             cases_dir=Path(args.cases_dir),
                             samples=args.samples)
    print(f'run {run.run_id} ({run.ts}, {run.model_label()})')
    _print_regrade(result)
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
                            "0-2 (default with --allow-spend: the routing "
                            "file's cheapest rung, else the cheapest Claude "
                            "tier of the first remote adapter; "
                            f"'{RUBRIC_OFF}' turns grading off)")
    run_p.add_argument('--rubric-min', type=int, default=None, metavar='N',
                       help='fail a rubric case graded below N (default: '
                            'a grade is reported only)')
    run_p.add_argument('--samples', type=int, default=1, metavar='N',
                       help='ask the rubric judge N times per case and '
                            'record the median (ties to the lower value); '
                            'default 1')
    run_p.add_argument('--repeat', type=int, default=1, metavar='N',
                       help='run the selection N times (one run file each) '
                            'and print an aggregate; exit 1 when a case '
                            'passed fewer than ceil(N/2) times (default 1)')
    run_p.add_argument('--cases-dir', default=str(cases.CASES_DIR),
                       help=argparse.SUPPRESS)
    run_p.set_defaults(func=_cmd_run)

    mx_p = sub.add_parser(
        'matrix', help='run the selection once per model and print one '
                       'row per model: passed, tokens, tool bytes, seconds, '
                       'cost, tool-contract score (compares; never gates)')
    mx_p.add_argument('--models', required=True, metavar='SPECS',
                      help="comma-separated 'Adapter|model' specs, each run "
                           'as the main model')
    mx_p.add_argument('--cases', default='',
                      help='comma-separated case names (default: all)')
    mx_p.add_argument('--tags', default='',
                      help='comma-separated tags; cases having any of them')
    mx_p.add_argument('--routing', default=None, metavar='FILE',
                      help='experiment TOML routing the sub-agents (the '
                           'model under test is the main/controller model)')
    mx_p.add_argument('--allow-spend', action='store_true',
                      help='grant the remote-spend question for every run')
    mx_p.add_argument('--num-ctx', type=int, default=None,
                      help='context window to load each model at; 0 = GPU '
                           'auto-fit (default: [evals] num_ctx, else '
                           f'{config.EVALS_NUM_CTX})')
    mx_p.add_argument('--out', default=str(DEFAULT_OUT),
                      help=f'run directory (default: {DEFAULT_OUT})')
    mx_p.add_argument('--note', default='',
                      help='free text appended to every trajectory row')
    mx_p.add_argument('--cases-dir', default=str(cases.CASES_DIR),
                      help=argparse.SUPPRESS)
    mx_p.add_argument('--models-dir', default=str(DEFAULT_MODELS_DIR),
                      help=argparse.SUPPRESS)
    mx_p.set_defaults(func=_cmd_matrix)

    cmp_p = sub.add_parser('compare', help='diff two run files')
    cmp_p.add_argument('old')
    cmp_p.add_argument('new')
    cmp_p.set_defaults(func=_cmd_compare)

    grade_p = sub.add_parser(
        'grade', help='re-grade a stored run offline with one or more '
                      'rubric judges and compare with the hand grades; '
                      '--show prints the packet for grading by hand')
    grade_p.add_argument('run_id', metavar='RUN_ID',
                         help='the 12-hex run id (or a path to a run file)')
    grade_p.add_argument('--rubric', action='append', default=None,
                         metavar='SPEC',
                         help="'Adapter|model' judge; repeat for a column "
                              'per judge (required unless --show)')
    grade_p.add_argument('--show', action='store_true',
                         help='print what the judge sees per rubric case '
                              '(prompt, rubric, evidence, answer) with a '
                              '[[label]] stub, for grading by hand; '
                              'without --rubric nothing is graded')
    grade_p.add_argument('--samples', type=int, default=1, metavar='N',
                         help='ask each judge N times per case; the cell '
                              'is the median (ties to the lower value) '
                              'then the samples, and the agreement line '
                              'gains a stability share (default 1)')
    grade_p.add_argument('--out', default=str(DEFAULT_OUT),
                         help=f'run directory (default: {DEFAULT_OUT})')
    grade_p.add_argument('--labels', default=str(labels.DEFAULT_LABELS_FILE),
                         metavar='FILE',
                         help='hand grades TOML (default: '
                              f'{labels.DEFAULT_LABELS_FILE})')
    grade_p.add_argument('--cases-dir', default=str(cases.CASES_DIR),
                         help=argparse.SUPPRESS)
    grade_p.set_defaults(func=_cmd_grade)

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
    """Entry point; returns the exit code. Logging goes to
    ``~/.guru/guru.log`` as in the TUI (an agent turn that raises is only
    logged by the orchestrator; without this it vanished)."""
    log.setup()
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':
    sys.exit(main())
