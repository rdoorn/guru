"""Probe: the ``labels`` decision point (sub-agent task complexity) on real
controller-written tasks, NLI encoder judge vs GLiNER2.5-Decide.

Cases: ``labels_cases.json`` - unique sub-agent tasks from the eval-run
ledgers, the controller's own label(s), and two independent gold labels
(``gold_a`` / ``gold_b``); only cases where both gold labels agree are
scored.

Both judges get guru's own question (``decisions.label_questions``): the
NLI judge through ``guru.judges.encoder.EncoderJudge`` exactly as guru runs
it, Decide with the same option texts as label descriptions and the same
instructions as prompt. Besides accuracy it reports what the active
tie-breaker would do at ``labels_margin``: how often the judge overrides
the controller, and whether those overrides fix or break the label.

Needs the ``judge`` extra (``uv sync --extra judge``)::

    .venv/bin/python -m bench.primitives.probe_labels \
        [--devices cpu,mps] [--margin 0.15] [--out FILE]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path
from typing import Callable

from gliner2 import AutoExtractor
from transformers import pipeline

from guru.domain import decisions
from guru.judges.encoder import NLI_MODEL, EncoderJudge

HERE = Path(__file__).parent
DECIDE_MODEL = 'fastino/GLiNER2.5-Decide'


def load_cases() -> list:
    rows = json.loads((HERE / 'labels_cases.json').read_text())
    return [r for r in rows if r['gold_a'] == r['gold_b']]


def nli_judge(device: str) -> Callable:
    judge = EncoderJudge(pipeline_factory=lambda: pipeline(
        'zero-shot-classification', model=NLI_MODEL, device=device))
    judge.warm_up()

    def ask(q: decisions.Question) -> dict:
        return judge.ask([q])[0].dist
    return ask


def decide_judge(device: str) -> Callable:
    model = AutoExtractor.from_pretrained(DECIDE_MODEL)
    model.to(device)
    model.classify_text('warm up', {'x': ['a', 'b']})

    def ask(q: decisions.Question) -> dict:
        # multi_label + softmax + threshold 0 returns every label with its
        # softmax probability: the full distribution, same argmax as the
        # single-label call.
        res = model.classify_text(q.state, {q.id: {
            'labels': dict(q.options), 'prompt': q.instructions,
            'multi_label': True, 'class_act': 'softmax',
            'cls_threshold': 0.0}}, include_confidence=True)[q.id]
        return {r['label']: float(r['confidence']) for r in res}
    return ask


def run(name: str, ask: Callable, cases: list, margin: float) -> dict:
    rows = []
    for case in cases:
        q = decisions.label_questions(case['task'])[0]
        t0 = time.perf_counter()
        dist = ask(q)
        ms = (time.perf_counter() - t0) * 1000
        ranked = sorted(dist, key=dist.__getitem__, reverse=True)
        top, gap = ranked[0], dist[ranked[0]] - dist[ranked[1]]
        ctrl = case['controller']
        overrides = top != ctrl and gap >= margin
        rows.append({'id': case['id'], 'gold': case['gold_a'],
                     'controller': ctrl, 'top': top, 'gap': round(gap, 3),
                     'override': overrides, 'ms': round(ms, 1)})
    return {'judge': name, 'rows': rows, 'summary': summarise(rows)}


def summarise(rows: list) -> dict:
    n = len(rows)
    ms = sorted(r['ms'] for r in rows)
    over = [r for r in rows if r['override']]
    routed = [r['top'] if r['override'] else r['controller'] for r in rows]
    confusion = collections.Counter((r['gold'], r['top']) for r in rows)
    return {
        'n': n,
        'accuracy': round(sum(r['top'] == r['gold'] for r in rows) / n, 3),
        'recall': {g: round(sum(1 for r in rows if r['gold'] == g
                                and r['top'] == g)
                            / max(1, sum(r['gold'] == g for r in rows)), 2)
                   for g in ('trivial', 'standard', 'hard')},
        'predicted': dict(collections.Counter(r['top'] for r in rows)),
        'controller_accuracy': round(
            sum(r['controller'] == r['gold'] for r in rows) / n, 3),
        'routed_accuracy': round(
            sum(t == r['gold'] for t, r in zip(routed, rows)) / n, 3),
        'overrides': len(over),
        'overrides_fixed': sum(r['top'] == r['gold'] for r in over),
        'overrides_broke': sum(r['controller'] == r['gold'] for r in over),
        'ms_median': round(statistics.median(ms), 1),
        'ms_p95': round(ms[int(0.95 * (n - 1))], 1),
        'confusion': {f'{g}->{p}': c
                      for (g, p), c in sorted(confusion.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--devices', default='cpu,mps')
    ap.add_argument('--margin', type=float, default=0.15)
    ap.add_argument('--out', default=str(HERE / 'results-labels.json'))
    args = ap.parse_args()
    cases = load_cases()
    print(f'{len(cases)} gold cases', flush=True)
    results = []
    for device in args.devices.split(','):
        for name, make in (('nli', nli_judge), ('decide', decide_judge)):
            t0 = time.perf_counter()
            ask = make(device)
            load_s = round(time.perf_counter() - t0, 1)
            res = run(f'{name}@{device}', ask, cases, args.margin)
            res['load_s'] = load_s
            results.append(res)
            s = res['summary']
            print(f"== {res['judge']} load={load_s}s acc={s['accuracy']} "
                  f"recall={s['recall']} ctrl={s['controller_accuracy']} "
                  f"routed={s['routed_accuracy']} overrides={s['overrides']}"
                  f" (fixed {s['overrides_fixed']}, broke "
                  f"{s['overrides_broke']}) ms p50/p95={s['ms_median']}/"
                  f"{s['ms_p95']}", flush=True)
    Path(args.out).write_text(json.dumps(
        {'margin': args.margin, 'results': results}, indent=1))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
