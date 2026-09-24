"""Probe: zero-shot NLI cross-encoders (BERT-style, no generation) on the same
labelled cases as ``probe.py``.

Runs from a separate environment with torch + transformers installed (these
are deliberately NOT guru dependencies)::

    PYTHONPATH=. /tmp/nli-venv/bin/python bench/primitives/probe_nli.py \
        [--models a,b] [--device mps|cpu] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from transformers import pipeline

from bench.primitives.cases import TASKS

MODELS = ['MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33',
          'MoritzLaurer/deberta-v3-base-zeroshot-v2.0']

# Hypothesis phrasing per task: (template, {option_key: label}).
HYPOTHESES = {
    'stall': ('This reply {}.',
              {'yes': 'only announces a future action and gives no answer'}),
    'panel_sec': ('This code-review task {}.',
                  {'yes': 'involves security: authentication, secrets,'
                          ' untrusted input, or path handling'}),
    'panel_arch': ('This code-review task {}.',
                   {'yes': 'involves system architecture, module boundaries'
                           ' or service decomposition'}),
    'panel_sre': ('This code-review task {}.',
                  {'yes': 'involves reliability: deployments, retries,'
                          ' timeouts, alerting or operations'}),
    'tier': ('This request is {}.',
             {'small': 'trivial: a greeting, a simple fact, a conversion or'
                       ' a one-line lookup',
              'medium': 'moderate: explain or write a short code snippet,'
                        ' a regex, a summary or a small fix',
              'large': 'hard: a multi-file refactor, architecture, a subtle'
                       ' bug or race, a security review'}),
    'judge': ('The answer is {}.',
              {'wrong': 'wrong or unhelpful',
               'partial': 'partially correct or incomplete',
               'good': 'correct and complete'}),
}


def _tool_hypotheses() -> tuple:
    task = next(t for t in TASKS if t['name'] == 'tool')
    labels = {k: v.split(':', 1)[1].strip().split('.')[0]
              for k, v in task['options'].items()}
    return ('The right tool for this request is one that will {}.', labels)


def run(model: str, device: str) -> list:
    clf = pipeline('zero-shot-classification', model=model, device=device)
    out = []
    hyps = dict(HYPOTHESES)
    hyps['tool'] = _tool_hypotheses()
    for task in TASKS:
        template, labels = hyps[task['name']]
        inv = {v: k for k, v in labels.items()}
        keys = list(task['options'])
        for i, case in enumerate(task['cases']):
            if case.get('expected') is None:
                continue
            t0 = time.perf_counter()
            res = clf(case['state'], candidate_labels=list(labels.values()),
                      hypothesis_template=template,
                      multi_label=(task['kind'] == 'noul'))
            ms = (time.perf_counter() - t0) * 1000
            dist = {inv[lab]: round(s, 3)
                    for lab, s in zip(res['labels'], res['scores'])}
            if task['kind'] == 'noul':
                pred = dist['yes'] >= 0.5
                correct = pred == case['expected']
            elif task['kind'] == 'score':
                arg = max(dist, key=dist.__getitem__)
                pred = {'argmax': keys.index(arg),
                        'ev': round(sum(dist[k] * j
                                        for j, k in enumerate(keys)), 2)}
                correct = pred['argmax'] == case['expected']
            else:
                pred = max(dist, key=dist.__getitem__)
                correct = pred == case['expected']
            out.append({'model': model, 'task': task['name'], 'case': i,
                        'expected': case['expected'], 'predicted': pred,
                        'correct': correct, 'dist': dist, 'ms': round(ms)})
    return out


def summarise(outcomes: list) -> list:
    groups: dict = {}
    for o in outcomes:
        groups.setdefault((o['model'], o['task']), []).append(o)
    rows = []
    for (model, task), os_ in sorted(groups.items()):
        rows.append({
            'model': model, 'task': task, 'n': len(os_),
            'accuracy': round(sum(o['correct'] for o in os_) / len(os_), 2),
            'mean_ms': round(statistics.mean(o['ms'] for o in os_)),
            'mean_conf_correct': round(statistics.mean(
                max(o['dist'].values()) for o in os_ if o['correct']), 2)
            if any(o['correct'] for o in os_) else None,
            'mean_conf_wrong': round(statistics.mean(
                max(o['dist'].values()) for o in os_ if not o['correct']), 2)
            if any(not o['correct'] for o in os_) else None})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', default='bench/primitives/results-nli.json')
    args = ap.parse_args()
    all_out: list = []
    loads = {}
    for model in args.models.split(','):
        t0 = time.perf_counter()
        res = run(model, args.device)
        loads[model] = round((time.perf_counter() - t0) * 1000)
        all_out.extend(res)
        print(f"== {model} ({args.device})", flush=True)
        for s in summarise(res):
            print(f"  {s['task']:10s} acc={s['accuracy']:.2f} "
                  f"ms={s['mean_ms']:5d} conf ok/ko="
                  f"{s['mean_conf_correct']}/{s['mean_conf_wrong']}",
                  flush=True)
    Path(args.out).write_text(json.dumps({
        'device': args.device, 'total_ms_incl_load': loads,
        'summary': summarise(all_out), 'outcomes': all_out}, indent=1))
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
