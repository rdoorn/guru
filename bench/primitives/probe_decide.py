"""Probe: GLiNER2.5-Decide schema classifiers (fastino, DeBERTa-v3 encoders,
no generation) on the same labelled cases as ``probe.py`` / ``probe_nli.py``.

Three variants per model:

* ``bare``      - label names only, no prompt.
* ``prompt``    - the task instructions as the prompt, bare label names.
* ``described`` - the instructions as the prompt plus one description per
  label (the option texts; for yes/no gates the NLI hypothesis).
* ``reframed``  - the shape the model card uses: yes/no gates become named
  labels (stall: preamble vs answer) and the three panel questions become
  one multi-label head; tier / tool / judge as in ``described``.

Needs the ``judge`` extra (``uv sync --extra judge``)::

    .venv/bin/python -m bench.primitives.probe_decide \
        [--models a,b] [--variants bare,prompt,described] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from gliner2 import AutoExtractor

from bench.primitives.cases import TASKS
from bench.primitives.probe_nli import HYPOTHESES

MODELS = ['fastino/GLiNER2.5-Decide', 'fastino/GLiNER2.5-multi-Decide']
VARIANTS = ['bare', 'prompt', 'described', 'reframed']

STALL_LABELS = {
    'stalled_preamble': 'only announces or promises an action (reading,'
                        ' checking, running) and delivers no answer',
    'substantive_answer': 'delivers an actual answer, finding or result'}
PANEL_LABELS = {
    'security': 'authentication, secrets, untrusted input, path handling,'
                ' injection',
    'architecture': 'system design, module boundaries, service'
                    ' decomposition, dependency direction',
    'reliability': 'deployments, retries, timeouts, alerting, operations',
    'none': 'trivial change, no specialist needed'}
PANEL_KEY = {'panel_sec': 'security', 'panel_arch': 'architecture',
             'panel_sre': 'reliability'}


def _labels(task: dict, variant: str) -> list | dict:
    """Label spec for one task: a list of names or a name -> description
    mapping."""
    keys = list(task['options'])
    if variant != 'described':
        return keys
    if task['kind'] == 'noul':
        return {'yes': 'Yes, it ' + HYPOTHESES[task['name']][1]['yes'],
                'no': 'No'}
    return {k: task['options'][k] for k in keys}


def _spec(task: dict, variant: str) -> dict:
    spec: dict = {'labels': _labels(task, variant)}
    if variant != 'bare':
        spec['prompt'] = task['instructions']
    return {task['name']: spec}


def _score(task: dict, label: str, conf: float, expected) -> tuple:
    """Turn the top label + confidence into (predicted, correct, p_top)."""
    keys = list(task['options'])
    if task['kind'] == 'noul':
        p_yes = conf if label == 'yes' else 1.0 - conf
        pred = p_yes >= 0.5
        return {'yes': pred, 'p_yes': round(p_yes, 3)}, pred == expected
    if task['kind'] == 'score':
        idx = keys.index(label)
        return idx, idx == expected
    return label, label == expected


def _reframed(model, task: dict, text: str) -> tuple:
    """(label, conf, p_yes or None) for the ``reframed`` variant."""
    name = task['name']
    if name == 'stall':
        res = model.classify_text(text, {'reply_status': {
            'labels': STALL_LABELS,
            'prompt': 'What does this assistant reply do?'}},
            include_confidence=True)['reply_status']
        p = res['confidence']
        p_yes = p if res['label'] == 'stalled_preamble' else 1.0 - p
        return ('yes' if p_yes >= 0.5 else 'no'), p, p_yes
    if name in PANEL_KEY:
        res = model.classify_text(text, {'specialists': {
            'labels': PANEL_LABELS, 'multi_label': True,
            'cls_threshold': 0.0,
            'prompt': 'Which review specialists does this task need?'}},
            include_confidence=True)['specialists']
        scores = {r['label']: float(r['confidence']) for r in res}
        p_yes = scores.get(PANEL_KEY[name], 0.0)
        return ('yes' if p_yes >= 0.5 else 'no'), p_yes, p_yes
    res = model.classify_text(text, _spec(task, 'described'),
                              include_confidence=True)[name]
    return res['label'], float(res['confidence']), None


def run(model_name: str, variants: list) -> tuple:
    t0 = time.perf_counter()
    model = AutoExtractor.from_pretrained(model_name)
    load_ms = round((time.perf_counter() - t0) * 1000)
    # One warm-up call so the first case does not carry graph setup cost.
    model.classify_text('warm up', {'x': ['a', 'b']})
    out = []
    for variant in variants:
        for task in TASKS:
            spec = _spec(task, variant)
            for i, case in enumerate(task['cases']):
                if case.get('expected') is None:
                    continue
                t0 = time.perf_counter()
                if variant == 'reframed':
                    label, conf, p_yes = _reframed(model, task, case['state'])
                else:
                    top = model.classify_text(case['state'], spec,
                                              include_confidence=True)
                    label = top[task['name']]['label']
                    conf = float(top[task['name']]['confidence'])
                    p_yes = None
                ms = (time.perf_counter() - t0) * 1000
                if p_yes is not None:
                    conf = p_yes if label == 'yes' else 1.0 - p_yes
                pred, correct = _score(task, label, conf, case['expected'])
                out.append({'model': model_name, 'variant': variant,
                            'task': task['name'], 'case': i,
                            'expected': case['expected'], 'label': label,
                            'predicted': pred, 'correct': correct,
                            'conf': round(conf, 3), 'ms': round(ms)})
    return out, load_ms


def summarise(outcomes: list) -> list:
    groups: dict = {}
    for o in outcomes:
        groups.setdefault((o['model'], o['variant'], o['task']), []).append(o)
    rows = []
    for (model, variant, task), os_ in sorted(groups.items()):
        ok = [o['conf'] for o in os_ if o['correct']]
        ko = [o['conf'] for o in os_ if not o['correct']]
        rows.append({
            'model': model, 'variant': variant, 'task': task, 'n': len(os_),
            'accuracy': round(len(ok) / len(os_), 2),
            'mean_ms': round(statistics.mean(o['ms'] for o in os_)),
            'mean_conf_correct': round(statistics.mean(ok), 2) if ok else None,
            'mean_conf_wrong': round(statistics.mean(ko), 2) if ko else None})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--out', default='bench/primitives/results-decide.json')
    args = ap.parse_args()
    variants = args.variants.split(',')
    all_out: list = []
    loads = {}
    for name in args.models.split(','):
        res, loads[name] = run(name, variants)
        all_out.extend(res)
        print(f"== {name} (load {loads[name]} ms)", flush=True)
        for s in summarise(res):
            print(f"  {s['variant']:9s} {s['task']:10s} "
                  f"acc={s['accuracy']:.2f} ms={s['mean_ms']:4d} conf ok/ko="
                  f"{s['mean_conf_correct']}/{s['mean_conf_wrong']}",
                  flush=True)
    Path(args.out).write_text(json.dumps({
        'load_ms': loads, 'summary': summarise(all_out),
        'outcomes': all_out}, indent=1))
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
