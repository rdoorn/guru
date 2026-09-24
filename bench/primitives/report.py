"""Render probe results as Markdown tables (accuracy per model x variant, and
latency per model x task)."""
import json
import sys
from collections import defaultdict


def main(path: str) -> None:
    d = json.load(open(path))
    rows = d['summary']
    tasks = sorted({r['task'] for r in rows})
    models = list(dict.fromkeys(r['model'] for r in rows))
    variants = list(dict.fromkeys(r['variant'] for r in rows))
    acc: dict = defaultdict(dict)
    lat: dict = defaultdict(dict)
    mass: dict = defaultdict(dict)
    for r in rows:
        acc[(r['model'], r['variant'])][r['task']] = r['accuracy']
        lat[(r['model'], r['variant'])][r['task']] = r['mean_ms']
        mass[(r['model'], r['variant'])][r['task']] = r['mean_in_option_mass']
    print('## Accuracy (fraction correct)\n')
    print('| model | variant | ' + ' | '.join(tasks) + ' | mean |')
    print('|' + '---|' * (len(tasks) + 3))
    for m in models:
        for v in variants:
            a = acc.get((m, v))
            if not a:
                continue
            vals = [a.get(t) for t in tasks]
            mean = sum(x for x in vals if x is not None) / len(vals)
            print(f"| {m} | {v} | "
                  + ' | '.join(f"{x:.2f}" for x in vals)
                  + f" | **{mean:.2f}** |")
    print('\n## Mean latency (ms) and in-option mass, plain variant\n')
    print('| model | ' + ' | '.join(tasks) + ' |')
    print('|' + '---|' * (len(tasks) + 1))
    for m in models:
        lt = lat.get((m, 'plain'), {})
        print(f"| {m} | " + ' | '.join(f"{lt.get(t, 0)}" for t in tasks)
              + ' |')
    print('\n## In-option mass (how often the model answered with a letter)\n')
    print('| model | variant | ' + ' | '.join(tasks) + ' |')
    print('|' + '---|' * (len(tasks) + 2))
    for m in models:
        for v in variants:
            ms = mass.get((m, v))
            if ms:
                print(f"| {m} | {v} | "
                      + ' | '.join(f"{ms.get(t, 0):.2f}" for t in tasks)
                      + ' |')
    print('\nloads:', json.dumps(d['loads']))
    print('baselines:', d['baselines'])


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'bench/primitives/results.json')
