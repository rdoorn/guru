"""Measure the economics of routing a request to a smaller model.

For each model: cold load time, and wall time for a trivial and a medium
request (prompt eval, generation, tokens/s). Then check whether a small model
can stay resident next to a big one, and what a reload after eviction costs.

    .venv/bin/python bench/primitives/switch_cost.py [--models a,b,c]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import ollama

MODELS = ['qwen3:4b', 'huihui_ai/qwen3-abliterated:8b', 'qwen3:14b']
PROMPTS = {
    'trivial': ("Translate 'good morning' to Dutch. Answer with the"
                " translation only.", 32),
    'medium': ('Write a Python regex that matches ISO-8601 dates'
               ' (YYYY-MM-DD) and explain it in three sentences.', 200),
}


def stop_all(client: ollama.Client) -> None:
    for m in client.ps().models:
        subprocess.run(['ollama', 'stop', m.model], check=False,
                       capture_output=True)
    time.sleep(1)


def resident(client: ollama.Client) -> dict:
    return {m.model: round((m.size_vram or 0) / 2**30, 1)
            for m in client.ps().models}


def timed(client: ollama.Client, model: str, prompt: str, n: int) -> dict:
    t0 = time.perf_counter()
    r = client.generate(model=model, prompt=prompt, think=False,
                        options={'num_predict': n, 'temperature': 0,
                                 'num_ctx': 8192},
                        keep_alive='10m')
    wall = time.perf_counter() - t0
    ns = 1e9
    return {'wall_s': round(wall, 2),
            'load_s': round((r.load_duration or 0) / ns, 2),
            'prompt_tokens': r.prompt_eval_count,
            'prompt_s': round((r.prompt_eval_duration or 0) / ns, 2),
            'gen_tokens': r.eval_count,
            'gen_s': round((r.eval_duration or 0) / ns, 2),
            'tok_per_s': round(r.eval_count / (r.eval_duration / ns), 1)
            if r.eval_duration else None,
            'answer': (r.response or '').strip()[:60]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--out', default='bench/primitives/results-switch.json')
    args = ap.parse_args()
    client = ollama.Client()
    models = args.models.split(',')
    out: dict = {'per_model': {}, 'coexistence': {}}

    for m in models:
        stop_all(client)
        rec: dict = {}
        t0 = time.perf_counter()
        client.generate(model=m, prompt='hi', think=False,
                        options={'num_predict': 1, 'num_ctx': 8192},
                        keep_alive='10m')
        rec['cold_load_s'] = round(time.perf_counter() - t0, 2)
        rec['resident'] = resident(client)
        for name, (prompt, n) in PROMPTS.items():
            rec[name] = timed(client, m, prompt, n)
        out['per_model'][m] = rec
        print(m, json.dumps(rec), flush=True)

    # Coexistence: load the biggest, then the smallest; does the big one stay?
    # Conservative pair: the 8B, not the largest, next to the 4B.
    big, small = models[min(1, len(models) - 1)], models[0]
    stop_all(client)
    client.generate(model=big, prompt='hi', think=False,
                    options={'num_predict': 1, 'num_ctx': 8192},
                    keep_alive='10m')
    t0 = time.perf_counter()
    client.generate(model=small, prompt='hi', think=False,
                    options={'num_predict': 1, 'num_ctx': 8192},
                    keep_alive='10m')
    co = {'small_load_next_to_big_s': round(time.perf_counter() - t0, 2),
          'resident_after': resident(client)}
    # Now go back to the big one: warm (still resident) or a reload?
    t0 = time.perf_counter()
    client.generate(model=big, prompt='hi', think=False,
                    options={'num_predict': 1, 'num_ctx': 8192},
                    keep_alive='10m')
    co['big_again_s'] = round(time.perf_counter() - t0, 2)
    co['resident_final'] = resident(client)
    out['coexistence'] = co
    print('coexistence', json.dumps(co), flush=True)
    stop_all(client)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
