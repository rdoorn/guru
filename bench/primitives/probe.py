"""Probe: can a small local model act as a "System One" judge?

Runs labelled Choice / Score / Noul cases against several Ollama models using
single-token log-probabilities (no text generation), under a few steering
variants, and reports accuracy, latency, and how much probability mass lands
on the allowed answers. Compares against guru's current heuristics where one
exists (``looks_like_preamble`` and ``_match_tools``).

Usage::

    .venv/bin/python bench/primitives/probe.py [--models a,b] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import string
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import ollama

from bench.primitives.cases import TASKS
from guru.adapters.turn import looks_like_preamble
from guru.domain.tools import _match_tools

MODELS = ['qwen3:0.6b', 'gemma3:1b', 'qwen3:1.7b',
          'huihui_ai/qwen3-abliterated:8b']
VARIANTS = ['plain', 'plain-rev', 'chat', 'json']
LETTERS = string.ascii_uppercase
KEEP_ALIVE = '10m'


@dataclass
class Outcome:
    """One (model, variant, task, case) evaluation."""
    model: str
    variant: str
    task: str
    case: int
    expected: Any
    predicted: Any
    correct: bool
    dist: dict
    in_option_mass: float
    ms: float
    prompt_tokens: int
    raw: str = ''
    error: str = ''


@dataclass
class Summary:
    model: str
    variant: str
    task: str
    n: int
    accuracy: float
    mean_ms: float
    p50_ms: float
    mean_prompt_tokens: float
    mean_in_option_mass: float
    mean_conf_correct: Optional[float]
    mean_conf_wrong: Optional[float]
    errors: int
    extra: dict = field(default_factory=dict)


def _prompt(task: dict, case: dict, reverse: bool) -> tuple:
    """Build the question text. Returns (prompt, letter->option-key)."""
    keys = list(task['options'])
    if reverse:
        keys = keys[::-1]
    lines = [task['instructions'], '', case['state'], '', 'Options:']
    mapping = {}
    for i, key in enumerate(keys):
        letter = LETTERS[i]
        mapping[letter] = key
        lines.append(f"{letter}) {task['options'][key]}")
    lines.append('')
    lines.append('Answer with the letter only.')
    return '\n'.join(lines), mapping


def _letter_mass(top: list, mapping: dict) -> dict:
    """Sum the probability of every top token that reads as an option
    letter (allowing 'A', ' A', 'A)', 'A.')."""
    mass = {k: 0.0 for k in mapping}
    for tl in top:
        tok = tl.token.strip()
        if not tok:
            continue
        head, tail = tok[0].upper(), tok[1:]
        if head in mapping and tail in ('', ')', '.', ':'):
            mass[head] += math.exp(tl.logprob)
    return mass


def _call(client: ollama.Client, model: str, variant: str, prompt: str,
          mapping: dict) -> tuple:
    """Run one question. Returns (letter_mass, raw_text, prompt_tokens)."""
    opts = {'num_predict': 1, 'temperature': 0}
    if variant in ('plain', 'plain-rev'):
        r = client.generate(model=model, prompt=prompt + '\nAnswer:',
                            think=False, logprobs=True, top_logprobs=20,
                            options=opts, keep_alive=KEEP_ALIVE)
        top = r.logprobs[0].top_logprobs if r.logprobs else []
        return _letter_mass(top, mapping), r.response, r.prompt_eval_count
    if variant == 'chat':
        r = client.chat(
            model=model, think=False, logprobs=True, top_logprobs=20,
            options=opts, keep_alive=KEEP_ALIVE,
            messages=[
                {'role': 'system', 'content':
                 'You are a strict classifier. Read the question and reply'
                 ' with exactly one option letter and nothing else.'},
                {'role': 'user', 'content': prompt}])
        top = r.logprobs[0].top_logprobs if r.logprobs else []
        return (_letter_mass(top, mapping), r.message.content,
                r.prompt_eval_count)
    if variant == 'json':
        schema = {'type': 'object',
                  'properties': {'answer': {'type': 'string',
                                            'enum': list(mapping)}},
                  'required': ['answer']}
        r = client.generate(model=model, prompt=prompt, think=False,
                            format=schema, logprobs=True, top_logprobs=20,
                            options={'num_predict': 12, 'temperature': 0},
                            keep_alive=KEEP_ALIVE)
        mass = {k: 0.0 for k in mapping}
        # Find the token position that emitted the letter; read its
        # alternatives there. Fall back to the parsed answer with mass 1.
        for lp in (r.logprobs or []):
            tok = lp.token.strip().strip('"')
            if tok in mapping:
                mass = _letter_mass(lp.top_logprobs or [], mapping)
                break
        try:
            ans = json.loads(r.response).get('answer')
        except (ValueError, AttributeError):
            ans = None
        if ans in mapping and sum(mass.values()) == 0:
            mass[ans] = 1.0
        return mass, r.response, r.prompt_eval_count
    raise ValueError(variant)


def _decide(task: dict, mass: dict, mapping: dict) -> tuple:
    """Turn letter mass into (prediction, normalised dist over option keys)."""
    total = sum(mass.values())
    dist = {mapping[k]: (v / total if total else 0.0) for k, v in mass.items()}
    if task['kind'] == 'noul':
        return (dist.get('yes', 0.0) >= 0.5, dist)
    if task['kind'] == 'score':
        keys = list(task['options'])
        expected_value = sum(dist[k] * i for i, k in enumerate(keys))
        argmax = max(dist, key=dist.__getitem__) if total else None
        return ({'argmax': keys.index(argmax) if argmax else None,
                 'ev': round(expected_value, 2)}, dist)
    return (max(dist, key=dist.__getitem__) if total else None, dist)


def _is_correct(task: dict, expected: Any, predicted: Any) -> bool:
    if task['kind'] == 'score':
        return predicted['argmax'] == expected
    return predicted == expected


def run_model(client: ollama.Client, model: str, variants: list) -> list:
    out: list = []
    for variant in variants:
        for task in TASKS:
            for i, case in enumerate(task['cases']):
                if case.get('expected') is None:
                    continue
                prompt, mapping = _prompt(task, case,
                                          reverse=(variant == 'plain-rev'))
                t0 = time.perf_counter()
                try:
                    mass, raw, ptoks = _call(client, model, variant, prompt,
                                             mapping)
                    err = ''
                except Exception as e:                   # noqa: BLE001
                    mass, raw, ptoks, err = ({k: 0.0 for k in mapping}, '',
                                             0, repr(e))
                ms = (time.perf_counter() - t0) * 1000
                predicted, dist = _decide(task, mass, mapping)
                out.append(Outcome(
                    model=model, variant=variant, task=task['name'], case=i,
                    expected=case['expected'], predicted=predicted,
                    correct=_is_correct(task, case['expected'], predicted),
                    dist={k: round(v, 3) for k, v in dist.items()},
                    in_option_mass=round(sum(mass.values()), 3), ms=round(ms),
                    prompt_tokens=ptoks or 0, raw=raw[:40], error=err))
    return out


def summarise(outcomes: list) -> list:
    groups: dict = {}
    for o in outcomes:
        groups.setdefault((o.model, o.variant, o.task), []).append(o)
    rows = []
    for (model, variant, task), os_ in sorted(groups.items()):
        conf_ok = [max(o.dist.values()) for o in os_ if o.correct and o.dist]
        conf_ko = [max(o.dist.values()) for o in os_
                   if not o.correct and o.dist]
        extra: dict = {}
        tdef = next(t for t in TASKS if t['name'] == task)
        if tdef['kind'] == 'score':
            extra['mae_ev'] = round(statistics.mean(
                abs(o.predicted['ev'] - o.expected) for o in os_), 2)
        rows.append(Summary(
            model=model, variant=variant, task=task, n=len(os_),
            accuracy=round(sum(o.correct for o in os_) / len(os_), 2),
            mean_ms=round(statistics.mean(o.ms for o in os_)),
            p50_ms=round(statistics.median(o.ms for o in os_)),
            mean_prompt_tokens=round(statistics.mean(
                o.prompt_tokens for o in os_)),
            mean_in_option_mass=round(statistics.mean(
                o.in_option_mass for o in os_), 2),
            mean_conf_correct=round(statistics.mean(conf_ok), 2)
            if conf_ok else None,
            mean_conf_wrong=round(statistics.mean(conf_ko), 2)
            if conf_ko else None,
            errors=sum(1 for o in os_ if o.error), extra=extra))
    return rows


def heuristic_baselines() -> dict:
    """Score guru's existing regex / keyword heuristics on the same cases."""
    out = {}
    stall = next(t for t in TASKS if t['name'] == 'stall')
    hits = sum(looks_like_preamble(c['state'].split('Reply:\n', 1)[-1])
               == c['expected'] for c in stall['cases'])
    out['stall/looks_like_preamble'] = round(hits / len(stall['cases']), 2)
    tool = next(t for t in TASKS if t['name'] == 'tool')
    hits = sum(_match_tools(c['state'].split('Request: ', 1)[-1])[0]
               == c['expected'] for c in tool['cases'])
    out['tool/_match_tools'] = round(hits / len(tool['cases']), 2)
    return out


def vram(client: ollama.Client, model: str) -> int:
    for m in client.ps().models:
        if m.model == model or m.name == model:
            return int(m.size_vram or 0)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--out', default='bench/primitives/results.json')
    args = ap.parse_args()
    client = ollama.Client()
    all_out: list = []
    loads: dict = {}
    for model in args.models.split(','):
        t0 = time.perf_counter()
        client.generate(model=model, prompt='hi', think=False,
                        options={'num_predict': 1}, keep_alive=KEEP_ALIVE)
        loads[model] = {'cold_load_ms': round((time.perf_counter() - t0)
                                              * 1000),
                        'vram_mb': round(vram(client, model) / 2**20)}
        print(f"== {model}  load={loads[model]}", flush=True)
        res = run_model(client, model, args.variants.split(','))
        all_out.extend(res)
        for s in summarise(res):
            print(f"  {s.variant:9s} {s.task:8s} acc={s.accuracy:.2f} "
                  f"ms={s.mean_ms:5d} mass={s.mean_in_option_mass:.2f} "
                  f"conf ok/ko={s.mean_conf_correct}/{s.mean_conf_wrong} "
                  f"{s.extra}", flush=True)
    summary = [asdict(s) for s in summarise(all_out)]
    base = heuristic_baselines()
    print('heuristic baselines:', base)
    Path(args.out).write_text(json.dumps({
        'loads': loads, 'baselines': base, 'summary': summary,
        'outcomes': [asdict(o) for o in all_out]}, indent=1))
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
