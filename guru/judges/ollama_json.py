"""Sidecar judge: a small Ollama model forced to answer with one option
letter via a JSON-schema enum; the option distribution is read from the
token log-probabilities at the letter position (no free text generated).

Measured in bench/primitives: qwen3:4b scores 0.92 on stall detection and
0.83 on tool choice this way at ~250 ms. Probabilities are NOT calibrated;
treat them as a ranking.
"""
from __future__ import annotations

import json
import math
import string
import time
from typing import Optional, Union

import ollama

from guru.domain.decisions import CHOICE, NOUL, SCORE, Answer, Question

LETTERS = string.ascii_uppercase
KEEP_ALIVE = '10m'
SIDECAR_TIMEOUT_S = 30      # a hung sidecar must not pin the judge thread


def build_prompt(q: Question) -> tuple:
    """Render ``q`` as lettered options; return ``(prompt, letter -> key)``."""
    mapping = {LETTERS[i]: key for i, key in enumerate(q.options)}
    lines = [q.instructions, '', q.state, '', 'Options:']
    lines += [f'{letter}) {q.options[key]}' for letter, key in mapping.items()]
    lines += ['', 'Answer with the letter only.']
    return '\n'.join(lines), mapping


def letter_mass(top: list, mapping: dict) -> dict:
    """Sum the probability of top tokens that read as an option letter."""
    mass = {k: 0.0 for k in mapping}
    for tl in top or []:
        tok = (tl.token or '').strip()
        if not tok:
            continue
        head, tail = tok[0].upper(), tok[1:]
        if head in mapping and tail in ('', ')', '.', ':'):
            mass[head] += math.exp(tl.logprob)
    return mass


def to_answer(q: Question, mass: dict, mapping: dict, judge: str,
              ms: int) -> Answer:
    """Normalise letter mass into an Answer for ``q``'s kind.

    Zero total mass (no letter seen, unparseable reply) yields
    ``chosen=None`` for every kind, an all-zero ``dist`` and confidence 0.
    """
    if q.kind not in (NOUL, SCORE, CHOICE):
        raise ValueError(f'unknown kind {q.kind!r}')
    total = sum(mass.values())
    dist = {mapping[k]: (round(v / total, 4) if total else 0.0)
            for k, v in mass.items()}
    if not total:
        return Answer(chosen=None, dist=dist, confidence=0.0, judge=judge,
                      ms=ms)
    top = max(dist, key=dist.__getitem__)
    n = len(dist)
    # A single option is certain by construction (matches the encoder judge).
    confidence = ((n * dist[top] - 1) / (n - 1)) if n > 1 else 1.0
    chosen: Union[str, bool, int, None]
    if q.kind == NOUL:
        chosen = dist.get('yes', 0.0) >= 0.5
    elif q.kind == SCORE:
        chosen = list(q.options).index(top)
    else:
        chosen = top
    return Answer(chosen=chosen, dist=dist, confidence=round(confidence, 4),
                  judge=judge, ms=ms)


class OllamaJsonJudge:
    """Judge backed by an Ollama model with a JSON-schema enum answer."""

    def __init__(self, model: str, url: Optional[str] = None,
                 client=None) -> None:
        self.model = model
        self.client = client or ollama.Client(host=url,
                                              timeout=SIDECAR_TIMEOUT_S)
        self.name = f'ollama-json:{model}'

    def ask(self, questions: list) -> list:
        """Answer each question with one constrained sidecar call."""
        return [self._ask_one(q) for q in questions]

    def _ask_one(self, q: Question) -> Answer:
        prompt, mapping = build_prompt(q)
        schema = {'type': 'object',
                  'properties': {'answer': {'type': 'string',
                                            'enum': list(mapping)}},
                  'required': ['answer']}
        t0 = time.perf_counter()
        r = self.client.generate(
            model=self.model, prompt=prompt, think=False, format=schema,
            logprobs=True, top_logprobs=20,
            options={'num_predict': 12, 'temperature': 0},
            keep_alive=KEEP_ALIVE)
        ms = round((time.perf_counter() - t0) * 1000)
        mass = {k: 0.0 for k in mapping}
        for lp in (getattr(r, 'logprobs', None) or []):
            if (lp.token or '').strip().strip('"') in mapping:
                mass = letter_mass(lp.top_logprobs, mapping)
                break
        if sum(mass.values()) == 0:
            try:
                ans = json.loads(r.response or '{}').get('answer')
            except ValueError:
                ans = None
            if ans in mapping:
                mass[ans] = 1.0
        return to_answer(q, mass, mapping, self.name, ms)
