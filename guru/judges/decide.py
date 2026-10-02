"""Decide judge: a GLiNER2.5-Decide schema classifier (DeBERTa-v3-large
encoder, no generation) over the option descriptions of a choice or score
question.

Measured on 325 real controller-written sub-agent tasks for the ``labels``
point (bench/primitives/README.md, 2026-10-02): 0.83 accuracy against 0.46
for the NLI encoder judge; as the margin-gated tie-breaker it lifts routed
accuracy from 0.60 (controller alone) to 0.83. 140-200 ms per question.

Yes/no (noul) questions are declined: as a yes/no answerer the model says
"no" to nearly everything (same probe), so those points stay on the NLI
or Ollama judges.

Needs the optional ``judge`` extra (gliner2 + torch + transformers),
imported lazily; ``available()`` tells. The model is loaded on first use
(~8 s), so callers warm it up ahead of time (``judges.install``).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional, Union

from guru.domain.decisions import CHOICE, SCORE, Answer, Question
from guru.judges import encoder

DECIDE_MODEL = 'fastino/GLiNER2.5-Decide'
# gliner2 does not truncate and its time grows with the input: ~0.24 s at
# 300 characters, ~0.95 s at 2000, ~1.65 s at 3000 (warm, MPS). 2000 keeps
# a question inside the 1500 ms active timeout. The 325 measured tasks are
# all shorter (longest 1574), so accuracy on cut tasks is unmeasured.
MAX_CHARS = 2000


def _import_extractor() -> Optional[Any]:
    try:
        from gliner2 import AutoExtractor
    except ImportError:
        return None
    return AutoExtractor


def available() -> bool:
    """True when gliner2 and the encoder stack (torch + transformers) are
    installed."""
    return encoder.available() and _import_extractor() is not None


def _factory(model: str) -> Callable[[], Any]:
    def make() -> Any:
        extractor = _import_extractor()
        if extractor is None:
            raise RuntimeError('decide judge needs: uv sync --extra judge')
        loaded = extractor.from_pretrained(model)
        loaded.to(encoder._device())
        return loaded
    return make


class DecideJudge:
    """Choice / score judge over short label descriptions."""

    def __init__(self, model: str = DECIDE_MODEL,
                 extractor_factory: Optional[Callable] = None) -> None:
        self.name = f'decide:{model.rsplit("/", 1)[-1]}'
        self._extractor = encoder._LazyPipeline(
            extractor_factory or _factory(model))
        # The warm-up thread and the active and shadow workers share one
        # model: one forward pass at a time (concurrent passes on MPS are
        # a crash risk, and serialising costs nothing at ~150 ms each).
        self._infer = threading.Lock()

    def warm_up(self) -> float:
        """Load the model and classify one token so the weights are
        resident; never raises. Returns the seconds spent."""
        def run() -> None:
            extractor = self._extractor()
            with self._infer:
                extractor.classify_text('ok', {'x': ['a', 'b']})
        return encoder._warm(self.name, run)

    def ask(self, questions: list) -> list:
        """One classifier call per question."""
        return [self._ask_one(q) for q in questions]

    def _ask_one(self, q: Question) -> Answer:
        if q.kind not in (CHOICE, SCORE):
            raise ValueError(f'{self.name} answers choice/score questions,'
                             f' not {q.kind!r} ({q.id})')
        extractor = self._extractor()
        with self._infer:
            t0 = time.perf_counter()
            # multi_label + softmax + threshold 0 returns every label with
            # its softmax probability: the full distribution the margin
            # gate needs, with the same argmax as a single-label call.
            res = extractor.classify_text(q.state[:MAX_CHARS], {q.id: {
                'labels': dict(q.options), 'prompt': q.instructions,
                'multi_label': True, 'class_act': 'softmax',
                'cls_threshold': 0.0}}, include_confidence=True)[q.id]
            ms = round((time.perf_counter() - t0) * 1000)
        dist = {r['label']: round(float(r['confidence']), 4) for r in res}
        top = max(dist, key=dist.__getitem__)
        n = len(dist)
        confidence = round((n * dist[top] - 1) / (n - 1), 4) if n > 1 else 1.0
        chosen: Union[str, int] = (top if q.kind == CHOICE
                                   else list(q.options).index(top))
        return Answer(chosen=chosen, dist=dist, confidence=confidence,
                      judge=self.name, ms=ms)
