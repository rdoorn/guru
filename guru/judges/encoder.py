"""Encoder judges: BERT-family classifiers that return class probabilities
without generating text.

* ``EncoderJudge`` — zero-shot NLI (DeBERTa-v3). Noul = entailment
  probability of ``Question.hypothesis``; choice/score = softmax over option
  descriptions. Measured 0.86–1.00 on review-panel selection at 13–100 ms;
  weak on judgments about a reply's function and on code as input.
* ``InjectionJudge`` — prompt-injection classifier over fetched text
  (0.89; one false positive on raw code).

Both need the optional ``judge`` extra (torch + transformers) and import it
lazily; ``available()`` tells. The pipeline is built on first use (1-2 s),
which is longer than the active-decision timeout, so callers that care
about the first answer call ``warm_up()`` ahead of time.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional, Union

from guru import log
from guru.domain.decisions import CHOICE, NOUL, Answer, Question

NLI_MODEL = 'MoritzLaurer/deberta-v3-base-zeroshot-v2.0'
INJECTION_MODEL = 'protectai/deberta-v3-base-prompt-injection-v2'
MAX_CHARS = 2000        # ~500 tokens: the encoders' hard limit


def _import_pipeline() -> Optional[Callable]:
    try:
        from transformers import pipeline
    except ImportError:
        return None
    return pipeline


def available() -> bool:
    """True when torch + transformers are installed."""
    return _import_pipeline() is not None


def _device() -> str:
    # Apple Silicon (MPS) or CPU only: guru runs on a Mac; CUDA is not
    # probed. Extend here if a CUDA host ever runs the encoder judges.
    try:
        import torch
    except ImportError:
        return 'cpu'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def _factory(task: str, model: str, **kw: Any) -> Callable:
    def make() -> Any:
        pipeline = _import_pipeline()
        if pipeline is None:
            raise RuntimeError('encoder judge needs: uv sync --extra judge')
        return pipeline(task, model=model, device=_device(), **kw)
    return make


class _LazyPipeline:
    """Build the transformers pipeline on first use and keep it.

    The lock keeps a warm-up thread and the judge worker from both
    building (and holding) a copy.
    """

    def __init__(self, factory: Callable) -> None:
        self._factory = factory
        self._pipe: Any = None
        self._lock = threading.Lock()

    def __call__(self) -> Any:
        with self._lock:
            if self._pipe is None:
                self._pipe = self._factory()
            return self._pipe


def _warm(name: str, run: Callable[[], object]) -> float:
    """Run ``run`` (load + one tiny classification) and return the seconds
    it took; failures are logged and swallowed."""
    t0 = time.perf_counter()
    try:
        run()
    except Exception:
        log.exc(f'warm-up of {name} failed')
    return round(time.perf_counter() - t0, 3)


def _noul_answer(p: float, judge: str, ms: int) -> Answer:
    p = round(p, 4)
    return Answer(chosen=p >= 0.5, dist={'yes': p, 'no': round(1 - p, 4)},
                  confidence=round(abs(p - 0.5) * 2, 4), judge=judge, ms=ms)


class EncoderJudge:
    """Zero-shot NLI judge over short natural-language descriptions."""

    def __init__(self, model: str = NLI_MODEL,
                 pipeline_factory: Optional[Callable] = None) -> None:
        self.name = f'encoder:{model.rsplit("/", 1)[-1]}'
        self._pipeline = _LazyPipeline(pipeline_factory or _factory(
            'zero-shot-classification', model))

    def warm_up(self) -> float:
        """Load the pipeline and classify one token so the weights are
        resident; never raises. Returns the seconds spent."""
        return _warm(self.name, lambda: self._pipeline()(
            'ok', candidate_labels=['ok'], hypothesis_template='{}',
            multi_label=True))

    def ask(self, questions: list) -> list:
        """One classifier call per question."""
        return [self._ask_one(q) for q in questions]

    def _ask_one(self, q: Question) -> Answer:
        pipe = self._pipeline()
        state = q.state[:MAX_CHARS]
        t0 = time.perf_counter()
        if q.kind == NOUL:
            res = pipe(state,
                       candidate_labels=[q.hypothesis or q.instructions],
                       hypothesis_template='{}', multi_label=True)
            ms = round((time.perf_counter() - t0) * 1000)
            return _noul_answer(float(res['scores'][0]), self.name, ms)
        labels = {desc: key for key, desc in q.options.items()}
        res = pipe(state, candidate_labels=list(labels),
                   hypothesis_template='{}', multi_label=False)
        ms = round((time.perf_counter() - t0) * 1000)
        dist = {labels[lab]: round(float(s), 4)
                for lab, s in zip(res['labels'], res['scores'])}
        top = max(dist, key=dist.__getitem__)
        n = len(dist)
        confidence = round((n * dist[top] - 1) / (n - 1), 4) if n > 1 else 1.0
        chosen: Union[str, int] = (top if q.kind == CHOICE
                                   else list(q.options).index(top))
        return Answer(chosen=chosen, dist=dist, confidence=confidence,
                      judge=self.name, ms=ms)


class InjectionJudge:
    """Prompt-injection screen: a noul over fetched text."""

    def __init__(self, model: str = INJECTION_MODEL,
                 pipeline_factory: Optional[Callable] = None) -> None:
        self.name = f'injection:{model.rsplit("/", 1)[-1]}'
        self._pipeline = _LazyPipeline(pipeline_factory or _factory(
            'text-classification', model, truncation=True, max_length=512))

    def warm_up(self) -> float:
        """Load the pipeline and classify one token; never raises.
        Returns the seconds spent."""
        return _warm(self.name, lambda: self._pipeline()('ok'))

    def ask(self, questions: list) -> list:
        """Classify each question's state; INJECTION maps to yes."""
        pipe = self._pipeline()
        out = []
        for q in questions:
            t0 = time.perf_counter()
            r = pipe(q.state[:MAX_CHARS])[0]
            ms = round((time.perf_counter() - t0) * 1000)
            score = float(r['score'])
            injected = str(r['label']).upper().startswith('INJ')
            p = score if injected else 1 - score
            out.append(_noul_answer(p, self.name, ms))
        return out
