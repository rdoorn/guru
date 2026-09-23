"""Judge implementations for the decision seam, and the settings -> judge
factory. Specs (``[decisions.points]`` values): ``ollama`` /
``ollama:<model>``, ``encoder`` / ``encoder:<hf-model>``, ``injection`` /
``injection:<hf-model>``."""
from __future__ import annotations

from typing import Optional

from guru import config, log
from guru.domain import decisions
from guru.judges import encoder, ollama_json


def build(spec: str) -> Optional[decisions.Judge]:
    """Instantiate the judge for ``spec``, or None if unknown/unavailable."""
    kind, _, arg = spec.partition(':')
    if kind == 'ollama':
        return ollama_json.OllamaJsonJudge(
            arg or config.DECISIONS_SIDECAR_MODEL,
            url=config.DECISIONS_SIDECAR_URL)
    if kind in ('encoder', 'injection'):
        if not encoder.available():
            log.info('judge %r needs the extra: uv sync --extra judge', spec)
            return None
        if kind == 'encoder':
            return encoder.EncoderJudge(arg or encoder.NLI_MODEL)
        return encoder.InjectionJudge(arg or encoder.INJECTION_MODEL)
    log.info('unknown judge spec %r', spec)
    return None


def install() -> dict:
    """Register a judge per configured point (shadow and active modes).

    Returns ``{point: judge name}``.
    """
    decisions.clear_judges()
    if config.DECISIONS_MODE not in config.JUDGING_MODES:
        return {}
    installed: dict = {}
    for point, spec in config.DECISIONS_POINTS.items():
        judge = build(spec)
        if judge is not None:
            decisions.set_judge(point, judge)
            installed[point] = getattr(judge, 'name', spec)
    return installed
