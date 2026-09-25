"""Judge implementations for the decision seam, and the settings -> judge
factory. Specs (``[decisions.points]`` values): ``ollama`` /
``ollama:<model>``, ``encoder`` / ``encoder:<hf-model>``, ``injection`` /
``injection:<hf-model>``, ``llm:<Adapter>|<model>`` (the sandbox gate's
reviewer on a provider adapter; needs the registry from
:func:`set_registry`).

Judges load their model lazily, which takes longer than the active-decision
timeout, so the first active answer would always fall back to the
heuristic. ``install(warm=True)`` (the default; the TUI path) warms every
judge exposing ``warm_up()`` on a background thread; ``warm_up_all()`` does
the same synchronously with a deadline (the eval runner, before its first
case).
"""
from __future__ import annotations

import threading
from typing import Optional

from guru import config, log
from guru.domain import decisions
from guru.judges import encoder, llm, ollama_json


def set_registry(registry: object, routing_cfg: object = None) -> None:
    """Install the adapter registry (and optionally the routing settings)
    the ``llm:`` judges resolve against (the CLI and the eval runner)."""
    llm.set_registry(registry, routing_cfg)   # type: ignore[arg-type]


def build(spec: str) -> Optional[decisions.Judge]:
    """Instantiate the judge for ``spec``, or None if unknown/unavailable."""
    kind, _, arg = spec.partition(':')
    if kind == 'llm':
        return llm.reviewer_from_spec(arg)
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


def install(warm: bool = True) -> dict:
    """Register a judge per configured point (shadow and active modes).

    With ``warm`` every installed judge exposing ``warm_up()`` is warmed on
    a daemon thread (startup is not blocked; ``wait_warm_up`` joins it).
    Returns ``{point: judge name}``.
    """
    global _warm_thread
    _warm_thread = None
    decisions.clear_judges()
    if config.DECISIONS_MODE not in config.JUDGING_MODES:
        return {}
    installed: dict = {}
    for point, spec in config.DECISIONS_POINTS.items():
        judge = build(spec)
        if judge is not None:
            decisions.set_judge(point, judge)
            installed[point] = getattr(judge, 'name', spec)
    if warm and _warmable():
        _warm_thread = _start_warm_up({})
    return installed


_warm_thread: Optional[threading.Thread] = None


def _warmable() -> list:
    """Installed judges with a ``warm_up`` method, each once."""
    out: list = []
    for judge in decisions.installed_judges():
        if (callable(getattr(judge, 'warm_up', None))
                and all(j is not judge for j in out)):
            out.append(judge)
    return out


def _warm_all(into: dict) -> None:
    """Warm each warmable judge in turn, recording ``{name: seconds}`` in
    ``into`` as each finishes; one info line at the end. Never raises."""
    for judge in _warmable():
        name = getattr(judge, 'name', type(judge).__name__)
        try:
            into[name] = float(judge.warm_up())
        except Exception:
            log.exc(f'warm-up of {name} failed')
    log.info('judge warm-up: %s', ', '.join(
        f'{name}={secs:.2f}s' for name, secs in into.items()) or 'nothing')


def _start_warm_up(into: dict) -> threading.Thread:
    thread = threading.Thread(target=_warm_all, args=(into,),
                              name='guru-judge-warm-up', daemon=True)
    thread.start()
    return thread


def wait_warm_up(timeout_s: Optional[float] = None
                 ) -> Optional[threading.Thread]:
    """Join the background warm-up started by ``install`` (if any) and
    return its thread, or None when none was started."""
    thread = _warm_thread
    if thread is not None:
        thread.join(timeout_s)
    return thread


def warm_up_all(timeout_s: float = 30) -> dict:
    """Warm every installed judge now and wait up to ``timeout_s``.

    Returns ``{judge name: seconds}`` for the judges that finished; a
    warm-up still running at the deadline keeps going on its daemon thread
    (the model becomes resident later) and is reported with a warning.
    """
    if not _warmable():
        return {}
    done: dict = {}
    thread = _start_warm_up(done)
    thread.join(timeout_s)
    if thread.is_alive():
        log.warning('judge warm-up still running after %ss; finished: %s',
                    timeout_s, done or 'nothing')
    return dict(done)
