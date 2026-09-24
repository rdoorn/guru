"""The LLM reviewer for the sandbox quality gate (design plan §1
``gate`` judge, chunk S3): a :class:`LLMReviewer` answers a ``review``
question by asking a provider adapter for one JSON-only completion of the
fixed ``gate.GATE_QUESTIONS`` over the packet, parses it strictly
(``gate.parse_review``; garbage is an error, never a verdict) and returns
an ``Answer`` whose ``dist`` carries the five answers and whose ``chosen``
is the state ``gate.decide`` derives from them alone.

Which model reviews: ``[decisions.points] gate = "llm:<Adapter>|<model>"``
pins one (``reviewer_from_spec``, resolved through the adapter registry
the CLI installs with :func:`set_registry`). Without a configured judge,
:func:`default_reviewer` picks the routing ladder's ``standard`` rung
(``routing.resolve`` for a standard review task — so local-only mode, a
secret-scan finding on the diff and a declined spend confirmation all
keep the diff off a remote model), else the session's own adapter/model.
"""
from __future__ import annotations

import time
from typing import Optional

from guru import log
from guru.domain import decisions, gate, policy, routing, spend
from guru.repositories import settings as routing_settings

REVIEW_MAX_TOKENS = 400
_registry: object = None
_routing: Optional[routing_settings.RoutingSettings] = None


def set_registry(registry: object,
                 routing_cfg: Optional[routing_settings.RoutingSettings]
                 = None) -> None:
    """Install the ``AdapterRegistry`` (and optionally the
    ``RoutingSettings``) the reviewers resolve adapter names and ladders
    against; None clears them."""
    global _registry, _routing
    _registry = registry
    _routing = routing_cfg


def registry() -> object:
    """The installed adapter registry, or None."""
    return _registry


def _routing_settings() -> routing_settings.RoutingSettings:
    global _routing
    if _routing is None:
        try:
            _routing = routing_settings.load_routing()
        except ValueError as e:
            log.warning('routing: %s; gate reviewer uses defaults', e)
            _routing = routing_settings.RoutingSettings()
    return _routing


class LLMReviewer:
    """Judge for ``review`` questions backed by ``adapter.complete`` on
    ``model``. ``name`` is ``llm:<adapter>|<model>``."""

    def __init__(self, adapter: object, model: str,
                 max_tokens: int = REVIEW_MAX_TOKENS) -> None:
        self.adapter = adapter
        self.model = str(model)
        self.max_tokens = int(max_tokens)
        self.name = f'llm:{getattr(adapter, "name", "?")}|{self.model}'

    def ask(self, questions: list) -> list:
        """One completion per question; a non-review question or an
        unparsable answer raises (the seam records an ``error`` row)."""
        return [self._ask_one(q) for q in questions]

    def _ask_one(self, q: decisions.Question) -> decisions.Answer:
        if q.kind != decisions.REVIEW:
            raise ValueError(f'{self.name} answers review questions only, '
                             f'not {q.kind!r} ({q.id})')
        prompt = f'{q.instructions}\n\n{q.state}'
        t0 = time.perf_counter()
        text = self.adapter.complete(       # type: ignore[attr-defined]
            prompt, max_tokens=self.max_tokens, model=self.model)
        ms = round((time.perf_counter() - t0) * 1000)
        review = gate.parse_review(str(text))
        verdict = gate.decide([], review)
        return decisions.Answer(chosen=verdict.state, dist=review,
                                confidence=float(review['confidence']),
                                judge=self.name, ms=ms)


def reviewer_from_spec(arg: str) -> Optional[LLMReviewer]:
    """The reviewer for a ``llm:<Adapter>|<model>`` spec's argument, or
    None (logged) without a registry, an unknown adapter or no model."""
    adapter_name, sep, model = arg.partition('|')
    if not sep or not adapter_name.strip() or not model.strip():
        log.info('judge spec llm:%r needs <Adapter>|<model>', arg)
        return None
    if _registry is None:
        log.info('judge spec llm:%r needs the adapter registry '
                 '(judges.set_registry)', arg)
        return None
    adapter = _registry.get(              # type: ignore[attr-defined]
        adapter_name.strip())
    if adapter is None:
        log.info('judge spec llm:%r: adapter %r unknown to the registry',
                 arg, adapter_name)
        return None
    return LLMReviewer(adapter, model.strip())


def default_reviewer(diff_text: str, adapter: object, model: str
                     ) -> Optional[LLMReviewer]:
    """The reviewer when no ``gate`` judge is configured.

    With a registry and ladders: the rung ``routing.resolve`` picks for a
    ``standard`` ``review`` task under the routing mode, with the secret
    scan of ``diff_text`` (a finding strips remote rungs), the current
    spend confirmation (a pick that would still need the question falls
    through) and the session's ``adapter``/``model`` as the pre-approved
    fallback. Without a registry, ladders or a usable pick: the session's
    adapter/model; None when the session has no adapter or model.
    """
    fallback = (LLMReviewer(adapter, model)
                if adapter is not None and model else None)
    if _registry is None:
        return fallback
    cfg = _routing_settings()
    try:
        ladders = routing_settings.ladders_from_settings(
            cfg, _registry)    # type: ignore[arg-type]
    except Exception:                                # noqa: BLE001
        log.exc('gate reviewer: ladders unavailable')
        return fallback
    if not ladders:
        return fallback
    name = getattr(adapter, 'name', '')
    local_main = None
    if name and model:
        try:
            remote = _registry.is_remote(name)  # type: ignore[attr-defined]
            local_main = routing.Rung(name, model, 'hard', remote=remote,
                                      default=True)
        except KeyError:
            local_main = None
    findings = len(policy.scan(diff_text)) if cfg.secret_scan else 0
    route = routing.resolve(
        'review', 'standard', ladders, mode=cfg.mode,
        scan_findings=findings,
        confirmation=spend.status(cfg.spend_confirm),
        complexity_router=True, type_router=cfg.type_router,
        local_main=local_main)
    if route.refused or route.needs_confirmation:
        log.info('gate reviewer: route %s; using the session model',
                 '; '.join(route.reason))
        return fallback
    picked = _registry.get(route.adapter)   # type: ignore[attr-defined]
    if picked is None:
        return fallback
    return LLMReviewer(picked, route.model)
