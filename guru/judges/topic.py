"""The topic labeler endpoint (``guru.domain.usage.TopicLabeler``): one
JSON-only completion on the cheapest routed rung (``trivial`` ``explain``
through ``llm.routed_reviewer``, so local-only mode, a secret-scan finding
and a pending spend confirmation keep the request off a remote model).

Only a remote rung labels: never the session's own model (not even as
routing's main-model last resort) and never a local rung, so a label never
queues behind the turn on the local model or loads a second one; without
a usable remote rung (local-only mode, a secret-scan finding, a pending
spend confirmation, no ladder) the topic keeps its request text.
"""
from __future__ import annotations

from typing import Optional

from guru import log, session
from guru.domain import usage
from guru.judges import llm

MAX_TOKENS = 40


class RoutedTopicLabeler:
    """Labels a request with the cheapest routed model."""

    def label(self, request: str) -> Optional[str]:
        picked = llm.routed_reviewer(request, session.adapter, session.model,
                                     'explain', 'trivial', fallback=False)
        if picked is None or not getattr(picked.adapter, 'remote', False):
            return None
        text = picked.adapter.complete(           # type: ignore[attr-defined]
            usage.label_prompt(request), max_tokens=MAX_TOKENS,
            model=picked.model)
        label = usage.parse_label(str(text))
        if label is None:
            log.info('topic label: unparsable reply %r', str(text)[:120])
        return label
