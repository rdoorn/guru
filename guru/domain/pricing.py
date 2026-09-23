"""Price table and per-call cost. USD per million tokens.

Bundled defaults were verified against the live Anthropic pricing page on
2026-09-23; ``[pricing."<model>"]`` in settings.toml overrides any field.
Lookup is exact model ID first, then the longest table key contained in the
ID (LiteLLM route names such as ``anthropic/claude-sonnet-5``).

Bedrock/LiteLLM-style ids are normalised first (:func:`normalise_model_id`):
provider prefixes (``aws/``, ``anthropic/``, ``anthropic.``, ``bedrock/``)
and a trailing ``-YYYYMMDD`` date are stripped, and a version-first name
such as ``claude-5-sonnet`` or ``claude-4-5-haiku`` is reordered to the
first-party form (``claude-sonnet-5``, ``claude-haiku-4-5``). Prices for
those aliases are the first-party table; Bedrock partner pricing may
differ, so a cost computed through an alias is an approximation. Override
per model with ``[pricing."<model>"]`` when the exact figure matters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from guru import config, log

FIELDS = ('input_per_m', 'cache_write_5m_per_m', 'cache_write_1h_per_m',
          'cache_read_per_m', 'output_per_m')

DEFAULT_PRICES: dict = {
    'claude-fable-5-1': (10.0, 12.5, 20.0, 0.25, 50.0),
    'claude-fable-5': (10.0, 12.5, 20.0, 1.0, 50.0),
    'claude-opus-5-5': (4.0, 5.0, 8.0, 0.20, 20.0),
    'claude-opus-5': (5.0, 6.25, 10.0, 0.50, 25.0),
    'claude-opus-4-8': (5.0, 6.25, 10.0, 0.50, 25.0),
    'claude-opus-4-7': (5.0, 6.25, 10.0, 0.50, 25.0),
    'claude-opus-4-6': (5.0, 6.25, 10.0, 0.50, 25.0),
    'claude-sonnet-5': (2.0, 2.5, 4.0, 0.20, 10.0),
    'claude-sonnet-4-6': (3.0, 3.75, 6.0, 0.30, 15.0),
    'claude-haiku-4-5': (1.0, 1.25, 2.0, 0.10, 5.0),
}


@dataclass
class Usage:
    """Token counts for one provider call.

    ``cache_write_tokens`` are priced at the 5-minute rate; the SDK does not
    split 1-hour writes out yet, so those are under-priced until phase 2
    reads the per-TTL breakdown.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0       # 5-minute writes (the SDK default)


REQUIRED_FOR_NEW = ('input_per_m', 'output_per_m')
_warned: set = set()

# Provider prefixes stripped (repeatedly) from Bedrock/LiteLLM route names.
PROVIDER_PREFIXES = ('aws/', 'anthropic/', 'anthropic.', 'bedrock/')
_DATE_SUFFIX = re.compile(r'-\d{8}$')
# ``claude-5-sonnet`` / ``claude-4-5-haiku``: version before the family.
_VERSION_FIRST = re.compile(r'claude-(\d+(?:-\d+)?)-(opus|sonnet|haiku)\b')


def _table() -> dict:
    """Bundled table with settings.toml overrides merged per field.

    An override for a model already in the table may be partial; one that
    introduces a new model must carry at least ``input_per_m`` and
    ``output_per_m`` or it is ignored (warned once per model).
    """
    table = {k: dict(zip(FIELDS, v)) for k, v in DEFAULT_PRICES.items()}
    for model, fields in config.PRICING_OVERRIDES.items():
        known = {k: v for k, v in fields.items() if k in FIELDS}
        complete = all(k in known for k in REQUIRED_FOR_NEW)
        if model not in table and not complete:
            if model not in _warned:
                _warned.add(model)
                log.warning(
                    'ignoring [pricing."%s"]: a new model needs %s',
                    model, ' and '.join(REQUIRED_FOR_NEW))
            continue
        table.setdefault(model, {}).update(known)
    return table


def _strip_provider(model: str) -> str:
    stripped = True
    while stripped:
        stripped = False
        for prefix in PROVIDER_PREFIXES:
            if model.startswith(prefix):
                model = model[len(prefix):]
                stripped = True
    return model


def _lookup(table: dict, model: str) -> Optional[str]:
    """The table key for ``model``: exact, else the longest key contained."""
    if model in table:
        return model
    hits = [k for k in table if k in model]
    return max(hits, key=len) if hits else None


def normalise_model_id(model: str) -> str:
    """Map a provider-flavoured id onto the table key it stands for.

    Strips provider prefixes and a trailing date, tries an exact/contained
    match, then reorders a version-first name (``claude-5-sonnet`` ->
    ``claude-sonnet-5``) and tries again. An id that matches nothing is
    returned stripped (prefixes/date) but otherwise unchanged.
    """
    table = _table()
    bare = _DATE_SUFFIX.sub('', _strip_provider(model))
    key = _lookup(table, bare)
    if key is not None:
        return key
    reordered = _VERSION_FIRST.sub(r'claude-\2-\1', bare)
    key = _lookup(table, reordered)
    return key if key is not None else bare


def prices_for(model: str) -> Optional[dict]:
    """Return the price dict for ``model`` or None if unknown.

    ``model`` may be a first-party id, a LiteLLM route name or a Bedrock
    alias (see :func:`normalise_model_id`).
    """
    table = _table()
    key = _lookup(table, model)
    if key is None:
        key = _lookup(table, normalise_model_id(model))
    return table[key] if key is not None else None


def cost_usd(model: str, usage: Usage, local: bool = False) -> Optional[float]:
    """Cost of one call in USD; 0.0 for local models; None when unknown."""
    if local:
        return 0.0
    p = prices_for(model)
    if p is None:
        return None
    total = (usage.input_tokens * p.get('input_per_m', 0.0)
             + usage.output_tokens * p.get('output_per_m', 0.0)
             + usage.cache_read_tokens * p.get('cache_read_per_m', 0.0)
             + usage.cache_write_tokens * p.get('cache_write_5m_per_m', 0.0))
    return total / 1_000_000
