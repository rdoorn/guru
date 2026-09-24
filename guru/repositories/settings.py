"""Typed loaders for the ``[routing]`` table of ``~/.guru/settings.toml``
and the ``[decisions]`` table of an eval experiment file.

TOML shape (the design sketch wrote per-kind ladders as
``[[routing.ladder.<kind>]]``; that cannot work because ``[[routing.ladder]]``
makes ``ladder`` an array of tables, so ``[[routing.ladder.review]]`` nests
inside the *last default rung*. Per-kind ladders therefore live under a
sibling table)::

    [routing]
    mode = "local-and-remote"   # local-only | local-and-remote | remote-only
    controller = true           # default: on when any ladder rung is set
    complexity_router = true
    type_router = false
    spend_confirm = "ask"            # ask | auto | never
    secret_scan = true

    [[routing.ladder]]               # the default ladder, lowest rung first
    adapter = "Ollama"
    model = "qwen3:14b"
    max_complexity = "standard"      # trivial | standard | hard
    default = true                   # at most one per ladder

    [[routing.ladders.review]]       # per-kind ladder (kind from KINDS)
    adapter = "Anthropic"
    model = "claude-sonnet-5"
    max_complexity = "hard"

Validation is strict: unknown keys, unknown enum values and duplicate
defaults raise ``ValueError`` naming the offender, so a typo cannot silently
route a task to the wrong model.

``controller`` defaults to *on when any ladder rung is configured* (a
ladder without a controller is the over-reading configuration of triage
2026-09-24: the main agent inspects dozens of files itself before it
delegates); write ``controller = false`` next to a ladder to keep the main
agent hands-on. Without a ladder the default is off.
``RoutingSettings.controller`` is always the effective boolean.

``[decisions]`` (:func:`load_decisions`) carries the judge setup an eval
experiment installs for its run — ``mode`` plus the ``points``, ``active``
and ``thresholds`` sub-tables and the ``labels_margin`` tie-breaker margin
that ``config`` documents; the sidecar and timing keys stay in
``settings.toml``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from guru import config, log
from guru.domain import routing
from guru.domain.routing import Ladder, Rung
from guru.repositories.adapters import AdapterRegistry

SPEND_CONFIRM = ('ask', 'auto', 'never')

_FLAGS = ('controller', 'complexity_router', 'type_router', 'secret_scan')
_KNOWN_KEYS = frozenset(
    ('mode', 'spend_confirm', 'ladder', 'ladders') + _FLAGS)
_RUNG_KEYS = frozenset(('adapter', 'model', 'max_complexity', 'default'))


@dataclass(frozen=True)
class RungSpec:
    """One ``[[routing.ladder]]`` entry as written (no remote flag yet)."""
    adapter: str
    model: str
    max_complexity: str
    default: bool = False


@dataclass
class RoutingSettings:
    """The validated ``[routing]`` table. ``ladders`` maps ``'default'`` and
    per-kind names to their rung specs, lowest rung first.

    ``controller`` may be given as None (the key was not set): it then
    resolves to True when any ladder has a rung, else False, so after
    construction it is always the effective boolean.
    """
    mode: str = 'local-and-remote'
    controller: Optional[bool] = None
    complexity_router: bool = True
    type_router: bool = False
    spend_confirm: str = 'ask'
    secret_scan: bool = True
    ladders: dict[str, list[RungSpec]] = field(default_factory=dict)
    # True when a (non-empty) [routing] table was configured. Without one
    # guru behaves exactly as before: the CLI binds no secret scanner and
    # leaves ``config.SECRET_SCAN`` off, and routing is inert.
    present: bool = False

    def __post_init__(self) -> None:
        if self.controller is None:
            self.controller = any(
                bool(specs) for specs in self.ladders.values())


_DECISION_KEYS = frozenset(
    ('mode', 'points', 'active', 'thresholds', 'labels_margin'))
DEFAULT_LABELS_MARGIN = 0.15


@dataclass
class DecisionsSettings:
    """The validated ``[decisions]`` table of an experiment file: the
    values ``config.DECISIONS_MODE/POINTS/ACTIVE/THRESHOLDS`` and
    ``config.DECISIONS_LABELS_MARGIN`` take for a run."""
    mode: str = 'off'
    points: dict[str, str] = field(default_factory=dict)
    active: dict[str, bool] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    labels_margin: float = DEFAULT_LABELS_MARGIN


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _typed_table(section: dict, key: str, accept: Callable[[object], bool],
                 what: str, convert: Callable[[Any], Any] = lambda v: v
                 ) -> dict:
    """``section[key]`` as ``{str: convert(v)}`` for values ``accept``
    passes; ``ValueError`` (naming the offender) otherwise."""
    raw = section.get(key)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f'[decisions] {key} must be a table of {what}')
    out: dict = {}
    for k, v in raw.items():
        if not accept(v):
            raise ValueError(
                f'[decisions] {key}.{k} = {v!r}; expected {what}')
        out[str(k)] = convert(v)
    return out


def load_decisions(section: dict) -> DecisionsSettings:
    """Parse and validate a ``[decisions]`` table (an experiment file's).

    An empty table yields the ``off`` defaults. Raises ``ValueError`` on an
    unknown key, an unknown ``mode`` or a mistyped sub-table value.
    """
    unknown = sorted(set(section) - _DECISION_KEYS)
    if unknown:
        raise ValueError(
            '[decisions] unknown keys: ' + ', '.join(unknown))
    mode = section.get('mode', 'off')
    if mode not in config.DECISIONS_MODES:
        raise ValueError(
            f'[decisions] mode = {mode!r}; expected one of '
            + ', '.join(config.DECISIONS_MODES))
    margin = section.get('labels_margin', DEFAULT_LABELS_MARGIN)
    if not _is_number(margin) or margin < 0:
        raise ValueError(
            f'[decisions] labels_margin = {margin!r}; expected a '
            'non-negative number')
    return DecisionsSettings(
        mode=str(mode),
        points=_typed_table(section, 'points',
                            lambda v: isinstance(v, str), 'judge specs'),
        active=_typed_table(section, 'active',
                            lambda v: isinstance(v, bool), 'booleans'),
        thresholds=_typed_table(section, 'thresholds', _is_number,
                                'numbers', float),
        labels_margin=float(margin),
    )


def _enum(section: dict, key: str, allowed: tuple, default: str) -> str:
    value = section.get(key, default)
    if value not in allowed:
        raise ValueError(
            f'[routing] {key} = {value!r}; expected one of '
            + ', '.join(allowed))
    return str(value)


def _flag(section: dict, key: str, default: Optional[bool]
          ) -> Optional[bool]:
    """``section[key]`` as a bool; ``default`` when absent (None lets the
    dataclass resolve it)."""
    value = section.get(key, default)
    if value is not None and not isinstance(value, bool):
        raise ValueError(f'[routing] {key} = {value!r}; expected a boolean')
    return value


def _rung(raw: object, where: str) -> RungSpec:
    if not isinstance(raw, dict):
        raise ValueError(f'[routing] {where}: each rung must be a table')
    unknown = sorted(set(raw) - _RUNG_KEYS)
    if unknown:
        raise ValueError(
            f'[routing] {where}: unknown rung keys ' + ', '.join(unknown))
    for key in ('adapter', 'model'):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ValueError(
                f'[routing] {where}: rung needs a non-empty {key}')
    level = raw.get('max_complexity')
    if level not in routing.COMPLEXITY:
        raise ValueError(
            f'[routing] {where}: max_complexity = {level!r}; expected one '
            'of ' + ', '.join(routing.COMPLEXITY))
    default = raw.get('default', False)
    if not isinstance(default, bool):
        raise ValueError(
            f'[routing] {where}: default = {default!r}; expected a boolean')
    return RungSpec(raw['adapter'], raw['model'], str(level), default)


def _ladder(raw: object, where: str) -> list[RungSpec]:
    if not isinstance(raw, list):
        raise ValueError(
            f'[routing] {where} must be an array of tables ([[{where}]])')
    rungs = [_rung(r, where) for r in raw]
    defaults = sum(1 for r in rungs if r.default)
    if defaults > 1:
        raise ValueError(
            f'[routing] {where}: {defaults} rungs marked default; at most '
            'one allowed')
    return rungs


def load_routing(section: Optional[dict] = None) -> RoutingSettings:
    """Parse and validate the ``[routing]`` table.

    ``section`` defaults to ``config.settings_section('routing')``; a missing
    (or empty) table yields the documented defaults with ``present`` False.
    Raises ``ValueError`` on any invalid or unknown key.
    """
    if section is None:
        section = config.settings_section('routing')
    present = bool(section)
    unknown = sorted(set(section) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(
            '[routing] unknown keys: ' + ', '.join(unknown))
    ladders: dict[str, list[RungSpec]] = {}
    if 'ladder' in section:
        ladders[routing.DEFAULT_LADDER] = _ladder(
            section['ladder'], 'routing.ladder')
    if 'ladders' in section:
        per_kind = section['ladders']
        if not isinstance(per_kind, dict):
            raise ValueError(
                '[routing] ladders must hold per-kind tables '
                '([[routing.ladders.<kind>]])')
        for kind, raw in per_kind.items():
            if kind == routing.DEFAULT_LADDER:
                raise ValueError(
                    '[routing] ladders.default: the default ladder is '
                    '[[routing.ladder]]')
            if kind not in routing.KINDS:
                raise ValueError(
                    f'[routing] ladders.{kind}: unknown kind; expected one '
                    'of ' + ', '.join(routing.KINDS))
            ladders[str(kind)] = _ladder(raw, f'routing.ladders.{kind}')
    return RoutingSettings(
        mode=_enum(section, 'mode', routing.MODES, 'local-and-remote'),
        controller=_flag(section, 'controller', None),
        complexity_router=bool(_flag(section, 'complexity_router', True)),
        type_router=bool(_flag(section, 'type_router', False)),
        spend_confirm=_enum(section, 'spend_confirm', SPEND_CONFIRM, 'ask'),
        secret_scan=bool(_flag(section, 'secret_scan', True)),
        ladders=ladders,
        present=present,
    )


def ladders_from_settings(settings: RoutingSettings,
                          registry: AdapterRegistry) -> dict[str, Ladder]:
    """Turn rung specs into domain ``Ladder``s, filling ``Rung.remote`` from
    ``registry.is_remote(adapter)``.

    A rung naming an adapter the registry does not know is dropped with a
    warning (design §5); a ladder left with no rungs is omitted.
    """
    ladders: dict[str, Ladder] = {}
    for name, specs in settings.ladders.items():
        rungs: list[Rung] = []
        for spec in specs:
            try:
                remote = registry.is_remote(spec.adapter)
            except KeyError:
                log.warning(
                    'routing: dropping rung %s|%s from ladder %r: adapter '
                    '%r is not configured', spec.adapter, spec.model, name,
                    spec.adapter)
                continue
            rungs.append(Rung(spec.adapter, spec.model, spec.max_complexity,
                              remote=remote, default=spec.default))
        if rungs:
            ladders[name] = Ladder(rungs)
    return ladders
