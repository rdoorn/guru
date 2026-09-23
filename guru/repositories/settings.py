"""Typed loader for the ``[routing]`` table of ``~/.guru/settings.toml``.

TOML shape (the design sketch wrote per-kind ladders as
``[[routing.ladder.<kind>]]``; that cannot work because ``[[routing.ladder]]``
makes ``ladder`` an array of tables, so ``[[routing.ladder.review]]`` nests
inside the *last default rung*. Per-kind ladders therefore live under a
sibling table)::

    [routing]
    mode = "local-and-remote"   # local-only | local-and-remote | remote-only
    controller = false
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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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
    per-kind names to their rung specs, lowest rung first."""
    mode: str = 'local-and-remote'
    controller: bool = False
    complexity_router: bool = True
    type_router: bool = False
    spend_confirm: str = 'ask'
    secret_scan: bool = True
    ladders: dict[str, list[RungSpec]] = field(default_factory=dict)
    # True when a (non-empty) [routing] table was configured. Without one
    # guru behaves exactly as before: the CLI binds no secret scanner and
    # leaves ``config.SECRET_SCAN`` off, and routing is inert.
    present: bool = False


def _enum(section: dict, key: str, allowed: tuple, default: str) -> str:
    value = section.get(key, default)
    if value not in allowed:
        raise ValueError(
            f'[routing] {key} = {value!r}; expected one of '
            + ', '.join(allowed))
    return str(value)


def _flag(section: dict, key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
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
        controller=_flag(section, 'controller', False),
        complexity_router=_flag(section, 'complexity_router', True),
        type_router=_flag(section, 'type_router', False),
        spend_confirm=_enum(section, 'spend_confirm', SPEND_CONFIRM, 'ask'),
        secret_scan=_flag(section, 'secret_scan', True),
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
