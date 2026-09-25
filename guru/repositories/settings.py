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
    adapter = "SBP Litellm"
    model = "aws/claude-5-sonnet"
    max_complexity = "standard"      # trivial | standard | hard
    default = true                   # at most one per ladder

    [[routing.ladders.review]]       # per-kind ladder (kind from KINDS)
    adapter = "SBP Litellm"
    model = "aws/claude-5-5-opus"
    max_complexity = "hard"

``mode = "off"`` keeps the table but disables it: :func:`load_routing`
then yields the no-table defaults (``present`` False, ``off`` True), so
guru behaves as without a ``[routing]`` table; ``load_routing(full=True)``
keeps the ladders for display (``/routing``). :func:`switch_routing`
flips that key in place and :func:`ensure_default_routing` writes the
measured default block (:func:`default_routing_toml`) into a settings
file that has no ``[routing]`` yet.

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

``.guru/tools.toml`` (:func:`load_tools_policy`) is the per-project tool
policy — the ``ToolsPolicy`` entity lives in ``guru.domain.toolpolicy``::

    [tools]
    enabled = ["read_file", "run_tests"]   # non-empty: allowlist
    disabled = ["web_search"]              # always wins
    [tools.tests]
    runner = "pytest"                      # pytest | unittest
    [tools.limits]
    timeout_s = 60                         # see config.PROC_LIMIT_KEYS

An absent file is the default policy (everything enabled); an unreadable
file, unknown keys, an unknown runner or a mistyped limit raise
``ValueError`` naming the file (the CLI then fails closed: every registry
tool disabled).

``[sandbox]`` (:func:`load_sandbox`) configures sandboxed execution. The
global table in ``settings.toml`` sets the defaults; a project's
``.guru/sandbox.toml`` carries the same table and wins key by key, and is
the *only* place ``enabled`` may be set (a sandbox is a per-project
opt-in)::

    [sandbox]
    enabled = true                       # project file only
    runtime = "docker"
    base_image = "python:3.12-slim@sha256:…"   # digest-pinned, always
    cpus = 2.0
    memory_mb = 2048
    pids = 256
    timeout_s = 600
    proxy_image = "docker.io/kalaksi/tinyproxy:latest@sha256:…"
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from guru import config, log
from guru.domain import routing
from guru.domain.routing import Ladder, Rung
from guru.domain.toolpolicy import ToolsPolicy
from guru.repositories.adapters import AdapterRegistry

__all__ = ['DecisionsSettings', 'RoutingSettings', 'RungSpec',
           'SandboxSettings', 'ToolsPolicy', 'default_routing_toml',
           'ensure_default_routing', 'ladders_from_settings',
           'load_decisions', 'load_routing', 'load_sandbox',
           'load_tools_policy', 'switch_routing']

MODE_OFF = 'off'

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
    # True when the table says ``mode = "off"`` (kept but disabled).
    off: bool = False

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


def load_routing(section: Optional[dict] = None, *,
                 full: bool = False) -> RoutingSettings:
    """Parse and validate the ``[routing]`` table.

    ``section`` defaults to ``config.settings_section('routing')``; a missing
    (or empty) table yields the documented defaults with ``present`` False.
    ``mode = "off"`` validates the whole table and then yields the same
    defaults with ``off`` True (guru runs as without a table) — unless
    ``full`` is set, which keeps the parsed ladders and flags (for display)
    with ``mode`` at its default. Raises ``ValueError`` on any invalid or
    unknown key.
    """
    if section is None:
        section = config.settings_section('routing')
    present = bool(section)
    off = section.get('mode') == MODE_OFF
    if off:
        section = {**section, 'mode': routing.MODES[1]}
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
    parsed = RoutingSettings(
        mode=_enum(section, 'mode', routing.MODES, 'local-and-remote'),
        controller=_flag(section, 'controller', None),
        complexity_router=bool(_flag(section, 'complexity_router', True)),
        type_router=bool(_flag(section, 'type_router', False)),
        spend_confirm=_enum(section, 'spend_confirm', SPEND_CONFIRM, 'ask'),
        secret_scan=bool(_flag(section, 'secret_scan', True)),
        ladders=ladders,
        present=present,
        off=off,
    )
    if off and not full:
        return RoutingSettings(off=True)
    return parsed


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


# --- the default [routing] block -------------------------------------------

# Adapter kinds (``type`` in adapters.toml) whose models run off-machine.
REMOTE_ADAPTER_KINDS = ('litellm', 'anthropic')
# The Claude tiers per adapter kind, cheapest first (Haiku 4.5, Sonnet 5,
# Opus 5.5): LiteLLM/Bedrock route names, and the first-party ids that key
# ``guru.domain.pricing.DEFAULT_PRICES``.
DEFAULT_TIER_MODELS = {
    'litellm': ('aws/claude-4-5-haiku', 'aws/claude-5-sonnet',
                'aws/claude-5-5-opus'),
    'anthropic': ('claude-haiku-4-5', 'claude-sonnet-5', 'claude-opus-5-5'),
}
_ROUTING_HEADER = """\
# Routing defaults written by guru: the configuration measured in
# evals/routing/claude-tiers-judges.toml (evals/triage/2026-09-24-*).
# Sub-agent tasks run on Claude tiers picked by the controller's
# complexity label; reviews never run on Haiku. Measured with Haiku 4.5
# as the main (controller) model: pick it in /models.
# /routing shows this; /routing off disables it (mode = "off");
# guru never rewrites an existing [routing] table.
"""
_ROUTING_TABLE = """
[routing]
mode = "local-and-remote"   # local-only | local-and-remote | remote-only | off
controller = true           # the main agent only spawns, checks and joins
complexity_router = true    # lowest rung whose max_complexity covers the task
type_router = true          # review tasks use [[routing.ladders.review]]
spend_confirm = "ask"       # ask (once per run) | auto | never
secret_scan = true          # findings force local; remote tool output redacted

"""
_DECISIONS_TABLE = """
[decisions]
mode = "active"             # judges act on the points listed under active
labels_margin = 0.15        # the labels judge's tier must beat its runner-up

[decisions.points]
labels = "encoder"          # complexity tie-breaker for the controller's label
panel = "encoder"           # needs_security: one extra security reviewer
injection = "injection"     # shadow: fetched pages checked for injection

[decisions.active]
{note}labels = {flag}
panel = {flag}
"""
_JUDGE_EXTRA_NOTE = """\
# The encoder judges need the judge extra (uv sync --extra judge);
# until it is installed labels and panel stay shadow (verdicts are
# logged, nothing changes). Set both to true afterwards.
"""


def _judge_extra_available() -> bool:
    """True when the ``judge`` extra (torch + transformers) is importable;
    a ``find_spec`` probe, so nothing heavy is loaded."""
    import importlib.util
    try:
        return all(importlib.util.find_spec(name) is not None
                   for name in ('torch', 'transformers'))
    except (ImportError, ValueError):
        return False


def _rung_toml(table: str, adapter: str, model: str, level: str,
               default: bool = False, note: str = '') -> str:
    head = f'[[{table}]]'
    if note:
        head = f'{head:<28}# {note}'
    lines = [head, f'adapter = "{adapter}"', f'model = "{model}"',
             f'max_complexity = "{level}"']
    if default:
        lines.append('default = true')
    return '\n'.join(lines) + '\n'


def default_routing_toml(adapter_name: str, adapter_kind: str, *,
                         judges_available: Optional[bool] = None,
                         decisions: bool = True) -> str:
    """The default ``[routing]`` (and ``[decisions]``) block for the remote
    adapter ``adapter_name`` of kind ``adapter_kind`` (a key of
    :data:`DEFAULT_TIER_MODELS`), as TOML text.

    Ladder: Haiku for trivial, Sonnet (default rung) for standard, Opus
    for hard; a ``review`` ladder starting at Sonnet (``type_router`` on
    so review-kind tasks take it). Judges: the ``labels`` tie-breaker
    (margin 0.15) and the ``panel`` point active on the encoder judge,
    ``injection`` shadow — active only when ``judges_available`` (default:
    probe the ``judge`` extra); otherwise both are written ``false`` with a
    note. ``decisions=False`` omits the ``[decisions]`` table (the file
    already has one). Raises ``ValueError`` for an unknown kind or a name
    that cannot sit in a TOML basic string.
    """
    if adapter_kind not in DEFAULT_TIER_MODELS:
        raise ValueError(
            f'no default ladder for adapter kind {adapter_kind!r}; expected '
            'one of ' + ', '.join(DEFAULT_TIER_MODELS))
    if not adapter_name or any(ch in adapter_name for ch in '"\\\n'):
        raise ValueError(f'adapter name {adapter_name!r} cannot be written '
                         'as a TOML string')
    if judges_available is None:
        judges_available = _judge_extra_available()
    haiku, sonnet, opus = DEFAULT_TIER_MODELS[adapter_kind]
    out = [_ROUTING_HEADER, _ROUTING_TABLE,
           _rung_toml('routing.ladder', adapter_name, haiku, 'trivial',
                      note='trivial: lookups, one-file summaries'),
           '\n',
           _rung_toml('routing.ladder', adapter_name, sonnet, 'standard',
                      default=True,
                      note='standard: a few files, one bug, one edit'),
           '\n',
           _rung_toml('routing.ladder', adapter_name, opus, 'hard',
                      note='hard: multi-file work, whole-repo reviews'),
           '\n',
           _rung_toml('routing.ladders.review', adapter_name, sonnet,
                      'standard', default=True,
                      note='review-kind tasks: never Haiku'),
           '\n',
           _rung_toml('routing.ladders.review', adapter_name, opus, 'hard')]
    if decisions:
        out.append(_DECISIONS_TABLE.format(
            flag='true' if judges_available else 'false',
            note='' if judges_available else _JUDGE_EXTRA_NOTE))
    return ''.join(out)


def _remote_adapter(adapters: list) -> Optional[tuple[str, str]]:
    """``(name, kind)`` of the first enabled remote adapter config."""
    for cfg in adapters:
        if not isinstance(cfg, dict):
            continue
        kind = str(cfg.get('type', ''))
        name = str(cfg.get('name', '') or kind)
        if (kind in REMOTE_ADAPTER_KINDS and cfg.get('enable', True)
                and name):
            return name, kind
    return None


def ensure_default_routing(adapters: Optional[list] = None,
                           path: Optional[Path] = None, *,
                           judges_available: Optional[bool] = None) -> str:
    """Write the default routing block into the settings file when it has
    no ``[routing]`` table and an enabled remote adapter exists.

    ``adapters`` are the raw adapter configs (default
    ``config.load_adapter_configs()``); ``path`` the settings file (default
    ``config.GLOBAL_SETTINGS_PATH``), created when missing. The block is
    appended after the existing text; no existing table is touched, and
    the ``[decisions]`` part is skipped when the file already has one.
    After writing, ``config`` re-reads the settings so the new
    ``[decisions]`` table takes effect in this process.

    Returns ``'written'``, ``'exists'`` (a ``[routing]`` table, even an
    empty one), ``'no-remote'`` (no enabled litellm/anthropic adapter) or
    ``'invalid'`` (the file cannot be parsed or written; nothing changed).
    """
    if adapters is None:
        adapters = config.load_adapter_configs()
    remote = _remote_adapter(adapters)
    if remote is None:
        return 'no-remote'
    target = Path(path) if path is not None else config.GLOBAL_SETTINGS_PATH
    try:
        text = target.read_text(encoding='utf-8')
    except FileNotFoundError:
        text = ''
    except OSError as e:
        log.warning('routing: cannot read %s: %s', target, e)
        return 'invalid'
    try:
        data = tomllib.loads(text)
    except ValueError as e:                        # TOMLDecodeError
        log.warning('routing: %s is not valid TOML (%s); not writing the '
                    'default block', target, e)
        return 'invalid'
    if 'routing' in data:
        return 'exists'
    block = default_routing_toml(*remote, judges_available=judges_available,
                                 decisions='decisions' not in data)
    joiner = '' if not text else ('\n' if text.endswith('\n') else '\n\n')
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + joiner + block, encoding='utf-8')
    except OSError as e:
        log.warning('routing: cannot write %s: %s', target, e)
        return 'invalid'
    config._apply_settings()
    return 'written'


_TABLE_HEADER = re.compile(r'^\s*\[')
_ROUTING_HEADER_LINE = re.compile(r'^\s*\[routing\]\s*(#.*)?$')
_MODE_LINE = re.compile(
    r'^(?P<head>\s*mode\s*=\s*)"(?P<mode>[^"]*)"(?P<gap>[ \t]*)'
    r'(?:#[ \t]*(?P<comment>.*))?$')
_WAS = re.compile(r'^was "(?P<was>[^"]*)"(?: -- (?P<rest>.*))?$')


def _mode_line(head: str, mode: str, comment: str, column: int) -> str:
    """``head"mode"`` plus ``# comment`` (when any) starting at ``column``
    where possible, so a switch keeps the file's comment alignment."""
    value = f'{head}"{mode}"'
    if not comment:
        return value
    return f'{value:<{max(column, len(value) + 2)}}# {comment}'


def switch_routing(on: bool, path: Optional[Path] = None) -> str:
    """Flip ``mode`` in the settings file's ``[routing]`` table: ``on``
    False writes ``mode = "off"`` (remembering the previous mode in a
    trailing ``# was "…"`` comment), True restores that mode (default
    ``local-and-remote``). Only the ``mode`` line changes; every other
    line, key and comment is kept byte for byte. A table without a
    ``mode`` line gets one inserted under its header when switching off.

    Returns the mode now in force. Raises ``ValueError`` when the file
    has no ``[routing]`` table or cannot be read.
    """
    target = Path(path) if path is not None else config.GLOBAL_SETTINGS_PATH
    try:
        lines = target.read_text(encoding='utf-8').splitlines(keepends=True)
    except OSError as e:
        raise ValueError(f'{target}: cannot read: {e}') from e
    start = next((i for i, ln in enumerate(lines)
                  if _ROUTING_HEADER_LINE.match(ln)), None)
    if start is None:
        raise ValueError(f'{target}: no [routing] table to switch')
    end = next((i for i in range(start + 1, len(lines))
                if _TABLE_HEADER.match(lines[i])), len(lines))
    at = next((i for i in range(start + 1, end)
               if _MODE_LINE.match(lines[i])), None)
    if at is None:
        if on:
            return routing.MODES[1]
        lines.insert(start + 1, f'mode = "{MODE_OFF}"\n')
        target.write_text(''.join(lines), encoding='utf-8')
        return MODE_OFF
    match = _MODE_LINE.match(lines[at])
    assert match is not None
    head, mode, comment = (match.group('head'), match.group('mode'),
                           (match.group('comment') or '').strip())
    column = len(head) + len(mode) + 2 + len(match.group('gap'))
    newline = '\n' if lines[at].endswith('\n') else ''
    if not on:
        if mode == MODE_OFF:
            return MODE_OFF
        tail = f'was "{mode}"' + (f' -- {comment}' if comment else '')
        new_mode = MODE_OFF
    else:
        if mode != MODE_OFF:
            return mode
        was = _WAS.match(comment)
        new_mode = (was.group('was') if was else '') or routing.MODES[1]
        tail = (was.group('rest') or '') if was else comment
    lines[at] = _mode_line(head, new_mode, tail, column) + newline
    target.write_text(''.join(lines), encoding='utf-8')
    return new_mode


# --- .guru/tools.toml --------------------------------------------------------

TEST_RUNNERS = ('pytest', 'unittest')
_TOOLS_KEYS = frozenset(('enabled', 'disabled', 'tests', 'limits'))


def _name_list(table: dict, key: str, where: str) -> set:
    raw = table.get(key, [])
    if not isinstance(raw, list) or not all(
            isinstance(x, str) for x in raw):
        raise ValueError(f'{where}: [tools] {key} must be a list of tool '
                         'names')
    return set(raw)


def _tools_limits(raw: object, where: str) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f'{where}: [tools.limits] must be a table')
    unknown = sorted(set(raw) - set(config.PROC_LIMIT_KEYS))
    if unknown:
        raise ValueError(
            f'{where}: [tools.limits] unknown keys: ' + ', '.join(unknown)
            + '; expected ' + ', '.join(config.PROC_LIMIT_KEYS))
    out: dict = {}
    for key, value in raw.items():
        if not _is_number(value) or value <= 0:
            raise ValueError(
                f'{where}: [tools.limits] {key} = {value!r}; expected a '
                'positive number')
        out[str(key)] = int(value)
    return out


def load_tools_policy(path: Optional[Path] = None) -> ToolsPolicy:
    """Parse and validate a project's ``.guru/tools.toml``.

    ``path`` defaults to ``config.TOOLS_POLICY_PATH``. An *absent* file
    yields the default policy (everything enabled); a file that exists but
    cannot be read raises like an invalid one, so the caller can fail
    closed. Raises ``ValueError`` naming the file on an unreadable file,
    invalid TOML, an unknown key or table, a runner outside
    ``TEST_RUNNERS`` or a bad ``[tools.limits]`` value.
    """
    target = Path(path) if path is not None else config.TOOLS_POLICY_PATH
    where = str(target)
    try:
        text = target.read_text(encoding='utf-8')
    except FileNotFoundError:
        return ToolsPolicy()
    except OSError as e:
        raise ValueError(f'{where}: cannot read tool policy: {e}') from e
    try:
        data = tomllib.loads(text)
    except ValueError as e:                        # TOMLDecodeError
        raise ValueError(f'{where}: invalid TOML: {e}') from e
    unknown = sorted(set(data) - {'tools'})
    if unknown:
        raise ValueError(f'{where}: unknown tables: ' + ', '.join(unknown)
                         + '; expected [tools]')
    table = data.get('tools', {})
    if not isinstance(table, dict):
        raise ValueError(f'{where}: [tools] must be a table')
    unknown = sorted(set(table) - _TOOLS_KEYS)
    if unknown:
        raise ValueError(f'{where}: [tools] unknown keys: '
                         + ', '.join(unknown) + '; expected '
                         + ', '.join(sorted(_TOOLS_KEYS)))
    tests = table.get('tests', {})
    if not isinstance(tests, dict) or set(tests) - {'runner'}:
        raise ValueError(f'{where}: [tools.tests] takes only runner')
    runner = tests.get('runner', 'pytest')
    if runner not in TEST_RUNNERS:
        raise ValueError(
            f'{where}: [tools.tests] runner = {runner!r}; expected one of '
            + ', '.join(TEST_RUNNERS))
    return ToolsPolicy(
        enabled=_name_list(table, 'enabled', where),
        disabled=_name_list(table, 'disabled', where),
        test_runner=str(runner),
        limits=_tools_limits(table.get('limits', {}), where))


# --- [sandbox] / .guru/sandbox.toml -----------------------------------------

SANDBOX_RUNTIMES = ('docker',)
# docker.io/library/python:3.12-slim, multi-arch index digest as served on
# 2026-09-24 (``docker buildx imagetools inspect python:3.12-slim``).
DEFAULT_BASE_IMAGE = ('python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a'
                      '686e2cb9a83de48e70534b94cd8ebbe06a9')
# The provisioning proxy: docker.io/kalaksi/tinyproxy (tinyproxy 1.11.3 on
# Alpine, 6.9 MB, runs unprivileged as 57981:57981). Chosen over
# ubuntu/squid for size and because its URL filter expresses exactly
# "CONNECT to <host>:443 only, deny everything else" (guru.sandbox.proxy);
# the tinyproxy project publishes no official image. Multi-arch index
# digest obtained on 2026-09-24 with ``docker pull kalaksi/tinyproxy:latest``
# then ``docker image inspect --format '{{index .RepoDigests 0}}'``
# (amd64 + arm64; ``docker manifest inspect`` of the digest shows an OCI
# image index). Repin the same way when bumping.
DEFAULT_PROXY_IMAGE = ('docker.io/kalaksi/tinyproxy:latest@sha256:fafafc7079c'
                       'a29c6704564de1353f61d038f2166f09b01d4e460e8e499bf6b57')
_SANDBOX_KEYS = frozenset(('enabled', 'runtime', 'base_image', 'cpus',
                           'memory_mb', 'pids', 'timeout_s', 'proxy_image'))
_SANDBOX_INTS = ('memory_mb', 'pids', 'timeout_s')


@dataclass
class SandboxSettings:
    """The validated, merged ``[sandbox]`` table (global defaults, project
    overrides). ``enabled`` is False unless the project file sets it."""
    enabled: bool = False
    runtime: str = 'docker'
    base_image: str = DEFAULT_BASE_IMAGE
    cpus: float = 2.0
    memory_mb: int = 2048
    pids: int = 256
    timeout_s: int = 600
    proxy_image: str = DEFAULT_PROXY_IMAGE


def _pinned(value: object, key: str, where: str) -> str:
    if not isinstance(value, str) or '@sha256:' not in value or any(
            ch.isspace() for ch in value):
        raise ValueError(f'{where}: {key} = {value!r}; expected a '
                         'digest-pinned image reference <image>@sha256:<hex>')
    return value


def _apply_sandbox(out: SandboxSettings, table: dict, where: str,
                   allow_enabled: bool) -> None:
    """Validate ``table`` and apply its keys onto ``out`` in place."""
    unknown = sorted(set(table) - _SANDBOX_KEYS)
    if unknown:
        raise ValueError(f'{where}: [sandbox] unknown keys: '
                         + ', '.join(unknown) + '; expected '
                         + ', '.join(sorted(_SANDBOX_KEYS)))
    if 'enabled' in table:
        if not allow_enabled:
            raise ValueError(f'{where}: [sandbox] enabled may only be set in '
                             'a project\'s .guru/sandbox.toml')
        if not isinstance(table['enabled'], bool):
            raise ValueError(f'{where}: [sandbox] enabled = '
                             f'{table["enabled"]!r}; expected a boolean')
        out.enabled = table['enabled']
    if 'runtime' in table:
        if table['runtime'] not in SANDBOX_RUNTIMES:
            raise ValueError(f'{where}: [sandbox] runtime = '
                             f'{table["runtime"]!r}; expected one of '
                             + ', '.join(SANDBOX_RUNTIMES))
        out.runtime = str(table['runtime'])
    for key in ('base_image', 'proxy_image'):
        if key in table:
            setattr(out, key, _pinned(table[key], key, where))
    if 'cpus' in table:
        cpus = table['cpus']
        if not _is_number(cpus) or cpus <= 0:
            raise ValueError(f'{where}: [sandbox] cpus = {cpus!r}; expected '
                             'a positive number')
        out.cpus = float(cpus)
    for key in _SANDBOX_INTS:
        if key in table:
            value = table[key]
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value <= 0):
                raise ValueError(f'{where}: [sandbox] {key} = {value!r}; '
                                 'expected a positive integer')
            setattr(out, key, int(value))


def _read_toml_table(target: Path, table: str) -> Optional[dict]:
    """The ``[table]`` of the TOML file ``target``; None when the file is
    absent; ``ValueError`` naming the file when it is unreadable, invalid
    TOML, holds other top-level tables or ``table`` is not a table."""
    where = str(target)
    try:
        text = target.read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    except OSError as e:
        raise ValueError(f'{where}: cannot read: {e}') from e
    try:
        data = tomllib.loads(text)
    except ValueError as e:                        # TOMLDecodeError
        raise ValueError(f'{where}: invalid TOML: {e}') from e
    unknown = sorted(set(data) - {table})
    if unknown:
        raise ValueError(f'{where}: unknown tables: ' + ', '.join(unknown)
                         + f'; expected [{table}]')
    section = data.get(table, {})
    if not isinstance(section, dict):
        raise ValueError(f'{where}: [{table}] must be a table')
    return section


def load_sandbox(section: Optional[dict] = None,
                 path: Optional[Path] = None) -> SandboxSettings:
    """Parse and merge the ``[sandbox]`` settings.

    ``section`` is the global table (default
    ``config.settings_section('sandbox')``); ``path`` the project file
    (default ``config.SANDBOX_POLICY_PATH``), whose ``[sandbox]`` table
    overrides the global key by key. ``enabled`` is accepted from the
    project file only. An absent project file leaves ``enabled`` False.
    Raises ``ValueError`` naming the offender on an unknown key, an
    unpinned image, a bad number or a broken project file.
    """
    if section is None:
        section = config.settings_section('sandbox')
    target = Path(path) if path is not None else config.SANDBOX_POLICY_PATH
    out = SandboxSettings()
    _apply_sandbox(out, dict(section), 'settings.toml', allow_enabled=False)
    project = _read_toml_table(target, 'sandbox')
    if project is not None:
        _apply_sandbox(out, project, str(target), allow_enabled=True)
    return out
