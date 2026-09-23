"""Regex secret scanner: a ``ContentScanner`` for keys, tokens and private
key blocks, plus per-project sensitive markers and an allow-list.

The provider patterns key on distinctive prefixes. The generic
``<name>[:=] <value>`` rule is deliberately broad on the *name* (any word
ending in api_key/secret/password/token, optionally followed by ``_key``)
and strict on the *value*: 16+ token characters that are not a hex digest, a
UUID, or a Python call/subscript/attribute — so ``os.environ[...]`` lookups,
function calls, checksums and ids do not fire. It still fires on some
non-secrets, e.g. masked or prose values (``password: ****************``);
that direction is fail-safe: a false finding only keeps a task on a local
model and redacts a span, it never leaks anything.

Findings are ``guru.domain.policy.Finding`` spans whose ``sample`` is masked
(first four characters, an ellipsis, the span length) so the ledger and UI
never carry more than four characters of a secret. ``load_project_scanner``
reads the project's marker and allow files.
"""
from __future__ import annotations

import re
from typing import Iterable

from guru import config, log
from guru.domain.policy import Finding

SAMPLE_CHARS = 4

# (kind, compiled regex). Group 1, when present, is the secret span; otherwise
# the whole match is. Order matters only for the sample kind of a collapsed
# duplicate span: specific kinds come before the generic rule.
PATTERNS: list = [
    ('aws_access_key', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    ('aws_secret_key', re.compile(
        r'aws_secret_access_key\s*[:=]\s*["\']?([A-Za-z0-9/+=]{40})'
        r'(?![A-Za-z0-9/+=])', re.IGNORECASE)),
    ('github_token', re.compile(r'\bgh[pousr]_[A-Za-z0-9]{36,}\b')),
    ('slack_token', re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{10,}\b')),
    ('google_api_key', re.compile(
        r'\bAIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])')),
    # The whole block up to the matching END line; header only when the END
    # line is missing (a truncated paste is still a private key).
    ('private_key', re.compile(
        r'-----BEGIN [A-Z ]*PRIVATE KEY-----'
        r'(?:[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----)?')),
    # No leading \b: client_secret, access_token, GITHUB_TOKEN, db_password
    # and the suffix forms secret_key / api_key_id all end in one of these.
    ('generic_secret', re.compile(
        r'(?:api[_-]?key|secret|password|token)(?:[_-]?key)?\s*[:=]\s*'
        r'["\']?([A-Za-z0-9_\-!@#$%^&*/+=]{16,})', re.IGNORECASE)),
    ('jwt', re.compile(
        r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b')),
]

# A generic value that is a plain hex digest or a UUID is not a secret
# assignment we can act on (it is far more often a checksum or an id).
_HEX = re.compile(r'^[0-9a-f]{16,}$', re.IGNORECASE)
_UUID = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE)


def _plausible_generic(value: str, tail: str) -> bool:
    """Reject generic matches that are code rather than a literal secret:
    an identifier followed by ``(``, ``[`` or ``.`` (a call/lookup/attr) or
    a hex digest / UUID."""
    if tail[:1] in ('(', '[', '.'):
        return False
    if _HEX.match(value) or _UUID.match(value):
        return False
    return True


def mask(secret: str) -> str:
    """``'AKIA…(20)'``: the first ``SAMPLE_CHARS`` characters, an ellipsis
    and the full length — enough to recognise a finding, never enough to
    reconstruct it."""
    return f'{secret[:SAMPLE_CHARS]}…({len(secret)})'


class SecretScanner:
    """``ContentScanner`` over ``PATTERNS`` plus literal project markers.

    ``markers`` are case-insensitive literals (project code names, customer
    names) reported as kind ``marker``. ``allow`` holds regexes; a finding is
    suppressed when one matches the secret itself or the line it sits on
    (so an inline ``# scan-ok`` style annotation can be allowed).
    """

    def __init__(self, markers: Iterable[str] = (),
                 allow: Iterable[str] = ()) -> None:
        self.markers = [m.strip() for m in markers if m.strip()]
        self.allow = list(allow)
        self._markers = [re.compile(re.escape(m), re.IGNORECASE)
                         for m in self.markers]
        self._allow = []
        for pattern in self.allow:
            try:
                self._allow.append(re.compile(pattern))
            except re.error as exc:
                log.warning('scan_allow: skipping invalid regex %r (%s)',
                            pattern, exc)

    def _allowed(self, text: str, start: int, end: int) -> bool:
        if not self._allow:
            return False
        line_start = text.rfind('\n', 0, start) + 1
        line_end = text.find('\n', end)
        line = text[line_start:len(text) if line_end < 0 else line_end]
        return any(rx.search(line) for rx in self._allow)

    def scan(self, text: str) -> list:
        """Findings sorted by start; one finding per distinct span."""
        if not text:
            return []
        found: dict = {}
        for kind, rx in PATTERNS:
            for m in rx.finditer(text):
                group = 1 if m.lastindex else 0
                start, end = m.start(group), m.end(group)
                if kind == 'generic_secret' and not _plausible_generic(
                        m.group(1), text[end:end + 1]):
                    continue
                found.setdefault((start, end), kind)
        for rx in self._markers:
            for m in rx.finditer(text):
                found.setdefault((m.start(), m.end()), 'marker')
        out = []
        for (start, end), kind in sorted(found.items()):
            if self._allowed(text, start, end):
                continue
            out.append(Finding(kind, start, end, mask(text[start:end])))
        return out


def _read_lines(path) -> list:
    """Non-blank, non-comment lines of ``path`` (missing file -> [])."""
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            out.append(line)
    return out


def load_project_scanner() -> SecretScanner:
    """A scanner with the project's ``.guru/sensitive_markers.txt`` literals
    and ``.guru/scan_allow.txt`` regexes (either may be absent)."""
    return SecretScanner(
        markers=_read_lines(config.SENSITIVE_MARKERS_PATH),
        allow=_read_lines(config.SCAN_ALLOW_PATH))
