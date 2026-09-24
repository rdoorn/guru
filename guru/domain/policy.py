"""Content policy: findings, redaction and the pluggable scanner seam.

A ``ContentScanner`` (the regex secret scanner in ``guru.scanners`` is one)
returns ``Finding`` spans over a text. The rule "any finding forces local"
lives in ``guru.domain.routing.resolve`` (``scan_findings``); this module
owns the finding shape, ``redact`` for text bound to a remote provider, and
the process-wide scanner binding. ``scan`` never raises: a broken scanner
logs and yields no findings, so the orchestrator keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from guru import log


@dataclass(frozen=True)
class Finding:
    """A sensitive span ``[start, end)`` of kind ``kind`` (e.g.
    ``aws_access_key``, ``marker``). ``sample`` is a short excerpt for the
    ledger/UI; it may be truncated and must not be relied on for length."""
    kind: str
    start: int
    end: int
    sample: str


@runtime_checkable
class ContentScanner(Protocol):
    """Anything that finds sensitive spans in a text."""

    def scan(self, text: str) -> list[Finding]: ...


_scanner: Optional[ContentScanner] = None


def set_scanner(scanner: Optional[ContentScanner]) -> None:
    """Bind the process-wide scanner (None disables scanning)."""
    global _scanner
    _scanner = scanner


def scanner() -> Optional[ContentScanner]:
    """The bound scanner, or None when scanning is off."""
    return _scanner


def scan(text: str) -> list[Finding]:
    """Findings from the bound scanner; ``[]`` without one, on empty text,
    or when the scanner raises (logged)."""
    if _scanner is None or not text:
        return []
    try:
        return list(_scanner.scan(text))
    except Exception:
        log.exc('content scanner failed; treating text as clean')
        return []


def redact(text: str, findings: list[Finding]) -> str:
    """Replace each finding's span with ``[REDACTED:<kind>]``.

    Spans are clamped to the text and sorted; overlapping spans are merged
    (the earliest finding's kind labels the merged span) while merely
    adjacent spans stay separate, each keeping its own kind; empty spans are
    ignored. Replacement runs from the end so earlier offsets stay valid.
    """
    spans: list[tuple] = sorted(
        (max(0, f.start), min(len(text), f.end), f.kind)
        for f in findings)
    merged: list[list] = []
    for start, end, kind in spans:
        if end <= start:
            continue
        if merged and start < merged[-1][1]:      # overlap (not adjacency)
            merged[-1][1] = max(merged[-1][1], end)
            continue
        merged.append([start, end, kind])
    for start, end, kind in reversed(merged):
        text = f'{text[:start]}[REDACTED:{kind}]{text[end:]}'
    return text
