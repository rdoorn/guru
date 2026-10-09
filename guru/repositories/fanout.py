"""A ledger repository that writes every row to a primary repository and
to extra sinks (the CLI: the JSONL ledger plus the SQLite usage store).

The primary answers everything else (``rows``, ``save_transcript``,
``transcript_path``, ``dir``); a failing sink never stops the others.
"""
from __future__ import annotations

from typing import Any

from guru import log


class FanOutLedger:
    """``append`` to ``primary`` and each of ``extra``."""

    def __init__(self, primary: Any, *extra: Any) -> None:
        self.primary = primary
        self.extra = [e for e in extra if e is not None]

    def append(self, stream: str, row: dict) -> None:
        for sink in [self.primary, *self.extra]:
            try:
                sink.append(stream, row)
            except Exception:                            # noqa: BLE001
                log.exc(f'ledger sink {type(sink).__name__} failed'
                        f' ({stream})')

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)
