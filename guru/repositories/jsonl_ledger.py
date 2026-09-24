"""Append-only JSONL ledger: one file per stream per UTC day."""
from __future__ import annotations

import gzip
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from guru import log


class JsonlLedger:
    """``LedgerRepository`` over ``<dir>/<stream>-YYYY-MM-DD.jsonl``."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.disabled = False
        self._lock = threading.Lock()

    def _path(self, stream: str) -> Path:
        day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        return self.dir / f'{stream}-{day}.jsonl'

    def append(self, stream: str, row: dict) -> None:
        """Append one row; on the first OSError disable this repository."""
        if self.disabled:
            return
        try:
            with self._lock:
                self.dir.mkdir(parents=True, exist_ok=True)
                with self._path(stream).open('a', encoding='utf-8') as fh:
                    fh.write(json.dumps(row, ensure_ascii=False,
                                        default=str) + '\n')
        except OSError:
            self.disabled = True
            log.exc(f'ledger disabled: cannot write {self.dir}')

    def transcript_path(self, task_id: str) -> Path:
        """Where ``save_transcript`` puts ``task_id``'s transcript."""
        return self.dir / 'transcripts' / f'{task_id}.json.gz'

    def save_transcript(self, task_id: str, messages: list) -> Path:
        """Write ``messages`` as gzip JSON to :meth:`transcript_path`.

        Returns the path. Raises OSError when disabled or unwritable (the
        domain helper swallows it).
        """
        if self.disabled:
            raise OSError('ledger disabled')
        target = self.transcript_path(task_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(target, 'wt', encoding='utf-8') as fh:
            json.dump(messages, fh, ensure_ascii=False, default=str)
        return target

    def rows(self, stream: str, run_id: Optional[str] = None) -> list:
        """All rows of ``stream`` across days, oldest first; with ``run_id``
        only that run's rows.

        Corrupt (non-JSON or non-object) lines are skipped. Reads bypass
        the write lock: a half-written last line is dropped as corrupt.
        """
        out: list = []
        for p in sorted(self.dir.glob(f'{stream}-*.jsonl')):
            try:
                lines = p.read_text(encoding='utf-8').splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                if run_id is None or row.get('run_id') == run_id:
                    out.append(row)
        return out
