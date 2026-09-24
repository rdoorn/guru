"""Aggregate the ledger across days and print a Markdown report.

    .venv/bin/python bench/ledger_report.py [--dir ~/.guru/ledger]

Which models are called how often, task latency p50/p95 per (kind,
complexity), tokens and cost per model, fallback/retry rates, how the
shadow judges agree with the heuristics and with ``/good`` ``/bad`` labels,
and the tools called (per tool: calls, mean seconds, bytes shown vs
produced, denials).
The aggregation lives in ``guru.domain.ledger_report``; this script only
loads the streams and prints.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root

from guru import config                                         # noqa: E402
from guru.domain import ledger_report                           # noqa: E402
from guru.repositories.jsonl_ledger import JsonlLedger          # noqa: E402

STREAMS = ('calls', 'tasks', 'turns', 'decisions', 'labels',
           'tool_events')


def load(directory: Path) -> dict:
    """Every stream's rows from ``directory`` (empty lists when missing)."""
    repo = JsonlLedger(directory)
    return {name: repo.rows(name) for name in STREAMS}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dir', type=Path, default=config.LEDGER_DIR,
                        help=f'ledger directory (default {config.LEDGER_DIR})')
    args = parser.parse_args(argv)
    report = ledger_report.build_report(**load(args.dir))
    print(ledger_report.render_markdown(report), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
