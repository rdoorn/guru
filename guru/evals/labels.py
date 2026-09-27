"""Hand rubric grades: ``evals/rubric-labels.toml`` -> :class:`HandLabel`.

The file is the reference the rubric judges are scored against (``python
-m guru.evals grade``). One ``[[label]]`` table per hand grade:

.. code-block:: toml

    [[label]]
    case = "guru-add-version-flag"   # the case name
    run = "*"                        # a run id, or "*" for any run
    score = 2                        # 0 | 1 | 2, as the rubric judge scores
    note = "why (triage note, date)" # optional

A grade for a specific run wins over a ``*`` grade for the same case; two
grades for the same ``(case, run)`` are a mistake and raise. Unknown keys
and wrong types raise ``ValueError`` naming the offender, as the case
files do. Parsing and lookup are pure; the runner's ledger recording is
elsewhere.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from guru.evals import cases, rubric

DEFAULT_LABELS_FILE = cases.REPO_ROOT / 'evals' / 'rubric-labels.toml'
ANY_RUN = '*'
HAND_LABELLER = 'hand'
_KEYS = ('case', 'run', 'score', 'note')


@dataclass(frozen=True)
class HandLabel:
    """One hand grade: ``score`` for ``case`` in ``run`` (``*`` = any)."""
    case: str
    run: str
    score: int
    note: str = ''


def parse_labels(text: str, where: str = 'rubric-labels.toml'
                 ) -> list[HandLabel]:
    """The ``[[label]]`` tables of ``text``; ``ValueError`` (naming
    ``where``) for invalid TOML, an unknown key, a wrong type, a score
    outside :data:`rubric.SCORES` or a duplicate ``(case, run)``."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f'{where}: invalid TOML: {e}') from e
    for key in data:
        if key != 'label':
            raise ValueError(f'{where}: unknown top-level key {key!r} '
                             '(only [[label]] tables)')
    raw = data.get('label', [])
    if not isinstance(raw, list):
        raise ValueError(f'{where}: label must be an array of tables '
                         '([[label]])')
    out: list[HandLabel] = []
    seen: set[tuple[str, str]] = set()
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f'{where}: label #{i} is not a table')
        for key in item:
            if key not in _KEYS:
                raise ValueError(f'{where}: label #{i}: unknown key {key!r} '
                                 f'(known: {", ".join(_KEYS)})')
        case = item.get('case')
        if not isinstance(case, str) or not case.strip():
            raise ValueError(f'{where}: label #{i}: case must be a '
                             'non-empty string')
        run = item.get('run', ANY_RUN)
        if not isinstance(run, str) or not run.strip():
            raise ValueError(f'{where}: label #{i}: run must be a run id '
                             f'or "{ANY_RUN}"')
        score = item.get('score')
        if isinstance(score, bool) or score not in rubric.SCORES:
            raise ValueError(f'{where}: label #{i}: score must be one of '
                             f'{", ".join(str(s) for s in rubric.SCORES)}, '
                             f'got {score!r}')
        note = item.get('note', '')
        if not isinstance(note, str):
            raise ValueError(f'{where}: label #{i}: note must be a string')
        key_pair = (case.strip(), run.strip())
        if key_pair in seen:
            raise ValueError(f'{where}: label #{i}: duplicate grade for '
                             f'case {key_pair[0]!r}, run {key_pair[1]!r}')
        seen.add(key_pair)
        out.append(HandLabel(key_pair[0], key_pair[1], int(score),
                             ' '.join(note.split())))
    return out


def load_labels(path: Optional[Path] = None) -> list[HandLabel]:
    """Parse the labels file (default :data:`DEFAULT_LABELS_FILE`); an
    absent file is an empty list, an unreadable or invalid one raises
    ``ValueError``."""
    target = Path(path) if path is not None else DEFAULT_LABELS_FILE
    try:
        text = target.read_text(encoding='utf-8')
    except FileNotFoundError:
        return []
    except OSError as e:
        raise ValueError(f'{target}: cannot read labels file: {e}') from e
    return parse_labels(text, str(target))


def hand_label(labels: list[HandLabel], case: str, run_id: str
               ) -> Optional[HandLabel]:
    """The grade for ``case`` in ``run_id``: the run-specific one when
    present, else the ``*`` one, else None."""
    fallback: Optional[HandLabel] = None
    for lab in labels:
        if lab.case != case:
            continue
        if lab.run == run_id:
            return lab
        if lab.run == ANY_RUN:
            fallback = lab
    return fallback


def agreement(pairs: list[tuple[Optional[int], Optional[int]]]
              ) -> tuple[int, int]:
    """``(agreed, compared)`` over ``(judge score, hand score)`` pairs;
    a pair with either side None is not compared."""
    both = [(j, h) for j, h in pairs if j is not None and h is not None]
    return sum(1 for j, h in both if j == h), len(both)
