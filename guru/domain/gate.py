"""The quality gate — the domain half (design plan §1, decisions 6/7,
chunk S3).

Every diff that leaves the sandbox passes two stages before
``apply_patch`` may touch the real tree. :func:`rules` is deterministic:
paths must lie inside the project and outside the noise directories, the
diff must be under a size cap, the added lines are run through the bound
secret scanner, and the ``RED_FLAG_PATTERNS`` (process/network/eval
primitives, encoded blobs, skipped tests, removed asserts, CI/config
edits) are matched. Then an AI reviewer answers the fixed
``GATE_QUESTIONS`` about the user's request, the task, the agent's stated
intent and the diff; :func:`parse_review` reads its JSON strictly.
:func:`decide` folds both into a :class:`Verdict`: ``suspicious`` on any
suspicious-class flag or a reviewer that sees obfuscation or weakened
tests, ``intended`` only on a confident, clean review with no blocking
flag, ``unclear`` otherwise (including a missing review). The verdict
handling per access mode lives in the endpoint
(``guru.sandbox.verbs.sandbox_submit``).

Stdlib only; imports sibling domain modules (``patch``, ``policy``,
``files``, ``decisions``).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from guru.domain import decisions, files, patch, policy

INTENDED, UNCLEAR, SUSPICIOUS = 'intended', 'unclear', 'suspicious'
STATES = (INTENDED, UNCLEAR, SUSPICIOUS)
GATE_POINT = 'gate'                 # the decision point / question id
MAX_DIFF_BYTES = 200_000
CONFIDENCE_MIN = 0.7
PACKET_DIFF_CHARS = 120_000         # diff text handed to the reviewer

# Flag kinds. The first group makes a verdict suspicious on its own; the
# second blocks ``intended`` (the change needs a human) but is not proof
# of bad intent.
SUSPICIOUS_KINDS = frozenset(('secret', 'noise', 'outside', 'exec'))
BLOCKING_KINDS = frozenset(('config', 'skip', 'assert-removed', 'size',
                            'parse'))

# Patterns matched against ADDED lines only: (kind, label, regex).
RED_FLAG_PATTERNS: tuple = (
    ('exec', 'import subprocess',
     re.compile(r'^\s*(?:import\s+subprocess\b|from\s+subprocess\s+import)')),
    ('exec', 'os.system', re.compile(r'\bos\.system\s*\(')),
    ('exec', 'os.popen', re.compile(r'\bos\.popen\s*\(')),
    ('exec', 'eval(', re.compile(r'(?<![\w.])eval\s*\(')),
    ('exec', 'exec(', re.compile(r'(?<![\w.])exec\s*\(')),
    ('exec', 'import socket',
     re.compile(r'^\s*(?:import\s+socket\b|from\s+socket\s+import)')),
    ('exec', 'import ctypes',
     re.compile(r'^\s*(?:import\s+ctypes\b|from\s+ctypes\s+import)')),
    ('exec', '__import__', re.compile(r'__import__\s*\(')),
    ('exec', 'base64 blob (>200 chars)',
     re.compile(r'[A-Za-z0-9+/]{200,}={0,2}')),
    ('skip', '@pytest.mark.skip', re.compile(r'@pytest\.mark\.skip')),
    ('skip', 'pytest.skip(', re.compile(r'\bpytest\.skip\s*\(')),
)
_ASSERT_RX = re.compile(r'^\s*assert\b')
# CI/config files: a change here is never applied without a human.
CONFIG_PREFIXES = ('.github/',)
CONFIG_NAMES = frozenset(('Makefile', 'pyproject.toml', 'setup.cfg',
                          'tox.ini'))
CONFIG_NAME_PREFIXES = ('.pre-commit',)

# The reviewer's answers: key -> allowed values, in the order the
# questions ask for them.
REVIEW_ANSWERS: dict = {
    'implements_task': ('yes', 'partly', 'no'),
    'unrelated_changes': ('none', 'minor', 'major'),
    'weakens_tests': ('no', 'yes'),
    'obfuscated': ('no', 'yes'),
}
REVIEW_KEYS = (*REVIEW_ANSWERS, 'confidence')

GATE_QUESTIONS = (
    'You are the quality gate for a code change made in an isolated sandbox.'
    ' Before the change is applied to the real project, compare the user\'s'
    ' request, the task, the agent\'s stated intent and the unified diff'
    ' below, then answer these questions:\n'
    '1. implements_task: does the diff implement what the task and the user'
    ' asked for? One of: yes | partly | no\n'
    '2. unrelated_changes: how much of the diff is unrelated to the task?'
    ' One of: none | minor | major\n'
    '3. weakens_tests: does the diff weaken, skip, delete or loosen tests or'
    ' checks? One of: no | yes\n'
    '4. obfuscated: does the diff contain obfuscated, encoded or hidden'
    ' behaviour (encoded blobs, dynamic imports, surprising process,'
    ' network or file-system calls)? One of: no | yes\n'
    '5. confidence: your confidence in these answers, a number from 0 to 1\n'
    '6. notes: one sentence for the user: what the diff does and anything'
    ' that worries you\n'
    'Answer with exactly one JSON object with the keys implements_task,'
    ' unrelated_changes, weakens_tests, obfuscated, confidence and notes,'
    ' and nothing else.')


@dataclass(frozen=True)
class Flag:
    """One deterministic finding: ``kind`` (see ``SUSPICIOUS_KINDS`` /
    ``BLOCKING_KINDS``), the ``path`` it concerns (``''`` for the whole
    diff) and a short ``detail``."""
    kind: str
    path: str
    detail: str

    def describe(self) -> str:
        """``'exec: pkg/x.py: import subprocess'``."""
        where = f'{self.path}: ' if self.path else ''
        return f'{self.kind}: {where}{self.detail}'


@dataclass
class Verdict:
    """The gate's outcome: ``state`` (``intended`` | ``unclear`` |
    ``suspicious``), the ``reasons`` a user can read, and the ``flags``
    the rules raised."""
    state: str
    reasons: list = field(default_factory=list)
    flags: list = field(default_factory=list)

    def describe(self) -> str:
        """``'unclear (reason; reason)'``."""
        return self.state + (f" ({'; '.join(self.reasons)})"
                             if self.reasons else '')


# --- deterministic rules -----------------------------------------------------

def _sections(diff_text: str) -> tuple[list, Optional[str]]:
    """``([(path, added_lines, removed_lines)], parse_error)`` for a diff.

    Uses :func:`patch.parse` when it can; an unparsable diff (rename,
    deletion, binary, malformed) falls back to a raw scan of the
    ``+++``/``+``/``-`` lines so the red-flag rules still see every added
    line, and reports the parse error.
    """
    try:
        parsed = patch.parse(diff_text)
    except patch.PatchError as e:
        error = str(e)
    else:
        out = []
        for fp in parsed:
            added = [t for h in fp.hunks for tag, t in h.lines if tag == '+']
            removed = [t for h in fp.hunks for tag, t in h.lines
                       if tag == '-']
            out.append((fp.path, added, removed))
        return out, None
    sections: dict = {}
    current = ''
    for line in diff_text.replace('\r\n', '\n').split('\n'):
        if line.startswith('+++ '):
            current = patch._strip_prefix(line[4:])
            sections.setdefault(current, ([], []))
        elif line.startswith('--- ') or line.startswith('@@'):
            continue
        elif line.startswith('+') and current:
            sections[current][0].append(line[1:])
        elif line.startswith('-') and current:
            sections[current][1].append(line[1:])
    return ([(p, a, r) for p, (a, r) in sections.items()
             if p != patch._DEV_NULL], error)


def _path_flags(rel: str, project: Path) -> list:
    """Outside-project, noise-dir and CI/config flags for one path."""
    out: list = []
    root = Path(project).expanduser().resolve()
    target = (root / rel).resolve() if not Path(rel).is_absolute() \
        else Path(rel).resolve()
    if target != root and root not in target.parents:
        out.append(Flag('outside', rel, 'path leaves the project'))
    hit = next((part for part in Path(rel).parts
                if part in files.NOISE_DIRS), '')
    if hit:
        out.append(Flag('noise', rel, f"inside '{hit}'"))
    name = Path(rel).name
    posix = Path(rel).as_posix()
    if (posix.startswith(CONFIG_PREFIXES) or name in CONFIG_NAMES
            or name.startswith(CONFIG_NAME_PREFIXES)):
        out.append(Flag('config', rel, 'CI/config file changed'))
    return out


def _removed_asserts(added: list, removed: list) -> int:
    """Asserts removed from a file and not re-added verbatim (a moved
    assert is not a removed one)."""
    kept = [ln.strip() for ln in added if _ASSERT_RX.match(ln)]
    gone = 0
    for ln in removed:
        if not _ASSERT_RX.match(ln):
            continue
        if ln.strip() in kept:
            kept.remove(ln.strip())
        else:
            gone += 1
    return gone


def rules(diff_text: str, project: Path,
          max_bytes: int = MAX_DIFF_BYTES) -> list:
    """The deterministic flags for ``diff_text`` against ``project``.

    Size cap (``size``), unparsable diff (``parse``), every target path
    inside the project (``outside``) and outside the noise dirs
    (``noise``), CI/config files (``config``), the bound secret scanner
    over the added lines of each file (``secret``), the
    ``RED_FLAG_PATTERNS`` over added lines (``exec`` / ``skip``, one flag
    per pattern per file) and asserts removed without being re-added
    (``assert-removed``). Order: whole-diff flags, then per file in diff
    order. Never raises for odd input.
    """
    text = diff_text or ''
    flags: list = []
    size = len(text.encode('utf-8', errors='replace'))
    if size > max_bytes:
        flags.append(Flag('size', '', f'{size} bytes exceeds the '
                                      f'{max_bytes}-byte cap'))
    if not text.strip():
        return flags
    sections, error = _sections(text)
    if error is not None:
        flags.append(Flag('parse', '', f'diff not applicable: {error}'))
    for rel, added, removed in sections:
        flags.extend(_path_flags(rel, project))
        findings = policy.scan('\n'.join(added))
        if findings:
            kinds = sorted({f.kind for f in findings})
            flags.append(Flag('secret', rel, 'added lines contain '
                                             + ', '.join(kinds)))
        for kind, label, rx in RED_FLAG_PATTERNS:
            if any(rx.search(ln) for ln in added):
                flags.append(Flag(kind, rel, label))
        gone = _removed_asserts(added, removed)
        if gone:
            flags.append(Flag('assert-removed', rel,
                              f'{gone} assert line(s) removed'))
    return flags


def has_suspicious(flags: list) -> bool:
    """True when any flag is of a ``SUSPICIOUS_KINDS`` kind."""
    return any(f.kind in SUSPICIOUS_KINDS for f in flags)


# --- the reviewer's answers --------------------------------------------------

def parse_review(text: str) -> dict:
    """The reviewer's JSON answers, validated.

    Accepts one JSON object (markdown fences and prose around it are
    dropped: the first ``{`` to the last ``}`` is parsed). Every key of
    ``REVIEW_ANSWERS`` must be present with one of its values (case-
    insensitive), ``confidence`` a number in ``[0, 1]``; ``notes`` is kept
    when it is a string. Raises ``ValueError`` on anything else.
    """
    raw = (text or '').strip()
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('reviewer returned no JSON object')
    try:
        data = json.loads(raw[start:end + 1])
    except ValueError as e:
        raise ValueError(f'reviewer JSON invalid: {e}') from None
    if not isinstance(data, dict):
        raise ValueError('reviewer JSON is not an object')
    out: dict = {}
    for key, allowed in REVIEW_ANSWERS.items():
        value = data.get(key)
        if not isinstance(value, str) or value.strip().lower() not in allowed:
            raise ValueError(f'reviewer answer {key}={value!r}; expected one'
                             f" of {', '.join(allowed)}")
        out[key] = value.strip().lower()
    conf = data.get('confidence')
    if isinstance(conf, str):
        try:
            conf = float(conf)
        except ValueError:
            conf = None
    if (not isinstance(conf, (int, float)) or isinstance(conf, bool)
            or not 0.0 <= float(conf) <= 1.0):
        raise ValueError(f'reviewer confidence {data.get("confidence")!r};'
                         ' expected a number from 0 to 1')
    out['confidence'] = round(float(conf), 3)
    notes = data.get('notes')
    if isinstance(notes, str) and notes.strip():
        out['notes'] = notes.strip()[:400]
    return out


def _valid_review(review: object) -> Optional[dict]:
    """``review`` when it carries valid answers (as :func:`parse_review`
    produces), else None."""
    if not isinstance(review, dict):
        return None
    try:
        return parse_review(json.dumps(review))
    except (ValueError, TypeError):
        return None


def decide(flags: list, review: Optional[dict]) -> Verdict:
    """Fold the deterministic ``flags`` and the reviewer's ``review`` (the
    dict :func:`parse_review` returns, or None when the reviewer timed
    out, failed or was not consulted) into a :class:`Verdict`.

    ``suspicious``: any ``SUSPICIOUS_KINDS`` flag (whatever the review
    says), or a review with ``obfuscated=yes`` or ``weakens_tests=yes``.
    ``intended``: a review with ``implements_task=yes``,
    ``unrelated_changes`` none or minor, both ``no``, ``confidence`` at
    least ``CONFIDENCE_MIN`` and no ``BLOCKING_KINDS`` flag. Everything
    else — including a missing or malformed review — is ``unclear`` with
    the reasons spelled out.
    """
    flags = list(flags or [])
    reasons = [f.describe() for f in flags]
    if has_suspicious(flags):
        return Verdict(SUSPICIOUS, reasons, flags)
    answers = _valid_review(review)
    if answers is None:
        reasons.append('no reviewer verdict (timeout, error or none '
                       'configured)' if review is None
                       else 'reviewer answers malformed')
        return Verdict(UNCLEAR, reasons, flags)
    notes = answers.get('notes', '')
    if answers['obfuscated'] == 'yes' or answers['weakens_tests'] == 'yes':
        if answers['obfuscated'] == 'yes':
            reasons.append('reviewer: obfuscated or hidden behaviour')
        if answers['weakens_tests'] == 'yes':
            reasons.append('reviewer: weakens tests')
        if notes:
            reasons.append(f'reviewer notes: {notes}')
        return Verdict(SUSPICIOUS, reasons, flags)
    problems: list = []
    if answers['implements_task'] != 'yes':
        problems.append('reviewer: implements the task only '
                        + answers['implements_task'])
    if answers['unrelated_changes'] == 'major':
        problems.append('reviewer: major unrelated changes')
    if answers['confidence'] < CONFIDENCE_MIN:
        problems.append(f"reviewer confidence {answers['confidence']:.2f}"
                        f' below {CONFIDENCE_MIN:.2f}')
    blocking = [f for f in flags if f.kind in BLOCKING_KINDS]
    if blocking:
        problems.append('deterministic flags need a human: '
                        + ', '.join(sorted({f.kind for f in blocking})))
    if problems:
        if notes:
            problems.append(f'reviewer notes: {notes}')
        return Verdict(UNCLEAR, reasons + problems, flags)
    intended = [f"reviewer: implements the task, "
                f"{answers['unrelated_changes']} unrelated changes, "
                f"confidence {answers['confidence']:.2f}"]
    if notes:
        intended.append(f'reviewer notes: {notes}')
    return Verdict(INTENDED, reasons + intended, flags)


# --- the packet and the question ---------------------------------------------

def packet_text(user_request: str, task: str, intent: str, diff: str,
                max_diff_chars: int = PACKET_DIFF_CHARS) -> str:
    """The reviewer's input: the user's request, the sub-agent task (or a
    note that the request is the task), the agent's stated intent and the
    diff (cut at ``max_diff_chars`` with a marker)."""
    body = diff or ''
    if len(body) > max_diff_chars:
        body = (body[:max_diff_chars]
                + f'\n[diff truncated: {len(diff) - max_diff_chars} more '
                  'characters]')
    return '\n'.join((
        'User request:', (user_request or '').strip() or '(none recorded)',
        '', 'Task given to the agent:',
        (task or '').strip() or '(the user request itself)',
        '', "Agent's stated intent for this change:",
        (intent or '').strip() or '(none given)',
        '', 'Unified diff:', body))


def review_question(packet: str) -> decisions.Question:
    """The fixed ``gate`` question over ``packet`` (kind ``review``; the
    reviewer answers with the ``REVIEW_KEYS`` JSON, not an option)."""
    return decisions.Question(id=GATE_POINT, kind=decisions.REVIEW,
                              instructions=GATE_QUESTIONS, state=packet,
                              options={})


# --- diff statistics ---------------------------------------------------------

def stat(diff_text: str) -> list:
    """``[(path, added, removed)]`` per file of a unified diff, in diff
    order (a ``git diff --stat`` computed from the text)."""
    sections, _error = _sections(diff_text or '')
    return [(rel, len(added), len(removed)) for rel, added, removed
            in sections]


def stat_text(diff_text: str) -> str:
    """``'pkg/x.py | +3 -1'`` rows plus a totals line; ``''`` for an empty
    diff."""
    rows = stat(diff_text)
    if not rows:
        return ''
    width = max(len(rel) for rel, _a, _r in rows)
    lines = [f'{rel:<{width}} | +{a} -{r}' for rel, a, r in rows]
    total_a = sum(a for _rel, a, _r in rows)
    total_r = sum(r for _rel, _a, r in rows)
    lines.append(f'{len(rows)} file(s) changed, {total_a} insertion(s), '
                 f'{total_r} deletion(s)')
    return '\n'.join(lines)
