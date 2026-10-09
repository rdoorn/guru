"""The answer check: before the lead's final answer to a request it worked
on (spawned workers or changed files), a reviewer compares the answer with
the request and with the work actually done (the repository's changes),
once per request.

Evidence (dogfood evals 2026-10-08): in three runs out of three the
workers built and tested the usage store but nothing installed it in the
running application, and one answer opened with "wired into the existing
ledger" while saying further down that it was not installed. The rubric
judge passed both. The check catches the claim the changes do not support
and the requirement left undone, and sends the answer back to the lead
once to finish it.

This module holds the rules: the Protocol, the prompt, the strict verdict
parser and the texts the lead sees. The reviewer that calls a model
is ``guru.judges.claims``; the CLI and the eval runner install it with
:func:`set_checker`. Without one installed nothing is checked.
"""
from __future__ import annotations

import json
import re
from typing import Optional, Protocol

from guru import log

MAX_PROBLEMS = 5
# The evidence cap: the diff and the new files, cut to keep the check one
# cheap call (~8k tokens).
MAX_EVIDENCE_CHARS = 32000
# The prefix of the send-back that carries the problems to the lead
# (``already_checked`` finds it in the history).
PREFIX = 'Answer checked against the work: '


class AnswerChecker(Protocol):
    """Compares a final answer with the request and the work done."""

    def check(self, request: str, answer: str) -> list[str]:
        """The problems found (one sentence each); empty when the answer
        is true to the work and the request is met. Never raises."""


_checker: Optional[AnswerChecker] = None


def set_checker(checker: Optional[AnswerChecker]) -> None:
    """Install the answer checker (None: nothing is checked)."""
    global _checker
    _checker = checker


def checker() -> Optional[AnswerChecker]:
    """The installed answer checker, or None."""
    return _checker


INSTRUCTIONS = (
    'You check a coding assistant\'s final answer before the user sees it.'
    ' Compare it with the REQUEST and with the EVIDENCE: the repository\'s'
    ' changes (git diff and new files). Report a problem only when the'
    ' evidence shows it:\n'
    '1. a claim in the ANSWER the changes do not support (for example it'
    ' says something is wired in, installed, enabled or tested, and the'
    ' changes do not show it);\n'
    '2. a requirement the REQUEST states that the changes leave undone,'
    ' even when the answer admits it.\n'
    'The evidence may be cut at a size limit: something missing from cut'
    ' evidence is not evidence that it was not done. Do not report style,'
    ' design preferences or things the request did not ask for. Reply'
    ' with one JSON object: {"problems": ["..."]}, at'
    f' most {MAX_PROBLEMS} problems, each one sentence naming the file or'
    ' the requirement; {"problems": []} when the answer is accurate and'
    ' the request is met.')


def prompt(request: str, answer: str, evidence: str) -> str:
    """The reviewer's prompt; the evidence is cut at the cap."""
    if len(evidence) > MAX_EVIDENCE_CHARS:
        evidence = evidence[:MAX_EVIDENCE_CHARS] + '\n… (evidence cut)'
    return (f'{INSTRUCTIONS}\n\nREQUEST:\n{request}\n\nANSWER:\n{answer}'
            f'\n\nEVIDENCE:\n{evidence or "(no changes in the repository)"}')


_OBJECT_RE = re.compile(r'\{.*\}', re.DOTALL)


def parse(text: str) -> Optional[list[str]]:
    """The problems in a reviewer reply, or None when the reply is not the
    expected JSON object (the check then passes: it never blocks an
    answer on a reviewer's garbage)."""
    match = _OBJECT_RE.search(text or '')
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    problems = data.get('problems') if isinstance(data, dict) else None
    if not isinstance(problems, list) or not all(
            isinstance(p, str) for p in problems):
        return None
    return [p.strip() for p in problems if p.strip()][:MAX_PROBLEMS]


def problems_text(problems: list[str]) -> str:
    """The send-back that carries the answer's problems to the lead."""
    lines = '\n'.join(f'- {p}' for p in problems)
    return (f'{PREFIX}not delivered.\n{lines}\nDo the missing part now'
            ' (yourself or with a worker), or answer stating exactly what'
            ' is not done. Do not claim what the changes do not show.')


def already_checked(messages: list) -> bool:
    """Whether this request's answer was checked already: the problems
    text in ``messages`` (the request's span)."""
    return any(isinstance(m, dict) and m.get('role') in ('tool', 'user')
               and str(m.get('content', '')).startswith(PREFIX)
               for m in messages)


# The calls that count as work on a request: delegating it, or a write.
_WORK_TOOLS = frozenset(('spawn', 'write_file', 'edit_file', 'apply_patch',
                         'delete_file', 'sandbox_submit', 'apply_work'))


# Results of a call that did nothing (refused, failed, invalid).
_NOT_DONE = ('Refused', 'Tool error:', 'Invalid arguments', 'Not run:',
             'Already called', 'Unknown tool:', 'Could not spawn',
             'Nothing to')


def worked_on(messages: list) -> bool:
    """Whether ``messages`` (a request's span) hold work: a ``spawn`` or
    write call that did something, or a mailbox delivery (the workers'
    results)."""
    from guru.domain import conversation
    for m in messages:
        if not isinstance(m, dict):
            continue
        if (m.get('role') == 'tool' and m.get('tool_name') in _WORK_TOOLS
                and not str(m.get('content', '')).startswith(_NOT_DONE)):
            return True
        if (m.get('role') == 'user'
                and conversation.is_mailbox(str(m.get('content', '')))):
            return True
    return False


def check_answer(messages: list, answer: str) -> Optional[str]:
    """The answer check for the lead's final answer: once per request,
    when the request's span holds work (:func:`worked_on`). Returns the
    problems text that sends it back, or None to deliver it. Nothing is
    checked without an installed checker (``[decisions] answer_check =
    false`` installs none)."""
    from guru.domain import conversation
    if not answer or _checker is None:
        return None
    span = messages[conversation.request_start(messages):]
    if not worked_on(span) or already_checked(span):
        return None
    problems = run(conversation.request_in(messages, cap=None), answer)
    if not problems:
        return None
    log.info('answer check: %d problem(s): %s', len(problems),
             '; '.join(problems))
    return problems_text(problems)


def run(request: str, answer: str) -> list[str]:
    """Ask the installed checker; empty without one or on any failure."""
    c = _checker
    if c is None:
        return []
    try:
        return list(c.check(request, answer))
    except Exception:                                    # noqa: BLE001
        log.exc('answer check failed')
        return []
