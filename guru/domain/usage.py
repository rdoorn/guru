"""Usage tracking rules: what a call's topic is, the short label a cheap
model gives it, the dashboard's time ranges, and the seams the store and
the labeler plug into.

A *topic* belongs to one user request. When a user turn starts on an
agent the user talks to (not a sub-agent running a task, not a mailbox
delivery), :func:`begin_topic` makes that turn's id the session's
``topic_id``; every call of that request — the synthesis rounds of a
lead and every worker it delegates to — carries it, so the dashboard
groups all the work one request caused under one topic. The topic row
holds the request text (redacted, then cut to ``TOPIC_CHARS``) at once,
and a 3-6 word label once the installed :class:`TopicLabeler` has produced
one on a background thread (never blocking the turn).

The store (``guru.repositories.usage_sqlite``) and the labeler
(``guru.judges.topic``) are installed by the CLI; without them this module
records nothing extra and labels nothing.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from guru import config, log, session
from guru.domain import ledger, policy

TOPIC_CHARS = 200          # the request text kept as a topic's fallback
LABEL_CHARS = 60           # a generated label, at most
LABEL_WORDS = 8            # ... and at most this many words
TOPICS_STREAM = 'topics'

RANGES = ('today', '7d', '30d', 'all')
GROUPS = ('model', 'project', 'topic')
SOURCES = ('all', 'cli', 'eval')


# --- topic text and label ----------------------------------------------------

# Token-shaped strings redacted from persisted topic and task text on top
# of the project's secret scanner (which is bound only with a routing table
# and does not know every provider's key format): provider keys, key=value
# credentials, and long opaque base64/hex runs.
_TOKEN_RES = (
    re.compile(r'\b(?:sk|pk|rk)-[A-Za-z0-9_-]{8,}'),
    re.compile(r'\b(?:ghp|gho|ghu|ghs|github_pat|glpat)_[A-Za-z0-9_]{16,}'),
    re.compile(r'\bxox[abposr]-[A-Za-z0-9-]{10,}'),
    re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(r'(?i)\b(?:api[_-]?key|token|secret|passw(?:or)?d|pwd)'
               r'\s*[:=]\s*[^\s,;]+'),
    re.compile(r'\b[A-Za-z0-9+_=-]{40,}\b'),     # no '/': paths survive
)
REDACTED = '[REDACTED]'


def topic_text(text: str) -> str:
    """``text`` as it may be persisted: secrets redacted first — the
    project scanner's findings, then token-shaped strings — so a secret
    split by the cut cannot slip past; whitespace collapsed; then cut to
    ``TOPIC_CHARS``."""
    clean = policy.redact(text or '', policy.scan(text or ''))
    for pattern in _TOKEN_RES:
        clean = pattern.sub(REDACTED, clean)
    clean = ' '.join(clean.split())
    if len(clean) > TOPIC_CHARS:
        clean = clean[:TOPIC_CHARS - 1].rstrip() + '…'
    return clean


LABEL_INSTRUCTIONS = (
    'Give a short topic label (3 to 6 words) for this request to a coding'
    ' assistant, naming what the user is working on (e.g. "usage dashboard'
    ' store", "fix login redirect"). Reply with one JSON object:'
    ' {"topic": "..."}.')


def label_prompt(request: str) -> str:
    """The labeler's prompt for an (already redacted) request."""
    return f'{LABEL_INSTRUCTIONS}\n\nREQUEST:\n{request}'


_OBJECT_RE = re.compile(r'\{.*\}', re.DOTALL)


def parse_label(text: str) -> Optional[str]:
    """The label in a labeler reply, or None when it is not the expected
    JSON object or the label is empty or too long to be a label."""
    match = _OBJECT_RE.search(text or '')
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    label = data.get('topic') if isinstance(data, dict) else None
    if not isinstance(label, str):
        return None
    label = ' '.join(label.split()).strip(' .,"\'')
    if not label or len(label.split()) > LABEL_WORDS:
        return None
    return label[:LABEL_CHARS]


# --- seams -------------------------------------------------------------------

class TopicLabeler(Protocol):
    """Produces a short label for a request (an endpoint calls a model)."""

    def label(self, request: str) -> Optional[str]:
        """The label, or None when no model may or could label it."""


class UsageQueries(Protocol):
    """Read side of the usage store, for the dashboard (``since`` is an
    ISO-8601 UTC timestamp or None for all time; ``source`` one of
    ``SOURCES``)."""

    def totals(self, since: Optional[str], source: str) -> dict: ...

    def daily(self, since: Optional[str], source: str) -> list: ...

    def groups(self, by: str, since: Optional[str], source: str,
               limit: int) -> list: ...

    def calls(self, since: Optional[str], source: str, limit: int,
              offset: int) -> list: ...


_labeler: Optional[TopicLabeler] = None


def set_labeler(labeler: Optional[TopicLabeler]) -> None:
    """Install the topic labeler (None: topics keep the request text)."""
    global _labeler
    _labeler = labeler


def labeler() -> Optional[TopicLabeler]:
    """The installed topic labeler, or None."""
    return _labeler


# --- topic lifecycle ---------------------------------------------------------

def begin_topic(request: str) -> None:
    """Start the topic of a new user request on the bound session: its
    ``topic_id`` becomes the current turn id, the ``topics`` stream gets
    the request text, and the label is produced in the background (with
    the session bound, so its call is accounted to this turn). Never
    raises."""
    try:
        topic_id = session.turn_id
        if not topic_id:
            return
        session.topic_id = topic_id
        text = topic_text(request)
        ledger.submit(TOPICS_STREAM, {**ledger.base_row(),
                                      'topic_id': topic_id,
                                      'request': text, 'label': ''})
        if _labeler is None or not config.TOPIC_LABELS or not text:
            return
        state = _label_state(topic_id)
        threading.Thread(target=_label_in, args=(state, topic_id, text),
                         name='guru-topic-label', daemon=True).start()
    except Exception:                                    # noqa: BLE001
        log.exc('begin_topic failed')


def _label_state(topic_id: str) -> session.SessionState:
    """A session of its own for the label call: the turn's identity (so
    the call is filed under this agent, turn and topic) and its model
    context, but fresh counters, so the background call never races the
    turn's cost and call counters."""
    cur = session.current()
    state = session.SessionState()
    for name in ('agent_id', 'task_id', 'turn_id', 'adapter', 'model'):
        setattr(state, name, getattr(cur, name))
    state.topic_id = topic_id
    return state


def _label_in(state: session.SessionState, topic_id: str, text: str) -> None:
    token = session.use(state)
    try:
        _label(topic_id, text)
    finally:
        session.reset(token)


def _label(topic_id: str, text: str) -> None:
    labeler_ = _labeler
    if labeler_ is None:
        return
    try:
        label = labeler_.label(text)
    except Exception:                                    # noqa: BLE001
        log.exc('topic label failed')
        return
    if label:
        ledger.submit(TOPICS_STREAM, {**ledger.base_row(),
                                      'topic_id': topic_id, 'label': label})


# --- time ranges -------------------------------------------------------------

def since(range_name: str, now: Optional[datetime] = None) -> Optional[str]:
    """The UTC ISO timestamp a dashboard range starts at: ``today`` from
    local midnight, ``7d``/``30d`` that many days back, ``all`` None.
    Unknown names raise ``ValueError``."""
    if range_name not in RANGES:
        raise ValueError(f'range must be one of {", ".join(RANGES)}')
    if range_name == 'all':
        return None
    now = now or datetime.now().astimezone()
    if range_name == 'today':
        # Local midnight with the offset in force at midnight (correct on
        # a daylight-saving change day, unlike now's offset).
        local = now.astimezone()
        start = datetime.combine(local.date(), datetime.min.time()
                                 ).astimezone()
    else:
        start = now - timedelta(days=int(range_name[:-1]))
    return start.astimezone(timezone.utc).isoformat(timespec='milliseconds')
