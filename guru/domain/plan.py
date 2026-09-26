"""The typed controller plan: schema, parsing and validation (pure).

A controller answers every user turn (and every mailbox delivery) with one
``plan`` tool call::

    {outcome: "answer" | "delegate",
     answer?: str,
     tasks?: [{goal: str, kind: <routing.KINDS>,
               complexity: <routing.COMPLEXITY>,
               files?: [str], role?: str, skill?: str}]}

``answer`` is the reply itself: no worker runs. ``delegate`` hands every
task to guru, which spawns them on routed workers, joins them and resumes
the controller with their results. ``answer`` is always a valid outcome: a
simple question has no task.

Validation is code, not prose. :func:`parse` turns the tool arguments into
a :class:`Plan` or a list of *hard* errors (neither outcome, mistyped
fields, ``delegate`` without a task); an ``answer`` is never an error, not
even without text (the loop then takes the round's own text, and only
when both are empty falls to its empty-reply re-prompt). :func:`validate`
checks the labels against the routing vocabularies (hard) and, for a
request that names several concerns from :data:`CONCERNS` joined by a
coordinator (:func:`coverage_concerns`), that every named concern is
covered by some task goal (soft: one re-ask naming the missing concern,
then the plan runs as given). The handler passes the *full* request text
(not the ledger's capped ``request`` column), so a long request's later
concerns count. :func:`evaluate` does both and returns a
:class:`Verdict`. Re-asks and re-prompts are counted in the conversation
itself (:func:`reasks_in`), so the loop and the handler agree without
shared state.

No I/O and no session access: the orchestrator and the turn loop call in
with what they know.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from guru.domain import routing

OUTCOMES = ('answer', 'delegate')

# The concern vocabulary and, per concern, the words (or phrases) that
# count as naming it — in the user's request and in a task goal alike.
# Noun forms and unambiguous review words only: everyday task verbs and
# nouns that name an artefact rather than a concern (``fix``, ``fast``,
# ``test``, ``comments``, ``structure``, ``failure``, ``auth``, ``secret``,
# ``retry``, ``timeout``) were false positives ("run the fast tests and fix
# the failure" is one task, not a three-concern review) and are out.
CONCERNS: dict[str, tuple[str, ...]] = {
    'correctness': ('correctness', 'correct', 'bug', 'bugs', 'logic',
                    'behaviour', 'behavior', 'wrong', 'broken',
                    'regression', 'regressions'),
    'security': ('security', 'secure', 'vulnerability', 'vulnerabilities',
                 'vulnerable', 'injection', 'authz', 'authn', 'secrets',
                 'traversal', 'xss', 'csrf', 'exploit', 'exploits'),
    'performance': ('performance', 'perf', 'slow', 'latency',
                    'throughput', 'speed', 'efficiency', 'efficient',
                    'memory usage', 'hot path'),
    'reliability': ('reliability', 'reliable', 'resilience', 'resilient',
                    'robust', 'robustness', 'crash', 'crashes',
                    'error handling', 'sre'),
    'design': ('design', 'architecture', 'architectural', 'modularity',
               'coupling', 'abstraction', 'maintainability',
               'readability'),
    'tests': ('tests', 'testing', 'test suite', 'coverage', 'pytest',
              'unittest'),
    'docs': ('docs', 'documentation', 'readme', 'docstring', 'docstrings'),
}

# Delegate rounds per user request: after this many ``delegate`` plans
# for one request (each resumed by a joined delivery) a further
# ``delegate`` is refused and the controller must answer from the results
# it has. Dogfood run 7730e6c1c39c: seven delegate rounds, seven workers,
# 577k tokens and no submit — every follow-up worker got a fresh sandbox
# copy, found the previous edits gone and the controller re-delegated.
MAX_DELEGATE_ROUNDS = 3

# Coverage is checked only when the request names at least this many
# distinct concerns ("several") AND joins two of them with a coordinator
# — "and", a comma, "&" — or names them in separate sentences
# (:func:`coverage_concerns`): a single-concern request never re-asks,
# nor does one that merely happens to contain two vocabulary words.
COVERAGE_MIN_CONCERNS = 2
_COORDINATOR_RE = re.compile(r'(?:\band\b|[,&;.?!\n])', re.IGNORECASE)

# --- the texts the loop and the handler exchange with the model ------------

# Tool result (or user message on the text path) of a rejected plan; the
# errors follow. ``conversation.is_nudge`` recognises the prefix so the
# re-ask is never mistaken for the user's request.
REASK_PREFIX = 'Plan not accepted: '
REASK_SUFFIX = (' Call plan again; outcome answer (with your text) is always'
                ' accepted.')
# Deterministic re-prompts of the turn contract: a text-only reply where a
# tool call was forced. One per turn; then the text is the answer.
REPROMPT_TEXT = ('Reply with a tool call: call the tool you need next, or'
                 ' final_answer with your complete answer.')
PLAN_REPROMPT_TEXT = ('Reply with one plan tool call: outcome answer with'
                      ' your text, or outcome delegate with the tasks.')
# Tool result of an accepted ``answer`` plan (the loop takes the text).
ANSWER_ACK = 'Answer delivered to the user.'
DELEGATED_PREFIX = 'Delegated: '
REFUSED_PREFIX = 'Could not delegate: '


@dataclass
class Task:
    """One delegated task as the controller wrote it (labels normalised
    to lower case, unknown values kept for :func:`validate` to name)."""
    goal: str
    kind: str = routing.DEFAULT_KIND
    complexity: str = routing.DEFAULT_COMPLEXITY
    files: list[str] = field(default_factory=list)
    role: str = ''
    skill: str = ''


@dataclass
class Plan:
    """A parsed plan; ``tasks`` is empty for ``answer``."""
    outcome: str
    answer: str = ''
    tasks: list[Task] = field(default_factory=list)


@dataclass
class Verdict:
    """:func:`evaluate`'s result: the plan when it parsed, the hard errors
    (malformed: cannot run) and the soft ones (concerns no task goal
    covers). ``ok`` means run it as given."""
    plan: Optional[Plan]
    errors: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.errors and not self.missing

    @property
    def messages(self) -> list[str]:
        """Every problem, hard first, as the re-ask lists them."""
        return list(self.errors) + [
            f"no task goal covers the concern '{c}' the request names"
            f" (say {' or '.join(CONCERNS[c][:3])})" for c in self.missing]


# --- JSON schema (the forced tool's input) -----------------------------------

def _labels(descriptions: dict) -> str:
    return '; '.join(f'{k} = {v}' for k, v in descriptions.items())


TASK_SCHEMA: dict = {
    'type': 'object',
    'properties': {
        'goal': {
            'type': 'string',
            'description': ('A clear, self-contained instruction for the'
                            ' worker (it has the file, code and web tools'
                            ' you lack); name the project path.')},
        'kind': {
            'type': 'string', 'enum': list(routing.KINDS),
            'description': 'What kind of task this is: '
                           + _labels(routing.KIND_DESCRIPTIONS)},
        'complexity': {
            'type': 'string', 'enum': list(routing.COMPLEXITY),
            'description': ('How hard it is (picks the model that runs it;'
                            ' use all three tiers): '
                            + _labels(routing.COMPLEXITY_DESCRIPTIONS))},
        'files': {
            'type': 'array', 'items': {'type': 'string'},
            'description': 'Files or directories the task concerns'},
        'role': {'type': 'string',
                 'description': 'Persona from the catalog, or omit'},
        'skill': {'type': 'string',
                  'description': 'Method from the catalog, or omit'},
    },
    'required': ['goal', 'kind', 'complexity'],
}

SCHEMA: dict = {
    'type': 'object',
    'properties': {
        'outcome': {
            'type': 'string', 'enum': list(OUTCOMES),
            'description': ('answer: your text is the reply and no worker'
                            ' runs. delegate: guru runs every task on a'
                            ' routed worker in parallel and resumes you'
                            ' with their results. You have no other'
                            ' tools: anything that needs a file, a'
                            ' command, a package, a test or the sandbox'
                            ' must be delegated; answer is for replies'
                            ' that need no work.')},
        'answer': {
            'type': 'string',
            'description': 'The reply to the user (outcome answer)'},
        'tasks': {
            'type': 'array', 'items': TASK_SCHEMA,
            'description': ('The tasks to run in parallel (outcome'
                            ' delegate; at least one)')},
    },
    'required': ['outcome'],
}


# --- parsing -----------------------------------------------------------------

def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ''


def _label(value: object) -> str:
    return str(value).strip().lower() if isinstance(value, str) else ''


def _listish(value: object) -> object:
    """A JSON-encoded list some models send as a string, decoded."""
    if isinstance(value, str) and value.lstrip().startswith('['):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _parse_task(index: int, raw: object, errors: list[str]
                ) -> Optional[Task]:
    tag = f'task {index}'
    if not isinstance(raw, dict):
        errors.append(f'{tag}: must be an object with goal, kind and'
                      ' complexity')
        return None
    goal = _text(raw.get('goal'))
    if not goal:
        errors.append(f'{tag}: goal must be a non-empty string')
    files_raw = _listish(raw.get('files'))
    files: list[str] = []
    if files_raw not in (None, ''):
        if not isinstance(files_raw, list) or not all(
                isinstance(f, str) for f in files_raw):
            errors.append(f'{tag}: files must be a list of strings')
        else:
            files = [f.strip() for f in files_raw if f.strip()]
    for key in ('role', 'skill'):
        if raw.get(key) not in (None, '') and not isinstance(raw[key], str):
            errors.append(f'{tag}: {key} must be a string')
    return Task(goal=goal,
                kind=_label(raw.get('kind')) or routing.DEFAULT_KIND,
                complexity=(_label(raw.get('complexity'))
                            or routing.DEFAULT_COMPLEXITY),
                files=files, role=_text(raw.get('role')),
                skill=_text(raw.get('skill')))


def parse(args: object) -> tuple[Optional[Plan], list[str]]:
    """The :class:`Plan` in the tool arguments ``args`` and the hard
    errors found; the plan is None when it cannot be built at all (no
    known outcome) and is never None otherwise, even with errors."""
    if not isinstance(args, dict):
        return None, ['plan must be an object with an outcome']
    outcome = _label(args.get('outcome'))
    if outcome not in OUTCOMES:
        return None, ["outcome must be 'answer' or 'delegate'"]
    errors: list[str] = []
    if outcome == 'answer':
        # Never rejected: without text the loop uses the round's own text.
        return Plan('answer', answer=_text(args.get('answer'))), errors
    raw_tasks = _listish(args.get('tasks'))
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return Plan('delegate'), ['outcome delegate needs at least one task']
    tasks = [_parse_task(i + 1, raw, errors)
             for i, raw in enumerate(raw_tasks)]
    return Plan('delegate', tasks=[t for t in tasks if t is not None]), errors


# --- validation --------------------------------------------------------------

def _word_re(term: str) -> re.Pattern:
    return re.compile(r'(?<![\w-])' + re.escape(term) + r'(?![\w-])',
                      re.IGNORECASE)


_CONCERN_RES = {c: [_word_re(t) for t in terms]
                for c, terms in CONCERNS.items()}


def concerns_in(text: str) -> list[str]:
    """The concerns ``text`` names, in vocabulary order."""
    return [c for c, res in _CONCERN_RES.items()
            if any(r.search(text) for r in res)]


def _mentions(text: str) -> list[tuple[int, int, str]]:
    """Every concern mention in ``text`` as ``(start, end, concern)``,
    in text order."""
    out: list[tuple[int, int, str]] = []
    for concern, res in _CONCERN_RES.items():
        for r in res:
            out.extend((m.start(), m.end(), concern)
                       for m in r.finditer(text))
    return sorted(out)


def coverage_concerns(request: str) -> list[str]:
    """The concerns ``request`` names *for coverage*: at least
    :data:`COVERAGE_MIN_CONCERNS` distinct ones, two of which are joined
    by a coordinator ("and", ",", "&") or sit in separate sentences —
    "review this for correctness and security" names two; "add docstrings
    to the security module" names two words but one concern list is not
    a panel, so it is empty. Returns them in vocabulary order, or ``[]``
    when the request does not qualify."""
    mentions = _mentions(request)
    named = {c for _, _, c in mentions}
    if len(named) < COVERAGE_MIN_CONCERNS:
        return []
    for (_, end, a), (start, _, b) in zip(mentions, mentions[1:]):
        if a != b and _COORDINATOR_RE.search(request[end:start]):
            return [c for c in CONCERNS if c in named]
    return []


def schema_errors(plan: Plan) -> list[str]:
    """Hard errors :func:`parse` cannot see: unknown labels."""
    errors: list[str] = []
    for i, t in enumerate(plan.tasks, 1):
        if t.kind not in routing.KINDS:
            errors.append(f"task {i}: unknown kind '{t.kind}'; one of "
                          + ', '.join(routing.KINDS))
        if t.complexity not in routing.COMPLEXITY:
            errors.append(f"task {i}: unknown complexity '{t.complexity}';"
                          ' one of ' + ', '.join(routing.COMPLEXITY))
    return errors


def missing_concerns(request: str, plan: Plan) -> list[str]:
    """The concerns ``request`` names for coverage
    (:func:`coverage_concerns`) that no task goal of a ``delegate`` plan
    covers. Empty for ``answer`` and for a request that does not qualify."""
    if plan.outcome != 'delegate' or not plan.tasks:
        return []
    named = coverage_concerns(request)
    if not named:
        return []
    covered: set[str] = set()
    for t in plan.tasks:
        covered.update(concerns_in(t.goal))
    return [c for c in named if c not in covered]


def validate(request: str, plan: Plan, followup: bool = False
             ) -> list[str]:
    """Every problem with ``plan`` for ``request``: unknown labels, then
    the concerns it leaves uncovered. ``followup`` (a mailbox turn: the
    request was already planned once) skips the coverage check. An
    ``answer`` plan is never rejected here."""
    errors = schema_errors(plan)
    if not followup:
        errors.extend(Verdict(plan, [], missing_concerns(request, plan))
                      .messages)
    return errors


def evaluate(request: str, args: object, followup: bool = False) -> Verdict:
    """Parse and validate the tool arguments ``args`` against ``request``
    (see :func:`validate` for ``followup``)."""
    plan, errors = parse(args)
    if plan is None:
        return Verdict(None, errors)
    errors.extend(schema_errors(plan))
    missing = [] if (errors or followup) else missing_concerns(request, plan)
    return Verdict(plan, errors, missing)


# --- the text path (adapters that cannot force a tool call) ------------------

_FENCE_RE = re.compile(r'```(?:json)?\s*(.*?)```', re.DOTALL | re.IGNORECASE)
_MAX_SCAN = 64


def _find_plan(text: str) -> Optional[tuple[dict, int, int]]:
    """The first JSON object in ``text`` with an ``outcome`` key and the
    span of ``text`` it occupies (the whole fence when it was fenced)."""
    if not text or '{' not in text:
        return None
    # (body to scan, span of the whole candidate in text, fenced?)
    candidates = [(m.group(1), m.start(), m.end(), True)
                  for m in _FENCE_RE.finditer(text)]
    candidates.append((text, 0, len(text), False))
    decoder = json.JSONDecoder()
    for body, lo, hi, fenced in candidates:
        starts = [i for i, ch in enumerate(body) if ch == '{'][:_MAX_SCAN]
        for i in starts:
            try:
                obj, end = decoder.raw_decode(body[i:])
            except ValueError:
                continue
            if isinstance(obj, dict) and 'outcome' in obj:
                return (obj, lo, hi) if fenced else (obj, i, i + end)
    return None


def from_text(text: str) -> Optional[dict]:
    """The plan object a model wrote as text: the first JSON object in
    ``text`` (fenced or bare) that has an ``outcome`` key; None when there
    is none. Used where a tool call cannot be forced (Ollama)."""
    found = _find_plan(text)
    return found[0] if found is not None else None


def prose_around(text: str) -> str:
    """``text`` without the plan object :func:`from_text` finds in it (the
    prose a model wrote around its JSON), stripped; ``text`` itself when
    there is no plan object. The answer of an ``answer`` plan whose
    ``answer`` field is empty on the text path."""
    found = _find_plan(text)
    if found is None:
        return text.strip()
    _, lo, hi = found
    return (text[:lo] + ' ' + text[hi:]).strip()


# --- texts -------------------------------------------------------------------

def reask_text(problems: list[str]) -> str:
    """The one re-ask for a rejected plan."""
    return REASK_PREFIX + '; '.join(problems) + '.' + REASK_SUFFIX


def is_reask(text: str) -> bool:
    return text.startswith(REASK_PREFIX)


def reask_problems(text: str) -> str:
    """The problem list a re-ask text carries (its prefix and suffix
    stripped)."""
    body = text[len(REASK_PREFIX):] if is_reask(text) else text
    if body.endswith(REASK_SUFFIX):
        body = body[:-len(REASK_SUFFIX)]
    return body.rstrip('.').strip()


def is_reprompt(text: str) -> bool:
    """True for a loop-injected re-ask or re-prompt (never the user's)."""
    return text in (REPROMPT_TEXT, PLAN_REPROMPT_TEXT) or is_reask(text)


def reasks_in(messages: list) -> int:
    """How many plan re-asks ``messages`` (this turn's slice) carry: the
    ``plan`` tool results and the user messages that start with
    :data:`REASK_PREFIX`."""
    n = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role')
        if role == 'tool' and m.get('tool_name') != 'plan':
            continue
        if role in ('tool', 'user') and is_reask(m.get('content') or ''):
            n += 1
    return n


def delegated_text(titles: list[str], tasks: list[Task]) -> str:
    """Tool result of an accepted ``delegate``: what runs where, and that
    the turn ends here."""
    parts = [f'{title} ({t.kind}/{t.complexity})'
             for title, t in zip(titles, tasks)]
    return (DELEGATED_PREFIX + ', '.join(parts) + '. This turn ends here;'
            ' you will be resumed with a [joined results] message and'
            ' answer it with plan (outcome answer, synthesising them).')


def delegations_in(messages: list) -> int:
    """How many ``delegate`` plans ran for the request ``messages`` (the
    slice from :func:`conversation.request_start`) carries: the ``plan``
    tool results that start with :data:`DELEGATED_PREFIX` (the tool
    path) or the mailbox deliveries that resumed the controller (the text
    path leaves no tool result) — whichever is more, they count the same
    rounds."""
    from guru.domain import conversation
    delegated = deliveries = 0
    for m in messages:
        role = conversation.msg_role(m)
        text = conversation.msg_content(m) or ''
        if role == 'tool':
            if isinstance(m, dict) and m.get('tool_name') == 'plan' \
                    and text.startswith(DELEGATED_PREFIX):
                delegated += 1
        elif role == 'user' and conversation.is_mailbox(text.strip()):
            deliveries += 1
    return max(delegated, deliveries)


def delegate_cap_text(rounds: int) -> str:
    """Tool result of a ``delegate`` refused by :data:`MAX_DELEGATE_ROUNDS`:
    a refusal (the loop keeps the turn going) that says what to do."""
    return (REFUSED_PREFIX + f'this request has already been delegated'
            f' {rounds} times (the cap). Answer the user from the results'
            ' you have: call plan with outcome answer, saying what was done'
            ' and what remains.')


def refused_text(reasons: list[str]) -> str:
    """Tool result when routing refused every task."""
    why = '; '.join(reasons) or 'no model is allowed to run these tasks'
    return (REFUSED_PREFIX + why + '. Answer the user yourself: call plan'
            ' with outcome answer.')


def fallback_text(text: str, args: object, problems: list[str]) -> str:
    """The plain-text answer when the second plan is malformed too: the
    model's own text if it wrote any, else its ``answer`` field, else a
    one-paragraph account of the failed plan (task goals included) so the
    user sees what was attempted."""
    if text.strip():
        return text.strip()
    if isinstance(args, dict):
        answer = _text(args.get('answer'))
        if answer:
            return answer
    goals: list[str] = []
    if isinstance(args, dict):
        tasks = _listish(args.get('tasks'))
        if isinstance(tasks, list):
            goals = [_text(t.get('goal')) for t in tasks
                     if isinstance(t, dict) and _text(t.get('goal'))]
    out = ('(guru could not run the controller\'s plan: '
           + '; '.join(problems) + '.)')
    if goals:
        out += '\nProposed tasks:\n' + '\n'.join(f'- {g}' for g in goals)
    return out


def final_text(args: object) -> str:
    """The answer in a ``final_answer`` call: its ``text``, else the first
    string argument (a model that named the field differently)."""
    if not isinstance(args, dict):
        return ''
    text = args.get('text')
    if isinstance(text, str):
        return text.strip()
    return next((v.strip() for v in args.values() if isinstance(v, str)),
                '')


# --- task text ---------------------------------------------------------------

def brief_hook(task_text: str) -> str:
    """Identity: the task text stays the plain goal (it is the ledger's
    ``task`` column and the join titles). The project brief slice reaches
    the worker through its system context instead — see
    ``Orchestrator._brief_block``."""
    return task_text


def task_text(task: Task) -> str:
    """The worker's task text: the goal, the files it names, and the
    brief hook."""
    text = task.goal
    if task.files:
        text += '\nFiles: ' + ', '.join(task.files)
    return brief_hook(text)
