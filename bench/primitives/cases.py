"""Labelled cases for the primitives probe.

Each task is a dict: ``name``, ``kind`` (choice / score / noul),
``instructions``, ``options`` (ordered key -> description; for noul the keys
are ``yes``/``no``, for score the keys are ordered levels) and ``cases`` (each
``state`` text + ``expected`` label; ``expected: None`` skips scoring).
"""
from guru.domain.tools import TOOL_REGISTRY

YES_NO = {'yes': 'Yes', 'no': 'No'}


def _reply(text: str) -> str:
    return 'Reply:\n' + text


def _req(text: str) -> str:
    return 'Request: ' + text


STALL = {
    'name': 'stall', 'kind': 'noul',
    'instructions': (
        'An AI assistant produced the reply below at the end of its turn. Is'
        ' the reply a stalled preamble - it announces or promises an action'
        ' (reading, checking, running something) but does not actually'
        ' deliver an answer or result? Substantive answers are NOT preambles'
        ' even if they contain phrases like "let me" or "I\'ll".'),
    'options': YES_NO,
    'cases': [
        {'state': _reply("Let me read the remaining files and then I'll get"
                         " back to you with a summary:"), 'expected': True},
        {'state': _reply("I'll start by listing the directory."),
         'expected': True},
        {'state': _reply('First, I need to look at the config file.'),
         'expected': True},
        {'state': _reply('Checking the tests now...'), 'expected': True},
        {'state': _reply('Next, I will run the tests to confirm.'),
         'expected': True},
        {'state': _reply('Okay.'), 'expected': True},
        {'state': _reply('The function returns None when the list is empty'
                         ' because the loop never executes.'),
         'expected': False},
        {'state': _reply("I'll be honest: the code is fine. The loop is O(n)"
                         ' and the tests cover the edge cases.'),
         'expected': False},
        {'state': _reply('Let me know if you need anything else. Summary: 3'
                         ' files changed, tests pass.'), 'expected': False},
        {'state': _reply('Going to sleep mode is handled in power.py line'
                         ' 40, which calls suspend().'), 'expected': False},
        {'state': _reply("Here's what I found: the bug is in parse_range"
                         ' where end is exclusive.'), 'expected': False},
        {'state': _reply("The next step for you would be to run the tests;"
                         " I've finished my changes."), 'expected': False},
    ],
}

TIER = {
    'name': 'tier', 'kind': 'choice',
    'instructions': (
        'A user sent the request below to a coding assistant. Which size of'
        ' language model is the SMALLEST that would give a good enough'
        ' answer?'),
    'options': {
        'small': ('Small (1-4B): greetings, trivial facts, simple'
                  ' formatting or conversions, one-line lookups'),
        'medium': ('Medium (8-14B): explain or write a short snippet,'
                   ' summarise a single document, a regex, a small fix'),
        'large': ('Large (30B+ / frontier): multi-file refactors,'
                  ' architecture, subtle bugs or races, security review,'
                  ' rigorous reasoning'),
    },
    'cases': [
        {'state': _req("hi, what's the time zone of Amsterdam?"),
         'expected': 'small'},
        {'state': _req('convert this list to JSON: apples, pears, bananas'),
         'expected': 'small'},
        {'state': _req("what does 'ls -la' do?"), 'expected': 'small'},
        {'state': _req("translate 'good morning' to Dutch"),
         'expected': 'small'},
        {'state': _req('Explain what this Python function does:'
                       ' def f(x): return [i*i for i in x if i%2]'),
         'expected': 'medium'},
        {'state': _req('Summarise the README of this repo in 3 bullets'),
         'expected': 'medium'},
        {'state': _req('Write a regex that matches ISO-8601 dates'),
         'expected': 'medium'},
        {'state': _req('Fix the flake8 warnings in this 20-line function'),
         'expected': 'medium'},
        {'state': _req('Refactor the adapter layer so tool-calling is shared'
                       ' across Ollama, Anthropic and LiteLLM without'
                       ' breaking the tests'), 'expected': 'large'},
        {'state': _req('Review this repository for security vulnerabilities:'
                       ' injection, path traversal, secrets handling'),
         'expected': 'large'},
        {'state': _req("There's a race condition somewhere between the"
                       " orchestrator's join barrier and the worker threads;"
                       ' find it'), 'expected': 'large'},
        {'state': _req('Design a multi-tenant auth architecture with OAuth,'
                       ' RBAC and audit logging for a Kubernetes platform'),
         'expected': 'large'},
    ],
}

TOOL = {
    'name': 'tool', 'kind': 'choice',
    'instructions': ('Which ONE tool should the assistant call first to'
                     ' handle the request below?'),
    'options': {name: f"{name}: {info['description']}"
                for name, info in TOOL_REGISTRY.items()},
    'cases': [
        {'state': _req("what's the weather in Utrecht right now"),
         'expected': 'web_search'},
        {'state': _req('read https://example.com/changelog and tell me what'
                       ' changed'), 'expected': 'web_fetch'},
        {'state': _req("what's the latest version of kubernetes/kubernetes"),
         'expected': 'fetch_github_releases'},
        {'state': _req('what files are in the current directory'),
         'expected': 'list_dir'},
        {'state': _req('show me the structure of this project'),
         'expected': 'list_tree'},
        {'state': _req('open guru/config.py'), 'expected': 'read_file'},
        {'state': _req('where is compact_messages defined'),
         'expected': 'search_code'},
        {'state': _req('create a new file notes.md with a todo list'),
         'expected': 'write_file'},
        {'state': _req('change the timeout in fetch from 15 to 30 seconds'),
         'expected': 'edit_file'},
        {'state': _req('remove the temporary file tmp.out'),
         'expected': 'delete_file'},
        {'state': _req('find all usages of session.messages'),
         'expected': 'search_code'},
        {'state': _req('who won the election yesterday'),
         'expected': 'web_search'},
    ],
}


def _panel(question: str, name: str, cases: list) -> dict:
    return {
        'name': name, 'kind': 'noul',
        'instructions': ('A code-review task is described below. ' + question),
        'options': YES_NO,
        'cases': [{'state': 'Task: ' + t, 'expected': e} for t, e in cases],
    }


_PANEL_TASKS = [
    ('review the new login endpoint that stores session tokens in cookies',
     True, False, False),
    ('review the retry and timeout logic of the background job runner',
     False, False, True),
    ('fix the typo in the README', False, False, False),
    ('review the file upload handler that writes user-provided paths to'
     ' disk', True, False, False),
    ('review the Helm chart changes for the production rollout and the'
     ' alerting rules', False, False, True),
    ('review the proposal to split the monolith into three services with'
     ' a shared event bus', False, True, None),
    ('rename a local variable in one function', False, False, False),
    ('review the new module boundaries and dependency direction between'
     ' the domain and adapter layers', False, True, False),
]

PANEL_SEC = _panel(
    'Does it need a SECURITY specialist (injection, authz, secrets, path'
    ' traversal, untrusted input)?', 'panel_sec',
    [(t, s) for t, s, _, _ in _PANEL_TASKS])
PANEL_ARCH = _panel(
    'Does it need a software ARCHITECT (system design, module boundaries,'
    ' service decomposition)?', 'panel_arch',
    [(t, a) for t, _, a, _ in _PANEL_TASKS])
PANEL_SRE = _panel(
    'Does it need an SRE / reliability specialist (deployments, retries,'
    ' timeouts, alerting, operations)?', 'panel_sre',
    [(t, r) for t, _, _, r in _PANEL_TASKS])

JUDGE = {
    'name': 'judge', 'kind': 'score',
    'instructions': ('Rate the quality of the answer to the question'
                     ' below.'),
    'options': {
        'wrong': 'Wrong or unhelpful',
        'partial': 'Partially correct or incomplete',
        'good': 'Correct and complete',
    },
    'cases': [
        {'state': 'Question: What does HTTP status 404 mean?\nAnswer: The'
                  ' requested resource could not be found on the server.',
         'expected': 2},
        {'state': 'Question: What does HTTP status 404 mean?\nAnswer: It'
                  ' means the server had an internal error.', 'expected': 0},
        {'state': 'Question: How do I list files in a directory in Python?'
                  '\nAnswer: Use os.listdir(path) or Path(path).iterdir().',
         'expected': 2},
        {'state': 'Question: How do I list files in a directory in Python?'
                  '\nAnswer: You can use a library.', 'expected': 1},
        {'state': 'Question: What is 17*3?\nAnswer: 51', 'expected': 2},
        {'state': 'Question: What is 17*3?\nAnswer: 54', 'expected': 0},
        {'state': 'Question: Explain what git rebase does.\nAnswer: It'
                  ' rewrites commits so they appear on top of another base'
                  ' commit, replaying your changes.', 'expected': 2},
        {'state': 'Question: Explain what git rebase does.\nAnswer: It moves'
                  " commits around, but I'm not sure how.", 'expected': 1},
        {'state': 'Question: Which port does HTTPS use by default?\nAnswer:'
                  ' 443', 'expected': 2},
        {'state': 'Question: Which port does HTTPS use by default?\nAnswer:'
                  ' 80', 'expected': 0},
    ],
}

TASKS = [STALL, TIER, TOOL, PANEL_SEC, PANEL_ARCH, PANEL_SRE, JUDGE]
