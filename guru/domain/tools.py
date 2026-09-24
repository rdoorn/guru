"""Tool directory: the tools themselves, discovery, gating, and execution.

Tool execution is provider-agnostic. Adapters call ``execute_tool`` when a
model requests a tool; this module handles the domain allow-list gate,
``search_tools`` activation, and running the tool, returning a result string.
"""
import time

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from guru import config, log, session, skills, ui
from guru.domain import (code, decisions, files, gitread, ledger, patch,
                         policy, procs, quality, routing, toolpolicy)

# The tools a controller (``[routing] controller = true``) keeps: it
# coordinates and never executes (design doc §2).
CONTROLLER_TOOLS = frozenset(('spawn', 'check', 'join', 'use_skill'))
# The project tool policy seam lives in guru.domain.toolpolicy (so the
# audited verbs can read it without importing this module); re-exported
# here so tools.set_policy / is_enabled / active_policy keep working.
ALWAYS_ON_TOOLS = toolpolicy.ALWAYS_ON_TOOLS
ToolsPolicy = toolpolicy.ToolsPolicy
set_policy = toolpolicy.set_policy
active_policy = toolpolicy.active_policy
is_enabled = toolpolicy.is_enabled

_STOP_WORDS = {
    'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'for',
    'of', 'and', 'or', 'but', 'what', 'how', 'when', 'where', 'who',
    'which', 'that', 'this', 'are', 'was', 'were', 'be', 'been',
    'being', 'do', 'does', 'did', 'me', 'my', 'you', 'your', 'its',
}


# --- project tool policy (.guru/tools.toml) ----------------------------------

def _advertised() -> list:
    """The registry tool names the project policy lets run, in registry
    order. Discovery, pre-activation and the specs sent to the model all
    draw from this list, so a disabled tool is never described to the
    model (and ``execute_tool`` refuses it anyway if called by name)."""
    return [name for name in TOOL_REGISTRY if is_enabled(name)]


# Pluggable domain approval — overridable by the TUI so it doesn't call the
# blocking input() inside a full-screen app. Signature: (domain) -> bool.
_domain_asker = None


def set_domain_asker(fn) -> None:
    """Install a custom domain-approval prompt (used by the TUI)."""
    global _domain_asker
    _domain_asker = fn


# Pluggable sub-agent spawner — installed by the TUI. Signature:
# (task, role, skill, kind, complexity) -> str. Absent in the REPL, where
# there are no viewports to delegate into.
_spawn_handler = None


def set_spawn_handler(fn) -> None:
    """Install the sub-agent spawner (used by the TUI)."""
    global _spawn_handler
    _spawn_handler = fn


def spawn(task: str, role: str = '', skill: str = '', kind: str = 'other',
          complexity: str = 'standard') -> str:
    """
    Delegate a self-contained task to a new sub-agent that runs in parallel.

    Label the task: ``kind`` is one of debug, build, refactor, review,
    explain, docs, ops, other and ``complexity`` one of trivial, standard,
    hard -- the labels pick the model that runs it. Optionally give it a
    ``role`` (persona) and/or ``skill`` (method) from the catalog -- e.g.
    role='security-engineer', skill='code-review'. The sub-agent works in its
    own viewport and context and returns only its conclusion; it cannot spawn
    further agents.
    """
    if _spawn_handler is None:
        return (
            "Spawning sub-agents is not available in this mode."
            " Handle this task yourself instead.")
    return _spawn_handler(task, role, skill, kind, complexity)


_SPAWN_SPEC = {
    'name': 'spawn',
    'description': (
        'Delegate a self-contained task to a new sub-agent that runs in'
        ' parallel in its own viewport and context. Label the task with kind'
        ' and complexity (they pick the model that runs it). Optionally set'
        ' role (persona) and skill (method) from the catalog. The sub-agent'
        ' cannot spawn further agents. Returns immediately -- its result is'
        ' delivered back to you automatically when it finishes.'),
    'parameters': {
        'task': 'A clear, self-contained instruction for the sub-agent',
        'kind': ('What kind of task this is, one of: '
                 + ', '.join(routing.KINDS) + ' (default other)'),
        'complexity': ('How hard the task is, one of: '
                       + ', '.join(routing.COMPLEXITY)
                       + ' (default standard)'),
        'role': 'Optional persona name from the catalog (or empty)',
        'skill': 'Optional method name from the catalog (or empty)',
    },
    'optional': ['kind', 'complexity', 'role', 'skill'],
}


# Pluggable non-blocking sub-agent status query — installed by the TUI.
_check_handler = None
# Pluggable non-blocking join/barrier — installed by the TUI.
_join_handler = None


def set_check_handler(fn) -> None:
    """Install the sub-agent status query (used by the TUI)."""
    global _check_handler
    _check_handler = fn


def set_join_handler(fn) -> None:
    """Install the sub-agent join/barrier (used by the TUI)."""
    global _join_handler
    _join_handler = fn


def check(target: str) -> str:
    """
    Check the status and any finished results of your sub-agents, without
    blocking. Pass a sub-agent name (e.g. "agent2") or "all". Returns
    immediately, so you can keep working or delegate more while others run.
    Do not poll in a loop: when everything is still running, call join.
    """
    if _check_handler is None:
        return "Checking sub-agents is not available in --classic (REPL) mode."
    return _check_handler(target)


def join(targets: str) -> str:
    """
    Ask to be resumed automatically once the named sub-agents all finish,
    then end your turn. Non-blocking: you stay free to take other work. Pass
    one or more sub-agent names separated by spaces or commas. Their combined
    results are delivered to you when the whole group is done.
    """
    if _join_handler is None:
        return "Joining sub-agents is not available in --classic (REPL) mode."
    return _join_handler(targets)


_CHECK_SPEC = {
    'name': 'check',
    'description': (
        'Check the status and any finished results of your sub-agents without'
        ' blocking. Pass a sub-agent name (e.g. "agent2") or "all". Returns'
        ' immediately. Do not poll in a loop: when everything is still'
        ' running, call join instead.'
    ),
    'parameters': {
        'target': 'A sub-agent name, or "all" for every one',
    },
}

_JOIN_SPEC = {
    'name': 'join',
    'description': (
        'Ask to be automatically resumed once the named sub-agents all'
        ' finish, then end your turn. Non-blocking — you stay free to take'
        ' other work meanwhile. Pass one or more sub-agent names separated by'
        ' spaces or commas; their combined results are delivered to you when'
        ' the group completes.'
    ),
    'parameters': {
        'targets': 'Sub-agent names, space- or comma-separated',
    },
}


def use_skill(name: str) -> str:
    """
    Adopt a methodology (skill) from the catalog for the current task, e.g.
    'code-review' or 'systematic-debugging'. It stays active until you switch
    skills. Roles (personas) are set with spawn(role=...) or the user's /role.
    """
    entry = skills.get(name)
    if entry is None or entry.kind != skills.SKILL:
        avail = ', '.join(
            skills.names(skills.REGISTRY, skills.SKILL)) or 'none'
        return f"No skill '{name}'. Available skills: {avail}."
    session.active_skill = name
    return f"Skill '{name}' is now active for this agent."


_USE_SKILL_SPEC = {
    'name': 'use_skill',
    'description': (
        'Adopt a methodology (skill) from the catalog for the current task'
        ' (e.g. code-review, systematic-debugging). Stays active until'
        ' switched. Use the catalog names shown in your context.'),
    'parameters': {'name': 'A skill name from the catalog'},
}


def _ask_domain(question: str) -> bool:
    """Default terminal approval prompt. Only an explicit yes (Enter, y, or
    yes) approves; any other input, or an error, denies."""
    try:
        answer = input(f"{question}\n[Y/n] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return answer in ('', 'y', 'yes')


def ensure_domain_allowed(domain: str) -> bool:
    """Return True if the domain is allowed, prompting per the access mode.

    Web fetches/searches are reads, so this uses the read domain list. auto
    mode approves silently (unless ``config.AUTO_GRANT`` is off, then it
    asks); ask prompts; approvals are persisted.
    """
    domain = domain.lower()
    if domain in config.ALLOWED_DOMAINS:
        return True
    if config.MODE == config.MODE_AUTO and config.AUTO_GRANT:
        config.ALLOWED_DOMAINS.add(domain)
        config.persist_domain(domain)
        return True
    asker = _domain_asker or _ask_domain
    if asker(f"Allow web access to '{domain}'?"):
        config.ALLOWED_DOMAINS.add(domain)
        config.persist_domain(domain)
        ui.console.print(f"[green]Allowed[/green] {domain}.")
        return True
    ui.console.print(f"[red]Denied[/red] {domain}.")
    return False


def _relevance_score(query: str, result: dict) -> int:
    """Score a search result by keyword overlap with the query."""
    keywords = {
        w.lower() for w in query.split()
        if w.lower() not in _STOP_WORDS and len(w) > 2
    }
    text = (
        (result.get('title') or '') + ' ' + (result.get('body') or '')
    ).lower()
    return sum(1 for kw in keywords if kw in text)


def web_search(query: str) -> str:
    """
    Search the internet for current information.
    ONLY use this for: current events, recent news, live prices/weather,
    information that changes over time, or topics likely after your training
    cutoff. Do NOT use for math, logic, coding, or stable well-known facts.
    """
    if not ensure_domain_allowed(config.SEARCH_BACKEND_DOMAIN):
        return (
            f"Access to the search engine '{config.SEARCH_BACKEND_DOMAIN}'"
            " was denied by the user. The search was not performed."
        )
    results = list(DDGS().text(query, max_results=10))

    scored = sorted(
        results,
        key=lambda r: _relevance_score(query, r),
        reverse=True,
    )
    relevant = [r for r in scored if _relevance_score(query, r) > 0][:5]
    if not relevant:
        relevant = scored[:3]

    output = []
    for i, result in enumerate(relevant, 1):
        entry = (
            f"{i}. {result.get('title')}\n"
            f"URL: {result.get('href')}\n"
            f"Summary: {result.get('body')}\n"
        )
        output.append(entry)
        ui.console.print(entry, style="dim yellow", markup=False)

    return "\n".join(output)


def web_fetch(url: str) -> str:
    """
    Fetch and read the text content of a webpage.
    Use this after web_search when you need more information from a result.
    """
    domain = config.domain_of(url)
    if not ensure_domain_allowed(domain):
        return f"Access to domain '{domain}' was denied by the user."
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup([
        "script", "style", "nav", "footer", "header", "aside",
    ]):
        element.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Shadow-mode injection screen: a judge scores the fetched text; the
    # page is returned unchanged (see guru.domain.decisions).
    decisions.shadow('injection', [decisions.injection_question(text, url)])
    # Don't dump enormous pages into the model.
    return text[:15000]


def fetch_github_releases(repo: str) -> str:
    """
    Fetch the latest release for a GitHub repository.
    Use this for software version questions when the project is on
    GitHub. Pass the repo as 'owner/repo', e.g. 'kubernetes/kubernetes'.
    Do NOT use web_search for version questions about GitHub projects —
    use this tool directly.
    """
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Mozilla/5.0",
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    notes = (data.get('body') or '')[:500]
    return (
        f"Repository: {repo}\n"
        f"Latest release: {data.get('tag_name')}\n"
        f"Name: {data.get('name')}\n"
        f"Published: {data.get('published_at')}\n"
        f"URL: {data.get('html_url')}\n"
        f"Release notes: {notes}\n"
    )


# Each entry: fn (callable), description, tags, parameters.
# search_tools matches against all of these fields.
TOOL_REGISTRY: dict = {
    "web_search": {
        "fn": web_search,
        "description": (
            "Search the web and return ranked results with titles, URLs, and "
            "text snippets. Use for current events, live data, news, prices, "
            "weather, or any information that changes over time."
        ),
        "tags": [
            "search", "web", "internet", "find", "lookup", "query",
            "google", "news", "current", "live", "weather", "prices",
            "events", "information", "recent", "today", "latest",
        ],
        "parameters": {
            "query": "Search terms to look up on the web",
        },
        "retain": "summarize",
    },
    "web_fetch": {
        "fn": web_fetch,
        "description": (
            "Fetch and read the full text content of any webpage given its"
            " URL. Use to read a specific page, follow a link, retrieve an"
            " endpoint response, or read content from a known URL."
        ),
        "tags": [
            "fetch", "read", "url", "webpage", "page", "html", "content",
            "download", "retrieve", "open", "link", "site", "http", "https",
            "get", "load", "visit", "endpoint", "curl", "request",
        ],
        "parameters": {
            "url": "Full URL to fetch (e.g. https://example.com/page)",
        },
        "retain": "summarize",
    },
    "fetch_github_releases": {
        "fn": fetch_github_releases,
        "description": (
            "Get the latest release version, tag, and release notes for any"
            " GitHub repository. Use for software version questions on"
            " projects hosted on GitHub."
        ),
        "tags": [
            "github", "release", "version", "latest", "software", "package",
            "tag", "changelog", "update", "repo", "repository", "library",
            "binary", "open source", "project",
        ],
        "parameters": {
            "repo": (
                "Repository in 'owner/repo' format"
                " (e.g. 'kubernetes/kubernetes')"
            ),
        },
    },
    "list_dir": {
        "fn": files.list_dir,
        "description": (
            "List the immediate contents of a directory (non-recursive) with"
            " octal permissions and sizes. Restricted to allowed directories"
            " (the working directory by default; asks once for others)."
        ),
        "tags": [
            "list", "ls", "dir", "directory", "folder", "files", "contents",
            "browse", "filesystem", "path", "permissions", "stat", "local",
        ],
        "parameters": {
            "path": "Directory to list (default: current directory)",
        },
        "optional": ["path"],
    },
    "list_tree": {
        "fn": files.list_tree,
        "description": (
            "List a directory tree recursively with octal permissions and"
            " sizes. Noise dirs (.git, node_modules, __pycache__, …) are"
            " shown but not expanded — pass a noise dir's full path to look"
            " inside. Restricted to allowed directories."
        ),
        "tags": [
            "tree", "recursive", "list", "directory", "files", "find", "walk",
            "structure", "filesystem", "browse", "depth", "local",
        ],
        "parameters": {
            "path": "Directory to walk (default: current directory)",
            "depth": "Max levels to recurse (default: 3)",
        },
        "optional": ["path", "depth"],
    },
    "read_file": {
        "fn": files.read_file,
        "description": (
            "Read the text content of a file. For large files, pass 'lines'"
            " as a 1-based inclusive range like '10-20' to read just that"
            " span. Output is line-numbered. Restricted to allowed"
            " directories."
        ),
        "tags": [
            "read", "file", "open", "cat", "view", "content", "source",
            "text", "lines", "show", "filesystem", "code", "local",
        ],
        "parameters": {
            "path": "File to read",
            "lines": "Optional line range 'start-end' (e.g. '10-20')",
        },
        "optional": ["lines"],
        "retain": "outline",
    },
    "search_code": {
        "fn": files.search_code,
        "description": (
            "Search file contents for a string or regex under a directory"
            " (like grep), returning 'relpath:line: text' rows. Case-"
            "insensitive unless your pattern has an uppercase letter. Pass"
            " glob to limit which files are searched: one or more globs,"
            " comma-separated (e.g. '*.py,*.md'). Use to"
            " find where something is defined or used before concluding code"
            " is missing. Skips noise dirs. Restricted to allowed directories."
        ),
        "tags": [
            "grep", "search", "find", "code", "definition", "symbol",
            "usage", "references", "locate", "where", "contents", "text",
            "regex", "pattern", "audit", "filesystem", "local",
        ],
        "parameters": {
            "pattern": "String or regex to search for (e.g. 'def my_func')",
            "path": "Directory or file to search (default: current directory)",
            "glob": (
                "Optional filename glob(s) to limit files: one or more globs,"
                " comma-separated (e.g. '*.py,*.md')"),
        },
        "optional": ["path", "glob"],
    },
    "write_file": {
        "fn": files.write_file,
        "description": (
            "Create or overwrite a file with the given content. Requires"
            " write access to the directory (asked once, showing the exact"
            " write); refused in read-only mode. Prefer edit_file to change"
            " part of an existing file."
        ),
        "tags": [
            "write", "create", "save", "file", "overwrite", "new", "output",
            "generate", "filesystem", "local", "change", "modify",
        ],
        "parameters": {
            "path": "File to write",
            "content": "Full text content to write to the file",
        },
    },
    "edit_file": {
        "fn": files.edit_file,
        "description": (
            "Replace a single unique occurrence of 'old' with 'new' in a"
            " file ('old' must appear exactly once; include surrounding"
            " context). Pass 'sha' — the value the last read_file/write_file/"
            "edit_file of this file returned (reuse it; no re-read needed)."
            " Returns the new sha so edits chain. Refused if the file changed"
            " since that sha. Requires write access; refused in read-only."
        ),
        "tags": [
            "edit", "change", "modify", "replace", "patch", "file", "update",
            "fix", "refactor", "filesystem", "local", "code", "write",
        ],
        "parameters": {
            "path": "File to edit",
            "old": "Exact text to replace (must be unique in the file)",
            "new": "Replacement text",
            "sha": "sha the last read/write/edit of this file returned",
        },
    },
    "delete_file": {
        "fn": files.delete_file,
        "description": (
            "Delete a single file. Destructive and write-gated: refused in"
            " read-only mode, asked once per directory (showing which file)."
            " Does not delete directories."
        ),
        "tags": [
            "delete", "remove", "rm", "unlink", "erase", "file", "trash",
            "destructive", "filesystem", "local",
        ],
        "parameters": {
            "path": "File to delete",
        },
    },
    # --- audited coding verbs (design plan chunk B) --------------------------
    "outline": {
        "fn": code.outline,
        "description": (
            "Outline a source file instead of reading it: one row per"
            " def/class with its line range and signature (nested ones"
            " indented) plus the module docstring, so you can pick the exact"
            " lines to read_file next. Non-Python files show their first 40"
            " numbered lines. Returns the file's sha. Use this BEFORE"
            " read_file on any file you have not seen."
        ),
        "tags": [
            "outline", "structure", "skeleton", "functions", "classes",
            "signatures", "summary", "navigate", "code", "file", "ast",
            "overview", "local",
        ],
        "parameters": {
            "path": "Source file to outline",
        },
    },
    "find_symbol": {
        "fn": code.find_symbol,
        "description": (
            "Find where a function, class, method or module-level name is"
            " defined and where it is referenced across the project's Python"
            " files, in one call ('def: path:line (kind)' rows, then"
            " 'ref: path:line: text' rows, max 30). kind narrows to 'def' or"
            " 'ref'. Prefer this over search_code for 'where is X defined /"
            " who calls X'."
        ),
        "tags": [
            "symbol", "definition", "references", "callers", "usages",
            "find", "where", "defined", "function", "class", "locate",
            "code", "local",
        ],
        "parameters": {
            "name": "The identifier to look up (e.g. 'compact_messages')",
            "kind": "Optional filter: 'def' or 'ref' (default both)",
        },
        "optional": ["kind"],
    },
    "run_tests": {
        "fn": quality.run_tests,
        "description": (
            "Run the project's test suite (pytest by default; unittest when"
            " the project policy says so) and get a short digest: the"
            " summary line and the failing test ids with their assertion"
            " line. target = a test file, directory or 'file::test' node id"
            " (default: whole project); k = pytest -k expression; maxfail ="
            " stop after N failures (default 1); detail = a failing test id"
            " to see that failure in full. Always run this to verify an edit"
            " before reporting it done."
        ),
        "tags": [
            "test", "tests", "pytest", "unittest", "run", "verify", "check",
            "failing", "suite", "assert", "regression", "ci", "local",
        ],
        "parameters": {
            "target": "Test file, directory or node id (default: project)",
            "k": "Optional pytest -k expression to select tests",
            "maxfail": "Stop after this many failures (default 1)",
            "detail": "A failing test id to expand into its full block",
        },
        "optional": ["target", "k", "maxfail", "detail"],
        "retain": "keep",
    },
    "check_syntax": {
        "fn": quality.check_syntax,
        "description": (
            "Compile one Python file (in-process, nothing written) and"
            " report 'ok' or the SyntaxError with line, column and text."
            " Cheap: call it after every edit to a .py file, before"
            " run_tests."
        ),
        "tags": [
            "syntax", "compile", "check", "verify", "python", "error",
            "parse", "valid", "code", "local",
        ],
        "parameters": {
            "path": "Python file to compile",
        },
        "retain": "keep",
    },
    "lint": {
        "fn": quality.lint,
        "description": (
            "Run the project's linters (flake8; mypy when the project"
            " configures it) on a file or directory and get counts plus the"
            " first 10 issues per linter. detail = 'flake8' or 'mypy'"
            " returns that linter's output (up to 4 KB). Read-only."
        ),
        "tags": [
            "lint", "flake8", "mypy", "style", "pep8", "typecheck", "types",
            "static", "analysis", "quality", "warnings", "code", "local",
        ],
        "parameters": {
            "path": "File or directory to lint (default: project)",
            "detail": "Optional: 'flake8' or 'mypy' to expand that output",
        },
        "optional": ["path", "detail"],
        "retain": "keep",
    },
    "git_status": {
        "fn": gitread.git_status,
        "description": (
            "Show the working tree status of the project's git repository"
            " (porcelain 'XY path' rows incl. untracked files) with a count."
            " Read-only: never stages or commits."
        ),
        "tags": [
            "git", "status", "changed", "modified", "untracked", "staged",
            "working tree", "dirty", "repository", "vcs", "local",
        ],
        "parameters": {},
        "retain": "keep",
    },
    "git_diff": {
        "fn": gitread.git_diff,
        "description": (
            "Show unstaged changes in the project's git repository: a"
            " per-file +/- stat by default, optionally limited to path;"
            " detail=true returns the unified diff itself (capped at 8 KB)."
            " Read-only."
        ),
        "tags": [
            "git", "diff", "changes", "patch", "modified", "review", "what"
            " changed", "unstaged", "repository", "vcs", "local",
        ],
        "parameters": {
            "path": "Optional file or directory to limit the diff to",
            "detail": "true to return the unified diff instead of the stat",
        },
        "optional": ["path", "detail"],
        "retain": "keep",
    },
    "apply_patch": {
        "fn": patch.apply_patch,
        "description": (
            "Apply a unified diff (one or more files, '--- a/x' / '+++ b/x'"
            " or plain paths relative to the working directory). Every"
            " hunk's context must match the file exactly; the whole patch is"
            " validated first and applied all-or-nothing. New files inside"
            " the project are allowed; renames, deletions and binary patches"
            " are refused. Write-gated like edit_file (refused in read-only"
            " mode). Returns per file the hunks applied and the new sha."
            " Verify with check_syntax/run_tests afterwards."
        ),
        "tags": [
            "patch", "diff", "apply", "unified", "hunk", "edit", "change",
            "modify", "multi-file", "write", "code", "local",
        ],
        "parameters": {
            "diff": "The unified diff text to apply",
        },
    },
}


def _match_tools(query: str) -> list:
    """Rank TOOL_REGISTRY entries by weighted match across metadata fields."""
    keywords = [
        w.lower() for w in query.replace('-', ' ').split()
        if len(w) > 2 and w.lower() not in _STOP_WORDS
    ]
    names = _advertised()
    if not keywords:
        return names
    scores: dict = {}
    for name in names:
        info = TOOL_REGISTRY[name]
        score = 0
        tags_text = ' '.join(info['tags']).lower()
        desc_text = info['description'].lower()
        params_text = ' '.join(info['parameters'].values()).lower()
        for kw in keywords:
            if kw in name.lower():
                score += 5   # exact name match
            if kw in tags_text:
                score += 3   # tag hit
            if kw in desc_text:
                score += 2   # description hit
            if kw in params_text:
                score += 1   # parameter description hit
        if score > 0:
            scores[name] = score
    if not scores:
        return names
    return sorted(scores, key=scores.__getitem__, reverse=True)


def search_tools(query: str) -> str:
    """
    Search the tool directory for tools matching an action you want to perform.

    Call this with a short phrase describing WHAT YOU WANT TO DO — not the
    user's question. Use action-oriented terms that match what a tool does:
      search_tools("search the web")
      search_tools("fetch webpage url")
      search_tools("get latest github release version")

    Matched tools are added to your active tool set and can be called directly.
    """
    matched = _match_tools(query)
    lines: list = [f"Tools matching '{query}':\n"]
    for name in matched:
        info = TOOL_REGISTRY[name]
        param_lines = "\n".join(
            f"      {k}: {v}" for k, v in info['parameters'].items()
        )
        lines.append(
            f"  {name}\n"
            f"    {info['description']}\n"
            f"    Parameters:\n{param_lines}\n"
        )
    lines.append("These tools are now active — call them directly by name.")
    return "\n".join(lines)


_SEARCH_TOOLS_SPEC = {
    'name': 'search_tools',
    'description': (
        'Search the tool directory for tools matching an action you want to'
        ' perform. Call with a short phrase describing WHAT YOU WANT TO DO.'
        ' Matched tools become active and can then be called directly.'
    ),
    'parameters': {
        'query': 'A short phrase describing the action you want to perform',
    },
}


def retain_policy(name: str) -> str:
    """How a tool's output should be retained after a turn: 'keep' (default),
    'summarize' (query-focused, for bulky web output), or 'outline' (code
    skeleton, for large file reads)."""
    info = TOOL_REGISTRY.get(name)
    if not info:
        return 'keep'
    return info.get('retain', 'keep')


def active_specs() -> list:
    """Return provider-neutral specs for the currently active tools.

    Each spec is ``{name, description, parameters}``. Adapters translate this
    to their native tool schema. search_tools is always present; discovered
    registry tools are added as they are activated.
    """
    return specs_for(session.active_tool_names, session.can_spawn,
                     session.controller)


def specs_for(active_tool_names, can_spawn: bool,
              controller: bool = False) -> list:
    """Provider-neutral specs for a given tool set (no session routing).

    Lets callers (e.g. the context breakdown) price a specific agent's tool
    schemas without binding that agent's session context. A controller gets
    only spawn/check/join/use_skill, whatever ``active_tool_names`` holds.
    A tool the project policy disables is left out even if it is in
    ``active_tool_names`` (e.g. activated before the policy changed).
    """
    if controller:
        return [_SPAWN_SPEC, _CHECK_SPEC, _JOIN_SPEC, _USE_SKILL_SPEC]
    specs = [_SEARCH_TOOLS_SPEC, _USE_SKILL_SPEC]
    if can_spawn:
        specs.extend([_SPAWN_SPEC, _CHECK_SPEC, _JOIN_SPEC])
    for name in _advertised():
        if name in active_tool_names:
            info = TOOL_REGISTRY[name]
            specs.append({
                'name': name,
                'description': info['description'],
                'parameters': info['parameters'],
                'optional': info.get('optional', []),
            })
    return specs


def _core_tool_fns() -> list:
    """(name, fn) for the pre-activated toolset — the tools an agent can call
    directly without going through search_tools first. Normally the
    config-driven core set; when [tools] flat = true it is the ENTIRE registry,
    so a capable model gets the whole toolset up front (no search_tools
    hop). Either way a tool the project policy disables is skipped."""
    names = (list(TOOL_REGISTRY) if config.FLAT_TOOLS
             else config.PREACTIVATE_TOOLS)
    out = []
    for name in names:
        info = TOOL_REGISTRY.get(name)
        if info and is_enabled(name):
            out.append((name, info['fn']))
    return out


def initial_tools(can_spawn: bool, controller: bool = False) -> tuple:
    """The active tool list + activated-name set an agent starts a turn with:
    the always-on tools (search_tools, use_skill, and spawn/check/join when
    delegation-capable) plus the pre-activated core toolset. A controller
    gets exactly spawn/check/join/use_skill and no core tools."""
    if controller:
        return [spawn, check, join, use_skill], set()
    base = [search_tools, use_skill]
    if can_spawn:
        base.extend([spawn, check, join])
    names = set()
    for name, fn in _core_tool_fns():
        base.append(fn)
        names.add(name)
    return base, names


def reset_active_tools() -> None:
    """Reset the active tool set to the always-on + pre-activated core tools.

    search_tools/use_skill are always present; spawn/check/join are added for
    delegation-capable agents (so the Ollama adapter, which introspects the
    callables, sees them); the pre-activated core toolset is added so common
    file tools can be called without a search_tools hop.
    """
    base, names = initial_tools(session.can_spawn, session.controller)
    session.active_tools[:] = base
    session.active_tool_names.clear()
    session.active_tool_names.update(names)


def activate(name: str) -> None:
    """Add a registry tool to the active set if not already present; a
    tool the project policy disables is never activated."""
    if (name in TOOL_REGISTRY and is_enabled(name)
            and name not in session.active_tool_names):
        session.active_tool_names.add(name)
        session.active_tools.append(TOOL_REGISTRY[name]['fn'])
        ui.console.print(f"[green]\\[ACTIVATED][/green] {name}")


def _redact_for_remote(name: str, result: str) -> str:
    """Redact secrets from a tool result bound for a remote provider.

    Applies only when the bound session's adapter is remote and the secret
    scan is on (``config.SECRET_SCAN``); findings are replaced with typed
    markers (design doc §2 step 6), counted into ``struggle['redactions']``
    and logged. Local adapters see the raw result.
    """
    if not config.SECRET_SCAN or not getattr(session.adapter, 'remote', False):
        return result
    findings = policy.scan(result)
    if not findings:
        return result
    ledger.bump('redactions', len(findings))
    kinds = sorted({f.kind for f in findings})
    log.info('redacted %d finding(s) (%s) from %s result for remote '
             'adapter %s', len(findings), ', '.join(kinds), name,
             getattr(session.adapter, 'name', '?'))
    ui.console.print(
        f"[yellow]\\[REDACTED][/yellow] {len(findings)} sensitive span(s)"
        f" ({', '.join(kinds)}) removed before sending to a remote model")
    return policy.redact(result, findings)


def _mode_denial(result: str) -> bool:
    """Whether a tool result text is an access-mode/allow-list refusal
    (message texts owned by guru.domain.files / ensure_domain_allowed), or
    the subprocess runner's own refusal (``procs.DENIED_PREFIX``, which the
    verbs surface bare or behind ``Refused: ``)."""
    head = result[:300]
    return (head.startswith('Refused: read-only mode')
            or head.startswith(procs.DENIED_PREFIX)
            or head.startswith(f'Refused: {procs.DENIED_PREFIX}')
            or 'was denied by the user' in head
            or (head.startswith('Write access to ') and head.rstrip()
                .endswith('was denied.')))


def _record_event(name: str, arguments: dict, raw: str, shown: str,
                  seconds: float, denied: str) -> None:
    """One ``tool_events`` audit row for a finished execute_tool call."""
    ok = not denied and not raw.startswith(('Tool error:', 'Unknown tool:'))
    ledger.record_tool_event(
        name, arguments, seconds=seconds, ok=ok, produced_bytes=len(raw),
        shown_bytes=len(shown), files_touched=_files_touched(name, arguments),
        denied=denied)


def _files_touched(name: str, arguments: dict) -> list:
    """The paths a tool call named: its ``path`` (or ``target`` for
    run_tests) argument, or every file an apply_patch diff addresses."""
    if not isinstance(arguments, dict):
        return []
    if name == 'apply_patch':
        return patch.targets(str(arguments.get('diff', '')))
    for key in ('path', 'target'):
        value = arguments.get(key)
        if value:
            return [str(value)]
    return []


def execute_tool(name: str, arguments: dict) -> str:
    """Run a tool the model requested and return its result text.

    Handles search_tools activation, the project tool policy, and
    unknown/error cases; the domain and directory allow-list gates are
    applied inside the individual tools. Every call — including refused
    and unknown ones — writes one ``tool_events`` ledger row.
    """
    # File-write tools render their own '⏺ Verb(file)' diff block, so the raw
    # note (which would dump the whole content/old/new) is shown as just the
    # path — or skipped for delete, which prints its own line.
    if name in ('write_file', 'edit_file'):
        ui.note_tool(name, str(arguments.get('path', '')))
    elif name == 'apply_patch':
        ui.note_tool(name, ', '.join(patch.targets(
            str(arguments.get('diff', '')))))
    elif name != 'delete_file':
        ui.note_tool(name, ' '.join(str(v) for v in arguments.values()))
    denied = ''
    started = time.monotonic()
    if session.controller and name not in CONTROLLER_TOOLS:
        # A controller coordinates only; CONTROLLER_HINT promises it has no
        # other tools, so keep that true (the attempt is still measured by
        # turn.controller_executed).
        result = f"Unknown tool: {name}"
        denied = 'controller'
    elif name in TOOL_REGISTRY and not is_enabled(name):
        result = f"Tool '{name}' is disabled by .guru/tools.toml"
        denied = 'policy'
    elif name == "search_tools":
        result = search_tools(**arguments)
        for tn in _match_tools(arguments.get("query", "")):
            activate(tn)
    elif name == "use_skill":
        result = use_skill(**arguments)
    elif name == "spawn":
        result = spawn(**arguments)
    elif name == "check":
        result = check(**arguments)
    elif name == "join":
        result = join(**arguments)
    elif name in TOOL_REGISTRY:
        try:
            result = TOOL_REGISTRY[name]["fn"](**arguments)
        except Exception as e:                       # noqa: BLE001
            result = f"Tool error: {e}"
    else:
        result = f"Unknown tool: {name}"
    seconds = time.monotonic() - started
    if not denied and _mode_denial(result):
        denied = 'mode'
    raw = result
    result = _redact_for_remote(name, result)
    # Struggle counters: a tool that raised, or an edit_file refused because
    # the model passed a stale sha (message text owned by files.edit_file).
    if result.startswith('Tool error:'):
        ledger.bump('tool_errors')
    elif name == 'edit_file' and result.startswith('sha mismatch:'):
        ledger.bump('sha_mismatches')
    _record_event(name, arguments, raw, result, seconds, denied)
    # Show the output's size — the context cost of this tool result.
    ui.note_tool_result(len(result))
    return result
