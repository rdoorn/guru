"""Tool directory: the tools themselves, discovery, gating, and execution.

Tool execution is provider-agnostic. Adapters call ``execute_tool`` when a
model requests a tool; this module handles the domain allow-list gate,
``search_tools`` activation, and running the tool, returning a result string.
"""
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from guru import config, session, ui

_STOP_WORDS = {
    'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'for',
    'of', 'and', 'or', 'but', 'what', 'how', 'when', 'where', 'who',
    'which', 'that', 'this', 'are', 'was', 'were', 'be', 'been',
    'being', 'do', 'does', 'did', 'me', 'my', 'you', 'your', 'its',
}


def ensure_domain_allowed(domain: str) -> bool:
    """Return True if the domain is allowed, prompting the user if unknown.

    On approval the domain is added to the in-memory set and persisted.
    Denial (no / empty / Ctrl+C) returns False and asks again next time.
    """
    domain = domain.lower()
    if domain in config.ALLOWED_DOMAINS:
        return True
    ui.console.print(
        f"\n[yellow]\\[ACCESS][/yellow] Request to access"
        f" [bold]{domain}[/bold]."
    )
    try:
        answer = input(f"Allow access to '{domain}'? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        answer = ''
    if answer in ('y', 'yes'):
        config.ALLOWED_DOMAINS.add(domain)
        config.persist_domain(domain)
        ui.console.print(
            f"[green]Allowed[/green] {domain} (saved to allow-list)."
        )
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
    # Tool output shown on screen is debug info for the user (the same text
    # is sent to the model as normal input), so render it in the debug style.
    ui.console.print(f"\n[SEARCH] {query}", style="dim yellow", markup=False)

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
    ui.console.print(f"\n[cyan]\\[FETCH][/cyan] {url}")

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup([
        "script", "style", "nav", "footer", "header", "aside",
    ]):
        element.decompose()

    text = soup.get_text(separator="\n", strip=True)
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
    ui.console.print(f"\n[cyan]\\[GITHUB][/cyan] {repo}")
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
}


def _match_tools(query: str) -> list:
    """Rank TOOL_REGISTRY entries by weighted match across metadata fields."""
    keywords = [
        w.lower() for w in query.replace('-', ' ').split()
        if len(w) > 2 and w.lower() not in _STOP_WORDS
    ]
    if not keywords:
        return list(TOOL_REGISTRY.keys())
    scores: dict = {}
    for name, info in TOOL_REGISTRY.items():
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
        return list(TOOL_REGISTRY.keys())
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
    ui.console.print(f"\n[cyan]\\[TOOL_SEARCH][/cyan] {query}")
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


def active_specs() -> list:
    """Return provider-neutral specs for the currently active tools.

    Each spec is ``{name, description, parameters}``. Adapters translate this
    to their native tool schema. search_tools is always present; discovered
    registry tools are added as they are activated.
    """
    specs = [_SEARCH_TOOLS_SPEC]
    for name in TOOL_REGISTRY:
        if name in session.active_tool_names:
            info = TOOL_REGISTRY[name]
            specs.append({
                'name': name,
                'description': info['description'],
                'parameters': info['parameters'],
            })
    return specs


def reset_active_tools() -> None:
    """Reset the active tool set to just the search_tools meta-tool."""
    session.active_tool_names.clear()
    session.active_tools[:] = [search_tools]


def activate(name: str) -> None:
    """Add a registry tool to the active set if not already present."""
    if name in TOOL_REGISTRY and name not in session.active_tool_names:
        session.active_tool_names.add(name)
        session.active_tools.append(TOOL_REGISTRY[name]['fn'])
        ui.console.print(f"[green]\\[ACTIVATED][/green] {name}")


def execute_tool(name: str, arguments: dict) -> str:
    """Run a tool the model requested and return its result text.

    Handles search_tools activation and unknown/error cases. The domain
    allow-list gate is applied inside the individual web tools.
    """
    ui.console.print(
        f"[cyan]\\[TOOL][/cyan] [bold]{name}[/bold]: {arguments}"
    )
    if name == "search_tools":
        result = search_tools(**arguments)
        for tn in _match_tools(arguments.get("query", "")):
            activate(tn)
        return result
    if name in TOOL_REGISTRY:
        try:
            return TOOL_REGISTRY[name]["fn"](**arguments)
        except Exception as e:
            return f"Tool error: {e}"
    return f"Unknown tool: {name}"
