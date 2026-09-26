"""The ``/brief`` command body (endpoint layer): ``guru.domain.brief`` over
the JSON store ``guru.repositories.briefs``. The TUI imports this; the
domain module imports neither."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from guru.domain import brief
from guru.repositories import briefs

BRIEF_USAGE = ('/brief            show the project brief (build or load)\n'
               '/brief refresh    rebuild it for the current HEAD\n'
               '/brief slice <task text>   preview the slice a worker gets')


def brief_command(args: str, root: Optional[Path] = None,
                  store: Optional[brief.BriefStore] = None) -> str:
    """``''`` shows the map (building or loading), ``refresh`` rebuilds,
    ``slice <text>`` previews a worker's slice; anything else is the usage
    text. Returns the text to print -- one line on any failure (the brief
    is a saving, never a dependency, so a broken project file or store
    must not take the command down)."""
    root = Path(root or Path.cwd()).resolve()
    words = (args or '').strip().split(None, 1)
    verb = words[0].lower() if words else ''
    if verb not in ('', 'refresh', 'slice'):
        return BRIEF_USAGE
    try:
        b = brief.current(root, store if store is not None else briefs,
                          refresh=(verb == 'refresh'))
    except Exception as e:                               # noqa: BLE001
        return f"brief: cannot build for {root}: {type(e).__name__}: {e}"
    if verb == 'slice':
        text = words[1] if len(words) > 1 else ''
        return brief.slice(b, text)
    tail = (f"built in {b.build_seconds}s at {b.built_at};"
            f" {len(b.outlines)} modules outlined,"
            f" {len(b.symbols)} symbols"
            + (" (outlining truncated at the budget)" if b.truncated
               else ''))
    if b.head_sha == brief.NO_HEAD:
        tail += "; not a git checkout, so not stored"
    return f"{brief.render_map(b)}\n{tail}"
