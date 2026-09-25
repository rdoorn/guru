"""Code-health metrics per Python function, and the before/after delta
the quality gate and the ``code_health`` verb report (improve-loop plan,
package B; the CodeScene reading).

Pure ``ast`` over source text: no files, no processes. For every
function (methods qualified ``Class.method``, nested functions
``outer.inner``) :func:`function_health` measures

- ``lines``: the def's line span (decorators excluded);
- ``complexity``: cyclomatic complexity ``1 +`` every ``if``/``elif``,
  ``for``, ``while``, ``except``, ``with``, ``assert``, comprehension
  ``if``, ternary, ``match`` case, and every boolean operator operand
  after the first (``a and b or c`` adds 2);
- ``nesting``: the deepest chain of compound statements (``if``/``for``/
  ``while``/``with``/``try``/``match``) inside the body — an ``elif`` sits
  at its ``if``'s depth;
- ``args``: positional and keyword parameters, ``self``/``cls`` excluded
  (``*args``/``**kwargs`` not counted);
- ``returns``: ``return`` statements.

A nested def is measured on its own and does not count towards the
enclosing function. Lambdas are part of the function that contains them.

:func:`delta` matches the functions of two versions of a module by
qualified name and gives each changed or new function a verdict against
the fixed thresholds (``LINES_MAX`` 60, ``COMPLEXITY_MAX`` 10,
``NESTING_MAX`` 4, ``ARGS_MAX`` 6):

- ``degraded``: a metric crosses a threshold it was under before; a new
  function is over a threshold; or a metric already over its threshold
  grows by more than ``WORSE_RATIO`` (20%);
- ``improved``: no degradation, and a metric that was over its threshold
  shrank (whether or not it is under the threshold now);
- ``unchanged``: everything else — pre-existing debt that stays as it is
  (or grows within 20%) is not news, and a healthy function that changed
  within the thresholds is not either.

Functions the change removes are not reported (a removal is a
``destructive`` question, not a health one). A source that does not
parse yields an empty result; nothing here raises for odd input.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterator, Optional

LINES_MAX = 60
COMPLEXITY_MAX = 10
NESTING_MAX = 4
ARGS_MAX = 6
WORSE_RATIO = 0.2                # growth of an already-over metric
IMPROVED, UNCHANGED, DEGRADED = 'improved', 'unchanged', 'degraded'
VERDICTS = (IMPROVED, UNCHANGED, DEGRADED)
# (metric, threshold) in report order; ``returns`` is measured and shown
# but has no threshold.
THRESHOLDS: tuple[tuple[str, int], ...] = (
    ('lines', LINES_MAX), ('complexity', COMPLEXITY_MAX),
    ('nesting', NESTING_MAX), ('args', ARGS_MAX))
METRICS = tuple(m for m, _t in THRESHOLDS) + ('returns',)

_FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEF_NODES = (*_FUNC_NODES, ast.ClassDef)
_BRANCH_NODES = (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While,
                 ast.ExceptHandler, ast.With, ast.AsyncWith, ast.Assert,
                 ast.match_case)
_NESTING_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With,
                  ast.AsyncWith, ast.Try, ast.TryStar, ast.Match)
_SELF_NAMES = frozenset(('self', 'cls'))


@dataclass(frozen=True)
class FunctionHealth:
    """The metrics of one function: its qualified ``name``, first
    ``lineno`` and the five measures the module docstring defines."""
    name: str
    lineno: int
    lines: int
    complexity: int
    nesting: int
    args: int
    returns: int

    def metric(self, name: str) -> int:
        """The value of ``name`` (one of ``METRICS``)."""
        return int(getattr(self, name))

    def over(self) -> tuple[str, ...]:
        """The metrics over their threshold, in report order."""
        return tuple(m for m, limit in THRESHOLDS if self.metric(m) > limit)

    def describe(self) -> str:
        """``'lines 12, complexity 3, nesting 1, args 2, returns 1'`` with
        ``>N`` after a metric over its threshold."""
        limits = dict(THRESHOLDS)
        parts = []
        for m in METRICS:
            value = self.metric(m)
            mark = f' >{limits[m]}' if m in limits and value > limits[m] \
                else ''
            parts.append(f'{m} {value}{mark}')
        return ', '.join(parts)


@dataclass(frozen=True)
class FunctionDelta:
    """One changed or new function: ``before`` (None for a new one),
    ``after``, the ``verdict`` (``VERDICTS``) and the ``reasons`` per
    metric that decided it (``'lines 55→72 >60'``)."""
    name: str
    before: Optional[FunctionHealth]
    after: FunctionHealth
    verdict: str
    reasons: tuple[str, ...]

    def describe(self) -> str:
        """``'f: degraded (lines 55→72 >60)'``; a new function says
        ``new``; an unchanged verdict lists the after metrics."""
        if self.reasons:
            why = '; '.join(self.reasons)
        else:
            why = self.after.describe()
        prefix = 'new; ' if self.before is None else ''
        return f'{self.name}: {self.verdict} ({prefix}{why})'


# --- metrics -----------------------------------------------------------------

def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Every node in ``node``'s body, not descending into nested defs
    (the def nodes themselves are not yielded either)."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, _DEF_NODES):
            continue
        yield child
        stack.extend(ast.iter_child_nodes(child))


def _complexity(node: ast.AST) -> int:
    total = 1
    for child in _own_nodes(node):
        if isinstance(child, _BRANCH_NODES):
            total += 1
        elif isinstance(child, ast.BoolOp):
            total += max(0, len(child.values) - 1)
        elif isinstance(child, ast.comprehension):
            total += len(child.ifs)
    return total


def _nesting(body: list, depth: int = 0) -> int:
    """The deepest compound-statement chain in ``body``."""
    deepest = depth
    for stmt in body:
        if isinstance(stmt, _DEF_NODES):
            continue
        if not isinstance(stmt, _NESTING_NODES):
            continue
        inner = depth + 1
        blocks: list = [] if isinstance(stmt, ast.Match) else [stmt.body]
        if isinstance(stmt, ast.If):
            # ``elif``: the orelse is exactly one If — same depth as the if.
            if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
                deepest = max(deepest, _nesting(stmt.orelse, depth))
            else:
                blocks.append(stmt.orelse)
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            blocks.append(stmt.orelse)
        elif isinstance(stmt, (ast.Try, ast.TryStar)):
            blocks.extend(h.body for h in stmt.handlers)
            blocks.extend((stmt.orelse, stmt.finalbody))
        elif isinstance(stmt, ast.Match):
            blocks.extend(c.body for c in stmt.cases)
        for block in blocks:
            deepest = max(deepest, _nesting(block, inner))
    return deepest


def _args(node: ast.AST) -> int:
    assert isinstance(node, _FUNC_NODES)
    params = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    if params and params[0].arg in _SELF_NAMES:
        params = params[1:]
    return len(params)


def _returns(node: ast.AST) -> int:
    return sum(1 for child in _own_nodes(node)
               if isinstance(child, ast.Return))


def function_health(node: ast.AST, name: str = '') -> FunctionHealth:
    """The :class:`FunctionHealth` of a ``FunctionDef``/``AsyncFunctionDef``
    node; ``name`` overrides the node's own (the caller qualifies it)."""
    assert isinstance(node, _FUNC_NODES)
    end = getattr(node, 'end_lineno', None) or node.lineno
    return FunctionHealth(
        name=name or node.name, lineno=node.lineno,
        lines=end - node.lineno + 1, complexity=_complexity(node),
        nesting=_nesting(node.body), args=_args(node),
        returns=_returns(node))


def _functions(body: list, prefix: str) -> Iterator[FunctionHealth]:
    for node in body:
        if isinstance(node, _FUNC_NODES):
            name = f'{prefix}{node.name}'
            yield function_health(node, name)
            yield from _functions(node.body, f'{name}.')
        elif isinstance(node, ast.ClassDef):
            yield from _functions(node.body, f'{prefix}{node.name}.')


def file_health(source: str) -> list[FunctionHealth]:
    """Every function of ``source`` in definition order, methods as
    ``Class.method`` and nested functions as ``outer.inner``. ``[]`` when
    the source does not parse."""
    try:
        tree = ast.parse(source or '')
    except (SyntaxError, ValueError):
        return []
    return list(_functions(tree.body, ''))


# --- delta -------------------------------------------------------------------

def _judge(before: Optional[FunctionHealth], after: FunctionHealth
           ) -> tuple[str, tuple[str, ...]]:
    """``(verdict, reasons)`` per the module docstring."""
    degraded: list[str] = []
    improved: list[str] = []
    for metric, limit in THRESHOLDS:
        now = after.metric(metric)
        if before is None:
            if now > limit:
                degraded.append(f'{metric} {now} >{limit}')
            continue
        was = before.metric(metric)
        if now == was:
            continue
        arrow = f'{metric} {was}→{now}'
        if now > limit and was <= limit:
            degraded.append(f'{arrow} >{limit}')
        elif now > limit and now > was * (1 + WORSE_RATIO):
            pct = round((now - was) * 100 / was)
            degraded.append(f'{arrow} >{limit} (+{pct}%)')
        elif was > limit and now < was:
            improved.append(arrow + (f' >{limit}' if now > limit else ''))
    if degraded:
        return DEGRADED, tuple(degraded)
    if improved:
        return IMPROVED, tuple(improved)
    return UNCHANGED, ()


def _same_metrics(a: FunctionHealth, b: FunctionHealth) -> bool:
    return all(a.metric(m) == b.metric(m) for m in METRICS)


def delta(before_src: Optional[str], after_src: str) -> list[FunctionDelta]:
    """The changed and new functions of ``after_src`` against
    ``before_src`` (``None``/``''`` for a new file), matched by qualified
    name, in ``after`` definition order, each with its verdict. A function
    whose metrics did not move is left out; a removed function is not
    reported. ``[]`` when either source fails to parse."""
    try:
        before_tree = ast.parse(before_src or '')
        after_tree = ast.parse(after_src or '')
    except (SyntaxError, ValueError):
        return []
    before = {fh.name: fh for fh in _functions(before_tree.body, '')}
    out: list[FunctionDelta] = []
    for fh in _functions(after_tree.body, ''):
        was = before.get(fh.name)
        if was is not None and _same_metrics(was, fh):
            continue
        verdict, reasons = _judge(was, fh)
        out.append(FunctionDelta(fh.name, was, fh, verdict, reasons))
    return out
