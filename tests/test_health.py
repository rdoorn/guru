"""Tests for ``guru.domain.health``: the per-function metrics, the
qualified names, the before/after delta and its verdicts (improve-loop
plan, package B)."""
import textwrap

import pytest

from guru.domain import health


def _one(src: str) -> health.FunctionHealth:
    [fh] = health.file_health(textwrap.dedent(src))
    return fh


# --- metrics -----------------------------------------------------------------

class TestMetrics:
    def test_trivial_function(self) -> None:
        fh = _one('''
            def f():
                pass
        ''')
        assert fh == health.FunctionHealth('f', 2, 2, 1, 0, 0, 0)
        assert fh.over() == ()
        assert fh.describe() == ('lines 2, complexity 1, nesting 0, '
                                 'args 0, returns 0')

    def test_lines_span_the_def_not_the_decorators(self) -> None:
        fh = _one('''
            @dec
            @other
            def f():
                a = 1
                return a
        ''')
        assert fh.lineno == 4 and fh.lines == 3

    @pytest.mark.parametrize('body,expected', [
        ('if a: pass', 2),
        ('if a: pass\nelif b: pass\nelse: pass', 3),
        ('for x in a: pass', 2),
        ('while a: pass', 2),
        ('try:\n    pass\nexcept ValueError:\n    pass\n'
         'except KeyError:\n    pass', 3),
        ('with a: pass', 2),
        ('assert a', 2),
        ('x = [i for i in a if i if i > 1]', 3),
        ('x = 1 if a else 2', 2),
        ('match a:\n    case 1: pass\n    case 2: pass', 3),
        ('x = a and b', 2),
        ('x = a and b and c or d', 4),
        ('x = 1', 1),
    ])
    def test_complexity_per_construct(self, body, expected) -> None:
        src = 'def f(a, b, c, d):\n' + textwrap.indent(body, '    ') + '\n'
        [fh] = health.file_health(src)
        assert fh.complexity == expected, body

    def test_nesting_depth_counts_compound_statements(self) -> None:
        fh = _one('''
            def f(a):
                if a:
                    for x in a:
                        with a:
                            try:
                                pass
                            except ValueError:
                                pass
                return 1
        ''')
        assert fh.nesting == 4

    def test_elif_does_not_deepen_and_else_does(self) -> None:
        fh = _one('''
            def f(a):
                if a:
                    pass
                elif a > 1:
                    if a:
                        pass
                else:
                    if a:
                        pass
        ''')
        assert fh.nesting == 2

    def test_match_and_try_blocks_nest(self) -> None:
        fh = _one('''
            def f(a):
                match a:
                    case 1:
                        if a:
                            pass
                try:
                    pass
                finally:
                    while a:
                        pass
        ''')
        assert fh.nesting == 2

    def test_args_exclude_self_cls_and_star_args(self) -> None:
        src = textwrap.dedent('''
            class C:
                def m(self, a, /, b, *args, c, d=1, **kw):
                    pass

                @classmethod
                def k(cls, a):
                    pass

            def f(*args, **kw):
                pass
        ''')
        rows = {fh.name: fh.args for fh in health.file_health(src)}
        assert rows == {'C.m': 4, 'C.k': 1, 'f': 0}

    def test_return_count(self) -> None:
        fh = _one('''
            def f(a):
                if a:
                    return 1
                if a > 1:
                    return 2
                return 3
        ''')
        assert fh.returns == 3

    def test_nested_functions_are_measured_apart(self) -> None:
        src = textwrap.dedent('''
            def outer(a):
                def inner(b):
                    if b:
                        return b
                    return 0
                x = lambda q: q if a else 0
                return inner(a)

            async def co(a):
                async for x in a:
                    async with x:
                        pass
        ''')
        rows = {fh.name: fh for fh in health.file_health(src)}
        assert set(rows) == {'outer', 'outer.inner', 'co'}
        # the lambda's ternary counts for outer; inner's branch does not
        assert rows['outer'].complexity == 2
        assert rows['outer'].returns == 1 and rows['outer'].nesting == 0
        assert rows['outer.inner'].complexity == 2
        assert rows['outer.inner'].returns == 2
        assert rows['co'].complexity == 3 and rows['co'].nesting == 2

    def test_methods_are_qualified_by_class(self) -> None:
        src = textwrap.dedent('''
            class A:
                def m(self):
                    pass

                class B:
                    def n(self):
                        def deep():
                            pass
        ''')
        assert [fh.name for fh in health.file_health(src)] == [
            'A.m', 'A.B.n', 'A.B.n.deep']

    def test_over_thresholds_and_describe(self) -> None:
        body = '\n'.join(f'    x{i} = {i}' for i in range(61))
        src = 'def f(a, b, c, d, e, f, g):\n' + body + '\n'
        [fh] = health.file_health(src)
        assert fh.lines == 62 and fh.args == 7
        assert fh.over() == ('lines', 'args')
        assert fh.describe().startswith('lines 62 >60, complexity 1, ')
        assert 'args 7 >6' in fh.describe()

    def test_thresholds(self) -> None:
        assert (health.LINES_MAX, health.COMPLEXITY_MAX, health.NESTING_MAX,
                health.ARGS_MAX) == (60, 10, 4, 6)
        assert health.WORSE_RATIO == 0.2

    @pytest.mark.parametrize('src', ['def f(:', 'x = (', '\x00', ''])
    def test_unparsable_or_empty_source_is_empty(self, src) -> None:
        assert health.file_health(src) == []

    def test_function_health_on_a_node(self) -> None:
        import ast
        node = ast.parse('def f(a, b):\n    return a or b\n').body[0]
        fh = health.function_health(node, 'Q.f')
        assert fh.name == 'Q.f' and fh.complexity == 2 and fh.args == 2
        assert health.function_health(node).name == 'f'


# --- delta -------------------------------------------------------------------

def _fn(name: str, n_args: int = 1, body_lines: int = 1,
        branches: int = 0) -> str:
    args = ', '.join(f'a{i}' for i in range(n_args))
    lines = [f'def {name}({args}):']
    lines += [f'    if a0 == {i}:\n        pass' for i in range(branches)]
    lines += [f'    x{i} = {i}' for i in range(body_lines)]
    lines.append('    return 0')
    return '\n'.join(lines) + '\n\n'


class TestDelta:
    def test_untouched_functions_are_left_out(self) -> None:
        src = _fn('f') + _fn('g', 2)
        assert health.delta(src, src) == []

    def test_metric_change_within_thresholds_is_unchanged(self) -> None:
        [d] = health.delta(_fn('f'), _fn('f', 3, 5, 2))
        assert d.name == 'f' and d.verdict == health.UNCHANGED
        assert d.reasons == () and d.before is not None
        assert d.describe().startswith('f: unchanged (lines ')

    def test_crossing_a_threshold_is_degraded(self) -> None:
        [d] = health.delta(_fn('f', 6), _fn('f', 7))
        assert d.verdict == health.DEGRADED
        assert d.reasons == ('args 6→7 >6',)
        assert d.describe() == 'f: degraded (args 6→7 >6)'

    def test_crossing_lines_and_complexity(self) -> None:
        before = _fn('f', 1, 58, 0)
        after = _fn('f', 1, 40, 12)
        [d] = health.delta(before, after)
        assert d.verdict == health.DEGRADED
        assert d.reasons == ('lines 60→66 >60', 'complexity 1→13 >10')

    def test_new_function_over_a_threshold_is_degraded(self) -> None:
        [d] = health.delta('', _fn('f', 7))
        assert d.before is None and d.verdict == health.DEGRADED
        assert d.describe() == 'f: degraded (new; args 7 >6)'
        assert health.delta(None, _fn('f', 7)) == [d]

    def test_new_function_under_thresholds_is_unchanged(self) -> None:
        [d] = health.delta(_fn('f'), _fn('f') + _fn('g'))
        assert d.name == 'g' and d.verdict == health.UNCHANGED
        assert d.describe().startswith('g: unchanged (new; lines ')

    def test_existing_debt_growing_a_little_is_unchanged(self) -> None:
        # already over: 7 args -> 8 is +14%, within WORSE_RATIO
        [d] = health.delta(_fn('f', 7), _fn('f', 8))
        assert d.verdict == health.UNCHANGED

    def test_existing_debt_growing_over_20_percent_is_degraded(self) -> None:
        [d] = health.delta(_fn('f', 7), _fn('f', 9))
        assert d.verdict == health.DEGRADED
        assert d.reasons == ('args 7→9 >6 (+29%)',)

    def test_reducing_debt_is_improved(self) -> None:
        [d] = health.delta(_fn('f', 9), _fn('f', 7))
        assert d.verdict == health.IMPROVED
        assert d.reasons == ('args 9→7 >6',)
        [d] = health.delta(_fn('f', 9), _fn('f', 2))
        assert d.verdict == health.IMPROVED and d.reasons == ('args 9→2',)

    def test_degraded_wins_over_improved(self) -> None:
        [d] = health.delta(_fn('f', 9, 1, 0), _fn('f', 7, 1, 12))
        assert d.verdict == health.DEGRADED
        assert d.reasons == ('complexity 1→13 >10',)

    def test_removed_functions_are_not_reported(self) -> None:
        assert health.delta(_fn('f', 9) + _fn('g'), _fn('g')) == []

    def test_methods_match_by_qualified_name(self) -> None:
        before = 'class C:\n    def m(self, a):\n        return a\n'
        after = ('class C:\n    def m(self, a, b, c, d, e, f, g):\n'
                 '        return a\n')
        [d] = health.delta(before, after)
        assert d.name == 'C.m' and d.verdict == health.DEGRADED
        # a module-level m is a different (here: new) function
        deltas = health.delta(before, after + '\n\ndef m(a):\n    return a\n')
        assert [(d.name, d.verdict, d.before is None) for d in deltas] == [
            ('C.m', health.DEGRADED, False), ('m', health.UNCHANGED, True)]

    def test_after_order_is_kept(self) -> None:
        after = _fn('b', 7) + _fn('a', 7)
        assert [d.name for d in health.delta('', after)] == ['b', 'a']

    def test_unparsable_side_is_empty(self) -> None:
        assert health.delta('def f(:', _fn('f', 7)) == []
        assert health.delta(_fn('f'), 'def f(:') == []

    def test_notable(self) -> None:
        deltas = health.delta(_fn('f', 9) + _fn('g'), _fn('f', 2)
                              + _fn('g', 2) + _fn('h', 7))
        assert [d.verdict for d in deltas] == [
            health.IMPROVED, health.UNCHANGED, health.DEGRADED]
        assert [d.name for d in health.notable(deltas)] == ['f', 'h']
