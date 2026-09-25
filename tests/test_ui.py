"""Tests for guru.ui formatting and status helpers."""
from guru import cli, config, session, ui


class TestHumanCtx:
    """Tests for cli._human_ctx formatting."""

    def test_kilo(self) -> None:
        assert cli._human_ctx(65536) == '64k'
        assert cli._human_ctx(4096) == '4k'

    def test_mega(self) -> None:
        assert cli._human_ctx(1024 * 1024) == '1M'


class TestStatusParts:
    """Tests for ui._status_parts colour thresholds and segments."""

    def test_green_yellow_red(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'num_ctx', 1000)
        monkeypatch.setattr(session, 'model', 'demo:latest')

        monkeypatch.setattr(session, 'ctx_used', 100)
        assert ui._status_parts()[3] == 'green'

        monkeypatch.setattr(session, 'ctx_used', 780)
        left, ctx_segment, right, colour = ui._status_parts()
        assert colour == 'yellow'
        assert '78%' in ctx_segment

        monkeypatch.setattr(session, 'ctx_used', 900)
        assert ui._status_parts()[3] == 'red'

    def test_segments_include_expected_fields(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'num_ctx', 1000)
        monkeypatch.setattr(session, 'ctx_used', 100)
        monkeypatch.setattr(session, 'model', 'demo:latest')
        monkeypatch.setattr(session, 'model_size', '32B')
        monkeypatch.setattr(session, 'session_in', 1234)
        monkeypatch.setattr(session, 'session_out', 56)
        monkeypatch.setattr(session, 'git_branch', 'main')

        left, ctx_segment, right, _ = ui._status_parts()
        assert 'demo' in left and '32B' in left
        assert '🧠' in ctx_segment
        assert '1234' in right and '56' in right and 'main' in right


class TestFormatBytes:
    """Tests for ui.format_bytes."""

    def test_gigabytes(self) -> None:
        assert ui.format_bytes(8_200_000_000) == '7.6 GB'

    def test_megabytes(self) -> None:
        assert ui.format_bytes(5_000_000) == '4.8 MB'


class TestStatusCost:
    """The status bar shows the run cost after the token counters."""

    def _base(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'num_ctx', 1000)
        monkeypatch.setattr(session, 'ctx_used', 100)
        monkeypatch.setattr(session, 'model', 'demo:latest')
        monkeypatch.setattr(session, 'session_in', 12)
        monkeypatch.setattr(session, 'session_out', 34)
        monkeypatch.setattr(session, 'git_branch', 'main')

    def test_no_cost_fragment_when_nothing_spent(self, monkeypatch) -> None:
        self._base(monkeypatch)
        monkeypatch.setattr(session, 'cost_usd', 0.0)
        monkeypatch.setattr(session, 'cost_known', True)
        right = ui._status_parts()[2]
        assert '$' not in right

    def test_cost_with_four_decimals_after_tokens(self, monkeypatch) -> None:
        self._base(monkeypatch)
        monkeypatch.setattr(session, 'cost_usd', 0.01234)
        monkeypatch.setattr(session, 'cost_known', True)
        right = ui._status_parts()[2]
        assert ' | ↑ 34 | $0.0123 | 📁' in right

    def test_unknown_cost_shows_question_mark(self, monkeypatch) -> None:
        self._base(monkeypatch)
        monkeypatch.setattr(session, 'cost_usd', 0.0)
        monkeypatch.setattr(session, 'cost_known', False)
        right = ui._status_parts()[2]
        assert ' | ↑ 34 | $? | 📁' in right

    def test_partial_cost_shows_spend_plus_marker(self, monkeypatch) -> None:
        self._base(monkeypatch)
        monkeypatch.setattr(session, 'cost_usd', 0.5)
        monkeypatch.setattr(session, 'cost_known', False)
        right = ui._status_parts()[2]
        assert ' | ↑ 34 | $0.5000+? | 📁' in right


class _RowsRepo:
    """LedgerRepository with ``rows(stream, run_id=None)`` like JsonlLedger."""

    def __init__(self, seed=None) -> None:
        self.rows_by_stream: dict = dict(seed or {})

    def append(self, stream: str, row: dict) -> None:
        self.rows_by_stream.setdefault(stream, []).append(row)

    def rows(self, stream: str, run_id=None) -> list:
        out = list(self.rows_by_stream.get(stream, []))
        if run_id is not None:
            out = [r for r in out if r.get('run_id') == run_id]
        return out


class _AppendOnlyRepo:
    """Minimal LedgerRepository. ``rows`` is a list attribute on purpose
    (like conftest.FakeRepo): the cli must treat a non-callable ``rows`` as
    "cannot read back", not crash on it."""

    def __init__(self) -> None:
        self.rows: list = []

    def append(self, stream: str, row: dict) -> None:
        self.rows.append((stream, row))


class TestLabelCommands:
    """/good and /bad label the last completed turn and its tasks."""

    def _install(self, monkeypatch, repo: object) -> None:
        from guru.domain import ledger
        ledger.set_repository(repo)
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        monkeypatch.setattr(session, 'turn_id', 'turn1')

    def test_labels_turn_and_its_tasks(self, monkeypatch) -> None:
        from guru.domain import ledger
        repo = _RowsRepo({'tasks': [
            {'run_id': ledger.RUN_ID, 'task_id': 'tA', 'turn_id': 'turn1'},
            {'run_id': ledger.RUN_ID, 'task_id': 'tA', 'turn_id': 'turn1'},
            {'run_id': ledger.RUN_ID, 'task_id': 'tB', 'turn_id': 'turn1'},
            {'run_id': ledger.RUN_ID, 'task_id': 'tZ', 'turn_id': 'other'},
            {'run_id': 'old-run', 'task_id': 'tO', 'turn_id': 'turn1'},
        ]})
        self._install(monkeypatch, repo)
        try:
            cli._label_command('good', 'clean answer')
            ledger.flush()
        finally:
            ledger.set_repository(None)
        labels = repo.rows('labels')
        assert [(r['target_id'], r['label'], r['labeller'], r['note'])
                for r in labels] == [
            ('turn1', 'good', 'user', 'clean answer'),
            ('tA', 'good', 'user', 'clean answer'),
            ('tB', 'good', 'user', 'clean answer')]

    def test_bad_without_rows_support_labels_only_the_turn(
            self, monkeypatch, capsys) -> None:
        from guru.domain import ledger
        repo = _AppendOnlyRepo()
        self._install(monkeypatch, repo)
        try:
            cli._label_command('bad')
            ledger.flush()
        finally:
            ledger.set_repository(None)
        assert [(s, r['target_id'], r['label'], r['note'])
                for s, r in repo.rows] == [('labels', 'turn1', 'bad', '')]
        assert 'turn1' in capsys.readouterr().out

    def test_no_turn_yet(self, monkeypatch, capsys) -> None:
        from guru.domain import ledger
        repo = _AppendOnlyRepo()
        self._install(monkeypatch, repo)
        monkeypatch.setattr(session, 'turn_id', '')
        try:
            cli._label_command('good')
            ledger.flush()
        finally:
            ledger.set_repository(None)
        assert repo.rows == []
        assert 'No completed turn' in capsys.readouterr().out

    def test_no_repository(self, monkeypatch, capsys) -> None:
        from guru.domain import ledger
        ledger.set_repository(None)
        monkeypatch.setattr(session, 'turn_id', 'turn1')
        cli._label_command('good')
        assert 'ledger' in capsys.readouterr().out.lower()


class TestLedgerCommand:
    """/ledger prints the current run's summary."""

    def test_prints_models_tasks_and_top(self, monkeypatch, capsys) -> None:
        from guru.domain import ledger
        rid = ledger.RUN_ID
        repo = _RowsRepo({
            'calls': [
                {'run_id': rid, 'adapter': 'Ollama', 'model': 'qwen3:8b',
                 'tokens_in': 100, 'tokens_out': 20, 'cost_usd': 0.0},
                {'run_id': rid, 'adapter': 'Anthropic', 'model': 'opus',
                 'tokens_in': 10, 'tokens_out': 2, 'cost_usd': 0.25},
                {'run_id': 'old', 'adapter': 'X', 'model': 'ghost',
                 'tokens_in': 1, 'tokens_out': 1, 'cost_usd': 5.0}],
            'tasks': [
                {'run_id': rid, 'task_id': 't1', 'status': 'done',
                 'adapter': 'Ollama', 'model': 'qwen3:8b', 'cost_usd': 0.0,
                 'seconds': 3.5, 'task': 'review upload.py for bugs',
                 'role': 'security-engineer', 'kind': 'review'}]})
        ledger.set_repository(repo)
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        try:
            cli._ledger_command()
        finally:
            ledger.set_repository(None)
        out = capsys.readouterr().out
        assert 'qwen3:8b' in out and 'opus' in out and 'ghost' not in out
        assert '0.2500' in out and 'security-engineer' in out
        assert rid in out

    def test_without_repository(self, capsys) -> None:
        from guru.domain import ledger
        ledger.set_repository(None)
        cli._ledger_command()
        assert 'ledger' in capsys.readouterr().out.lower()

    def test_format_handles_unknown_cost(self) -> None:
        summary = {'run_id': 'r', 'models': {'A|m': {
            'adapter': 'A', 'model': 'm', 'calls': 1, 'tokens_in': 1,
            'tokens_out': 1, 'cache_read': 0, 'cache_write': 0,
            'cost_usd': None}}, 'tasks': {},
            'top_tasks': [], 'totals': {'calls': 1, 'tokens_in': 1,
                                        'tokens_out': 1, 'cache_read': 0,
                                        'cache_write': 0, 'cost_usd': None,
                                        'tasks': 0}}
        text = cli._format_run_summary(summary)
        assert '$?' in text and 'A|m' in text


class TestTurnLine:
    """The per-turn cost line and the exit summary behind
    ``[ledger] turn_line``."""

    def _record(self, monkeypatch, turn_id: str) -> None:
        from guru.domain import ledger, pricing
        monkeypatch.setattr(session, 'turn_id', turn_id)
        ledger.record_call(adapter='A', model='claude-haiku-4-5',
                           usage=pricing.Usage(590, 10, 410, 0),
                           seconds=1.0, phase='step')
        ledger.record_call(adapter='A', model='claude-sonnet-5',
                           usage=pricing.Usage(100, 50), seconds=1.0,
                           phase='step')

    def test_turn_line_formats_the_turn(self, monkeypatch, fake_repo) -> None:
        monkeypatch.setattr(config, 'LEDGER_TURN_LINE', True)
        self._record(monkeypatch, 'turn-x')
        line = cli._turn_line('turn-x')
        assert line.startswith('turn: $0.00 · 2 calls · haiku, sonnet')
        assert line.endswith('cache 37%')             # 410 of 1100
        assert cli._turn_line('other') == ''
        assert cli._turn_line('') == ''

    def test_off_switch(self, monkeypatch, fake_repo) -> None:
        monkeypatch.setattr(config, 'LEDGER_TURN_LINE', False)
        self._record(monkeypatch, 'turn-y')
        assert cli._turn_line('turn-y') == ''
        assert cli._session_line() == ''

    def test_session_line_covers_every_turn(self, monkeypatch,
                                            fake_repo) -> None:
        monkeypatch.setattr(config, 'LEDGER_TURN_LINE', True)
        self._record(monkeypatch, 'turn-1')
        self._record(monkeypatch, 'turn-2')
        assert cli._session_line().startswith('session: $')
        assert '4 calls' in cli._session_line()

    def test_cost_omitted_when_ledger_disabled(self, monkeypatch) -> None:
        from guru.domain import ledger
        monkeypatch.setattr(config, 'LEDGER_TURN_LINE', True)
        monkeypatch.setattr(config, 'LEDGER_ENABLED', False)
        ledger.set_repository(None)
        self._record(monkeypatch, 'turn-z')
        assert cli._turn_line('turn-z') == \
            'turn: 2 calls · haiku, sonnet · cache 37%'


class TestToolsCommand:
    """/tools prints the last turn's tool events as a table."""

    def _events(self, rid: str) -> list:
        return [
            {'run_id': rid, 'turn_id': 'turn1', 'tool': 'read_file',
             'args': {'path': 'guru/cli.py'}, 'seconds': 0.0123,
             'produced_bytes': 9000, 'shown_bytes': 400, 'denied': '',
             'ok': True},
            {'run_id': rid, 'turn_id': 'turn1', 'tool': 'write_file',
             'args': {'path': 'x.py', 'content': 'c' * 300},
             'seconds': 0.5, 'produced_bytes': 40, 'shown_bytes': 40,
             'denied': 'mode', 'ok': False},
            {'run_id': rid, 'turn_id': 'other', 'tool': 'web_fetch',
             'args': {'url': 'https://ghost.example'}, 'seconds': 1.0,
             'produced_bytes': 1, 'shown_bytes': 1, 'denied': '',
             'ok': True},
            {'run_id': 'old', 'turn_id': 'turn1', 'tool': 'list_dir',
             'args': {'path': 'stale'}, 'seconds': 1.0,
             'produced_bytes': 1, 'shown_bytes': 1, 'denied': '',
             'ok': True}]

    def test_prints_last_turn_events(self, monkeypatch, capsys) -> None:
        from guru.domain import ledger
        repo = _RowsRepo({'tool_events': self._events(ledger.RUN_ID)})
        ledger.set_repository(repo)
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        monkeypatch.setattr(session, 'turn_id', 'turn1')
        try:
            cli._tools_command()
        finally:
            ledger.set_repository(None)
        out = capsys.readouterr().out
        assert 'read_file' in out and 'guru/cli.py' in out
        assert 'write_file' in out and 'mode' in out
        assert 'ghost' not in out and 'stale' not in out
        assert '400' in out and '9000' in out           # shown/produced
        assert 'c' * 300 not in out                     # args head only

    def test_no_events_in_turn(self, monkeypatch, capsys) -> None:
        from guru.domain import ledger
        repo = _RowsRepo({'tool_events': []})
        ledger.set_repository(repo)
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        monkeypatch.setattr(session, 'turn_id', 'turn1')
        try:
            cli._tools_command()
        finally:
            ledger.set_repository(None)
        assert 'no tool calls' in capsys.readouterr().out.lower()

    def test_without_repository(self, capsys) -> None:
        from guru.domain import ledger
        ledger.set_repository(None)
        cli._tools_command()
        assert 'ledger' in capsys.readouterr().out.lower()

    def test_format_table(self) -> None:
        text = cli._format_tool_events(self._events('r')[:2])
        lines = text.splitlines()
        assert lines[0].split()[:2] == ['tool', 'args']
        assert any(ln.startswith('read_file') for ln in lines)
        assert '0.01' in text and '0.50' in text


class TestLoadToolsPolicyFailClosed:
    """cli.load_tools_policy: absent file -> default; present but invalid or
    unreadable -> every registry tool disabled (fail closed)."""

    def test_absent_file_is_the_default_policy(self, tmp_path,
                                               monkeypatch) -> None:
        from guru.domain import tools
        monkeypatch.setattr(config, 'TOOLS_POLICY_PATH',
                            tmp_path / 'tools.toml')
        pol = cli.load_tools_policy()
        assert pol == tools.ToolsPolicy()
        assert tools.is_enabled('read_file')

    def test_invalid_file_disables_every_registry_tool(
            self, tmp_path, monkeypatch, capsys, caplog) -> None:
        from guru.domain import tools
        p = tmp_path / 'tools.toml'
        p.write_text('[tools]\nenable = ["x"]\n', encoding='utf-8')
        monkeypatch.setattr(config, 'TOOLS_POLICY_PATH', p)
        with caplog.at_level('WARNING', logger='guru'):
            pol = cli.load_tools_policy()
        assert pol.enabled == set()
        assert pol.disabled == set(tools.TOOL_REGISTRY)
        assert any(str(p) in r.getMessage() for r in caplog.records)
        assert 'disabled' in capsys.readouterr().out.lower()
        tools.set_policy(pol)
        try:
            assert all(not tools.is_enabled(n) for n in tools.TOOL_REGISTRY)
            assert tools.is_enabled('search_tools')
        finally:
            tools.set_policy(None)

    def test_unreadable_file_fails_closed(self, tmp_path,
                                          monkeypatch) -> None:
        from guru.domain import tools
        d = tmp_path / 'tools.toml'
        d.mkdir()
        monkeypatch.setattr(config, 'TOOLS_POLICY_PATH', d)
        pol = cli.load_tools_policy()
        assert pol.disabled == set(tools.TOOL_REGISTRY)
