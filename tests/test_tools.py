"""Tests for the tool registry, activation, and delegation tools."""
import pytest

from guru import config, session, ui
from guru.domain import files, tools


class TestMatchTools:
    """Tests for tools._match_tools ranking."""

    def test_github_query_ranks_github_tool_first(self) -> None:
        ranked = tools._match_tools('get latest github release version')
        assert ranked[0] == 'fetch_github_releases'

    def test_fetch_query_ranks_web_fetch_first(self) -> None:
        assert tools._match_tools('fetch a webpage url')[0] == 'web_fetch'


class TestActiveSpecs:
    """Tests for tools.active_specs."""

    def test_search_tools_always_present(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'active_tool_names', set())
        specs = tools.active_specs()
        assert [s['name'] for s in specs] == ['search_tools', 'use_skill']

    def test_activated_tool_included(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'active_tool_names', {'web_fetch'})
        names = [s['name'] for s in tools.active_specs()]
        assert 'search_tools' in names and 'web_fetch' in names


class TestSpecsFor:
    """tools.specs_for builds tool specs without session routing."""

    def test_search_tools_always(self) -> None:
        assert [s['name'] for s in tools.specs_for(set(), False)] == [
            'search_tools', 'use_skill']

    def test_can_spawn_and_activated(self) -> None:
        names = {s['name'] for s in tools.specs_for({'web_fetch'}, True)}
        assert {'search_tools', 'spawn', 'check', 'join', 'web_fetch'} <= names


class TestToolSizeFormat:
    """Human byte/token formatting for the per-tool result size line."""

    def test_fmt_bytes(self) -> None:
        assert ui._fmt_bytes(80) == '80b'
        assert ui._fmt_bytes(1536) == '1.5k'
        assert ui._fmt_bytes(51200) == '50k'
        assert ui._fmt_bytes(5 * 1024 * 1024) == '5.0M'

    def test_fmt_size_includes_tokens(self) -> None:
        assert ui._fmt_size(40) == '40b · ~10 tok'
        assert '~2.0k tok' in ui._fmt_size(8000)   # 8000/4 = 2000 tok

    def test_execute_tool_reports_size_and_returns_result(
            self, monkeypatch) -> None:
        sizes = []
        monkeypatch.setattr(ui, 'note_tool_result', sizes.append)
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        tools.set_spawn_handler(lambda t, r, s, k, c: 'RESULT-9')
        try:
            out = tools.execute_tool('spawn', {'task': 't'})
        finally:
            tools.set_spawn_handler(None)
        assert out == 'RESULT-9'
        assert sizes == [len('RESULT-9')]


class TestSpawnTool:
    """Tests for the spawn delegation tool and its handler injection."""

    def test_spawn_without_handler_reports_repl(self) -> None:
        tools.set_spawn_handler(None)
        out = tools.spawn('do something')
        assert 'not available in this mode' in out

    def test_spawn_with_handler_delegates(self) -> None:
        seen: list = []
        tools.set_spawn_handler(
            lambda t, r, s, k, c: seen.append(t) or f'ok:{t}')
        try:
            assert tools.spawn('research topic') == 'ok:research topic'
            assert seen == ['research topic']
        finally:
            tools.set_spawn_handler(None)

    def test_execute_tool_routes_spawn(self) -> None:
        seen: list = []
        tools.set_spawn_handler(
            lambda t, r, s, k, c: seen.append(t) or 'done')
        try:
            assert tools.execute_tool('spawn', {'task': 'go'}) == 'done'
            assert seen == ['go']
        finally:
            tools.set_spawn_handler(None)

    def test_delegation_specs_gated_by_can_spawn(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'active_tool_names', set())
        monkeypatch.setattr(session, 'can_spawn', False)
        names = [s['name'] for s in tools.active_specs()]
        assert not ({'spawn', 'check', 'join'} & set(names))
        monkeypatch.setattr(session, 'can_spawn', True)
        names = [s['name'] for s in tools.active_specs()]
        assert {'spawn', 'check', 'join'} <= set(names)

    def test_reset_active_tools_honours_can_spawn(self) -> None:
        capable = session.SessionState()
        capable.can_spawn = True
        token = session.use(capable)
        try:
            tools.reset_active_tools()
            for fn in (tools.spawn, tools.check, tools.join,
                       tools.search_tools):
                assert fn in capable.active_tools
        finally:
            session.reset(token)
        plain = session.SessionState()
        token = session.use(plain)
        try:
            tools.reset_active_tools()
            assert tools.spawn not in plain.active_tools
            assert tools.check not in plain.active_tools
            assert tools.join not in plain.active_tools
        finally:
            session.reset(token)


class TestCollectTools:
    """Tests for the non-blocking check/join tools and their injection."""

    def test_check_without_handler_reports_repl(self) -> None:
        tools.set_check_handler(None)
        assert '--classic' in tools.check('all')

    def test_join_without_handler_reports_repl(self) -> None:
        tools.set_join_handler(None)
        assert '--classic' in tools.join('agent2')

    def test_check_and_join_delegate_and_route(self) -> None:
        seen: list = []
        tools.set_check_handler(lambda t: seen.append(('check', t)) or 'c')
        tools.set_join_handler(lambda t: seen.append(('join', t)) or 'j')
        try:
            assert tools.execute_tool('check', {'target': 'all'}) == 'c'
            assert tools.execute_tool('join', {'targets': 'a b'}) == 'j'
            assert seen == [('check', 'all'), ('join', 'a b')]
        finally:
            tools.set_check_handler(None)
            tools.set_join_handler(None)


class TestRetainPolicy:
    def test_web_tools_summarize(self) -> None:
        assert tools.retain_policy('web_search') == 'summarize'
        assert tools.retain_policy('web_fetch') == 'summarize'

    def test_read_file_outline(self) -> None:
        assert tools.retain_policy('read_file') == 'outline'

    def test_local_tools_keep(self) -> None:
        for name in ('search_code', 'list_dir', 'list_tree',
                     'write_file', 'edit_file', 'delete_file'):
            assert tools.retain_policy(name) == 'keep'

    def test_unknown_keeps(self) -> None:
        assert tools.retain_policy('nope') == 'keep'
        assert tools.retain_policy('') == 'keep'


class TestInitialTools:
    """The pre-activated core toolset lets weak models skip search_tools."""

    def test_initial_tools_includes_core(self, monkeypatch) -> None:
        monkeypatch.setattr(
            config, 'PREACTIVATE_TOOLS', ['read_file', 'search_code'])
        base, names = tools.initial_tools(can_spawn=False)
        assert names == {'read_file', 'search_code'}
        assert tools.search_tools in base and tools.use_skill in base
        assert tools.spawn not in base
        assert tools.TOOL_REGISTRY['read_file']['fn'] in base

    def test_initial_tools_spawn_and_empty_core(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'PREACTIVATE_TOOLS', [])
        base, names = tools.initial_tools(can_spawn=True)
        assert tools.spawn in base and names == set()

    def test_flat_activates_entire_registry(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'FLAT_TOOLS', True)
        monkeypatch.setattr(config, 'PREACTIVATE_TOOLS', [])
        base, names = tools.initial_tools(can_spawn=False)
        assert names == set(tools.TOOL_REGISTRY)     # every registry tool
        assert len(names) > len(['read_file', 'search_code'])


class TestStruggleCounters:
    """execute_tool counts tool errors and edit_file sha mismatches."""

    def _arm(self, monkeypatch) -> None:
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)

    def test_tool_error_counted(self, monkeypatch) -> None:
        self._arm(monkeypatch)

        def boom(**kw):
            raise ValueError('bad args')
        monkeypatch.setitem(tools.TOOL_REGISTRY, 'boom', {'fn': boom})
        out = tools.execute_tool('boom', {})
        assert out.startswith('Tool error:')
        assert session.struggle['tool_errors'] == 1
        assert session.struggle['sha_mismatches'] == 0

    def test_successful_tool_not_counted(self, monkeypatch) -> None:
        self._arm(monkeypatch)
        monkeypatch.setitem(tools.TOOL_REGISTRY, 'ok', {'fn': lambda: 'fine'})
        assert tools.execute_tool('ok', {}) == 'fine'
        assert session.struggle['tool_errors'] == 0

    def test_sha_mismatch_counted(self, tmp_path, monkeypatch) -> None:
        self._arm(monkeypatch)
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS',
                            {str(tmp_path.resolve())})
        monkeypatch.setattr(config, 'persist_write_dir', lambda d: None)
        p = tmp_path / 'c.py'
        p.write_text('hello\n', encoding='utf-8')
        out = tools.execute_tool('edit_file', {
            'path': str(p), 'old': 'hello', 'new': 'bye', 'sha': 'stale'})
        assert out.startswith('sha mismatch:')
        assert session.struggle['sha_mismatches'] == 1
        assert session.struggle['tool_errors'] == 0
        good = files._sha('hello\n')
        out = tools.execute_tool('edit_file', {
            'path': str(p), 'old': 'hello', 'new': 'bye', 'sha': good})
        assert out.startswith('Edited')
        assert session.struggle['sha_mismatches'] == 1


class TestDomainAutoGrant:
    """ensure_domain_allowed honours config.AUTO_GRANT in auto mode."""

    def test_auto_grant_false_denies_and_does_not_persist(
            self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_AUTO)
        monkeypatch.setattr(config, 'AUTO_GRANT', False)
        monkeypatch.setattr(config, 'ALLOWED_DOMAINS', set())
        saved: list = []
        monkeypatch.setattr(config, 'persist_domain', saved.append)
        asked: list = []

        def deny(q):
            asked.append(q)
            return False
        tools.set_domain_asker(deny)
        try:
            assert tools.ensure_domain_allowed('Example.com') is False
        finally:
            tools.set_domain_asker(None)
        assert asked and 'example.com' in asked[0]
        assert config.ALLOWED_DOMAINS == set() and saved == []

    def test_auto_grant_true_grants_and_persists(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_AUTO)
        monkeypatch.setattr(config, 'AUTO_GRANT', True)
        monkeypatch.setattr(config, 'ALLOWED_DOMAINS', set())
        saved: list = []
        monkeypatch.setattr(config, 'persist_domain', saved.append)

        def boom(q):
            raise AssertionError('should not prompt')
        tools.set_domain_asker(boom)
        try:
            assert tools.ensure_domain_allowed('example.com') is True
        finally:
            tools.set_domain_asker(None)
        assert config.ALLOWED_DOMAINS == {'example.com'}
        assert saved == ['example.com']


class TestControllerTools:
    """Controller mode (Task 4.5): only spawn/check/join/use_skill."""

    def test_initial_tools_controller(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'PREACTIVATE_TOOLS', ['read_file'])
        base, names = tools.initial_tools(can_spawn=True, controller=True)
        assert base == [tools.spawn, tools.check, tools.join,
                        tools.use_skill]
        assert names == set()

    def test_initial_tools_non_controller_unchanged(self, monkeypatch):
        monkeypatch.setattr(config, 'PREACTIVATE_TOOLS', ['read_file'])
        base, names = tools.initial_tools(can_spawn=True)
        assert tools.search_tools in base and names == {'read_file'}

    def test_specs_for_controller(self) -> None:
        names = [s['name'] for s in
                 tools.specs_for({'read_file', 'web_fetch'}, True,
                                 controller=True)]
        assert names == ['spawn', 'check', 'join', 'use_skill']

    def test_active_specs_honours_session_controller(self, monkeypatch):
        monkeypatch.setattr(session, 'active_tool_names', {'read_file'})
        monkeypatch.setattr(session, 'can_spawn', True)
        monkeypatch.setattr(session, 'controller', True)
        names = [s['name'] for s in tools.active_specs()]
        assert names == ['spawn', 'check', 'join', 'use_skill']
        monkeypatch.setattr(session, 'controller', False)
        names = [s['name'] for s in tools.active_specs()]
        assert 'search_tools' in names and 'read_file' in names

    def test_reset_active_tools_honours_controller(self, monkeypatch):
        monkeypatch.setattr(session, 'active_tools', [])
        monkeypatch.setattr(session, 'active_tool_names', set())
        monkeypatch.setattr(session, 'can_spawn', True)
        monkeypatch.setattr(session, 'controller', True)
        tools.reset_active_tools()
        assert session.active_tools == [tools.spawn, tools.check, tools.join,
                                        tools.use_skill]
        assert session.active_tool_names == set()

    def test_execute_tool_hides_other_tools_from_a_controller(
            self, monkeypatch) -> None:
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        monkeypatch.setattr(session, 'controller', True)
        monkeypatch.setattr(session, 'active_tool_names', set())
        assert tools.execute_tool('read_file', {'path': 'x'}) == \
            'Unknown tool: read_file'
        assert tools.execute_tool('search_tools', {'query': 'web'}) == \
            'Unknown tool: search_tools'
        assert session.active_tool_names == set()       # nothing activated
        monkeypatch.setattr(tools, 'use_skill', lambda name: f'ok:{name}')
        assert tools.execute_tool('use_skill', {'name': 's'}) == 'ok:s'

    def test_spawn_spec_has_optional_labels(self) -> None:
        spec = tools._SPAWN_SPEC
        assert {'kind', 'complexity'} <= set(spec['parameters'])
        assert {'kind', 'complexity'} <= set(spec['optional'])
        assert 'debug' in spec['parameters']['kind']
        assert 'hard' in spec['parameters']['complexity']


class _Marker:
    """Scanner that flags every ``SECRET``."""

    def scan(self, text: str) -> list:
        from guru.domain.policy import Finding
        return [Finding('marker', i, i + 6, 'SECRET')
                for i in range(len(text)) if text.startswith('SECRET', i)]


class TestRemoteRedaction:
    """execute_tool redacts results bound for a remote adapter (Task 4.6)."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        from types import SimpleNamespace
        from guru.domain import policy
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        monkeypatch.setattr(tools, 'use_skill',
                            lambda name: f'token SECRET for {name}')
        monkeypatch.setattr(config, 'SECRET_SCAN', True)
        monkeypatch.setattr(session, 'adapter',
                            SimpleNamespace(name='Remote', remote=True))
        policy.set_scanner(_Marker())
        yield
        policy.set_scanner(None)

    def test_remote_result_is_redacted_and_counted(self) -> None:
        out = tools.execute_tool('use_skill', {'name': 'x'})
        assert out == 'token [REDACTED:marker] for x'
        assert session.struggle['redactions'] == 1

    def test_local_adapter_sees_raw_result(self, monkeypatch) -> None:
        from types import SimpleNamespace
        monkeypatch.setattr(session, 'adapter',
                            SimpleNamespace(name='Ollama', remote=False))
        out = tools.execute_tool('use_skill', {'name': 'x'})
        assert out == 'token SECRET for x'
        assert session.struggle['redactions'] == 0

    def test_scan_off_sees_raw_result(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'SECRET_SCAN', False)
        out = tools.execute_tool('use_skill', {'name': 'x'})
        assert out == 'token SECRET for x'
        assert session.struggle['redactions'] == 0

    def test_clean_result_untouched(self, monkeypatch) -> None:
        monkeypatch.setattr(tools, 'use_skill', lambda name: 'all clear')
        assert tools.execute_tool('use_skill', {'name': 'x'}) == 'all clear'
        assert session.struggle['redactions'] == 0


class TestToolsPolicy:
    """tools.is_enabled / set_policy: .guru/tools.toml gating (A3)."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        yield
        tools.set_policy(None)

    def test_default_policy_enables_everything(self) -> None:
        tools.set_policy(None)
        assert tools.is_enabled('read_file') and tools.is_enabled('web_fetch')
        assert tools.active_policy() == tools.ToolsPolicy()

    def test_disabled_wins_over_enabled(self) -> None:
        tools.set_policy(tools.ToolsPolicy(enabled={'read_file'},
                                           disabled={'read_file'}))
        assert tools.is_enabled('read_file') is False

    def test_enabled_list_is_an_allowlist(self) -> None:
        tools.set_policy(tools.ToolsPolicy(enabled={'read_file'}))
        assert tools.is_enabled('read_file') is True
        assert tools.is_enabled('web_fetch') is False

    def test_always_on_tools_are_never_gated(self) -> None:
        tools.set_policy(tools.ToolsPolicy(
            enabled={'read_file'},
            disabled={'search_tools', 'use_skill', 'spawn', 'check', 'join'}))
        for name in ('search_tools', 'use_skill', 'spawn', 'check', 'join'):
            assert tools.is_enabled(name) is True


class TestToolEvents:
    """execute_tool writes one tool_events row per call (A2, A3)."""

    @pytest.fixture(autouse=True)
    def _quiet(self, monkeypatch, fake_repo):
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        monkeypatch.setattr(session, 'controller', False)
        monkeypatch.setattr(session, 'turn_id', 'turnX')
        self.repo = fake_repo
        yield
        tools.set_policy(None)

    def _event(self) -> dict:
        from guru.domain import ledger
        ledger.flush()
        [row] = self.repo.stream('tool_events')
        return row

    def test_registry_tool_event(self, monkeypatch) -> None:
        monkeypatch.setitem(tools.TOOL_REGISTRY, 'read_file', {
            **tools.TOOL_REGISTRY['read_file'],
            'fn': lambda path, lines='': 'x' * 40})
        out = tools.execute_tool('read_file', {'path': 'a.py'})
        assert out == 'x' * 40
        row = self._event()
        assert row['tool'] == 'read_file' and row['turn_id'] == 'turnX'
        assert row['ok'] is True and row['denied'] == ''
        assert row['produced_bytes'] == 40 and row['shown_bytes'] == 40
        assert row['files_touched'] == ['a.py']
        assert row['args'] == {'path': 'a.py'}
        assert row['seconds'] >= 0

    def test_unknown_tool_event(self) -> None:
        out = tools.execute_tool('teleport', {'to': 'mars'})
        assert out.startswith('Unknown tool')
        row = self._event()
        assert row['tool'] == 'teleport' and row['ok'] is False
        assert row['files_touched'] == []
        assert row['produced_bytes'] == len(out)

    def test_tool_error_is_not_ok(self, monkeypatch) -> None:
        def boom(path, lines=''):
            raise RuntimeError('nope')
        monkeypatch.setitem(tools.TOOL_REGISTRY, 'read_file', {
            **tools.TOOL_REGISTRY['read_file'], 'fn': boom})
        tools.execute_tool('read_file', {'path': 'a.py'})
        row = self._event()
        assert row['ok'] is False and row['denied'] == ''

    def test_disabled_tool_is_refused_and_denied(self, monkeypatch) -> None:
        called = []
        monkeypatch.setitem(tools.TOOL_REGISTRY, 'read_file', {
            **tools.TOOL_REGISTRY['read_file'],
            'fn': lambda path, lines='': called.append(path) or 'ran'})
        tools.set_policy(tools.ToolsPolicy(disabled={'read_file'}))
        out = tools.execute_tool('read_file', {'path': 'a.py'})
        assert out == "Tool 'read_file' is disabled by .guru/tools.toml"
        assert called == []
        row = self._event()
        assert row['denied'] == 'policy' and row['ok'] is False

    def test_read_only_refusal_is_denied_mode(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_READ_ONLY)
        out = tools.execute_tool('write_file', {'path': '/tmp/x',
                                                'content': 'c'})
        assert out.startswith('Refused: read-only mode')
        row = self._event()
        assert row['denied'] == 'mode' and row['files_touched'] == ['/tmp/x']

    def test_controller_refusal_is_denied(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'controller', True)
        tools.execute_tool('read_file', {'path': 'a.py'})
        row = self._event()
        assert row['denied'] == 'controller' and row['ok'] is False

    def test_shown_bytes_after_redaction(self, monkeypatch) -> None:
        from types import SimpleNamespace
        from guru.domain import policy
        monkeypatch.setattr(tools, 'use_skill',
                            lambda name: 'token SECRET for x')
        monkeypatch.setattr(config, 'SECRET_SCAN', True)
        monkeypatch.setattr(session, 'adapter',
                            SimpleNamespace(name='Remote', remote=True))
        policy.set_scanner(_Marker())
        try:
            out = tools.execute_tool('use_skill', {'name': 'x'})
        finally:
            policy.set_scanner(None)
        row = self._event()
        assert row['produced_bytes'] == len('token SECRET for x')
        assert row['shown_bytes'] == len(out) != row['produced_bytes']


class TestCodeVerbRegistry:
    """Plan B5: the audited coding verbs are registry tools with clear
    specs, are pre-activated where the plan says, dispatch through
    execute_tool and audit the files they touch."""

    VERBS = ('outline', 'find_symbol', 'run_tests', 'check_syntax', 'lint',
             'git_status', 'git_diff', 'apply_patch')

    @pytest.fixture(autouse=True)
    def _quiet(self, monkeypatch, fake_repo):
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        monkeypatch.setattr(session, 'controller', False)
        self.repo = fake_repo
        yield
        tools.set_policy(None)

    def _event(self) -> dict:
        from guru.domain import ledger
        ledger.flush()
        return self.repo.stream('tool_events')[-1]

    def test_entries_are_complete(self) -> None:
        from guru.domain import code, gitread, patch, quality
        fns = {'outline': code.outline, 'find_symbol': code.find_symbol,
               'run_tests': quality.run_tests,
               'check_syntax': quality.check_syntax, 'lint': quality.lint,
               'git_status': gitread.git_status, 'git_diff': gitread.git_diff,
               'apply_patch': patch.apply_patch}
        for name in self.VERBS:
            info = tools.TOOL_REGISTRY[name]
            assert info['fn'] is fns[name]
            assert len(info['description']) > 80 and info['tags']
            assert isinstance(info['parameters'], dict)
            for opt in info.get('optional', []):
                assert opt in info['parameters'], (name, opt)
        assert tools.TOOL_REGISTRY['run_tests']['optional'] == [
            'target', 'k', 'maxfail', 'detail']
        assert tools.TOOL_REGISTRY['git_status']['parameters'] == {}
        assert tools.TOOL_REGISTRY['apply_patch']['parameters'] == {
            'diff': tools.TOOL_REGISTRY['apply_patch']['parameters']['diff']}

    def test_retain_policies(self) -> None:
        for name in ('run_tests', 'lint', 'check_syntax', 'git_status',
                     'git_diff', 'outline', 'find_symbol', 'apply_patch'):
            assert tools.retain_policy(name) == 'keep', name

    def test_preactivated_set(self) -> None:
        for name in ('outline', 'find_symbol', 'run_tests', 'check_syntax'):
            assert name in config.PREACTIVATE_TOOLS
        for name in ('lint', 'git_status', 'git_diff', 'apply_patch'):
            assert name not in config.PREACTIVATE_TOOLS
        base, names = tools.initial_tools(can_spawn=False)
        assert {'outline', 'find_symbol', 'run_tests',
                'check_syntax'} <= names

    def test_search_tools_finds_the_verbs(self) -> None:
        assert tools._match_tools('run the tests')[0] == 'run_tests'
        assert tools._match_tools('apply a unified diff patch')[0] == \
            'apply_patch'
        assert tools._match_tools('outline file structure')[0] == 'outline'
        assert 'find_symbol' in tools._match_tools(
            'find symbol definition references')[:2]

    def test_execute_dispatches_and_audits(self, tmp_path,
                                           monkeypatch) -> None:
        monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', {str(tmp_path)})
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(files, '_show_change', lambda block: None)
        monkeypatch.chdir(tmp_path)
        files.set_path_asker(lambda q: False)
        try:
            (tmp_path / 'm.py').write_text('def f():\n    return 1\n')
            out = tools.execute_tool('outline', {'path': 'm.py'})
            assert 'L1-2 def f()' in out
            assert self._event()['files_touched'] == ['m.py']
            out = tools.execute_tool('check_syntax', {'path': 'm.py'})
            assert out.startswith('ok:')
            out = tools.execute_tool('find_symbol', {'name': 'f'})
            assert 'def: m.py:1 (def)' in out
            diff = ('--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n def f():\n'
                    '-    return 1\n+    return 2\n'
                    '--- /dev/null\n+++ b/n.py\n@@ -0,0 +1 @@\n+x = 1\n')
            out = tools.execute_tool('apply_patch', {'diff': diff})
            assert out.startswith('Applied patch:')
            row = self._event()
            assert row['files_touched'] == ['m.py', 'n.py']
            assert row['ok'] is True and row['denied'] == ''
            assert (tmp_path / 'm.py').read_text().endswith('return 2\n')
        finally:
            files.set_path_asker(None)

    def test_run_tests_audits_target(self, monkeypatch) -> None:
        from guru.domain import quality
        monkeypatch.setattr(quality, 'run_tests',
                            lambda target='', k='', maxfail=1, detail='':
                            f'ran {target}')
        monkeypatch.setitem(tools.TOOL_REGISTRY, 'run_tests', {
            **tools.TOOL_REGISTRY['run_tests'], 'fn': quality.run_tests})
        out = tools.execute_tool('run_tests', {'target': 'tests/test_x.py'})
        assert out == 'ran tests/test_x.py'
        assert self._event()['files_touched'] == ['tests/test_x.py']

    def test_apply_patch_read_only_is_denied_mode(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_READ_ONLY)
        out = tools.execute_tool('apply_patch', {
            'diff': '--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n'})
        assert out.startswith('Refused: read-only mode')
        row = self._event()
        assert row['denied'] == 'mode' and row['files_touched'] == ['x']

    def test_policy_can_disable_a_verb(self) -> None:
        tools.set_policy(tools.ToolsPolicy(disabled={'apply_patch'}))
        out = tools.execute_tool('apply_patch', {'diff': 'x'})
        assert out == "Tool 'apply_patch' is disabled by .guru/tools.toml"
        assert self._event()['denied'] == 'policy'

    def test_prompts_mention_the_verbs(self) -> None:
        assert 'run_tests' in config.CONTROLLER_HINT
        assert 'check_syntax' in config.CONTROLLER_HINT
        for text in (config.SYSTEM_PROMPT, config.DELEGATION_HINT):
            assert 'outline' in text and 'find_symbol' in text
            assert 'run_tests' in text


class TestDisabledToolsNotAdvertised:
    """A tool the project policy disables is neither preactivated, nor
    discoverable, nor described to the model — and still refused if the
    model calls it by name anyway (security review follow-up)."""

    @pytest.fixture(autouse=True)
    def _policy(self, monkeypatch, fake_repo):
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        monkeypatch.setattr(session, 'controller', False)
        monkeypatch.setattr(session, 'active_tool_names', set())
        monkeypatch.setattr(session, 'active_tools', [])
        tools.set_policy(tools.ToolsPolicy(disabled={'read_file'}))
        yield
        tools.set_policy(None)

    def test_not_preactivated(self, monkeypatch) -> None:
        monkeypatch.setattr(
            config, 'PREACTIVATE_TOOLS', ['read_file', 'search_code'])
        assert [n for n, _ in tools._core_tool_fns()] == ['search_code']
        _, names = tools.initial_tools(can_spawn=False)
        assert names == {'search_code'}

    def test_flat_mode_skips_disabled(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'FLAT_TOOLS', True)
        names = {n for n, _ in tools._core_tool_fns()}
        assert 'read_file' not in names and 'search_code' in names

    def test_activate_is_a_no_op(self) -> None:
        tools.activate('read_file')
        assert 'read_file' not in session.active_tool_names
        assert tools.TOOL_REGISTRY['read_file']['fn'] \
            not in session.active_tools
        tools.activate('search_code')
        assert 'search_code' in session.active_tool_names

    def test_search_tools_does_not_list_it(self) -> None:
        assert 'read_file' not in tools._match_tools('read a file')
        assert 'read_file' not in tools._match_tools('')      # fallback
        assert 'read_file' not in tools._match_tools('zzqqxx')  # no hits
        out = tools.search_tools('read the contents of a file')
        listed = [ln.strip() for ln in out.splitlines()
                  if ln.startswith('  ') and not ln.startswith('    ')]
        assert 'read_file' not in listed and 'search_code' in listed

    def test_specs_skip_it(self) -> None:
        names = [s['name'] for s in tools.specs_for(
            {'read_file', 'search_code'}, can_spawn=False)]
        assert 'read_file' not in names
        assert names[:2] == ['search_tools', 'use_skill']
        assert 'search_code' in names

    def test_execute_still_refuses(self, fake_repo) -> None:
        from guru.domain import ledger
        out = tools.execute_tool('read_file', {'path': 'a.py'})
        assert out == "Tool 'read_file' is disabled by .guru/tools.toml"
        ledger.flush()
        [row] = fake_repo.stream('tool_events')
        assert row['denied'] == 'policy'

    def test_allowlist_hides_the_rest(self) -> None:
        tools.set_policy(tools.ToolsPolicy(enabled={'search_code'}))
        assert tools._match_tools('read a file') == ['search_code']
        names = [s['name'] for s in tools.specs_for(
            set(tools.TOOL_REGISTRY), can_spawn=True)]
        assert set(names) == {'search_tools', 'use_skill', 'spawn',
                              'check', 'join', 'search_code'}


class TestRunnerDenialIsMode:
    """A verb surfacing the runner's ``Denied:`` text is recorded as a
    mode denial in tool_events."""

    def test_mode_denial_recognises_the_prefix(self) -> None:
        from guru.domain import procs
        text = f"{procs.DENIED_PREFIX} cwd '/x' is outside the allowed dirs"
        assert tools._mode_denial(text) is True
        assert tools._mode_denial(f'Refused: {text}') is True
        assert tools._mode_denial('Denied by nobody') is False
        assert tools._mode_denial('ok') is False

    def test_event_row_is_denied_mode(self, monkeypatch, fake_repo) -> None:
        from guru.domain import ledger, procs
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        monkeypatch.setattr(session, 'controller', False)
        monkeypatch.setitem(tools.TOOL_REGISTRY, 'run_tests', {
            **tools.TOOL_REGISTRY['run_tests'],
            'fn': lambda **kw: (f"Refused: {procs.DENIED_PREFIX} cwd '/x'"
                                ' is outside the allowed directories')})
        tools.execute_tool('run_tests', {'target': '/x'})
        ledger.flush()
        [row] = fake_repo.stream('tool_events')
        assert row['denied'] == 'mode' and row['ok'] is False
