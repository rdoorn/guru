"""Unit tests for guru's pure helper functions across the package."""
import json
from pathlib import Path
from types import SimpleNamespace

from guru import cli, config, session, skills, ui
from guru.adapters import anthropic as anth
from guru.adapters import litellm as lite
from guru.adapters.ollama import OllamaAdapter
from guru.domain import conversation, files, tools


class TestAgentManager:
    """Tests for the multi-viewport agent model."""

    def test_starts_with_main(self) -> None:
        from guru.agents import AgentManager
        m = AgentManager()
        assert m.active.title == 'main'
        assert m.tabs() == [(True, 'main')]

    def test_add_and_switch(self) -> None:
        from guru.agents import AgentManager
        m = AgentManager()
        m.add('research')
        assert [t for _, t in m.tabs()] == ['main', 'research']
        m.switch(1)
        assert m.active.title == 'research'
        m.switch(1)                     # wraps
        assert m.active.title == 'main'
        m.switch(-1)
        assert m.active.title == 'research'

    def test_append_and_text(self) -> None:
        from guru.agents import Agent
        a = Agent(id='x')
        a.append('one')
        a.append('two')
        assert a.text == 'one\ntwo'

    def test_agents_are_hashable_by_identity(self) -> None:
        from guru.agents import Agent
        a, b = Agent(id='x'), Agent(id='x')
        # Usable as dict keys (join barriers) and distinct despite same id.
        d = {a: 1, b: 2}
        assert len(d) == 2 and d[a] == 1 and a != b


class TestHumanCtx:
    """Tests for cli._human_ctx formatting."""

    def test_kilo(self) -> None:
        assert cli._human_ctx(65536) == '64k'
        assert cli._human_ctx(4096) == '4k'

    def test_mega(self) -> None:
        assert cli._human_ctx(1024 * 1024) == '1M'


class TestDomainOf:
    """Tests for config.domain_of."""

    def test_strips_scheme_and_lowercases(self) -> None:
        assert config.domain_of('https://Example.COM/path') == 'example.com'

    def test_strips_port(self) -> None:
        assert config.domain_of('http://example.com:8443/x') == 'example.com'

    def test_bare_host_without_scheme(self) -> None:
        assert config.domain_of('example.com/path') == 'example.com'

    def test_subdomain_preserved(self) -> None:
        assert config.domain_of('https://api.github.com') == 'api.github.com'


class TestProjectMemoryDir:
    """The project memory dir lives under ./.guru/memory."""

    def test_points_into_project_guru_dir(self) -> None:
        assert config.PROJECT_MEMORY_DIR == Path.cwd() / '.guru' / 'memory'


class TestBuildSystemPrompt:
    """Tests for config.build_system_prompt assembly."""

    def test_appends_global_and_local(self, tmp_path, monkeypatch) -> None:
        global_md = tmp_path / 'GURU.md'
        local_md = tmp_path / '.GURU.md'
        global_md.write_text('GLOBAL RULES', encoding='utf-8')
        local_md.write_text('LOCAL RULES', encoding='utf-8')
        monkeypatch.setattr(config, 'GURU_MD_PATH', global_md)
        monkeypatch.setattr(config, 'PROJECT_GURU_MD', local_md)

        prompt = config.build_system_prompt()

        assert config.SYSTEM_PROMPT.strip() in prompt
        assert 'GLOBAL RULES' in prompt
        assert 'LOCAL RULES' in prompt
        assert prompt.index('GLOBAL RULES') < prompt.index('LOCAL RULES')

    def test_missing_files_are_skipped(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'GURU_MD_PATH', tmp_path / 'nope.md')
        monkeypatch.setattr(config, 'PROJECT_GURU_MD', tmp_path / 'nope2.md')
        assert config.build_system_prompt() == config.SYSTEM_PROMPT.strip()


class TestMessageToDict:
    """Tests for conversation.message_to_dict normalisation."""

    def test_plain_dict_user_message(self) -> None:
        msg = {'role': 'user', 'content': 'hello'}
        assert conversation.message_to_dict(msg) == {
            'role': 'user', 'content': 'hello'}

    def test_tool_message_keeps_tool_name(self) -> None:
        msg = {'role': 'tool', 'tool_name': 'web_fetch', 'content': 'x'}
        assert conversation.message_to_dict(msg)['tool_name'] == 'web_fetch'

    def test_none_content_becomes_empty_string(self) -> None:
        msg = {'role': 'assistant', 'content': None}
        assert conversation.message_to_dict(msg)['content'] == ''

    def test_pydantic_like_object_via_model_dump(self) -> None:
        class FakeMessage:
            def model_dump(self) -> dict:
                return {'role': 'assistant', 'content': 'hi'}

        assert conversation.message_to_dict(FakeMessage()) == {
            'role': 'assistant', 'content': 'hi'}


class TestFirstUserMessage:
    """Tests for conversation._first_user_message."""

    def test_returns_first_user_content(self, tmp_path) -> None:
        path = tmp_path / 'c.memory'
        path.write_text(json.dumps([
            {'role': 'assistant', 'content': 'ignored'},
            {'role': 'user', 'content': 'the question'},
        ]), encoding='utf-8')
        assert conversation._first_user_message(path) == 'the question'

    def test_truncates_long_message(self, tmp_path) -> None:
        path = tmp_path / 'c.memory'
        path.write_text(
            json.dumps([{'role': 'user', 'content': 'x' * 100}]),
            encoding='utf-8')
        result = conversation._first_user_message(path)
        assert result.endswith('…')
        assert len(result) == 61

    def test_unreadable_file(self, tmp_path) -> None:
        assert conversation._first_user_message(
            tmp_path / 'missing.memory') == '(unreadable)'


class TestMatchTools:
    """Tests for tools._match_tools ranking."""

    def test_github_query_ranks_github_tool_first(self) -> None:
        ranked = tools._match_tools('get latest github release version')
        assert ranked[0] == 'fetch_github_releases'

    def test_fetch_query_ranks_web_fetch_first(self) -> None:
        assert tools._match_tools('fetch a webpage url')[0] == 'web_fetch'


class _FakeInfo:
    def __init__(self, modelinfo: dict, parameters: str) -> None:
        self.modelinfo = modelinfo
        self.parameters = parameters


class TestResolveContextWindow:
    """Tests for OllamaAdapter._resolve_context_window."""

    def _adapter(self) -> OllamaAdapter:
        return OllamaAdapter()

    def test_uses_modelfile_num_ctx_capped_at_ceiling(
            self, monkeypatch) -> None:
        info = _FakeInfo(
            {'general.architecture': 'qwen3', 'qwen3.context_length': 40960},
            'num_ctx 32768\ntemperature 0.6')
        monkeypatch.setattr(
            'guru.adapters.ollama.ollama.show', lambda m: info)
        monkeypatch.setattr(session, 'num_ctx_override', 0)
        assert self._adapter()._resolve_context_window('m') == (32768, 40960)

    def test_defaults_when_modelfile_has_no_num_ctx(self, monkeypatch) -> None:
        info = _FakeInfo(
            {'general.architecture': 'llama', 'llama.context_length': 8192},
            'temperature 0.7')
        monkeypatch.setattr(
            'guru.adapters.ollama.ollama.show', lambda m: info)
        monkeypatch.setattr(session, 'num_ctx_override', 0)
        num_ctx, ceiling = self._adapter()._resolve_context_window('m')
        assert num_ctx == config.DEFAULT_NUM_CTX
        assert ceiling == 8192

    def test_cli_override_wins_but_is_capped(self, monkeypatch) -> None:
        info = _FakeInfo(
            {'general.architecture': 'qwen3', 'qwen3.context_length': 40960},
            'num_ctx 8192')
        monkeypatch.setattr(
            'guru.adapters.ollama.ollama.show', lambda m: info)
        monkeypatch.setattr(session, 'num_ctx_override', 100000)
        num_ctx, _ = self._adapter()._resolve_context_window('m')
        assert num_ctx == 40960

    def test_show_failure_falls_back(self, monkeypatch) -> None:
        def _boom(m: str) -> None:
            raise RuntimeError('no server')

        monkeypatch.setattr('guru.adapters.ollama.ollama.show', _boom)
        monkeypatch.setattr(session, 'num_ctx_override', 0)
        num_ctx, ceiling = self._adapter()._resolve_context_window('m')
        assert num_ctx == config.DEFAULT_NUM_CTX
        assert ceiling == 0


class TestGpuAutoFit:
    """GPU auto-fit default: KV math, budget clamp, and the default policy."""

    def _adapter(self) -> OllamaAdapter:
        return OllamaAdapter()

    def test_kv_bytes_per_token(self, monkeypatch) -> None:
        info = SimpleNamespace(modelinfo={
            'general.architecture': 'qwen3',
            'qwen3.block_count': 40,
            'qwen3.attention.head_count': 40,
            'qwen3.attention.head_count_kv': 8,
            'qwen3.embedding_length': 5120,
        })
        monkeypatch.setattr(
            'guru.adapters.ollama.ollama.show', lambda m: info)
        # head_dim = 5120/40 = 128; 2*40*8*128*2.0 = 163840
        assert self._adapter()._kv_bytes_per_token('m') == 163840

    def test_kv_bytes_missing_metadata_is_zero(self, monkeypatch) -> None:
        info = SimpleNamespace(modelinfo={'general.architecture': 'qwen3'})
        monkeypatch.setattr(
            'guru.adapters.ollama.ollama.show', lambda m: info)
        assert self._adapter()._kv_bytes_per_token('m') == 0

    def test_max_gpu_ctx_math(self, monkeypatch) -> None:
        a = self._adapter()
        monkeypatch.setattr(a, '_total_gpu_bytes', lambda: 24 * 1024 ** 3)
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 163840)
        monkeypatch.setattr(a, '_weight_bytes', lambda m: 9 * 1024 ** 3)
        assert a._max_gpu_ctx('m', 262144) == 63488      # not capped
        assert a._max_gpu_ctx('m', 40960) == 40960       # capped at ceiling

    def test_max_gpu_ctx_zero_when_no_budget(self, monkeypatch) -> None:
        a = self._adapter()
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 163840)
        monkeypatch.setattr(a, '_weight_bytes', lambda m: 0)
        monkeypatch.setattr(a, '_total_gpu_bytes', lambda: 0)
        assert a._max_gpu_ctx('m', 40960) == 0

    def test_max_gpu_ctx_zero_when_weights_exceed(self, monkeypatch) -> None:
        a = self._adapter()
        monkeypatch.setattr(a, '_total_gpu_bytes', lambda: 20 * 1024 ** 3)
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 163840)
        monkeypatch.setattr(a, '_weight_bytes', lambda m: 19 * 1024 ** 3)
        # budget 16G, weights 19G -> avail < 0 -> 0 (not the 2k floor)
        assert a._max_gpu_ctx('m', 40960) == 0

    def test_calibrated_ctx_from_measurement(self, monkeypatch) -> None:
        a = self._adapter()
        # weights 15e9, kv 163840/token; high probe spills to an 18e9 budget.
        measured = {
            2048: (15335544320, 15335544320),     # fits (vram == size)
            32768: (20368709120, 18000000000),    # spills -> budget 18e9
        }
        monkeypatch.setattr(a, '_measure_at', lambda ctx: measured[int(ctx)])
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 163840)
        # (18e9*0.95 - 15e9)/163840 -> 12288 after rounding to 1024
        assert a._calibrated_ctx('m', 262144) == 12288

    # sizes chosen so the probe deltas give kv = 100000 bytes/token exactly:
    #   (8_072_000_000 - 5_000_000_000) / (32768 - 2048) = 100000
    _FIT = {2048: (5_000_000_000, 5_000_000_000),
            32768: (8_072_000_000, 8_072_000_000)}      # both 100% GPU

    def test_calibrated_ctx_both_fit_ceiling_probe_fits(self, monkeypatch):
        a = self._adapter()
        measured = dict(self._FIT)
        measured[131072] = (13_000_000_000, 13_000_000_000)   # ceiling fits
        monkeypatch.setattr(a, '_measure_at', lambda ctx: measured[int(ctx)])
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 100000)
        monkeypatch.setattr(a, '_total_gpu_bytes', lambda: 10 ** 12)  # huge
        # ceiling probe fits -> use the full ceiling, not the 32k probe cap
        assert a._calibrated_ctx('m', 131072) == 131072

    def test_calibrated_ctx_extend_probe_spills(self, monkeypatch):
        a = self._adapter()
        measured = dict(self._FIT)
        measured[131072] = (30_000_000_000, 18_000_000_000)   # spills
        monkeypatch.setattr(a, '_measure_at', lambda ctx: measured[int(ctx)])
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 100000)
        monkeypatch.setattr(a, '_total_gpu_bytes', lambda: 10 ** 12)
        # weights = 5e9 - 2048*100000 = 4_795_200_000
        # (18e9*0.95 - weights)/100000 = 123048 -> round down to 122880
        assert a._calibrated_ctx('m', 131072) == 122880

    def test_calibrated_ctx_no_budget_keeps_probe(self, monkeypatch):
        a = self._adapter()
        measured = dict(self._FIT)
        monkeypatch.setattr(a, '_measure_at', lambda ctx: measured[int(ctx)])
        monkeypatch.setattr(a, '_kv_bytes_per_token', lambda m: 100000)
        monkeypatch.setattr(a, '_total_gpu_bytes', lambda: 0)   # unknown
        assert a._calibrated_ctx('m', 262144) == 32768

    def test_calibrated_ctx_zero_when_measure_fails(self, monkeypatch):
        a = self._adapter()
        monkeypatch.setattr(a, '_measure_at', lambda ctx: (0, 0))
        assert a._calibrated_ctx('m', 262144) == 0

    def test_default_ctx_prefers_stored(self, monkeypatch) -> None:
        a = self._adapter()
        monkeypatch.setattr(config, 'load_model_ctx', lambda: {'m': 16384})
        assert a._default_ctx('m', 4096, 40960) == 16384
        assert a._default_ctx('m', 4096, 8192) == 8192   # capped at ceiling

    def test_default_ctx_autofit_when_no_stored(self, monkeypatch) -> None:
        a = self._adapter()
        monkeypatch.setattr(config, 'load_model_ctx', lambda: {})
        monkeypatch.setattr(a, '_max_gpu_ctx', lambda m, c: 32768)
        assert a._default_ctx('m', 4096, 40960) == 32768

    def test_default_ctx_falls_back_to_resolved(self, monkeypatch) -> None:
        a = self._adapter()
        monkeypatch.setattr(config, 'load_model_ctx', lambda: {})
        monkeypatch.setattr(a, '_max_gpu_ctx', lambda m, c: 0)
        assert a._default_ctx('m', 4096, 40960) == 4096

    def test_list_models_reports_ceiling(self, monkeypatch) -> None:
        models = SimpleNamespace(models=[
            SimpleNamespace(model='qwen3:14b', size=9_000_000_000)])
        monkeypatch.setattr(
            'guru.adapters.ollama.ollama.list', lambda: models)
        a = self._adapter()
        monkeypatch.setattr(
            a, '_resolve_context_window', lambda m: (4096, 40960))
        monkeypatch.setattr(a, '_param_size', lambda m: '14B')
        infos = a.list_models()
        assert infos[0].context_window == 40960


class TestModelCtxStore:
    """Per-model context persistence (~/.guru/model_ctx.json)."""

    def _isolate(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'GURU_HOME', tmp_path)
        monkeypatch.setattr(
            config, 'MODEL_CTX_PATH', tmp_path / 'model_ctx.json')

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch) -> None:
        self._isolate(tmp_path, monkeypatch)
        config.save_model_ctx('qwen3:14b', 32768)
        config.save_model_ctx('devstral', 65536)
        assert config.load_model_ctx() == {
            'qwen3:14b': 32768, 'devstral': 65536}

    def test_save_ignores_empty(self, tmp_path, monkeypatch) -> None:
        self._isolate(tmp_path, monkeypatch)
        config.save_model_ctx('', 100)
        config.save_model_ctx('m', 0)
        assert config.load_model_ctx() == {}

    def test_load_missing_returns_empty(self, tmp_path, monkeypatch) -> None:
        self._isolate(tmp_path, monkeypatch)
        assert config.load_model_ctx() == {}


class TestStreamingCancel:
    """The streamed turn accumulates chunks and aborts on cancel_requested."""

    def _adapter(self) -> OllamaAdapter:
        return OllamaAdapter()

    def _chunk(self, content='', tool_calls=None, pin=0, ein=0):
        return SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
            prompt_eval_count=pin, eval_count=ein)

    def _bind(self, monkeypatch, a) -> None:
        monkeypatch.setattr(a, '_supports_thinking', lambda m: False)
        monkeypatch.setattr(session, 'model', 'm')
        monkeypatch.setattr(session, 'messages', [])
        monkeypatch.setattr(session, 'active_tools', [])
        monkeypatch.setattr(session, 'session_in', 0)
        monkeypatch.setattr(session, 'session_out', 0)

    def test_accumulates_content_and_counts(self, monkeypatch) -> None:
        a = self._adapter()
        self._bind(monkeypatch, a)
        monkeypatch.setattr(session, 'cancel_requested', False)

        def fake(*args, **kw):
            yield self._chunk('Hel')
            yield self._chunk('lo', pin=12, ein=5)
        monkeypatch.setattr('guru.adapters.ollama.ollama.chat', fake)
        msg = a._collect_response()
        assert msg.content == 'Hello' and not msg.tool_calls
        assert session.session_in == 12 and session.session_out == 5

    def test_returns_none_and_closes_on_cancel(self, monkeypatch) -> None:
        a = self._adapter()
        self._bind(monkeypatch, a)
        monkeypatch.setattr(session, 'cancel_requested', True)
        closed = {'v': False}

        def fake(*args, **kw):
            try:
                while True:
                    yield self._chunk('x')
            finally:
                closed['v'] = True
        monkeypatch.setattr('guru.adapters.ollama.ollama.chat', fake)
        assert a._collect_response() is None
        assert closed['v'] is True        # stream closed -> generation aborted


class TestGroupMessages:
    """Tests for conversation.group_messages turn-grouping."""

    def test_splits_at_each_user_message(self) -> None:
        msgs = [
            {'role': 'user', 'content': 'q1'},
            {'role': 'assistant', 'content': 'a1'},
            {'role': 'tool', 'content': 't1'},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 'a2'},
        ]
        groups = conversation.group_messages(msgs)
        assert len(groups) == 2
        assert len(groups[0]) == 3
        assert len(groups[1]) == 2


class TestEstimateTokens:
    """Tests for conversation.estimate_tokens."""

    def test_four_chars_per_token(self) -> None:
        assert conversation.estimate_tokens(
            [{'role': 'user', 'content': 'x' * 40}]) == 10


class TestCompactMessages:
    """Tests for conversation.compact_messages eviction path."""

    def test_evicts_old_tool_output_without_summary(
            self, monkeypatch) -> None:
        class _Adapter:
            def summarise(self, transcript: str) -> str:
                raise AssertionError('summary should not be called')

        monkeypatch.setattr(session, 'adapter', _Adapter())
        monkeypatch.setattr(config, 'KEEP_RECENT_GROUPS', 1)
        monkeypatch.setattr(session, 'num_ctx', 100)
        monkeypatch.setattr(session, 'messages', [
            {'role': 'system', 'content': 'SYS'},
            {'role': 'user', 'content': 'q1'},
            {'role': 'assistant', 'content': 'a1'},
            {'role': 'tool', 'tool_name': 'web_fetch', 'content': 'X' * 2000},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 'a2'},
        ])
        conversation.compact_messages(force=False)

        result = session.messages
        assert result[0]['content'] == 'SYS'
        tool_msg = next(m for m in result if m.get('role') == 'tool')
        assert 'evicted' in tool_msg['content']
        assert result[-1]['content'] == 'a2'


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


class TestFileTools:
    """Tests for the filesystem tools and the directory allow-list gate."""

    def _only(self, monkeypatch, *dirs) -> None:
        """Make ALLOWED_READ_DIRS contain exactly ``dirs`` for this test."""
        monkeypatch.setattr(
            config, 'ALLOWED_READ_DIRS',
            {str(Path(d).resolve()) for d in dirs})

    def test_list_dir_shows_perms_and_size(self, monkeypatch) -> None:
        self._only(monkeypatch, Path.cwd())
        out = files.list_dir('guru')
        assert 'session.py' in out and '644' in out

    def test_read_file_range_is_line_numbered(self, monkeypatch) -> None:
        self._only(monkeypatch, Path.cwd())
        out = files.read_file('guru/session.py', '1-3')
        assert 'lines 1-3 of' in out
        assert '\n     1\t' in out

    def test_read_file_bad_range(self, monkeypatch) -> None:
        self._only(monkeypatch, Path.cwd())
        assert 'Invalid line range' in files.read_file('guru/session.py', 'x')

    def test_read_file_caps_large_files(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        big = tmp_path / 'big.txt'
        big.write_text('\n'.join(str(i) for i in range(1, 501)) + '\n')
        out = files.read_file(str(big))
        assert 'lines 1-400 of 500' in out
        assert 'showing first 400 of 500' in out

    def test_read_file_refuses_binary(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        b = tmp_path / 'b.bin'
        b.write_bytes(b'\x00\x01\x02data')
        assert 'binary' in files.read_file(str(b))

    def test_list_tree_skips_noise_dirs(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / '.git').mkdir()
        (tmp_path / '.git' / 'INSIDE_GIT').write_text('y')
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'a.py').write_text('z')
        out = files.list_tree(str(tmp_path), '3')
        assert '*skip' in out
        assert 'INSIDE_GIT' not in out   # noise dir not descended
        assert 'src/a.py' in out         # normal dir descended, flat relpath

    def test_cwd_is_not_auto_allowed(self, monkeypatch) -> None:
        # A fresh project: nothing pre-allowed, so even cwd must be approved.
        self._only(monkeypatch)                       # empty allow-list
        files.set_path_asker(lambda d: False)
        try:
            assert 'denied' in files.list_dir('.').lower()
        finally:
            files.set_path_asker(None)

    def test_gate_denies_outside(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch)
        files.set_path_asker(lambda d: False)
        try:
            assert 'denied' in files.list_dir(str(tmp_path)).lower()
        finally:
            files.set_path_asker(None)

    def test_gate_approves_and_persists(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch)
        saved: list = []
        monkeypatch.setattr(config, 'persist_read_dir', saved.append)
        files.set_path_asker(lambda d: True)
        (tmp_path / 'f.txt').write_text('hello\n')
        try:
            out = files.list_dir(str(tmp_path))
            resolved = str(tmp_path.resolve())
            assert 'f.txt' in out
            assert resolved in config.ALLOWED_READ_DIRS
            assert saved == [resolved]
        finally:
            files.set_path_asker(None)

    def test_parse_range(self) -> None:
        assert files._parse_range('', 500) == (1, 400)
        assert files._parse_range('10-20', 500) == (10, 20)
        assert files._parse_range('bad', 10) == (None, None)
        assert files._parse_range('5-3', 10) == (None, None)
        assert files._parse_range('1-9999', 50) == (1, 50)

    def test_search_code_finds_matches(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / 'a.py').write_text('def foo():\n    return 1\n')
        (tmp_path / 'b.py').write_text('x = foo()\n')
        out = files.search_code('foo', str(tmp_path))
        assert 'a.py:1:' in out and 'b.py:1:' in out

    def test_search_code_regex(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / 'a.py').write_text('def   foo():\n')
        assert 'a.py:1:' in files.search_code(r'def\s+foo', str(tmp_path))

    def test_search_code_invalid_regex_falls_back(
            self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / 'c.py').write_text('foo(bar)\n')
        # '(' is not a valid regex -> literal search
        assert 'c.py:1:' in files.search_code('(', str(tmp_path))

    def test_search_code_no_match(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / 'd.py').write_text('hello\n')
        assert 'No matches' in files.search_code('zzz', str(tmp_path))

    def test_search_code_skips_noise_dirs(
            self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch, tmp_path)
        (tmp_path / '.git').mkdir()
        (tmp_path / '.git' / 'x.py').write_text('TOKEN\n')
        (tmp_path / 'src.py').write_text('TOKEN\n')
        out = files.search_code('TOKEN', str(tmp_path))
        assert 'src.py:1:' in out and '.git' not in out

    def test_search_code_gate_denies(self, tmp_path, monkeypatch) -> None:
        self._only(monkeypatch)              # empty allow-list
        files.set_path_asker(lambda d: False)
        try:
            assert 'denied' in files.search_code('x', str(tmp_path)).lower()
        finally:
            files.set_path_asker(None)


class TestSpecsFor:
    """tools.specs_for builds tool specs without session routing."""

    def test_search_tools_always(self) -> None:
        assert [s['name'] for s in tools.specs_for(set(), False)] == [
            'search_tools', 'use_skill']

    def test_can_spawn_and_activated(self) -> None:
        names = {s['name'] for s in tools.specs_for({'web_fetch'}, True)}
        assert {'search_tools', 'spawn', 'check', 'join', 'web_fetch'} <= names


class TestContextBreakdown:
    """conversation.context_breakdown categorises resident context tokens."""

    def test_categorises_and_totals(self) -> None:
        msgs = [
            {'role': 'system', 'content': 'x' * 40},    # 10 tokens
            {'role': 'user', 'content': 'y' * 20},      # 5 tokens
            {'role': 'assistant', 'content': 'z' * 8},  # 2 tokens
        ]
        bd = conversation.context_breakdown(msgs, set(), False)
        assert bd['sys'] == 10 and bd['in'] == 5 and bd['out'] == 2
        assert bd['tools'] > 0        # search_tools schema always present
        assert bd['total'] == (bd['sys'] + bd['in'] + bd['out']
                               + bd['toolout'] + bd['tools'])

    def test_can_spawn_adds_schema_tokens(self) -> None:
        base = conversation.context_breakdown([], set(), False)['tools']
        more = conversation.context_breakdown([], set(), True)['tools']
        assert more > base            # spawn/check/join schemas add tokens


class TestApplyRetention:
    def _msgs(self):
        return [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'find the k8s setting'},
            {'role': 'assistant', 'content': ''},        # text-less step
            {'role': 'tool', 'tool_name': 'web_fetch',
             'content': 'X' * 9000},                      # large -> summarize
            {'role': 'tool', 'tool_name': 'search_code',
             'content': 'a.py:1: hit'},                   # keep
            {'role': 'assistant', 'content': 'the answer'},
        ]

    def test_large_web_summarized_small_kept(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'WEB_SUMMARIZE_OVER_CHARS', 6000)

        class _Ad:
            def summarise(self, t):
                return 'GIST'
        monkeypatch.setattr(session, 'adapter', _Ad())
        msgs = self._msgs()
        conversation.apply_retention(msgs)
        assert all(not (m.get('role') == 'assistant'
                        and not m.get('content')) for m in msgs)
        web = next(m for m in msgs if m.get('tool_name') == 'web_fetch')
        assert 'GIST' in web['content'] and len(web['content']) < 9000
        grep = next(m for m in msgs if m.get('tool_name') == 'search_code')
        assert grep['content'] == 'a.py:1: hit'

    def test_small_web_not_summarized(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'WEB_SUMMARIZE_OVER_CHARS', 6000)

        class _Ad:
            def summarise(self, t):
                raise AssertionError('should not summarise small output')
        monkeypatch.setattr(session, 'adapter', _Ad())
        msgs = [
            {'role': 'user', 'content': 'q'},
            {'role': 'tool', 'tool_name': 'web_fetch', 'content': 'short'},
        ]
        conversation.apply_retention(msgs)
        assert msgs[-1]['content'] == 'short'

    def test_large_read_file_outlined(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'OUTLINE_FILE_OVER_CHARS', 50)
        src = "def foo():\n    return 1\n" * 10
        header = "/tmp/m.py (lines 1-20 of 20, sha:abc):"
        body = "\n".join(f"{i:>6}\t{ln}"
                         for i, ln in enumerate(src.splitlines(), 1))
        msgs = [
            {'role': 'user', 'content': 'q'},
            {'role': 'tool', 'tool_name': 'read_file',
             'content': f"{header}\n{body}"},
        ]
        conversation.apply_retention(msgs)
        assert '[outline]' in msgs[-1]['content']
        assert 'return 1' not in msgs[-1]['content']


class TestWriteTools:
    """Write gate + write_file/edit_file + access modes."""

    def _write_allowed(self, monkeypatch, *dirs):
        monkeypatch.setattr(
            config, 'ALLOWED_WRITE_DIRS',
            {str(Path(d).resolve()) for d in dirs})
        monkeypatch.setattr(config, 'persist_write_dir', lambda d: None)

    def test_write_file_creates(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'x.txt'
        out = files.write_file(str(p), 'hello')
        assert p.read_text() == 'hello' and 'Wrote' in out

    def test_write_refused_in_read_only(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_READ_ONLY)
        self._write_allowed(monkeypatch, tmp_path)
        out = files.write_file(str(tmp_path / 'x.txt'), 'hi')
        assert 'read-only' in out and not (tmp_path / 'x.txt').exists()

    def test_read_allow_does_not_grant_write(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(
            config, 'ALLOWED_READ_DIRS', {str(tmp_path.resolve())})
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
        files.set_path_asker(lambda q: False)     # deny the write prompt
        try:
            out = files.write_file(str(tmp_path / 'z.txt'), 'hi')
            assert 'denied' in out.lower()
            assert not (tmp_path / 'z.txt').exists()
        finally:
            files.set_path_asker(None)

    def test_write_gate_uses_write_list_and_persists(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
        saved: list = []
        monkeypatch.setattr(config, 'persist_write_dir', saved.append)
        files.set_path_asker(lambda q: True)
        try:
            files.write_file(str(tmp_path / 'y.txt'), 'hi')
            resolved = str(tmp_path.resolve())
            assert resolved in config.ALLOWED_WRITE_DIRS
            assert saved == [resolved]
        finally:
            files.set_path_asker(None)

    def test_auto_mode_writes_without_prompt(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_AUTO)
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
        monkeypatch.setattr(config, 'persist_write_dir', lambda d: None)

        def boom(q):
            raise AssertionError('should not prompt in auto mode')
        files.set_path_asker(boom)
        try:
            files.write_file(str(tmp_path / 'a.txt'), 'hi')
            assert (tmp_path / 'a.txt').read_text() == 'hi'
        finally:
            files.set_path_asker(None)

    def test_edit_file_unique_replace(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'c.py'
        p.write_text('a = 1\nb = 2\n')
        sha = files._sha(p.read_text())
        out = files.edit_file(str(p), 'b = 2', 'b = 3', sha)
        assert p.read_text() == 'a = 1\nb = 3\n' and 'Edited' in out

    def test_edit_file_not_found_and_ambiguous(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'c.py'
        p.write_text('x\nx\n')
        sha = files._sha(p.read_text())
        assert 'not found' in files.edit_file(str(p), 'zzz', 'q', sha)
        assert 'appears 2 times' in files.edit_file(str(p), 'x', 'y', sha)

    def test_edit_file_sha_mismatch_refuses(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'c.py'
        p.write_text('a = 1\n')
        out = files.edit_file(str(p), 'a = 1', 'a = 2', 'deadbeef1234')
        assert 'sha mismatch' in out and p.read_text() == 'a = 1\n'

    def test_read_and_write_report_sha(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        monkeypatch.setattr(
            config, 'ALLOWED_READ_DIRS', {str(tmp_path.resolve())})
        p = tmp_path / 'c.py'
        assert 'sha:' in files.write_file(str(p), 'hello\n')
        out = files.read_file(str(p))
        assert 'sha:' in out
        # the sha from read_file is accepted by edit_file
        sha = out.split('sha:')[1].split(')')[0].split(':')[0].strip()
        edited = files.edit_file(str(p), 'hello', 'bye', sha)
        assert 'Edited' in edited and p.read_text() == 'bye\n'

    def test_delete_file_removes(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'd.txt'
        p.write_text('bye')
        out = files.delete_file(str(p))
        assert not p.exists() and 'Deleted' in out

    def test_delete_refused_in_read_only(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_READ_ONLY)
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'd.txt'
        p.write_text('bye')
        out = files.delete_file(str(p))
        assert 'read-only' in out and p.exists()

    def test_delete_refuses_directory_and_missing(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        self._write_allowed(monkeypatch, tmp_path)
        assert 'directory' in files.delete_file(str(tmp_path))
        assert 'No such file' in files.delete_file(str(tmp_path / 'nope'))

    def test_delete_uses_write_gate(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', set())
        p = tmp_path / 'd.txt'
        p.write_text('bye')
        files.set_path_asker(lambda q: False)         # deny
        try:
            out = files.delete_file(str(p))
            assert 'denied' in out.lower() and p.exists()
        finally:
            files.set_path_asker(None)


class TestFileShaLedger:
    """The file->sha ledger + its rendering into the system prompt."""

    def _write_allowed(self, monkeypatch, *dirs):
        monkeypatch.setattr(
            config, 'ALLOWED_WRITE_DIRS',
            {str(Path(d).resolve()) for d in dirs})
        monkeypatch.setattr(config, 'persist_write_dir', lambda d: None)

    def test_read_write_edit_populate_ledger(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(session, 'file_shas', {})
        self._write_allowed(monkeypatch, tmp_path)
        monkeypatch.setattr(
            config, 'ALLOWED_READ_DIRS', {str(tmp_path.resolve())})
        p = tmp_path / 'c.py'
        key = str(p.resolve())
        files.write_file(str(p), 'hello\n')
        assert session.file_shas[key] == files._sha('hello\n')
        files.read_file(str(p))
        assert session.file_shas[key] == files._sha('hello\n')
        files.edit_file(str(p), 'hello', 'bye', session.file_shas[key])
        assert session.file_shas[key] == files._sha('bye\n')

    def test_delete_forgets_sha(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(session, 'file_shas', {})
        self._write_allowed(monkeypatch, tmp_path)
        p = tmp_path / 'd.txt'
        files.write_file(str(p), 'bye')
        assert str(p.resolve()) in session.file_shas
        files.delete_file(str(p))
        assert str(p.resolve()) not in session.file_shas

    def test_ledger_caps_and_drops_oldest(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'file_shas', {})
        cap = files._SHA_LEDGER_CAP
        for i in range(cap + 3):
            files._remember_sha(Path(f'/tmp/f{i}'), f'sha{i}')
        assert len(session.file_shas) == cap
        assert '/tmp/f0' not in session.file_shas
        assert session.file_shas[f'/tmp/f{cap + 2}'] == f'sha{cap + 2}'

    def test_remember_moves_key_to_newest(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'file_shas', {})
        files._remember_sha(Path('/tmp/a'), '1')
        files._remember_sha(Path('/tmp/b'), '2')
        files._remember_sha(Path('/tmp/a'), '3')     # re-touch a
        assert list(session.file_shas) == ['/tmp/b', '/tmp/a']
        assert session.file_shas['/tmp/a'] == '3'

    def test_refresh_adds_block_and_is_idempotent(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'file_shas', {'/tmp/x.txt': 'abc123'})
        monkeypatch.setattr(
            session, 'messages',
            [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        body = session.messages[0]['content']
        assert body.startswith('BASE')
        assert '[open files]' in body and 'abc123' in body
        conversation.refresh_system_context()             # again
        assert session.messages[0]['content'].count('[open files]') == 1

    def test_refresh_strips_block_when_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'file_shas', {'/tmp/x.txt': 'abc123'})
        monkeypatch.setattr(
            session, 'messages',
            [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        session.file_shas.clear()
        conversation.refresh_system_context()
        assert session.messages[0]['content'] == 'BASE'

    def test_refresh_shows_relative_path(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        key = str((tmp_path / 'sub' / 'f.py').resolve())
        monkeypatch.setattr(session, 'file_shas', {key: 'deadbeef'})
        monkeypatch.setattr(
            session, 'messages',
            [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        body = session.messages[0]['content']
        assert 'sub/f.py (sha:deadbeef)' in body

    def test_refresh_noop_without_system_message(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'file_shas', {'/tmp/x': 'y'})
        monkeypatch.setattr(session, 'messages', [])
        conversation.refresh_system_context()             # must not raise
        assert session.messages == []


class TestSystemContext:
    """Catalog + role + skill overlays rendered onto messages[0]."""

    def _reg(self):
        return {
            'developer': skills.SkillEntry(
                'developer', 'role', 'dev', '', 'BE A DEV'),
            'code-review': skills.SkillEntry(
                'code-review', 'skill', 'review', '', 'REVIEW METHOD'),
        }

    def test_catalog_rendered_when_registry_present(
            self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(session, 'active_role', None)
        monkeypatch.setattr(session, 'active_skill', None)
        monkeypatch.setattr(
            session, 'messages', [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        body = session.messages[0]['content']
        assert body.startswith('BASE')
        assert 'Available specialists' in body and 'developer' in body

    def test_role_and_skill_bodies_rendered(self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(session, 'active_role', 'developer')
        monkeypatch.setattr(session, 'active_skill', 'code-review')
        monkeypatch.setattr(
            session, 'messages', [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        body = session.messages[0]['content']
        assert '[role: developer]' in body and 'BE A DEV' in body
        assert '[skill: code-review]' in body and 'REVIEW METHOD' in body

    def test_idempotent_across_calls(self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(session, 'active_role', 'developer')
        monkeypatch.setattr(session, 'active_skill', None)
        monkeypatch.setattr(
            session, 'messages', [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()
        conversation.refresh_system_context()
        body = session.messages[0]['content']
        assert body.count('[role: developer]') == 1
        assert body.startswith('BASE')

    def test_unknown_active_name_ignored(self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(session, 'active_role', 'nonexistent')
        monkeypatch.setattr(session, 'active_skill', None)
        monkeypatch.setattr(
            session, 'messages', [{'role': 'system', 'content': 'BASE'}])
        conversation.refresh_system_context()      # must not raise
        assert '[role:' not in session.messages[0]['content']


class TestAccessPromptDefaults:
    """Access prompts allow only on an explicit yes (Enter/y/yes); any other
    input, junk, escape-sequence text, or error denies."""

    def test_path_enter_allows(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: '')
        assert files._ask_path('/x') is True

    def test_path_y_and_yes_allow(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: 'y')
        assert files._ask_path('/x') is True
        monkeypatch.setattr('builtins.input', lambda *a: 'YES')
        assert files._ask_path('/x') is True

    def test_path_junk_denies(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: 'x')
        assert files._ask_path('/x') is False

    def test_path_escape_sequence_denies(self, monkeypatch) -> None:
        # The CSI-u text a modifyOtherKeys terminal emits for Ctrl+C must NOT
        # be read as approval (the reported bug).
        monkeypatch.setattr('builtins.input', lambda *a: '\x1b[27;5;99~')
        assert files._ask_path('/x') is False

    def test_domain_junk_denies(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: 'maybe')
        assert tools._ask_domain('x.com') is False

    def test_path_explicit_no_denies(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: 'no')
        assert files._ask_path('/x') is False

    def test_path_error_denies(self, monkeypatch) -> None:
        def boom(*a):
            raise EOFError
        monkeypatch.setattr('builtins.input', boom)
        assert files._ask_path('/x') is False

    def test_domain_enter_allows(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: '  ')
        assert tools._ask_domain('x.com') is True

    def test_domain_no_denies(self, monkeypatch) -> None:
        monkeypatch.setattr('builtins.input', lambda *a: 'n')
        assert tools._ask_domain('x.com') is False

    def test_domain_error_denies(self, monkeypatch) -> None:
        def boom(*a):
            raise KeyboardInterrupt
        monkeypatch.setattr('builtins.input', boom)
        assert tools._ask_domain('x.com') is False


class TestOptionalToolParams:
    """Optional params are excluded from adapter 'required' schemas."""

    def test_anthropic_marks_optional_not_required(self) -> None:
        spec = {'name': 'read_file', 'description': 'd',
                'parameters': {'path': 'p', 'lines': 'l'},
                'optional': ['lines']}
        defn = anth.tool_defs([spec])[0]
        req = defn['input_schema']['required']
        assert req == ['path']

    def test_litellm_marks_optional_not_required(self) -> None:
        spec = {'name': 'read_file', 'description': 'd',
                'parameters': {'path': 'p', 'lines': 'l'},
                'optional': ['lines']}
        defn = lite.openai_tool_defs([spec])[0]
        req = defn['function']['parameters']['required']
        assert req == ['path']


class TestSpawnTool:
    """Tests for the spawn delegation tool and its handler injection."""

    def test_spawn_without_handler_reports_repl(self) -> None:
        tools.set_spawn_handler(None)
        out = tools.spawn('do something')
        assert 'not available in this mode' in out

    def test_spawn_with_handler_delegates(self) -> None:
        seen: list = []
        tools.set_spawn_handler(
            lambda t, r, s: seen.append(t) or f'ok:{t}')
        try:
            assert tools.spawn('research topic') == 'ok:research topic'
            assert seen == ['research topic']
        finally:
            tools.set_spawn_handler(None)

    def test_execute_tool_routes_spawn(self) -> None:
        seen: list = []
        tools.set_spawn_handler(lambda t, r, s: seen.append(t) or 'done')
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


class TestSessionRouting:
    """Tests for the per-context SessionState routing (parallel isolation)."""

    def test_use_and_reset(self) -> None:
        st = session.SessionState()
        token = session.use(st)
        session.model = 'bound-model'
        session.session_in += 7
        assert st.model == 'bound-model'
        assert st.session_in == 7
        session.reset(token)
        assert session.model != 'bound-model'

    def test_threads_are_isolated(self) -> None:
        import threading

        states = [session.SessionState() for _ in range(3)]
        errors: list = []
        start = threading.Barrier(len(states))

        def worker(idx: int) -> None:
            token = session.use(states[idx])
            try:
                start.wait()
                for _ in range(1000):
                    session.session_in += 1
                    session.model = f'model-{idx}'
                if (session.session_in != 1000
                        or session.model != f'model-{idx}'):
                    errors.append(idx)
            finally:
                session.reset(token)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(len(states))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert [s.session_in for s in states] == [1000, 1000, 1000]
        assert [s.model for s in states] == ['model-0', 'model-1', 'model-2']


class TestAdapterConfigRoundTrip:
    """Tests for config.save_adapter_configs / load_adapter_configs."""

    def test_round_trip_preserves_enable_and_fields(
            self, tmp_path, monkeypatch) -> None:
        path = tmp_path / 'adapters.toml'
        monkeypatch.setattr(config, 'ADAPTERS_PATH', path)
        monkeypatch.setattr(config, 'GURU_HOME', tmp_path)
        configs = [
            {'name': 'Ollama', 'type': 'ollama',
             'url': 'http://localhost:11434', 'enable': True},
            {'name': 'Anthropic Enterprise', 'type': 'anthropic',
             'auth': 'oauth', 'profile': 'guru', 'enable': False,
             'thinking': True},
        ]
        config.save_adapter_configs(configs)
        loaded = config.load_adapter_configs()
        assert loaded[0]['name'] == 'Ollama'
        assert loaded[0]['enable'] is True
        assert loaded[1]['enable'] is False
        assert loaded[1]['auth'] == 'oauth'
        assert loaded[1]['profile'] == 'guru'
        assert loaded[1]['thinking'] is True


class TestSettings:
    """Tests for config.load_settings / save_settings."""

    def test_round_trip(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / '.guru' / 'settings.json'
        monkeypatch.setattr(config, 'PROJECT_SETTINGS_PATH', path)
        assert config.load_settings() == {}
        config.save_settings({'adapter': 'Claude Code', 'model': 'x'})
        loaded = config.load_settings()
        assert loaded['adapter'] == 'Claude Code'
        assert loaded['model'] == 'x'


class TestAnthropicTranslation:
    """Tests for the pure Anthropic translation helpers."""

    def test_system_merged_and_history_flattened(self) -> None:
        messages = [
            {'role': 'system', 'content': 'BASE'},
            {'role': 'system', 'content': 'SUMMARY'},
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': '',
             'tool_calls': [{'function': {'name': 'web_search',
                                          'arguments': {'query': 'x'}}}]},
            {'role': 'tool', 'tool_name': 'web_search', 'content': 'results'},
        ]
        system, native = anth.to_anthropic_messages(messages)
        assert system == 'BASE\n\nSUMMARY'
        assert native[0] == {'role': 'user', 'content': 'hello'}
        assert native[1] == {'role': 'assistant', 'content': '(used tools)'}
        assert native[2]['role'] == 'user'
        assert 'web_search result' in native[2]['content']

    def test_tool_defs_schema(self) -> None:
        specs = [{
            'name': 'web_fetch',
            'description': 'fetch a page',
            'parameters': {'url': 'the url'},
        }]
        defs = anth.tool_defs(specs)
        assert defs[0]['name'] == 'web_fetch'
        schema = defs[0]['input_schema']
        assert schema['properties']['url']['type'] == 'string'
        assert schema['required'] == ['url']

    def test_neutral_assistant_with_and_without_tools(self) -> None:
        plain = anth.neutral_assistant('hi', [])
        assert plain == {'role': 'assistant', 'content': 'hi'}
        with_tools = anth.neutral_assistant(
            '', [('web_search', {'query': 'x'})])
        assert with_tools['tool_calls'][0]['function']['name'] == 'web_search'


class TestLiteLLMTranslation:
    """Tests for the pure LiteLLM/OpenAI translation helpers."""

    def test_messages_flattened(self) -> None:
        messages = [
            {'role': 'system', 'content': 'SYS'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': '',
             'tool_calls': [{'function': {'name': 'web_search',
                                          'arguments': {'query': 'x'}}}]},
            {'role': 'tool', 'tool_name': 'web_search', 'content': 'results'},
        ]
        out = lite.to_openai_messages(messages)
        assert out[0] == {'role': 'system', 'content': 'SYS'}
        assert out[1] == {'role': 'user', 'content': 'hi'}
        assert out[2] == {'role': 'assistant', 'content': '(used tools)'}
        assert out[3]['role'] == 'user'
        assert 'web_search result' in out[3]['content']

    def test_tool_defs_openai_shape(self) -> None:
        specs = [{
            'name': 'web_fetch',
            'description': 'fetch a page',
            'parameters': {'url': 'the url'},
        }]
        defs = lite.openai_tool_defs(specs)
        assert defs[0]['type'] == 'function'
        fn = defs[0]['function']
        assert fn['name'] == 'web_fetch'
        assert fn['parameters']['properties']['url']['type'] == 'string'
        assert fn['parameters']['required'] == ['url']

    def test_base_url_trailing_slash_stripped(self) -> None:
        a = lite.LiteLLMAdapter(base_url='https://proxy/v1/')
        assert a.base_url == 'https://proxy/v1'

    def test_inline_api_key_used_when_env_unset(self, monkeypatch) -> None:
        monkeypatch.delenv('MY_LLM_KEY', raising=False)
        a = lite.LiteLLMAdapter(
            base_url='https://p/v1', api_key_env='MY_LLM_KEY',
            api_key='sk-123')
        assert a._key() == 'sk-123'

    def test_env_key_overrides_inline(self, monkeypatch) -> None:
        monkeypatch.setenv('MY_LLM_KEY', 'sk-env')
        a = lite.LiteLLMAdapter(
            base_url='https://p/v1', api_key_env='MY_LLM_KEY',
            api_key='sk-123')
        assert a._key() == 'sk-env'


class TestSkillsRegistry:
    """Frontmatter parsing, seeding, token cap, lookup."""

    def test_parse_entry_reads_frontmatter_and_body(self) -> None:
        text = (
            "---\n"
            "name: developer\n"
            "kind: role\n"
            "description: General coding\n"
            "---\n"
            "Be a developer.\n")
        e = skills.parse_entry(text)
        assert e.name == 'developer' and e.kind == 'role'
        assert e.description == 'General coding'
        assert e.body == 'Be a developer.'

    def test_parse_entry_rejects_missing_frontmatter(self) -> None:
        assert skills.parse_entry("no frontmatter here") is None

    def test_body_is_capped(self) -> None:
        big = "x" * (skills._MAX_BODY_CHARS + 500)
        text = f"---\nname: t\nkind: skill\ndescription: d\n---\n{big}"
        e = skills.parse_entry(text)
        assert len(e.body) <= skills._MAX_BODY_CHARS + len(skills._TRUNC)
        assert e.body.endswith(skills._TRUNC)

    def test_seed_writes_missing_then_load(self, tmp_path) -> None:
        skills.seed_defaults(tmp_path, reset=False)
        reg = skills.load_registry(tmp_path)
        assert 'developer' in reg and reg['developer'].kind == 'role'
        assert 'code-review' in reg and reg['code-review'].kind == 'skill'

    def test_seed_does_not_overwrite_user_edit(self, tmp_path) -> None:
        skills.seed_defaults(tmp_path, reset=False)
        f = tmp_path / 'developer.md'
        f.write_text(f.read_text() + "\nUSER EDIT\n", encoding='utf-8')
        skills.seed_defaults(tmp_path, reset=False)          # again
        assert 'USER EDIT' in f.read_text()

    def test_reset_overwrites_defaults_not_extras(self, tmp_path) -> None:
        skills.seed_defaults(tmp_path, reset=False)
        dev = tmp_path / 'developer.md'
        dev.write_text("---\nname: developer\nkind: role\n"
                       "description: d\n---\nMANGLED\n", encoding='utf-8')
        extra = tmp_path / 'my-role.md'
        extra.write_text("---\nname: my-role\nkind: role\n"
                         "description: mine\n---\nkeep\n", encoding='utf-8')
        skills.seed_defaults(tmp_path, reset=True)
        assert 'MANGLED' not in dev.read_text()
        assert extra.read_text().endswith('keep\n')

    def test_names_by_kind(self, tmp_path) -> None:
        reg = {'a': skills.SkillEntry('a', 'role', 'd', '', 'b'),
               'c': skills.SkillEntry('c', 'skill', 'd', '', 'b')}
        assert skills.names(reg, 'role') == ['a']
        assert skills.names(reg, 'skill') == ['c']

    def test_setup_populates_registry(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'GURU_SKILLS_DIR', tmp_path / 'skills')
        skills.setup(reset=True)
        assert 'architect' in skills.REGISTRY
        skills.REGISTRY.clear()          # leave global clean for other tests


class TestSkillTools:
    def _reg(self):
        return {
            'developer': skills.SkillEntry(
                'developer', 'role', 'dev', '', 'BE A DEV'),
            'code-review': skills.SkillEntry(
                'code-review', 'skill', 'review', '', 'REVIEW'),
        }

    def test_use_skill_sets_active_skill(self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'active_skill', None)
        out = tools.use_skill('code-review')
        assert session.active_skill == 'code-review' and 'code-review' in out

    def test_use_skill_rejects_role_or_unknown(self, monkeypatch) -> None:
        monkeypatch.setattr(skills, 'REGISTRY', self._reg())
        monkeypatch.setattr(session, 'active_skill', None)
        assert 'No skill' in tools.use_skill('developer')   # wrong kind
        assert 'No skill' in tools.use_skill('nope')
        assert session.active_skill is None

    def test_spawn_passes_role_and_skill(self, monkeypatch) -> None:
        seen = {}

        def handler(task, role, skill):
            seen.update(task=task, role=role, skill=skill)
            return "ok"
        tools.set_spawn_handler(handler)
        try:
            tools.spawn('do it', role='developer', skill='code-review')
            assert seen == {'task': 'do it', 'role': 'developer',
                            'skill': 'code-review'}
        finally:
            tools.set_spawn_handler(None)

    def test_spawn_defaults_role_skill_empty(self, monkeypatch) -> None:
        seen = {}
        tools.set_spawn_handler(
            lambda task, role, skill: seen.update(
                role=role, skill=skill) or "ok")
        try:
            tools.spawn('t')
            assert seen == {'role': '', 'skill': ''}
        finally:
            tools.set_spawn_handler(None)


class TestContextSettings:
    """Global ~/.guru/settings.toml overrides context thresholds."""

    def test_load_missing_returns_empty(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            config, 'GLOBAL_SETTINGS_PATH', tmp_path / 'settings.toml')
        assert config.load_context_settings() == {}

    def test_load_reads_context_section(
            self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'settings.toml'
        p.write_text(
            "[context]\nweb_summarize_over_chars = 1234\n", encoding='utf-8')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', p)
        assert config.load_context_settings() == {
            'web_summarize_over_chars': 1234}

    def test_load_invalid_returns_empty(
            self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'settings.toml'
        p.write_text("not = valid = toml", encoding='utf-8')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', p)
        assert config.load_context_settings() == {}

    def test_apply_overrides_defaults(
            self, tmp_path, monkeypatch) -> None:
        p = tmp_path / 'settings.toml'
        p.write_text(
            "[context]\nweb_summarize_over_chars = 999\n"
            "outline_file_over_chars = 111\n", encoding='utf-8')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH', p)
        monkeypatch.setattr(config, 'WEB_SUMMARIZE_OVER_CHARS', 6000)
        monkeypatch.setattr(config, 'OUTLINE_FILE_OVER_CHARS', 8000)
        config._apply_settings()
        assert config.WEB_SUMMARIZE_OVER_CHARS == 999
        assert config.OUTLINE_FILE_OVER_CHARS == 111


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


class TestFocusedSummary:
    def test_recent_question_scans_backwards(self) -> None:
        msgs = [
            {'role': 'system', 'content': 's'},
            {'role': 'user', 'content': 'first'},
            {'role': 'assistant', 'content': 'a'},
            {'role': 'user', 'content': 'the question'},
            {'role': 'tool', 'tool_name': 'web_fetch', 'content': 'x'},
        ]
        assert conversation._recent_question(msgs, 4) == 'the question'

    def test_summarize_relevant_uses_adapter(self, monkeypatch) -> None:
        seen = {}

        class _Ad:
            def summarise(self, transcript):
                seen['t'] = transcript
                return 'RELEVANT BITS'
        monkeypatch.setattr(session, 'adapter', _Ad())
        out = conversation._summarize_relevant(
            'k8s setting?', 'huge content', 'web_fetch')
        assert 'RELEVANT BITS' in out
        assert 'web_fetch summary' in out and 'k8s setting?' in out
        assert 'huge content' in seen['t'] and 'k8s setting?' in seen['t']

    def test_summarize_relevant_falls_back_on_error(
            self, monkeypatch) -> None:
        class _Ad:
            def summarise(self, transcript):
                raise RuntimeError('down')
        monkeypatch.setattr(session, 'adapter', _Ad())
        out = conversation._summarize_relevant('q', 'A' * 100, 'web_fetch')
        assert 'truncated' in out and out.count('A') > 0


class TestOutlineCode:
    def _read_output(self, path, source):
        lines = source.splitlines()
        header = f"{path} (lines 1-{len(lines)} of {len(lines)}, sha:abc):"
        body = "\n".join(f"{i:>6}\t{ln}" for i, ln in enumerate(lines, 1))
        return f"{header}\n{body}"

    def test_outline_python_keeps_signatures_drops_bodies(self) -> None:
        src = (
            "import os\n"
            "\n"
            "def foo(a, b=2) -> int:\n"
            "    '''Add things.'''\n"
            "    return a + b\n"
            "\n"
            "class C:\n"
            "    def method(self, x):\n"
            "        '''Do it.'''\n"
            "        return x\n")
        out = conversation._outline_code(
            self._read_output('/tmp/m.py', src))
        assert 'def foo(a, b=2) -> int:' in out
        assert 'Add things.' in out
        assert 'class C' in out
        assert 'def method(self, x):' in out
        assert 'return a + b' not in out      # body dropped
        assert '/tmp/m.py' in out             # header kept

    def test_outline_non_python_truncates(self) -> None:
        src = "\n".join(f"line {i}" for i in range(200))
        out = conversation._outline_code(
            self._read_output('/tmp/notes.txt', src))
        assert '/tmp/notes.txt' in out
        assert len(out) < len(src)            # shrunk

    def test_outline_unparseable_python_falls_back(self) -> None:
        src = "def broken(:\n    pass\nimport sys\n"
        out = conversation._outline_code(
            self._read_output('/tmp/b.py', src))
        # regex fallback keeps def/import lines even when AST fails
        assert 'import sys' in out or 'def broken' in out
