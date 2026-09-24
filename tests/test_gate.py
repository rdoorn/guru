"""Tests for the sandbox quality gate (S3): the deterministic rules and
verdict mapping of ``guru.domain.gate``, ``patch.render``/``rebase``, the
LLM reviewer (``guru.judges.llm``) with a fake adapter, the ``llm:`` judge
spec, and ``decisions.decide_review``."""
import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from guru import config, session
from guru.domain import decisions, gate, ledger, patch, policy, routing
from guru.domain.policy import Finding
from guru.judges import llm
import guru.judges as judges
from guru.repositories.adapters import AdapterRegistry
from guru.repositories.settings import RoutingSettings, RungSpec
from tests.test_adapters_registry import FakeAdapter


def _diff(path: str, old: list, new: list, new_file: bool = False) -> str:
    """A one-hunk unified diff replacing ``old`` lines by ``new`` in
    ``path`` (git headers included)."""
    src = '/dev/null' if new_file else f'a/{path}'
    body = [f'-{ln}' for ln in old] + [f'+{ln}' for ln in new]
    return (f'diff --git a/{path} b/{path}\n--- {src}\n+++ b/{path}\n'
            f'@@ -1,{len(old)} +1,{len(new)} @@\n' + '\n'.join(body) + '\n')


GOOD_REVIEW = {'implements_task': 'yes', 'unrelated_changes': 'none',
               'weakens_tests': 'no', 'obfuscated': 'no', 'confidence': 0.9,
               'notes': 'Renames the helper.'}


class MarkerScanner:
    def scan(self, text):
        i = text.find('MARKER_SECRET')
        return [Finding('marker', i, i + 13, 'MARKER')] if i >= 0 else []


@pytest.fixture
def marker_scanner():
    policy.set_scanner(MarkerScanner())
    try:
        yield
    finally:
        policy.set_scanner(None)


# --- gate.rules --------------------------------------------------------------

class TestRules:
    def _kinds(self, flags):
        return sorted(f.kind for f in flags)

    def test_clean_diff_has_no_flags(self, tmp_path) -> None:
        d = _diff('pkg/mod.py', ['    return 1'], ['    return 2'])
        assert gate.rules(d, tmp_path) == []

    @pytest.mark.parametrize('line,label', [
        ('import subprocess', 'import subprocess'),
        ('from subprocess import run', 'import subprocess'),
        ('    os.system("ls")', 'os.system'),
        ('x = os.popen("ls").read()', 'os.popen'),
        ('y = eval(user)', 'eval('),
        ('exec(code)', 'exec('),
        ('import socket', 'import socket'),
        ('from ctypes import CDLL', 'import ctypes'),
        ('m = __import__("os")', '__import__'),
        ('blob = "' + 'QUJD' * 60 + '"', 'base64 blob (>200 chars)'),
    ])
    def test_exec_class_patterns(self, tmp_path, line, label) -> None:
        [flag] = gate.rules(_diff('pkg/m.py', [], [line]), tmp_path)
        assert flag.kind == 'exec' and flag.detail == label
        assert flag.path == 'pkg/m.py'

    @pytest.mark.parametrize('line', [
        'v = ast.literal_eval(text)', 'model.eval()', 'self.exec_cmd()',
        'x = "subprocess"', 'import subprocess_helper',
        'sha = "' + 'a' * 64 + '"'])
    def test_near_misses_are_not_exec_flags(self, tmp_path, line) -> None:
        assert gate.rules(_diff('pkg/m.py', [], [line]), tmp_path) == []

    @pytest.mark.parametrize('line', ['@pytest.mark.skip(reason="x")',
                                      '@pytest.mark.skipif(True)',
                                      '    pytest.skip("later")'])
    def test_skip_class(self, tmp_path, line) -> None:
        [flag] = gate.rules(_diff('tests/test_a.py', [], [line]), tmp_path)
        assert flag.kind == 'skip'

    def test_removed_assert_is_flagged(self, tmp_path) -> None:
        d = _diff('tests/test_a.py', ['    assert x == 1', '    assert y'],
                  ['    pass'])
        [flag] = gate.rules(d, tmp_path)
        assert flag.kind == 'assert-removed' and '2 assert' in flag.detail

    def test_moved_assert_is_not_removed(self, tmp_path) -> None:
        d = _diff('tests/test_a.py', ['    assert x == 1', '    y = 2'],
                  ['    y = 2', '    assert x == 1'])
        assert gate.rules(d, tmp_path) == []

    def test_only_added_lines_are_pattern_scanned(self, tmp_path) -> None:
        d = _diff('pkg/m.py', ['import subprocess'], ['import shlex'])
        assert gate.rules(d, tmp_path) == []

    @pytest.mark.parametrize('path', [
        '.github/workflows/ci.yml', 'Makefile', 'pyproject.toml',
        'setup.cfg', 'tox.ini', '.pre-commit-config.yaml', 'sub/Makefile'])
    def test_config_files(self, tmp_path, path) -> None:
        [flag] = gate.rules(_diff(path, ['a'], ['b']), tmp_path)
        assert flag.kind == 'config' and flag.path == path

    @pytest.mark.parametrize('path', ['.git/hooks/pre-commit',
                                      '.venv/lib/x.py',
                                      'pkg/__pycache__/x.py'])
    def test_noise_dirs(self, tmp_path, path) -> None:
        kinds = self._kinds(gate.rules(_diff(path, ['a'], ['b']), tmp_path))
        assert 'noise' in kinds

    @pytest.mark.parametrize('path', ['../outside.py', '/etc/passwd',
                                      'pkg/../../x.py'])
    def test_outside_project(self, tmp_path, path) -> None:
        kinds = self._kinds(gate.rules(_diff(path, ['a'], ['b']), tmp_path))
        assert 'outside' in kinds

    def test_inside_project_via_dotdot_is_fine(self, tmp_path) -> None:
        d = _diff('pkg/../mod.py', ['a'], ['b'])
        assert gate.rules(d, tmp_path) == []

    def test_size_cap(self, tmp_path) -> None:
        d = _diff('pkg/m.py', [], ['x = 1'])
        [flag] = gate.rules(d, tmp_path, max_bytes=20)
        assert flag.kind == 'size' and flag.path == ''
        assert gate.rules(d, tmp_path, max_bytes=len(d.encode())) == []

    def test_secret_in_added_lines(self, tmp_path, marker_scanner) -> None:
        d = _diff('pkg/m.py', ['TOKEN = "MARKER_SECRET"'], ['TOKEN = ""'])
        assert gate.rules(d, tmp_path) == []      # removed: not scanned
        d = _diff('pkg/m.py', [], ['TOKEN = "MARKER_SECRET"'])
        [flag] = gate.rules(d, tmp_path)
        assert flag.kind == 'secret' and 'marker' in flag.detail

    def test_unparsable_diff_flags_parse_and_still_scans(self, tmp_path):
        d = ('--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1,1 +1,1 @@\n'
             '-x = 1\n+import subprocess\n\n'
             'rename from a\nrename to b\n')
        kinds = self._kinds(gate.rules(d, tmp_path))
        assert kinds == ['exec', 'parse']

    def test_empty_diff(self, tmp_path) -> None:
        assert gate.rules('', tmp_path) == []
        assert gate.rules('   \n', tmp_path) == []

    def test_several_files_in_order(self, tmp_path) -> None:
        d = (_diff('a.py', [], ['import socket'])
             + _diff('Makefile', ['x'], ['y']))
        flags = gate.rules(d, tmp_path)
        assert [(f.kind, f.path) for f in flags] == [
            ('exec', 'a.py'), ('config', 'Makefile')]
        assert flags[0].describe() == 'exec: a.py: import socket'


# --- gate.decide -------------------------------------------------------------

class TestDecide:
    def _review(self, **over):
        return {**GOOD_REVIEW, **over}

    def test_intended_on_clean_confident_review(self) -> None:
        v = gate.decide([], self._review())
        assert v.state == gate.INTENDED
        assert any('implements the task' in r for r in v.reasons)
        assert any('Renames the helper' in r for r in v.reasons)

    def test_minor_unrelated_is_still_intended(self) -> None:
        assert gate.decide([], self._review(unrelated_changes='minor')
                           ).state == gate.INTENDED

    @pytest.mark.parametrize('kind', sorted(gate.SUSPICIOUS_KINDS))
    def test_suspicious_flag_wins_over_a_good_review(self, kind) -> None:
        v = gate.decide([gate.Flag(kind, 'p', 'd')], self._review())
        assert v.state == gate.SUSPICIOUS and v.reasons == [f'{kind}: p: d']

    def test_missing_review_is_unclear(self) -> None:
        v = gate.decide([], None)
        assert v.state == gate.UNCLEAR
        assert 'no reviewer verdict' in v.reasons[0]

    def test_malformed_review_is_unclear(self) -> None:
        v = gate.decide([], {'implements_task': 'maybe'})
        assert v.state == gate.UNCLEAR and 'malformed' in v.reasons[0]

    @pytest.mark.parametrize('over', [{'obfuscated': 'yes'},
                                      {'weakens_tests': 'yes'}])
    def test_obfuscated_or_weakened_tests_is_suspicious(self, over) -> None:
        v = gate.decide([], self._review(**over))
        assert v.state == gate.SUSPICIOUS
        assert any(r.startswith('reviewer:') for r in v.reasons)

    @pytest.mark.parametrize('over,needle', [
        ({'implements_task': 'partly'}, 'only partly'),
        ({'implements_task': 'no'}, 'only no'),
        ({'unrelated_changes': 'major'}, 'major unrelated'),
        ({'confidence': 0.69}, 'confidence 0.69 below 0.70'),
    ])
    def test_unclear_reasons(self, over, needle) -> None:
        v = gate.decide([], self._review(**over))
        assert v.state == gate.UNCLEAR
        assert any(needle in r for r in v.reasons), v.reasons

    def test_confidence_threshold_is_inclusive(self) -> None:
        assert gate.decide([], self._review(confidence=0.7)
                           ).state == gate.INTENDED

    @pytest.mark.parametrize('kind', sorted(gate.BLOCKING_KINDS))
    def test_blocking_flags_make_a_good_review_unclear(self, kind) -> None:
        v = gate.decide([gate.Flag(kind, 'p', 'd')], self._review())
        assert v.state == gate.UNCLEAR
        assert any(kind in r and 'need a human' in r for r in v.reasons)

    def test_describe(self) -> None:
        v = gate.Verdict(gate.UNCLEAR, ['a', 'b'])
        assert v.describe() == 'unclear (a; b)'
        assert gate.Verdict(gate.INTENDED).describe() == 'intended'


# --- parse_review / packet / question ----------------------------------------

class TestParseReview:
    def test_plain_json(self) -> None:
        out = gate.parse_review(json.dumps(GOOD_REVIEW))
        assert out == {**GOOD_REVIEW, 'confidence': 0.9}

    def test_fenced_and_uppercase(self) -> None:
        text = ('Here you go:\n```json\n{"implements_task": "Yes",'
                ' "unrelated_changes": "NONE", "weakens_tests": "no",'
                ' "obfuscated": "no", "confidence": "0.8"}\n```')
        out = gate.parse_review(text)
        assert out['implements_task'] == 'yes'
        assert out['unrelated_changes'] == 'none'
        assert out['confidence'] == 0.8 and 'notes' not in out

    @pytest.mark.parametrize('bad', [
        '', 'no json here', '{not json}', '[1, 2]',
        json.dumps({k: v for k, v in GOOD_REVIEW.items()
                    if k != 'obfuscated'}),
        json.dumps({**GOOD_REVIEW, 'implements_task': 'maybe'}),
        json.dumps({**GOOD_REVIEW, 'confidence': 1.5}),
        json.dumps({**GOOD_REVIEW, 'confidence': 'high'}),
        json.dumps({**GOOD_REVIEW, 'confidence': True}),
    ])
    def test_rejects(self, bad) -> None:
        with pytest.raises(ValueError):
            gate.parse_review(bad)

    def test_notes_are_clipped(self) -> None:
        out = gate.parse_review(json.dumps({**GOOD_REVIEW,
                                            'notes': 'n' * 1000}))
        assert len(out['notes']) == 400

    def test_gate_questions_are_stable(self) -> None:
        # The reviewer's rubric is part of the gate's contract: a change here
        # changes what every recorded gate row meant. Update deliberately.
        digest = hashlib.sha256(gate.GATE_QUESTIONS.encode()).hexdigest()
        assert digest[:16] == 'cde66b28bd44c902'
        for key in gate.REVIEW_KEYS + ('notes',):
            assert key in gate.GATE_QUESTIONS

    def test_packet_text_and_truncation(self) -> None:
        text = gate.packet_text('fix it', '', 'I fixed it', 'd' * 50,
                                max_diff_chars=10)
        assert 'User request:\nfix it' in text
        assert '(the user request itself)' in text
        assert "intent for this change:\nI fixed it" in text
        assert text.endswith('d' * 10 + '\n[diff truncated: 40 more '
                             'characters]')

    def test_review_question(self) -> None:
        q = gate.review_question('packet')
        assert q.kind == decisions.REVIEW and q.id == 'gate'
        assert q.instructions == gate.GATE_QUESTIONS and q.state == 'packet'

    def test_stat(self) -> None:
        d = (_diff('a.py', ['x'], ['y', 'z']) + _diff('b/c.py', [], ['n']))
        assert gate.stat(d) == [('a.py', 2, 1), ('b/c.py', 1, 0)]
        text = gate.stat_text(d)
        assert 'a.py   | +2 -1' in text and 'b/c.py | +1 -0' in text
        assert text.endswith('2 file(s) changed, 3 insertion(s), '
                             '1 deletion(s)')
        assert gate.stat_text('') == ''


# --- patch.render / rebase ---------------------------------------------------

class TestRebase:
    def test_round_trip_and_absolute_paths(self, tmp_path) -> None:
        d = (_diff('pkg/mod.py', ['    return 1'], ['    return 2'])
             + _diff('new.py', [], ['x = 1'], new_file=True))
        out = patch.rebase(d, tmp_path)
        parsed = patch.parse(out)
        assert [fp.path for fp in parsed] == [
            str(tmp_path / 'pkg/mod.py'), str(tmp_path / 'new.py')]
        assert parsed[1].new_file is True
        assert parsed[0].hunks[0].lines == [('-', '    return 1'),
                                            ('+', '    return 2')]
        assert patch.render(parsed) == out

    def test_no_newline_markers_survive(self, tmp_path) -> None:
        d = ('--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-a\n'
             '\\ No newline at end of file\n+b\n'
             '\\ No newline at end of file\n')
        [fp] = patch.parse(patch.rebase(d, tmp_path))
        h = fp.hunks[0]
        assert h.old_no_newline and h.new_no_newline

    def test_unparsable_is_returned_unchanged(self, tmp_path) -> None:
        d = 'rename from a\nrename to b\n'
        assert patch.rebase(d, tmp_path) == d

    def test_absolute_paths_are_kept(self, tmp_path) -> None:
        d = _diff('/abs/x.py', ['a'], ['b'])
        [fp] = patch.parse(patch.rebase(d, tmp_path))
        assert fp.path == '/abs/x.py'


# --- the LLM reviewer --------------------------------------------------------

class CompletingAdapter(FakeAdapter):
    """FakeAdapter with a scripted ``complete``."""

    def __init__(self, name='Fake', reply='', remote=True, delay=0.0,
                 error=None) -> None:
        super().__init__(name, remote=remote)
        self.reply, self.delay, self.error = reply, delay, error
        self.calls: list = []

    def complete(self, prompt, max_tokens=1024, model=''):
        self.calls.append((prompt, max_tokens, model))
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.reply


def _question() -> decisions.Question:
    return gate.review_question(gate.packet_text('req', 'task', 'intent',
                                                 'diff'))


class TestLLMReviewer:
    def test_json_reply_becomes_an_answer(self) -> None:
        adapter = CompletingAdapter(reply=json.dumps(GOOD_REVIEW))
        r = llm.LLMReviewer(adapter, 'sonnet')
        [a] = r.ask([_question()])
        assert r.name == 'llm:Fake|sonnet' and a.judge == r.name
        assert a.chosen == gate.INTENDED and a.confidence == 0.9
        assert a.dist['implements_task'] == 'yes'
        prompt, max_tokens, model = adapter.calls[0]
        assert prompt.startswith(gate.GATE_QUESTIONS)
        assert 'User request:\nreq' in prompt
        assert model == 'sonnet' and max_tokens == llm.REVIEW_MAX_TOKENS

    def test_suspicious_review_state(self) -> None:
        adapter = CompletingAdapter(reply=json.dumps(
            {**GOOD_REVIEW, 'obfuscated': 'yes'}))
        [a] = llm.LLMReviewer(adapter, 'm').ask([_question()])
        assert a.chosen == gate.SUSPICIOUS

    def test_garbage_raises(self) -> None:
        r = llm.LLMReviewer(CompletingAdapter(reply='I think it is fine.'),
                            'm')
        with pytest.raises(ValueError, match='no JSON'):
            r.ask([_question()])

    def test_only_review_questions(self) -> None:
        r = llm.LLMReviewer(CompletingAdapter(reply='{}'), 'm')
        with pytest.raises(ValueError, match='review questions only'):
            r.ask([decisions.stall_question('x')])

    def test_base_adapter_has_no_complete(self) -> None:
        with pytest.raises(NotImplementedError):
            FakeAdapter('x').complete('p')


class TestDecideReview:
    @pytest.fixture(autouse=True)
    def _clean(self, fake_repo, monkeypatch):
        decisions.clear_judges()
        monkeypatch.setattr(config, 'DECISIONS_GATE_TIMEOUT_MS', 2000)
        self.repo = fake_repo
        yield
        decisions.clear_judges()

    def _rows(self):
        ledger.flush()
        return self.repo.stream('decisions')

    def test_answers_and_row(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'task_id', 'T9')
        judge = llm.LLMReviewer(
            CompletingAdapter(reply=json.dumps(GOOD_REVIEW)), 'm')
        out = decisions.decide_review('gate', _question(), judge=judge)
        assert out is not None and out['implements_task'] == 'yes'
        [row] = self._rows()
        assert row['point'] == 'gate' and row['kind'] == 'review'
        assert row['used'] == 'judge' and row['fallback_reason'] == ''
        assert row['chosen'] == gate.INTENDED and row['task_id'] == 'T9'
        assert row['dist']['confidence'] == 0.9 and row['threshold'] is None
        assert row['heuristic'] is None and row['mode'] == 'active'

    def test_registered_judge_is_used_without_active_mode(self,
                                                          monkeypatch):
        monkeypatch.setattr(config, 'DECISIONS_MODE', 'off')
        decisions.set_judge('gate', llm.LLMReviewer(
            CompletingAdapter(reply=json.dumps(GOOD_REVIEW)), 'm'))
        assert decisions.decide_review('gate', _question()) is not None

    def test_no_judge(self) -> None:
        assert decisions.decide_review('gate', _question()) is None
        [row] = self._rows()
        assert row['used'] == 'heuristic'
        assert row['fallback_reason'] == 'no_judge'

    def test_garbage_is_an_error_row(self) -> None:
        judge = llm.LLMReviewer(CompletingAdapter(reply='nope'), 'm')
        assert decisions.decide_review('gate', _question(), judge) is None
        [row] = self._rows()
        assert row['used'] == 'heuristic'
        assert row['fallback_reason'] == 'error'
        assert 'no JSON' in row['error']

    def test_provider_error_is_an_error_row(self) -> None:
        judge = llm.LLMReviewer(
            CompletingAdapter(error=RuntimeError('boom')), 'm')
        assert decisions.decide_review('gate', _question(), judge) is None
        [row] = self._rows()
        assert row['fallback_reason'] == 'error' and 'boom' in row['error']

    def test_timeout(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'DECISIONS_GATE_TIMEOUT_MS', 50)
        release = threading.Event()

        class Slow:
            name = 'slow'

            def ask(self, questions):
                release.wait(2)
                return [None]
        assert decisions.decide_review('gate', _question(), Slow()) is None
        release.set()
        [row] = self._rows()
        assert row['fallback_reason'] == 'timeout' and row['used'] == \
            'heuristic'
        assert isinstance(row['queued_ms'], int)

    def test_runs_on_its_own_worker(self) -> None:
        seen: dict = {}

        class Where:
            name = 'where'

            def ask(self, questions):
                seen['thread'] = threading.current_thread().name
                return [decisions.Answer(gate.INTENDED, dict(GOOD_REVIEW),
                                         0.9, 'where', 1)]
        assert decisions.decide_review('gate', _question(), Where())
        assert seen['thread'] == 'guru-judge-gate'


# --- judges.build('llm:...') and the default reviewer -----------------------

class TestLLMSpec:
    @pytest.fixture(autouse=True)
    def _registry(self):
        self.anthropic = CompletingAdapter('Anthropic', remote=True)
        self.ollama = CompletingAdapter('Ollama', remote=False)
        self.registry = AdapterRegistry([self.anthropic, self.ollama])
        yield
        llm.set_registry(None)

    def test_build_with_registry(self) -> None:
        judges.set_registry(self.registry)
        j = judges.build('llm:Anthropic|claude-sonnet-4-5')
        assert isinstance(j, llm.LLMReviewer)
        assert j.adapter is self.anthropic and j.model == 'claude-sonnet-4-5'
        assert j.name == 'llm:Anthropic|claude-sonnet-4-5'

    @pytest.mark.parametrize('spec', ['llm:Nope|m', 'llm:Anthropic',
                                      'llm:Anthropic|', 'llm:|m', 'llm:'])
    def test_build_unknown_or_malformed(self, spec) -> None:
        judges.set_registry(self.registry)
        assert judges.build(spec) is None

    def test_build_without_registry(self) -> None:
        judges.set_registry(None)
        assert judges.build('llm:Anthropic|m') is None

    def test_install_registers_the_gate_point(self, monkeypatch) -> None:
        judges.set_registry(self.registry)
        monkeypatch.setattr(config, 'DECISIONS_MODE', 'shadow')
        monkeypatch.setattr(config, 'DECISIONS_POINTS',
                            {'gate': 'llm:Ollama|qwen'})
        try:
            installed = judges.install(warm=False)
            assert installed == {'gate': 'llm:Ollama|qwen'}
            assert isinstance(decisions.judge_for('gate'), llm.LLMReviewer)
        finally:
            decisions.clear_judges()

    def _routing(self, mode='local-and-remote') -> RoutingSettings:
        return RoutingSettings(
            mode=mode, spend_confirm='never', controller=False,
            ladders={'default': [
                RungSpec('Ollama', 'qwen', 'trivial'),
                RungSpec('Anthropic', 'sonnet', 'standard'),
                RungSpec('Anthropic', 'opus', 'hard')]})

    def test_default_reviewer_is_the_standard_rung(self) -> None:
        llm.set_registry(self.registry, self._routing())
        r = llm.default_reviewer('diff', self.ollama, 'local-model')
        assert r is not None and r.adapter is self.anthropic
        assert r.model == 'sonnet'

    def test_local_only_mode_keeps_the_diff_local(self) -> None:
        llm.set_registry(self.registry, self._routing('local-only'))
        r = llm.default_reviewer('diff', self.ollama, 'local-model')
        assert r is not None and r.adapter is self.ollama
        assert r.model == 'qwen'          # the surviving rung, not main

    def test_secret_finding_forces_local(self, marker_scanner) -> None:
        llm.set_registry(self.registry, self._routing())
        r = llm.default_reviewer('x = MARKER_SECRET', self.ollama, 'main')
        assert r is not None and r.adapter is self.ollama

    def test_pending_spend_falls_back_to_the_session(self) -> None:
        cfg = self._routing()
        cfg.spend_confirm = 'ask'
        llm.set_registry(self.registry, cfg)
        r = llm.default_reviewer('diff', self.ollama, 'main')
        assert r is not None and r.adapter is self.ollama and r.model == \
            'main'

    def test_without_registry_uses_the_session(self) -> None:
        llm.set_registry(None)
        r = llm.default_reviewer('diff', self.ollama, 'main')
        assert r is not None and r.adapter is self.ollama and r.model == \
            'main'
        assert llm.default_reviewer('diff', None, '') is None
        assert llm.default_reviewer('diff', self.ollama, '') is None

    def test_without_ladders_uses_the_session(self) -> None:
        llm.set_registry(self.registry, RoutingSettings(controller=False))
        r = llm.default_reviewer('diff', self.anthropic, 'main')
        assert r is not None and r.model == 'main'

    def test_rung_shape(self) -> None:
        assert routing.Rung('a', 'm', 'standard', remote=True).default is False
        assert Path('.').exists()
