"""Tests for the sandbox verbs (S3, ``guru.sandbox.verbs``) with the
Colima runtime faked: run/python/diff digests and refusals, the submit
paths per verdict and access mode (a real ``apply_patch`` on a temp
project), the per-task copy lifecycle, the tool-registry wiring
(advertising, pre-activation, controller) and ``/sandbox status|gate``."""
import json
import shutil
from pathlib import Path

import pytest

from guru import config, session
from guru.domain import decisions, files, gate, ledger, tools
from guru.domain import sandbox as sb
from guru.repositories import sandbox_images as images
from guru.repositories.settings import load_sandbox
from guru.sandbox import colima, provision, verbs
from tests.test_gate import GOOD_REVIEW

MOD = 'def f():\n    return 1\n'
MOD_DIFF = ('diff --git a/pkg/mod.py b/pkg/mod.py\n'
            '--- a/pkg/mod.py\n+++ b/pkg/mod.py\n'
            '@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n')
EXEC_DIFF = ('--- a/pkg/mod.py\n+++ b/pkg/mod.py\n'
             '@@ -1,2 +1,3 @@\n def f():\n+    import subprocess\n'
             '     return 1\n')
DELETE_DIFF = ('diff --git a/pkg/mod.py b/pkg/mod.py\n'
               'deleted file mode 100644\n--- a/pkg/mod.py\n+++ /dev/null\n'
               '@@ -1,2 +0,0 @@\n-def f():\n-    return 1\n')


class FakeReviewer:
    """Gate judge answering with a fixed review dict (or raising)."""
    name = 'fake-reviewer'

    def __init__(self, review=None, fail=False) -> None:
        self.review = dict(GOOD_REVIEW if review is None else review)
        self.fail = fail
        self.calls: list = []

    def ask(self, questions):
        self.calls.extend(questions)
        if self.fail:
            raise RuntimeError('reviewer down')
        state = gate.decide([], self.review).state
        return [decisions.Answer(state, dict(self.review),
                                 float(self.review['confidence']),
                                 self.name, 2) for _ in questions]


class FakeColima:
    """Stands in for ``colima.prepare_copy``/``run``/``diff``: copies the
    tree without git, answers runs from a script, returns a scripted
    diff."""

    def __init__(self) -> None:
        self.runs: list = []
        self.diff_text = ''
        self.rc, self.out, self.err = 0, 'ok\n', ''
        self.scripts: list = []          # (name, content) sandbox_python ran

    def prepare_copy(self, project, dest, excludes):
        shutil.copytree(project, dest, symlinks=True,
                        ignore=shutil.ignore_patterns('.git'))
        (Path(dest) / colima.COPY_MARKER).write_text('copy\n')
        return Path(dest)

    def run(self, spec, argv, copy, network='none', env=None):
        self.runs.append((list(argv), Path(copy)))
        if argv[0] == 'python' and argv[1].startswith(verbs.SCRIPT_PREFIX):
            script = Path(copy) / argv[1]
            assert script.is_file(), 'script must exist while running'
            self.scripts.append((argv[1], script.read_text()))
        return colima.RunResult(list(argv), self.rc, self.out, self.err,
                                0.25)

    def diff(self, copy, project=None):
        return self.diff_text


@pytest.fixture
def sandboxed(tmp_path, monkeypatch, fake_repo):
    """A provisioned (recorded, not built) project the verbs may run on,
    with the runtime faked; yields ``(project, fake)``."""
    root = tmp_path / 'proj'
    (root / 'pkg').mkdir(parents=True)
    (root / '.guru').mkdir()
    (root / 'pyproject.toml').write_text(
        '[project]\nname = "pkg"\nversion = "0.1"\n'
        'requires-python = ">=3.12"\ndependencies = []\n')
    (root / 'uv.lock').write_text('version = 1\nrequires-python = ">=3.12"\n')
    (root / 'pkg' / 'mod.py').write_text(MOD)
    monkeypatch.setattr(config, 'SANDBOX_HOME', tmp_path / 'sbhome')
    monkeypatch.setattr(config, 'PROJECT_GURU_DIR', root / '.guru')
    monkeypatch.setattr(config, 'SANDBOX_POLICY_PATH',
                        root / '.guru' / 'sandbox.toml')
    monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH',
                        tmp_path / 'no-settings.toml')
    monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
    monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', {str(tmp_path)})
    monkeypatch.setattr(config, 'MODE', config.MODE_AUTO)
    monkeypatch.setattr(config, 'AUTO_GRANT', True)
    monkeypatch.setattr(config, 'DECISIONS_GATE_TIMEOUT_MS', 2000)
    monkeypatch.setattr(session, 'task_id', '')
    monkeypatch.setattr(session, 'agent_id', 'main')
    monkeypatch.setattr(session, 'task_text', '')
    monkeypatch.setattr(session, 'messages', [
        {'role': 'system', 'content': 's'},
        {'role': 'user', 'content': 'make f return 2'}])
    fake = FakeColima()
    monkeypatch.setattr(colima, 'prepare_copy', fake.prepare_copy)
    monkeypatch.setattr(colima, 'run', fake.run)
    monkeypatch.setattr(colima, 'diff', fake.diff)
    monkeypatch.setattr(colima, 'copy_excludes_for', lambda p: [])
    spec = sb.spec_from(root, load_sandbox())
    images.record_built(spec, sb.dockerfile_for(root, spec.base_image),
                        'sha256:cafe')
    decisions.clear_judges()
    files.set_path_asker(lambda q: False)
    provision.set_approve_asker(lambda q: False)
    try:
        yield root, fake
    finally:
        verbs.cleanup_all()
        decisions.clear_judges()
        files.set_path_asker(None)
        provision.set_approve_asker(None)


def _events(repo, kind):
    ledger.flush()
    return [r for r in repo.stream('sandbox_events') if r['kind'] == kind]


# --- availability ------------------------------------------------------------

class TestAvailable:
    def test_needs_an_image_record(self, sandboxed) -> None:
        root, _fake = sandboxed
        assert verbs.available() is True
        spec = verbs.spec_for()
        (images.record_dir(spec) / images.RECORD_FILE).unlink()
        assert verbs.available() is False
        assert verbs.sandbox_run(['python', '-c', '1']) == \
            verbs.NOT_PROVISIONED
        assert verbs.sandbox_python('1') == verbs.NOT_PROVISIONED
        assert verbs.sandbox_diff() == verbs.NOT_PROVISIONED
        assert verbs.sandbox_submit('x') == verbs.NOT_PROVISIONED
        assert verbs.request_dependency('six') == verbs.NOT_PROVISIONED

    def test_no_lockfile_or_bad_settings(self, sandboxed) -> None:
        root, _fake = sandboxed
        (root / '.guru' / 'sandbox.toml').write_text('[sandbox]\ncpus="x"\n')
        assert verbs.available() is False
        (root / '.guru' / 'sandbox.toml').unlink()
        (root / 'uv.lock').unlink()
        assert verbs.available() is False and verbs.spec_for() is None


# --- sandbox_run / sandbox_python / sandbox_diff -----------------------------

class TestRun:
    def test_digest_and_copy(self, sandboxed, fake_repo) -> None:
        root, fake = sandboxed
        fake.out = '\n'.join(f'line {i}' for i in range(40)) + '\n'
        fake.err = 'warn\n'
        out = verbs.sandbox_run(['pytest', '-q'])
        assert out.startswith('exit 0 in 0.2s\n--- stdout ---\nline 0\n')
        assert 'line 29' in out and 'line 30' not in out
        assert '…truncated (10 more lines' in out
        assert out.endswith('--- stderr ---\nwarn')
        [(argv, copy)] = fake.runs
        assert argv == ['pytest', '-q']
        assert copy.parent == images.work_root(verbs.spec_for())
        assert (copy / 'pkg' / 'mod.py').read_text() == MOD
        assert (copy / colima.COPY_MARKER).is_file()

    def test_argv_shapes(self, sandboxed) -> None:
        _root, fake = sandboxed
        verbs.sandbox_run(json.dumps(['python', '-c', 'print(1)']))
        verbs.sandbox_run('pytest -q tests/test_a.py')
        assert [a for a, _c in fake.runs] == [
            ['python', '-c', 'print(1)'], ['pytest', '-q', 'tests/test_a.py']]
        assert verbs.parse_argv('[not json') == ['[not', 'json']

    @pytest.mark.parametrize('argv', [['bash', '-c', 'x'], 'sh', '',
                                      'pip install x'])
    def test_argv0_refused_without_a_run(self, sandboxed, fake_repo, argv):
        _root, fake = sandboxed
        out = verbs.sandbox_run(argv)
        assert out.startswith('Refused: ') and 'sandbox' in out
        assert fake.runs == []
        assert tools._mode_denial(out)
        [ev] = _events(fake_repo, 'run')
        assert ev['ok'] is False

    def test_detail_tail_and_previous_run(self, sandboxed) -> None:
        _root, fake = sandboxed
        fake.out = 'x' * 5000
        fake.rc = 3
        out = verbs.sandbox_run(['pytest'], detail='true')
        assert out.startswith('exit 3\n… (904 earlier chars omitted)\n')
        assert out.endswith('x' * 4096)
        assert verbs.sandbox_run('', detail='true') == out
        assert len(fake.runs) == 1
        assert verbs.sandbox_run([], detail='yes') == out
        verbs.cleanup_all()
        assert verbs.sandbox_run('', detail='true') == \
            'No previous sandbox run.'

    def test_denied_and_timed_out_digests(self) -> None:
        res = colima.RunResult(['python'], -1, '', '', 0.0,
                               denied='DENIED: nope')
        assert verbs.digest(res) == 'Refused: DENIED: nope'
        assert verbs.tail(res) == 'Refused: DENIED: nope'
        res = colima.RunResult(['python'], 137, 'o', 'e', 9.0,
                               timed_out=True, truncated=True)
        head = ('exit 137 in 9.0s (timed out; the container was killed) '
                '[output capped]')
        assert verbs.digest(res).startswith(head)

    def test_python_writes_and_removes_the_script(self, sandboxed) -> None:
        _root, fake = sandboxed
        out = verbs.sandbox_python('print(1)')
        assert out.startswith('exit 0')
        [(name, content)] = fake.scripts
        assert name.startswith('.guru-sandbox-') and name.endswith('.py')
        assert content == 'print(1)\n'
        [(argv, copy)] = fake.runs
        assert argv == ['python', name]
        assert not (copy / name).exists()
        assert verbs.sandbox_python('   ').startswith('Refused: ')

    def test_script_removed_even_when_the_run_raises(
            self, sandboxed, monkeypatch) -> None:
        _root, fake = sandboxed
        seen: dict = {}

        def boom(spec, argv, copy, network='none', env=None):
            seen['script'] = Path(copy) / argv[1]
            raise RuntimeError('docker gone')
        monkeypatch.setattr(colima, 'run', boom)
        with pytest.raises(RuntimeError):
            verbs.sandbox_python('print(1)')
        assert not seen['script'].exists()

    def test_diff_digest(self, sandboxed) -> None:
        _root, fake = sandboxed
        assert verbs.sandbox_diff() == 'The sandbox copy is unchanged.'
        fake.diff_text = MOD_DIFF
        out = verbs.sandbox_diff()
        assert 'pkg/mod.py | +1 -1' in out and 'sandbox_submit' in out


# --- sandbox_submit ----------------------------------------------------------

class TestSubmit:
    def test_needs_intent_and_changes(self, sandboxed) -> None:
        _root, fake = sandboxed
        assert verbs.sandbox_submit('').startswith('Refused: sandbox_submit')
        assert verbs.sandbox_submit('x') == \
            'Nothing to submit: the sandbox copy is unchanged.'

    def test_intended_auto_applies_and_removes_the_copy(
            self, sandboxed, fake_repo) -> None:
        root, fake = sandboxed
        fake.diff_text = MOD_DIFF
        reviewer = FakeReviewer()
        decisions.set_judge('gate', reviewer)
        verbs.sandbox_run(['pytest'])           # a copy exists
        assert len(verbs.copies()) == 1
        out = verbs.sandbox_submit('make f return 2 as asked')
        assert out.startswith('Gate verdict: intended\n')
        assert 'Applied patch:' in out and 'pkg/mod.py: 1 hunk(s)' in out
        assert (root / 'pkg' / 'mod.py').read_text() == \
            'def f():\n    return 2\n'
        assert verbs.copies() == {}
        assert not list(images.work_root(verbs.spec_for()).iterdir())
        [q] = reviewer.calls
        assert q.kind == decisions.REVIEW
        assert 'User request:\nmake f return 2' in q.state
        assert 'make f return 2 as asked' in q.state and MOD_DIFF in q.state
        [submit] = _events(fake_repo, 'submit')
        assert submit['ok'] is True and submit['detail'].startswith(
            'intended: ')
        [apply] = _events(fake_repo, 'apply')
        assert apply['ok'] is True
        ledger.flush()
        [row] = [r for r in fake_repo.stream('decisions')
                 if r['point'] == 'gate']
        assert row['used'] == 'judge' and row['chosen'] == 'intended'

    def test_requested_deletion_is_applied(self, sandboxed, fake_repo):
        root, fake = sandboxed
        fake.diff_text = DELETE_DIFF
        reviewer = FakeReviewer()
        decisions.set_judge('gate', reviewer)
        out = verbs.sandbox_submit('remove the obsolete module as asked')
        assert out.startswith('Gate verdict: intended\n')
        assert '  - delete: deletes pkg/mod.py (2 lines)' in out
        assert 'pkg/mod.py | +0 -2 deleted' in out
        assert 'Deleted files: pkg/mod.py' in out
        assert f'deleted {root / "pkg" / "mod.py"} (2 lines)' in out
        assert not (root / 'pkg' / 'mod.py').exists()
        assert verbs.copies() == {}
        [q] = reviewer.calls
        assert '1 file(s) deleted: pkg/mod.py' in q.state
        [submit] = _events(fake_repo, 'submit')
        assert submit['detail'].endswith('; deletes: pkg/mod.py')

    def test_unrequested_deletion_asks(self, sandboxed) -> None:
        root, fake = sandboxed
        fake.diff_text = DELETE_DIFF
        decisions.set_judge('gate', FakeReviewer(
            {**GOOD_REVIEW, 'deletions_requested': 'no'}))
        asked: list = []
        provision.set_approve_asker(lambda q: asked.append(q) or False)
        out = verbs.sandbox_submit('clean up')
        assert out.startswith('Declined: ') and 'unclear' in out
        assert 'did not ask to delete' in out
        [q] = asked
        assert 'Deleted files' not in q and 'pkg/mod.py | +0 -2 deleted' in q
        assert (root / 'pkg' / 'mod.py').read_text() == MOD

    def test_task_text_reaches_the_reviewer(self, sandboxed, monkeypatch):
        _root, fake = sandboxed
        fake.diff_text = MOD_DIFF
        monkeypatch.setattr(session, 'task_text', 'sub-agent task T')
        reviewer = FakeReviewer()
        decisions.set_judge('gate', reviewer)
        verbs.sandbox_submit('intent')
        assert 'Task given to the agent:\nsub-agent task T' in \
            reviewer.calls[0].state

    def test_intended_ask_mode_asks_first(self, sandboxed, monkeypatch,
                                          fake_repo) -> None:
        root, fake = sandboxed
        fake.diff_text = MOD_DIFF
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        decisions.set_judge('gate', FakeReviewer())
        asked: list = []
        provision.set_approve_asker(lambda q: asked.append(q) or False)
        out = verbs.sandbox_submit('intent')
        assert out.startswith('Declined: ')
        assert (root / 'pkg' / 'mod.py').read_text() == MOD
        assert len(verbs.copies()) == 1              # kept for another go
        [q] = asked
        assert 'gate verdict intended' in q and 'pkg/mod.py | +1 -1' in q
        assert 'Intent: intent' in q and str(root) in q
        [ev] = _events(fake_repo, 'apply')
        assert ev['ok'] is False and ev['detail'] == 'declined'
        provision.set_approve_asker(lambda q: True)
        out = verbs.sandbox_submit('intent')
        assert 'Applied patch:' in out
        assert (root / 'pkg' / 'mod.py').read_text().endswith('return 2\n')

    def test_auto_without_auto_grant_asks(self, sandboxed, monkeypatch):
        root, fake = sandboxed
        fake.diff_text = MOD_DIFF
        monkeypatch.setattr(config, 'AUTO_GRANT', False)
        decisions.set_judge('gate', FakeReviewer())
        asked: list = []
        provision.set_approve_asker(lambda q: asked.append(q) or False)
        assert verbs.sandbox_submit('intent').startswith('Declined: ')
        assert len(asked) == 1
        assert (root / 'pkg' / 'mod.py').read_text() == MOD

    def test_unclear_asks_even_in_auto_mode(self, sandboxed, fake_repo):
        root, fake = sandboxed
        fake.diff_text = MOD_DIFF
        decisions.set_judge('gate', FakeReviewer(
            {**GOOD_REVIEW, 'implements_task': 'partly',
             'notes': 'Only half of it.'}))
        asked: list = []
        provision.set_approve_asker(lambda q: asked.append(q) or False)
        out = verbs.sandbox_submit('intent')
        assert out.startswith('Declined: ') and 'unclear' in out
        assert 'only partly' in out and 'Only half of it.' in out
        [q] = asked
        assert 'gate verdict unclear' in q and 'Only half of it.' in q
        assert (root / 'pkg' / 'mod.py').read_text() == MOD
        [submit] = _events(fake_repo, 'submit')
        assert submit['ok'] is False and submit['detail'].startswith(
            'unclear: ')

    def test_unclear_approved_applies(self, sandboxed) -> None:
        root, fake = sandboxed
        fake.diff_text = MOD_DIFF
        decisions.set_judge('gate', FakeReviewer(
            {**GOOD_REVIEW, 'confidence': 0.4}))
        provision.set_approve_asker(lambda q: True)
        out = verbs.sandbox_submit('intent')
        assert out.startswith('Gate verdict: unclear') and 'Applied' in out
        assert (root / 'pkg' / 'mod.py').read_text().endswith('return 2\n')

    def test_no_reviewer_is_unclear(self, sandboxed, monkeypatch) -> None:
        root, fake = sandboxed
        fake.diff_text = MOD_DIFF
        monkeypatch.setattr(session, 'adapter', None)
        monkeypatch.setattr(session, 'model', '')
        asked: list = []
        provision.set_approve_asker(lambda q: asked.append(q) or False)
        out = verbs.sandbox_submit('intent')
        assert 'unclear' in out and 'no reviewer verdict' in out
        assert len(asked) == 1
        assert (root / 'pkg' / 'mod.py').read_text() == MOD

    def test_reviewer_failure_is_unclear(self, sandboxed) -> None:
        root, fake = sandboxed
        fake.diff_text = MOD_DIFF
        decisions.set_judge('gate', FakeReviewer(fail=True))
        asked: list = []
        provision.set_approve_asker(lambda q: asked.append(q) or False)
        assert 'unclear' in verbs.sandbox_submit('intent')
        assert len(asked) == 1

    def test_suspicious_refuses_without_asking_or_reviewing(
            self, sandboxed, fake_repo) -> None:
        root, fake = sandboxed
        fake.diff_text = EXEC_DIFF
        reviewer = FakeReviewer()
        decisions.set_judge('gate', reviewer)
        asked: list = []
        provision.set_approve_asker(lambda q: asked.append(q) or True)
        out = verbs.sandbox_submit('intent')
        assert out.startswith('Refused: the quality gate found the change '
                              'suspicious; nothing was applied.')
        assert 'exec: pkg/mod.py: import subprocess' in out
        assert asked == [] and reviewer.calls == []
        assert (root / 'pkg' / 'mod.py').read_text() == MOD
        assert len(verbs.copies()) == 1
        [submit] = _events(fake_repo, 'submit')
        assert submit['detail'].startswith('suspicious: exec:')
        assert _events(fake_repo, 'apply') == []

    def test_reviewer_suspicious_refuses(self, sandboxed) -> None:
        root, fake = sandboxed
        fake.diff_text = MOD_DIFF
        decisions.set_judge('gate', FakeReviewer(
            {**GOOD_REVIEW, 'weakens_tests': 'yes'}))
        out = verbs.sandbox_submit('intent')
        assert out.startswith('Refused: ') and 'weakens tests' in out
        assert (root / 'pkg' / 'mod.py').read_text() == MOD

    def test_read_only_reports_without_consulting_the_reviewer(
            self, sandboxed, monkeypatch, fake_repo) -> None:
        root, fake = sandboxed
        fake.diff_text = MOD_DIFF
        monkeypatch.setattr(config, 'MODE', config.MODE_READ_ONLY)
        reviewer = FakeReviewer()
        decisions.set_judge('gate', reviewer)
        out = verbs.sandbox_submit('intent')
        assert out.startswith('read-only: not applied')
        assert 'pkg/mod.py | +1 -1' in out
        assert reviewer.calls == []            # the diff never left
        assert (root / 'pkg' / 'mod.py').read_text() == MOD
        [ev] = _events(fake_repo, 'submit')
        assert ev['ok'] is False and ev['detail'] == 'read-only: not applied'
        ledger.flush()
        assert [r for r in fake_repo.stream('decisions')
                if r['point'] == 'gate'] == []

    def test_vanished_copy_is_a_clean_refusal(self, sandboxed) -> None:
        _root, fake = sandboxed
        fake.diff_text = MOD_DIFF
        verbs.sandbox_run(['pytest'])
        (_key, copy), = verbs.copies().items()
        shutil.rmtree(copy)
        assert verbs.sandbox_submit('intent') == verbs.COPY_GONE
        assert verbs.copies() == {}
        # A later verb makes a fresh copy and submit works again.
        fake.diff_text = ''
        assert verbs.sandbox_submit('intent') == \
            'Nothing to submit: the sandbox copy is unchanged.'

    def test_diff_failure_on_a_vanished_copy(self, sandboxed, monkeypatch):
        _root, fake = sandboxed
        verbs.sandbox_run(['pytest'])
        (_key, copy), = verbs.copies().items()

        def gone(copy_path, project=None):
            shutil.rmtree(copy_path)
            raise RuntimeError('git diff failed')
        monkeypatch.setattr(colima, 'diff', gone)
        assert verbs.sandbox_submit('intent') == verbs.COPY_GONE

    def test_copy_lock_holds_from_review_to_apply(self, sandboxed,
                                                  monkeypatch) -> None:
        import threading
        root, fake = sandboxed
        fake.diff_text = MOD_DIFF
        monkeypatch.setattr(session, 'task_id', 'T1')
        entered, release = threading.Event(), threading.Event()

        class Slow(FakeReviewer):
            def ask(self, questions):
                entered.set()
                release.wait(5)
                return super().ask(questions)
        decisions.set_judge('gate', Slow())
        verbs.sandbox_run(['pytest'])
        (_key, copy), = verbs.copies().items()
        result: dict = {}
        submitter = threading.Thread(
            target=lambda: result.setdefault(
                'out', verbs.sandbox_submit('intent')))
        submitter.start()
        assert entered.wait(5)
        cleaner = threading.Thread(target=verbs.cleanup_task, args=('T1',))
        cleaner.start()
        cleaner.join(0.3)
        assert cleaner.is_alive()             # blocked on the copy lock
        assert copy.is_dir()
        release.set()
        submitter.join(5)
        cleaner.join(5)
        assert not cleaner.is_alive() and not submitter.is_alive()
        assert 'Applied patch:' in result['out']
        assert (root / 'pkg' / 'mod.py').read_text().endswith('return 2\n')
        assert not copy.exists() and verbs.copies() == {}

    def test_question_verdict(self) -> None:
        q = verbs.SUBMIT_QUESTION.format(state='intended') + '\nIntent: x'
        assert verbs.question_verdict(q) == 'intended'
        q = verbs.SUBMIT_QUESTION.format(state='unclear')
        assert verbs.question_verdict(q) == 'unclear'
        assert verbs.question_verdict('Add dependency six?') == ''
        assert verbs.question_verdict('') == ''

    def test_script_prefix_is_git_excluded(self) -> None:
        assert verbs.SCRIPT_PREFIX == colima.SCRIPT_PREFIX
        assert f'{colima.SCRIPT_PREFIX}*' in colima._GIT_EXCLUDE.split('\n')

    def test_patch_that_does_not_apply_keeps_the_copy(self, sandboxed):
        root, fake = sandboxed
        (root / 'pkg' / 'mod.py').write_text('def f():\n    return 9\n')
        fake.diff_text = MOD_DIFF
        decisions.set_judge('gate', FakeReviewer())
        out = verbs.sandbox_submit('intent')
        assert out.startswith('Patch rejected:') and 'copy is kept' in out
        assert len(verbs.copies()) == 1

    def test_default_reviewer_from_session(self, sandboxed, monkeypatch):
        from tests.test_gate import CompletingAdapter
        root, fake = sandboxed
        fake.diff_text = MOD_DIFF
        adapter = CompletingAdapter(reply=json.dumps(GOOD_REVIEW))
        monkeypatch.setattr(session, 'adapter', adapter)
        monkeypatch.setattr(session, 'model', 'main-model')
        out = verbs.sandbox_submit('intent')
        assert 'Applied patch' in out
        [(prompt, _max, model)] = adapter.calls
        assert model == 'main-model' and MOD_DIFF in prompt


# --- copies per task ---------------------------------------------------------

class TestCopies:
    def test_one_copy_per_task_and_cleanup(self, sandboxed, monkeypatch):
        _root, fake = sandboxed
        monkeypatch.setattr(session, 'task_id', 'T1')
        verbs.sandbox_run(['pytest'])
        verbs.sandbox_python('print(1)')
        assert len(verbs.copies()) == 1
        (key, copy_a), = verbs.copies().items()
        assert key[1] == 'T1' and copy_a.is_dir()
        assert 'task-t1-' in copy_a.name
        monkeypatch.setattr(session, 'task_id', 'T2')
        verbs.sandbox_run(['pytest'])
        assert len(verbs.copies()) == 2
        assert verbs.cleanup_task('T1') == 1
        assert not copy_a.exists() and len(verbs.copies()) == 1
        assert verbs.cleanup_task('T1') == 0
        assert verbs.cleanup_all() == 1 and verbs.copies() == {}

    def test_main_agent_key_is_the_agent_id(self, sandboxed) -> None:
        verbs.sandbox_run(['pytest'])
        (key, _copy), = verbs.copies().items()
        assert key[1] == 'main'

    def test_vanished_copy_is_remade(self, sandboxed) -> None:
        _root, fake = sandboxed
        verbs.sandbox_run(['pytest'])
        (_key, copy), = verbs.copies().items()
        shutil.rmtree(copy)
        verbs.sandbox_run(['pytest'])
        (_key, again), = verbs.copies().items()
        assert again.is_dir() and again != copy

    def test_orchestrator_finish_task_cleans_up(self, monkeypatch,
                                                fake_repo) -> None:
        from guru.orchestrator import Orchestrator
        cleaned: list = []
        monkeypatch.setattr(verbs, 'cleanup_task', cleaned.append)
        o = Orchestrator()
        main = o.manager.active
        main.busy = True
        child = o._make_child(main, task='do it')
        child.queue.clear()
        child.started = 0.0
        assert child.state.task_text == 'do it'
        task_id = child.task_rec.task_id
        child.state.messages.append({'role': 'assistant', 'content': 'A'})
        o.on_done(child)
        assert cleaned == [task_id]


# --- tool registry wiring ----------------------------------------------------

class TestToolWiring:
    @pytest.fixture(autouse=True)
    def _policy(self, monkeypatch):
        tools.set_policy(None)
        monkeypatch.setattr(config, 'FLAT_TOOLS', False)
        monkeypatch.setattr(config, 'PREACTIVATE_TOOLS', ['read_file'])
        yield
        tools.set_policy(None)

    def test_registry_entries(self) -> None:
        for name in tools.SANDBOX_TOOLS:
            info = tools.TOOL_REGISTRY[name]
            assert info['fn'].__name__ == name and info['fn'].__doc__
            assert set(info.get('optional', [])) <= set(info['parameters'])

    def test_not_advertised_without_an_image(self, monkeypatch) -> None:
        monkeypatch.setattr(verbs, 'available', lambda project=None: False)
        assert not set(tools._advertised()) & set(tools.SANDBOX_TOOLS)
        _base, names = tools.initial_tools(can_spawn=False)
        assert names == {'read_file'}
        specs = tools.specs_for(set(tools.SANDBOX_TOOLS), can_spawn=False)
        assert not {s['name'] for s in specs} & set(tools.SANDBOX_TOOLS)
        assert 'sandbox_run' not in tools.search_tools('run in sandbox')

    def test_write_tools_present_without_an_image(self, monkeypatch):
        monkeypatch.setattr(verbs, 'available', lambda project=None: False)
        monkeypatch.setattr(config, 'PREACTIVATE_TOOLS',
                            ['read_file', 'edit_file', 'apply_patch'])
        assert tools.DIRECT_WRITE_TOOLS <= set(tools._advertised())
        _base, names = tools.initial_tools(can_spawn=False)
        assert names == {'read_file', 'edit_file', 'apply_patch'}
        specs = {s['name'] for s in tools.specs_for(
            tools.DIRECT_WRITE_TOOLS, can_spawn=False)}
        assert specs >= tools.DIRECT_WRITE_TOOLS
        assert 'edit_file' in tools.search_tools('edit a file')

    def test_write_tools_hidden_with_an_image(self, monkeypatch) -> None:
        monkeypatch.setattr(verbs, 'available', lambda project=None: True)
        monkeypatch.setattr(config, 'PREACTIVATE_TOOLS',
                            ['read_file', 'edit_file', 'apply_patch'])
        assert not tools.DIRECT_WRITE_TOOLS & set(tools._advertised())
        _base, names = tools.initial_tools(can_spawn=False)
        assert names == {'read_file', *tools.SANDBOX_TOOLS}
        specs = {s['name'] for s in tools.specs_for(
            set(tools.DIRECT_WRITE_TOOLS), can_spawn=False)}
        assert not specs & tools.DIRECT_WRITE_TOOLS
        listing = tools.search_tools('edit write patch delete file')
        assert not any(name in listing for name in tools.DIRECT_WRITE_TOOLS)
        monkeypatch.setattr(config, 'FLAT_TOOLS', True)
        _base, names = tools.initial_tools(can_spawn=False)
        assert names == set(tools.TOOL_REGISTRY) - tools.DIRECT_WRITE_TOOLS

    @pytest.mark.parametrize('name,args', [
        ('write_file', {'path': 'x.py', 'content': '1'}),
        ('edit_file', {'path': 'x.py', 'old': 'a', 'new': 'b', 'sha': 's'}),
        ('apply_patch', {'diff': MOD_DIFF}),
        ('delete_file', {'path': 'x.py'})])
    def test_execute_tool_refuses_direct_writes_with_an_image(
            self, sandboxed, monkeypatch, fake_repo, name, args) -> None:
        from guru import ui
        root, _fake = sandboxed
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        monkeypatch.setattr(session, 'controller', False)
        called: list = []
        monkeypatch.setitem(tools.TOOL_REGISTRY[name], 'fn',
                            lambda **kw: called.append(kw) or 'ran')
        assert tools.execute_tool(name, args) == tools.SANDBOX_WRITE_REFUSAL
        assert called == []
        ledger.flush()
        [row] = [r for r in fake_repo.stream('tool_events')
                 if r['tool'] == name]
        assert row['denied'] == 'policy' and row['ok'] is False
        # ... and the same call runs once the image is gone.
        (images.record_dir(verbs.spec_for()) / images.RECORD_FILE).unlink()
        assert tools.execute_tool(name, args) == 'ran'

    def test_preactivated_with_an_image(self, monkeypatch) -> None:
        monkeypatch.setattr(verbs, 'available', lambda project=None: True)
        base, names = tools.initial_tools(can_spawn=False)
        assert names == {'read_file', *tools.SANDBOX_TOOLS}
        assert tools.sandbox_submit in base
        assert 'sandbox_run' in tools.search_tools('run in sandbox')
        tools.set_policy(tools.ToolsPolicy(disabled={'sandbox_submit'}))
        _base, names = tools.initial_tools(can_spawn=False)
        assert 'sandbox_submit' not in names and 'sandbox_run' in names

    def test_flat_mode_follows_availability(self, monkeypatch) -> None:
        monkeypatch.setattr(config, 'FLAT_TOOLS', True)
        monkeypatch.setattr(verbs, 'available', lambda project=None: False)
        _base, names = tools.initial_tools(can_spawn=False)
        assert names == set(tools.TOOL_REGISTRY) - set(tools.SANDBOX_TOOLS)
        assert tools.DIRECT_WRITE_TOOLS <= names

    def test_controller_never_gets_the_verbs(self, monkeypatch) -> None:
        from guru import ui
        monkeypatch.setattr(verbs, 'available', lambda project=None: True)
        base, names = tools.initial_tools(can_spawn=True, controller=True)
        assert base == [tools.spawn, tools.check, tools.join,
                        tools.use_skill] and names == set()
        assert not {s['name'] for s in tools.specs_for(
            set(tools.SANDBOX_TOOLS), True, controller=True)} & set(
                tools.SANDBOX_TOOLS)
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        monkeypatch.setattr(session, 'controller', True)
        assert tools.execute_tool('sandbox_run', {'argv': 'pytest'}) == \
            'Unknown tool: sandbox_run'

    def test_execute_tool_refuses_when_not_provisioned(
            self, monkeypatch, tmp_path) -> None:
        from guru import ui
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        monkeypatch.setattr(session, 'controller', False)
        monkeypatch.setattr(config, 'PROJECT_GURU_DIR', tmp_path / '.guru')
        monkeypatch.setattr(config, 'SANDBOX_HOME', tmp_path / 'sbhome')
        out = tools.execute_tool('sandbox_python', {'code': 'print(1)'})
        assert out == verbs.NOT_PROVISIONED
        out = tools.execute_tool('request_dependency', {'name': 'six'})
        assert out == verbs.NOT_PROVISIONED

    def test_execute_tool_runs_the_verbs(self, sandboxed, monkeypatch):
        from guru import ui
        _root, fake = sandboxed
        monkeypatch.setattr(ui, 'note_tool', lambda *a: None)
        monkeypatch.setattr(ui, 'note_tool_result', lambda n: None)
        monkeypatch.setattr(session, 'controller', False)
        out = tools.execute_tool('sandbox_run', {'argv': 'pytest -q'})
        assert out.startswith('exit 0')
        out = tools.execute_tool('request_dependency',
                                 {'name': 'six', 'constraint': '>=1'})
        assert out.startswith('Recorded dependency request six>=1')
        assert [r.spec for r in images.pending_requests(verbs.spec_for())] \
            == ['six>=1']


# --- /sandbox status | gate --------------------------------------------------

class TestSandboxCommands:
    def test_status_shows_requests_and_copies(self, sandboxed, monkeypatch,
                                              capsys) -> None:
        import guru.cli as cli
        root, _fake = sandboxed
        monkeypatch.setattr(colima, 'available', lambda *a, **k: True)
        cli._sandbox_command('status')
        out = capsys.readouterr().out
        assert 'pending dependency requests: none' in out
        assert 'task copies: none' in out
        verbs.request_dependency('six', '>=1')
        verbs.sandbox_run(['pytest'])
        (_key, copy), = verbs.copies().items()
        stale = images.work_root(verbs.spec_for()) / 'build-old'
        stale.mkdir()
        (stale / colima.COPY_MARKER).write_text('x')
        cli._sandbox_command('status')
        flat = capsys.readouterr().out.replace('\n', '')   # console wraps
        assert 'pending dependency requests: six>=1' in flat
        assert f'{copy.name} (task main)' in flat
        assert f'{stale.name} (stale; safe to delete)' in flat

    def test_gate_rows(self, sandboxed, fake_repo, capsys, tmp_path):
        import guru.cli as cli
        from guru.repositories.jsonl_ledger import JsonlLedger
        _root, fake = sandboxed
        # The command reads back through ``rows()``, which the in-memory
        # FakeRepo lacks: use the JSONL repository for this test.
        ledger.set_repository(JsonlLedger(tmp_path / 'ledger'))
        cli._sandbox_command('gate')
        out = capsys.readouterr().out
        assert 'gate verdicts (last submits):\n  none this run' in out
        assert 'reviewer rows (decisions/gate):\n  none this run' in out
        fake.diff_text = MOD_DIFF
        decisions.set_judge('gate', FakeReviewer())
        verbs.sandbox_submit('make it two')
        out = cli._sandbox_gate()             # unwrapped (no console)
        assert 'intended: reviewer: implements the task' in out
        assert '[intent: make it two]' in out
        assert 'fake-reviewer: used=judge chosen=intended' in out
        assert 'implements_task=yes' in out
        ledger.set_repository(fake_repo)

    def test_gate_without_ledger(self, monkeypatch, capsys) -> None:
        import guru.cli as cli
        ledger.set_repository(None)
        cli._sandbox_command('gate')
        assert 'no readable ledger' in capsys.readouterr().out

    def test_usage_mentions_gate(self) -> None:
        import guru.cli as cli
        assert 'gate' in cli._SANDBOX_USAGE
