"""The answer check: domain rules (guru.domain.claims) and the git
evidence of the endpoint (guru.judges.claims)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from guru.domain import claims


class TestParse:
    def test_problems(self) -> None:
        assert claims.parse('{"problems": ["a", " b ", ""]}') == ['a', 'b']

    def test_wrapped_in_prose(self) -> None:
        assert claims.parse('Here:\n{"problems": []}\nok') == []

    def test_garbage_is_none(self) -> None:
        for text in ('no json', '{"problems": "a"}', '{"x": []}', '[1]',
                     '{"problems": [1]}', ''):
            assert claims.parse(text) is None

    def test_capped(self) -> None:
        many = '{"problems": [' + ','.join(['"p"'] * 9) + ']}'
        assert len(claims.parse(many)) == claims.MAX_PROBLEMS


class TestTexts:
    def test_prompt_cuts_the_evidence(self) -> None:
        text = claims.prompt('req', 'ans', 'x' * (claims.MAX_EVIDENCE_CHARS
                                                  + 10))
        assert 'REQUEST:\nreq' in text and 'ANSWER:\nans' in text
        assert text.endswith('… (evidence cut)')
        empty = claims.prompt('r', 'a', '')
        assert '(no changes in the repository)' in empty

    def test_problems_text_is_recognised(self) -> None:
        text = claims.problems_text(['a'])
        assert claims.already_checked([{'role': 'tool', 'content': text}])
        # The text path delivers it as a user message.
        assert claims.already_checked([{'role': 'user', 'content': text}])
        assert not claims.already_checked([{'role': 'assistant',
                                            'content': text}])

    def test_run_without_a_checker(self) -> None:
        claims.set_checker(None)
        assert claims.run('r', 'a') == []


def _repo(tmp_path: Path) -> Path:
    def git(*a):
        subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t',
                        *a], cwd=tmp_path, check=True, capture_output=True)
    git('init', '-q')
    (tmp_path / 'a.py').write_text('x = 1\n')
    git('add', '.')
    git('commit', '-qm', 'init')
    return tmp_path


class TestEvidence:
    def test_diff_and_new_files(self, tmp_path) -> None:
        from guru.judges import claims as endpoint
        root = _repo(tmp_path)
        (root / 'a.py').write_text('x = 2\n')
        (root / 'new.py').write_text('NEW = True\n')
        ev = endpoint.evidence(root)
        assert '-x = 1' in ev and '+x = 2' in ev
        assert '--- new file: new.py\nNEW = True' in ev

    def test_outside_a_repo(self, tmp_path) -> None:
        from guru.judges import claims as endpoint
        assert endpoint.evidence(tmp_path) == ''


class TestReviewer:
    def test_asks_the_routed_reviewer(self, tmp_path, monkeypatch) -> None:
        from guru.judges import claims as endpoint
        from guru.judges import llm
        root = _repo(tmp_path)
        monkeypatch.chdir(root)
        (root / 'a.py').write_text('x = 3\n')
        seen: dict = {}

        class Adapter:
            def complete(self, prompt, max_tokens=0, model=''):
                seen['prompt'], seen['model'] = prompt, model
                return '{"problems": ["cli.py not changed"]}'

        monkeypatch.setattr(llm, 'default_reviewer',
                            lambda ev, a, m: SimpleNamespace(
                                adapter=Adapter(), model='sonnet'))
        out = endpoint.ClaimsReviewer().check('wire it', 'wired')
        assert out == ['cli.py not changed']
        assert seen['model'] == 'sonnet' and '+x = 3' in seen['prompt']

    def test_unparsable_reply_passes(self, tmp_path, monkeypatch) -> None:
        from guru.judges import claims as endpoint
        from guru.judges import llm

        class Adapter:
            def complete(self, prompt, max_tokens=0, model=''):
                return 'looks fine to me'
        monkeypatch.chdir(_repo(tmp_path))
        monkeypatch.setattr(llm, 'default_reviewer',
                            lambda ev, a, m: SimpleNamespace(
                                adapter=Adapter(), model='m'))
        assert endpoint.ClaimsReviewer().check('r', 'a') == []


class TestEvidenceBudget:
    def test_new_files_first_and_a_big_diff_is_cut(self, tmp_path) -> None:
        from guru.judges import claims as endpoint
        root = _repo(tmp_path)
        (root / 'a.py').write_text('y = 2\n' * 20000)        # huge diff
        (root / 'new.py').write_text('NEW = True\n')
        (root / 'blob.bin').write_bytes(b'\0\1\2' * 10)
        ev = endpoint.evidence(root)
        assert ev.startswith('--- new file: new.py\nNEW = True')
        assert 'blob.bin' not in ev
        assert '… (diff cut)' in ev
        assert len(ev) <= claims.MAX_EVIDENCE_CHARS + 100

    def test_no_commit_yet(self, tmp_path) -> None:
        from guru.judges import claims as endpoint
        subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
        (tmp_path / 'x.py').write_text('x')
        assert endpoint.evidence(tmp_path) == ''


def test_secret_findings_skip_a_remote_reviewer(tmp_path,
                                                monkeypatch) -> None:
    from guru.domain import policy
    from guru.judges import claims as endpoint
    from guru.judges import llm
    monkeypatch.chdir(_repo(tmp_path))
    called: list = []

    class Remote:
        remote = True

        def complete(self, *a, **k):
            called.append(1)
            return '{"problems": ["x"]}'

    monkeypatch.setattr(llm, 'default_reviewer',
                        lambda ev, a, m: SimpleNamespace(adapter=Remote(),
                                                         model='m'))
    monkeypatch.setattr(policy, 'scan', lambda text: ['finding'])
    assert endpoint.ClaimsReviewer().check('r', 'a') == []
    assert called == []
