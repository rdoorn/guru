"""The eval fixtures are self-contained repos: each one's own pytest must
behave exactly as FIXTURE.md promises when run from a copy."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / 'evals' / 'fixtures'


def _run_pytest(name: str, tmp_path: Path) -> subprocess.CompletedProcess:
    copy = tmp_path / name
    shutil.copytree(FIXTURES / name, copy)
    return subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p',
                           'no:cacheprovider'],
                          cwd=copy, capture_output=True, text=True,
                          timeout=120)


@pytest.mark.parametrize('name', ['flaskish', 'cli-tool', 'docs-only'])
def test_fixture_has_manifest(name: str) -> None:
    text = (FIXTURES / name / 'FIXTURE.md').read_text(encoding='utf-8')
    assert 'Planted' in text or 'planted' in text


def test_flaskish_happy_path_tests_pass(tmp_path: Path) -> None:
    proc = _run_pytest('flaskish', tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'failed' not in proc.stdout


def test_cli_tool_has_exactly_one_failing_test(tmp_path: Path) -> None:
    proc = _run_pytest('cli-tool', tmp_path)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    summary = proc.stdout.strip().splitlines()[-1]
    m = re.search(r'(\d+) failed', summary)
    assert m and m.group(1) == '1', summary
    assert 'newline' in proc.stdout


def test_docs_only_has_no_tests() -> None:
    assert not list((FIXTURES / 'docs-only').rglob('test_*.py'))
    assert (FIXTURES / 'docs-only' / 'README.md').exists()
    assert (FIXTURES / 'docs-only' / 'CHANGELOG.md').exists()


def test_flaskish_planted_bugs_are_present() -> None:
    """Guard against someone 'fixing' the fixture by accident."""
    upload = (FIXTURES / 'flaskish' / 'app' / 'upload.py').read_text()
    assert 'os.path.join(base_dir, user_path)' in upload
    assert 'normpath' not in upload and 'realpath' not in upload
    sess = (FIXTURES / 'flaskish' / 'app' / 'session.py').read_text()
    assert "now < token['expires_at']" in sess


def test_cli_tool_bug_is_present() -> None:
    src = (FIXTURES / 'cli-tool' / 'wordcount.py').read_text()
    assert "split(' ')" in src
