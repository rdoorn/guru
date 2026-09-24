"""Container integration test for the Colima sandbox runtime (chunk S1).

Marked ``sandbox`` — excluded by ``make test``, run with
``make test-sandbox`` — and skipped unless ``docker info`` succeeds. Builds
a tiny image from a temp uv project (network on for the build: that is the
provisioning phase; S2 puts the proxy in front of it), then checks the run
phase: ``python -c`` works, ``--network none`` blocks egress, ``/tmp`` is a
tmpfs that does not persist, and the working copy's diff shows an edit.
The image and the copy are removed afterwards.

The copy lives under ``~/.guru/sandbox/`` because Colima mounts ``$HOME``
into its VM; a macOS temp dir would not be visible to the daemon.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from guru import config
from guru.domain import files
from guru.domain import sandbox as sb
from guru.repositories import sandbox_images as images
from guru.repositories.settings import DEFAULT_BASE_IMAGE, SandboxSettings
from guru.sandbox import colima

pytestmark = pytest.mark.sandbox

_LOCK = '''version = 1
revision = 3
requires-python = ">=3.12"

[[package]]
name = "sbfixture"
version = "0.1.0"
source = { virtual = "." }
'''
_PYPROJECT = '''[project]
name = "sbfixture"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []
'''


@pytest.fixture(scope='module')
def runtime_ok():
    colima.reset_cache()
    if not colima.available(Path.cwd(), refresh=True):
        pytest.skip('docker info failed; Colima not running')
    return True


@pytest.fixture
def allowed(tmp_path, monkeypatch):
    home = Path(config.SANDBOX_HOME)
    monkeypatch.setattr(config, 'ALLOWED_READ_DIRS',
                        {str(tmp_path), str(home)})
    monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
    files.set_path_asker(lambda question: False)
    try:
        yield tmp_path
    finally:
        files.set_path_asker(None)


@pytest.fixture
def project(allowed):
    root = allowed / 'sbfixture'
    root.mkdir()
    (root / 'pyproject.toml').write_text(_PYPROJECT, encoding='utf-8')
    (root / 'uv.lock').write_text(_LOCK, encoding='utf-8')
    (root / 'hello.py').write_text('print("ok")\n', encoding='utf-8')
    (root / '.env').write_text('SECRET=1\n', encoding='utf-8')
    return root


@pytest.fixture
def built(runtime_ok, project, monkeypatch):
    """A built image for the fixture project plus a working copy under
    the Colima-visible sandbox home; both removed afterwards."""
    home = Path(config.SANDBOX_HOME) / '_integration'
    home.mkdir(parents=True, exist_ok=True)
    sbhome = Path(tempfile.mkdtemp(prefix='t-', dir=home))
    monkeypatch.setattr(config, 'SANDBOX_HOME', sbhome)
    settings = SandboxSettings(base_image=DEFAULT_BASE_IMAGE, cpus=1.0,
                               memory_mb=512, pids=64, timeout_s=120)
    spec = sb.spec_from(project, settings)
    text = sb.dockerfile_for(spec.project, spec.base_image)
    dockerfile = images.write_dockerfile(spec, text)
    copy = colima.prepare_copy(project, images.work_root(spec) / 'ctx',
                               sb.copy_excludes(project))
    assert not (copy / '.env').exists()
    res = colima.build(spec, dockerfile, copy, network='default')
    assert res.ok, res.stderr[-2000:]
    assert res.digest.startswith('sha256:')
    assert images.needs_build(spec, text) is False
    try:
        yield spec, copy
    finally:
        subprocess.run(['docker', 'rmi', '-f', spec.image_tag],
                       capture_output=True)
        shutil.rmtree(sbhome, ignore_errors=True)


def test_run_no_network_no_persistence_and_diff(built, project) -> None:
    spec, copy = built
    res = colima.run(spec, ['python', '-c', 'print("ok")'], copy)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == 'ok'
    assert res.docker_argv[:4] == ['docker', 'run', '--rm', '--name']

    res = colima.run(spec, ['python', 'hello.py'], copy)
    assert res.returncode == 0 and res.stdout.strip() == 'ok'

    net = colima.run(spec, ['python', '-c',
                            'import urllib.request; urllib.request.urlopen('
                            '"https://pypi.org", timeout=5)'], copy)
    assert net.returncode != 0, 'egress should be impossible'
    assert 'URLError' in net.stderr or 'gaierror' in net.stderr

    wrote = colima.run(spec, ['python', '-c',
                              'open("/tmp/marker", "w").write("x")'], copy)
    assert wrote.returncode == 0, wrote.stderr
    gone = colima.run(spec, ['python', '-c',
                             'import os; raise SystemExit(0 if not '
                             'os.path.exists("/tmp/marker") else 3)'], copy)
    assert gone.returncode == 0, 'a /tmp file persisted across runs'

    ro = colima.run(spec, ['python', '-c',
                           'open("/etc/marker", "w").write("x")'], copy)
    assert ro.returncode != 0, 'the root filesystem should be read-only'

    edit = colima.run(spec, ['python', '-c',
                             'open("hello.py", "a").write("print(2)\\n")'],
                      copy)
    assert edit.returncode == 0, edit.stderr
    text = colima.diff(copy, project=project)
    assert '+print(2)' in text and 'hello.py' in text
    assert (project / 'hello.py').read_text() == 'print("ok")\n'

    denied = colima.run(spec, ['sh', '-c', 'id'], copy)
    assert denied.denied and denied.returncode == -1


def test_timeout_kills_the_container(built) -> None:
    spec, copy = built
    quick = sb.SandboxSpec(**{**spec.__dict__, 'timeout_s': 3})
    res = colima.run(quick, ['python', '-c', 'import time; time.sleep(60)'],
                     copy)
    assert res.timed_out and res.returncode == -1
    listing = subprocess.run(['docker', 'ps', '-q', '--filter',
                              f'name={res.name}'], capture_output=True,
                             text=True)
    assert listing.stdout.strip() == '', 'container survived the timeout'
