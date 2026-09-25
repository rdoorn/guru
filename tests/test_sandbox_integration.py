"""Container integration tests for the Colima sandbox runtime (S1) and the
provisioning proxy (S2).

Marked ``sandbox`` — excluded by ``make test``, run with
``make test-sandbox`` — and skipped unless ``docker info`` succeeds. S1:
builds a tiny image from a temp uv project (on the default network), then
checks the run phase: ``python -c`` works, ``--network none`` blocks
egress, ``/tmp`` is a tmpfs that does not persist, and the working copy's
diff shows an edit. S2: an internal network plus the pinned tinyproxy lets
a client reach ``pypi.org``/``files.pythonhosted.org`` on 443 only (other
hosts, plain HTTP and the no-proxy route all fail); ``provision`` builds
the fixture image through that proxy and ``apply_dependency`` runs a real
``uv add`` through it, patches the lockfile back and rebuilds. Images,
copies, networks and containers are removed afterwards.

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


# --- S2: proxy, provisioning, dependency requests ----------------------------

_CLIENT_IMAGE = DEFAULT_BASE_IMAGE


def _client(network: str, env: dict, code: str, timeout: int = 90
            ) -> subprocess.CompletedProcess:
    """Run ``python -c code`` in a throwaway container on ``network``
    (test plumbing, not guru's argv)."""
    argv = ['docker', 'run', '--rm', '--network', network]
    for key, value in env.items():
        argv += ['-e', f'{key}={value}']
    argv += [_CLIENT_IMAGE, 'python', '-c', code]
    return subprocess.run(argv, capture_output=True, text=True,
                          timeout=timeout)


@pytest.fixture
def ledger_repo(monkeypatch):
    from guru.domain import ledger
    from tests.conftest import FakeRepo
    repo = FakeRepo()
    ledger.set_repository(repo)
    monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
    try:
        yield repo
    finally:
        ledger.flush()
        ledger.set_repository(None)


def test_proxy_allows_only_listed_hosts_on_443(runtime_ok, allowed,
                                               ledger_repo) -> None:
    from guru.domain import ledger
    from guru.sandbox import proxy
    net, name = 'guru-provision-itest', 'guru-proxy-itest'
    cfg_dir = Path(config.SANDBOX_HOME) / '_integration' / 'proxy-itest'
    proxy.network_up(net, cwd=allowed)
    try:
        subnet = proxy.network_subnet(net, cwd=allowed)
        assert '/' in subnet
        proxy.write_config(cfg_dir, proxy.allowlist_from_domains(
            {'pypi.org', 'files.pythonhosted.org', 'test-host.example'}),
            client_cidr=subnet)        # a hyphenated host must parse
        proxy.proxy_up(name, net, cfg_dir, SandboxSettings().proxy_image,
                       cwd=allowed)
        try:
            url = f'http://{name}:{proxy.PROXY_PORT}'
            env = {'HTTPS_PROXY': url, 'HTTP_PROXY': url}
            ok = _client(net, env, 'import urllib.request as u; r = u.urlopen'
                         '("https://pypi.org/simple/six/", timeout=30); '
                         'print(r.status)')
            assert ok.returncode == 0 and ok.stdout.strip() == '200', (
                ok.stderr[-1500:])
            wheel = _client(net, env, 'import subprocess, sys; sys.exit('
                            'subprocess.call([sys.executable, "-m", "pip", '
                            '"download", "--no-deps", "-q", "-d", "/tmp/w", '
                            '"six==1.16.0"]))', timeout=180)
            assert wheel.returncode == 0, wheel.stderr[-1500:]
            denied = _client(net, env, 'import urllib.request as u; '
                             'u.urlopen("https://example.com/", timeout=30)')
            assert denied.returncode != 0, 'example.com should be refused'
            assert '403' in denied.stderr, denied.stderr[-800:]
            plain = _client(net, env, 'import urllib.request as u; '
                            'u.urlopen("http://pypi.org/simple/", '
                            'timeout=30)')
            assert plain.returncode != 0, 'plain HTTP should be refused'
            assert '403' in plain.stderr
            direct = _client(net, {}, 'import urllib.request as u; '
                             'u.urlopen("https://pypi.org/", timeout=10)')
            assert direct.returncode != 0, 'no route without the proxy'
            rows = proxy.tail_access_log(name, cwd=allowed)
        finally:
            proxy.proxy_down(name, cwd=allowed)
    finally:
        proxy.network_down(net, cwd=allowed)
        shutil.rmtree(cfg_dir, ignore_errors=True)
    outcomes = {(r['host'], r['port'], r['allowed']) for r in rows}
    assert ('pypi.org', 443, True) in outcomes
    assert ('files.pythonhosted.org', 443, True) in outcomes
    assert ('example.com', 443, False) in outcomes
    assert ('pypi.org', 80, False) in outcomes
    assert not any(r['allowed'] for r in rows
                   if r['host'] not in ('pypi.org', 'files.pythonhosted.org'))
    from guru.repositories import sandbox_images as images_repo
    images_repo.record_net_events(rows, 'itest')
    ledger.flush()
    written = ledger_repo.stream('net_events')
    assert {r['host'] for r in written} >= {'pypi.org', 'example.com'}
    listing = subprocess.run(['docker', 'network', 'ls', '-q', '--filter',
                              f'name={net}'], capture_output=True, text=True)
    assert listing.stdout.strip() == ''


def test_provision_and_uv_add_through_the_proxy(runtime_ok, project,
                                                monkeypatch, ledger_repo
                                                ) -> None:
    from guru.domain import deps, ledger, tools
    from guru.sandbox import provision
    home = Path(config.SANDBOX_HOME) / '_integration'
    home.mkdir(parents=True, exist_ok=True)
    sbhome = Path(tempfile.mkdtemp(prefix='p-', dir=home))
    monkeypatch.setattr(config, 'SANDBOX_HOME', sbhome)
    monkeypatch.setattr(config, 'ALLOWED_DOMAINS', set())
    monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', {str(project.parent)})
    monkeypatch.setattr(config, 'persist_domain', lambda d: None)
    monkeypatch.setattr(config, 'persist_write_dir', lambda d: None)
    monkeypatch.setattr(files, '_show_change', lambda block: None)
    monkeypatch.chdir(project)
    asked: list = []
    tools.set_domain_asker(lambda q: asked.append(q) or True)
    provision.set_approve_asker(lambda q: True)
    settings = SandboxSettings(base_image=DEFAULT_BASE_IMAGE, cpus=1.0,
                               memory_mb=1024, pids=128, timeout_s=300)
    tags: list = []
    try:
        rec = provision.provision(project, settings)
        tags.append(rec.tag)
        assert rec.digest.startswith('sha256:')
        assert len(asked) == 2
        spec = sb.spec_from(project, settings)
        assert images.needs_build(
            spec, sb.dockerfile_for(project, DEFAULT_BASE_IMAGE)) is False
        assert provision.provision(project, settings) == rec   # current
        ledger.flush()
        net_rows = ledger_repo.stream('net_events')
        hosts = {r['host'] for r in net_rows if r['allowed']}
        assert hosts == {'pypi.org', 'files.pythonhosted.org'}
        assert all(r['phase'] == 'build' for r in net_rows)
        assert not any(r['host'] not in hosts for r in net_rows), net_rows
        # Proxy build args are Docker-predefined: not in the image history.
        history = subprocess.run(['docker', 'history', '--no-trunc',
                                  '--format', '{{.CreatedBy}}', rec.tag],
                                 capture_output=True, text=True)
        assert history.returncode == 0, history.stderr
        assert 'guru-proxy' not in history.stdout
        assert 'HTTPS_PROXY=' not in history.stdout
        # The built image runs offline with uv installed.
        copy = colima.prepare_copy(project, images.work_root(spec) / 'c',
                                   sb.copy_excludes(project))
        res = colima.run(spec, ['uv', '--version'], copy)
        assert res.returncode == 0 and res.stdout.startswith('uv '), (
            res.stderr)
        colima.remove_copy(copy)
        # A dependency request installs nothing ...
        out = provision.request_dependency('six', '==1.16.0',
                                           project=project,
                                           settings=settings)
        assert 'nothing was installed' in out
        req, = images.pending_requests(spec)
        assert req.spec == 'six==1.16.0'
        assert images.load_record(spec) == rec
        # ... and applying it locks through the proxy, patches the real
        # tree via apply_patch and rebuilds under the new lockfile tag.
        out = provision.apply_dependency(project, req, settings)
        assert out.startswith('Added six==1.16.0'), out
        assert 'added six==1.16.0' in out
        assert 'six==1.16.0' in (project / 'pyproject.toml').read_text()
        assert deps.lock_packages((project / 'uv.lock').read_text()).get(
            'six') == '1.16.0'
        new_spec = sb.spec_from(project, settings)
        tags.append(new_spec.image_tag)
        assert new_spec.image_tag != spec.image_tag
        new_rec = images.load_record(new_spec)
        assert new_rec is not None and new_rec.digest != rec.digest
        assert images.pending_requests(new_spec) == []
        copy = colima.prepare_copy(project, images.work_root(new_spec) / 'd',
                                   sb.copy_excludes(project))
        res = colima.run(new_spec, ['python', '-c',
                                    'import six; print(six.__version__)'],
                         copy)
        assert res.returncode == 0 and res.stdout.strip() == '1.16.0', (
            res.stderr)
        colima.remove_copy(copy)
        ledger.flush()
        phases = {r['phase'] for r in ledger_repo.stream('net_events')}
        assert phases == {'build', 'uv add'}
    finally:
        tools.set_domain_asker(None)
        provision.set_approve_asker(None)
        for tag in tags:
            subprocess.run(['docker', 'rmi', '-f', tag], capture_output=True)
        shutil.rmtree(sbhome, ignore_errors=True)
    for kind in ('network', 'container'):
        listing = subprocess.run(
            ['docker', kind, 'ls', '-q', '--filter', 'name=guru-pro'],
            capture_output=True, text=True)
        assert listing.stdout.strip() == '', f'stale {kind}'


def test_sandbox_verbs_end_to_end(built, project, monkeypatch, ledger_repo
                                  ) -> None:
    """S3: the verbs against the real runtime — run, edit through
    ``sandbox_python`` in the task's copy, diff, and a ``sandbox_submit``
    that a fake reviewer finds intended in auto mode patches the real
    fixture tree through ``apply_patch`` and removes the copy."""
    from guru import session
    from guru.domain import decisions, ledger
    from guru.sandbox import verbs
    from tests.test_sandbox_verbs import FakeReviewer
    monkeypatch.setattr(config, 'PROJECT_GURU_DIR', project / '.guru')
    monkeypatch.setattr(config, 'SANDBOX_POLICY_PATH',
                        project / '.guru' / 'sandbox.toml')
    monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH',
                        project.parent / 'no-settings.toml')
    monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', {str(project.parent)})
    monkeypatch.setattr(config, 'MODE', config.MODE_AUTO)
    monkeypatch.setattr(config, 'AUTO_GRANT', True)
    monkeypatch.setattr(config, 'persist_write_dir', lambda d: None)
    monkeypatch.setattr(files, '_show_change', lambda block: None)
    monkeypatch.setattr(session, 'task_id', 'INTEG')
    monkeypatch.setattr(session, 'task_text', 'make hello.py print more')
    monkeypatch.setattr(session, 'messages', [
        {'role': 'user', 'content': 'make hello.py print more'}])
    monkeypatch.chdir(project)
    reviewer = FakeReviewer()
    decisions.set_judge('gate', reviewer)
    try:
        assert verbs.available() is True
        out = verbs.sandbox_run(['python', '-c', 'print(1)'])
        assert out.startswith('exit 0') and out.endswith('--- stdout ---\n1')
        assert verbs.sandbox_run('bash -c id').startswith('Refused: ')
        out = verbs.sandbox_python(
            "import pathlib\np = pathlib.Path('hello.py')\n"
            "p.write_text(p.read_text() + 'print(\"more\")\\n')\n"
            "print('edited')\n")
        assert 'edited' in out, out
        (_key, copy), = verbs.copies().items()
        assert _key[1] == 'INTEG'
        assert (copy / 'hello.py').read_text().endswith('print("more")\n')
        assert not list(copy.glob(f'{verbs.SCRIPT_PREFIX}*.py'))
        assert (project / 'hello.py').read_text() == 'print("ok")\n'
        assert 'hello.py | +1 -0' in verbs.sandbox_diff()
        out = verbs.sandbox_submit('append a second print to hello.py')
        assert out.startswith('Gate verdict: intended'), out
        assert 'Applied patch:' in out
        assert (project / 'hello.py').read_text() == \
            'print("ok")\nprint("more")\n'
        assert verbs.copies() == {} and not copy.exists()
        [q] = reviewer.calls
        assert '+print("more")' in q.state
        # A file deleted in the copy round-trips: git diff -> gate stat ->
        # apply_patch deletes it in the real tree.
        (project / 'obsolete.py').write_text('X = 1\nY = 2\n',
                                             encoding='utf-8')
        out = verbs.sandbox_python("import os\nos.remove('obsolete.py')\n"
                                   "print('gone')")
        assert 'gone' in out, out
        diff_out = verbs.sandbox_diff()
        assert 'obsolete.py | +0 -2 deleted' in diff_out, diff_out
        out = verbs.sandbox_submit('delete obsolete.py as asked')
        assert out.startswith('Gate verdict: intended'), out
        assert 'delete: deletes obsolete.py (2 lines)' in out
        assert f'deleted {project / "obsolete.py"} (2 lines)' in out
        assert not (project / 'obsolete.py').exists()
        assert (project / 'hello.py').exists()
        assert 'Task given to the agent:\nmake hello.py print more' in q.state
        ledger.flush()
        kinds = [r['kind'] for r in ledger_repo.stream('sandbox_events')]
        assert 'run' in kinds and 'submit' in kinds and 'apply' in kinds
    finally:
        decisions.clear_judges()
        verbs.cleanup_all()
