"""Tests for the sandbox (S1): the domain spec and Dockerfile generation,
the image records repository, the Colima runtime's fixed docker argv (with
a fake ``procs.run``) and the ``/sandbox status`` command."""
import hashlib
import json
import re
from pathlib import Path

import pytest

from guru import config
from guru.domain import files, policy, procs
from guru.domain import sandbox as sb
from guru.domain.policy import Finding
from guru.repositories import sandbox_images as images
from guru.repositories.settings import SandboxSettings

PINNED = 'python:3.12-slim@sha256:' + 'a' * 64


def _project(tmp_path: Path, name: str = 'proj') -> Path:
    """A minimal uv project: pyproject.toml, uv.lock and one module."""
    root = tmp_path / name
    (root / 'pkg').mkdir(parents=True)
    (root / 'pyproject.toml').write_text(
        '[project]\nname = "pkg"\nversion = "0.1"\n'
        'requires-python = ">=3.12"\ndependencies = []\n', encoding='utf-8')
    (root / 'uv.lock').write_text('version = 1\nrequires-python = ">=3.12"\n',
                                  encoding='utf-8')
    (root / 'pkg' / '__init__.py').write_text('X = 1\n', encoding='utf-8')
    return root


class MarkerScanner:
    """Flags every occurrence of ``MARKER_SECRET`` as a ``marker`` finding."""

    def scan(self, text: str) -> list:
        out = []
        start = 0
        while True:
            i = text.find('MARKER_SECRET', start)
            if i < 0:
                return out
            out.append(Finding('marker', i, i + 13, 'MARKER_SECRET'))
            start = i + 1


@pytest.fixture
def marker_scanner():
    policy.set_scanner(MarkerScanner())
    try:
        yield
    finally:
        policy.set_scanner(None)


# --- guru.domain.sandbox -----------------------------------------------------

class TestDockerfile:
    def test_pinned_from_uv_sync_user_and_workdir(self, tmp_path) -> None:
        root = _project(tmp_path)
        text = sb.dockerfile_for(root, PINNED)
        lines = [ln for ln in text.splitlines()
                 if ln and not ln.startswith('#')]
        assert lines[0] == f'FROM {PINNED}'
        assert f'RUN pip install --no-cache-dir uv=={sb.UV_VERSION}' in lines
        assert 'COPY pyproject.toml uv.lock ./' in lines
        assert any(ln.startswith('RUN uv sync --frozen --all-groups')
                   for ln in lines)
        assert 'COPY . /work' in lines
        assert 'USER 1000:1000' in lines
        assert 'WORKDIR /work' in lines
        # The venv lives outside /work: the copy is bind-mounted over it.
        assert 'UV_PROJECT_ENVIRONMENT=/opt/venv' in text
        assert 'PATH=/opt/venv/bin:$PATH' in text
        # No install at run time: uv is offline and never syncs.
        assert 'UV_OFFLINE=1' in text and 'UV_NO_SYNC=1' in text
        # The lockfile sha is embedded so the Dockerfile changes with it.
        assert sb.lockfile_sha(root) in text

    def test_user_comes_after_installs(self, tmp_path) -> None:
        text = sb.dockerfile_for(_project(tmp_path), PINNED)
        assert text.index('USER 1000:1000') > text.rindex('RUN ')

    @pytest.mark.parametrize('base', ['python:3.12-slim', 'python:3.12-slim@'
                                      'md5:abc', '', 'python@sha256'])
    def test_refuses_an_unpinned_base(self, tmp_path, base) -> None:
        with pytest.raises(ValueError, match='sha256'):
            sb.dockerfile_for(_project(tmp_path), base)

    def test_refuses_a_base_with_a_newline(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            sb.dockerfile_for(_project(tmp_path), PINNED + '\nRUN evil')

    def test_needs_pyproject_and_lockfile(self, tmp_path) -> None:
        root = _project(tmp_path)
        (root / 'uv.lock').unlink()
        with pytest.raises(ValueError, match='uv.lock'):
            sb.dockerfile_for(root, PINNED)
        (root / 'uv.lock').write_text('x', encoding='utf-8')
        (root / 'pyproject.toml').unlink()
        with pytest.raises(ValueError, match='pyproject.toml'):
            sb.dockerfile_for(root, PINNED)


class TestLockfileSha:
    def test_sha256_of_the_lockfile_bytes(self, tmp_path) -> None:
        root = _project(tmp_path)
        want = hashlib.sha256((root / 'uv.lock').read_bytes()).hexdigest()
        assert sb.lockfile_sha(root) == want

    def test_empty_when_absent(self, tmp_path) -> None:
        assert sb.lockfile_sha(tmp_path) == ''

    def test_changes_with_the_lockfile(self, tmp_path) -> None:
        root = _project(tmp_path)
        before = sb.lockfile_sha(root)
        (root / 'uv.lock').write_text('version = 2\n', encoding='utf-8')
        assert sb.lockfile_sha(root) != before


class TestCopyExcludes:
    def test_noise_dirs_and_dotenv_always_excluded(self, tmp_path) -> None:
        excludes = sb.copy_excludes(_project(tmp_path))
        for noise in files.NOISE_DIRS:
            assert noise in excludes
        assert '.env' in excludes and '.env*' in excludes

    def test_file_with_a_finding_is_excluded_and_reported(
            self, tmp_path, marker_scanner) -> None:
        root = _project(tmp_path)
        (root / 'pkg' / 'settings.py').write_text(
            'TOKEN = "MARKER_SECRET"\n', encoding='utf-8')
        (root / 'clean.py').write_text('print(1)\n', encoding='utf-8')
        flagged = sb.flagged_files(root)
        assert flagged == {'pkg/settings.py': ['marker']}
        excludes = sb.copy_excludes(root)
        assert 'pkg/settings.py' in excludes
        assert 'clean.py' not in excludes and 'pkg/__init__.py' not in excludes

    def test_large_and_binary_files_are_not_scanned(
            self, tmp_path, marker_scanner) -> None:
        root = _project(tmp_path)
        (root / 'big.txt').write_text(
            'MARKER_SECRET' + 'x' * sb.SCAN_MAX_BYTES, encoding='utf-8')
        (root / 'blob.bin').write_bytes(b'\x00MARKER_SECRET')
        assert sb.flagged_files(root) == {}

    def test_noise_dir_contents_are_not_scanned(
            self, tmp_path, marker_scanner) -> None:
        root = _project(tmp_path)
        (root / '.venv').mkdir()
        (root / '.venv' / 'leak.py').write_text('MARKER_SECRET',
                                                encoding='utf-8')
        assert sb.flagged_files(root) == {}

    def test_without_a_scanner_nothing_is_flagged(self, tmp_path) -> None:
        root = _project(tmp_path)
        (root / 'a.py').write_text('MARKER_SECRET', encoding='utf-8')
        policy.set_scanner(None)
        assert sb.flagged_files(root) == {}


class TestSpecFrom:
    def test_fields_from_project_and_settings(self, tmp_path) -> None:
        root = _project(tmp_path, 'My_Proj')
        settings = SandboxSettings(enabled=True, base_image=PINNED, cpus=1.5,
                                   memory_mb=512, pids=64, timeout_s=30)
        spec = sb.spec_from(root, settings)
        assert spec.project == root.resolve()
        assert spec.name == sb.project_key(root)
        assert re.fullmatch(r'my_proj-[0-9a-f]{8}', spec.name)
        assert spec.base_image == PINNED
        assert spec.lockfile_sha == sb.lockfile_sha(root)
        assert spec.image_tag == (
            f'guru-sandbox/{spec.name}:{spec.lockfile_sha[:12]}')
        assert (spec.cpus, spec.memory_mb, spec.pids, spec.timeout_s) == (
            1.5, 512, 64, 30)

    def test_name_is_a_safe_docker_repository_component(self, tmp_path):
        root = _project(tmp_path, 'Weird Name!!')
        spec = sb.spec_from(root, SandboxSettings(base_image=PINNED))
        assert re.fullmatch(r'weird-name-[0-9a-f]{8}', spec.name)

    def test_same_basename_different_path_is_a_different_key(
            self, tmp_path) -> None:
        a = _project(tmp_path / 'one', 'proj')
        b = _project(tmp_path / 'two', 'proj')
        assert sb.project_key(a) != sb.project_key(b)
        assert sb.project_key(a) == sb.project_key(
            tmp_path / 'one' / '.' / 'proj')          # resolved path
        assert sb.spec_from(a, SandboxSettings(base_image=PINNED)).image_tag \
            != sb.spec_from(b, SandboxSettings(base_image=PINNED)).image_tag

    def test_requires_a_lockfile(self, tmp_path) -> None:
        root = _project(tmp_path)
        (root / 'uv.lock').unlink()
        with pytest.raises(ValueError, match='uv.lock'):
            sb.spec_from(root, SandboxSettings(base_image=PINNED))


class TestRunners:
    def test_allow_listed_argv0(self) -> None:
        assert sb.RUNNERS == frozenset(
            ('python', 'python3', 'pytest', 'uv', 'ruff', 'mypy', 'flake8',
             'make'))
        assert sb.check_argv(['pytest', '-q']) == ''
        assert sb.check_argv(['/usr/bin/python3', '-c', 'x']) == ''

    @pytest.mark.parametrize('argv', [['bash', '-c', 'x'], ['sh'],
                                      ['curl', 'x'], ['pip', 'install', 'y'],
                                      [], ['python', 3]])
    def test_denied(self, argv) -> None:
        assert sb.check_argv(argv).startswith(procs.DENIED_PREFIX)


# --- guru.repositories.sandbox_images ---------------------------------------

def _spec(tmp_path: Path) -> sb.SandboxSpec:
    return sb.spec_from(_project(tmp_path), SandboxSettings(base_image=PINNED))


class TestImageRecords:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, 'SANDBOX_HOME', tmp_path / 'sbhome')

    def test_no_record_means_build(self, tmp_path) -> None:
        spec = _spec(tmp_path)
        dockerfile = sb.dockerfile_for(spec.project, spec.base_image)
        assert images.load_record(spec) is None
        assert images.needs_build(spec, dockerfile) is True

    def test_record_round_trip_under_sandbox_home(self, tmp_path) -> None:
        spec = _spec(tmp_path)
        dockerfile = sb.dockerfile_for(spec.project, spec.base_image)
        path = images.write_dockerfile(spec, dockerfile)
        assert path == config.SANDBOX_HOME / spec.name / 'Dockerfile'
        assert path.read_text(encoding='utf-8') == dockerfile
        rec = images.record_built(spec, dockerfile, digest='sha256:abc')
        assert rec.tag == spec.image_tag and rec.digest == 'sha256:abc'
        assert rec.lockfile_sha == spec.lockfile_sha
        assert rec.dockerfile_sha == sb.sha_text(dockerfile)
        assert rec.built_at.endswith('+00:00')
        stored = json.loads(
            (config.SANDBOX_HOME / spec.name / 'image.json').read_text())
        assert stored['tag'] == spec.image_tag
        assert images.load_record(spec) == rec
        assert images.needs_build(spec, dockerfile) is False

    def test_lockfile_change_needs_build(self, tmp_path) -> None:
        spec = _spec(tmp_path)
        dockerfile = sb.dockerfile_for(spec.project, spec.base_image)
        images.record_built(spec, dockerfile, digest='sha256:abc')
        (spec.project / 'uv.lock').write_text('version = 9\n',
                                              encoding='utf-8')
        spec2 = sb.spec_from(spec.project, SandboxSettings(base_image=PINNED))
        dockerfile2 = sb.dockerfile_for(spec2.project, spec2.base_image)
        assert images.needs_build(spec2, dockerfile2) is True

    def test_dockerfile_change_needs_build(self, tmp_path) -> None:
        spec = _spec(tmp_path)
        dockerfile = sb.dockerfile_for(spec.project, spec.base_image)
        images.record_built(spec, dockerfile, digest='sha256:abc')
        assert images.needs_build(spec, dockerfile + 'RUN x\n') is True

    def test_corrupt_record_means_build(self, tmp_path) -> None:
        spec = _spec(tmp_path)
        d = config.SANDBOX_HOME / spec.name
        d.mkdir(parents=True)
        (d / 'image.json').write_text('{not json', encoding='utf-8')
        assert images.load_record(spec) is None
        assert images.needs_build(spec, 'FROM x\n') is True


class TestSandboxEvents:
    def test_row_shape(self, monkeypatch) -> None:
        from guru.domain import ledger
        from tests.conftest import FakeRepo
        repo = FakeRepo()
        ledger.set_repository(repo)
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        try:
            images.record_sandbox_event(
                'run', ['pytest', '-q', 'x' * 500], 1.5, True, 'exit 0')
            ledger.flush()
        finally:
            ledger.set_repository(None)
        (stream, row), = repo.rows
        assert stream == 'sandbox_events'
        assert row['kind'] == 'run' and row['ok'] is True
        assert row['seconds'] == 1.5 and row['detail'] == 'exit 0'
        assert row['argv'][:2] == ['pytest', '-q']
        assert len(row['argv'][2]) <= ledger.ARGS_HEAD
        assert {'ts', 'run_id', 'project', 'agent', 'task_id',
                'turn_id'} <= set(row)

    def test_never_raises(self, monkeypatch) -> None:
        from guru.domain import ledger

        class Boom:
            def append(self, stream, row):
                raise RuntimeError('x')
        ledger.set_repository(Boom())
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        try:
            images.record_sandbox_event('build', object(), 0.0, False, '')
            ledger.flush()
        finally:
            ledger.set_repository(None)


# --- guru.sandbox.colima -----------------------------------------------------

from guru.sandbox import colima  # noqa: E402


class FakeRun:
    """Stands in for ``procs.run``: records argv, answers per argv[1]."""

    def __init__(self, **answers) -> None:
        self.calls: list = []
        self.answers = answers          # {'build': (rc, out), 'run': ...}

    def __call__(self, argv, cwd, limits=None, env_extra=None):
        self.calls.append({'argv': list(argv), 'cwd': Path(cwd),
                           'limits': limits, 'env': dict(env_extra or {})})
        key = argv[1] if argv[0] == 'docker' else argv[0]
        rc, out = self.answers.get(key, (0, ''))
        timed_out = rc == 'timeout'
        return procs.ProcResult(list(argv), -1 if timed_out else rc, out,
                                '' if rc == 0 else 'boom', 0.25,
                                timed_out=timed_out)


@pytest.fixture
def fake_run(monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(procs, 'run', fake)
    colima.reset_cache()
    yield fake
    colima.reset_cache()


@pytest.fixture
def sbhome(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'SANDBOX_HOME', tmp_path / 'sbhome')
    return tmp_path / 'sbhome'


class TestAvailable:
    def test_docker_info_argv_and_cache(self, fake_run, tmp_path) -> None:
        fake_run.answers['info'] = (0, '28.3.3\n')
        assert colima.available(tmp_path) is True
        assert colima.available(tmp_path) is True
        assert len(fake_run.calls) == 1
        call = fake_run.calls[0]
        assert call['argv'] == ['docker', 'info', '--format',
                                '{{.ServerVersion}}']
        assert call['cwd'] == tmp_path
        assert call['limits'].timeout_s == colima.INFO_TIMEOUT_S
        assert call['env']['DOCKER_CONFIG'].endswith('.docker')

    def test_failure_is_false_and_refreshable(self, fake_run, tmp_path):
        fake_run.answers['info'] = (1, '')
        assert colima.available(tmp_path) is False
        fake_run.answers['info'] = (0, 'x')
        assert colima.available(tmp_path) is False          # cached
        assert colima.available(tmp_path, refresh=True) is True

    def test_docker_env_passthrough(self, monkeypatch) -> None:
        monkeypatch.setenv('DOCKER_HOST', 'unix:///x.sock')
        monkeypatch.setenv('DOCKER_CONFIG', '/cfg')
        monkeypatch.delenv('DOCKER_CONTEXT', raising=False)
        env = colima.docker_env()
        assert env == {'DOCKER_CONFIG': '/cfg',
                       'DOCKER_HOST': 'unix:///x.sock'}


class TestBuild:
    def test_argv_digest_record_and_event(self, fake_run, sbhome, tmp_path,
                                          monkeypatch) -> None:
        from guru.domain import ledger
        from tests.conftest import FakeRepo
        repo = FakeRepo()
        ledger.set_repository(repo)
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        spec = _spec(tmp_path)
        text = sb.dockerfile_for(spec.project, spec.base_image)
        dockerfile = images.write_dockerfile(spec, text)
        fake_run.answers['image'] = (0, 'sha256:feedface\n')
        try:
            res = colima.build(spec, dockerfile, tmp_path / 'ctx',
                               network='guru-provision')
            ledger.flush()
        finally:
            ledger.set_repository(None)
        assert res.ok and res.digest == 'sha256:feedface'
        assert res.tag == spec.image_tag
        build_call, inspect_call = fake_run.calls
        assert build_call['argv'] == [
            'docker', 'build', '--network', 'guru-provision', '-t',
            spec.image_tag, '-f', str(dockerfile), str(tmp_path / 'ctx')]
        assert build_call['cwd'] == spec.project
        assert build_call['limits'].timeout_s == colima.BUILD_TIMEOUT_S
        assert inspect_call['argv'] == [
            'docker', 'image', 'inspect', '--format', '{{.Id}}',
            spec.image_tag]
        rec = images.load_record(spec)
        assert rec is not None and rec.digest == 'sha256:feedface'
        assert images.needs_build(spec, text) is False
        (stream, row), = repo.rows
        assert stream == 'sandbox_events' and row['kind'] == 'build'
        assert row['ok'] is True and row['detail'] == 'sha256:feedface'

    def test_default_network_is_none(self, fake_run, sbhome, tmp_path):
        spec = _spec(tmp_path)
        dockerfile = images.write_dockerfile(
            spec, sb.dockerfile_for(spec.project, spec.base_image))
        colima.build(spec, dockerfile, tmp_path)
        argv = fake_run.calls[0]['argv']
        assert argv[2:4] == ['--network', 'none']

    def test_failure_records_nothing(self, fake_run, sbhome, tmp_path):
        spec = _spec(tmp_path)
        dockerfile = images.write_dockerfile(
            spec, sb.dockerfile_for(spec.project, spec.base_image))
        fake_run.answers['build'] = (1, '')
        res = colima.build(spec, dockerfile, tmp_path)
        assert not res.ok and res.digest == '' and res.stderr == 'boom'
        assert len(fake_run.calls) == 1               # no inspect
        assert images.load_record(spec) is None


class TestRun:
    def test_exact_docker_argv_in_order(self, fake_run, sbhome, tmp_path):
        spec = _spec(tmp_path)
        copy = tmp_path / 'copy'
        res = colima.run(spec, ['pytest', '-q', 'tests'], copy)
        assert res.returncode == 0 and res.denied == ''
        assert res.argv == ['pytest', '-q', 'tests']
        assert re.fullmatch(r'guru-sb-[0-9a-f]{12}', res.name)
        call, = fake_run.calls
        assert call['argv'] == [
            'docker', 'run', '--rm', '--name', res.name,
            '--stop-timeout', str(colima.STOP_TIMEOUT_S),
            '--network', 'none', '--user', '1000:1000',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--read-only', '--tmpfs', '/tmp',
            '--pids-limit', '256', '--memory', '2048m', '--cpus', '2.0',
            '-v', f'{copy}:/work', '-w', '/work',
            spec.image_tag, 'pytest', '-q', 'tests']
        assert res.docker_argv == call['argv']
        assert call['cwd'] == spec.project
        assert call['limits'].timeout_s == spec.timeout_s
        assert call['env']['DOCKER_CONFIG']

    def test_limits_come_from_the_spec(self, fake_run, sbhome, tmp_path):
        spec = sb.spec_from(_project(tmp_path), SandboxSettings(
            base_image=PINNED, cpus=0.5, memory_mb=300, pids=7,
            timeout_s=9))
        colima.run(spec, ['python', '-c', 'x'], tmp_path)
        argv = fake_run.calls[0]['argv']
        for flag, value in (('--pids-limit', '7'), ('--memory', '300m'),
                            ('--cpus', '0.5')):
            assert argv[argv.index(flag) + 1] == value
        assert fake_run.calls[0]['limits'].timeout_s == 9

    def test_names_are_unique(self) -> None:
        names = {colima.container_name() for _ in range(50)}
        assert len(names) == 50
        assert all(n.startswith('guru-sb-') for n in names)

    @pytest.mark.parametrize('argv', [['bash', '-c', 'x'], ['sh', 'x.sh'],
                                      ['pip', 'install', 'y'], ['curl', 'u'],
                                      ['rm', '-rf', '/'], []])
    def test_argv0_outside_the_allow_list_never_starts(
            self, fake_run, sbhome, tmp_path, argv) -> None:
        res = colima.run(_spec(tmp_path), argv, tmp_path)
        assert res.returncode == -1
        assert res.denied.startswith(procs.DENIED_PREFIX)
        assert fake_run.calls == []

    def test_model_text_stays_after_the_fixed_prefix(
            self, fake_run, sbhome, tmp_path) -> None:
        spec = _spec(tmp_path)
        colima.run(spec, ['python', '--privileged', '-v', '/:/host'],
                   tmp_path)
        argv = fake_run.calls[0]['argv']
        assert argv[argv.index(spec.image_tag) + 1:] == [
            'python', '--privileged', '-v', '/:/host']
        assert argv.index('--network') < argv.index(spec.image_tag)

    def test_timeout_kills_the_container_by_name(self, fake_run, sbhome,
                                                 tmp_path) -> None:
        fake_run.answers['run'] = ('timeout', '')
        fake_run.answers['kill'] = (0, '')
        res = colima.run(_spec(tmp_path), ['pytest'], tmp_path)
        assert res.timed_out and res.killed and res.returncode == -1
        run_call, kill_call = fake_run.calls
        assert kill_call['argv'] == ['docker', 'kill', res.name]
        assert kill_call['limits'].timeout_s == colima.KILL_TIMEOUT_S

    def test_no_kill_without_timeout(self, fake_run, sbhome, tmp_path):
        fake_run.answers['run'] = (3, 'failed test')
        res = colima.run(_spec(tmp_path), ['pytest'], tmp_path)
        assert res.returncode == 3 and not res.killed
        assert len(fake_run.calls) == 1


@pytest.fixture
def allowed(tmp_path, monkeypatch):
    """The temp dir under the read allow-list so git runs through
    procs.run; the prompt denies any escalation."""
    monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
    monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
    files.set_path_asker(lambda question: False)
    try:
        yield tmp_path
    finally:
        files.set_path_asker(None)


class TestPrepareCopyAndDiff:
    def _rich_project(self, tmp_path, marker_scanner=False) -> Path:
        root = _project(tmp_path)
        (root / '.env').write_text('SECRET=1\n', encoding='utf-8')
        (root / '.env.local').write_text('S=2\n', encoding='utf-8')
        (root / '.venv' / 'lib').mkdir(parents=True)
        (root / '.venv' / 'lib' / 'x.py').write_text('x', encoding='utf-8')
        (root / 'pkg' / '__pycache__').mkdir()
        (root / 'pkg' / '__pycache__' / 'a.pyc').write_bytes(b'\x00')
        (root / 'pkg' / 'secret.py').write_text('K = "MARKER_SECRET"\n',
                                                encoding='utf-8')
        (root / 'README.md').write_text('# pkg\n', encoding='utf-8')
        return root

    def test_copy_applies_excludes_and_is_a_git_repo(
            self, allowed, marker_scanner) -> None:
        root = self._rich_project(allowed)
        dest = allowed / 'copies' / 'one'
        excludes = sb.copy_excludes(root)
        got = colima.prepare_copy(root, dest, excludes)
        assert got == dest
        assert (dest / 'pkg' / '__init__.py').is_file()
        assert (dest / 'README.md').is_file()
        assert (dest / 'pyproject.toml').is_file()
        for gone in ('.env', '.env.local', '.venv', 'pkg/__pycache__',
                     'pkg/secret.py'):
            assert not (dest / gone).exists(), gone
        assert (dest / '.git' / 'HEAD').is_file()      # fresh init
        assert (dest / colima.COPY_MARKER).is_file()
        assert colima.diff(dest, project=root) == ''  # clean baseline

    def _git(self, root: Path, *args: str) -> None:
        res = colima._git(list(args), root, root)
        assert res.returncode == 0, res.stderr

    def test_git_repo_copies_the_positive_list_only(self, allowed) -> None:
        root = _project(allowed)
        (root / '.gitignore').write_text('secrets.yaml\n*.log\nbuild/\n',
                                         encoding='utf-8')
        (root / 'secrets.yaml').write_text('token: abc\n', encoding='utf-8')
        (root / 'debug.log').write_text('x\n', encoding='utf-8')
        (root / 'build').mkdir()
        (root / 'build' / 'out.bin').write_bytes(b'\x00')
        (root / 'notes.md').write_text('untracked, kept\n', encoding='utf-8')
        self._git(root, 'init', '-q')
        self._git(root, 'add', 'pyproject.toml', 'uv.lock', 'pkg',
                  '.gitignore')
        self._git(root, 'commit', '-q', '-m', 'base')
        (root / 'pkg' / 'gone.py').write_text('x', encoding='utf-8')
        self._git(root, 'add', 'pkg/gone.py')
        (root / 'pkg' / 'gone.py').unlink()          # tracked but deleted
        (root / 'link').symlink_to('pkg/__init__.py')
        dest = colima.prepare_copy(root, allowed / 'copy',
                                   colima.copy_excludes_for(root))
        assert (dest / 'pkg' / '__init__.py').is_file()
        assert (dest / 'notes.md').is_file()          # untracked, not ignored
        assert (dest / '.gitignore').is_file()
        assert (dest / 'link').is_symlink()
        for gone in ('secrets.yaml', 'debug.log', 'build', 'pkg/gone.py'):
            assert not (dest / gone).exists(), gone
        assert 'base' not in colima._git(['log', '--oneline'], dest,
                                         root).stdout   # fresh history
        assert colima.diff(dest, project=root) == ''

    def test_broken_git_dir_fails_closed(self, allowed) -> None:
        root = _project(allowed)
        (root / '.git').mkdir()
        (root / '.git' / 'HEAD').write_text('ref: x\n', encoding='utf-8')
        with pytest.raises(RuntimeError, match='ls-files'):
            colima.prepare_copy(root, allowed / 'copy', [])
        assert not (allowed / 'copy').exists()

    def test_flagged_paths_match_exactly_names_by_glob(self, allowed):
        root = _project(allowed)
        (root / 'pkg' / 'secrets[1].py').write_text('K = 1\n')
        (root / 'pkg' / 'secrets1.py').write_text('K = 2\n')
        (root / 'pkg' / 'other.pyc').write_text('x')
        excludes = ['pkg/secrets[1].py', '*.pyc']
        dest = colima.prepare_copy(root, allowed / 'copy', excludes)
        assert not (dest / 'pkg' / 'secrets[1].py').exists()
        assert (dest / 'pkg' / 'secrets1.py').is_file()
        assert not (dest / 'pkg' / 'other.pyc').exists()
        # Same rule on the git positive-list path.
        self._git(root, 'init', '-q')
        self._git(root, 'add', '-A')
        self._git(root, 'commit', '-q', '-m', 'base')
        dest2 = colima.prepare_copy(root, allowed / 'copy2', excludes)
        assert not (dest2 / 'pkg' / 'secrets[1].py').exists()
        assert (dest2 / 'pkg' / 'secrets1.py').is_file()

    def test_unbound_scanner_still_excludes_secrets(self, allowed,
                                                    monkeypatch) -> None:
        policy.set_scanner(None)
        monkeypatch.setattr(config, 'SENSITIVE_MARKERS_PATH',
                            allowed / 'none' / 'markers.txt')
        monkeypatch.setattr(config, 'SCAN_ALLOW_PATH',
                            allowed / 'none' / 'allow.txt')
        root = _project(allowed)
        (root / 'pkg' / 'creds.py').write_text(
            'AWS_KEY = "AKIA' + 'A1B2C3D4E5F6G7H8' + '"\n', encoding='utf-8')
        assert sb.flagged_files(root) == {}         # domain: no scanner
        excludes = colima.copy_excludes_for(root)   # endpoint binds one
        assert 'pkg/creds.py' in excludes
        dest = colima.prepare_copy(root, allowed / 'copy', excludes)
        assert not (dest / 'pkg' / 'creds.py').exists()
        assert (dest / 'pkg' / '__init__.py').is_file()

    def test_refuses_an_existing_dest(self, allowed) -> None:
        root = _project(allowed)
        (allowed / 'exists').mkdir()
        with pytest.raises(FileExistsError):
            colima.prepare_copy(root, allowed / 'exists', [])

    def test_diff_shows_edits_new_and_deleted_files(self, allowed) -> None:
        root = _project(allowed)
        dest = colima.prepare_copy(root, allowed / 'copy', [])
        (dest / 'pkg' / '__init__.py').write_text('X = 2\n', encoding='utf-8')
        (dest / 'pkg' / 'new.py').write_text('Y = 1\n', encoding='utf-8')
        (dest / 'uv.lock').unlink()
        (dest / 'pkg' / '__pycache__').mkdir()
        (dest / 'pkg' / '__pycache__' / 'z.pyc').write_bytes(b'\x00')
        text = colima.diff(dest, project=root)
        assert '-X = 1' in text and '+X = 2' in text
        assert 'b/pkg/new.py' in text and '+Y = 1' in text
        assert 'a/uv.lock' in text and 'deleted file' in text
        assert '__pycache__' not in text
        assert text == colima.diff(dest, project=root)  # idempotent

    def test_remove_copy_needs_the_marker(self, allowed) -> None:
        root = _project(allowed)
        dest = colima.prepare_copy(root, allowed / 'copy', [])
        with pytest.raises(ValueError, match='refusing'):
            colima.remove_copy(root)
        assert root.exists()
        colima.remove_copy(dest)
        assert not dest.exists()


# --- /sandbox status ---------------------------------------------------------

class TestSandboxStatusCommand:
    @pytest.fixture
    def project(self, tmp_path, monkeypatch, sbhome):
        root = _project(tmp_path)
        (root / '.guru').mkdir()
        monkeypatch.setattr(config, 'PROJECT_GURU_DIR', root / '.guru')
        monkeypatch.setattr(config, 'SANDBOX_POLICY_PATH',
                            root / '.guru' / 'sandbox.toml')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH',
                            tmp_path / 'no-settings.toml')
        return root

    def test_status_lines(self, project, monkeypatch, capsys) -> None:
        import guru.cli as cli
        monkeypatch.setattr(colima, 'available', lambda *a, **k: True)
        (project / '.guru' / 'sandbox.toml').write_text(
            '[sandbox]\nenabled = true\n', encoding='utf-8')
        cli._sandbox_command('status')
        out = capsys.readouterr().out
        assert 'docker available' in out
        assert 'present' in out and 'enabled: yes' in out
        assert 'guru-sandbox/proj-' in out
        assert 'not built' in out and 'needs build: yes' in out

    def test_recorded_image(self, project, monkeypatch, capsys) -> None:
        import guru.cli as cli
        from guru.repositories.settings import load_sandbox
        monkeypatch.setattr(colima, 'available', lambda *a, **k: False)
        spec = sb.spec_from(project, load_sandbox({}, project / 'none'))
        text = sb.dockerfile_for(spec.project, spec.base_image)
        images.record_built(spec, text, 'sha256:cafe')
        cli._sandbox_command('')
        out = capsys.readouterr().out
        assert 'unavailable' in out and 'enabled: no' in out
        assert 'sha256:cafe' in out and 'needs build: no' in out

    def test_no_lockfile_and_bad_settings(self, project, monkeypatch,
                                          capsys) -> None:
        import guru.cli as cli
        monkeypatch.setattr(colima, 'available', lambda *a, **k: True)
        (project / 'uv.lock').unlink()
        cli._sandbox_command()
        assert 'uv.lock' in capsys.readouterr().out
        (project / '.guru' / 'sandbox.toml').write_text(
            '[sandbox]\ncpus = "many"\n', encoding='utf-8')
        cli._sandbox_command()
        assert 'invalid' in capsys.readouterr().out

    def test_unknown_subcommand(self, project, capsys) -> None:
        import guru.cli as cli
        cli._sandbox_command('build')
        assert 'usage' in capsys.readouterr().out.lower()
