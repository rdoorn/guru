"""Tests for the sandbox provisioning phase (S2): the dependency domain
(``guru.domain.deps``), the tinyproxy allow-list/config rendering and the
fixed docker argv of ``guru.sandbox.proxy`` (fake ``procs.run``), build
args on ``colima.build``, ``net_events`` recording, the pending-request
store, and ``guru.sandbox.provision`` end to end with fakes."""
import json
import re
from pathlib import Path

import pytest

from guru import config, session
from guru.domain import deps, files, ledger, procs
from guru.domain import sandbox as sb
from guru.repositories import sandbox_images as images
from guru.repositories.settings import (DEFAULT_PROXY_IMAGE,
                                        SandboxSettings)
from guru.sandbox import colima, provision, proxy
from tests.conftest import FakeRepo
from tests.test_sandbox import PINNED, FakeRun, _project

OLD_LOCK = '''version = 1
requires-python = ">=3.12"

[[package]]
name = "pkg"
version = "0.1"
source = { virtual = "." }

[[package]]
name = "six"
version = "1.16.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "attrs"
version = "23.1.0"
source = { registry = "https://pypi.org/simple" }
'''
NEW_LOCK = '''version = 1
requires-python = ">=3.12"

[[package]]
name = "pkg"
version = "0.1"
source = { virtual = "." }

[[package]]
name = "six"
version = "1.17.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "rich"
version = "13.7.0"
source = { registry = "https://pypi.org/simple" }
'''


@pytest.fixture
def sbhome(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'SANDBOX_HOME', tmp_path / 'sbhome')
    return tmp_path / 'sbhome'


@pytest.fixture
def fake_run(monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(procs, 'run', fake)
    colima.reset_cache()
    yield fake
    colima.reset_cache()


@pytest.fixture
def repo(monkeypatch):
    fake = FakeRepo()
    ledger.set_repository(fake)
    monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
    try:
        yield fake
    finally:
        ledger.flush()
        ledger.set_repository(None)


# --- guru.domain.deps --------------------------------------------------------

class TestDependencyRequest:
    @pytest.mark.parametrize('name', ['six', 'Django', 'zope.interface',
                                      'ruamel_yaml', 'a', 'A1-b2_c3.d4'])
    def test_valid_names(self, name) -> None:
        assert deps.check_name(name) == ''

    @pytest.mark.parametrize('name', ['', ' six', 'six ', '-six', 'six-',
                                      'six[extra]', 'six>=1', 'a b', '.x',
                                      'x' * 201, 'ünicode', 'pkg;rm'])
    def test_invalid_names(self, name) -> None:
        assert deps.check_name(name) != ''

    @pytest.mark.parametrize('constraint', ['', '==1.0', '>=1.2,<2',
                                            '~=1.4.2', '!=1.*', '>1,<=3',
                                            '===foobar', '>=1.0rc1'])
    def test_valid_constraints(self, constraint) -> None:
        assert deps.check_constraint(constraint) == ''

    @pytest.mark.parametrize('constraint', ['1.0', '>= 1.2', '>=1.2, <2',
                                            '=1.0', '>=1;python', '@ file',
                                            '--index-url x', '>=1.2\n', ',',
                                            '>=', '>=' + '1' * 200])
    def test_invalid_constraints(self, constraint) -> None:
        assert deps.check_constraint(constraint) != ''

    def test_request_from(self) -> None:
        req = deps.request_from('Six', '>=1.16')
        assert req.name == 'Six' and req.constraint == '>=1.16'
        assert req.spec == 'Six>=1.16'
        assert req.key == 'six'
        assert deps.request_from('ruamel.YAML').key == 'ruamel-yaml'
        with pytest.raises(ValueError, match='name'):
            deps.request_from('six extra')
        with pytest.raises(ValueError, match='constraint'):
            deps.request_from('six', '1.0')


class TestLockDiff:
    def test_packages(self) -> None:
        assert deps.lock_packages(OLD_LOCK) == {
            'pkg': '0.1', 'six': '1.16.0', 'attrs': '23.1.0'}
        assert deps.lock_packages('') == {}

    def test_summary(self) -> None:
        summary = deps.lock_diff_summary(OLD_LOCK, NEW_LOCK)
        assert summary == {'added': ['rich==13.7.0'],
                           'removed': ['attrs==23.1.0'],
                           'changed': ['six: 1.16.0 -> 1.17.0']}
        text = deps.summary_text(summary)
        assert 'added rich==13.7.0' in text
        assert 'removed attrs==23.1.0' in text
        assert 'changed six: 1.16.0 -> 1.17.0' in text
        assert deps.summary_text(deps.lock_diff_summary(OLD_LOCK, OLD_LOCK)
                                 ) == 'no package changes'

    def test_invalid_lock_raises(self) -> None:
        with pytest.raises(ValueError):
            deps.lock_packages('[[package]\nname = 1')

    def test_unified_diff_applies_with_apply_patch(self, tmp_path,
                                                   monkeypatch) -> None:
        from guru.domain import patch
        monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
        monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', {str(tmp_path)})
        monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
        monkeypatch.setattr(session, 'file_shas', {})
        monkeypatch.setattr(files, '_show_change', lambda block: None)
        target = tmp_path / 'uv.lock'
        target.write_text(OLD_LOCK, encoding='utf-8')
        diff = deps.unified_diff(target, OLD_LOCK, NEW_LOCK)
        assert diff.startswith(f'--- {target}\n+++ {target}\n')
        assert '+version = "1.17.0"' in diff
        assert deps.unified_diff(target, OLD_LOCK, OLD_LOCK) == ''
        out = patch.apply_patch(diff)
        assert out.startswith('Applied patch'), out
        assert target.read_text(encoding='utf-8') == NEW_LOCK


# --- guru.sandbox.proxy: rendering -------------------------------------------

class TestAllowlist:
    def test_lowercased_sorted_deduplicated(self) -> None:
        got = proxy.allowlist_from_domains(
            {'PyPI.org', 'files.pythonhosted.org', ' pypi.org ', ''})
        assert got == ['files.pythonhosted.org', 'pypi.org']

    def test_no_implicit_additions(self) -> None:
        # pypi.org alone does NOT pull in files.pythonhosted.org: both must
        # be approved by the user.
        assert proxy.allowlist_from_domains({'pypi.org'}) == ['pypi.org']

    @pytest.mark.parametrize('bad', ['exa mple.com', 'a/b', 'x;y', '-a.com',
                                     'a..b', 'host:443', '^.*$', 'a_b.com'])
    def test_malformed_entries_are_dropped(self, bad) -> None:
        assert proxy.allowlist_from_domains({bad, 'ok.example'}) == [
            'ok.example']


class TestRenderConfig:
    def test_filter_lists_only_the_allowed_hosts_on_443(self) -> None:
        text = proxy.render_filter(['pypi.org', 'files.pythonhosted.org'])
        assert text.splitlines() == [r'^pypi\.org:443$',
                                     r'^files\.pythonhosted\.org:443$']
        # '-' stays bare: '\-' outside brackets is undefined in POSIX ERE.
        assert proxy.render_filter(['test-host.example']) == (
            '^test-host\\.example:443$\n')
        assert 'example.com' not in text
        # Plain HTTP URLs never match: the patterns are anchored host:port.
        assert 'http' not in text

    def test_config_denies_by_default_and_connects_443_only(self) -> None:
        text = proxy.render_config(['pypi.org'])
        for line in ('Port 8888', 'Listen 0.0.0.0', 'FilterDefaultDeny Yes',
                     'FilterURLs On', 'FilterType ere', 'ConnectPort 443',
                     f'Filter "{proxy.CONFIG_MOUNT}/{proxy.FILTER_FILE}"',
                     'LogLevel Connect', 'DisableViaHeader Yes'):
            assert line in text.splitlines(), line
        assert 'LogFile' not in text            # logs go to docker logs
        assert 'Upstream' not in text
        assert 'ConnectPort 80' not in text
        assert 'Allow ' not in text             # no client CIDR given

    def test_client_cidr_restricts_clients(self) -> None:
        text = proxy.render_config(['pypi.org'], client_cidr='172.25.0.0/16')
        assert 'Allow 172.25.0.0/16' in text.splitlines()
        with pytest.raises(ValueError):
            proxy.render_config(['pypi.org'], client_cidr='not a cidr')

    def test_write_config(self, tmp_path) -> None:
        conf = proxy.write_config(tmp_path / 'p', ['pypi.org'],
                                  client_cidr='10.0.0.0/24')
        assert conf == tmp_path / 'p' / proxy.CONF_FILE
        assert (tmp_path / 'p' / proxy.FILTER_FILE).read_text().strip() == (
            r'^pypi\.org:443$')
        assert 'Allow 10.0.0.0/24' in conf.read_text()


# --- guru.sandbox.proxy: docker argv -----------------------------------------

class TestNetwork:
    def test_up_down_and_subnet_argv(self, fake_run, tmp_path) -> None:
        fake_run.answers['network'] = (0, '172.25.0.0/16\n')
        proxy.network_up('guru-provision-x', cwd=tmp_path)
        assert proxy.network_subnet('guru-provision-x', cwd=tmp_path) == (
            '172.25.0.0/16')
        proxy.network_down('guru-provision-x', cwd=tmp_path)
        argvs = [c['argv'] for c in fake_run.calls]
        assert argvs == [
            ['docker', 'network', 'create', '--internal', 'guru-provision-x'],
            ['docker', 'network', 'inspect', '--format',
             '{{(index .IPAM.Config 0).Subnet}}', 'guru-provision-x'],
            ['docker', 'network', 'rm', 'guru-provision-x']]
        assert all(c['cwd'] == tmp_path for c in fake_run.calls)
        assert all(c['env']['DOCKER_CONFIG'] for c in fake_run.calls)

    @staticmethod
    def _existing(internal: str):
        calls: list = []

        def run(argv, cwd, limits=None, env_extra=None):
            calls.append(list(argv))
            if argv[1:3] == ['network', 'create']:
                return procs.ProcResult(list(argv), 1, '', 'Error response '
                                        'from daemon: network with name x '
                                        'already exists', 0.1)
            if argv[1:3] == ['network', 'inspect']:
                return procs.ProcResult(list(argv), 0, internal + '\n', '',
                                        0.1)
            return procs.ProcResult(list(argv), 0, '', '', 0.1)
        return run, calls

    def test_up_reuses_an_existing_internal_network(
            self, monkeypatch, tmp_path) -> None:
        run, calls = self._existing('true')
        monkeypatch.setattr(procs, 'run', run)
        proxy.network_up('x', cwd=tmp_path)             # no raise
        assert calls[1] == ['docker', 'network', 'inspect', '--format',
                            '{{.Internal}}', 'x']

    def test_up_refuses_an_existing_routable_network(
            self, monkeypatch, tmp_path) -> None:
        run, calls = self._existing('false')
        monkeypatch.setattr(procs, 'run', run)
        with pytest.raises(proxy.ProxyError, match='not internal'):
            proxy.network_up('x', cwd=tmp_path)

    def test_up_failure_raises(self, fake_run, tmp_path) -> None:
        fake_run.answers['network'] = (1, '')
        with pytest.raises(proxy.ProxyError, match='network create'):
            proxy.network_up('x', cwd=tmp_path)


class TestProxyContainer:
    def test_up_argv_in_order(self, fake_run, tmp_path) -> None:
        fake_run.answers['inspect'] = (0, 'true\n')
        name = proxy.proxy_up('guru-proxy-x', 'guru-provision-x',
                              tmp_path / 'cfg', DEFAULT_PROXY_IMAGE,
                              cwd=tmp_path)
        assert name == 'guru-proxy-x'
        argvs = [c['argv'] for c in fake_run.calls]
        assert argvs[0] == ['docker', 'rm', '-f', 'guru-proxy-x']
        assert argvs[1] == [
            'docker', 'run', '-d', '--rm', '--name', 'guru-proxy-x',
            '--network', 'guru-provision-x',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--read-only', '--tmpfs', '/tmp',
            '--pids-limit', str(proxy.PROXY_PIDS),
            '--memory', f'{proxy.PROXY_MEMORY_MB}m',
            '-v', f'{tmp_path / "cfg"}:{proxy.CONFIG_MOUNT}:ro',
            DEFAULT_PROXY_IMAGE,
            proxy.TINYPROXY_BIN, '-d', '-c',
            f'{proxy.CONFIG_MOUNT}/{proxy.CONF_FILE}']
        assert argvs[2] == ['docker', 'network', 'connect', 'bridge',
                            'guru-proxy-x']
        assert argvs[3] == ['docker', 'inspect', '--format',
                            '{{.State.Running}}', 'guru-proxy-x']
        assert len(argvs) == 4
        proxy.proxy_down('guru-proxy-x', cwd=tmp_path)
        assert fake_run.calls[-1]['argv'] == ['docker', 'rm', '-f',
                                              'guru-proxy-x']

    def test_up_failure_tears_down_and_raises(self, fake_run, tmp_path):
        fake_run.answers['run'] = (125, '')
        with pytest.raises(proxy.ProxyError, match='proxy'):
            proxy.proxy_up('p', 'n', tmp_path, DEFAULT_PROXY_IMAGE,
                           cwd=tmp_path)
        assert fake_run.calls[-1]['argv'] == ['docker', 'rm', '-f', 'p']

    def test_not_running_after_start_raises(self, fake_run, tmp_path):
        fake_run.answers['inspect'] = (0, 'false\n')
        with pytest.raises(proxy.ProxyError, match='not running'):
            proxy.proxy_up('p', 'n', tmp_path, DEFAULT_PROXY_IMAGE,
                           cwd=tmp_path)


SAMPLE_LOG = '''NOTICE    Sep 24 19:39:36.888 [1]: Initializing tinyproxy ...
CONNECT   Sep 24 19:39:43.250 [1]: Connect (file descriptor 4): 172.25.0.3
CONNECT   Sep 24 19:39:43.250 [1]: Request (file descriptor 4): CONNECT pypi.org:443 HTTP/1.1
CONNECT   Sep 24 19:39:43.261 [1]: Established connection to host "pypi.org" using file descriptor 5.
CONNECT   Sep 24 19:39:43.530 [1]: Connect (file descriptor 4): 172.25.0.3
CONNECT   Sep 24 19:39:43.530 [1]: Request (file descriptor 4): CONNECT example.com:443 HTTP/1.1
NOTICE    Sep 24 19:39:43.530 [1]: Proxying refused on filtered url "example.com:443"
CONNECT   Sep 24 19:39:43.696 [1]: Request (file descriptor 4): GET http://pypi.org/simple/ HTTP/1.1
NOTICE    Sep 24 19:39:43.696 [1]: Proxying refused on filtered url "http://pypi.org/simple/"
CONNECT   Sep 24 19:39:43.869 [1]: Request (file descriptor 4): CONNECT pypi.org:8443 HTTP/1.1
CONNECT   Sep 24 19:39:44.000 [1]: Request (file descriptor 6): CONNECT files.pythonhosted.org:443 HTTP/1.1
CONNECT   Sep 24 19:39:44.001 [1]: Request (file descriptor 7): CONNECT files.pythonhosted.org:443 HTTP/1.1
CONNECT   Sep 24 19:39:44.010 [1]: Established connection to host "files.pythonhosted.org" using file descriptor 8.
'''  # noqa: E501


class TestAccessLog:
    def test_parse_pairs_requests_with_outcomes(self) -> None:
        rows = proxy.parse_log(SAMPLE_LOG)
        assert [(r['method'], r['host'], r['port'], r['allowed'])
                for r in rows] == [
            ('CONNECT', 'pypi.org', 443, True),
            ('CONNECT', 'example.com', 443, False),
            ('GET', 'pypi.org', 80, False),
            ('CONNECT', 'pypi.org', 8443, False),
            ('CONNECT', 'files.pythonhosted.org', 443, True),
            ('CONNECT', 'files.pythonhosted.org', 443, False)]
        assert rows[1]['reason'] == 'filtered'
        assert rows[2]['reason'] == 'filtered'
        assert rows[3]['reason'] == 'no upstream connection'
        assert rows[0]['reason'] == '' and rows[0]['ts'].startswith('Sep 24')
        assert proxy.parse_log('') == []

    def test_tail_argv_and_recording(self, fake_run, tmp_path, repo) -> None:
        fake_run.answers['logs'] = (0, SAMPLE_LOG)
        rows = proxy.tail_access_log('guru-proxy-x', since='2026-09-24T00:00'
                                     ':00Z', cwd=tmp_path)
        assert fake_run.calls[0]['argv'] == [
            'docker', 'logs', '--since', '2026-09-24T00:00:00Z',
            'guru-proxy-x']
        assert len(rows) == 6
        proxy.tail_access_log('guru-proxy-x', cwd=tmp_path)
        assert fake_run.calls[1]['argv'] == ['docker', 'logs', 'guru-proxy-x']
        n = images.record_net_events(rows, phase='build')
        ledger.flush()
        written = repo.stream('net_events')
        assert n == len(written) == 6          # aggregated per outcome
        by_key = {(r['host'], r['port'], r['allowed']): r for r in written}
        assert by_key[('files.pythonhosted.org', 443, True)]['count'] == 1
        assert by_key[('files.pythonhosted.org', 443, False)]['count'] == 1
        assert by_key[('pypi.org', 443, True)]['count'] == 1
        assert by_key[('pypi.org', 80, False)]['method'] == 'GET'
        assert all(r['phase'] == 'build' for r in written)
        assert {'ts', 'run_id', 'project', 'agent', 'task_id', 'turn_id',
                'reason', 'method'} <= set(written[0])

    def test_recording_never_raises(self, monkeypatch) -> None:
        class Boom:
            def append(self, stream, row):
                raise RuntimeError('x')
        ledger.set_repository(Boom())
        monkeypatch.setattr(config, 'LEDGER_ENABLED', True)
        try:
            images.record_net_events([{'host': 1}], phase='x')
            ledger.flush()
        finally:
            ledger.set_repository(None)


# --- colima.build with build args; Dockerfile ARGs ---------------------------

class TestBuildArgs:
    def test_proxy_build_args_cover_both_cases_and_clear_no_proxy(self):
        args = sb.proxy_build_args('http://guru-proxy-x:8888')
        assert args == {'HTTP_PROXY': 'http://guru-proxy-x:8888',
                        'HTTPS_PROXY': 'http://guru-proxy-x:8888',
                        'http_proxy': 'http://guru-proxy-x:8888',
                        'https_proxy': 'http://guru-proxy-x:8888',
                        'NO_PROXY': '', 'no_proxy': ''}

    def test_dockerfile_declares_no_proxy_args(self, tmp_path) -> None:
        # Docker predefines the proxy build args; declaring them with ARG
        # would record their values in `docker history`.
        text = sb.dockerfile_for(_project(tmp_path), PINNED)
        lines = text.splitlines()
        assert not any(ln.startswith('ARG ') for ln in lines)
        assert not any('PROXY' in ln.upper() for ln in lines)

    def test_build_argv_with_args_uses_the_legacy_builder(
            self, fake_run, sbhome, tmp_path) -> None:
        spec = sb.spec_from(_project(tmp_path), SandboxSettings(
            base_image=PINNED))
        dockerfile = images.write_dockerfile(
            spec, sb.dockerfile_for(spec.project, spec.base_image))
        fake_run.answers['image'] = (0, 'sha256:1\n')
        res = colima.build(spec, dockerfile, tmp_path / 'ctx',
                           network='guru-provision-pkg',
                           build_args={'HTTPS_PROXY': 'http://p:8888',
                                       'NO_PROXY': ''})
        assert res.ok
        call = fake_run.calls[0]
        assert call['argv'] == [
            'docker', 'build', '--network', 'guru-provision-pkg',
            '--build-arg', 'HTTPS_PROXY=http://p:8888',
            '--build-arg', 'NO_PROXY=',
            '-t', spec.image_tag, '-f', str(dockerfile),
            str(tmp_path / 'ctx')]
        # BuildKit refuses a named network; the classic builder takes it.
        assert call['env']['DOCKER_BUILDKIT'] == '0'
        assert call['env']['DOCKER_CONFIG']

    def test_default_network_keeps_buildkit(self, fake_run, sbhome, tmp_path):
        spec = sb.spec_from(_project(tmp_path), SandboxSettings(
            base_image=PINNED))
        dockerfile = images.write_dockerfile(
            spec, sb.dockerfile_for(spec.project, spec.base_image))
        colima.build(spec, dockerfile, tmp_path)
        assert 'DOCKER_BUILDKIT' not in fake_run.calls[0]['env']
        assert '--build-arg' not in fake_run.calls[0]['argv']

    @pytest.mark.parametrize('key', ['A B', 'A=B', '', '-x', 'a\nb'])
    def test_bad_build_arg_names_are_refused(self, fake_run, sbhome,
                                             tmp_path, key) -> None:
        spec = sb.spec_from(_project(tmp_path), SandboxSettings(
            base_image=PINNED))
        dockerfile = images.write_dockerfile(
            spec, sb.dockerfile_for(spec.project, spec.base_image))
        with pytest.raises(ValueError):
            colima.build(spec, dockerfile, tmp_path, network='n',
                         build_args={key: 'v'})
        assert fake_run.calls == []


class TestRunNetworkAndEnv:
    def test_run_on_a_network_with_env(self, fake_run, sbhome, tmp_path):
        spec = sb.spec_from(_project(tmp_path), SandboxSettings(
            base_image=PINNED))
        res = colima.run(spec, ['uv', 'add', 'six'], tmp_path / 'c',
                         network='guru-provision-pkg',
                         env={'HTTPS_PROXY': 'http://p:8888',
                              'UV_OFFLINE': '0'})
        argv = fake_run.calls[0]['argv']
        assert argv[argv.index('--network') + 1] == 'guru-provision-pkg'
        i = argv.index('-w') + 2
        assert argv[i:i + 4] == ['-e', 'HTTPS_PROXY=http://p:8888',
                                 '-e', 'UV_OFFLINE=0']
        assert argv[argv.index(spec.image_tag) + 1:] == ['uv', 'add', 'six']
        assert res.returncode == 0

    def test_default_stays_network_none_without_env(self, fake_run, sbhome,
                                                    tmp_path) -> None:
        spec = sb.spec_from(_project(tmp_path), SandboxSettings(
            base_image=PINNED))
        colima.run(spec, ['pytest'], tmp_path)
        argv = fake_run.calls[0]['argv']
        assert argv[argv.index('--network') + 1] == 'none'
        assert '-e' not in argv


# --- pending requests store --------------------------------------------------

class TestPendingStore:
    def test_add_list_remove(self, sbhome, tmp_path) -> None:
        spec = sb.spec_from(_project(tmp_path), SandboxSettings(
            base_image=PINNED))
        assert images.pending_requests(spec) == []
        req = deps.request_from('six', '>=1.16')
        images.add_request(spec, req)
        images.add_request(spec, deps.request_from('rich'))
        images.add_request(spec, deps.request_from('SIX', '<2'))  # replaces
        pending = images.pending_requests(spec)
        assert [(r.name, r.constraint) for r in pending] == [
            ('rich', ''), ('SIX', '<2')]
        assert all(r.requested_at for r in pending)
        stored = json.loads((sbhome / spec.name / images.DEPS_FILE)
                            .read_text())
        assert {r['name'] for r in stored['requests']} == {'rich', 'SIX'}
        assert images.remove_request(spec, 'six') is True
        assert images.remove_request(spec, 'six') is False
        assert [r.name for r in images.pending_requests(spec)] == ['rich']

    def test_corrupt_store_is_empty(self, sbhome, tmp_path) -> None:
        spec = sb.spec_from(_project(tmp_path), SandboxSettings(
            base_image=PINNED))
        (sbhome / spec.name).mkdir(parents=True)
        (sbhome / spec.name / images.DEPS_FILE).write_text('{nope')
        assert images.pending_requests(spec) == []


# --- guru.sandbox.provision --------------------------------------------------

@pytest.fixture
def allowed(tmp_path, monkeypatch, sbhome):
    """A read+write allow-listed temp dir with the sandbox home inside it,
    cwd inside it, prompts denying, no domain persistence."""
    monkeypatch.setattr(config, 'ALLOWED_READ_DIRS', {str(tmp_path)})
    monkeypatch.setattr(config, 'ALLOWED_WRITE_DIRS', {str(tmp_path)})
    monkeypatch.setattr(config, 'MODE', config.MODE_ASK)
    monkeypatch.setattr(config, 'ALLOWED_DOMAINS', set())
    monkeypatch.setattr(config, 'persist_domain', lambda d: None)
    monkeypatch.setattr(config, 'persist_write_dir', lambda d: None)
    monkeypatch.setattr(session, 'file_shas', {})
    monkeypatch.setattr(files, '_show_change', lambda block: None)
    monkeypatch.chdir(tmp_path)
    files.set_path_asker(lambda q: False)
    try:
        yield tmp_path
    finally:
        files.set_path_asker(None)
        provision.set_approve_asker(None)


@pytest.fixture
def domain_yes(monkeypatch):
    from guru.domain import tools
    asked: list = []
    tools.set_domain_asker(lambda q: asked.append(q) or True)
    try:
        yield asked
    finally:
        tools.set_domain_asker(None)


def _settings() -> SandboxSettings:
    return SandboxSettings(enabled=True, base_image=PINNED, cpus=1.0,
                           memory_mb=512, pids=64, timeout_s=60)


class TestNames:
    def test_network_and_proxy_names_are_per_project_and_session(
            self, tmp_path) -> None:
        spec = sb.spec_from(_project(tmp_path), _settings())
        assert re.fullmatch(r'guru-provision-proj-[0-9a-f]{8}-[0-9a-f]{8}',
                            provision.network_name(spec))
        assert provision.proxy_name(spec) == (
            f'guru-proxy-{spec.name}-{provision.SESSION}')
        assert provision.proxy_url(spec) == (
            f'http://{provision.proxy_name(spec)}:8888')


class TestDiscard:
    def test_missing_marker_is_left_in_place(self, tmp_path) -> None:
        keep = tmp_path / 'real'
        keep.mkdir()
        (keep / 'f').write_text('x')
        provision._discard(keep)
        assert (keep / 'f').exists()
        provision._discard(None)

    def test_marked_copy_is_removed(self, tmp_path) -> None:
        copy = tmp_path / 'copy'
        copy.mkdir()
        (copy / colima.COPY_MARKER).write_text('x')
        provision._discard(copy)
        assert not copy.exists()


class TestProvision:
    def test_full_sequence_with_fakes(self, allowed, fake_run, domain_yes,
                                      repo) -> None:
        root = _project(allowed)
        fake_run.answers['network'] = (0, '172.25.0.0/16\n')
        fake_run.answers['inspect'] = (0, 'true\n')
        fake_run.answers['image'] = (0, 'sha256:built\n')
        fake_run.answers['logs'] = (0, SAMPLE_LOG)
        rec = provision.provision(root, _settings())
        assert rec.digest == 'sha256:built'
        spec = sb.spec_from(root, _settings())
        # Both PyPI hosts were approved (asked, ask mode) and persisted.
        assert sorted(domain_yes) == [
            "Allow web access to 'files.pythonhosted.org'?",
            "Allow web access to 'pypi.org'?"]
        assert {'pypi.org',
                'files.pythonhosted.org'} <= config.ALLOWED_DOMAINS
        docker = [c['argv'][1:] for c in fake_run.calls
                  if c['argv'][0] == 'docker']
        kinds = [' '.join(a[:2]) for a in docker]
        net, pxy = provision.network_name(spec), provision.proxy_name(spec)
        assert kinds == ['network create', 'network inspect', 'rm -f',
                         'run -d', 'network connect', 'inspect --format',
                         'build --network', 'image inspect', f'logs {pxy}',
                         'rm -f', 'network rm']
        build = next(a for a in docker if a[0] == 'build')
        assert build[1:3] == ['--network', net]
        url = f'http://{pxy}:{proxy.PROXY_PORT}'
        assert f'HTTPS_PROXY={url}' in build and f'http_proxy={url}' in build
        assert 'NO_PROXY=' in build
        assert build[-1].startswith(str(images.work_root(spec)))
        # The proxy config carries the allow-list and the client subnet.
        cfg_dir = images.record_dir(spec) / provision.PROXY_DIR
        assert (cfg_dir / proxy.FILTER_FILE).read_text().splitlines() == [
            r'^files\.pythonhosted\.org:443$', r'^pypi\.org:443$']
        conf = (cfg_dir / proxy.CONF_FILE).read_text()
        assert 'Allow 172.25.0.0/16' in conf
        # The build context copy is gone; the record is saved.
        assert list(images.work_root(spec).iterdir()) == []
        text = sb.dockerfile_for(root, PINNED)
        assert images.needs_build(spec, text) is False
        ledger.flush()
        assert len(repo.stream('net_events')) == 6
        kinds = [r['kind'] for r in repo.stream('sandbox_events')]
        assert 'build' in kinds and 'provision' in kinds

    def test_skips_when_the_record_is_current(self, allowed, fake_run,
                                              domain_yes) -> None:
        root = _project(allowed)
        spec = sb.spec_from(root, _settings())
        text = sb.dockerfile_for(root, PINNED)
        images.record_built(spec, text, 'sha256:old')
        rec = provision.provision(root, _settings())
        assert rec.digest == 'sha256:old'
        assert fake_run.calls == [] and domain_yes == []

    def test_force_rebuilds(self, allowed, fake_run, domain_yes) -> None:
        root = _project(allowed)
        spec = sb.spec_from(root, _settings())
        images.record_built(spec, sb.dockerfile_for(root, PINNED), 'sha256:o')
        fake_run.answers['inspect'] = (0, 'true\n')
        fake_run.answers['image'] = (0, 'sha256:new\n')
        assert provision.provision(root, _settings(), force=True).digest == (
            'sha256:new')

    def test_domain_denied_means_no_network_at_all(self, allowed, fake_run,
                                                   monkeypatch) -> None:
        from guru.domain import tools
        tools.set_domain_asker(lambda q: False)
        try:
            with pytest.raises(provision.ProvisionError, match='pypi.org'):
                provision.provision(_project(allowed), _settings())
        finally:
            tools.set_domain_asker(None)
        assert fake_run.calls == []

    def test_build_failure_tears_down_and_raises(self, allowed, fake_run,
                                                 domain_yes) -> None:
        root = _project(allowed)
        fake_run.answers['inspect'] = (0, 'true\n')
        fake_run.answers['build'] = (1, '')
        with pytest.raises(provision.ProvisionError, match='build failed'):
            provision.provision(root, _settings())
        tails = [' '.join(c['argv'][1:3]) for c in fake_run.calls[-2:]]
        assert tails == ['rm -f', 'network rm']
        spec = sb.spec_from(root, _settings())
        assert images.load_record(spec) is None
        assert list(images.work_root(spec).iterdir()) == []

    def test_proxy_failure_removes_the_network(self, allowed, fake_run,
                                               domain_yes) -> None:
        fake_run.answers['run'] = (125, '')
        with pytest.raises(provision.ProvisionError):
            provision.provision(_project(allowed), _settings())
        assert fake_run.calls[-1]['argv'][1:3] == ['network', 'rm']
        assert not any(c['argv'][1] == 'build' for c in fake_run.calls)


class TestRequestDependency:
    def test_records_without_installing(self, allowed, fake_run, repo):
        root = _project(allowed)
        out = provision.request_dependency('six', '>=1.16', project=root,
                                           settings=_settings())
        assert 'six>=1.16' in out and 'nothing was installed' in out
        spec = sb.spec_from(root, _settings())
        assert [r.spec for r in images.pending_requests(spec)] == ['six>=1.16']
        assert fake_run.calls == []
        ledger.flush()
        row, = repo.stream('sandbox_events')
        assert row['kind'] == 'dep_request' and row['ok'] is True

    def test_invalid_request_is_refused(self, allowed, fake_run) -> None:
        root = _project(allowed)
        out = provision.request_dependency('six; rm -rf', '', project=root,
                                           settings=_settings())
        assert out.startswith('Refused')
        out = provision.request_dependency('six', '1.0', project=root,
                                           settings=_settings())
        assert out.startswith('Refused') and 'constraint' in out
        assert images.pending_requests(
            sb.spec_from(root, _settings())) == []


class FakeUvAdd:
    """Stands in for ``colima.run``: a successful ``uv add`` that rewrites
    the copy's pyproject.toml and uv.lock, remembering the call."""

    def __init__(self, rc: int = 0) -> None:
        self.calls: list = []
        self.rc = rc

    def __call__(self, spec, argv, copy, network='none', env=None):
        self.calls.append({'argv': list(argv), 'copy': Path(copy),
                           'network': network, 'env': dict(env or {})})
        if self.rc == 0:
            py = Path(copy) / 'pyproject.toml'
            py.write_text(py.read_text().replace(
                'dependencies = []', 'dependencies = ["six>=1.16"]'))
            (Path(copy) / 'uv.lock').write_text(NEW_LOCK, encoding='utf-8')
        return colima.RunResult(list(argv), self.rc, '', 'uv failed'
                                if self.rc else '', 0.5)


class TestApplyDependency:
    @pytest.fixture
    def prepared(self, allowed, fake_run, domain_yes, monkeypatch):
        root = _project(allowed)
        (root / 'uv.lock').write_text(OLD_LOCK, encoding='utf-8')
        settings = _settings()
        spec = sb.spec_from(root, settings)
        images.record_built(spec, sb.dockerfile_for(root, PINNED), 'sha256:o')
        req = deps.request_from('six', '>=1.16')
        images.add_request(spec, req)
        fake_run.answers['network'] = (0, '10.9.0.0/16\n')
        fake_run.answers['inspect'] = (0, 'true\n')
        fake_run.answers['logs'] = (0, SAMPLE_LOG)
        uv = FakeUvAdd()
        monkeypatch.setattr(colima, 'run', uv)
        rebuilt: list = []

        def fake_provision(project, settings=None, force=False):
            rebuilt.append((Path(project), force))
            new_spec = sb.spec_from(project, settings)
            return images.record_built(
                new_spec, sb.dockerfile_for(project, PINNED), 'sha256:n')
        monkeypatch.setattr(provision, 'provision', fake_provision)
        return root, settings, req, uv, rebuilt

    def test_happy_path(self, prepared, repo) -> None:
        root, settings, req, uv, rebuilt = prepared
        questions: list = []
        provision.set_approve_asker(lambda q: questions.append(q) or True)
        out = provision.apply_dependency(root, req, settings)
        assert questions == ['Add dependency six>=1.16 to uv.lock and '
                             'rebuild the sandbox image?']
        call, = uv.calls
        assert call['argv'] == ['uv', 'add', '--no-sync', 'six>=1.16']
        spec = sb.spec_from(root, settings)
        assert call['network'] == provision.network_name(spec)
        url = f'http://{provision.proxy_name(spec)}:{proxy.PROXY_PORT}'
        assert call['env']['HTTPS_PROXY'] == url
        assert call['env']['HTTP_PROXY'] == url
        assert call['env']['UV_OFFLINE'] == '0'
        assert call['env']['UV_CACHE_DIR'].startswith('/tmp')
        assert call['copy'].parent == images.work_root(spec)
        assert not call['copy'].exists()                 # removed afterwards
        assert call['network'].endswith(provision.SESSION)
        # The real tree changed through apply_patch, the lockfile diff is
        # in the digest and the image was rebuilt.
        assert (root / 'uv.lock').read_text() == NEW_LOCK
        assert 'six>=1.16' in (root / 'pyproject.toml').read_text()
        assert 'added rich==13.7.0' in out
        assert 'removed attrs==23.1.0' in out
        assert 'changed six: 1.16.0 -> 1.17.0' in out
        assert 'sha256:n' in out
        assert rebuilt == [(root, False)]
        new_spec = sb.spec_from(root, settings)
        assert images.pending_requests(new_spec) == []
        ledger.flush()
        kinds = [r['kind'] for r in repo.stream('sandbox_events')]
        assert 'dep_apply' in kinds
        assert repo.stream('net_events')            # uv add traffic logged

    def test_decline_changes_nothing(self, prepared, fake_run) -> None:
        root, settings, req, uv, rebuilt = prepared
        provision.set_approve_asker(lambda q: False)
        before = fake_run.calls[:]
        out = provision.apply_dependency(root, req, settings)
        assert out.startswith('Declined')
        assert uv.calls == [] and rebuilt == []
        assert fake_run.calls == before
        assert (root / 'uv.lock').read_text() == OLD_LOCK
        spec = sb.spec_from(root, settings)
        assert [r.spec for r in images.pending_requests(spec)] == [
            'six>=1.16']                        # still pending

    def test_read_only_refuses(self, prepared, monkeypatch) -> None:
        root, settings, req, uv, rebuilt = prepared
        monkeypatch.setattr(config, 'MODE', config.MODE_READ_ONLY)
        provision.set_approve_asker(lambda q: True)
        out = provision.apply_dependency(root, req, settings)
        assert out.startswith('Refused') and uv.calls == []

    def test_auto_mode_does_not_ask(self, prepared, monkeypatch) -> None:
        root, settings, req, uv, rebuilt = prepared
        monkeypatch.setattr(config, 'MODE', config.MODE_AUTO)
        monkeypatch.setattr(config, 'AUTO_GRANT', True)
        provision.set_approve_asker(lambda q: False)
        out = provision.apply_dependency(root, req, settings)
        assert 'rebuilt' in out and len(uv.calls) == 1

    def test_uv_failure_leaves_the_tree(self, prepared, monkeypatch) -> None:
        root, settings, req, uv, rebuilt = prepared
        monkeypatch.setattr(colima, 'run', FakeUvAdd(rc=2))
        provision.set_approve_asker(lambda q: True)
        out = provision.apply_dependency(root, req, settings)
        assert out.startswith('uv add failed') and 'uv failed' in out
        assert (root / 'uv.lock').read_text() == OLD_LOCK and rebuilt == []

    def test_needs_a_built_image(self, allowed, fake_run) -> None:
        root = _project(allowed)
        provision.set_approve_asker(lambda q: True)
        out = provision.apply_dependency(root, deps.request_from('six'),
                                         _settings())
        assert 'not built' in out and fake_run.calls == []

    def test_default_asker_denies_on_eof(self, prepared, monkeypatch) -> None:
        root, settings, req, uv, rebuilt = prepared
        provision.set_approve_asker(None)
        monkeypatch.setattr('builtins.input',
                            lambda prompt='': (_ for _ in ()).throw(EOFError))
        assert provision.apply_dependency(root, req, settings).startswith(
            'Declined')


# --- /sandbox provision, /sandbox deps ---------------------------------------

class TestSandboxCommands:
    @pytest.fixture
    def cli_project(self, allowed, monkeypatch):
        root = _project(allowed)
        (root / '.guru').mkdir()
        monkeypatch.setattr(config, 'PROJECT_GURU_DIR', root / '.guru')
        monkeypatch.setattr(config, 'SANDBOX_POLICY_PATH',
                            root / '.guru' / 'sandbox.toml')
        monkeypatch.setattr(config, 'GLOBAL_SETTINGS_PATH',
                            allowed / 'no-settings.toml')
        (root / '.guru' / 'sandbox.toml').write_text(
            '[sandbox]\nenabled = true\n', encoding='utf-8')
        return root

    def test_provision_command(self, cli_project, monkeypatch, capsys):
        import guru.cli as cli
        seen: list = []

        def fake(project, settings=None, force=False):
            seen.append((Path(project), force))
            return images.ImageRecord('t', 'sha256:x', 'l', 'now', 'd')
        monkeypatch.setattr(provision, 'provision', fake)
        cli._sandbox_command('provision')
        out = capsys.readouterr().out
        assert 'sha256:x' in out and seen == [(cli_project, False)]
        cli._sandbox_command('provision --force')
        assert seen[-1] == (cli_project, True)

    def test_provision_command_reports_errors(self, cli_project,
                                              monkeypatch, capsys) -> None:
        import guru.cli as cli

        def boom(project, settings=None, force=False):
            raise provision.ProvisionError('docker build failed: nope')
        monkeypatch.setattr(provision, 'provision', boom)
        cli._sandbox_command('provision')
        assert 'docker build failed: nope' in capsys.readouterr().out

    def test_deps_list_request_apply(self, cli_project, monkeypatch,
                                     capsys) -> None:
        import guru.cli as cli
        cli._sandbox_command('deps')
        assert 'no pending' in capsys.readouterr().out
        cli._sandbox_command('deps request six>=1.16')
        out = capsys.readouterr().out
        assert 'six>=1.16' in out
        cli._sandbox_command('deps')
        assert 'six>=1.16' in capsys.readouterr().out
        applied: list = []
        monkeypatch.setattr(
            provision, 'apply_dependency',
            lambda project, req, settings=None: applied.append(req.spec)
            or 'Added six>=1.16')
        cli._sandbox_command('deps apply six')
        assert applied == ['six>=1.16']
        assert 'Added six>=1.16' in capsys.readouterr().out
        cli._sandbox_command('deps apply nothere')
        assert 'no pending request' in capsys.readouterr().out
        cli._sandbox_command('deps bogus')
        assert 'usage' in capsys.readouterr().out.lower()

    def test_deps_request_rejects_a_bad_spec(self, cli_project, capsys):
        import guru.cli as cli
        cli._sandbox_command('deps request "six; rm"')
        assert 'Refused' in capsys.readouterr().out
