"""The usage dashboard: guru.domain.dashboard rules and the
guru.dashboard.server API, safety checks and single-server election."""
from __future__ import annotations

import http.client
import json
import random
import socket

import pytest

from guru.dashboard import server as srv
from guru.domain import dashboard as rules
from guru.repositories.usage_sqlite import SqliteUsage
from tests.test_usage import _call


def _free_port() -> int:
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _get(port: int, path: str, host: str = '', method: str = 'GET'):
    con = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    con.putrequest(method, path, skip_host=True)
    con.putheader('Host', host or f'127.0.0.1:{port}')
    con.endheaders()
    r = con.getresponse()
    body = r.read()
    con.close()
    return r.status, dict(r.getheaders()), body


@pytest.fixture
def served(tmp_path):
    store = SqliteUsage(tmp_path / 'usage.db')
    store.append('topics', {'topic_id': 't1', 'request': 'fix login',
                            'label': 'login fix'})
    store.append('calls', _call(ts='2099-01-01T00:00:00.000+00:00'))
    d = srv.Dashboard(store, _free_port(), delay=lambda: 0.05)
    assert d.start() == rules.SERVING
    yield d
    d.stop()


class TestRules:
    def test_parse_health(self) -> None:
        assert rules.parse_health({'app': rules.APP, 'pid': 42}) == \
            rules.Holder(42, '')
        for bad in ({'app': 'x', 'pid': 1}, {'app': rules.APP},
                    {'app': rules.APP, 'pid': True}, [], None):
            assert rules.parse_health(bad) is None

    def test_retry_delay_is_jittered(self) -> None:
        r = random.Random(1)
        delays = {round(rules.retry_delay(r), 3) for _ in range(20)}
        assert len(delays) > 10
        assert all(rules.RETRY_S * 0.7 <= d <= rules.RETRY_S * 1.3
                   for d in delays)

    def test_status_text(self) -> None:
        url = 'http://127.0.0.1:7340/'
        assert rules.status_text(rules.SERVING, url, None) == \
            f'serving at {url}'
        assert rules.status_text(rules.OTHER_GURU, url,
                                 rules.Holder(7)) == \
            f'served by guru pid 7 at {url}'
        assert 'another program' in rules.status_text(rules.PORT_BUSY,
                                                      url, None)


class TestApi:
    def test_page_and_assets(self, served) -> None:
        status, headers, body = _get(served.port, '/')
        assert status == 200 and b'guru' in body
        assert "script-src 'self'" in headers['Content-Security-Policy']
        assert headers['X-Content-Type-Options'] == 'nosniff'
        assert 'Access-Control-Allow-Origin' not in headers
        assert _get(served.port, '/app.js')[0] == 200
        assert _get(served.port, '/app.css')[0] == 200

    def test_summary(self, served) -> None:
        status, _, body = _get(served.port, '/api/summary?range=all')
        data = json.loads(body)
        assert status == 200 and data['totals']['calls'] == 1
        assert data['groups']['topic'][0]['key'] == 'login fix'
        assert data['daily'][0]['model'] == 'claude-sonnet-5'

    def test_calls_paging_and_bounds(self, served) -> None:
        status, _, body = _get(served.port,
                               '/api/calls?range=all&limit=9999&offset=-3')
        data = json.loads(body)
        assert status == 200 and data['limit'] == srv.PAGE_MAX
        assert data['offset'] == 0 and data['calls'][0]['topic'] == \
            'login fix'

    @pytest.mark.parametrize('query', ['range=forever', 'source=root',
                                       'limit=x'])
    def test_bad_parameters_are_400(self, served, query) -> None:
        path = ('/api/calls?' if 'limit' in query else '/api/summary?')
        assert _get(served.port, path + query)[0] == 400

    def test_foreign_host_is_refused(self, served) -> None:
        assert _get(served.port, '/api/summary',
                    host=f'evil.example:{served.port}')[0] == 403
        assert _get(served.port, '/', host=f'localhost:{served.port}')[0] \
            == 200

    def test_read_only(self, served) -> None:
        assert _get(served.port, '/api/summary', method='POST')[0] == 405
        assert _get(served.port, '/nope')[0] == 404

    def test_health(self, served) -> None:
        holder = srv.probe(served.port)
        assert holder is not None and holder.pid > 0

    def test_access_log_stays_out_of_the_terminal(self, served,
                                                  capfd) -> None:
        _get(served.port, '/api/health')
        out, err = capfd.readouterr()
        assert 'GET /api/health' not in out + err


class TestElection:
    def test_second_guru_defers_then_takes_over(self, served,
                                                tmp_path) -> None:
        other = srv.Dashboard(SqliteUsage(tmp_path / 'b.db'), served.port,
                              delay=lambda: 0.05)
        try:
            assert other.start() == rules.OTHER_GURU
            assert other.holder is not None and 'served by guru pid' in \
                other.status()
            served.stop()
            assert srv.wait_for(lambda: other.state == rules.SERVING, 5)
            assert _get(other.port, '/api/health')[0] == 200
        finally:
            other.stop()

    def test_a_foreign_program_on_the_port(self, tmp_path) -> None:
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        d = srv.Dashboard(SqliteUsage(tmp_path / 'u.db'), port,
                          delay=lambda: 60)
        try:
            assert d.start() == rules.PORT_BUSY
            assert 'another program' in d.status()
        finally:
            d.stop()
            sock.close()

    def test_a_wildcard_listener_is_respected(self, tmp_path) -> None:
        sock = socket.socket()
        sock.bind(('0.0.0.0', 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        d = srv.Dashboard(SqliteUsage(tmp_path / 'u.db'), port,
                          delay=lambda: 60)
        try:
            assert d.start() == rules.PORT_BUSY
        finally:
            d.stop()
            sock.close()

    def test_a_stopped_dashboard_starts_again(self, tmp_path) -> None:
        d = srv.Dashboard(SqliteUsage(tmp_path / 'u.db'), _free_port(),
                          delay=lambda: 0.05)
        assert d.start() == rules.SERVING
        d.stop()
        assert d.start() == rules.SERVING
        d.stop()

    def test_stop_ends_the_retries(self, served, tmp_path) -> None:
        other = srv.Dashboard(SqliteUsage(tmp_path / 'b.db'), served.port,
                              delay=lambda: 0.05)
        other.start()
        other.stop()
        served.stop()
        assert not srv.wait_for(lambda: other.state == rules.SERVING, 0.3)
