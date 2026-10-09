"""The usage dashboard server: stdlib ``ThreadingHTTPServer`` on
``127.0.0.1`` serving one page (``static/``) and a read-only JSON API over
the usage store (``guru.domain.usage.UsageQueries``).

Safety: bound to the loopback address only; GET and HEAD only; a request
whose ``Host`` is not ``127.0.0.1:<port>`` or ``localhost:<port>`` is
refused (a web page cannot reach the API through DNS rebinding); no CORS
headers; a strict Content-Security-Policy (no inline script); access lines
go to the guru log, never the terminal.

:class:`Dashboard` runs the single-server rule (``guru.domain.dashboard``):
bind or find out who holds the port, then retry in the background.
"""
from __future__ import annotations

import errno
import http.client
import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlsplit

from guru import log
from guru.domain import dashboard as rules
from guru.domain import usage

STATIC = Path(__file__).resolve().parent / 'static'
FILES = {'/': ('index.html', 'text/html; charset=utf-8'),
         '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
         '/app.css': ('app.css', 'text/css; charset=utf-8')}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self';"
       " img-src 'self'; connect-src 'self'; frame-ancestors 'none';"
       " base-uri 'none'; form-action 'none'")
HEALTH_TIMEOUT_S = 1.0
GROUP_LIMIT = 25
PAGE_MAX = 200


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # SO_REUSEADDR: a takeover may bind a port whose old connections are
    # still in TIME_WAIT. Two listeners on 127.0.0.1:<port> are refused on
    # Linux and BSD/macOS alike, but BSD lets a specific address join a
    # wildcard (0.0.0.0) listener, so Dashboard probes before binding.
    allow_reuse_address = True

    def __init__(self, address, handler, queries: usage.UsageQueries,
                 health: dict) -> None:
        super().__init__(address, handler)
        self.queries = queries
        self.health = health


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    server_version = 'guru-dashboard'
    sys_version = ''

    def log_message(self, format: str, *args) -> None:   # noqa: A002
        log.log.debug('dashboard: ' + format, *args)

    # --- plumbing ------------------------------------------------------------

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', CSP)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _json(self, status: int, data: object) -> None:
        self._send(status, json.dumps(data, default=str).encode('utf-8'),
                   'application/json; charset=utf-8')

    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        host = (self.headers.get('Host') or '').strip().lower()
        return host in (f'127.0.0.1:{port}', f'localhost:{port}')

    # --- verbs ---------------------------------------------------------------

    def do_GET(self) -> None:                            # noqa: N802
        if not self._host_ok():
            self._json(HTTPStatus.FORBIDDEN, {'error': 'bad Host header'})
            return
        parts = urlsplit(self.path)
        try:
            if parts.path in FILES:
                name, ctype = FILES[parts.path]
                self._send(HTTPStatus.OK, (STATIC / name).read_bytes(),
                           ctype)
            elif parts.path == '/api/health':
                self._json(HTTPStatus.OK, self.server.health)
            elif parts.path == '/api/summary':
                self._json(HTTPStatus.OK, self._summary(parse_qs(
                    parts.query)))
            elif parts.path == '/api/calls':
                self._json(HTTPStatus.OK, self._calls(parse_qs(parts.query)))
            else:
                self._json(HTTPStatus.NOT_FOUND, {'error': 'not found'})
        except ValueError as e:
            self._json(HTTPStatus.BAD_REQUEST, {'error': str(e)})
        except Exception:                                # noqa: BLE001
            log.exc('dashboard request failed')
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {'error': 'internal error (see the guru log)'})

    do_HEAD = do_GET

    def _refuse(self) -> None:
        self._json(HTTPStatus.METHOD_NOT_ALLOWED, {'error': 'read-only'})

    do_POST = do_PUT = do_DELETE = do_PATCH = _refuse    # noqa: N815

    # --- API -----------------------------------------------------------------

    @staticmethod
    def _one(query: dict, name: str, default: str) -> str:
        values = query.get(name) or [default]
        return values[0]

    def _filters(self, query: dict) -> tuple:
        since = usage.since(self._one(query, 'range', '7d'))
        source = self._one(query, 'source', 'all')
        if source not in usage.SOURCES:
            raise ValueError('source must be one of '
                             + ', '.join(usage.SOURCES))
        return since, source

    def _summary(self, query: dict) -> dict:
        since, source = self._filters(query)
        q = self.server.queries
        return {'totals': q.totals(since, source),
                'daily': q.daily(since, source),
                'groups': {by: q.groups(by, since, source, GROUP_LIMIT)
                           for by in usage.GROUPS}}

    def _calls(self, query: dict) -> dict:
        since, source = self._filters(query)
        try:
            limit = int(self._one(query, 'limit', '50'))
            offset = int(self._one(query, 'offset', '0'))
        except ValueError:
            raise ValueError('limit and offset must be integers') from None
        limit = max(1, min(limit, PAGE_MAX))
        offset = max(0, offset)
        return {'calls': self.server.queries.calls(since, source, limit,
                                                   offset),
                'limit': limit, 'offset': offset}


def listening(port: int, timeout: float = 0.3) -> bool:
    """Whether anything accepts connections on 127.0.0.1:``port`` (a
    guru, or another program listening on loopback or the wildcard)."""
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex(('127.0.0.1', port)) == 0


def probe(port: int, timeout: float = HEALTH_TIMEOUT_S
          ) -> Optional[rules.Holder]:
    """Who serves ``127.0.0.1:port``: the guru holding it, or None (no
    answer, or not a guru dashboard)."""
    # http.client, not urllib: a configured HTTP proxy must never see (or
    # swallow) a loopback probe.
    con = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        con.request('GET', '/api/health')
        resp = con.getresponse()
        return rules.parse_health(json.loads(resp.read(4096)))
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        con.close()


class Dashboard:
    """The single-server rule for one guru process: :meth:`start` binds
    the port (``serving``) or finds who holds it (``other-guru`` /
    ``port-busy``) and then retries in the background; :meth:`stop` shuts
    down the server and the retries."""

    def __init__(self, queries: usage.UsageQueries, port: int,
                 delay: Callable[[], float] = rules.retry_delay,
                 version: str = '') -> None:
        self.queries = queries
        self.port = port
        self.state = rules.OFF
        self.holder: Optional[rules.Holder] = None
        self._delay = delay
        self._version = version
        self._server: Optional[_Server] = None
        self._stop = threading.Event()
        self._retry: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self.port}/'

    def status(self) -> str:
        return rules.status_text(self.state, self.url, self.holder)

    def _try_bind(self) -> bool:
        health = {'app': rules.APP, 'pid': os.getpid(), 'port': self.port,
                  'version': self._version,
                  'started': datetime.now(timezone.utc).isoformat(
                      timespec='seconds')}
        if listening(self.port):
            return False                 # someone holds it: never join them
        try:
            server = _Server(('127.0.0.1', self.port), _Handler,
                             self.queries, health)
        except OSError as e:
            if e.errno not in (errno.EADDRINUSE, errno.EACCES):
                log.warning('dashboard: cannot bind %s: %s', self.port, e)
            return False
        self._server = server
        threading.Thread(target=server.serve_forever,
                         name='guru-dashboard', daemon=True).start()
        self.state, self.holder = rules.SERVING, None
        log.info('dashboard serving at %s', self.url)
        return True

    def start(self) -> str:
        """Serve, or find the holder and start retrying; returns the
        state. A stopped dashboard can be started again."""
        self._stop.clear()
        with self._lock:
            if self._try_bind():
                return self.state
            self._observe()
        self._retry = threading.Thread(target=self._retry_loop,
                                       name='guru-dashboard-retry',
                                       daemon=True)
        self._retry.start()
        return self.state

    def _observe(self) -> None:
        self.holder = probe(self.port)
        self.state = (rules.OTHER_GURU if self.holder is not None
                      else rules.PORT_BUSY)

    def _retry_loop(self) -> None:
        while not self._stop.wait(self._delay()):
            with self._lock:
                if self._stop.is_set() or self.state == rules.SERVING:
                    return
                if self._try_bind():
                    return
                self._observe()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            server, self._server = self._server, None
            self.state = rules.OFF
        if server is not None:
            server.shutdown()
            server.server_close()


def wait_for(predicate: Callable[[], bool], timeout_s: float) -> bool:
    """Poll ``predicate`` until true or ``timeout_s`` passes (tests)."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()
