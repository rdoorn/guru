"""Startup domain: host classification and the judge warm-up status text."""
from __future__ import annotations

import pytest

from guru.domain import startup


class TestHostLocation:
    @pytest.mark.parametrize('url, host', [
        ('http://localhost:11434', 'localhost:11434'),
        ('http://127.0.0.1:11434', '127.0.0.1:11434'),
        ('http://127.1.2.3', '127.1.2.3'),
        ('http://[::1]:8080/v1', '[::1]:8080'),
        ('http://0.0.0.0:11434', '0.0.0.0:11434'),
    ])
    def test_loopback_is_local(self, url: str, host: str) -> None:
        assert startup.host_location(url, True) == startup.Location(
            'local', host)

    def test_other_hosts_are_remote(self) -> None:
        loc = startup.host_location('https://proxy.example/v1', False)
        assert loc == startup.Location('remote', 'proxy.example')

    def test_no_url_uses_the_default(self) -> None:
        assert startup.host_location(None, False) == startup.Location(
            'local', '')
        assert startup.host_location('', True) == startup.Location(
            'remote', '')

    def test_str(self) -> None:
        assert str(startup.Location('local', 'localhost:11434')) == (
            'local, localhost:11434')
        assert str(startup.Location('remote', '')) == 'remote'


class TestStatusText:
    def test_idle_is_empty(self) -> None:
        assert startup.status_text(startup.WarmStatus(), now=0) == ''

    def test_loading(self) -> None:
        st = startup.WarmStatus(state='loading', name='decide')
        assert startup.status_text(st, now=0) == 'judges: loading decide…'

    def test_ready_is_shown_then_expires(self) -> None:
        st = startup.WarmStatus(state='ready', seconds=8.14, at=100.0)
        assert startup.status_text(st, now=105) == 'judges ready 8.1s'
        assert startup.status_text(
            st, now=100 + startup.READY_SHOWN_S + 0.1) == ''

    def test_failed_stays(self) -> None:
        st = startup.WarmStatus(state='failed', failed=('decide', 'enc'),
                                at=0.0)
        assert startup.status_text(st, now=10_000) == (
            'judges: decide, enc failed (see log)')
