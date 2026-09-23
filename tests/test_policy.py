"""Content policy seam (guru.domain.policy): findings, redaction, scanner."""
import pytest

from guru.domain import policy
from guru.domain.policy import Finding, redact


class FakeScanner:
    """Flags every occurrence of ``needle`` as kind ``fake``."""

    def __init__(self, needle: str = 'SECRET') -> None:
        self.needle = needle
        self.calls: list = []

    def scan(self, text: str) -> list:
        self.calls.append(text)
        out, start = [], 0
        while True:
            i = text.find(self.needle, start)
            if i < 0:
                return out
            out.append(Finding('fake', i, i + len(self.needle),
                               text[i:i + len(self.needle)]))
            start = i + len(self.needle)


class Exploding:
    def scan(self, text: str) -> list:
        raise RuntimeError('boom')


@pytest.fixture(autouse=True)
def _no_scanner():
    policy.set_scanner(None)
    yield
    policy.set_scanner(None)


class TestFinding:
    def test_fields(self) -> None:
        f = Finding('aws_access_key', 3, 23, 'AKIA…')
        assert (f.kind, f.start, f.end, f.sample) == (
            'aws_access_key', 3, 23, 'AKIA…')

    def test_is_protocol_conformant_scanner(self) -> None:
        assert isinstance(FakeScanner(), policy.ContentScanner)


class TestRedact:
    def test_replaces_span_and_preserves_surroundings(self) -> None:
        text = 'key = SECRET; done'
        out = redact(text, [Finding('fake', 6, 12, 'SECRET')])
        assert out == 'key = [REDACTED:fake]; done'

    def test_multiple_spans_in_any_order(self) -> None:
        text = 'a SECRET b SECRET c'
        findings = [Finding('x', 11, 17, ''), Finding('y', 2, 8, '')]
        assert redact(text, findings) == 'a [REDACTED:y] b [REDACTED:x] c'

    def test_overlapping_spans_merge(self) -> None:
        text = '0123456789'
        findings = [Finding('a', 2, 6, ''), Finding('b', 4, 8, '')]
        out = redact(text, findings)
        assert out == '01[REDACTED:a]89'

    def test_adjacent_spans_stay_separate(self) -> None:
        text = '0123456789'
        findings = [Finding('a', 2, 5, ''), Finding('b', 5, 8, '')]
        assert redact(text, findings) == '01[REDACTED:a][REDACTED:b]89'

    def test_no_findings_returns_text_unchanged(self) -> None:
        assert redact('hello', []) == 'hello'

    def test_out_of_range_spans_clamped(self) -> None:
        assert redact('abc', [Finding('k', -2, 99, '')]) == '[REDACTED:k]'

    def test_empty_span_ignored(self) -> None:
        assert redact('abc', [Finding('k', 1, 1, '')]) == 'abc'


class TestScannerSeam:
    def test_default_no_scanner(self) -> None:
        assert policy.scanner() is None
        assert policy.scan('SECRET') == []

    def test_set_and_use(self) -> None:
        fake = FakeScanner()
        policy.set_scanner(fake)
        assert policy.scanner() is fake
        found = policy.scan('x SECRET y')
        assert [(f.kind, f.start, f.end) for f in found] == [('fake', 2, 8)]

    def test_reset_to_none(self) -> None:
        policy.set_scanner(FakeScanner())
        policy.set_scanner(None)
        assert policy.scan('SECRET') == []

    def test_scan_never_raises(self, caplog) -> None:
        policy.set_scanner(Exploding())
        with caplog.at_level('DEBUG', logger='guru'):
            assert policy.scan('anything') == []
        assert 'scanner failed' in caplog.text
        assert 'boom' in caplog.text

    def test_scan_empty_text_short_circuits(self) -> None:
        fake = FakeScanner()
        policy.set_scanner(fake)
        assert policy.scan('') == []
        assert fake.calls == []

    def test_scan_and_redact_together(self) -> None:
        policy.set_scanner(FakeScanner())
        text = 'token: SECRET'
        assert redact(text, policy.scan(text)) == 'token: [REDACTED:fake]'
