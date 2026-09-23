"""Regex secret scanner (guru.scanners.secrets): positives, known false
positives, project markers, allow-list, and the project loader."""
import pytest

from guru import config
from guru.domain import policy
from guru.domain.policy import redact
from guru.scanners.secrets import SecretScanner, load_project_scanner

AWS_KEY = 'AKIAIOSFODNN7EXAMPLE'
AWS_SECRET = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
GITHUB = 'ghp_' + 'A' * 36
# Assembled at runtime so no token-shaped literal sits in the file (GitHub
# push protection flags realistic Slack tokens even in test corpora).
SLACK = '-'.join(['xoxb', '1' * 12, '2' * 13, 'AbCdEfGhIjKlMnOpQrStUvWx'])
GOOGLE = 'AIza' + 'Sy' + 'D' * 33
GOOGLE_DASH = 'AIza' + 'D' * 34 + '-'
AWS_SECRET_PLUS = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKE+'
JWT = ('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.'
       'eyJzdWIiOiIxMjM0NTY3ODkwIn0.'
       'dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U')
PEM = ('-----BEGIN RSA PRIVATE KEY-----\n'
       'MIIEowIBAAKCAQEA\n'
       '-----END RSA PRIVATE KEY-----')


def kinds(findings) -> list:
    return sorted({f.kind for f in findings})


class TestPositives:
    @pytest.mark.parametrize('text, kind', [
        (f'aws_access_key_id = {AWS_KEY}', 'aws_access_key'),
        (f'aws_secret_access_key = {AWS_SECRET}', 'aws_secret_key'),
        (f'AWS_SECRET_ACCESS_KEY: "{AWS_SECRET}"', 'aws_secret_key'),
        (f'aws_secret_access_key = {AWS_SECRET_PLUS}', 'aws_secret_key'),
        (f'export GITHUB_TOKEN={GITHUB}', 'github_token'),
        ('gho_' + 'b' * 40, 'github_token'),
        (f'slack: {SLACK}', 'slack_token'),
        ('xoxp-1-2-3-abcdef', 'slack_token'),
        (f'key: {GOOGLE}', 'google_api_key'),
        (f'key: {GOOGLE_DASH}', 'google_api_key'),
        (PEM, 'private_key'),
        ('-----BEGIN OPENSSH PRIVATE KEY-----', 'private_key'),
        ('-----BEGIN PRIVATE KEY-----', 'private_key'),
        ('api_key = "abcdefghijklmnop1234"', 'generic_secret'),
        ("API-KEY: 'abcdefghijklmnop1234'", 'generic_secret'),
        ('password=Sup3rS3cretPassw0rd!', 'generic_secret'),
        ('secret: A1b2C3d4E5f6G7h8I9', 'generic_secret'),
        ('token = xyzxyzxyzxyzxyzxyz', 'generic_secret'),
        ('client_secret = abcdefghijklmnop1234', 'generic_secret'),
        ('access_token: abcdefghijklmnop1234', 'generic_secret'),
        ('auth_token=abcdefghijklmnop1234', 'generic_secret'),
        ('api_token = "abcdefghijklmnop1234"', 'generic_secret'),
        ('db_password = abcdefghijklmnop1234', 'generic_secret'),
        ('export GITHUB_TOKEN=abcdefghijklmnop1234', 'generic_secret'),
        ('secret_key = abcdefghijklmnop1234', 'generic_secret'),
        ('SECRET_KEY = "abcdefghijklmnop1234"', 'generic_secret'),
        ('secret = YWJjZGVmZ2hpamtsbW5vcA==', 'generic_secret'),
        ('password: ****************', 'generic_secret'),
        (f'Authorization: Bearer {JWT}', 'jwt'),
    ])
    def test_each_regex(self, text, kind) -> None:
        found = SecretScanner().scan(text)
        assert kind in kinds(found), (text, found)

    def test_spans_cover_the_secret(self) -> None:
        text = f'before {AWS_KEY} after'
        (f,) = SecretScanner().scan(text)
        assert text[f.start:f.end] == AWS_KEY

    def test_sample_is_masked(self) -> None:
        (f,) = SecretScanner().scan(AWS_KEY)
        assert f.sample == 'AKIA…(20)'
        assert AWS_KEY[4:] not in f.sample
        (f,) = SecretScanner().scan(f'token = {GITHUB}')
        assert f.sample == 'ghp_…(40)'

    def test_private_key_span_reaches_end_line(self) -> None:
        text = f'x\n{PEM}\ny'
        (f,) = SecretScanner().scan(text)
        assert text[f.start:f.end] == PEM
        assert text[f.start:f.end].endswith('-----END RSA PRIVATE KEY-----')
        assert redact(text, [f]) == 'x\n[REDACTED:private_key]\ny'

    def test_private_key_header_only_when_end_missing(self) -> None:
        text = '-----BEGIN PRIVATE KEY-----\nMIIEow\nrest of file'
        (f,) = SecretScanner().scan(text)
        assert text[f.start:f.end] == '-----BEGIN PRIVATE KEY-----'

    def test_two_private_keys_are_two_findings(self) -> None:
        text = f'{PEM}\n\n{PEM}'
        found = SecretScanner().scan(text)
        assert [f.kind for f in found] == ['private_key', 'private_key']
        assert text[found[0].start:found[0].end] == PEM

    def test_multiple_findings_in_one_text(self) -> None:
        text = f'{AWS_KEY}\n{GITHUB}\n{JWT}\n'
        assert kinds(SecretScanner().scan(text)) == [
            'aws_access_key', 'github_token', 'jwt']

    def test_findings_sorted_by_start(self) -> None:
        text = f'{JWT} then {AWS_KEY}'
        found = SecretScanner().scan(text)
        assert [f.start for f in found] == sorted(f.start for f in found)

    def test_duplicate_spans_collapsed(self) -> None:
        # A GitHub token also looks like a generic 'token = ...' value;
        # the same span must not be reported twice.
        text = f'token = {GITHUB}'
        found = SecretScanner().scan(text)
        spans = [(f.start, f.end) for f in found]
        assert len(spans) == len(set(spans))


class TestFalsePositives:
    @pytest.mark.parametrize('text', [
        # sha256 hex digest as a token value (exercises _plausible_generic)
        'token = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b'
        '7852b855',
        'sha256 = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b'
        '7852b855',
        # UUID as a secret value
        'secret: 123e4567-e89b-12d3-a456-426614174000',
        'id = 123e4567-e89b-12d3-a456-426614174000',
        'password = some_very_long_function_name()',
        'token = SomeVeryLongIdentifierName.value',
        "password = os.environ['DB_PASSWORD']",
        'password = os.environ.get("DB_PASSWORD")',
        'token = get_token()',
        'key = value',
        'api_key = ""',
        'secret = config.secret',
        'token: ${{ secrets.GITHUB_TOKEN }}',
        'The token is passed as a keyword argument to the function.',
        'AKIA is the prefix of an AWS access key id.',
        'eyJ is how base64 JSON starts.',
        'git commit -m "rotate password handling"',
        'def get_secret_key(self) -> str:',
        'read_dirs_allow.txt  write_dirs_allow.txt',
    ])
    def test_does_not_match(self, text) -> None:
        assert SecretScanner().scan(text) == [], text

    def test_short_generic_values_ignored(self) -> None:
        assert SecretScanner().scan('password = abcdefghijklmno') == []

    def test_empty_text(self) -> None:
        assert SecretScanner().scan('') == []


class TestMarkers:
    def test_marker_detected_case_insensitive(self) -> None:
        s = SecretScanner(markers=['Project Falcon', 'acme-internal'])
        found = s.scan('notes on PROJECT FALCON and Acme-Internal')
        assert [f.kind for f in found] == ['marker', 'marker']
        assert [f.sample for f in found] == ['PROJ…(14)', 'Acme…(13)']

    def test_marker_is_literal_not_regex(self) -> None:
        s = SecretScanner(markers=['a.b'])
        assert s.scan('axb') == []
        assert len(s.scan('a.b')) == 1

    def test_blank_markers_ignored(self) -> None:
        s = SecretScanner(markers=['', '  '])
        assert s.scan('anything at all') == []


class TestAllowList:
    def test_allow_regex_suppresses(self) -> None:
        text = f'aws_access_key_id = {AWS_KEY}'
        assert SecretScanner().scan(text)
        assert SecretScanner(allow=[r'AKIA\w*EXAMPLE']).scan(text) == []

    def test_allow_only_suppresses_matching_findings(self) -> None:
        text = f'{AWS_KEY}\n{GITHUB}'
        found = SecretScanner(allow=['EXAMPLE']).scan(text)
        assert kinds(found) == ['github_token']

    def test_allow_matches_context_line(self) -> None:
        # An allow pattern matching anywhere on the finding's line wins, so
        # `# scan-ok` style annotations work.
        text = f'x = "{GITHUB}"  # scan-ok'
        assert SecretScanner(allow=[r'#\s*scan-ok']).scan(text) == []

    def test_invalid_allow_regex_skipped(self, caplog) -> None:
        with caplog.at_level('WARNING', logger='guru'):
            s = SecretScanner(allow=['(unclosed'])
        assert 'unclosed' in caplog.text
        assert s.scan(AWS_KEY)


class TestRedactIntegration:
    def test_redact_output(self) -> None:
        text = f'aws_access_key_id = {AWS_KEY}\nok'
        out = redact(text, SecretScanner().scan(text))
        assert out == 'aws_access_key_id = [REDACTED:aws_access_key]\nok'

    def test_protocol_conformance(self) -> None:
        assert isinstance(SecretScanner(), policy.ContentScanner)


class TestProjectLoader:
    def test_slack_needs_ten_chars_after_prefix(self) -> None:
        assert SecretScanner().scan('xoxb-short') == []
        assert SecretScanner().scan('xoxb-' + 'a' * 10)

    def test_config_paths(self) -> None:
        assert config.SENSITIVE_MARKERS_PATH == (
            config.PROJECT_GURU_DIR / 'sensitive_markers.txt')
        assert config.SCAN_ALLOW_PATH == (
            config.PROJECT_GURU_DIR / 'scan_allow.txt')

    def test_missing_files_give_plain_scanner(
            self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, 'SENSITIVE_MARKERS_PATH',
                            tmp_path / 'm.txt')
        monkeypatch.setattr(config, 'SCAN_ALLOW_PATH', tmp_path / 'a.txt')
        s = load_project_scanner()
        assert isinstance(s, SecretScanner)
        assert s.markers == [] and s.allow == []

    def test_reads_markers_and_allow(self, tmp_path, monkeypatch) -> None:
        m, a = tmp_path / 'm.txt', tmp_path / 'a.txt'
        m.write_text('# comment\nProject Falcon\n\n  acme  \n',
                     encoding='utf-8')
        a.write_text('EXAMPLE\n# no\n', encoding='utf-8')
        monkeypatch.setattr(config, 'SENSITIVE_MARKERS_PATH', m)
        monkeypatch.setattr(config, 'SCAN_ALLOW_PATH', a)
        s = load_project_scanner()
        assert s.markers == ['Project Falcon', 'acme']
        assert s.allow == ['EXAMPLE']
        assert s.scan(AWS_KEY) == []
        assert [f.kind for f in s.scan('ACME rocks')] == ['marker']
