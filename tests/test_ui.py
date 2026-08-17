"""Tests for guru.ui formatting and status helpers."""
from guru import cli, session, ui


class TestHumanCtx:
    """Tests for cli._human_ctx formatting."""

    def test_kilo(self) -> None:
        assert cli._human_ctx(65536) == '64k'
        assert cli._human_ctx(4096) == '4k'

    def test_mega(self) -> None:
        assert cli._human_ctx(1024 * 1024) == '1M'


class TestStatusParts:
    """Tests for ui._status_parts colour thresholds and segments."""

    def test_green_yellow_red(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'num_ctx', 1000)
        monkeypatch.setattr(session, 'model', 'demo:latest')

        monkeypatch.setattr(session, 'ctx_used', 100)
        assert ui._status_parts()[3] == 'green'

        monkeypatch.setattr(session, 'ctx_used', 780)
        left, ctx_segment, right, colour = ui._status_parts()
        assert colour == 'yellow'
        assert '78%' in ctx_segment

        monkeypatch.setattr(session, 'ctx_used', 900)
        assert ui._status_parts()[3] == 'red'

    def test_segments_include_expected_fields(self, monkeypatch) -> None:
        monkeypatch.setattr(session, 'num_ctx', 1000)
        monkeypatch.setattr(session, 'ctx_used', 100)
        monkeypatch.setattr(session, 'model', 'demo:latest')
        monkeypatch.setattr(session, 'model_size', '32B')
        monkeypatch.setattr(session, 'session_in', 1234)
        monkeypatch.setattr(session, 'session_out', 56)
        monkeypatch.setattr(session, 'git_branch', 'main')

        left, ctx_segment, right, _ = ui._status_parts()
        assert 'demo' in left and '32B' in left
        assert '🧠' in ctx_segment
        assert '1234' in right and '56' in right and 'main' in right


class TestFormatBytes:
    """Tests for ui.format_bytes."""

    def test_gigabytes(self) -> None:
        assert ui.format_bytes(8_200_000_000) == '7.6 GB'

    def test_megabytes(self) -> None:
        assert ui.format_bytes(5_000_000) == '4.8 MB'
