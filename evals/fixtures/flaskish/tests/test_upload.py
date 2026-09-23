"""Happy-path tests for uploads."""
from pathlib import Path

from app import upload


def test_save_then_read_round_trip(tmp_path: Path) -> None:
    stored = upload.save_upload(str(tmp_path), 'alice/notes.txt', b'hello')
    assert Path(stored) == tmp_path / 'alice' / 'notes.txt'
    assert upload.read_upload(str(tmp_path), 'alice/notes.txt') == b'hello'


def test_save_creates_nested_directories(tmp_path: Path) -> None:
    upload.save_upload(str(tmp_path), 'a/b/c.bin', b'\x00\x01')
    assert (tmp_path / 'a' / 'b' / 'c.bin').read_bytes() == b'\x00\x01'
