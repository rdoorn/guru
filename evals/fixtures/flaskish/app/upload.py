"""Store files uploaded by authenticated users under a per-service root."""
import os


def save_upload(base_dir: str, user_path: str, data: bytes) -> str:
    """Write ``data`` to ``user_path`` below ``base_dir``; return the path.

    ``user_path`` comes straight from the request (the client picks the
    file name, optionally with sub-directories).
    """
    target = os.path.join(base_dir, user_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'wb') as fh:
        fh.write(data)
    return target


def read_upload(base_dir: str, user_path: str) -> bytes:
    """Return the bytes previously stored under ``user_path``."""
    with open(os.path.join(base_dir, user_path), 'rb') as fh:
        return fh.read()
