import hashlib
from collections.abc import Iterable
from pathlib import Path

from .constants import HASH_CHUNK_BYTES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)

    return digest.hexdigest()


def list_files(directory: Path, extensions: Iterable[str]) -> list[Path]:
    allowed = frozenset(extensions)

    if not directory.is_dir():
        return []

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    )
