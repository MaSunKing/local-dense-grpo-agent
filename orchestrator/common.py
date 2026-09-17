from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_bytes(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=".writing-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def safe_relative(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("nonempty relative path required")
    candidate = (Path(root).resolve() / relative).resolve()
    if not candidate.is_relative_to(Path(root).resolve()):
        raise ValueError("path escapes package root")
    return candidate

