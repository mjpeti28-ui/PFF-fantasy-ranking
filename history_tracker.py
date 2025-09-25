from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple


BASE_HISTORY_DIR = Path("history")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}


def _save_manifest(manifest_path: Path, data: dict) -> None:
    manifest_path.write_text(json.dumps(data, indent=2))


def _category_path(category: str) -> Path:
    return BASE_HISTORY_DIR / category


def _copy_with_timestamp(src: Path, dest_dir: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{timestamp}_{src.name}"
    dest_path.write_bytes(src.read_bytes())


def archive_if_changed(entries: Iterable[Tuple[str, Path]]) -> None:
    BASE_HISTORY_DIR.mkdir(exist_ok=True)
    manifest_path = BASE_HISTORY_DIR / "manifest.json"
    manifest = _load_manifest(manifest_path)

    for category, path in entries:
        if not path.exists():
            continue
        file_hash = _hash_file(path)
        entry_key = f"{category}:{path.name}"
        if manifest.get(entry_key) == file_hash:
            continue
        _copy_with_timestamp(path, _category_path(category))
        manifest[entry_key] = file_hash

    _save_manifest(manifest_path, manifest)
