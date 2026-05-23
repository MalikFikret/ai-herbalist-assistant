"""Persisted registry of which PDFs are embedded in the local Chroma index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


def manifest_path(persist_dir: Path) -> Path:
    return Path(persist_dir) / "index_manifest.json"


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_manifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_manifest()
    if not isinstance(data, dict):
        return _empty_manifest()
    files = data.get("files")
    if not isinstance(files, dict):
        data["files"] = {}
    return data


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["version"] = MANIFEST_VERSION
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _empty_manifest() -> dict[str, Any]:
    return {"version": MANIFEST_VERSION, "files": {}}


def file_mtime_ns(pdf_path: Path) -> int:
    return pdf_path.stat().st_mtime_ns


def manifest_entry_for_file(pdf_path: Path, *, chunk_count: int) -> dict[str, Any]:
    return {"mtime_ns": file_mtime_ns(pdf_path), "chunk_count": chunk_count}


def is_file_unchanged(manifest: dict[str, Any], pdf_path: Path) -> bool:
    entry = manifest.get("files", {}).get(pdf_path.name)
    if not entry or "mtime_ns" not in entry:
        return False
    try:
        return int(entry["mtime_ns"]) == file_mtime_ns(pdf_path)
    except OSError:
        return False
