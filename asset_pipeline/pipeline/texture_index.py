"""Stage 6c: load/save the texture registry (library/textures/index.json).

Mirrors catalog_writer.py exactly: single JSON file, Pydantic-validated on
load, texture_id uniqueness enforced on write -- raise, don't silently
overwrite.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

import config
from models.texture_models import TextureAsset

_IndexAdapter = TypeAdapter(list[TextureAsset])


def index_path() -> Path:
    return config.LIBRARY_DIR / "textures" / "index.json"


def load_index(path: Path | None = None) -> list[TextureAsset]:
    p = path or index_path()
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8-sig"))
    return _IndexAdapter.validate_python(raw)


def save_index(entries: list[TextureAsset], path: Path | None = None) -> None:
    p = path or index_path()
    _assert_unique_ids(entries)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([e.model_dump(mode="json") for e in entries], indent=2),
        encoding="utf-8",
    )


def append_texture(entry: TextureAsset, path: Path | None = None) -> list[TextureAsset]:
    entries = load_index(path)
    if any(e.texture_id == entry.texture_id for e in entries):
        raise ValueError(
            f"texture_id {entry.texture_id!r} already exists in the texture index; "
            "use a different id or update the existing entry explicitly."
        )
    entries.append(entry)
    save_index(entries, path)
    return entries


def find_texture(texture_id: str, path: Path | None = None) -> TextureAsset | None:
    for entry in load_index(path):
        if entry.texture_id == texture_id:
            return entry
    return None


def _assert_unique_ids(entries: list[TextureAsset]) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.texture_id in seen:
            raise ValueError(f"Duplicate texture_id in index: {entry.texture_id!r}")
        seen.add(entry.texture_id)
