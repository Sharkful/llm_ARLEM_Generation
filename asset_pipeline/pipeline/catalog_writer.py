"""Load/save the authoritative asset catalog (library/catalog.json).

Enforces asset_id uniqueness on write -- raises rather than silently
overwriting, per the "no silent fallback" rule in the requirements doc.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

import config
from models.catalog_models import AssetCatalogEntry

_CatalogAdapter = TypeAdapter(list[AssetCatalogEntry])


def load_catalog(path: Path | None = None) -> list[AssetCatalogEntry]:
    catalog_path = path or config.CATALOG_PATH
    if not catalog_path.exists():
        return []
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    return _CatalogAdapter.validate_python(raw)


def save_catalog(entries: list[AssetCatalogEntry], path: Path | None = None) -> None:
    catalog_path = path or config.CATALOG_PATH
    _assert_unique_ids(entries)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps([e.model_dump(mode="json") for e in entries], indent=2),
        encoding="utf-8",
    )


def append_entry(
    entry: AssetCatalogEntry, path: Path | None = None
) -> list[AssetCatalogEntry]:
    entries = load_catalog(path)
    if any(e.asset_id == entry.asset_id for e in entries):
        raise ValueError(
            f"asset_id {entry.asset_id!r} already exists in the catalog; "
            "use a different id or update the existing entry explicitly."
        )
    entries.append(entry)
    save_catalog(entries, path)
    return entries


def find_by_id(asset_id: str, path: Path | None = None) -> AssetCatalogEntry | None:
    for entry in load_catalog(path):
        if entry.asset_id == asset_id:
            return entry
    return None


def _assert_unique_ids(entries: list[AssetCatalogEntry]) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.asset_id in seen:
            raise ValueError(f"Duplicate asset_id in catalog: {entry.asset_id!r}")
        seen.add(entry.asset_id)
