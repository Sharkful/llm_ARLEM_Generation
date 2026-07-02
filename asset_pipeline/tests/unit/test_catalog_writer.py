import pytest

from models.catalog_models import AssetCatalogEntry, ProvenanceInfo
from pipeline.catalog_writer import append_entry, find_by_id, load_catalog, save_catalog


def _entry(asset_id="sphere_basic"):
    return AssetCatalogEntry(
        asset_id=asset_id,
        display_name="Basic Sphere",
        asset_class="primitive",
        address="Primitives/Sphere",
        canonical_bounds_m=[1.0, 1.0, 1.0],
        provenance=ProvenanceInfo(source_type="core", license_status="approved"),
    )


def test_load_catalog_returns_empty_list_when_file_missing(tmp_path):
    assert load_catalog(tmp_path / "nope.json") == []


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "catalog.json"
    save_catalog([_entry()], path)
    loaded = load_catalog(path)
    assert len(loaded) == 1
    assert loaded[0].asset_id == "sphere_basic"


def test_save_catalog_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "catalog.json"
    with pytest.raises(ValueError, match="Duplicate asset_id"):
        save_catalog([_entry("dup"), _entry("dup")], path)


def test_append_entry_rejects_existing_id(tmp_path):
    path = tmp_path / "catalog.json"
    save_catalog([_entry("sphere_basic")], path)
    with pytest.raises(ValueError, match="already exists"):
        append_entry(_entry("sphere_basic"), path)


def test_append_entry_adds_new_id(tmp_path):
    path = tmp_path / "catalog.json"
    save_catalog([_entry("sphere_basic")], path)
    append_entry(_entry("cube_basic"), path)
    loaded = load_catalog(path)
    assert {e.asset_id for e in loaded} == {"sphere_basic", "cube_basic"}


def test_find_by_id(tmp_path):
    path = tmp_path / "catalog.json"
    save_catalog([_entry("sphere_basic")], path)
    assert find_by_id("sphere_basic", path) is not None
    assert find_by_id("does_not_exist", path) is None
