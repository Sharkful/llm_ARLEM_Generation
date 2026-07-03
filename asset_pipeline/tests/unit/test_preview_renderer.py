import pytest

import config
from models.catalog_models import AssetCatalogEntry, ProvenanceInfo
from pipeline import preview_renderer


def _entry(asset_id="sphere_basic", address="Primitives/Sphere", **overrides):
    defaults = dict(
        asset_id=asset_id,
        display_name=asset_id,
        asset_class="primitive",
        address=address,
        canonical_bounds_m=[1.0, 1.0, 1.0],
        provenance=ProvenanceInfo(source_type="core"),
    )
    defaults.update(overrides)
    return AssetCatalogEntry(**defaults)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path / "library")
    (tmp_path / "library" / "previews").mkdir(parents=True)
    return tmp_path / "library"


def test_spec_for_glb_entry_uses_library_url():
    entry = _entry(
        asset_id="table", asset_class="imported",
        address="library/imported/table/model.glb",
    )
    spec = preview_renderer.spec_for_entry(entry)
    assert spec == {"kind": "glb", "url": "/library/imported/table/model.glb"}


def test_spec_for_unity_primitive_maps_name_and_bounds():
    entry = _entry(address="Primitives/Cylinder", canonical_bounds_m=[1.0, 2.0, 1.0])
    spec = preview_renderer.spec_for_entry(entry)
    assert spec == {"kind": "primitive", "name": "cylinder", "bounds": [1.0, 2.0, 1.0]}


def test_spec_for_composite_address_maps_name():
    entry = _entry(address="Composites/Arrow")
    assert preview_renderer.spec_for_entry(entry)["name"] == "arrow"


def test_render_previews_skips_existing_without_touching_playwright(sandbox, monkeypatch):
    # Force an import error if the code tries to reach playwright -- with
    # every preview already on disk it must return early instead.
    monkeypatch.setitem(__import__("sys").modules, "playwright.sync_api", None)
    entries = [_entry(), _entry(asset_id="cube_basic", address="Primitives/Cube")]
    for e in entries:
        preview_renderer.preview_path(e.asset_id).write_bytes(b"png")

    written, skipped = preview_renderer.render_previews(entries, force=False)

    assert written == []
    assert set(skipped) == {"sphere_basic", "cube_basic"}


def test_contact_sheet_lists_all_entries_and_flags_missing_previews(sandbox):
    entries = [_entry(), _entry(asset_id="cube_basic", address="Primitives/Cube")]
    preview_renderer.preview_path("sphere_basic").write_bytes(b"png")

    sheet = preview_renderer.write_contact_sheet(entries)
    html = sheet.read_text(encoding="utf-8")

    assert 'src="sphere_basic.png"' in html
    assert "cube_basic" in html
    assert "no preview" in html  # cube has no PNG yet
    assert "2 asset(s)" in html
