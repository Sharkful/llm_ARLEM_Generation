import json

import pytest

import config
from models.catalog_models import AssetCatalogEntry, MaterialDef, ProvenanceInfo
from pipeline.catalog_writer import save_catalog
from pipeline.sync import sync_assets


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    monkeypatch.setattr(config, "LIBRARY_DIR", lib)
    monkeypatch.setattr(config, "CATALOG_PATH", lib / "catalog.json")
    monkeypatch.setattr(config, "MATERIALS_DIR", lib / "materials")
    monkeypatch.setattr(config, "TEXTURES_DIR", lib / "materials" / "textures")
    (lib / "materials" / "textures").mkdir(parents=True)
    (lib / "generated" / "bracket").mkdir(parents=True)
    (lib / "textures" / "earth_daymap").mkdir(parents=True)
    return tmp_path


def _catalog(sandbox):
    (sandbox / "library" / "generated" / "bracket" / "model.glb").write_bytes(b"glb")
    save_catalog([
        AssetCatalogEntry(
            asset_id="bracket", display_name="Bracket", asset_class="parametric",
            address="library/generated/bracket/model.glb",
            canonical_bounds_m=[0.06, 0.04, 0.005],
            provenance=ProvenanceInfo(source_type="generated"),
        ),
        AssetCatalogEntry(
            asset_id="sphere_basic", display_name="Sphere", asset_class="primitive",
            address="Primitives/Sphere", canonical_bounds_m=[1, 1, 1],
            provenance=ProvenanceInfo(source_type="core"),
        ),
    ])


def test_sync_copies_glb_material_and_flattened_texture(sandbox):
    _catalog(sandbox)
    (sandbox / "library" / "textures" / "earth_daymap" / "albedo.jpg").write_bytes(b"jpg")
    mat = MaterialDef(
        material_id="sphere_basic_mat", base_color="#3a7bd5",
        texture="textures/earth_daymap/albedo.jpg", texture_id="earth_daymap",
    )
    (sandbox / "library" / "materials" / "sphere_basic_mat.json").write_text(
        json.dumps(mat.model_dump(mode="json")), encoding="utf-8"
    )
    target = sandbox / "viewer_assets"

    result = sync_assets(["bracket", "sphere_basic"], target)

    assert (target / "models" / "bracket.glb").exists()
    assert result.models_copied == ["bracket"]
    assert any("built-in" in s for s in result.skipped)  # sphere has no file
    # Registry texture flattened to a unique name; copied material rewritten.
    assert (target / "textures" / "earth_daymap.jpg").exists()
    synced_mat = json.loads(
        (target / "materials" / "sphere_basic_mat.json").read_text(encoding="utf-8")
    )
    assert synced_mat["texture"] == "textures/earth_daymap.jpg"
    # ...while the library copy is untouched.
    original = json.loads(
        (sandbox / "library" / "materials" / "sphere_basic_mat.json").read_text(encoding="utf-8")
    )
    assert original["texture"] == "textures/earth_daymap/albedo.jpg"


def test_sync_carries_object_level_materials(sandbox):
    """A scene object's material is named after the object (moon_mat), not
    the catalog asset it resolved to (sphere_basic) -- regression for the
    apollo scene where nothing synced."""
    _catalog(sandbox)
    (sandbox / "library" / "textures" / "earth_daymap" / "albedo.jpg").write_bytes(b"jpg")
    mat = MaterialDef(
        material_id="moon_mat", base_color="#888888",
        texture="textures/earth_daymap/albedo.jpg", texture_id="earth_daymap",
    )
    (sandbox / "library" / "materials" / "moon_mat.json").write_text(
        json.dumps(mat.model_dump(mode="json")), encoding="utf-8"
    )
    target = sandbox / "viewer_assets"

    result = sync_assets(["sphere_basic"], target, material_ids=["moon_mat"])

    assert "moon_mat" in result.materials_copied
    assert (target / "materials" / "moon_mat.json").exists()
    assert (target / "textures" / "earth_daymap.jpg").exists()


def test_sync_reports_unknown_and_missing(sandbox):
    _catalog(sandbox)
    (sandbox / "library" / "generated" / "bracket" / "model.glb").unlink()
    target = sandbox / "viewer_assets"

    result = sync_assets(["bracket", "ghost"], target)

    assert result.models_copied == []
    assert any("missing file" in s for s in result.skipped)
    assert any("not in catalog" in s for s in result.skipped)
