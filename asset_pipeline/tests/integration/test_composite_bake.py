"""Integration tier: real Blender bake of a multi-part composite. Self-skips
if Blender isn't installed/detected, per the Stage 1.1 testing design (same
pattern as test_mesh_normalization.py).

Exit test for the "Frosty is all gray" fix: a composite with distinctly
colored parts (white body, black hat) bakes into ONE glTF whose vertex
colors/material assignment are genuinely different per region -- checked
here via per-material base_color in the exported GLB, and via the merged
mesh actually being a single object with multiple material slots.
"""
from pathlib import Path

import pygltflib
import pytest

import config
from models.catalog_models import AssetCatalogEntry, MaterialDef, ProvenanceInfo
from pipeline.catalog_writer import save_catalog
from pipeline.composite_baker import bake_composite
from pipeline.composite_builder import CompositeFragment, CompositePart
from models.spec_models import AssetSpec

pytestmark = pytest.mark.skipif(
    not config.BLENDER_BIN, reason="Blender binary not found on this machine"
)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    monkeypatch.setattr(config, "LIBRARY_DIR", lib)
    monkeypatch.setattr(config, "CATALOG_PATH", lib / "catalog.json")
    monkeypatch.setattr(config, "MATERIALS_DIR", lib / "materials")
    (lib / "materials").mkdir(parents=True)
    (lib / "composites").mkdir(parents=True)
    save_catalog([
        AssetCatalogEntry(
            asset_id="sphere_basic", display_name="Sphere", asset_class="primitive",
            address="Primitives/Sphere", canonical_bounds_m=[1, 1, 1],
            provenance=ProvenanceInfo(source_type="core"),
        ),
        AssetCatalogEntry(
            asset_id="cone_basic", display_name="Cone", asset_class="primitive",
            address="Primitives/Cone", canonical_bounds_m=[1, 1, 1],
            provenance=ProvenanceInfo(source_type="core"),
        ),
    ])
    return lib


def _material(material_id, color):
    mat = MaterialDef(material_id=material_id, base_color=color)
    (config.MATERIALS_DIR / f"{material_id}.json").write_text(
        mat.model_dump_json(), encoding="utf-8"
    )


def test_bake_produces_multi_material_glb_with_distinct_colors(sandbox):
    _material("snowman_body_mat", "#f5f5f5")   # near-white
    _material("snowman_nose_mat", "#e07b1a")   # orange

    fragment = CompositeFragment(
        composite_id="snowman_test", display_name="Test Snowman",
        source_description="a small test snowman",
        parts=[
            CompositePart(
                part_id="body", asset_spec=AssetSpec(object_id="body", description="white sphere"),
                resolved_asset_id="sphere_basic", material_id="snowman_body_mat",
                position=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0],
            ),
            CompositePart(
                part_id="nose", asset_spec=AssetSpec(object_id="nose", description="orange cone"),
                resolved_asset_id="cone_basic", material_id="snowman_nose_mat",
                position=[0.0, 0.0, 0.6], rotation=[90.0, 0.0, 0.0], scale=[0.15, 0.15, 0.3],
            ),
        ],
    )

    result = bake_composite(fragment, "snowman_test", target_size_m=0.3)

    assert result.success is True, result.error_message
    glb_path = Path(result.glb_path)
    assert glb_path.exists() and glb_path.stat().st_size > 0

    gltf = pygltflib.GLTF2().load(str(glb_path))
    # One merged mesh...
    assert len(gltf.meshes) == 1
    # ...but with two distinct materials (one per colored part), and two
    # primitives (one per material slot) -- exactly what "baked, not just
    # positioned" means: Blender's join preserved per-face material data.
    assert len(gltf.materials) == 2
    colors = [
        tuple(round(c, 2) for c in m.pbrMetallicRoughness.baseColorFactor[:3])
        for m in gltf.materials
    ]
    saturations = [max(c) - min(c) for c in colors]
    # white-ish and orange-ish, not two copies of the same gray default:
    # one near-neutral (white), one clearly saturated (orange).
    assert colors[0] != colors[1]
    assert min(saturations) < 0.05
    assert max(saturations) > 0.15

    assert result.final_bounds_m is not None
    assert max(result.final_bounds_m) > 0
