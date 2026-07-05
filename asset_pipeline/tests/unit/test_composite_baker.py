import json

import pytest

import config
from models.catalog_models import AssetCatalogEntry, MaterialDef, ProvenanceInfo
from models.spec_models import AssetSpec
from pipeline import composite_baker
from pipeline.catalog_writer import save_catalog
from pipeline.composite_baker import CompositeBakeError, bake_composite
from pipeline.composite_builder import CompositeFragment, CompositePart


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    monkeypatch.setattr(config, "LIBRARY_DIR", lib)
    monkeypatch.setattr(config, "CATALOG_PATH", lib / "catalog.json")
    monkeypatch.setattr(config, "MATERIALS_DIR", lib / "materials")
    (lib / "materials").mkdir(parents=True)
    (lib / "composites").mkdir(parents=True)
    return tmp_path


def _sphere_entry():
    return AssetCatalogEntry(
        asset_id="sphere_basic", display_name="Sphere", asset_class="primitive",
        address="Primitives/Sphere", canonical_bounds_m=[1, 1, 1],
        provenance=ProvenanceInfo(source_type="core"),
    )


def _fragment(n_parts=2, with_materials=True):
    parts = []
    for i in range(n_parts):
        parts.append(CompositePart(
            part_id=f"p{i}",
            asset_spec=AssetSpec(object_id=f"p{i}", description=f"part {i}"),
            resolved_asset_id="sphere_basic",
            material_id=f"frosty_p{i}_mat" if with_materials else None,
            position=[0.0, i * 0.3, 0.0], scale=[0.3, 0.3, 0.3],
        ))
    return CompositeFragment(
        composite_id="frosty_composite", display_name="Frosty",
        source_description="a snowman", parts=parts,
    )


def test_bake_requires_blender(sandbox, monkeypatch):
    monkeypatch.setattr(config, "BLENDER_BIN", None)
    save_catalog([_sphere_entry()])

    with pytest.raises(CompositeBakeError, match="Blender binary not found"):
        bake_composite(_fragment(), "frosty", target_size_m=0.3)


def test_bake_reports_failure_when_no_parts_bakeable(sandbox, monkeypatch):
    monkeypatch.setattr(config, "BLENDER_BIN", "blender")
    save_catalog([])  # sphere_basic missing -> every part gets skipped

    result = bake_composite(_fragment(), "frosty", target_size_m=0.3)

    assert result.success is False
    assert "No parts could be baked" in result.error_message


def test_bake_builds_correct_payload_for_primitive_parts(sandbox, monkeypatch):
    monkeypatch.setattr(config, "BLENDER_BIN", "blender")
    save_catalog([_sphere_entry()])
    for i in range(2):
        mat = MaterialDef(material_id=f"frosty_p{i}_mat", base_color="#ffffff" if i == 0 else "#111111")
        (config.MATERIALS_DIR / f"frosty_p{i}_mat.json").write_text(
            json.dumps(mat.model_dump(mode="json")), encoding="utf-8"
        )

    captured = {}

    def fake_run(cmd, capture_output, text, timeout):
        captured["cmd"] = cmd
        # Simulate the Blender-side script by writing the report ourselves.
        report_path = cmd[-2]
        parts_payload = json.loads(cmd[cmd.index("--") + 1])
        with open(report_path, "w") as fh:
            json.dump({
                "success": True, "final_bounds_m": [0.3, 0.6, 0.3],
                "triangle_count": 200, "uv_status": "generated",
            }, fh)
        return type("R", (), {"stdout": "", "stderr": ""})()

    monkeypatch.setattr(composite_baker.subprocess, "run", fake_run)

    result = bake_composite(_fragment(n_parts=2), "frosty", target_size_m=0.3)

    assert result.success is True
    assert result.final_bounds_m == [0.3, 0.6, 0.3]
    assert result.uv_status == "generated"
    cmd = captured["cmd"]
    parts_payload = json.loads(cmd[cmd.index("--") + 1])
    assert len(parts_payload) == 2
    assert parts_payload[0]["primitive_kind"] == "sphere"
    assert parts_payload[0]["base_color"] == "#ffffff"
    assert parts_payload[1]["base_color"] == "#111111"


def test_bake_skips_unresolved_parts_but_continues(sandbox, monkeypatch):
    monkeypatch.setattr(config, "BLENDER_BIN", "blender")
    save_catalog([_sphere_entry()])
    fragment = _fragment(n_parts=1, with_materials=False)
    fragment.parts.append(CompositePart(
        part_id="unresolved", asset_spec=AssetSpec(object_id="x", description="x"),
        resolved_asset_id=None,
    ))

    def fake_run(cmd, capture_output, text, timeout):
        report_path = cmd[-2]
        with open(report_path, "w") as fh:
            json.dump({"success": True, "final_bounds_m": [0.3, 0.3, 0.3],
                      "triangle_count": 100, "uv_status": "none"}, fh)
        return type("R", (), {"stdout": "", "stderr": ""})()

    monkeypatch.setattr(composite_baker.subprocess, "run", fake_run)

    result = bake_composite(fragment, "frosty", target_size_m=0.3)

    assert result.success is True
    assert "unresolved" in result.error_message  # noted, not fatal


def test_bake_builds_connector_payload_without_catalog_lookup(sandbox, monkeypatch):
    """A bond_between part needs no resolved_asset_id/catalog entry at all --
    it's always a plain cylinder built from scratch; its geometry comes from
    the two labeled atoms' real positions, computed inside Blender (pass 2),
    not from anything the Python side needs to resolve."""
    monkeypatch.setattr(config, "BLENDER_BIN", "blender")
    save_catalog([_sphere_entry()])
    mat = MaterialDef(material_id="bond_mat", base_color="#cccccc")
    (config.MATERIALS_DIR / "bond_mat.json").write_text(mat.model_dump_json(), encoding="utf-8")

    fragment = CompositeFragment(
        composite_id="molecule_composite", display_name="Molecule", source_description="test",
        parts=[
            CompositePart(part_id="C", asset_spec=AssetSpec(object_id="C", description="carbon"),
                         resolved_asset_id="sphere_basic", label="C", position=[0, 0, 0]),
            CompositePart(part_id="H1", asset_spec=AssetSpec(object_id="H1", description="hydrogen"),
                         resolved_asset_id="sphere_basic", label="H1", position=[0.3, 0.3, 0.3]),
            CompositePart(part_id="bond1", asset_spec=AssetSpec(object_id="bond1", description="bond"),
                         material_id="bond_mat", bond_between=["C", "H1"], bond_thickness=0.05,
                         resolved_asset_id=None),  # deliberately unresolved -- must not matter
        ],
    )

    captured = {}

    def fake_run(cmd, capture_output, text, timeout):
        captured["cmd"] = cmd
        report_path = cmd[-2]
        with open(report_path, "w") as fh:
            json.dump({"success": True, "final_bounds_m": [0.3, 0.3, 0.3],
                      "triangle_count": 100, "uv_status": "none"}, fh)
        return type("R", (), {"stdout": "", "stderr": ""})()

    monkeypatch.setattr(composite_baker.subprocess, "run", fake_run)

    result = bake_composite(fragment, "molecule", target_size_m=0.3)

    assert result.success is True
    assert result.error_message is None  # the "unresolved" bond part is not an error
    cmd = captured["cmd"]
    parts_payload = json.loads(cmd[cmd.index("--") + 1])
    assert len(parts_payload) == 3
    bond_payload = next(p for p in parts_payload if p["part_id"] == "bond1")
    assert bond_payload["bond_between"] == ["C", "H1"]
    assert bond_payload["bond_thickness"] == 0.05
    assert bond_payload["base_color"] == "#cccccc"
    assert "primitive_kind" not in bond_payload  # geometry decided entirely in Blender


def test_bake_reports_blender_crash_with_no_report(sandbox, monkeypatch):
    monkeypatch.setattr(config, "BLENDER_BIN", "blender")
    save_catalog([_sphere_entry()])

    def fake_run(cmd, capture_output, text, timeout):
        return type("R", (), {"stdout": "", "stderr": "segfault"})()

    monkeypatch.setattr(composite_baker.subprocess, "run", fake_run)

    result = bake_composite(_fragment(n_parts=1), "frosty", target_size_m=0.3)

    assert result.success is False
    assert "no report" in result.error_message
