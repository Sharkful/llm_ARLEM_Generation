import json

import pytest

import config
from models.catalog_models import AssetCatalogEntry, ProvenanceInfo
from pipeline import external_intake
from pipeline.catalog_writer import load_catalog, save_catalog
from pipeline.mesh_processor import MeshNormalizationResult


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point intake/library/catalog at tmp so tests never touch the real library."""
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")
    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(config, "CATALOG_PATH", tmp_path / "library" / "catalog.json")
    (tmp_path / "intake").mkdir()
    (tmp_path / "library").mkdir()
    return tmp_path


def _write_intake(sandbox, asset_id="stool", source_overrides=None, mesh_names=("model.glb",)):
    intake_dir = sandbox / "intake" / asset_id
    intake_dir.mkdir(parents=True)
    source = {
        "display_name": "Wooden lab stool",
        "license": "CC0",
        "source_site": "polyhaven.com",
        "source_url": "https://polyhaven.com/a/wooden_stool",
        "original_author": "Jane Modeler",
        "target_size_m": 0.45,
        "pivot": "base_center",
        "tags": ["furniture", "stool"],
    }
    source.update(source_overrides or {})
    for key in [k for k, v in source.items() if v is None]:
        del source[key]
    (intake_dir / "source.json").write_text(json.dumps(source), encoding="utf-8")
    for name in mesh_names:
        (intake_dir / name).write_bytes(b"fake mesh bytes")
    return intake_dir


def _fake_normalize_success(monkeypatch):
    calls = {}

    def fake(asset_id, source_path, source_format, glb_path, target_size_m, pivot):
        calls.update(
            asset_id=asset_id, source_path=source_path, source_format=source_format,
            glb_path=glb_path, target_size_m=target_size_m, pivot=pivot,
        )
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(b"fake glb")
        return MeshNormalizationResult(
            asset_id=asset_id, success=True,
            source_path=str(source_path), glb_path=str(glb_path),
            original_bounds_m=[1.0, 2.0, 1.0], final_bounds_m=[0.225, 0.45, 0.225],
            pivot=pivot, scale_applied=0.225,
            triangle_count_before=500, triangle_count_after=500, decimated=False,
        )

    monkeypatch.setattr(external_intake, "normalize_mesh", fake)
    return calls


# ── Rejections (each with a specific reason) ──────────────────────────────

def test_missing_intake_folder_is_rejected(sandbox):
    with pytest.raises(external_intake.IntakeError, match="No intake folder"):
        external_intake.intake_asset("nope")


def test_missing_source_json_is_rejected(sandbox):
    (sandbox / "intake" / "bare").mkdir()
    with pytest.raises(external_intake.IntakeError, match="source.json"):
        external_intake.intake_asset("bare")


def test_invalid_source_json_names_the_missing_field(sandbox):
    _write_intake(sandbox, source_overrides={"target_size_m": None})
    with pytest.raises(external_intake.IntakeError, match="target_size_m"):
        external_intake.intake_asset("stool")


def test_disallowed_license_is_rejected(sandbox):
    _write_intake(sandbox, source_overrides={"license": "CC-BY-NC"})
    with pytest.raises(external_intake.IntakeError, match="allowlist"):
        external_intake.intake_asset("stool")


def test_cc_by_without_author_is_rejected(sandbox):
    _write_intake(
        sandbox, source_overrides={"license": "CC-BY-4.0", "original_author": None}
    )
    with pytest.raises(external_intake.IntakeError, match="attribution"):
        external_intake.intake_asset("stool")


def test_no_mesh_file_is_rejected(sandbox, monkeypatch):
    _fake_normalize_success(monkeypatch)
    _write_intake(sandbox, mesh_names=())
    with pytest.raises(external_intake.IntakeError, match="No model file"):
        external_intake.intake_asset("stool")


def test_multiple_mesh_files_are_rejected(sandbox, monkeypatch):
    _fake_normalize_success(monkeypatch)
    _write_intake(sandbox, mesh_names=("a.glb", "b.obj"))
    with pytest.raises(external_intake.IntakeError, match="Multiple model files"):
        external_intake.intake_asset("stool")


def test_duplicate_asset_id_rejected_before_normalization(sandbox, monkeypatch):
    calls = _fake_normalize_success(monkeypatch)
    existing = AssetCatalogEntry(
        asset_id="stool", display_name="existing", asset_class="primitive",
        address="builtin://cube", canonical_bounds_m=[1, 1, 1],
        provenance=ProvenanceInfo(source_type="core"),
    )
    save_catalog([existing])
    _write_intake(sandbox)

    with pytest.raises(external_intake.IntakeError, match="already exists"):
        external_intake.intake_asset("stool")
    assert not calls  # Blender path never invoked for a rejected intake


# ── Happy path ────────────────────────────────────────────────────────────

def test_intake_catalogs_with_provenance_and_files(sandbox, monkeypatch):
    calls = _fake_normalize_success(monkeypatch)
    _write_intake(sandbox)

    result = external_intake.intake_asset("stool")

    assert result.success is True
    assert calls["source_format"] == "glb"
    assert calls["target_size_m"] == 0.45
    assert calls["pivot"] == "base_center"

    out_dir = sandbox / "library" / "imported" / "stool"
    assert (out_dir / "model.glb").exists()
    assert (out_dir / "original.glb").read_bytes() == b"fake mesh bytes"
    assert (out_dir / "meta.json").exists()
    assert "CC0" in (out_dir / "LICENSE.txt").read_text(encoding="utf-8")

    [entry] = [e for e in load_catalog() if e.asset_id == "stool"]
    assert entry.asset_class == "imported"
    assert entry.address == "library/imported/stool/model.glb"
    assert entry.canonical_bounds_m == [0.225, 0.45, 0.225]
    assert entry.pivot == "base_center"
    assert entry.review_level == 3
    assert entry.provenance.source_type == "external_approved"
    assert entry.provenance.license_status == "approved"
    assert entry.provenance.attribution_required is False
    assert entry.provenance.modified is True
    assert entry.provenance.date_imported_or_generated  # recorded


def test_cc_by_sets_attribution_required_and_license_file_says_so(sandbox, monkeypatch):
    _fake_normalize_success(monkeypatch)
    _write_intake(sandbox, source_overrides={"license": "CC-BY-4.0"})

    result = external_intake.intake_asset("stool")

    assert result.catalog_entry.provenance.attribution_required is True
    license_text = (sandbox / "library" / "imported" / "stool" / "LICENSE.txt").read_text(
        encoding="utf-8"
    )
    assert "Attribution REQUIRED" in license_text
    assert "Jane Modeler" in license_text


def test_pedagogically_critical_bumps_review_level(sandbox, monkeypatch):
    _fake_normalize_success(monkeypatch)
    _write_intake(sandbox, source_overrides={"pedagogically_critical": True})

    result = external_intake.intake_asset("stool")
    assert result.catalog_entry.review_level == 4


def test_failed_normalization_reports_and_does_not_catalog(sandbox, monkeypatch):
    def fake_fail(asset_id, source_path, source_format, glb_path, target_size_m, pivot):
        return MeshNormalizationResult(
            asset_id=asset_id, success=False, source_path=str(source_path),
            error_message="Blender exploded",
        )

    monkeypatch.setattr(external_intake, "normalize_mesh", fake_fail)
    _write_intake(sandbox)

    result = external_intake.intake_asset("stool")

    assert result.success is False
    assert "Blender exploded" in result.error_message
    assert load_catalog() == []
