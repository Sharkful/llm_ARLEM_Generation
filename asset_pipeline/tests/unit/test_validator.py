import json

import pytest

import config
from models.catalog_models import AssetCatalogEntry, ProvenanceInfo
from pipeline import validator
from pipeline.catalog_writer import save_catalog


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    monkeypatch.setattr(config, "LIBRARY_DIR", lib)
    monkeypatch.setattr(config, "CATALOG_PATH", lib / "catalog.json")
    monkeypatch.setattr(config, "MATERIALS_DIR", lib / "materials")
    monkeypatch.setattr(config, "TEXTURES_DIR", lib / "materials" / "textures")
    for sub in ("imported/table", "materials/textures", "composites", "generated"):
        (lib / sub).mkdir(parents=True)
    return lib


def _entry(**overrides):
    defaults = dict(
        asset_id="table",
        display_name="Table",
        asset_class="imported",
        address="library/imported/table/model.glb",
        canonical_bounds_m=[0.6, 0.3, 0.35],
        pivot="base_center",
        provenance=ProvenanceInfo(
            source_type="external_approved",
            license="CC0",
            license_status="approved",
            original_author="Jane",
        ),
        review_level=3,
    )
    defaults.update(overrides)
    return AssetCatalogEntry(**defaults)


def _write_healthy_library(lib):
    (lib / "imported" / "table" / "model.glb").write_bytes(b"glb bytes")
    (lib / "imported" / "table" / "LICENSE.txt").write_text("CC0", encoding="utf-8")
    (lib / "imported" / "table" / "meta.json").write_text(
        json.dumps({"triangle_count_after": 100}), encoding="utf-8"
    )
    save_catalog([_entry()])


def test_healthy_library_passes(sandbox):
    _write_healthy_library(sandbox)
    report = validator.validate_library()

    assert report.passed is True
    assert report.errors == 0
    assert report.checked_assets == 1


def test_malformed_catalog_is_schema_error(sandbox):
    (sandbox / "catalog.json").write_text('[{"asset_id": "x"}]', encoding="utf-8")
    report = validator.validate_library()

    assert report.passed is False
    [issue] = report.issues
    assert issue.category == "schema"
    assert "failed to load" in issue.message


def test_missing_address_file_is_completeness_error(sandbox):
    save_catalog([_entry()])  # model.glb never written
    report = validator.validate_library()

    assert any(
        i.level == "error" and i.category == "completeness" and "does not exist" in i.message
        for i in report.issues
    )


def test_zero_byte_glb_is_geometry_error(sandbox):
    _write_healthy_library(sandbox)
    (sandbox / "imported" / "table" / "model.glb").write_bytes(b"")
    report = validator.validate_library()

    assert any(
        i.level == "error" and i.category == "geometry" and "zero-byte" in i.message
        for i in report.issues
    )


def test_negative_bounds_is_geometry_error(sandbox):
    _write_healthy_library(sandbox)
    save_catalog([_entry(canonical_bounds_m=[0.6, -0.1, 0.35])], sandbox / "catalog.json")
    report = validator.validate_library()

    assert any(
        i.level == "error" and i.category == "geometry" and "negative" in i.message
        for i in report.issues
    )


def test_zero_dim_warns_on_mesh_assets_but_not_primitives(sandbox):
    _write_healthy_library(sandbox)
    save_catalog(
        [
            _entry(canonical_bounds_m=[0.6, 0.0, 0.35]),  # imported, flat -> warn
            _entry(
                asset_id="plane_basic", asset_class="primitive",
                address="builtin://plane", canonical_bounds_m=[10.0, 0.0, 10.0],
                provenance=ProvenanceInfo(source_type="core"), review_level=0,
            ),
        ],
        sandbox / "catalog.json",
    )
    report = validator.validate_library()

    zero_issues = [i for i in report.issues if "zero dimension" in i.message]
    assert [i.asset_id for i in zero_issues] == ["table"]
    assert all(i.level == "warning" for i in zero_issues)


def test_out_of_range_scale_is_warning_not_error(sandbox):
    _write_healthy_library(sandbox)
    save_catalog([_entry(canonical_bounds_m=[120.0, 3.0, 3.0])], sandbox / "catalog.json")
    report = validator.validate_library()

    assert report.passed is True  # warning only
    assert any(
        i.level == "warning" and i.category == "geometry" and "sane range" in i.message
        for i in report.issues
    )


def test_triangle_count_over_threshold_is_geometry_error(sandbox, monkeypatch):
    _write_healthy_library(sandbox)
    monkeypatch.setattr(config, "MAX_TRIANGLE_COUNT", 50)
    report = validator.validate_library()

    assert any(
        i.level == "error" and "triangle count" in i.message for i in report.issues
    )


def test_unapproved_license_is_licensing_error(sandbox):
    _write_healthy_library(sandbox)
    save_catalog(
        [_entry(provenance=ProvenanceInfo(
            source_type="external_approved", license="CC0", license_status="pending",
        ))],
        sandbox / "catalog.json",
    )
    report = validator.validate_library()

    assert any(
        i.level == "error" and i.category == "licensing" and "license_status" in i.message
        for i in report.issues
    )


def test_attribution_without_author_is_licensing_error(sandbox):
    _write_healthy_library(sandbox)
    save_catalog(
        [_entry(provenance=ProvenanceInfo(
            source_type="external_approved", license="CC-BY", license_status="approved",
            attribution_required=True, original_author=None,
        ))],
        sandbox / "catalog.json",
    )
    report = validator.validate_library()

    assert any(
        i.level == "error" and "original_author" in i.message for i in report.issues
    )


def test_invalid_material_and_missing_texture_are_caught(sandbox):
    _write_healthy_library(sandbox)
    (sandbox / "materials" / "bad.json").write_text('{"nope": 1}', encoding="utf-8")
    (sandbox / "materials" / "rusty.json").write_text(
        json.dumps({
            "material_id": "rusty", "base_color": "#8b2500",
            "texture": "textures/does_not_exist.png",
        }),
        encoding="utf-8",
    )
    report = validator.validate_library()

    assert any(i.category == "schema" and "bad.json" in i.message for i in report.issues)
    assert any(
        i.category == "completeness" and "missing texture" in i.message for i in report.issues
    )


def test_composite_with_unknown_part_is_completeness_error(sandbox):
    _write_healthy_library(sandbox)
    fragment = {
        "composite_id": "water",
        "display_name": "Water molecule",
        "source_description": "one O, two H",
        "parts": [
            {
                "part_id": "oxygen",
                "asset_spec": {"object_id": "oxygen", "description": "red sphere"},
                "resolved_asset_id": "no_such_asset",
            }
        ],
    }
    (sandbox / "composites" / "water.json").write_text(json.dumps(fragment), encoding="utf-8")
    report = validator.validate_library()

    assert any(
        i.level == "error" and "unknown asset_id 'no_such_asset'" in i.message
        for i in report.issues
    )


def test_unresolved_records_fail_runtime_readiness(sandbox, tmp_path):
    _write_healthy_library(sandbox)
    records = [
        {
            "object_id": "mystery",
            "requested_asset_spec": {"object_id": "mystery", "description": "??"},
            "requires_author_review": True,
            "review_reason": "Classified as 'unclear' (not yet implemented).",
        },
        {
            "object_id": "moon",
            "requested_asset_spec": {"object_id": "moon", "description": "gray moon"},
            "requires_author_review": False,
        },
    ]
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")

    report = validator.validate_library(records_path=records_path)

    assert report.checked_records == 2
    assert any(
        i.level == "error" and i.category == "runtime" and i.asset_id == "mystery"
        for i in report.issues
    )
    assert report.passed is False


def test_json_and_screen_output_agree_on_verdict(sandbox):
    _write_healthy_library(sandbox)
    report = validator.validate_library()

    payload = json.loads(validator.format_json_output(report))
    assert payload["summary"]["passed"] is True
    screen = validator.format_screen_text(report, use_color=False)
    assert "PASSED" in screen
    assert "Errors: 0" in screen
