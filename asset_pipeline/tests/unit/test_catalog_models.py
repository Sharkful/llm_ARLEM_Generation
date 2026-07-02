import pytest
from pydantic import ValidationError

from models.catalog_models import AssetCatalogEntry, MaterialDef, ProvenanceInfo


def _valid_entry_kwargs(**overrides):
    kwargs = dict(
        asset_id="sphere_basic",
        display_name="Basic Sphere",
        asset_class="primitive",
        address="Primitives/Sphere",
        material_slots=["surface"],
        canonical_bounds_m=[1.0, 1.0, 1.0],
        provenance=ProvenanceInfo(source_type="core", license_status="approved"),
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_entry_round_trips_through_json():
    entry = AssetCatalogEntry(**_valid_entry_kwargs())
    dumped = entry.model_dump(mode="json")
    reloaded = AssetCatalogEntry.model_validate(dumped)
    assert reloaded == entry


def test_canonical_bounds_must_have_exactly_three_values():
    with pytest.raises(ValidationError):
        AssetCatalogEntry(**_valid_entry_kwargs(canonical_bounds_m=[1.0, 1.0]))


def test_asset_class_rejects_unknown_value():
    with pytest.raises(ValidationError):
        AssetCatalogEntry(**_valid_entry_kwargs(asset_class="not_a_real_class"))


def test_review_level_rejects_out_of_range_value():
    with pytest.raises(ValidationError):
        AssetCatalogEntry(**_valid_entry_kwargs(review_level=9))


def test_defaults_are_sane():
    entry = AssetCatalogEntry(**_valid_entry_kwargs())
    assert entry.review_level == 0
    assert entry.pivot == "center"
    assert entry.capabilities.scalable is True
    assert entry.collider.shape == "none"


def test_material_def_alpha_bounds():
    with pytest.raises(ValidationError):
        MaterialDef(material_id="m1", base_color="#ffffff", alpha=1.5)

    ok = MaterialDef(material_id="m1", base_color="#ffffff", alpha=0.25, transparent=True)
    assert ok.alpha == 0.25
    assert ok.shader_family == "URP/Lit"
