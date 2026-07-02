import pytest
from pydantic import ValidationError

from models.spec_models import AssetSpec, ResolutionRecord, ResolvedAssetRef


def test_asset_spec_minimal():
    spec = AssetSpec(object_id="moon", description="a small gray moon with craters")
    assert spec.kind is None
    assert spec.desired_size_m is None


def test_resolution_record_requires_review_defaults_false():
    spec = AssetSpec(object_id="moon", description="a small gray moon with craters")
    record = ResolutionRecord(object_id="moon", requested_asset_spec=spec)
    assert record.requires_author_review is False
    assert record.resolved_asset is None


def test_resolved_asset_ref_confidence_bounds():
    with pytest.raises(ValidationError):
        ResolvedAssetRef(asset_id="sphere_basic", resolution_method="catalog_match", confidence=1.5)

    ref = ResolvedAssetRef(asset_id="sphere_basic", resolution_method="catalog_match", confidence=0.94)
    assert ref.warnings == []


def test_resolution_method_rejects_unknown_value():
    spec = AssetSpec(object_id="moon", description="a small gray moon with craters")
    with pytest.raises(ValidationError):
        ResolvedAssetRef(asset_id="x", resolution_method="made_up_method", confidence=0.5)
