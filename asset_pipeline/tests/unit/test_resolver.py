import pytest

from models.catalog_models import AssetCatalogEntry, ProvenanceInfo
from models.classification_models import GeometryClassification
from models.spec_models import AssetSpec
from pipeline.resolver import resolve


def _entry(asset_id, tags):
    return AssetCatalogEntry(
        asset_id=asset_id,
        display_name=asset_id,
        asset_class="primitive",
        address=f"Primitives/{asset_id}",
        canonical_bounds_m=[1.0, 1.0, 1.0],
        tags=tags,
        provenance=ProvenanceInfo(source_type="core", license_status="approved"),
    )


def _catalog():
    return [_entry("sphere_basic", ["sphere", "primitive", "planet", "moon"])]


@pytest.fixture(autouse=True)
def _isolate_library(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path)


def test_resolve_catalog_match_does_not_require_review():
    spec = AssetSpec(object_id="moon", description="a gray moon", semantic_type="moon")
    record, logs = resolve(spec, _catalog())
    assert record.resolved_asset.resolution_method == "catalog_match"
    assert record.requires_author_review is False


def test_resolve_composite_builds_fragment_and_resolves_all_parts():
    spec = AssetSpec(object_id="water_molecule", description="a water molecule model")

    def fake_classifier(spec, candidates):
        if spec.object_id == "water_molecule":
            return (
                GeometryClassification(
                    geometry_class="composite",
                    reasoning="Oxygen + two hydrogens.",
                    composite_parts=["a sphere shape for oxygen (moon-like)", "a sphere shape for hydrogen (moon-like)"],
                ),
                None,
            )
        # sub-parts: resolve each part against the same sphere_basic catalog entry
        return (
            GeometryClassification(geometry_class="unclear", reasoning="fallback"),
            None,
        )

    record, logs = resolve(spec, _catalog(), classify_geometry_fn=fake_classifier)
    assert record.review_reason.startswith("Resolved as composite") or record.requires_author_review

    from pipeline.composite_builder import load_composite_fragment

    fragment = load_composite_fragment("water_molecule_composite")
    assert fragment is not None
    assert len(fragment.parts) == 2


def test_resolve_parametric_is_flagged_for_review_since_stage_3_not_built():
    spec = AssetSpec(object_id="bracket", description="an L-shaped mounting bracket")

    def fake_classifier(spec, candidates):
        return (
            GeometryClassification(geometry_class="parametric", reasoning="Mechanical bracket shape."),
            None,
        )

    record, logs = resolve(spec, [], classify_geometry_fn=fake_classifier)
    assert record.requires_author_review is True
    assert "parametric" in record.review_reason
    assert record.resolved_asset is None


def test_resolve_imported_is_flagged_for_review_since_stage_6_not_built():
    spec = AssetSpec(object_id="lab_table", description="a realistic wooden lab table")

    def fake_classifier(spec, candidates):
        return (
            GeometryClassification(geometry_class="imported", reasoning="Realistic furniture, not CSG-buildable."),
            None,
        )

    record, logs = resolve(spec, [], classify_geometry_fn=fake_classifier)
    assert record.requires_author_review is True
    assert "imported" in record.review_reason


def test_resolve_unclear_is_flagged_for_review():
    spec = AssetSpec(object_id="mystery", description="something ambiguous")

    def fake_classifier(spec, candidates):
        return (
            GeometryClassification(geometry_class="unclear", reasoning="Not enough information."),
            None,
        )

    record, logs = resolve(spec, [], classify_geometry_fn=fake_classifier)
    assert record.requires_author_review is True
    assert "Not enough information." in record.review_reason


def test_resolve_composite_recursion_depth_guard():
    spec = AssetSpec(object_id="turtles", description="a composite of composites forever")

    def always_composite(spec, candidates):
        return (
            GeometryClassification(
                geometry_class="composite",
                reasoning="infinite nesting",
                composite_parts=["another nested composite thing"],
            ),
            None,
        )

    # Should terminate (not recurse forever) thanks to the depth guard.
    record, logs = resolve(spec, [], classify_geometry_fn=always_composite)
    assert record is not None
