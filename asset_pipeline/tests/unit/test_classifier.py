from models.catalog_models import AssetCatalogEntry, ProvenanceInfo
from models.classification_models import GeometryClassification
from models.spec_models import AssetSpec
from pipeline.classifier import classify


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


def _never_called_classifier(spec, candidates):
    raise AssertionError("classify_geometry_fn should not be called for a confident catalog match")


def test_confident_match_returns_catalog_match_without_calling_llm():
    spec = AssetSpec(object_id="moon", description="a gray moon", semantic_type="moon")
    resolved, geometry, logs = classify(spec, _catalog(), classify_geometry_fn=_never_called_classifier)
    assert resolved is not None
    assert resolved.resolution_method == "catalog_match"
    assert resolved.asset_id == "sphere_basic"
    assert geometry is None
    assert logs == []


def test_weak_match_returns_variant_without_calling_llm():
    spec = AssetSpec(object_id="planet_marker", description="sphere shape marker for planet")
    resolved, geometry, logs = classify(
        spec, _catalog(), classify_geometry_fn=_never_called_classifier, match_threshold=0.99
    )
    assert resolved is not None
    assert resolved.resolution_method == "variant"
    assert resolved.warnings  # explains why it's a variant, not a match


def test_no_match_falls_through_to_geometry_classifier():
    spec = AssetSpec(object_id="water_molecule", description="a water molecule model")

    def fake_classifier(spec, candidates):
        return (
            GeometryClassification(
                geometry_class="composite",
                reasoning="Water molecule is oxygen + two hydrogens.",
                composite_parts=["1 red sphere (oxygen)", "2 white spheres (hydrogen)"],
            ),
            None,
        )

    resolved, geometry, logs = classify(spec, [], classify_geometry_fn=fake_classifier)
    assert resolved is None
    assert geometry.geometry_class == "composite"
    assert len(geometry.composite_parts) == 2


def test_classifier_log_entries_are_forwarded():
    from models.log_models import LLMCallEntry

    spec = AssetSpec(object_id="bracket", description="an L-shaped mounting bracket")
    fake_entry = LLMCallEntry(provider="anthropic", model="claude-sonnet-4-6", purpose="classification")

    def fake_classifier(spec, candidates):
        return (
            GeometryClassification(geometry_class="parametric", reasoning="Mechanical bracket shape."),
            fake_entry,
        )

    resolved, geometry, logs = classify(spec, [], classify_geometry_fn=fake_classifier)
    assert logs == [fake_entry]
