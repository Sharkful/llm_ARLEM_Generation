from models.catalog_models import AssetCatalogEntry, ProvenanceInfo
from models.spec_models import AssetSpec
from pipeline.catalog_matcher import find_candidates, score_entry


def _entry(asset_id, tags, display_name=None):
    return AssetCatalogEntry(
        asset_id=asset_id,
        display_name=display_name or asset_id,
        asset_class="primitive",
        address=f"Primitives/{asset_id}",
        canonical_bounds_m=[1.0, 1.0, 1.0],
        tags=tags,
        provenance=ProvenanceInfo(source_type="core", license_status="approved"),
    )


def _catalog():
    return [
        _entry("sphere_basic", ["sphere", "primitive", "planet", "moon", "atom"], "Basic Sphere"),
        _entry("cube_basic", ["cube", "box", "primitive", "block"], "Basic Cube"),
        _entry("lab_table_01", ["furniture", "table", "lab"], "Lab Table"),
    ]


def test_exact_semantic_type_match_scores_highest():
    spec = AssetSpec(object_id="moon", description="a small gray moon", semantic_type="moon")
    match = score_entry(spec, _catalog()[0])
    assert match.confidence == 1.0


def test_generic_category_semantic_type_does_not_shortcut():
    """Regression (Stage 9 exit test): 'worn leather armchair' parsed with
    semantic_type 'furniture' must not score 1.0 against a table carrying a
    'furniture' tag -- that is the silent-wrong-fallback failure mode."""
    armchair = AssetSpec(
        object_id="worn_leather_armchair",
        description="a realistic worn leather armchair",
        kind="composite", semantic_type="furniture",
        visual_style="realistic, worn leather",
    )
    match = score_entry(armchair, _catalog()[2])  # lab_table_01, tag 'furniture'
    assert match.confidence < 0.5  # falls through to token scoring


def test_unrelated_entry_scores_low():
    spec = AssetSpec(object_id="moon", description="a small gray moon with craters")
    match = score_entry(spec, _entry("lab_table_01", ["furniture", "table", "lab"]))
    assert match.confidence < 0.2


def test_find_candidates_ranks_best_match_first():
    spec = AssetSpec(object_id="moon", description="a small gray moon with visible craters")
    candidates = find_candidates(spec, _catalog())
    assert candidates[0].entry.asset_id == "sphere_basic"


def test_find_candidates_returns_empty_for_no_overlap():
    spec = AssetSpec(object_id="x", description="zzz qqq nonexistent")
    candidates = find_candidates(spec, _catalog())
    # "table" and "lab" won't match "zzz qqq nonexistent" at all
    assert all(c.confidence <= 0.2 for c in candidates) or candidates == []


def test_find_candidates_respects_top_k():
    spec = AssetSpec(object_id="x", description="a primitive shape")
    candidates = find_candidates(spec, _catalog(), top_k=1)
    assert len(candidates) <= 1
