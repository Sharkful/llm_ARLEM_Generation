import pytest

from models.catalog_models import AssetCatalogEntry, ProvenanceInfo
from models.spec_models import AssetSpec
from pipeline import source_assist
from pipeline.external_intake import IntakeResult
from pipeline.polyhaven import PolyHavenResult, PolyHavenSearchError
from pipeline.source_relevance import CandidateReview


def _spec(**overrides):
    defaults = dict(
        object_id="wooden_stool",
        description="a simple wooden stool, about 45cm tall",
        kind="stool", semantic_type="stool", desired_size_m=0.45,
    )
    defaults.update(overrides)
    return AssetSpec(**defaults)


def _hit(asset_id="wooden_stool_02", score=6, downloads=5000):
    return PolyHavenResult(
        asset_id=asset_id, name=asset_id.replace("_", " ").title(),
        download_count=downloads, score=score,
        thumbnail_url=f"https://cdn.example/{asset_id}.png",
    )


def _entry(asset_id):
    return AssetCatalogEntry(
        asset_id=asset_id, display_name=asset_id, asset_class="imported",
        address=f"library/imported/{asset_id}/model.glb",
        canonical_bounds_m=[0.3, 0.45, 0.3],
        provenance=ProvenanceInfo(source_type="external_approved", license="CC0",
                                  license_status="approved"),
    )


@pytest.fixture(autouse=True)
def _no_nasa_network(monkeypatch):
    """Default the NASA source to empty; tests that exercise it override."""
    monkeypatch.setattr(source_assist, "search_nasa3d", lambda q, limit: [])


@pytest.fixture(autouse=True)
def _stub_relevance_review(monkeypatch):
    """Default: every candidate 'matches' with no real LLM call, so routing
    tests exercise confidence/identity gating in isolation. Tests of the
    relevance gate itself override this."""
    def fake_review(description, candidates, provider=None, model=None):
        return [CandidateReview(matches=True, confidence=0.9, reasoning="stub") for _ in candidates], []

    monkeypatch.setattr(source_assist, "review_candidates", fake_review)


def test_build_query_prefers_identity_fields():
    q = source_assist.build_query(_spec())
    assert "stool" in q
    assert "45cm" not in q  # description filler only used as fallback


def test_build_query_prefers_explicit_keywords_over_mechanical_extraction():
    q = source_assist.build_query(
        _spec(description="A friendly snowman made from three rounded snowballs..."),
        keywords=["snowman", "christmas figure"],
    )
    assert q == "snowman christmas figure"
    assert "friendly" not in q and "made" not in q


def test_confident_cc0_match_with_size_auto_adopts(monkeypatch):
    monkeypatch.setattr(source_assist, "search_models",
                        lambda q, limit, asset_type: [_hit(score=6)])  # 3 tokens max -> high coverage
    adopted = {}

    def fake_adopt(candidate, intake_id, target_size_m):
        adopted.update(id=candidate.source_id, intake=intake_id, size=target_size_m)
        return IntakeResult(asset_id=intake_id, success=True,
                            catalog_entry=_entry(intake_id))

    monkeypatch.setattr(source_assist, "adopt_candidate", fake_adopt)

    result, logs = source_assist.assist_imported(_spec(), "wooden_stool")

    assert result.auto_adopted is not None
    assert adopted == {"id": "wooden_stool_02", "intake": "wooden_stool", "size": 0.45}


def test_no_size_means_candidates_not_auto(monkeypatch):
    monkeypatch.setattr(source_assist, "search_models",
                        lambda q, limit, asset_type: [_hit(score=6)])
    monkeypatch.setattr(source_assist, "adopt_candidate",
                        lambda *a, **k: pytest.fail("must not auto-adopt without a size"))

    result, logs = source_assist.assist_imported(_spec(desired_size_m=None), "wooden_stool")

    assert result.auto_adopted is None
    assert len(result.candidates) == 1
    assert "no real-world size" in result.note


def test_low_confidence_means_candidates_not_auto(monkeypatch):
    # score 2 over a ~3-token query: above the noise floor (shown as a
    # candidate) but below the auto-adopt bar (not downloaded).
    monkeypatch.setattr(source_assist, "search_models",
                        lambda q, limit, asset_type: [_hit(score=2)])
    monkeypatch.setattr(source_assist, "adopt_candidate",
                        lambda *a, **k: pytest.fail("must not auto-adopt a weak match"))

    result, logs = source_assist.assist_imported(_spec(), "wooden_stool")

    assert result.auto_adopted is None
    assert result.candidates  # visible for human approval
    assert result.candidates[0].confidence < source_assist.AUTO_ADOPT_CONFIDENCE


def test_no_hits_gives_specific_search_urls(monkeypatch):
    monkeypatch.setattr(source_assist, "search_models", lambda q, limit, asset_type: [])

    result, logs = source_assist.assist_imported(_spec(), "wooden_stool")

    assert result.candidates == []
    assert any("polyhaven.com" in u for u in result.search_urls)
    assert any("nasa3d" in u for u in result.search_urls)
    assert "intake/wooden_stool/" in result.note


def test_nasa3d_candidates_merge_and_rank(monkeypatch):
    monkeypatch.setattr(source_assist, "search_models",
                        lambda q, limit, asset_type: [_hit("bar_stool", score=2)])
    monkeypatch.setattr(source_assist, "search_nasa3d", lambda q, limit: [{
        "path": "3D Models/Apollo Lunar Module/Apollo Lunar Module.glb",
        "name": "Apollo Lunar Module", "folder": "3D Models/Apollo Lunar Module",
        "size": 700000, "score": 6,
        "thumbnail_url": "https://raw.example/thumb.png",
        "html_url": "https://github.com/nasa/NASA-3D-Resources/tree/master/x",
        "raw_url": "https://raw.example/model.glb",
    }])
    monkeypatch.setattr(source_assist, "adopt_candidate",
                        lambda *a, **k: pytest.fail("stool spec should not auto-adopt a lander"))

    result, logs = source_assist.assist_imported(_spec(desired_size_m=None), "lem")

    assert result.candidates[0].source == "nasa3d"
    assert result.candidates[0].license == "public-domain"
    assert result.candidates[0].name == "Apollo Lunar Module"


def test_junk_confidence_candidates_are_suppressed(monkeypatch):
    """Moon rocks must not be offered for a lunar lander -- below the noise
    floor only the search links remain."""
    monkeypatch.setattr(source_assist, "search_models",
                        lambda q, limit, asset_type: [_hit("moon_rock_01", score=1)])

    result, logs = source_assist.assist_imported(_spec(), "lem")

    assert result.candidates == []
    assert "search links" in result.note


def test_relevance_review_filters_out_non_matches(monkeypatch):
    """The actual Frosty regression: a coincidental word match (e.g. NASA's
    'Snowman Craters') must be dropped even at high keyword confidence, once
    the relevance reviewer says it isn't really the requested object."""
    monkeypatch.setattr(source_assist, "search_models", lambda q, limit, asset_type: [])
    monkeypatch.setattr(source_assist, "search_nasa3d", lambda q, limit: [{
        "path": "3D Printing/Vesta - Snowman Craters/Vesta - Snowman Craters.stl",
        "name": "Vesta - Snowman Craters", "folder": "x", "size": 1000, "score": 6,
        "thumbnail_url": "https://raw.example/thumb.png",
        "html_url": "https://github.com/nasa/NASA-3D-Resources/tree/master/x",
        "raw_url": "https://raw.example/model.stl",
    }])
    monkeypatch.setattr(
        source_assist, "review_candidates",
        lambda description, candidates, provider=None, model=None: (
            [CandidateReview(matches=False, confidence=0.1,
                             reasoning="This is an asteroid crater formation, not a snowman.")],
            [],
        ),
    )
    monkeypatch.setattr(source_assist, "adopt_candidate",
                        lambda *a, **k: pytest.fail("must not adopt a rejected candidate"))

    result, logs = source_assist.assist_imported(
        _spec(description="a snowman", semantic_type="snowman", desired_size_m=1.0), "frosty",
    )

    assert result.candidates == []
    assert result.auto_adopted is None
    assert "relevance review" in result.note


def test_relevance_review_unavailable_keeps_candidate_unreviewed(monkeypatch):
    """A transient review failure must not hide every candidate outright."""
    monkeypatch.setattr(source_assist, "search_models",
                        lambda q, limit, asset_type: [_hit(score=6)])
    monkeypatch.setattr(
        source_assist, "review_candidates",
        lambda description, candidates, provider=None, model=None: (
            [CandidateReview(matches=None, reasoning="Text review failed.")], [],
        ),
    )
    monkeypatch.setattr(source_assist, "adopt_candidate",
                        lambda *a, **k: pytest.fail("unreviewed must not auto-adopt"))

    result, logs = source_assist.assist_imported(_spec(), "wooden_stool")

    assert len(result.candidates) == 1
    assert result.candidates[0].relevance_match is None
    assert result.auto_adopted is None  # unreviewed is not the same as confirmed


def test_adopt_dispatches_by_source(monkeypatch):
    calls = {}
    monkeypatch.setattr(source_assist, "fetch_nasa3d_to_intake",
                        lambda path, intake_id, target_size_m: calls.update(nasa=path))
    monkeypatch.setattr(source_assist, "fetch_to_intake",
                        lambda sid, intake_id, target_size_m: calls.update(ph=sid))
    monkeypatch.setattr(source_assist, "intake_asset",
                        lambda intake_id: IntakeResult(asset_id=intake_id, success=True,
                                                       catalog_entry=_entry(intake_id)))

    nasa = source_assist.SourceCandidate(
        source="nasa3d", source_id="3D Models/X/X.glb", name="X",
        license="public-domain", url="u",
    )
    source_assist.adopt_candidate(nasa, "x", target_size_m=1.0)
    assert calls == {"nasa": "3D Models/X/X.glb"}


def test_network_failure_degrades_to_search_urls(monkeypatch):
    def boom(q, limit, asset_type):
        raise PolyHavenSearchError("offline")

    monkeypatch.setattr(source_assist, "search_models", boom)

    result, logs = source_assist.assist_imported(_spec(), "wooden_stool")

    assert result.auto_adopted is None
    assert result.search_urls
    assert "unavailable" in result.note
