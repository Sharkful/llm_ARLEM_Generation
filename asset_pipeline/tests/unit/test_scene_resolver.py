import json

import pytest

import config
from models.catalog_models import AssetCatalogEntry, MaterialDef, ProvenanceInfo
from models.generation_models import GenerationResult
from models.spec_models import AssetSpec
from pipeline import scene_resolver
from pipeline.asset_factory import DraftAsset
from pipeline.catalog_writer import save_catalog
from pipeline.validator import ValidationReport


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    monkeypatch.setattr(config, "LIBRARY_DIR", lib)
    monkeypatch.setattr(config, "CATALOG_PATH", lib / "catalog.json")
    monkeypatch.setattr(config, "MATERIALS_DIR", lib / "materials")
    (lib / "materials").mkdir(parents=True)
    (lib / "previews").mkdir(parents=True)
    # Batch plumbing is under test, not the stages themselves.
    monkeypatch.setattr(scene_resolver, "validate_library",
                        lambda records_path=None: ValidationReport())
    monkeypatch.setattr(scene_resolver, "render_previews", lambda entries: ([], []))
    monkeypatch.setattr(scene_resolver, "write_contact_sheet", lambda entries: None)
    return tmp_path


def _draft(object_id, status, **overrides):
    defaults = dict(
        asset_id=object_id, status=status, resolution_method="catalog_match",
        description=f"a {object_id}", spec=AssetSpec(object_id=object_id, description=f"a {object_id}"),
    )
    defaults.update(overrides)
    return DraftAsset(**defaults)


def _route_by_description(monkeypatch, drafts_by_keyword):
    def fake_create(description, asset_id=None, provider=None, model=None):
        for keyword, draft in drafts_by_keyword.items():
            if keyword in description:
                if isinstance(draft, Exception):
                    raise draft
                return draft, []
        raise AssertionError(f"no fake draft for {description!r}")

    monkeypatch.setattr(scene_resolver, "create_from_description", fake_create)
    monkeypatch.setattr(
        scene_resolver, "create_from_spec",
        lambda spec, provider=None, model=None: fake_create(spec.description),
    )


def _write_scene(tmp_path, items):
    path = tmp_path / "scene_specs.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def test_five_path_scene_report(sandbox, monkeypatch):
    """The implementation plan's Stage 9 exit-test shape: one item per
    resolution path, each reported with the expected status."""
    _route_by_description(monkeypatch, {
        "moon": _draft("moon", "existing", matched_asset_id="sphere_basic"),
        "water": _draft(
            "water", "draft", resolution_method="composite",
            composite_id="water_composite",
            glb_address="library/composites/water/model.glb",
            bounds_m=[0.1, 0.1, 0.1],
        ),
        "bracket": _draft(
            "bracket", "draft", resolution_method="parametric",
            glb_address="library/generated/bracket/model.glb",
            bounds_m=[0.06, 0.04, 0.005],
            generation=GenerationResult(asset_id="bracket", success=True, bounds_diverge=False),
        ),
        "couch": _draft("couch", "needs_human", resolution_method="imported",
                        message="Classified as 'imported'."),
        "mystery": RuntimeError("LLM exploded"),
    })

    saved = []
    def fake_save(asset_id, display_name=None, tags=None, render_preview=True):
        saved.append(asset_id)
        return AssetCatalogEntry(
            asset_id=asset_id, display_name=asset_id,
            asset_class="composite" if asset_id == "water" else "parametric",
            address=f"library/generated/{asset_id}/model.glb",
            canonical_bounds_m=[0.06, 0.04, 0.005],
            provenance=ProvenanceInfo(source_type="generated"), review_level=2,
        )
    monkeypatch.setattr(scene_resolver, "save_draft", fake_save)

    path = _write_scene(sandbox, [
        "a small gray moon",
        "a water molecule",
        "an L bracket",
        "a leather couch",
        "a mystery object",
    ])
    report = scene_resolver.resolve_scene(path)

    by_object_id = {i.object_id: i for i in report.items}
    assert by_object_id["moon"].status == "resolved_existing"
    assert by_object_id["moon"].asset_id == "sphere_basic"
    assert by_object_id["water"].status == "saved_generated"
    assert by_object_id["water"].asset_id == "water"  # baked composite, auto-cataloged like any generated asset
    assert by_object_id["bracket"].status == "saved_generated"
    assert by_object_id["bracket"].review_level == 2
    assert sorted(saved) == ["bracket", "water"]
    assert by_object_id["couch"].status == "needs_review"
    assert by_object_id["couch"].requires_author_review is True
    # Raw-string scene items have no object_id until the LLM parses one;
    # an exception before that point falls back to "?".
    assert "LLM exploded" in by_object_id["?"].message
    assert by_object_id["?"].status == "failed"
    assert (report.resolved, report.needs_review, report.failed) == (3, 1, 1)
    assert report.status == "failed"  # hard failure dominates


def test_diverging_parametric_bounds_halt_for_review(sandbox, monkeypatch):
    _route_by_description(monkeypatch, {
        "bracket": _draft(
            "bracket", "draft", resolution_method="parametric",
            glb_address="library/generated/bracket/model.glb",
            generation=GenerationResult(asset_id="bracket", success=True, bounds_diverge=True),
        ),
    })
    monkeypatch.setattr(
        scene_resolver, "save_draft",
        lambda *a, **k: pytest.fail("diverging draft must NOT be auto-saved"),
    )

    report = scene_resolver.resolve_scene(_write_scene(sandbox, ["an L bracket"]))

    [item] = report.items
    assert item.status == "needs_review"
    assert "diverge" in item.message.lower()
    assert report.status == "needs_review"


def test_texture_pending_flags_review_without_halting(sandbox, monkeypatch):
    _route_by_description(monkeypatch, {
        "earth": _draft(
            "earth", "existing", matched_asset_id="sphere_basic",
            texture_pending=True, texture_query="earth daymap",
            material=MaterialDef(material_id="earth_mat", base_color="#3a7bd5"),
        ),
    })

    report = scene_resolver.resolve_scene(_write_scene(sandbox, ["the earth"]))

    [item] = report.items
    assert item.status == "resolved_existing"  # geometry resolved fine
    assert item.requires_author_review is True  # ...but the fallback is recorded
    assert item.texture_query == "earth daymap"
    assert report.status == "needs_review"


def test_spec_objects_accepted_alongside_strings(sandbox, monkeypatch):
    _route_by_description(monkeypatch, {
        "moon": _draft("moon", "existing", matched_asset_id="sphere_basic"),
    })
    path = _write_scene(sandbox, [
        {"object_id": "moon", "description": "a small gray moon"},
    ])

    report = scene_resolver.resolve_scene(path)
    assert report.items[0].status == "resolved_existing"


def test_malformed_scene_items_rejected_with_clear_error(sandbox):
    path = _write_scene(sandbox, [{"not_a_spec": True}])
    with pytest.raises(ValueError, match="not a valid AssetSpec"):
        scene_resolver.resolve_scene(path)


def test_build_report_written_next_to_specs(sandbox, monkeypatch):
    _route_by_description(monkeypatch, {
        "moon": _draft("moon", "existing", matched_asset_id="sphere_basic"),
    })
    path = _write_scene(sandbox, ["a small gray moon"])
    report = scene_resolver.resolve_scene(path)

    out = scene_resolver.write_build_report(report, path)

    assert out.name == "scene_specs.build_report.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["items"][0]["asset_id"] == "sphere_basic"
