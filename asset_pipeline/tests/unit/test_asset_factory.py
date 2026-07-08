import json

import pytest

import config
from models.catalog_models import AssetCatalogEntry, ProvenanceInfo
from models.generation_models import GenerationResult, OpenSCADPlan
from models.log_models import LLMCallEntry
from models.spec_models import AssetSpec, ResolutionRecord, ResolvedAssetRef
from pipeline import asset_factory
from pipeline.catalog_writer import load_catalog, save_catalog
from pipeline.mesh_processor import MeshNormalizationResult

_SCAD = """\
width = 0.06;
height = 0.04;
thickness = 0.005;
cube([width, height, thickness]);
"""


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    monkeypatch.setattr(config, "LIBRARY_DIR", lib)
    monkeypatch.setattr(config, "CATALOG_PATH", lib / "catalog.json")
    monkeypatch.setattr(config, "MATERIALS_DIR", lib / "materials")
    (lib / "generated").mkdir(parents=True)
    (lib / "previews").mkdir(parents=True)
    (lib / "materials").mkdir(parents=True)
    (lib / "composites").mkdir(parents=True)
    return tmp_path


def _log(purpose="spec_parsing"):
    return LLMCallEntry(provider="anthropic", model="m", purpose=purpose)


def _spec(object_id="bracket", **kw):
    return AssetSpec(object_id=object_id, description="an L bracket", **kw)


def _patch_stage_1_2(monkeypatch, record):
    monkeypatch.setattr(
        asset_factory, "parse_description", lambda d, provider=None, model=None: (_spec(), _log())
    )
    monkeypatch.setattr(asset_factory, "resolve", lambda spec, catalog: (record, []))
    # The Stage 6c material bind is exercised separately; stub it here so
    # routing tests never reach a real LLM.
    monkeypatch.setattr(
        asset_factory, "_bind_material",
        lambda draft, provider, model, progress=None: [],
    )
    # ...and the sourcing assist so they never reach the network.
    from pipeline.source_assist import SourcingAssist
    monkeypatch.setattr(
        asset_factory, "assist_imported",
        lambda spec, intake_id, keywords=None, manual_query=None, provider=None, model=None, license_tier=None: (
            SourcingAssist(
                query="stub", search_urls=["Poly Haven (CC0): https://polyhaven.com/all?s=stub"],
                note="stubbed",
            ),
            [],
        ),
    )


# ── extract / substitute parameters ───────────────────────────────────────

def test_extract_scad_parameters_finds_top_level_numeric_assignments():
    params = asset_factory.extract_scad_parameters(_SCAD)
    assert params == {"width": 0.06, "height": 0.04, "thickness": 0.005}


def test_substitute_scad_parameters_rewrites_in_place():
    out = asset_factory.substitute_scad_parameters(_SCAD, {"width": 0.12})
    assert "width = 0.12;" in out
    assert "height = 0.04;" in out  # untouched


def test_substitute_unknown_parameter_raises_with_editable_list():
    with pytest.raises(asset_factory.AssetFactoryError, match="width"):
        asset_factory.substitute_scad_parameters(_SCAD, {"nope": 1.0})


# ── create_from_description routing ───────────────────────────────────────

def test_create_routes_catalog_match_to_existing(sandbox, monkeypatch):
    entry = AssetCatalogEntry(
        asset_id="sphere_basic", display_name="Sphere", asset_class="primitive",
        address="Primitives/Sphere", canonical_bounds_m=[1, 1, 1],
        provenance=ProvenanceInfo(source_type="core"),
    )
    save_catalog([entry])
    record = ResolutionRecord(
        object_id="bracket", requested_asset_spec=_spec(),
        resolved_asset=ResolvedAssetRef(
            asset_id="sphere_basic", resolution_method="catalog_match", confidence=0.95
        ),
    )
    _patch_stage_1_2(monkeypatch, record)

    draft, logs = asset_factory.create_from_description("a sphere")

    assert draft.status == "existing"
    assert draft.matched_asset_id == "sphere_basic"
    assert draft.bounds_m == [1, 1, 1]


def test_create_existing_match_skips_material_bind_for_imported_asset(sandbox, monkeypatch):
    """Regression (2026-07, 'lunar module comes back gray' incident): an
    'imported' catalog match already carries its own real, baked textures --
    binding a synthetic material over it would silently replace those with a
    flat procedural color, which is what happened once the catalog matcher
    was fixed to correctly resolve to a real imported asset instead of a
    wrong primitive/parametric one."""
    entry = AssetCatalogEntry(
        asset_id="lunar_excursion_module", display_name="Apollo Lunar Module",
        asset_class="imported", address="library/imported/lunar_excursion_module/model.glb",
        canonical_bounds_m=[0.15, 0.15, 0.12],
        provenance=ProvenanceInfo(source_type="external_approved"),
    )
    save_catalog([entry])
    record = ResolutionRecord(
        object_id="lm", requested_asset_spec=_spec(),
        resolved_asset=ResolvedAssetRef(
            asset_id="lunar_excursion_module", resolution_method="catalog_match", confidence=0.93
        ),
    )
    _patch_stage_1_2(monkeypatch, record)
    calls = []
    monkeypatch.setattr(
        asset_factory, "_bind_material",
        lambda draft, provider, model, progress=None: (calls.append(draft.asset_id), [])[1],
    )

    draft, logs = asset_factory.create_from_description("a lunar module")

    assert draft.status == "existing"
    assert draft.material is None
    assert calls == []  # _bind_material must never run for an imported match


def test_create_existing_match_binds_material_for_primitive_asset(sandbox, monkeypatch):
    """A bare primitive match (no baked texture of its own) still needs a
    bound material to become identifiable (the moon = sphere + moon texture)."""
    entry = AssetCatalogEntry(
        asset_id="sphere_basic", display_name="Sphere", asset_class="primitive",
        address="Primitives/Sphere", canonical_bounds_m=[1, 1, 1],
        provenance=ProvenanceInfo(source_type="core"),
    )
    save_catalog([entry])
    record = ResolutionRecord(
        object_id="moon", requested_asset_spec=_spec(),
        resolved_asset=ResolvedAssetRef(
            asset_id="sphere_basic", resolution_method="catalog_match", confidence=0.95
        ),
    )
    _patch_stage_1_2(monkeypatch, record)
    calls = []
    monkeypatch.setattr(
        asset_factory, "_bind_material",
        lambda draft, provider, model, progress=None: (calls.append(draft.asset_id), [])[1],
    )

    draft, logs = asset_factory.create_from_description("the moon")

    assert calls == [draft.asset_id]  # _bind_material must run for a primitive match


def test_create_routes_parametric_through_generate_and_normalize(sandbox, monkeypatch):
    record = ResolutionRecord(
        object_id="bracket", requested_asset_spec=_spec(),
        requires_author_review=True,
        review_reason="Classified as 'parametric' (not yet implemented). A bracket.",
    )
    _patch_stage_1_2(monkeypatch, record)

    def fake_generate(spec, asset_id, generate_plan_fn=None):
        stl = asset_factory._draft_dir(asset_id) / "source.stl"
        stl.parent.mkdir(parents=True, exist_ok=True)
        stl.write_bytes(b"stl")
        return (
            GenerationResult(
                asset_id=asset_id, success=True, scad_path="x.scad", stl_path=str(stl),
                actual_bounds_m=[0.06, 0.04, 0.005], expected_bounds_m=[0.06, 0.04, 0.005],
                repair_attempts=[{
                    "attempt_number": 1, "scad_source": _SCAD, "success": True,
                }],
            ),
            [_log("openscad_generation")],
        )

    def fake_normalize(asset_id, source_path, source_format, glb_path, target_size_m, pivot):
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(b"glb")
        return MeshNormalizationResult(
            asset_id=asset_id, success=True, source_path=str(source_path),
            glb_path=str(glb_path), final_bounds_m=[0.06, 0.04, 0.005],
            pivot=pivot, scale_applied=1.0,
            triangle_count_before=12, triangle_count_after=12,
        )

    monkeypatch.setattr(asset_factory, "generate_parametric_asset", fake_generate)
    monkeypatch.setattr(asset_factory, "normalize_mesh", fake_normalize)

    draft, logs = asset_factory.create_from_description("an L bracket")

    assert draft.status == "draft"
    assert draft.resolution_method == "parametric"
    assert draft.plan.parameters == {"width": 0.06, "height": 0.04, "thickness": 0.005}
    assert draft.glb_address == f"library/generated/{draft.asset_id}/model.glb"
    assert asset_factory.draft_path(draft.asset_id).exists()  # persisted


def test_create_routes_imported_to_needs_human(sandbox, monkeypatch):
    record = ResolutionRecord(
        object_id="couch", requested_asset_spec=_spec("couch"),
        requires_author_review=True,
        review_reason="Classified as 'imported' (not yet implemented). Organic form.",
    )
    _patch_stage_1_2(monkeypatch, record)

    draft, _ = asset_factory.create_from_description("a leather couch")

    assert draft.status == "needs_human"
    assert draft.resolution_method == "imported"
    assert draft.suggested_sources  # never "go search" without links
    assert asset_factory.draft_path(draft.asset_id).exists()  # persisted for the app


# ── create_from_spec forced_route (added 2026-07) ─────────────────────────
# Lets a human pick the creation route up front -- for A/B testing which
# route gives the best result for the same description -- instead of only
# being able to redirect a stuck draft after the fact (redirect_draft below).
# Each branch mirrors its automatic-routing counterpart above / its
# redirect_draft counterpart below, just entered directly.

def _assert_never_classifies(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("resolve()/classify() should not run for a forced route")
    monkeypatch.setattr(asset_factory, "resolve", boom)


def test_forced_route_existing_matches_catalog_without_llm_classification(sandbox, monkeypatch):
    entry = AssetCatalogEntry(
        asset_id="sphere_basic", display_name="Sphere", asset_class="primitive",
        address="Primitives/Sphere", canonical_bounds_m=[1, 1, 1],
        provenance=ProvenanceInfo(source_type="core"),
    )
    save_catalog([entry])
    monkeypatch.setattr(asset_factory, "_bind_material", lambda draft, provider, model, progress=None: [])
    _assert_never_classifies(monkeypatch)

    spec = AssetSpec(object_id="sphere_basic", description="a sphere")
    draft, logs = asset_factory.create_from_spec(spec, forced_route="existing")

    assert draft.status == "existing"
    assert draft.matched_asset_id == "sphere_basic"
    assert draft.resolution_method == "catalog_match"


def test_forced_route_existing_skips_material_bind_for_imported_asset(sandbox, monkeypatch):
    entry = AssetCatalogEntry(
        asset_id="lunar_excursion_module", display_name="Apollo Lunar Module",
        asset_class="imported", address="library/imported/lunar_excursion_module/model.glb",
        canonical_bounds_m=[0.15, 0.15, 0.12],
        provenance=ProvenanceInfo(source_type="external_approved"),
    )
    save_catalog([entry])
    _assert_never_classifies(monkeypatch)
    calls = []
    monkeypatch.setattr(
        asset_factory, "_bind_material",
        lambda draft, provider, model, progress=None: (calls.append(draft.asset_id), [])[1],
    )

    spec = AssetSpec(object_id="lm", description="a lunar module")
    draft, logs = asset_factory.create_from_spec(spec, forced_route="existing")

    assert draft.status == "existing"
    assert draft.material is None
    assert calls == []


def test_forced_route_existing_with_no_match_is_needs_human(sandbox):
    spec = AssetSpec(object_id="unicorn", description="a unicorn")
    draft, logs = asset_factory.create_from_spec(spec, forced_route="existing")

    assert draft.status == "needs_human"
    assert draft.resolution_method == "existing"
    assert "no catalog match" in draft.message.lower()


def test_forced_route_parametric_bypasses_classification(sandbox, monkeypatch):
    _assert_never_classifies(monkeypatch)

    def fake_generate(spec, asset_id, generate_plan_fn=None):
        stl = asset_factory._draft_dir(asset_id) / "source.stl"
        stl.parent.mkdir(parents=True, exist_ok=True)
        stl.write_bytes(b"stl")
        return (
            GenerationResult(
                asset_id=asset_id, success=True, scad_path="x.scad", stl_path=str(stl),
                actual_bounds_m=[0.06, 0.04, 0.005], expected_bounds_m=[0.06, 0.04, 0.005],
                repair_attempts=[{"attempt_number": 1, "scad_source": _SCAD, "success": True}],
            ),
            [_log("openscad_generation")],
        )

    def fake_normalize(asset_id, source_path, source_format, glb_path, target_size_m, pivot):
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(b"glb")
        return MeshNormalizationResult(
            asset_id=asset_id, success=True, glb_path=str(glb_path),
            final_bounds_m=[0.06, 0.04, 0.005], pivot=pivot, scale_applied=1.0,
            triangle_count_before=12, triangle_count_after=12,
        )

    monkeypatch.setattr(asset_factory, "generate_parametric_asset", fake_generate)
    monkeypatch.setattr(asset_factory, "normalize_mesh", fake_normalize)
    monkeypatch.setattr(asset_factory, "_bind_material", lambda draft, provider, model, progress=None: [])

    draft, logs = asset_factory.create_from_spec(_spec("bracket"), forced_route="parametric")

    assert draft.status == "draft"
    assert draft.resolution_method == "parametric"
    assert draft.glb_address == f"library/generated/{draft.asset_id}/model.glb"


def test_forced_route_composite_decomposes_and_bakes(sandbox, monkeypatch):
    from models.classification_models import CompositePartPlan

    forced_parts = [
        CompositePartPlan(description="large white sphere", color_hint="white"),
        CompositePartPlan(description="small white sphere", color_hint="white"),
    ]
    monkeypatch.setattr(
        asset_factory, "decompose_into_primitives",
        lambda spec, max_parts=12, provider=None, model=None: (forced_parts, _log("composite_decomposition")),
    )
    captured = {}

    def fake_resolve(spec, catalog, forced_composite_parts=None):
        captured["parts"] = forced_composite_parts
        return (
            ResolutionRecord(
                object_id=spec.object_id, requested_asset_spec=spec,
                requires_author_review=False,
                review_reason=f"Resolved as composite: {spec.object_id}_composite",
            ),
            [],
        )

    monkeypatch.setattr(asset_factory, "resolve", fake_resolve)

    baked = asset_factory.DraftAsset(
        asset_id="frosty", status="draft", resolution_method="composite",
        description="a friendly snowman", spec=_spec("frosty"),
        composite_id="frosty_composite", glb_address="library/composites/frosty/model.glb",
        bounds_m=[0.2, 0.3, 0.2],
    )
    monkeypatch.setattr(
        asset_factory, "_materialize_and_bake_composite",
        lambda composite_id, asset_id, target_size_m, provider, model, progress: (baked, [_log("material_generation")]),
    )

    draft, logs = asset_factory.create_from_spec(_spec("frosty"), forced_route="composite")

    assert draft.glb_address == "library/composites/frosty/model.glb"
    assert captured["parts"] == forced_parts


def test_forced_route_imported_searches_without_classification(sandbox, monkeypatch):
    from pipeline.source_assist import SourcingAssist

    _assert_never_classifies(monkeypatch)
    captured = {}

    def fake_assist(spec, intake_id, keywords=None, manual_query=None, provider=None, model=None, license_tier=None):
        captured["keywords"] = keywords
        return SourcingAssist(query="couch", note="stubbed"), []

    monkeypatch.setattr(asset_factory, "assist_imported", fake_assist)

    draft, logs = asset_factory.create_from_spec(_spec("couch"), forced_route="imported")

    assert draft.status == "needs_human"
    assert draft.resolution_method == "imported"
    assert captured["keywords"] is None  # no classifier ran, so no LLM keywords exist yet


def test_forced_route_blender_is_a_tbd_stub(sandbox):
    draft, logs = asset_factory.create_from_spec(_spec("widget"), forced_route="blender")

    assert draft.status == "needs_human"
    assert draft.resolution_method == "blender"
    assert "not implemented" in draft.message.lower()
    assert logs == []


def test_bind_material_attaches_material_and_pending_flag(sandbox, monkeypatch):
    from models.generation_models import MaterialGenerationResult, MaterialPlan
    from models.catalog_models import MaterialDef

    (sandbox / "library" / "materials").mkdir(parents=True, exist_ok=True)
    mat = MaterialDef(material_id="earth_mat", base_color="#3a6b9a")
    mat_path = sandbox / "library" / "materials" / "earth_mat.json"
    mat_path.write_text(json.dumps(mat.model_dump(mode="json")), encoding="utf-8")

    captured = {}

    def fake_generate_material(description, material_id=None, provider=None, model=None,
                               force=False, target_uv=None, target_bounds_m=None):
        captured.update(target_uv=target_uv, target_bounds_m=target_bounds_m)
        return (
            MaterialGenerationResult(
                material_id="earth_mat", success=True, material_path=str(mat_path),
                texture_source="pending", texture_pending=True, texture_query="earth daymap",
            ),
            MaterialPlan(material=mat, texture_need="authentic"),
            _log("material_generation"),
        )

    monkeypatch.setattr(asset_factory, "generate_material", fake_generate_material)

    from models.catalog_models import UVInfo
    draft = asset_factory.DraftAsset(
        asset_id="earth", status="existing", resolution_method="catalog_match",
        description="the planet earth", spec=_spec("earth"),
        bounds_m=[0.3, 0.3, 0.3],
        uv=UVInfo(status="builtin", convention="equirect"),
    )
    logs = asset_factory._bind_material(draft, None, None)

    assert len(logs) == 1
    assert draft.material.material_id == "earth_mat"
    assert draft.texture_pending is True
    assert "human sourcing" in draft.message
    assert captured["target_uv"].convention == "equirect"
    assert captured["target_bounds_m"] == [0.3, 0.3, 0.3]


# ── save / approve ────────────────────────────────────────────────────────

def _seed_draft(asset_id="bracket"):
    draft = asset_factory.DraftAsset(
        asset_id=asset_id, status="draft", resolution_method="parametric",
        description="an L bracket", spec=_spec(asset_id),
        plan=OpenSCADPlan(
            parameters={"width": 0.06}, scad_source=_SCAD,
            expected_bounds_m=[0.06, 0.04, 0.005],
        ),
        normalization=MeshNormalizationResult(
            asset_id=asset_id, success=True, pivot="center", scale_applied=1.0,
            final_bounds_m=[0.06, 0.04, 0.005],
        ),
        glb_address=f"library/generated/{asset_id}/model.glb",
        bounds_m=[0.06, 0.04, 0.005],
    )
    asset_factory.save_draft_state(draft)
    return draft


def test_save_draft_upserts_catalog_entry_with_generated_provenance(sandbox, monkeypatch):
    monkeypatch.setattr(asset_factory, "render_previews", lambda entries, force: ([], []))
    _seed_draft()

    entry = asset_factory.save_draft("bracket")

    assert entry.asset_class == "parametric"
    assert entry.review_level == 2
    assert entry.provenance.source_type == "generated"
    assert any("reviewed and saved by" in m for m in entry.provenance.modifications)
    [in_catalog] = load_catalog()
    assert in_catalog.asset_id == "bracket"
    assert asset_factory.load_draft_state("bracket").status == "saved"

    # Saving again after another iteration replaces, not duplicates.
    asset_factory.load_draft_state("bracket")
    asset_factory.save_draft("bracket")
    assert len(load_catalog()) == 1


def test_save_draft_composite_gets_composite_asset_class(sandbox, monkeypatch):
    """Regression: save_draft used to hardcode asset_class='parametric' for
    any draft with a glb_address, which would have mis-cataloged a baked
    composite (a snowman) as if it were an OpenSCAD part."""
    monkeypatch.setattr(asset_factory, "render_previews", lambda entries, force: ([], []))
    draft = asset_factory.DraftAsset(
        asset_id="frosty", status="draft", resolution_method="composite",
        description="a friendly snowman", spec=_spec("frosty"),
        composite_id="frosty_composite",
        glb_address="library/composites/frosty/model.glb",
        bounds_m=[0.2, 0.3, 0.2],
    )
    asset_factory.save_draft_state(draft)

    entry = asset_factory.save_draft("frosty")

    assert entry.asset_class == "composite"
    assert any("baked into one mesh" in m for m in entry.provenance.modifications)


def test_save_draft_without_glb_is_rejected(sandbox):
    draft = _seed_draft()
    draft.glb_address = None
    asset_factory.save_draft_state(draft)

    with pytest.raises(asset_factory.AssetFactoryError, match="no normalized GLB"):
        asset_factory.save_draft("bracket")


def test_save_binding_draft_records_approval_instead_of_cataloging(sandbox, monkeypatch):
    """Regression: 'Save' on a binding (moon = sphere + texture) used to fail
    with 'no normalized GLB' -- it must record who/when on the draft."""
    monkeypatch.setattr(asset_factory, "render_previews", lambda entries, force: ([], []))
    sphere = AssetCatalogEntry(
        asset_id="sphere_basic", display_name="Sphere", asset_class="primitive",
        address="Primitives/Sphere", canonical_bounds_m=[1, 1, 1],
        provenance=ProvenanceInfo(source_type="core"),
    )
    save_catalog([sphere])
    binding = asset_factory.DraftAsset(
        asset_id="moon", status="existing", resolution_method="catalog_match",
        description="the moon", spec=_spec("moon"), matched_asset_id="sphere_basic",
    )
    asset_factory.save_draft_state(binding)

    entry = asset_factory.save_draft("moon")

    assert entry.asset_id == "sphere_basic"  # underlying asset, no new entry
    assert len(load_catalog()) == 1
    saved = asset_factory.load_draft_state("moon")
    assert saved.status == "saved"
    assert saved.approved_by
    assert saved.approved_at is not None


def test_create_progress_events_are_emitted(sandbox, monkeypatch):
    record = ResolutionRecord(
        object_id="bracket", requested_asset_spec=_spec(),
        requires_author_review=True,
        review_reason="Classified as 'imported' (not yet implemented). X.",
    )
    _patch_stage_1_2(monkeypatch, record)
    events = []

    asset_factory.create_from_description("a couch", progress=events.append)

    assert any("Parsing description" in e for e in events)
    assert any("Checking local library" in e for e in events)
    assert any("searching allowlisted sources" in e for e in events)


def test_approve_asset_records_who_and_when(sandbox, monkeypatch):
    monkeypatch.setattr(asset_factory, "render_previews", lambda entries, force: ([], []))
    _seed_draft()
    asset_factory.save_draft("bracket")

    entry = asset_factory.approve_asset("bracket")

    assert "approved by" in entry.provenance.modifications[-1]


def test_approve_unknown_asset_raises(sandbox):
    with pytest.raises(asset_factory.AssetFactoryError, match="not found"):
        asset_factory.approve_asset("ghost")


# ── update_catalog_metadata (added 2026-07, "easier to search" request) ───

def test_update_catalog_metadata_renames_and_retags(sandbox, monkeypatch):
    monkeypatch.setattr(asset_factory, "render_previews", lambda entries, force: ([], []))
    _seed_draft()
    asset_factory.save_draft("bracket")

    entry = asset_factory.update_catalog_metadata(
        "bracket", display_name="Mounting Bracket v2", tags=["bracket", "mount", "6cm"]
    )

    assert entry.display_name == "Mounting Bracket v2"
    assert entry.tags == ["bracket", "mount", "6cm"]
    [in_catalog] = load_catalog()
    assert in_catalog.display_name == "Mounting Bracket v2"


def test_update_catalog_metadata_partial_update_leaves_other_field_alone(sandbox, monkeypatch):
    monkeypatch.setattr(asset_factory, "render_previews", lambda entries, force: ([], []))
    _seed_draft()
    asset_factory.save_draft("bracket")
    asset_factory.update_catalog_metadata("bracket", tags=["original"])

    entry = asset_factory.update_catalog_metadata("bracket", display_name="Renamed Only")

    assert entry.display_name == "Renamed Only"
    assert entry.tags == ["original"]  # untouched by the name-only edit


def test_update_catalog_metadata_rejects_blank_name(sandbox, monkeypatch):
    monkeypatch.setattr(asset_factory, "render_previews", lambda entries, force: ([], []))
    _seed_draft()
    asset_factory.save_draft("bracket")

    with pytest.raises(asset_factory.AssetFactoryError, match="cannot be blank"):
        asset_factory.update_catalog_metadata("bracket", display_name="   ")


def test_update_catalog_metadata_requires_some_field(sandbox, monkeypatch):
    monkeypatch.setattr(asset_factory, "render_previews", lambda entries, force: ([], []))
    _seed_draft()
    asset_factory.save_draft("bracket")

    with pytest.raises(asset_factory.AssetFactoryError, match="Nothing to update"):
        asset_factory.update_catalog_metadata("bracket")


def test_update_catalog_metadata_unknown_asset_raises(sandbox):
    with pytest.raises(asset_factory.AssetFactoryError, match="not found"):
        asset_factory.update_catalog_metadata("ghost", display_name="X")


# ── regenerate ────────────────────────────────────────────────────────────

def test_regenerate_with_parameters_recompiles_without_llm(sandbox, monkeypatch):
    _seed_draft()
    compiled = {}

    def fake_compile(source, scad_path, stl_path):
        compiled["source"] = source
        scad_path.parent.mkdir(parents=True, exist_ok=True)
        scad_path.write_text(source, encoding="utf-8")
        stl_path.write_bytes(b"stl")
        return True, ""

    import pipeline.openscad_generator as og
    monkeypatch.setattr(og, "compile_scad", fake_compile)
    monkeypatch.setattr(og, "measure_stl_bounds_m", lambda p: [0.12, 0.04, 0.005])

    def fake_normalize(asset_id, source_path, source_format, glb_path, target_size_m, pivot):
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(b"glb")
        return MeshNormalizationResult(
            asset_id=asset_id, success=True, glb_path=str(glb_path),
            final_bounds_m=[0.12, 0.04, 0.005], pivot=pivot, scale_applied=1.0,
            triangle_count_before=12, triangle_count_after=12,
        )

    monkeypatch.setattr(asset_factory, "normalize_mesh", fake_normalize)

    draft, logs = asset_factory.regenerate("bracket", parameters={"width": 0.12})

    assert logs == []  # no LLM call for a parameter-only edit
    assert "width = 0.12;" in compiled["source"]
    assert draft.plan.parameters["width"] == 0.12
    assert draft.bounds_m == [0.12, 0.04, 0.005]


def test_regenerate_requires_parametric_draft(sandbox):
    with pytest.raises(asset_factory.AssetFactoryError, match="No parametric draft"):
        asset_factory.regenerate("ghost", parameters={"w": 1})


def test_regenerate_requires_some_edit(sandbox):
    _seed_draft()
    with pytest.raises(asset_factory.AssetFactoryError, match="tweak"):
        asset_factory.regenerate("bracket")


# ── redirect_draft (manual override, added 2026-07) ───────────────────────

def _seed_stuck_draft(asset_id="frosty"):
    draft = asset_factory.DraftAsset(
        asset_id=asset_id, status="needs_human", resolution_method="imported",
        description="a friendly snowman", spec=_spec(asset_id),
        message="Classified as 'imported'.",
    )
    asset_factory.save_draft_state(draft)
    return draft


def test_redirect_unknown_draft_raises(sandbox):
    with pytest.raises(asset_factory.AssetFactoryError, match="No draft"):
        asset_factory.redirect_draft("ghost", "parametric")


def test_redirect_to_parametric_bypasses_classification(sandbox, monkeypatch):
    _seed_stuck_draft()

    def fake_generate(spec, asset_id, generate_plan_fn=None):
        stl = asset_factory._draft_dir(asset_id) / "source.stl"
        stl.parent.mkdir(parents=True, exist_ok=True)
        stl.write_bytes(b"stl")
        return (
            GenerationResult(
                asset_id=asset_id, success=True, scad_path="x.scad", stl_path=str(stl),
                actual_bounds_m=[0.2, 0.3, 0.2], expected_bounds_m=[0.2, 0.3, 0.2],
                repair_attempts=[{"attempt_number": 1, "scad_source": _SCAD, "success": True}],
            ),
            [_log("openscad_generation")],
        )

    def fake_normalize(asset_id, source_path, source_format, glb_path, target_size_m, pivot):
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(b"glb")
        return MeshNormalizationResult(
            asset_id=asset_id, success=True, glb_path=str(glb_path),
            final_bounds_m=[0.2, 0.3, 0.2], pivot=pivot, scale_applied=1.0,
            triangle_count_before=10, triangle_count_after=10,
        )

    monkeypatch.setattr(asset_factory, "generate_parametric_asset", fake_generate)
    monkeypatch.setattr(asset_factory, "normalize_mesh", fake_normalize)
    monkeypatch.setattr(asset_factory, "_bind_material", lambda draft, provider, model, progress=None: [])

    draft, logs = asset_factory.redirect_draft("frosty", "parametric")

    assert draft.status == "draft"
    assert draft.resolution_method == "parametric"
    assert draft.glb_address == "library/generated/frosty/model.glb"


def test_redirect_to_composite_uses_forced_parts(sandbox, monkeypatch):
    from models.classification_models import CompositePartPlan

    _seed_stuck_draft()
    forced_parts = [
        CompositePartPlan(description="large white sphere", color_hint="white"),
        CompositePartPlan(description="small white sphere", color_hint="white"),
        CompositePartPlan(description="thin black cylinder", color_hint="black"),
    ]
    monkeypatch.setattr(
        asset_factory, "decompose_into_primitives",
        lambda spec, max_parts=12, provider=None, model=None: (forced_parts, _log("composite_decomposition")),
    )
    captured = {}

    def fake_resolve(spec, catalog, forced_composite_parts=None):
        captured["parts"] = forced_composite_parts
        return (
            ResolutionRecord(
                object_id=spec.object_id, requested_asset_spec=spec,
                requires_author_review=False,
                review_reason=f"Resolved as composite: {spec.object_id}_composite",
            ),
            [],
        )

    monkeypatch.setattr(asset_factory, "resolve", fake_resolve)

    baked = asset_factory.DraftAsset(
        asset_id="frosty", status="draft", resolution_method="composite",
        description="a friendly snowman", spec=_spec("frosty"),
        composite_id="frosty_composite", glb_address="library/composites/frosty/model.glb",
        bounds_m=[0.2, 0.3, 0.2],
    )
    monkeypatch.setattr(
        asset_factory, "_materialize_and_bake_composite",
        lambda composite_id, asset_id, target_size_m, provider, model, progress: (baked, [_log("material_generation")]),
    )

    draft, logs = asset_factory.redirect_draft("frosty", "composite")

    assert draft.glb_address == "library/composites/frosty/model.glb"
    assert captured["parts"] == forced_parts
    assert len(logs) == 2  # decomposition + the (faked) materialize/bake call's log


def test_redirect_to_imported_passes_manual_query(sandbox, monkeypatch):
    from pipeline.source_assist import SourcingAssist

    _seed_stuck_draft()
    captured = {}

    def fake_assist(spec, intake_id, keywords=None, manual_query=None, provider=None, model=None, license_tier=None):
        captured["manual_query"] = manual_query
        return (
            SourcingAssist(query=manual_query or "x", note="2 candidate(s) found"),
            [],
        )

    monkeypatch.setattr(asset_factory, "assist_imported", fake_assist)

    draft, logs = asset_factory.redirect_draft("frosty", "imported", manual_query="snowman christmas decoration")

    assert captured["manual_query"] == "snowman christmas decoration"
    assert draft.status == "needs_human"


# ── _materialize_and_bake_composite (added 2026-07) ───────────────────────
# Regression coverage for "Frosty the Snowman is all gray": composites used
# to never generate per-part materials or bake a real mesh at all.

def _seed_fragment(sandbox, composite_id="frosty_composite", n_parts=2):
    from pipeline.composite_builder import (
        CompositeFragment, CompositePart, save_composite_fragment,
    )

    parts = [
        CompositePart(
            part_id=f"p{i}",
            asset_spec=AssetSpec(object_id=f"p{i}", description=f"part {i}"),
            resolved_asset_id="sphere_basic",
            color_hint=("white" if i == 0 else "black"),
        )
        for i in range(n_parts)
    ]
    fragment = CompositeFragment(
        composite_id=composite_id, display_name="Frosty", source_description="a snowman",
        parts=parts,
    )
    save_composite_fragment(fragment)
    return fragment


def test_materialize_and_bake_generates_one_material_per_part(sandbox, monkeypatch):
    from models.generation_models import MaterialGenerationResult

    _seed_fragment(sandbox, n_parts=3)
    calls = []

    def fake_generate_material(description, material_id=None, provider=None, model=None, force=False):
        calls.append((description, material_id))
        return (
            MaterialGenerationResult(material_id=material_id, success=True),
            None, _log("material_generation"),
        )

    monkeypatch.setattr(asset_factory, "generate_material", fake_generate_material)

    def raising_bake(fragment, asset_id, target_size_m):
        raise asset_factory.CompositeBakeError("stub: bake not under test")

    monkeypatch.setattr(asset_factory, "bake_composite", raising_bake)

    draft, logs = asset_factory._materialize_and_bake_composite(
        "frosty_composite", "frosty", None, None, None, lambda m: None
    )

    assert len(calls) == 3
    assert calls[0][1] == "frosty_composite_p0_mat"
    assert draft.status == "needs_human"  # bake failed, but materials were still generated
    from pipeline.composite_builder import load_composite_fragment
    fragment = load_composite_fragment("frosty_composite")
    assert all(p.material_id == f"frosty_composite_{p.part_id}_mat" for p in fragment.parts)


def test_materialize_and_bake_success_populates_glb_and_bounds(sandbox, monkeypatch):
    from models.generation_models import MaterialGenerationResult
    from pipeline.composite_baker import CompositeBakeResult

    _seed_fragment(sandbox, n_parts=2)
    monkeypatch.setattr(
        asset_factory, "generate_material",
        lambda description, material_id=None, provider=None, model=None, force=False: (
            MaterialGenerationResult(material_id=material_id, success=True), None, _log("material_generation"),
        ),
    )
    monkeypatch.setattr(
        asset_factory, "bake_composite",
        lambda fragment, asset_id, target_size_m: CompositeBakeResult(
            asset_id=asset_id, success=True, glb_path=str(sandbox / "x" / "model.glb"),
            final_bounds_m=[0.2, 0.3, 0.2], triangle_count=500, uv_status="generated",
        ),
    )

    draft, logs = asset_factory._materialize_and_bake_composite(
        "frosty_composite", "frosty", 0.3, None, None, lambda m: None
    )

    assert draft.status == "draft"
    assert draft.resolution_method == "composite"
    assert draft.glb_address == "library/composites/frosty/model.glb"
    assert draft.bounds_m == [0.2, 0.3, 0.2]
    assert draft.uv.status == "generated"


def test_materialize_and_bake_reports_bake_failure(sandbox, monkeypatch):
    from models.generation_models import MaterialGenerationResult
    from pipeline.composite_baker import CompositeBakeResult

    _seed_fragment(sandbox, n_parts=1)
    monkeypatch.setattr(
        asset_factory, "generate_material",
        lambda description, material_id=None, provider=None, model=None, force=False: (
            MaterialGenerationResult(material_id=material_id, success=True), None, _log("material_generation"),
        ),
    )
    monkeypatch.setattr(
        asset_factory, "bake_composite",
        lambda fragment, asset_id, target_size_m: CompositeBakeResult(
            asset_id=asset_id, success=False, error_message="Blender crashed",
        ),
    )

    draft, logs = asset_factory._materialize_and_bake_composite(
        "frosty_composite", "frosty", None, None, None, lambda m: None
    )

    assert draft.status == "needs_human"
    assert "Blender crashed" in draft.message


# ── edit_composite (added 2026-07, "I can't edit it" / methane incident) ──
# Mirrors the OpenSCAD regenerate() coverage above: numeric/geometry edits
# must not cost an LLM call, and rebake using the edited fragment.

def _seed_composite_draft(sandbox, asset_id="molecule", composite_id="molecule_composite"):
    from pipeline.composite_builder import CompositeFragment, CompositePart, save_composite_fragment

    fragment = CompositeFragment(
        composite_id=composite_id, display_name="Molecule", source_description="methane",
        parts=[
            CompositePart(
                part_id="C", asset_spec=AssetSpec(object_id="C", description="carbon"),
                resolved_asset_id="sphere_basic", label="C", position=[0.0, 0.0, 0.0],
                color_hint="dark gray",
            ),
            CompositePart(
                part_id="H1", asset_spec=AssetSpec(object_id="H1", description="hydrogen"),
                resolved_asset_id="sphere_basic", label="H1", position=[0.3, 0.3, 0.3],
                color_hint="white",
            ),
            CompositePart(
                part_id="bond1", asset_spec=AssetSpec(object_id="bond1", description="C-H bond"),
                bond_between=["C", "H1"], bond_thickness=0.06,
            ),
        ],
    )
    save_composite_fragment(fragment)
    draft = asset_factory.DraftAsset(
        asset_id=asset_id, status="draft", resolution_method="composite",
        description="methane", spec=_spec(asset_id), composite_id=composite_id,
        glb_address=f"library/composites/{asset_id}/model.glb", bounds_m=[0.3, 0.3, 0.3],
    )
    asset_factory.save_draft_state(draft)
    return draft, fragment


def test_edit_composite_updates_geometry_without_llm(sandbox, monkeypatch):
    from pipeline.composite_baker import CompositeBakeResult
    from pipeline.composite_builder import load_composite_fragment

    _seed_composite_draft(sandbox)

    def fake_bake(fragment, asset_id, target_size_m):
        return CompositeBakeResult(
            asset_id=asset_id, success=True,
            final_bounds_m=[0.4, 0.4, 0.4], triangle_count=100, uv_status="none",
        )

    monkeypatch.setattr(asset_factory, "bake_composite", fake_bake)

    draft, logs = asset_factory.edit_composite(
        "molecule",
        part_edits={"H1": {"position": [0.5, 0.5, 0.5]}, "bond1": {"bond_thickness": 0.1}},
    )

    assert logs == []  # no LLM call for a geometry-only edit
    assert draft.status == "draft"
    assert draft.bounds_m == [0.4, 0.4, 0.4]
    fragment = load_composite_fragment("molecule_composite")
    h1 = next(p for p in fragment.parts if p.part_id == "H1")
    assert h1.position == [0.5, 0.5, 0.5]
    bond = next(p for p in fragment.parts if p.part_id == "bond1")
    assert bond.bond_thickness == 0.1


def test_edit_composite_color_hex_writes_material_directly(sandbox, monkeypatch):
    from pipeline.composite_baker import CompositeBakeResult
    from models.catalog_models import MaterialDef

    _seed_composite_draft(sandbox)
    monkeypatch.setattr(
        asset_factory, "bake_composite",
        lambda fragment, asset_id, target_size_m: CompositeBakeResult(
            asset_id=asset_id, success=True, final_bounds_m=[0.3, 0.3, 0.3],
            triangle_count=10, uv_status="none",
        ),
    )

    draft, logs = asset_factory.edit_composite("molecule", part_edits={"C": {"color_hex": "#112233"}})

    assert logs == []  # no LLM call for a raw hex color edit
    mat_path = config.MATERIALS_DIR / "molecule_composite_C_mat.json"
    assert mat_path.is_file()
    mat = MaterialDef.model_validate_json(mat_path.read_text(encoding="utf-8"))
    assert mat.base_color == "#112233"


def test_edit_composite_rejects_bad_hex(sandbox):
    _seed_composite_draft(sandbox)
    with pytest.raises(asset_factory.AssetFactoryError, match="rrggbb"):
        asset_factory.edit_composite("molecule", part_edits={"C": {"color_hex": "red"}})


def test_edit_composite_rejects_unknown_part(sandbox):
    _seed_composite_draft(sandbox)
    with pytest.raises(asset_factory.AssetFactoryError, match="Unknown part"):
        asset_factory.edit_composite("molecule", part_edits={"ghost": {"position": [0, 0, 0]}})


def test_edit_composite_requires_composite_draft(sandbox):
    _seed_draft()  # a parametric draft, no composite_id
    with pytest.raises(asset_factory.AssetFactoryError, match="No composite draft"):
        asset_factory.edit_composite("bracket", part_edits={})


def test_edit_composite_regenerate_materials_for_calls_llm(sandbox, monkeypatch):
    from models.generation_models import MaterialGenerationResult
    from pipeline.composite_baker import CompositeBakeResult

    _seed_composite_draft(sandbox)
    calls = []

    def fake_generate_material(description, material_id=None, provider=None, model=None, force=False):
        calls.append((description, material_id))
        return (
            MaterialGenerationResult(material_id=material_id, success=True),
            None, _log("material_generation"),
        )

    monkeypatch.setattr(asset_factory, "generate_material", fake_generate_material)
    monkeypatch.setattr(
        asset_factory, "bake_composite",
        lambda fragment, asset_id, target_size_m: CompositeBakeResult(
            asset_id=asset_id, success=True, final_bounds_m=[0.3, 0.3, 0.3],
            triangle_count=10, uv_status="none",
        ),
    )

    draft, logs = asset_factory.edit_composite(
        "molecule", part_edits={"C": {"color_hint": "jet black"}},
        regenerate_materials_for=["C"],
    )

    assert len(logs) == 1  # exactly the one requested material regen
    assert calls[0] == ("jet black", "molecule_composite_C_mat")


def test_get_composite_fragment_returns_parts(sandbox):
    _seed_composite_draft(sandbox)
    fragment = asset_factory.get_composite_fragment("molecule")
    assert {p.part_id for p in fragment.parts} == {"C", "H1", "bond1"}


# ── review_and_repair (added 2026-07, "methane sticks don't connect"
#    visual-review auto-repair incident) ──────────────────────────────────

def test_review_and_repair_records_matching_verdict(sandbox, monkeypatch):
    from pipeline.visual_review import AssetVisualReview

    _seed_composite_draft(sandbox)
    monkeypatch.setattr(asset_factory, "render_glb_to_png", lambda glb, path: path)
    monkeypatch.setattr(
        asset_factory, "review_asset_render",
        lambda description, image_path, provider=None, model=None: (
            AssetVisualReview(matches=True, confidence=0.9, reasoning="looks correct"),
            _log("visual_review"),
        ),
    )

    draft, review, logs = asset_factory.review_and_repair("molecule")

    assert review.matches is True
    assert draft.visual_review.matches is True
    assert "matches the request" in draft.message
    assert len(logs) == 1  # just the review call, no repair triggered


def test_review_and_repair_triggers_composite_revision_on_mismatch(sandbox, monkeypatch):
    from pipeline.visual_review import AssetVisualReview
    from models.classification_models import CompositePartPlan
    from pipeline.composite_baker import CompositeBakeResult

    _seed_composite_draft(sandbox)
    monkeypatch.setattr(asset_factory, "render_glb_to_png", lambda glb, path: path)
    monkeypatch.setattr(
        asset_factory, "review_asset_render",
        lambda description, image_path, provider=None, model=None: (
            AssetVisualReview(
                matches=False, confidence=0.85,
                reasoning="the bond doesn't reach H1",
                suggested_fix="recompute the bond's geometry from the real atom positions",
            ),
            _log("visual_review"),
        ),
    )
    captured = {}

    def fake_revise(spec, current_parts, feedback, max_parts=12, provider=None, model=None):
        captured["current_parts"] = current_parts
        captured["feedback"] = feedback
        revised = [
            CompositePartPlan(description="carbon sphere", color_hint="dark gray", label="C"),
            CompositePartPlan(description="hydrogen sphere", color_hint="white", label="H1"),
            CompositePartPlan(description="bond", color_hint="white", bond_between=["C", "H1"], bond_thickness=0.08),
        ]
        return revised, _log("composite_revision")

    monkeypatch.setattr(asset_factory, "revise_composite_parts", fake_revise)

    def fake_resolve(spec, catalog, forced_composite_parts=None):
        captured["forced_parts"] = forced_composite_parts
        return (
            ResolutionRecord(
                object_id=spec.object_id, requested_asset_spec=spec,
                requires_author_review=False,
                review_reason=f"Resolved as composite: {spec.object_id}_composite",
            ),
            [],
        )

    monkeypatch.setattr(asset_factory, "resolve", fake_resolve)

    baked = asset_factory.DraftAsset(
        asset_id="molecule", status="draft", resolution_method="composite",
        description="methane", spec=_spec("molecule"),
        composite_id="molecule_composite", glb_address="library/composites/molecule/model.glb",
        bounds_m=[0.3, 0.3, 0.3],
    )
    monkeypatch.setattr(
        asset_factory, "_materialize_and_bake_composite",
        lambda composite_id, asset_id, target_size_m, provider, model, progress: (
            baked, [_log("material_generation")]
        ),
    )

    draft, review, logs = asset_factory.review_and_repair("molecule")

    assert review.matches is False
    assert len(captured["forced_parts"]) == 3  # the revised part list was passed straight through
    assert captured["forced_parts"][2].bond_thickness == 0.08
    assert len(captured["current_parts"]) == 3  # C, H1, bond1 from the seeded fragment
    assert "bond doesn't reach" in captured["feedback"]
    assert draft.glb_address == "library/composites/molecule/model.glb"
    assert draft.visual_review.matches is False
    assert len(logs) == 3  # review + revision + the (faked) materialize/bake log


def test_review_and_repair_skips_repair_when_auto_repair_false(sandbox, monkeypatch):
    from pipeline.visual_review import AssetVisualReview

    _seed_composite_draft(sandbox)
    monkeypatch.setattr(asset_factory, "render_glb_to_png", lambda glb, path: path)
    monkeypatch.setattr(
        asset_factory, "review_asset_render",
        lambda description, image_path, provider=None, model=None: (
            AssetVisualReview(matches=False, confidence=0.7, reasoning="wrong colors", suggested_fix="fix colors"),
            _log("visual_review"),
        ),
    )
    monkeypatch.setattr(
        asset_factory, "revise_composite_parts",
        lambda *a, **k: pytest.fail("auto_repair=False must not call the repair path"),
    )

    draft, review, logs = asset_factory.review_and_repair("molecule", auto_repair=False)

    assert review.matches is False
    assert "MISMATCH" in draft.message
    assert len(logs) == 1


def test_discard_draft_removes_parametric_draft_dir(sandbox):
    _seed_draft()
    assert asset_factory.draft_path("bracket").exists()

    asset_factory.discard_draft("bracket")

    assert not asset_factory._draft_dir("bracket").exists()
    assert asset_factory.load_draft_state("bracket") is None


def test_discard_draft_removes_composite_fragment_and_bake(sandbox):
    from pipeline.composite_builder import composite_path

    _seed_composite_draft(sandbox)
    frag_path = composite_path("molecule_composite")
    assert frag_path.exists()
    baked_dir = config.LIBRARY_DIR / "composites" / "molecule"
    baked_dir.mkdir(parents=True, exist_ok=True)
    (baked_dir / "model.glb").write_bytes(b"glb")

    asset_factory.discard_draft("molecule")

    assert not asset_factory._draft_dir("molecule").exists()
    assert not frag_path.exists()
    assert not baked_dir.exists()


def test_discard_draft_refuses_saved_draft(sandbox, monkeypatch):
    monkeypatch.setattr(asset_factory, "render_previews", lambda entries, force: ([], []))
    _seed_draft()
    asset_factory.save_draft("bracket")

    with pytest.raises(asset_factory.AssetFactoryError, match="already been saved"):
        asset_factory.discard_draft("bracket")

    assert asset_factory.draft_path("bracket").exists()  # untouched


def test_discard_draft_requires_existing_draft(sandbox):
    with pytest.raises(asset_factory.AssetFactoryError, match="No draft found"):
        asset_factory.discard_draft("ghost")


def test_review_and_repair_requires_glb(sandbox):
    draft = asset_factory.DraftAsset(
        asset_id="ghost", status="needs_human", resolution_method="imported",
        description="x", spec=_spec("ghost"),
    )
    asset_factory.save_draft_state(draft)

    with pytest.raises(asset_factory.AssetFactoryError, match="no renderable GLB"):
        asset_factory.review_and_repair("ghost")
