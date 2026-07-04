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
    (lib / "generated").mkdir(parents=True)
    (lib / "previews").mkdir(parents=True)
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
        lambda spec, intake_id, keywords=None, manual_query=None, provider=None, model=None: (
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
    _seed_stuck_draft()

    monkeypatch.setattr(
        asset_factory, "decompose_into_primitives",
        lambda spec, max_parts=12, provider=None, model=None: (
            ["large white sphere", "small white sphere", "thin black cylinder"],
            _log("composite_decomposition"),
        ),
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

    draft, logs = asset_factory.redirect_draft("frosty", "composite")

    assert draft.status == "composite"
    assert draft.composite_id == "frosty_composite"
    assert captured["parts"] == ["large white sphere", "small white sphere", "thin black cylinder"]
    assert len(logs) == 1  # only the decomposition call -- resolve() was faked with no logs


def test_redirect_to_imported_passes_manual_query(sandbox, monkeypatch):
    from pipeline.source_assist import SourcingAssist

    _seed_stuck_draft()
    captured = {}

    def fake_assist(spec, intake_id, keywords=None, manual_query=None, provider=None, model=None):
        captured["manual_query"] = manual_query
        return (
            SourcingAssist(query=manual_query or "x", note="2 candidate(s) found"),
            [],
        )

    monkeypatch.setattr(asset_factory, "assist_imported", fake_assist)

    draft, logs = asset_factory.redirect_draft("frosty", "imported", manual_query="snowman christmas decoration")

    assert captured["manual_query"] == "snowman christmas decoration"
    assert draft.status == "needs_human"
