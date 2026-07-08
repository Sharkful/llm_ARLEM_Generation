"""Stage 8b orchestration: description -> draft asset -> iterate -> save.

This is the seam the Flask review app (webapp/app.py) calls, and the same
seam Stage 9's resolve-scene batch will reuse: one function per lifecycle
step, no Flask/HTTP types anywhere here.

Lifecycle:
  create_from_description()  Stage 1 spec -> Stage 2 resolve -> route:
                               catalog_match/variant -> point at the match
                               parametric -> Stage 3 generate + Stage 4 normalize
                               composite -> fragment reference
                               imported/unclear -> needs_human
  regenerate()               param substitution (no LLM) or a reviewer
                               tweak instruction (LLM revision) -> recompile
                               -> renormalize
  save_draft()               promote the draft into catalog.json with
                               generated provenance + preview render
  approve_asset()            record the human approval (who/when) on an
                               existing catalog entry -- req. doc section 18's
                               human checkpoint

Drafts persist as library/generated/<asset_id>/draft.json so an app restart
(or a long worklist session) doesn't lose in-progress work.
"""
from __future__ import annotations

import getpass
import json
import re
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Literal, Optional

# Progress callback: receives one human-readable line per pipeline step so
# UIs can show "checking local library for possible matches..." live.
ProgressFn = Callable[[str], None]


def _noop_progress(message: str) -> None:
    pass

from pydantic import BaseModel, Field

import config
from models.catalog_models import AssetCatalogEntry, MaterialDef, ProvenanceInfo, UVInfo
from models.generation_models import GenerationResult, OpenSCADPlan
from models.log_models import LLMCallEntry
from models.spec_models import AssetSpec
from pipeline import uv_tools
from pipeline.catalog_matcher import find_candidates
from pipeline.catalog_writer import load_catalog, save_catalog
from pipeline.classifier import VARIANT_FLOOR
from pipeline.composite_baker import CompositeBakeError, bake_composite
from pipeline.composite_builder import (
    CompositeFragment,
    decompose_into_primitives,
    fragment_to_part_plans,
    load_composite_fragment,
    revise_composite_parts,
    save_composite_fragment,
)
from pipeline.source_assist import SourceCandidate, adopt_candidate, assist_imported
from pipeline.material_generator import MaterialGenerationError, generate_material
from pipeline.mesh_processor import MeshNormalizationResult, normalize_mesh, save_mesh_meta
from pipeline.openscad_generator import generate_openscad_plan, generate_parametric_asset
from pipeline.preview_renderer import PreviewError, render_glb_to_png, render_previews
from pipeline.resolver import resolve
from pipeline.spec_parser import parse_description
from pipeline.visual_review import AssetVisualReview, VisualReviewError, review_asset_render

DEFAULT_TARGET_SIZE_M = 0.15  # when neither the author nor the LLM sized it


class AssetFactoryError(RuntimeError):
    pass


class DraftAsset(BaseModel):
    asset_id: str
    status: Literal["draft", "saved", "needs_human", "existing", "composite"]
    resolution_method: str
    description: str
    spec: AssetSpec
    plan: Optional[OpenSCADPlan] = None
    generation: Optional[GenerationResult] = None
    normalization: Optional[MeshNormalizationResult] = None
    glb_address: Optional[str] = None  # library/... path for the viewport
    bounds_m: Optional[list[float]] = None
    matched_asset_id: Optional[str] = None
    composite_id: Optional[str] = None
    approved_by: Optional[str] = None  # human sign-off on a binding (who/when)
    approved_at: Optional[datetime] = None
    uv: Optional[UVInfo] = None
    material: Optional[MaterialDef] = None  # Stage 6c binding result
    texture_pending: bool = False  # authentic surface awaiting human sourcing
    texture_query: Optional[str] = None
    candidates: list[SourceCandidate] = Field(default_factory=list)  # sourcing assist hits
    suggested_sources: list[str] = Field(default_factory=list)  # search URLs when no hits
    message: Optional[str] = None
    visual_review: Optional[AssetVisualReview] = None  # Stage 8b.5 review verdict
    updated_at: datetime = Field(default_factory=datetime.utcnow)


def _draft_dir(asset_id: str) -> Path:
    return config.LIBRARY_DIR / "generated" / asset_id


def draft_path(asset_id: str) -> Path:
    return _draft_dir(asset_id) / "draft.json"


def save_draft_state(draft: DraftAsset) -> Path:
    path = draft_path(draft.asset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(draft.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def load_draft_state(asset_id: str) -> DraftAsset | None:
    path = draft_path(asset_id)
    if not path.is_file():
        return None
    return DraftAsset.model_validate_json(path.read_text(encoding="utf-8-sig"))


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:40] or "asset"


def _normalize_draft_mesh(draft: DraftAsset) -> DraftAsset:
    """Stage 4 pass over the draft's STL -> library/generated/<id>/model.glb."""
    stl_path = Path(draft.generation.stl_path)
    glb_path = _draft_dir(draft.asset_id) / "model.glb"
    target = (
        draft.spec.desired_size_m
        or (max(draft.generation.actual_bounds_m) if draft.generation.actual_bounds_m else None)
        or DEFAULT_TARGET_SIZE_M
    )
    normalization = normalize_mesh(
        asset_id=draft.asset_id,
        source_path=stl_path,
        source_format="stl",
        glb_path=glb_path,
        target_size_m=target,
        pivot="center",
    )
    draft.normalization = normalization
    if normalization.success:
        save_mesh_meta(draft.asset_id, normalization)
        draft.glb_address = f"library/generated/{draft.asset_id}/model.glb"
        draft.bounds_m = normalization.final_bounds_m
        draft.uv = _record_uv(draft.asset_id, glb_path, normalization.uv_status)
    else:
        draft.message = f"Mesh normalization failed: {normalization.error_message}"
    return draft


def _record_uv(asset_id: str, glb_path: Path, uv_status: str) -> UVInfo:
    """Stage 6c.2: hash the UV layer and export the island layout image."""
    if uv_status == "none":
        return UVInfo(status="none", convention="none")
    layout = uv_tools.export_uv_layout(glb_path, glb_path.parent / "uv_layout.png")
    return UVInfo(
        status=uv_status,
        convention="generic",
        layout_image=(
            f"generated/{asset_id}/uv_layout.png" if layout is not None else None
        ),
        uv_hash=uv_tools.uv_hash(glb_path),
    )


def _existing_match_needs_material_bind(matched: AssetCatalogEntry | None) -> bool:
    """An 'existing' match only needs a NEW bound material when it's a bare
    primitive/variant becoming identifiable through it (the moon = sphere +
    moon texture). An 'imported' match already carries its own real,
    non-regeneratable textures baked into the GLB -- binding a synthetic
    material over it silently replaces those real colors with a flat
    procedural one (2026-07, 'lunar module comes back gray' incident: fixing
    the catalog matcher to correctly resolve to lunar_excursion_module then
    surfaced this pre-existing bug, previously masked because most matches
    were bare primitives). Mirrors the "no bind" comment on the sourcing
    assist's auto-adopted-imported branch below, which never had this bug.
    """
    return matched is not None and matched.asset_class != "imported"


def _bind_material(
    draft: DraftAsset,
    provider: str | None,
    model: str | None,
    progress: ProgressFn | None = None,
) -> list[LLMCallEntry]:
    """Stage 6c.6: run the material/texture ladder for this draft's surface.

    Non-fatal by design -- a draft with no material is still reviewable; the
    message records what went wrong instead of failing the whole create.
    """
    progress = progress or _noop_progress
    progress("Binding material and texture (archive first, synthesis or sourcing after)...")
    try:
        result, plan, entry = generate_material(
            draft.description,
            material_id=f"{draft.asset_id}_mat",
            provider=provider,
            model=model,
            force=True,  # draft iteration overwrites its own material freely
            target_uv=draft.uv,
            target_bounds_m=draft.bounds_m,
        )
    except (MaterialGenerationError, RuntimeError) as exc:
        draft.message = f"{draft.message or ''} (material step failed: {exc})".strip()
        return []
    material = MaterialDef.model_validate_json(
        Path(result.material_path).read_text(encoding="utf-8-sig")
    )
    draft.material = material
    draft.texture_pending = result.texture_pending
    draft.texture_query = result.texture_query
    if result.texture_pending:
        draft.message = (
            f"{draft.message or ''} Texture flagged for human sourcing "
            f"(query: {result.texture_query!r}) -- intake a real map, then regenerate."
        ).strip()
    return [entry]


def _materialize_and_bake_composite(
    composite_id: str,
    asset_id: str,
    target_size_m: float | None,
    provider: str | None,
    model: str | None,
    progress: ProgressFn,
) -> tuple[DraftAsset, list[LLMCallEntry]]:
    """Give every part of a resolved composite its own material (reusing
    Stage 5, one call per part -- a snowman's hat/eyes/nose/arms all need
    independent colors, which is the whole point of building it as a
    composite rather than one fused CSG solid), then bake the assembly into
    one GLB (composite_baker.py). Shared by create_from_spec's composite
    branch and redirect_draft's forced-composite path.
    """
    logs: list[LLMCallEntry] = []
    fragment = load_composite_fragment(composite_id)
    if fragment is None:
        raise AssetFactoryError(f"Composite fragment {composite_id!r} not found after resolve().")

    progress(f"Generating a material for each of {len(fragment.parts)} part(s)...")
    for part in fragment.parts:
        if not part.resolved_asset_id:
            continue  # unresolved parts are skipped by the baker too
        try:
            result, _, entry = generate_material(
                part.color_hint or part.asset_spec.description,
                material_id=f"{composite_id}_{part.part_id}_mat",
                provider=provider, model=model, force=True,
            )
            logs.append(entry)
            part.material_id = result.material_id if result.success else None
        except (MaterialGenerationError, RuntimeError) as exc:
            part.material_id = None
            progress(f"  material failed for {part.part_id!r} ({exc}); it will use a default gray.")
    save_composite_fragment(fragment)

    progress("Baking the assembly into one multi-material mesh (Blender)...")
    size = target_size_m or DEFAULT_TARGET_SIZE_M
    try:
        bake = bake_composite(fragment, asset_id, target_size_m=size)
    except CompositeBakeError as exc:
        draft = DraftAsset(
            asset_id=asset_id, status="needs_human", resolution_method="composite",
            description=fragment.source_description,
            spec=fragment.parts[0].asset_spec.model_copy(update={"object_id": asset_id})
            if fragment.parts else AssetSpec(object_id=asset_id, description=fragment.source_description),
            composite_id=composite_id, message=f"Composite baking unavailable: {exc}",
        )
        save_draft_state(draft)
        return draft, logs
    if not bake.success:
        draft = DraftAsset(
            asset_id=asset_id, status="needs_human", resolution_method="composite",
            description=fragment.source_description,
            spec=AssetSpec(object_id=asset_id, description=fragment.source_description),
            composite_id=composite_id, message=f"Composite baking failed: {bake.error_message}",
        )
        save_draft_state(draft)
        return draft, logs

    uv = UVInfo(status=bake.uv_status, convention="generic") if bake.uv_status != "none" else UVInfo()
    message = "Built as a composite and baked into one mesh -- each part has its own color."
    if bake.error_message:  # non-fatal notes (skipped parts) even on success
        message += f" ({bake.error_message})"
    draft = DraftAsset(
        asset_id=asset_id, status="draft", resolution_method="composite",
        description=fragment.source_description,
        spec=AssetSpec(object_id=asset_id, description=fragment.source_description, desired_size_m=size),
        composite_id=composite_id,
        glb_address=f"library/composites/{asset_id}/model.glb",
        bounds_m=bake.final_bounds_m, uv=uv, message=message,
    )
    save_draft_state(draft)
    return draft, logs


# A human can pick the creation route up front instead of trusting
# resolve()'s catalog-match/LLM-classification decision tree -- added so the
# same description can be A/B tested across routes (does this snowman look
# better hand-composited or parametric?) without first having to land on a
# stuck/wrong draft to redirect. "existing" mirrors classify()'s own
# catalog-match/variant check but skips the LLM classification fallback
# when nothing matches; "imported"/"parametric"/"composite" mirror
# redirect_draft()'s post-hoc override paths below, run up front instead.
# "blender" is a placeholder for a planned native-Blender-script generation
# route (analogous to how "parametric" has the LLM write an OpenSCAD script)
# that does not exist yet -- forcing it returns a needs_human draft that
# says so, rather than silently falling back to another route.
ForcedRoute = Literal["existing", "imported", "parametric", "composite", "blender"]


def create_from_description(
    description: str,
    asset_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    progress: ProgressFn | None = None,
    forced_route: ForcedRoute | None = None,
    license_tier: str | None = None,
) -> tuple[DraftAsset, list[LLMCallEntry]]:
    """One-shot flow: free text -> routed, generated (if parametric), normalized draft.

    forced_route: bypass automatic routing and build via this route instead
    (see ForcedRoute above).
    license_tier: source_assist.LICENSE_TIERS key from the review app's
    pull-down; governs which licenses the imported route's search shows.
    """
    progress = progress or _noop_progress
    progress("Parsing description into a structured asset spec (LLM)...")
    spec, entry = parse_description(description, provider=provider, model=model)
    if asset_id:
        spec = spec.model_copy(update={"object_id": asset_id})
    draft, logs = create_from_spec(
        spec, asset_id=asset_id, provider=provider, model=model, progress=progress,
        forced_route=forced_route, license_tier=license_tier,
    )
    return draft, [entry, *logs]


def create_from_spec(
    spec: AssetSpec,
    asset_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    progress: ProgressFn | None = None,
    forced_route: ForcedRoute | None = None,
    license_tier: str | None = None,
) -> tuple[DraftAsset, list[LLMCallEntry]]:
    """Same flow starting from an already-parsed AssetSpec (Stage 9 batch
    input, or a worklist of pre-authored specs -- skips the Stage 1 call).

    forced_route: bypass automatic routing and build via this route instead
    (see ForcedRoute above).
    license_tier: see create_from_description.
    """
    progress = progress or _noop_progress
    final_id = asset_id or _slug(spec.object_id)

    if forced_route is not None:
        return _create_via_forced_route(
            spec, final_id, forced_route, provider, model, progress,
            license_tier=license_tier,
        )

    logs: list[LLMCallEntry] = []
    description = spec.description

    catalog = load_catalog()
    progress(
        f"Checking local library for possible matches ({len(catalog)} assets, "
        "then LLM classification if nothing fits)..."
    )
    record, resolve_logs = resolve(spec, catalog)
    logs.extend(resolve_logs)

    if record.resolved_asset is not None:
        # catalog_match / variant: nothing to build -- show the match.
        progress(
            f"Matched existing catalog asset {record.resolved_asset.asset_id!r} "
            f"({record.resolved_asset.resolution_method}, confidence "
            f"{record.resolved_asset.confidence:.2f}) -- no new geometry needed."
        )
        matched = next(
            (e for e in catalog if e.asset_id == record.resolved_asset.asset_id), None
        )
        draft = DraftAsset(
            asset_id=final_id,
            status="existing",
            resolution_method=record.resolved_asset.resolution_method,
            description=description,
            spec=spec,
            matched_asset_id=record.resolved_asset.asset_id,
            glb_address=matched.address if matched and matched.address.startswith("library/") else None,
            bounds_m=list(matched.canonical_bounds_m) if matched else None,
            uv=matched.uv if matched else None,
            message=(
                f"Resolved to existing catalog asset {record.resolved_asset.asset_id!r} "
                f"({record.resolved_asset.resolution_method}, "
                f"confidence {record.resolved_asset.confidence:.2f})."
            ),
        )
        if _existing_match_needs_material_bind(matched):
            logs.extend(_bind_material(draft, provider, model, progress=progress))
        save_draft_state(draft)
        return draft, logs

    # Route on the resolver's exact message prefixes (resolver.py owns these
    # strings; test_asset_factory pins them so a resolver wording change
    # fails loudly here instead of silently misrouting).
    reason = record.review_reason or ""
    if reason.startswith("Resolved as composite:") and not record.requires_author_review:
        composite_id = f"{spec.object_id}_composite"
        progress(f"Classified as composite -- resolved into {composite_id!r}, materializing...")
        draft, bake_logs = _materialize_and_bake_composite(
            composite_id, final_id, spec.desired_size_m, provider, model, progress
        )
        logs.extend(bake_logs)
        return draft, logs

    if reason.startswith("Classified as 'parametric'"):
        progress(
            "Classified as parametric -- generating an OpenSCAD script (LLM) and "
            "compiling it (up to 3 repair attempts)..."
        )
        generation, gen_logs = generate_parametric_asset(spec, final_id)
        logs.extend(gen_logs)
        if not generation.success:
            draft = DraftAsset(
                asset_id=final_id, status="needs_human", resolution_method="parametric",
                description=description, spec=spec, generation=generation,
                message=generation.error_message,
            )
            save_draft_state(draft)
            return draft, logs
        plan = OpenSCADPlan(
            parameters={},  # filled below from the last successful attempt
            scad_source=generation.repair_attempts[-1].scad_source,
            expected_bounds_m=generation.expected_bounds_m or [0, 0, 0],
        )
        draft = DraftAsset(
            asset_id=final_id, status="draft", resolution_method="parametric",
            description=description, spec=spec, plan=plan, generation=generation,
            message="Generated via OpenSCAD. Edit parameters or send a tweak, then save.",
        )
        progress("Compile succeeded -- normalizing mesh in Blender (scale/pivot/UVs)...")
        draft.plan.parameters = extract_scad_parameters(plan.scad_source)
        draft = _normalize_draft_mesh(draft)
        logs.extend(_bind_material(draft, provider, model, progress=progress))
        save_draft_state(draft)
        return draft, logs

    # imported / unclear / composite-without-parts
    is_imported = reason.startswith("Classified as 'imported'")
    if is_imported:
        # Sourcing assist (policy 2026-07): search allowlisted sources;
        # auto-download a confident CC0 match; otherwise surface candidates
        # or specific search URLs -- the user should never have to hunt.
        progress(
            "Classified as imported -- searching allowlisted sources "
            "(Poly Haven, NASA 3D Resources)..."
        )
        assist, assist_logs = assist_imported(
            spec, final_id, keywords=record.search_keywords, provider=provider, model=model,
            license_tier=license_tier,
        )
        logs.extend(assist_logs)
        progress("Reviewing candidates for relevance to the request...")
        progress(assist.note or "Source search finished.")
        if assist.auto_adopted is not None and assist.auto_adopted.success:
            entry = assist.auto_adopted.catalog_entry
            draft = DraftAsset(
                asset_id=final_id, status="existing", resolution_method="imported",
                description=description, spec=spec,
                matched_asset_id=entry.asset_id,
                glb_address=entry.address,
                bounds_m=list(entry.canonical_bounds_m),
                uv=entry.uv,
                message=assist.note,
            )
            # Imported GLBs carry their own textures/materials -- no bind.
            save_draft_state(draft)
            return draft, logs
        draft = DraftAsset(
            asset_id=final_id, status="needs_human", resolution_method="imported",
            description=description, spec=spec,
            candidates=assist.candidates,
            suggested_sources=assist.search_urls,
            message=f"{reason} {assist.note}".strip(),
        )
        save_draft_state(draft)
        return draft, logs

    draft = DraftAsset(
        asset_id=final_id, status="needs_human", resolution_method="flagged_for_review",
        description=description, spec=spec,
        message=reason or "Flagged for human review.",
    )
    save_draft_state(draft)
    return draft, logs


def _create_via_forced_route(
    spec: AssetSpec,
    final_id: str,
    route: ForcedRoute,
    provider: str | None,
    model: str | None,
    progress: ProgressFn,
    license_tier: str | None = None,
) -> tuple[DraftAsset, list[LLMCallEntry]]:
    """Build final_id via a human-chosen route, skipping resolve()'s own
    catalog-match/LLM-classification decision for this call. Each branch
    mirrors the corresponding automatic-routing branch in create_from_spec
    (or redirect_draft's post-hoc override, for parametric/composite/
    imported), just entered directly instead of via a classifier verdict.
    """
    logs: list[LLMCallEntry] = []
    description = spec.description

    if route == "existing":
        progress("Forced route: existing -- checking the catalog for a match only (no LLM fallback)...")
        catalog = load_catalog()
        candidates = find_candidates(spec, catalog)
        if not candidates or candidates[0].confidence < VARIANT_FLOOR:
            best = candidates[0].confidence if candidates else 0.0
            draft = DraftAsset(
                asset_id=final_id, status="needs_human", resolution_method="existing",
                description=description, spec=spec,
                message=(
                    f"Forced route 'existing' found no catalog match (best confidence "
                    f"{best:.2f}, need >= {VARIANT_FLOOR:.2f})."
                ),
            )
            save_draft_state(draft)
            return draft, logs
        top = candidates[0]
        method = (
            "catalog_match" if top.confidence >= config.CATALOG_MATCH_CONFIDENCE_THRESHOLD
            else "variant"
        )
        progress(
            f"Matched existing catalog asset {top.entry.asset_id!r} "
            f"({method}, confidence {top.confidence:.2f})."
        )
        draft = DraftAsset(
            asset_id=final_id, status="existing", resolution_method=method,
            description=description, spec=spec,
            matched_asset_id=top.entry.asset_id,
            glb_address=top.entry.address if top.entry.address.startswith("library/") else None,
            bounds_m=list(top.entry.canonical_bounds_m),
            uv=top.entry.uv,
            message=(
                f"Resolved to existing catalog asset {top.entry.asset_id!r} "
                f"({method}, confidence {top.confidence:.2f})."
            ),
        )
        if _existing_match_needs_material_bind(top.entry):
            logs.extend(_bind_material(draft, provider, model, progress=progress))
        save_draft_state(draft)
        return draft, logs

    if route == "parametric":
        progress(
            "Forced route: parametric -- generating an OpenSCAD script (LLM) and "
            "compiling it (up to 3 repair attempts)..."
        )
        generation, gen_logs = generate_parametric_asset(spec, final_id)
        logs.extend(gen_logs)
        if not generation.success:
            draft = DraftAsset(
                asset_id=final_id, status="needs_human", resolution_method="parametric",
                description=description, spec=spec, generation=generation,
                message=generation.error_message,
            )
            save_draft_state(draft)
            return draft, logs
        plan = OpenSCADPlan(
            parameters={},
            scad_source=generation.repair_attempts[-1].scad_source,
            expected_bounds_m=generation.expected_bounds_m or [0, 0, 0],
        )
        draft = DraftAsset(
            asset_id=final_id, status="draft", resolution_method="parametric",
            description=description, spec=spec, plan=plan, generation=generation,
            message="Generated via OpenSCAD (forced route). Edit parameters or send a tweak, then save.",
        )
        progress("Compile succeeded -- normalizing mesh in Blender (scale/pivot/UVs)...")
        draft.plan.parameters = extract_scad_parameters(plan.scad_source)
        draft = _normalize_draft_mesh(draft)
        logs.extend(_bind_material(draft, provider, model, progress=progress))
        save_draft_state(draft)
        return draft, logs

    if route == "composite":
        progress("Forced route: composite -- decomposing into primitive parts (LLM)...")
        parts, decomp_entry = decompose_into_primitives(
            spec, max_parts=12, provider=provider, model=model
        )
        logs.append(decomp_entry)
        progress(
            f"Decomposed into {len(parts)} primitive part(s): "
            f"{', '.join(p.description for p in parts)}"
        )
        catalog = load_catalog()
        record, resolve_logs = resolve(spec, catalog, forced_composite_parts=parts)
        logs.extend(resolve_logs)
        composite_id = f"{spec.object_id}_composite"
        draft, bake_logs = _materialize_and_bake_composite(
            composite_id, final_id, spec.desired_size_m, provider, model, progress
        )
        logs.extend(bake_logs)
        return draft, logs

    if route == "imported":
        progress(
            "Forced route: imported -- searching allowlisted sources "
            "(Poly Haven, NASA 3D Resources)..."
        )
        assist, assist_logs = assist_imported(
            spec, final_id, provider=provider, model=model, license_tier=license_tier,
        )
        logs.extend(assist_logs)
        progress(assist.note or "Source search finished.")
        if assist.auto_adopted is not None and assist.auto_adopted.success:
            entry = assist.auto_adopted.catalog_entry
            draft = DraftAsset(
                asset_id=final_id, status="existing", resolution_method="imported",
                description=description, spec=spec,
                matched_asset_id=entry.asset_id,
                glb_address=entry.address,
                bounds_m=list(entry.canonical_bounds_m),
                uv=entry.uv,
                message=assist.note,
            )
            save_draft_state(draft)
            return draft, logs
        draft = DraftAsset(
            asset_id=final_id, status="needs_human", resolution_method="imported",
            description=description, spec=spec,
            candidates=assist.candidates,
            suggested_sources=assist.search_urls,
            message=assist.note,
        )
        save_draft_state(draft)
        return draft, logs

    # route == "blender": no native-Blender-script generation route exists
    # yet (Blender today only normalizes meshes and bakes composites, both
    # as an implementation detail of the other routes) -- surfaced
    # explicitly as a TBD stub so a test run never mistakes this for a real
    # result, while the wiring (CLI/webapp route selector) is ready for when
    # the route is actually implemented.
    progress("Forced route: blender -- not yet implemented (TBD).")
    draft = DraftAsset(
        asset_id=final_id, status="needs_human", resolution_method="blender",
        description=description, spec=spec,
        message="Blender procedural route is TBD -- not implemented yet, this is a wiring placeholder.",
    )
    save_draft_state(draft)
    return draft, logs


# ── Manual redirect (added 2026-07) ───────────────────────────────────────
#
# A stuck draft (needs_human: imported with bad/no candidates, or
# flagged_for_review) previously had no way out except editing OpenSCAD
# parameters that don't exist yet for it. This lets a human override the
# classifier's judgment directly: force it down the parametric, composite,
# or imported path instead of re-asking the same question that already
# produced the wrong answer.

RedirectTarget = Literal["parametric", "composite", "imported"]


def redirect_draft(
    asset_id: str,
    target: RedirectTarget,
    manual_query: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    progress: ProgressFn | None = None,
    license_tier: str | None = None,
) -> tuple[DraftAsset, list[LLMCallEntry]]:
    """Rebuild an existing draft via a human-chosen path, bypassing
    classify()/resolve()'s original verdict for this asset_id.

    manual_query: only used for target='imported' -- search these words
    verbatim instead of re-deriving them, for "search again" after a bad
    first attempt.
    """
    progress = progress or _noop_progress
    draft = load_draft_state(asset_id)
    if draft is None:
        raise AssetFactoryError(f"No draft found for {asset_id!r}.")
    spec = draft.spec
    logs: list[LLMCallEntry] = []

    if target == "parametric":
        progress("Redirecting to OpenSCAD generation (human override)...")
        generation, gen_logs = generate_parametric_asset(spec, asset_id)
        logs.extend(gen_logs)
        if not generation.success:
            draft.status = "needs_human"
            draft.resolution_method = "parametric"
            draft.generation = generation
            draft.message = f"Forced OpenSCAD generation failed: {generation.error_message}"
            save_draft_state(draft)
            return draft, logs
        plan = OpenSCADPlan(
            parameters={},
            scad_source=generation.repair_attempts[-1].scad_source,
            expected_bounds_m=generation.expected_bounds_m or [0, 0, 0],
        )
        plan.parameters = extract_scad_parameters(plan.scad_source)
        draft = DraftAsset(
            asset_id=asset_id, status="draft", resolution_method="parametric",
            description=draft.description, spec=spec, plan=plan, generation=generation,
            message="Generated via OpenSCAD (human override). Edit parameters or send a tweak, then save.",
        )
        progress("Normalizing mesh in Blender (scale/pivot/UVs)...")
        draft = _normalize_draft_mesh(draft)
        logs.extend(_bind_material(draft, provider, model, progress=progress))
        save_draft_state(draft)
        return draft, logs

    if target == "composite":
        progress("Redirecting to composite decomposition (human override)...")
        parts, decomp_entry = decompose_into_primitives(spec, max_parts=12, provider=provider, model=model)
        logs.append(decomp_entry)
        progress(f"Decomposed into {len(parts)} primitive part(s): {', '.join(p.description for p in parts)}")
        catalog = load_catalog()
        record, resolve_logs = resolve(spec, catalog, forced_composite_parts=parts)
        logs.extend(resolve_logs)
        composite_id = f"{spec.object_id}_composite"
        new_draft, bake_logs = _materialize_and_bake_composite(
            composite_id, asset_id, spec.desired_size_m, provider, model, progress
        )
        logs.extend(bake_logs)
        return new_draft, logs

    # target == "imported": re-run the sourcing assist, optionally with the
    # human's own search words instead of whatever produced bad results.
    progress(f"Searching allowlisted sources again{' with your keywords' if manual_query else ''}...")
    assist, assist_logs = assist_imported(
        spec, asset_id, manual_query=manual_query, provider=provider, model=model,
        license_tier=license_tier,
    )
    logs.extend(assist_logs)
    progress(assist.note or "Source search finished.")
    if assist.auto_adopted is not None and assist.auto_adopted.success:
        entry = assist.auto_adopted.catalog_entry
        draft = DraftAsset(
            asset_id=asset_id, status="existing", resolution_method="imported",
            description=draft.description, spec=spec,
            matched_asset_id=entry.asset_id, glb_address=entry.address,
            bounds_m=list(entry.canonical_bounds_m), uv=entry.uv, message=assist.note,
        )
    else:
        draft = DraftAsset(
            asset_id=asset_id, status="needs_human", resolution_method="imported",
            description=draft.description, spec=spec,
            candidates=assist.candidates, suggested_sources=assist.search_urls,
            message=assist.note,
        )
    save_draft_state(draft)
    return draft, logs


# ── Regenerate loop ───────────────────────────────────────────────────────

_PARAM_RE = re.compile(
    r"^(?P<indent>\s*)(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*;",
    re.MULTILINE,
)


def extract_scad_parameters(scad_source: str) -> dict[str, float]:
    """Top-level `name = <number>;` assignments -- the human-tweakable knobs."""
    return {
        m.group("name"): float(m.group("value"))
        for m in _PARAM_RE.finditer(scad_source)
    }


def substitute_scad_parameters(scad_source: str, overrides: dict[str, float]) -> str:
    """Rewrite numeric top-level assignments in place; unknown names raise."""
    known = extract_scad_parameters(scad_source)
    unknown = set(overrides) - set(known)
    if unknown:
        raise AssetFactoryError(
            f"Unknown OpenSCAD parameter(s): {', '.join(sorted(unknown))}. "
            f"Editable: {', '.join(sorted(known)) or '(none)'}"
        )

    def _sub(match: re.Match) -> str:
        name = match.group("name")
        if name in overrides:
            return f"{match.group('indent')}{name} = {overrides[name]:g};"
        return match.group(0)

    return _PARAM_RE.sub(_sub, scad_source)


def regenerate(
    asset_id: str,
    parameters: dict[str, float] | None = None,
    tweak: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    progress: ProgressFn | None = None,
) -> tuple[DraftAsset, list[LLMCallEntry]]:
    """Iterate on a parametric draft.

    parameters: numeric edits substituted directly into the stored .scad --
                recompile only, no LLM call, fast.
    tweak:      free-text change request -> LLM revision of the script.
    """
    from pipeline.openscad_generator import compile_scad, measure_stl_bounds_m

    progress = progress or _noop_progress
    draft = load_draft_state(asset_id)
    if draft is None or draft.plan is None:
        raise AssetFactoryError(
            f"No parametric draft for {asset_id!r} -- only OpenSCAD-generated drafts "
            "can be regenerated."
        )
    if not parameters and not tweak:
        raise AssetFactoryError("Provide parameter edits and/or a tweak instruction.")

    logs: list[LLMCallEntry] = []
    source = draft.plan.scad_source
    if parameters:
        progress(f"Substituting parameter edits into the script: {parameters}")
        source = substitute_scad_parameters(source, parameters)
    if tweak:
        progress(f"Requesting a script revision from the LLM: {tweak!r}...")
        plan, entry = generate_openscad_plan(
            draft.spec, provider=provider, model=model,
            revision_context=(source, tweak),
        )
        logs.append(entry)
        source = plan.scad_source
        draft.plan.expected_bounds_m = plan.expected_bounds_m
        draft.plan.notes = plan.notes

    progress("Compiling with OpenSCAD...")
    asset_dir = _draft_dir(asset_id)
    success, stderr = compile_scad(source, asset_dir / "source.scad", asset_dir / "source.stl")
    if not success:
        raise AssetFactoryError(f"Revised script failed to compile: {stderr[-800:]}")

    draft.plan.scad_source = source
    draft.plan.parameters = extract_scad_parameters(source)
    actual = measure_stl_bounds_m(asset_dir / "source.stl")
    draft.generation = GenerationResult(
        asset_id=asset_id, success=True,
        scad_path=str(asset_dir / "source.scad"), stl_path=str(asset_dir / "source.stl"),
        actual_bounds_m=actual, expected_bounds_m=draft.plan.expected_bounds_m,
    )
    draft.status = "draft"
    draft.updated_at = datetime.utcnow()
    progress("Normalizing mesh in Blender (scale/pivot/UVs)...")
    draft = _normalize_draft_mesh(draft)
    save_draft_state(draft)
    return draft, logs


# ── Composite edit loop (added 2026-07, "methane sticks don't connect" /
#    "I can't edit it" incident) ──────────────────────────────────────────
#
# Mirrors regenerate()'s numeric-parameter-edit pattern for composites: a
# human who can SEE the bad geometry (a bond not reaching its atom, a part
# scaled wrong) should be able to nudge position/scale/bond_between/color
# directly and rebake, the same way an OpenSCAD parameter edit recompiles --
# no LLM call, no re-asking a question that already produced the wrong
# answer. A `color_hex` edit writes straight to that part's material file
# (also no LLM); a free-text color_hint edit only updates the hint for a
# future LLM material regeneration (see regenerate_materials_for below) and
# leaves the current material untouched until then, so it never silently
# does nothing while looking like it took effect.

_EDITABLE_COMPOSITE_FIELDS = {
    "position", "scale", "rotation", "bond_between", "bond_thickness", "label",
    "color_hint", "color_hex",
}
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def get_composite_fragment(asset_id: str) -> CompositeFragment:
    """The composite fragment behind a composite draft, for an editor UI to
    show current part transforms/colors/bonds before submitting edits."""
    draft = load_draft_state(asset_id)
    if draft is None or not draft.composite_id:
        raise AssetFactoryError(f"No composite draft for {asset_id!r}.")
    fragment = load_composite_fragment(draft.composite_id)
    if fragment is None:
        raise AssetFactoryError(f"Composite fragment {draft.composite_id!r} not found.")
    return fragment


def edit_composite(
    asset_id: str,
    part_edits: dict[str, dict],
    regenerate_materials_for: list[str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    progress: ProgressFn | None = None,
) -> tuple[DraftAsset, list[LLMCallEntry]]:
    """Edit one or more parts of a composite draft directly, then rebake.

    part_edits: {part_id: {field: value, ...}, ...} -- editable fields are
    position/scale/rotation (list[float] x3), bond_between (list[str] of 2
    labels or null to clear), bond_thickness (float), label (str),
    color_hex ('#rrggbb', writes the material file immediately, no LLM),
    and color_hint (free text, stored for a future material regen only).

    regenerate_materials_for: part_ids whose color_hint should be sent
    through the normal LLM material generator (Stage 5) right now -- the
    one part of this flow that costs an LLM call, so it is opt-in.
    """
    progress = progress or _noop_progress
    draft = load_draft_state(asset_id)
    if draft is None or not draft.composite_id:
        raise AssetFactoryError(
            f"No composite draft for {asset_id!r} -- only composite drafts can be edited this way."
        )
    fragment = load_composite_fragment(draft.composite_id)
    if fragment is None:
        raise AssetFactoryError(f"Composite fragment {draft.composite_id!r} not found.")

    parts_by_id = {p.part_id: p for p in fragment.parts}
    unknown_parts = set(part_edits) - set(parts_by_id)
    if unknown_parts:
        raise AssetFactoryError(
            f"Unknown part id(s): {', '.join(sorted(unknown_parts))}. "
            f"Known: {', '.join(parts_by_id)}"
        )

    logs: list[LLMCallEntry] = []
    if part_edits:
        progress(f"Applying edits to {len(part_edits)} part(s) (no LLM call)...")
    for part_id, edits in part_edits.items():
        part = parts_by_id[part_id]
        unknown_fields = set(edits) - _EDITABLE_COMPOSITE_FIELDS
        if unknown_fields:
            raise AssetFactoryError(
                f"Unknown field(s) for part {part_id!r}: {', '.join(sorted(unknown_fields))}"
            )
        if "position" in edits:
            part.position = [float(v) for v in edits["position"]]
        if "scale" in edits:
            part.scale = [float(v) for v in edits["scale"]]
        if "rotation" in edits:
            part.rotation = [float(v) for v in edits["rotation"]]
        if "bond_between" in edits:
            part.bond_between = list(edits["bond_between"]) if edits["bond_between"] else None
        if "bond_thickness" in edits:
            part.bond_thickness = float(edits["bond_thickness"])
        if "label" in edits:
            part.label = str(edits["label"])
        if "color_hint" in edits:
            part.color_hint = str(edits["color_hint"])
        if "color_hex" in edits:
            hexval = str(edits["color_hex"])
            if not _HEX_COLOR_RE.match(hexval):
                raise AssetFactoryError(
                    f"color_hex for {part_id!r} must look like '#rrggbb', got {hexval!r}"
                )
            mat_id = part.material_id or f"{fragment.composite_id}_{part_id}_mat"
            mat_path = config.MATERIALS_DIR / f"{mat_id}.json"
            existing = (
                MaterialDef.model_validate_json(mat_path.read_text(encoding="utf-8-sig"))
                if mat_path.is_file() else MaterialDef(material_id=mat_id, base_color=hexval)
            )
            updated = existing.model_copy(update={"material_id": mat_id, "base_color": hexval})
            mat_path.parent.mkdir(parents=True, exist_ok=True)
            mat_path.write_text(
                json.dumps(updated.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            part.material_id = mat_id

    for part_id in regenerate_materials_for or []:
        if part_id not in parts_by_id:
            raise AssetFactoryError(f"Unknown part id for material regen: {part_id!r}")
        part = parts_by_id[part_id]
        progress(f"Regenerating material for {part_id!r} from its color hint (LLM)...")
        try:
            result, _, entry = generate_material(
                part.color_hint or part.asset_spec.description,
                material_id=f"{fragment.composite_id}_{part_id}_mat",
                provider=provider, model=model, force=True,
            )
            logs.append(entry)
            part.material_id = result.material_id if result.success else part.material_id
        except (MaterialGenerationError, RuntimeError) as exc:
            progress(f"  material regen failed for {part_id!r} ({exc}); keeping the current one.")

    save_composite_fragment(fragment)

    progress("Rebaking the assembly (Blender)...")
    size = (
        draft.spec.desired_size_m
        or (max(draft.bounds_m) if draft.bounds_m else None)
        or DEFAULT_TARGET_SIZE_M
    )
    try:
        bake = bake_composite(fragment, asset_id, target_size_m=size)
    except CompositeBakeError as exc:
        draft.status = "needs_human"
        draft.message = f"Composite baking unavailable: {exc}"
        draft.updated_at = datetime.utcnow()
        save_draft_state(draft)
        return draft, logs
    if not bake.success:
        draft.status = "needs_human"
        draft.message = f"Composite baking failed: {bake.error_message}"
        draft.updated_at = datetime.utcnow()
        save_draft_state(draft)
        return draft, logs

    draft.status = "draft"
    draft.glb_address = f"library/composites/{asset_id}/model.glb"
    draft.bounds_m = bake.final_bounds_m
    draft.uv = UVInfo(status=bake.uv_status, convention="generic") if bake.uv_status != "none" else UVInfo()
    draft.message = "Edited and rebaked -- each part has its own color."
    if bake.error_message:
        draft.message += f" ({bake.error_message})"
    draft.updated_at = datetime.utcnow()
    save_draft_state(draft)
    return draft, logs


# ── Visual review + auto-repair (added 2026-07, "methane sticks don't
#    connect" incident) ─────────────────────────────────────────────────
#
# The pipeline had no way to notice its own bad output -- a broken bake
# (disconnected bonds, ungraded colors) looked exactly as "successful" as a
# correct one until a human happened to look. This renders the draft's
# current GLB, asks a vision LLM the same question a human reviewer would
# ("does this actually match the request?"), and -- opt-in, since it's an
# LLM call plus a render -- feeds a real mismatch's specific complaint back
# into a repair pass instead of a human having to describe the bug.

def review_and_repair(
    asset_id: str,
    auto_repair: bool = True,
    provider: str | None = None,
    model: str | None = None,
    progress: ProgressFn | None = None,
) -> tuple[DraftAsset, AssetVisualReview, list[LLMCallEntry]]:
    """Render the draft, review it against the original description, and
    (when auto_repair and the review found a mismatch) feed the reviewer's
    complaint into a repair pass: regenerate()'s tweak flow for a parametric
    draft, or a composite part-list revision for a composite. Drafts with no
    generative repair path (imported/existing) are left as-is with the
    review attached -- never a silent no-op that looks like it did something.
    """
    progress = progress or _noop_progress
    draft = load_draft_state(asset_id)
    if draft is None:
        raise AssetFactoryError(f"No draft found for {asset_id!r}.")
    if not draft.glb_address:
        raise AssetFactoryError(f"Draft {asset_id!r} has no renderable GLB yet.")

    progress("Rendering the current result for visual review...")
    image_path = _draft_dir(asset_id) / "review_render.png"
    try:
        render_glb_to_png(draft.glb_address, image_path)
    except PreviewError as exc:
        raise AssetFactoryError(f"Could not render {asset_id!r} for review: {exc}") from exc

    progress("Asking a vision-capable LLM whether the render matches the request...")
    try:
        review, entry = review_asset_render(
            draft.description, image_path, provider=provider, model=model
        )
    except VisualReviewError as exc:
        raise AssetFactoryError(f"Visual review unavailable: {exc}") from exc
    logs = [entry]
    draft.visual_review = review

    if review.matches:
        progress("Visual review: matches the request.")
        draft.message = f"{draft.message or ''} Visual review: matches the request.".strip()
        draft.updated_at = datetime.utcnow()
        save_draft_state(draft)
        return draft, review, logs

    progress(f"Visual review found a mismatch: {review.reasoning}")
    draft.message = f"Visual review: MISMATCH -- {review.reasoning} Suggested fix: {review.suggested_fix}"
    if not auto_repair:
        draft.updated_at = datetime.utcnow()
        save_draft_state(draft)
        return draft, review, logs

    feedback = f"{review.reasoning} Fix: {review.suggested_fix}"

    if draft.resolution_method == "parametric" and draft.plan is not None:
        progress("Auto-repairing via an OpenSCAD revision (LLM)...")
        repaired, repair_logs = regenerate(
            asset_id, tweak=feedback, provider=provider, model=model, progress=progress
        )
        logs.extend(repair_logs)
        repaired.visual_review = review
        save_draft_state(repaired)
        return repaired, review, logs

    if draft.resolution_method == "composite" and draft.composite_id:
        progress("Auto-repairing the composite part list (LLM)...")
        fragment = load_composite_fragment(draft.composite_id)
        if fragment is None:
            raise AssetFactoryError(f"Composite fragment {draft.composite_id!r} not found.")
        current_plans = fragment_to_part_plans(fragment)
        revised_plans, revise_entry = revise_composite_parts(
            draft.spec, current_plans, feedback, provider=provider, model=model,
        )
        logs.append(revise_entry)
        catalog = load_catalog()
        record, resolve_logs = resolve(draft.spec, catalog, forced_composite_parts=revised_plans)
        logs.extend(resolve_logs)
        # resolve() derives the composite_id from spec.object_id itself
        # (see resolver.py) -- match that exactly rather than assuming it
        # equals asset_id, which can differ when a custom asset_id was given.
        composite_id = f"{draft.spec.object_id}_composite"
        repaired, bake_logs = _materialize_and_bake_composite(
            composite_id, asset_id, draft.spec.desired_size_m, provider, model, progress
        )
        logs.extend(bake_logs)
        repaired.visual_review = review
        save_draft_state(repaired)
        return repaired, review, logs

    # imported/existing/unclear: no generative repair path -- surface the
    # review so a human can act on it manually instead of silently no-oping.
    draft.message += " (auto-repair isn't available for this resolution method; edit manually.)"
    draft.updated_at = datetime.utcnow()
    save_draft_state(draft)
    return draft, review, logs


# ── Save / approve ────────────────────────────────────────────────────────

def save_draft(
    asset_id: str,
    display_name: str | None = None,
    tags: list[str] | None = None,
    render_preview: bool = True,
) -> AssetCatalogEntry:
    """Promote a draft into catalog.json (upsert) -- the human's save IS the
    review acknowledgment, recorded in provenance.

    For a BINDING draft (status 'existing': the moon = sphere + moon
    texture; an adopted import), there is no new geometry to catalog --
    saving records the human approval (who/when) on the draft itself and
    returns the underlying catalog entry.
    """
    draft = load_draft_state(asset_id)
    if draft is None:
        raise AssetFactoryError(f"No draft found for {asset_id!r}.")

    if draft.status in ("existing", "saved") and draft.matched_asset_id:
        entry = next(
            (e for e in load_catalog() if e.asset_id == draft.matched_asset_id), None
        )
        if entry is None:
            raise AssetFactoryError(
                f"Draft {asset_id!r} resolves to {draft.matched_asset_id!r}, which is "
                "no longer in the catalog."
            )
        draft.approved_by = getpass.getuser()
        draft.approved_at = datetime.utcnow()
        draft.status = "saved"
        draft.message = (
            f"Binding approved by {draft.approved_by} on {date.today().isoformat()} "
            f"(uses {entry.asset_id!r}"
            + (f" with material {draft.material.material_id!r}" if draft.material else "")
            + ")."
        )
        save_draft_state(draft)
        return entry

    if not draft.glb_address or not draft.bounds_m:
        raise AssetFactoryError(
            f"Draft {asset_id!r} has no normalized GLB yet -- generate/regenerate first."
        )

    is_composite = draft.resolution_method == "composite"
    entry = AssetCatalogEntry(
        asset_id=asset_id,
        display_name=display_name or draft.spec.semantic_type or asset_id,
        asset_class="composite" if is_composite else "parametric",
        address=draft.glb_address,
        canonical_bounds_m=draft.bounds_m,
        pivot=draft.normalization.pivot if draft.normalization else "base_center",
        uv=draft.uv,
        material_slots=["surface"] if draft.material else [],
        tags=tags if tags is not None else [t for t in [draft.spec.semantic_type, draft.spec.kind] if t],
        provenance=ProvenanceInfo(
            source_type="generated",
            license="internal",
            license_status="approved",
            modified=True,
            modifications=[
                "assembled from primitives and baked into one mesh (composite_baker.py)"
                if is_composite else "generated via OpenSCAD (Stage 3) + normalized (Stage 4)",
                f"reviewed and saved by {getpass.getuser()} on {date.today().isoformat()}",
            ],
            date_imported_or_generated=date.today().isoformat(),
        ),
        # implementation plan section 4: parametric default review level;
        # composites inherit the same level (still LLM-decomposed geometry
        # a human should eyeball, same trust tier as a generated CSG part).
        review_level=2,
    )

    entries = load_catalog()
    entries = [e for e in entries if e.asset_id != asset_id] + [entry]
    save_catalog(entries)

    draft.status = "saved"
    draft.updated_at = datetime.utcnow()
    save_draft_state(draft)

    if render_preview:
        try:
            render_previews([entry], force=True)
        except Exception:
            pass  # preview is a nicety at save time; `cli.py preview` can redo it
    return entry


def adopt_draft_candidate(
    asset_id: str, source_id: str, target_size_m: float
) -> DraftAsset:
    """One-click approval of a sourcing candidate (webapp POST /api/adopt):
    fetch the chosen file, run the normal intake gate, resolve the draft."""
    draft = load_draft_state(asset_id)
    if draft is None:
        raise AssetFactoryError(f"No draft found for {asset_id!r}.")
    candidate = next((c for c in draft.candidates if c.source_id == source_id), None)
    if candidate is None:
        raise AssetFactoryError(
            f"{source_id!r} is not one of this draft's candidates "
            f"({', '.join(c.source_id for c in draft.candidates) or 'none'})."
        )
    result = adopt_candidate(candidate, asset_id, target_size_m=target_size_m)
    if not result.success:
        raise AssetFactoryError(result.error_message or "intake failed")

    entry = result.catalog_entry
    draft.status = "existing"
    draft.matched_asset_id = entry.asset_id
    draft.glb_address = entry.address
    draft.bounds_m = list(entry.canonical_bounds_m)
    draft.uv = entry.uv
    draft.message = (
        f"Adopted {candidate.source_id!r} from {candidate.source} "
        f"({candidate.license}) and cataloged as {entry.asset_id!r}."
    )
    draft.updated_at = datetime.utcnow()
    save_draft_state(draft)
    return draft


# ── Discard draft (added 2026-07, "no way to clear/delete" incident) ─────
#
# A stuck, wrong, or stale draft previously had no way out at all -- once
# created, `library/generated/<id>/draft.json` sat there forever, showing
# up in the drafts grid indefinitely. Deliberately scoped to drafts only
# (never a saved/approved catalog asset): a draft is this author's own
# in-progress work, safe to throw away; a saved entry may already be
# referenced by a scene, so removing one is a separate, more careful
# operation this function does not perform.

def discard_draft(asset_id: str) -> None:
    """Delete an in-progress draft and everything it produced (OpenSCAD
    source/STL/GLB, or a composite's fragment + baked GLB + per-part
    materials). Raises if the draft was already saved/approved."""
    draft = load_draft_state(asset_id)
    if draft is None:
        raise AssetFactoryError(f"No draft found for {asset_id!r}.")
    if draft.status == "saved":
        raise AssetFactoryError(
            f"{asset_id!r} has already been saved/approved -- discard only removes "
            "in-progress drafts, not saved catalog assets."
        )

    draft_dir = _draft_dir(asset_id)
    if draft_dir.is_dir():
        shutil.rmtree(draft_dir)

    if draft.composite_id:
        fragment_path = config.LIBRARY_DIR / "composites" / f"{draft.composite_id}.json"
        if fragment_path.is_file():
            fragment_path.unlink()
        baked_dir = config.LIBRARY_DIR / "composites" / asset_id
        if baked_dir.is_dir():
            shutil.rmtree(baked_dir)
        for mat_path in config.MATERIALS_DIR.glob(f"{draft.composite_id}_*_mat.json"):
            mat_path.unlink()


def approve_asset(asset_id: str) -> AssetCatalogEntry:
    """Record a human approval on an existing catalog entry (who/when)."""
    entries = load_catalog()
    for i, entry in enumerate(entries):
        if entry.asset_id == asset_id:
            stamp = f"approved by {getpass.getuser()} on {date.today().isoformat()}"
            prov = entry.provenance.model_copy(
                update={"modifications": [*entry.provenance.modifications, stamp]}
            )
            entries[i] = entry.model_copy(update={"provenance": prov})
            save_catalog(entries)
            return entries[i]
    raise AssetFactoryError(f"asset_id {asset_id!r} not found in the catalog.")


def update_catalog_metadata(
    asset_id: str,
    display_name: str | None = None,
    tags: list[str] | None = None,
) -> AssetCatalogEntry:
    """Rename/re-tag a catalog entry (added 2026-07, "easier to search"
    request) -- edits only display_name/tags, never geometry/provenance/
    scale. `None` means "leave this field alone" (so editing just the name
    doesn't blank the tags); pass an empty list to clear tags explicitly.
    """
    updates: dict = {}
    if display_name is not None:
        stripped = display_name.strip()
        if not stripped:
            raise AssetFactoryError("display_name cannot be blank.")
        updates["display_name"] = stripped
    if tags is not None:
        updates["tags"] = [t.strip() for t in tags if t.strip()]
    if not updates:
        raise AssetFactoryError("Nothing to update -- provide display_name and/or tags.")

    entries = load_catalog()
    for i, entry in enumerate(entries):
        if entry.asset_id == asset_id:
            entries[i] = entry.model_copy(update=updates)
            save_catalog(entries)
            return entries[i]
    raise AssetFactoryError(f"asset_id {asset_id!r} not found in the catalog.")
