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
from pipeline.catalog_writer import load_catalog, save_catalog
from pipeline.composite_builder import decompose_into_primitives
from pipeline.source_assist import SourceCandidate, adopt_candidate, assist_imported
from pipeline.material_generator import MaterialGenerationError, generate_material
from pipeline.mesh_processor import MeshNormalizationResult, normalize_mesh, save_mesh_meta
from pipeline.openscad_generator import generate_openscad_plan, generate_parametric_asset
from pipeline.preview_renderer import render_previews
from pipeline.resolver import resolve
from pipeline.spec_parser import parse_description

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


def create_from_description(
    description: str,
    asset_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    progress: ProgressFn | None = None,
) -> tuple[DraftAsset, list[LLMCallEntry]]:
    """One-shot flow: free text -> routed, generated (if parametric), normalized draft."""
    progress = progress or _noop_progress
    progress("Parsing description into a structured asset spec (LLM)...")
    spec, entry = parse_description(description, provider=provider, model=model)
    if asset_id:
        spec = spec.model_copy(update={"object_id": asset_id})
    draft, logs = create_from_spec(
        spec, asset_id=asset_id, provider=provider, model=model, progress=progress
    )
    return draft, [entry, *logs]


def create_from_spec(
    spec: AssetSpec,
    asset_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    progress: ProgressFn | None = None,
) -> tuple[DraftAsset, list[LLMCallEntry]]:
    """Same flow starting from an already-parsed AssetSpec (Stage 9 batch
    input, or a worklist of pre-authored specs -- skips the Stage 1 call)."""
    progress = progress or _noop_progress
    logs: list[LLMCallEntry] = []
    description = spec.description
    final_id = asset_id or _slug(spec.object_id)

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
        logs.extend(_bind_material(draft, provider, model, progress=progress))
        save_draft_state(draft)
        return draft, logs

    # Route on the resolver's exact message prefixes (resolver.py owns these
    # strings; test_asset_factory pins them so a resolver wording change
    # fails loudly here instead of silently misrouting).
    reason = record.review_reason or ""
    if reason.startswith("Resolved as composite:") and not record.requires_author_review:
        draft = DraftAsset(
            asset_id=final_id, status="composite", resolution_method="composite",
            description=description, spec=spec,
            composite_id=f"{spec.object_id}_composite", message=reason,
        )
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
            spec, final_id, keywords=record.search_keywords, provider=provider, model=model
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
        progress(f"Decomposed into {len(parts)} primitive part(s): {', '.join(parts)}")
        catalog = load_catalog()
        record, resolve_logs = resolve(spec, catalog, forced_composite_parts=parts)
        logs.extend(resolve_logs)
        draft = DraftAsset(
            asset_id=asset_id, status="composite", resolution_method="composite",
            description=draft.description, spec=spec,
            composite_id=f"{spec.object_id}_composite",
            message=record.review_reason or "Built as a composite (human override).",
        )
        save_draft_state(draft)
        return draft, logs

    # target == "imported": re-run the sourcing assist, optionally with the
    # human's own search words instead of whatever produced bad results.
    progress(f"Searching allowlisted sources again{' with your keywords' if manual_query else ''}...")
    assist, assist_logs = assist_imported(
        spec, asset_id, manual_query=manual_query, provider=provider, model=model
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

    entry = AssetCatalogEntry(
        asset_id=asset_id,
        display_name=display_name or draft.spec.semantic_type or asset_id,
        asset_class="parametric",
        address=draft.glb_address,
        canonical_bounds_m=draft.bounds_m,
        pivot=draft.normalization.pivot if draft.normalization else "center",
        uv=draft.uv,
        material_slots=["surface"] if draft.material else [],
        tags=tags if tags is not None else [t for t in [draft.spec.semantic_type, draft.spec.kind] if t],
        provenance=ProvenanceInfo(
            source_type="generated",
            license="internal",
            license_status="approved",
            modified=True,
            modifications=[
                "generated via OpenSCAD (Stage 3) + normalized (Stage 4)",
                f"reviewed and saved by {getpass.getuser()} on {date.today().isoformat()}",
            ],
            date_imported_or_generated=date.today().isoformat(),
        ),
        review_level=2,  # implementation plan section 4: parametric default
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
