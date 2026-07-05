"""Stage 9: resolve a whole scene's asset list end-to-end.

Input: a JSON file containing a list where each item is either a free-text
description string or an AssetSpec-shaped object. Each item runs the full
pipeline via asset_factory (spec -> classify/resolve -> generate/normalize
-> material/texture ladder), then the batch finishes with validation,
preview rendering, and (optionally) a sync into a consumer assets folder.

Human-in-the-loop per req. doc section 18: an item the pipeline cannot
finish autonomously (imported/unclear classifications, diverging parametric
bounds, failed generation) HALTS for that asset and is reported distinctly
from hard failures -- the batch continues with the other items, and the
build report (req. doc section 17 shape) records every outcome, fallback,
and review flag. Deployable bundles should contain zero unresolved errors.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from models.log_models import LLMCallEntry
from models.spec_models import AssetSpec
from pipeline.asset_factory import (
    AssetFactoryError,
    DraftAsset,
    create_from_description,
    create_from_spec,
    save_draft,
)
from pipeline.catalog_writer import load_catalog
from pipeline.preview_renderer import PreviewError, render_previews, write_contact_sheet
from pipeline.sync import SyncResult, sync_assets
from pipeline.validator import validate_library

ItemStatus = Literal[
    "resolved_existing",  # catalog match / variant -- nothing built
    "saved_generated",    # generated (OpenSCAD or a baked composite), auto-cataloged at review level 2
    "needs_review",       # halted for a human: imported/unclear/diverging bounds
    "failed",             # hard error (generation/normalization crash)
]


class SceneItemReport(BaseModel):
    object_id: str
    description: str
    status: ItemStatus
    resolution_method: str = ""
    asset_id: Optional[str] = None  # catalog id this object resolves to
    review_level: Optional[int] = None
    requires_author_review: bool = False
    texture_id: Optional[str] = None
    texture_pending: bool = False
    texture_query: Optional[str] = None
    message: Optional[str] = None
    llm_calls: int = 0


class SceneBuildReport(BaseModel):
    scene_file: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    items: list[SceneItemReport] = Field(default_factory=list)
    resolved: int = 0
    needs_review: int = 0
    failed: int = 0
    validation_errors: int = 0
    validation_warnings: int = 0
    previews_rendered: int = 0
    sync: Optional[SyncResult] = None
    status: Literal["ok", "needs_review", "failed"] = "ok"


def load_scene_specs(path: Path) -> list[str | AssetSpec]:
    """Each list item: a raw description string, or an AssetSpec-shaped dict."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("Scene specs file must contain a JSON list.")
    items: list[str | AssetSpec] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            items.append(item)
        elif isinstance(item, dict):
            try:
                items.append(AssetSpec.model_validate(item))
            except ValidationError as exc:
                raise ValueError(f"Scene item {i} is not a valid AssetSpec: {exc}") from exc
        else:
            raise ValueError(f"Scene item {i} must be a string or an object, got {type(item).__name__}")
    return items


def _item_from_draft(
    draft: DraftAsset, description: str, logs: list[LLMCallEntry]
) -> SceneItemReport:
    base = dict(
        object_id=draft.spec.object_id,
        description=description,
        resolution_method=draft.resolution_method,
        texture_id=draft.material.texture_id if draft.material else None,
        texture_pending=draft.texture_pending,
        texture_query=draft.texture_query,
        message=draft.message,
        llm_calls=len(logs),
        # Req. doc section 17: a pending texture is a recorded fallback, so
        # it flags author review even when the geometry resolved cleanly.
        requires_author_review=draft.texture_pending,
    )
    if draft.status == "existing":
        return SceneItemReport(status="resolved_existing", asset_id=draft.matched_asset_id, **base)
    if draft.status == "draft":
        diverges = bool(draft.generation and draft.generation.bounds_diverge)
        if diverges or not draft.glb_address:
            base["requires_author_review"] = True
            base["message"] = (
                (base["message"] or "")
                + " Bounds diverge from the LLM's expectation -- inspect in `cli.py review` before saving."
            ).strip()
            return SceneItemReport(status="needs_review", asset_id=draft.asset_id, **base)
        entry = save_draft(draft.asset_id)
        return SceneItemReport(
            status="saved_generated", asset_id=entry.asset_id,
            review_level=entry.review_level, **base,
        )
    # needs_human
    base["requires_author_review"] = True
    return SceneItemReport(status="needs_review", asset_id=draft.asset_id, **base)


def resolve_scene(
    specs_path: Path,
    sync_target: Path | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> SceneBuildReport:
    report = SceneBuildReport(scene_file=str(specs_path))
    items = load_scene_specs(specs_path)

    for item in items:
        description = item if isinstance(item, str) else item.description
        object_id = None if isinstance(item, str) else item.object_id
        try:
            if isinstance(item, str):
                draft, logs = create_from_description(item, provider=provider, model=model)
            else:
                draft, logs = create_from_spec(item, provider=provider, model=model)
            report.items.append(_item_from_draft(draft, description, logs))
        except Exception as exc:  # hard failure: recorded, batch continues
            report.items.append(
                SceneItemReport(
                    object_id=object_id or "?", description=description,
                    status="failed", requires_author_review=True, message=str(exc),
                )
            )

    report.resolved = sum(
        1 for i in report.items
        if i.status in ("resolved_existing", "saved_generated")
    )
    report.needs_review = sum(1 for i in report.items if i.status == "needs_review")
    report.failed = sum(1 for i in report.items if i.status == "failed")

    validation = validate_library()
    report.validation_errors = validation.errors
    report.validation_warnings = validation.warnings

    # Previews for everything the scene touches (skip ones already rendered).
    catalog = {e.asset_id: e for e in load_catalog()}
    touched = [catalog[i.asset_id] for i in report.items if i.asset_id in catalog]
    try:
        written, _ = render_previews(touched)
        report.previews_rendered = len(written)
        if written:
            write_contact_sheet(list(catalog.values()))
    except PreviewError:
        pass  # previews are a nicety; the report records 0 rendered

    if sync_target is not None:
        report.sync = sync_assets(
            [i.asset_id for i in report.items if i.asset_id in catalog],
            sync_target,
            # Scene objects name their material after the object (moon_mat),
            # not the catalog asset they resolved to (sphere_basic).
            # A baked composite's colors are already inside its GLB (no
            # separate <id>_mat.json exists for it); harmless to include --
            # sync_assets() just finds nothing to copy for that name.
            material_ids=[
                f"{i.object_id}_mat" for i in report.items
                if i.status in ("resolved_existing", "saved_generated")
            ],
        )

    if report.failed or report.validation_errors:
        report.status = "failed"
    elif report.needs_review or any(i.requires_author_review for i in report.items):
        report.status = "needs_review"
    else:
        report.status = "ok"
    return report


def write_build_report(report: SceneBuildReport, specs_path: Path) -> Path:
    out = specs_path.with_name(specs_path.stem + ".build_report.json")
    out.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    return out
