"""Stage 2 orchestrator: AssetSpec -> ResolutionRecord.

Wires classify() (catalog match / variant / geometry classification) and,
for the composite case, composite_builder (decomposition into sub-parts,
each recursively resolved) into a single ResolutionRecord per the schema
in offline_ar_asset_pipeline_requirements.md section 5.2.

No new geometry is generated here -- "parametric" and "imported" results
are routed to Stage 3 (OpenSCAD) and Stage 6 (external intake)
respectively, which don't exist yet. Until then, resolve() reports those
classifications with requires_author_review=True so nothing silently
claims to be resolved when it isn't (per requirements doc section 17: no
silent fallback behavior).
"""
from __future__ import annotations

from models.catalog_models import AssetCatalogEntry
from models.log_models import LLMCallEntry
from models.spec_models import AssetSpec, ResolutionRecord
from pipeline.classifier import ClassifierFn, classify
from pipeline.composite_builder import (
    CompositeFragment,
    CompositePart,
    build_part_specs,
    save_composite_fragment,
)

# Stages 3 (OpenSCAD) and 6 (external intake) don't exist yet -- these
# classes are always flagged for review until those stages are built.
_UNIMPLEMENTED_GEOMETRY_CLASSES = {"parametric", "imported", "unclear"}


def resolve(
    spec: AssetSpec,
    catalog: list[AssetCatalogEntry],
    classify_geometry_fn: ClassifierFn | None = None,
    _depth: int = 0,
) -> tuple[ResolutionRecord, list[LLMCallEntry]]:
    all_log_entries: list[LLMCallEntry] = []

    resolved_ref, geometry, log_entries = classify(
        spec, catalog, classify_geometry_fn=classify_geometry_fn
    )
    all_log_entries.extend(log_entries)

    if resolved_ref is not None:
        # catalog_match or variant -- fully resolved, no review needed.
        record = ResolutionRecord(
            object_id=spec.object_id,
            requested_asset_spec=spec,
            resolved_asset=resolved_ref,
            requires_author_review=False,
        )
        return record, all_log_entries

    assert geometry is not None  # classify() always returns one or the other

    if geometry.geometry_class == "composite" and geometry.composite_parts:
        composite_id = f"{spec.object_id}_composite"
        part_specs = build_part_specs(spec.object_id, geometry.composite_parts)

        parts: list[CompositePart] = []
        any_part_needs_review = False
        for part_spec in part_specs:
            # Recursive resolution -- guard against pathological infinite
            # recursion (an LLM classifying a sub-part as composite again)
            # rather than trusting the LLM never loops.
            if _depth >= 3:
                part_record = ResolutionRecord(
                    object_id=part_spec.object_id,
                    requested_asset_spec=part_spec,
                    requires_author_review=True,
                    review_reason="Composite recursion depth limit reached.",
                )
            else:
                part_record, part_log_entries = resolve(
                    part_spec, catalog, classify_geometry_fn=classify_geometry_fn, _depth=_depth + 1
                )
                all_log_entries.extend(part_log_entries)

            resolved_part_id = (
                part_record.resolved_asset.asset_id if part_record.resolved_asset else None
            )
            if resolved_part_id is None:
                any_part_needs_review = True

            parts.append(
                CompositePart(
                    part_id=part_spec.object_id,
                    asset_spec=part_spec,
                    resolved_asset_id=resolved_part_id,
                )
            )

        fragment = CompositeFragment(
            composite_id=composite_id,
            display_name=spec.semantic_type or spec.kind or spec.object_id,
            source_description=spec.description,
            parts=parts,
        )
        save_composite_fragment(fragment)

        if any_part_needs_review:
            record = ResolutionRecord(
                object_id=spec.object_id,
                requested_asset_spec=spec,
                requires_author_review=True,
                review_reason=(
                    f"Composite {composite_id!r} built, but one or more sub-parts could "
                    "not be fully resolved (see library/composites/ fragment for detail)."
                ),
            )
        else:
            record = ResolutionRecord(
                object_id=spec.object_id,
                requested_asset_spec=spec,
                resolved_asset=None,  # composites resolve to a fragment, not a single asset_id
                requires_author_review=False,
                review_reason=f"Resolved as composite: {composite_id}",
            )
        return record, all_log_entries

    # parametric / imported / unclear / (composite with no parts listed):
    # flag for review since the stage that would actually build/import
    # these doesn't exist yet.
    reason = geometry.reasoning
    if geometry.geometry_class == "composite" and not geometry.composite_parts:
        reason = f"Classified as composite but no sub-parts were listed. {reason}"
    elif geometry.geometry_class in _UNIMPLEMENTED_GEOMETRY_CLASSES:
        reason = f"Classified as {geometry.geometry_class!r} (not yet implemented). {reason}"

    record = ResolutionRecord(
        object_id=spec.object_id,
        requested_asset_spec=spec,
        requires_author_review=True,
        review_reason=reason,
    )
    return record, all_log_entries
