"""Stage 2.5: turn a composite classification into a composite fragment.

A composite is NOT new geometry -- it's a list of existing (or recursively
resolved) asset_ids with relative transforms, stored as a JSON fragment in
library/composites/<id>.json. Each sub-part goes back through classify()
(recursively) so a composite of composites, or a composite containing a
part that itself needs generation, is handled the same way a top-level
spec would be.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

import config
from models.spec_models import AssetSpec


class CompositePart(BaseModel):
    part_id: str
    asset_spec: AssetSpec
    resolved_asset_id: str | None = None
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])


class CompositeFragment(BaseModel):
    composite_id: str
    display_name: str
    source_description: str
    parts: list[CompositePart]


def build_part_specs(parent_object_id: str, composite_parts: list[str]) -> list[AssetSpec]:
    """Turn the classifier's free-text part descriptions into AssetSpecs.

    This does not call an LLM -- part descriptions are already short,
    LLM-authored phrases (from GeometryClassification.composite_parts);
    wrapping each in a minimal AssetSpec is enough for the resolver to
    recursively classify/resolve it. semantic_type is left unset since a
    short phrase like "1 large red sphere" doesn't cleanly map to one
    without further guessing.
    """
    specs = []
    for i, part_description in enumerate(composite_parts):
        specs.append(
            AssetSpec(
                object_id=f"{parent_object_id}_part{i}",
                description=part_description,
            )
        )
    return specs


def composite_path(composite_id: str) -> Path:
    return config.LIBRARY_DIR / "composites" / f"{composite_id}.json"


def save_composite_fragment(fragment: CompositeFragment) -> Path:
    path = composite_path(fragment.composite_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fragment.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def load_composite_fragment(composite_id: str) -> CompositeFragment | None:
    path = composite_path(composite_id)
    if not path.exists():
        return None
    return CompositeFragment.model_validate(json.loads(path.read_text(encoding="utf-8")))
