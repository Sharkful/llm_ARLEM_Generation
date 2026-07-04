from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

ResolutionMethod = Literal[
    "catalog_match",
    "variant",
    "composite",
    "parametric",
    "imported",
    "flagged_for_review",
]


class AssetSpec(BaseModel):
    """Authoring-side, unresolved asset request.

    Matches offline_ar_asset_pipeline_requirements.md sections 5.1 / 6.1.
    Never loaded directly by the runtime AR application.
    """

    object_id: str
    description: str
    kind: Optional[str] = None
    semantic_type: Optional[str] = None
    desired_size_m: Optional[float] = None
    visual_style: Optional[str] = None
    position: Optional[list[float]] = None


class ResolvedAssetRef(BaseModel):
    asset_id: str
    resolution_method: ResolutionMethod
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class ResolutionRecord(BaseModel):
    """Matches offline_ar_asset_pipeline_requirements.md section 5.2.

    Produced by the resolver (pipeline/resolver.py, Stage 2) for a given
    AssetSpec. requires_author_review gates the Stage 9 batch checkpoint
    per section 18 -- the LLM is never the sole approval authority.
    """

    object_id: str
    requested_asset_spec: AssetSpec
    resolved_asset: Optional[ResolvedAssetRef] = None
    requires_author_review: bool = False
    review_reason: Optional[str] = None
    # LLM-authored identity words for external sourcing, populated when the
    # geometry classifier chose 'imported' (see GeometryClassification).
    # Carrying these forward avoids re-deriving search terms mechanically
    # from the raw description, which produces filler-word noise.
    search_keywords: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
