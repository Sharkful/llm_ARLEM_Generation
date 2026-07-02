from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]

AssetClass = Literal["primitive", "variant", "composite", "parametric", "imported"]
ReviewLevel = Literal[0, 1, 2, 3, 4]
Axis = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
LicenseStatus = Literal["approved", "pending", "rejected", "unknown"]
SourceType = Literal["core", "generated", "composite", "external_approved", "internal"]


class AssetCapabilities(BaseModel):
    scalable: bool = True
    supports_material_override: bool = True
    has_collider: bool = False


class ColliderInfo(BaseModel):
    shape: Literal["none", "box", "sphere", "capsule", "mesh"] = "none"
    is_trigger: bool = False


class ProvenanceInfo(BaseModel):
    """Matches offline_ar_asset_pipeline_requirements.md section 11 exactly."""

    source_type: SourceType
    source_site: str | None = None
    source_url: str | None = None
    original_author: str | None = None
    license: str | None = None
    license_status: LicenseStatus = "unknown"
    attribution_required: bool = False
    modified: bool = False
    modifications: list[str] = Field(default_factory=list)
    date_imported_or_generated: str | None = None


class AssetCatalogEntry(BaseModel):
    """Matches offline_ar_asset_pipeline_requirements.md section 5.3 exactly."""

    asset_id: str
    display_name: str
    asset_class: AssetClass
    address: str
    material_slots: list[str] = Field(default_factory=list)
    canonical_bounds_m: Vec3
    pivot: Literal["center", "base_center", "custom"] = "center"
    default_forward_axis: Axis = "+Z"
    default_up_axis: Axis = "+Y"
    tags: list[str] = Field(default_factory=list)
    capabilities: AssetCapabilities = Field(default_factory=AssetCapabilities)
    collider: ColliderInfo = Field(default_factory=ColliderInfo)
    provenance: ProvenanceInfo
    review_level: ReviewLevel = 0


class IntakeSource(BaseModel):
    """Hand-filled source.json stub accompanying a manually downloaded file
    in intake/<asset_id>/ (Stage 6a). The human sourcing the file records
    where it came from and under what license; the pipeline refuses intake
    if the license isn't in the allowlist or attribution info is missing.
    """

    display_name: str
    license: str
    source_site: str | None = None
    source_url: str | None = None
    original_author: str | None = None
    target_size_m: float = Field(
        gt=0.0,
        description="Largest dimension in meters after normalization -- required; nothing enters the catalog with unknown scale",
    )
    pivot: Literal["center", "base_center"] = "center"
    tags: list[str] = Field(default_factory=list)
    semantic_type: str | None = None
    pedagogically_critical: bool = False
    notes: str = ""


class MaterialDef(BaseModel):
    """Matches offline_ar_asset_pipeline_requirements.md section 9 exactly."""

    material_id: str
    shader_family: str = "URP/Lit"
    base_color: str
    alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    metallic: float = Field(default=0.0, ge=0.0, le=1.0)
    smoothness: float = Field(default=0.5, ge=0.0, le=1.0)
    transparent: bool = False
    emissive: bool = False
    texture: str | None = None
