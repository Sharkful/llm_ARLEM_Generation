from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from models.catalog_models import MaterialDef


class OpenSCADPlan(BaseModel):
    """Structured instructor response for Stage 3 parametric generation."""

    parameters: dict[str, float] = Field(default_factory=dict)
    scad_source: str
    expected_bounds_m: list[float] = Field(min_length=3, max_length=3)
    notes: str = ""


class CompileAttempt(BaseModel):
    attempt_number: int
    scad_source: str
    success: bool
    stderr: str = ""


class GenerationResult(BaseModel):
    asset_id: str
    success: bool
    scad_path: str | None = None
    stl_path: str | None = None
    actual_bounds_m: list[float] | None = None
    expected_bounds_m: list[float] | None = None
    bounds_diverge: bool = False
    repair_attempts: list[CompileAttempt] = Field(default_factory=list)
    error_message: str | None = None


TextureKind = Literal["noise", "craters", "stripes", "grid", "gradient", "rust"]


class ProceduralTextureSpec(BaseModel):
    """LLM-proposed procedural texture, synthesized locally with numpy+Pillow.

    Only simple pattern families are supported (req. doc section 9 / 20.2
    scope) -- photorealistic or AI-image-generated textures are deliberately
    not part of this model.
    """

    kind: TextureKind
    base_color: str = Field(description="Hex color like #8a8a8a for the dominant surface")
    accent_color: str = Field(description="Hex color for the pattern features (crater shadows, stripes, rust blotches)")
    scale: float = Field(
        default=8.0, gt=0.0, le=64.0,
        description="Feature frequency across the texture: ~4 = large sparse features, ~32 = fine dense features",
    )
    notes: str = ""


class MaterialPlan(BaseModel):
    """Structured instructor response for Stage 5 material generation.

    `material.texture` must be left null by the LLM -- the generator fills
    it in with the real relative path if `procedural_texture` is synthesized.
    """

    material: MaterialDef
    procedural_texture: ProceduralTextureSpec | None = None
    reasoning: str = ""


class MaterialGenerationResult(BaseModel):
    material_id: str
    success: bool
    material_path: str | None = None
    texture_path: str | None = None
    error_message: str | None = None
