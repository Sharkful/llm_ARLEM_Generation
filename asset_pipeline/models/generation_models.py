from __future__ import annotations

from pydantic import BaseModel, Field


class OpenSCADPlan(BaseModel):
    """Structured instructor response for Stage 3 parametric generation.

    Not consumed by any pipeline stage yet -- defined now so the model
    surface is stable before pipeline/openscad_generator.py is built.
    """

    parameters: dict[str, float] = Field(default_factory=dict)
    scad_source: str
    expected_bounds_m: list[float] = Field(min_length=3, max_length=3)
    notes: str = ""


class GenerationResult(BaseModel):
    asset_id: str
    success: bool
    stl_path: str | None = None
    actual_bounds_m: list[float] | None = None
    repair_attempts: int = 0
    error_message: str | None = None
