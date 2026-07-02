from __future__ import annotations

from pydantic import BaseModel, Field


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
