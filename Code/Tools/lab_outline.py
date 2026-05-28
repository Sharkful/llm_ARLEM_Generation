"""
AR Lab Outline Pydantic Model

Minimal response model used by the L1 "outline only" prompt level.
Produces a rough scene-by-scene sketch instead of a full Lab spec.
No discriminated unions or cross-field validators, so it works with
all providers including Gemini.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OutlineScene(BaseModel):
    scene_name: str
    brief_purpose: str
    key_visuals: list[str] = Field(min_length=1)
    student_actions: list[str] = Field(min_length=1)


class LabOutline(BaseModel):
    lab_title: str
    top_level_objectives: list[str] = Field(min_length=1)
    scenes: list[OutlineScene] = Field(min_length=3, max_length=12)
