from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SceneClipBrief(BaseModel):
    """
    Brief for one narration clip.

    This is NOT the narration text. It is a compact instructional brief
    that a downstream LLM should use to generate spoken narration.

    The narration-generating LLM should:
    - explain the scene change naturally,
    - connect the change to the instructional meaning,
    - avoid reading these fields verbatim unless the wording is already ideal.
    """

    model_config = ConfigDict(extra="forbid")

    change: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "What changes visually or interactively in this clip. "
            "This is source information for narration generation, not narration text."
        ),
    )

    meaning: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Why the change matters instructionally. "
            "This is source information for narration generation, not narration text."
        ),
    )

    @field_validator("change", "meaning")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be blank.")
        return value


class SceneModuleBrief(BaseModel):
    """
    Brief for one module within the AR scene.

    Modules are ordered by their position in the list.
    The optional title is for human readability only.
    """

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=160,
        description="Optional human-readable module title.",
    )

    clips: list[SceneClipBrief] = Field(
        ...,
        min_length=1,
        description="Ordered list of clip briefs for this module.",
    )

    @field_validator("title")
    @classmethod
    def strip_optional_title(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ARSceneNarrationBrief(BaseModel):
    """
    Lightweight scene-and-clip brief used to generate AR narration.

    This model is intended as the upstream input to a narration-generating LLM.
    It is created when the AR scene and clips are generated.

    It should remain lightweight and simple.

    Required structure:
    - scene: short description of what the learner is looking at
    - objectives: what the learner should understand
    - modules: ordered groups of clips
    - clips[].change: what changes in the scene
    - clips[].meaning: why that change matters

    Important:
    - This is not the final TTS narration JSON.
    - The text in `scene`, `objectives`, `change`, and `meaning` should not be
      treated as final narration text.
    - A downstream LLM should use this as the basis for generating spoken
      narration segments, pauses, style, and optional clip titles.
    - Module order and clip order come from list order.
    - No file names, voices, timing, AR anchors, or TTS-provider details belong here.

    Example:

        {
          "scene": "A 3D loss surface with a marker showing the current model parameters.",
          "objectives": [
            "Explain loss as model error.",
            "Show gradient descent as downhill movement."
          ],
          "modules": [
            {
              "title": "Gradient Descent Basics",
              "clips": [
                {
                  "change": "The loss surface and starting marker appear.",
                  "meaning": "Each position represents a possible model; height represents error."
                }
              ]
            }
          ]
        }
    """

    model_config = ConfigDict(extra="forbid")

    scene: str = Field(
        ...,
        min_length=1,
        max_length=1500,
        description=(
            "Short description of what the learner is looking at in the AR scene. "
            "This provides visual context for narration generation."
        ),
    )

    objectives: list[str] = Field(
        ...,
        min_length=1,
        max_length=8,
        description=(
            "Learning objectives for the scene. These should describe what the learner "
            "should understand after the narrated clips."
        ),
    )

    modules: list[SceneModuleBrief] = Field(
        ...,
        min_length=1,
        description="Ordered list of module briefs in the AR scene.",
    )

    @field_validator("scene")
    @classmethod
    def strip_scene(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Scene description cannot be blank.")
        return value

    @field_validator("objectives")
    @classmethod
    def clean_objectives(cls, values: list[str]) -> list[str]:
        cleaned = []
        for objective in values:
            objective = objective.strip()
            if objective:
                cleaned.append(objective)

        if not cleaned:
            raise ValueError("At least one non-blank objective is required.")

        return cleaned

    @model_validator(mode="after")
    def ensure_each_module_has_clips(self) -> "ARSceneNarrationBrief":
        """
        Redundant with field validation, but useful as a clear semantic check.
        """
        for i, module in enumerate(self.modules, start=1):
            if not module.clips:
                raise ValueError(f"Module {i} must contain at least one clip.")
        return self