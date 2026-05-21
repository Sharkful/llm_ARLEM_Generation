# AR Scene Narration Brief JSON Standard

## Purpose

This document defines a lightweight JSON format for describing the instructional intent of an AR scene and its narration clips.

The format is intended for two audiences:

1. **AI coding models** that will implement, validate, or extend the AR narration pipeline.
2. **Human reviewers** who need to understand what the JSON is supposed to represent.

This JSON is **not** the final narration script and is **not** sent directly to a text-to-speech system. Instead, it is an intermediate planning format used by a downstream LLM to generate the final TTS narration JSON.

The brief answers three questions:

- What is the learner looking at?
- What should the learner understand?
- For each clip, what changes in the scene, and why does that change matter?

## Pipeline Position

The intended pipeline is:

```text
AR scene generation
        ↓
AR Scene Narration Brief JSON
        ↓
Narration-generating LLM
        ↓
Final AR TTS Narration JSON
        ↓
TTS generation system
        ↓
Audio clips
```

This standard describes only the second stage: the **AR Scene Narration Brief JSON**.

## Design Goals

The format should be:

- **Lightweight**: only enough information to guide narration generation.
- **Simple**: easy for an LLM to generate and easy for a human to inspect.
- **Structured**: organized by scene, objectives, modules, and clips.
- **Non-final**: fields are source notes, not final narration text.
- **Order-based**: modules and clips are ordered by their position in lists.
- **Provider-independent**: no OpenAI, TTS, filename, voice, or audio-generation details.

## Non-Goals

This JSON should not include:

- Final narration text.
- Voice names.
- TTS model names.
- Audio filenames.
- Timing data.
- Pause lengths.
- AR anchors or object IDs.
- Clip keys.
- Module order numbers.
- File output paths.
- Style presets.
- Emphasis instructions.

Those details belong either in the final narration JSON or in the audio-generation pipeline.

## Core Concept

Each clip brief contains two ideas:

```json
{
  "change": "What changes visually or interactively in the scene.",
  "meaning": "Why that change matters instructionally."
}
```

The downstream narration-generating LLM should use these fields to create natural spoken explanation.

The LLM should **not** simply read `change` and `meaning` as narration text.

## Minimal JSON Structure

```json
{
  "scene": "A short description of what the learner is looking at.",
  "objectives": [
    "What the learner should understand."
  ],
  "modules": [
    {
      "clips": [
        {
          "change": "What changes in the scene.",
          "meaning": "Why that change matters."
        }
      ]
    }
  ]
}
```

## Recommended JSON Structure

The only optional field currently recommended is `modules[].title`, which improves readability for humans and may help downstream narration generation.

```json
{
  "scene": "A 3D loss surface with a marker showing the current model parameters and a path showing gradient descent steps.",
  "objectives": [
    "Explain loss as model error.",
    "Show gradient descent as downhill movement.",
    "Compare small and large learning rates."
  ],
  "modules": [
    {
      "title": "Gradient Descent Basics",
      "clips": [
        {
          "change": "The loss surface and starting marker appear.",
          "meaning": "Each position represents a possible model; height represents error."
        },
        {
          "change": "An arrow points downhill from the marker.",
          "meaning": "The gradient tells us how error changes, so we move in the opposite direction to reduce error."
        }
      ]
    },
    {
      "title": "Learning Rate",
      "clips": [
        {
          "change": "The marker moves in many tiny steps.",
          "meaning": "A small learning rate is stable but slow."
        },
        {
          "change": "The marker jumps across the valley and misses the minimum.",
          "meaning": "A large learning rate can overshoot and fail to converge."
        }
      ]
    }
  ]
}
```

## Required Fields

### Top Level

| Field | Type | Required | Purpose |
|---|---:|---:|---|
| `scene` | string | yes | Short description of what the learner sees in the AR scene. |
| `objectives` | list of strings | yes | Learning objectives for the scene. |
| `modules` | list | yes | Ordered list of scene/narration modules. |

### Module Level

| Field | Type | Required | Purpose |
|---|---:|---:|---|
| `title` | string | no | Optional human-readable module title. |
| `clips` | list | yes | Ordered list of clip briefs within the module. |

### Clip Level

| Field | Type | Required | Purpose |
|---|---:|---:|---|
| `change` | string | yes | What changes visually or interactively in the clip. |
| `meaning` | string | yes | Why the change matters instructionally. |

## Interpretation Rules

### `scene`

The `scene` field should briefly describe what the learner sees. It should provide visual and conceptual context for the narration-generating LLM.

Good example:

```json
"scene": "A 3D model shows Earth orbiting the Sun while Earth's tilted axis remains pointed in the same direction."
```

Poor example:

```json
"scene": "Welcome to this lesson about seasons."
```

The poor example sounds like narration. The field should describe the scene, not speak to the learner.

### `objectives`

The `objectives` list should describe what the learner should understand. These are not narration lines.

Good example:

```json
"objectives": [
  "Explain why Earth has seasons.",
  "Connect axial tilt to changes in sunlight angle and day length."
]
```

Poor example:

```json
"objectives": [
  "Today we are going to learn about the seasons."
]
```

### `change`

The `change` field should describe what happens in the scene during a clip.

Good example:

```json
"change": "Earth moves to the June position in its orbit."
```

Poor example:

```json
"change": "Now look at Earth in June."
```

### `meaning`

The `meaning` field should explain why the change matters for learning.

Good example:

```json
"meaning": "The northern hemisphere is tilted toward the Sun, producing longer days and more direct sunlight."
```

Poor example:

```json
"meaning": "This is very important, so pay attention."
```

## Pydantic Model

The following Pydantic v2 model defines the current standard.

```python
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SceneClipBrief(BaseModel):
    """
    Brief for one narration clip.

    This is NOT the narration text. It is a compact instructional brief
    that a downstream LLM should use to generate spoken narration.
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

    Important:
    - This is not the final TTS narration JSON.
    - The text in `scene`, `objectives`, `change`, and `meaning` should not be
      treated as final narration text.
    - A downstream LLM should use this as the basis for generating spoken
      narration segments, pauses, style, and optional clip titles.
    - Module order and clip order come from list order.
    - No file names, voices, timing, AR anchors, or TTS-provider details belong here.
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
        for i, module in enumerate(self.modules, start=1):
            if not module.clips:
                raise ValueError(f"Module {i} must contain at least one clip.")
        return self
```

## Validation Expectations

The model should reject:

- Unknown fields.
- Blank strings.
- Empty objective lists.
- Empty module lists.
- Empty clip lists.
- Clip briefs without both `change` and `meaning`.

The model should allow:

- Modules without titles.
- Any number of clips per module.
- Any short natural-language phrasing in `scene`, `objectives`, `change`, and `meaning`.

## Guidance for Coding Models

When extending this format, preserve the distinction between:

1. **Briefing data**: compact source notes used to generate narration.
2. **Narration data**: final spoken text, pauses, voices, style, and TTS parameters.
3. **Generation metadata**: filenames, durations, warnings, logs, and output paths.

This file describes only briefing data.

Do not add fields to this schema just because they may eventually be needed downstream. Most downstream details should be added to the final narration schema or to generation metadata instead.

## Recommended Prompt Instruction for Downstream Narration Generation

When using this JSON to generate the final narration script, include an instruction similar to:

```text
Use the scene narration brief as source information. Do not treat the fields as final narration text. Convert the scene description, objectives, visual changes, and instructional meanings into clear spoken narration suitable for students in an AR lab. Preserve the module and clip order. Generate narration clips with say/pause segments using the final AR TTS narration JSON schema.
```

## Future Expansion Notes

Future versions may add a small number of fields if they prove necessary. Possible additions include:

- `schema_version`, if the brief format becomes widely used across projects.
- `lab_title`, if human organization becomes difficult without it.
- `audience`, if narration must vary strongly by learner level.

However, these should not be added unless there is a clear implementation need.

The current design intentionally keeps the format minimal.
