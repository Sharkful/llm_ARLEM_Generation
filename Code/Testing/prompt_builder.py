"""
AR Lab Prompt Builder

Loads a lab-description YAML and assembles the prompt for one of four
specificity levels (L1-L4), pairing it with the appropriate Pydantic
response model for handoff to Instructor.

L1 produces a rough outline (LabOutline) for every spec. L2-L4 produce full
specs: for json_lab, --structure selects between a single multi-clip module,
multiple per-scene modules, or a bare DemoModule with no Lab wrapper; for arlem
and arlem_simple it produces an ARLEMScenario (--structure does not apply).

Pure module: no LLM calls, no benchmark_config import (to avoid a circular
dependency). The caller passes spec_type as a string value, which matches
benchmark_config.SpecType since that enum inherits from str.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Type

import yaml
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LAB_DESCRIPTIONS_DIR = PROJECT_ROOT / "Artifacts" / "Lab Descriptions"

REQUIRED_YAML_FIELDS = (
    "topic_name",
    "field",
    "course_context",
    "topic_description",
    "learning_objectives",
    "detailed_script",
)

_REVIEW_MARKER_RE = re.compile(r"\[REVIEW:[^\]]*\]")


# ── Enums ────────────────────────────────────────────────────────────

class Level(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


class Structure(str, Enum):
    SINGLE_MODULE = "single-module"
    MULTI_MODULE = "multi-module"
    MODULE_ONLY = "module-only"


# ── Lab Description (input schema) ───────────────────────────────────

@dataclass
class LabDescription:
    """Parsed and flag-stripped contents of a <topic>_lab.yaml file."""
    topic_name: str
    field: str
    course_context: str
    topic_description: str
    learning_objectives: list[str]
    detailed_script: str

    @classmethod
    def from_yaml(cls, path: Path) -> "LabDescription":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"YAML at {path} did not parse as a mapping")

        missing = [k for k in REQUIRED_YAML_FIELDS if k not in data]
        if missing:
            raise ValueError(
                f"YAML at {path} is missing required fields: {missing}"
            )

        return cls(
            topic_name=_strip_flags(data["topic_name"]),
            field=_strip_flags(data["field"]),
            course_context=_strip_flags(data["course_context"]),
            topic_description=_strip_flags(data["topic_description"]),
            learning_objectives=[_strip_flags(item) for item in data["learning_objectives"]],
            detailed_script=_strip_flags(data["detailed_script"]),
        )


def _strip_flags(value: str) -> str:
    """Remove leading '*' authoring flag and inline [REVIEW: ...] markers."""
    if not isinstance(value, str):
        return value
    cleaned = _REVIEW_MARKER_RE.sub("", value).strip()
    if cleaned.startswith("*"):
        cleaned = cleaned.lstrip("*").lstrip()
    return cleaned


# ── Lab discovery ────────────────────────────────────────────────────

def discover_labs(
    directory: Path = DEFAULT_LAB_DESCRIPTIONS_DIR,
) -> dict[str, Path]:
    """Glob *.yaml in the lab-description directory, keyed by topic_name."""
    out: dict[str, Path] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "topic_name" in data:
            out[_strip_flags(data["topic_name"])] = path
    return out


# ── Instruction wrappers ─────────────────────────────────────────────
# Module-level so they're easy to tune without touching dispatch logic.

OUTLINE_INSTRUCTION = """\
Given the lab topic below, produce a rough scene-by-scene outline of the
AR lab. Do NOT write a full specification with prefabs, scene objects, or
clip-level changes. Instead, describe each scene's purpose, the key visuals
the student should see, and what the student is asked to do in that scene,
all in short prose.

Aim for 4-8 scenes. Each scene should contribute one clear pedagogical step
toward the lab's goals.

{lab_context}
"""

FULL_SPEC_SINGLE_MODULE_INSTRUCTION = """\
Create a complete AR lab specification for the topic described below.

Structural requirements:
- The lab MUST contain exactly 1 module: a "demo" type module (bridge/demonstration)
- The module should guide the student through the entire topic using a sequence of clips
- Include at least {min_objects} 3D scene objects relevant to the topic
- Create at least {min_clips} clips that form a logical educational narrative covering
  every scene described in the input
- Each clip should use sparse object changes to animate/update the scene
- Objects should have realistic positions (within a ~3m workspace centered on the user)
- Scales should be appropriate for a tabletop AR experience (objects typically 0.02-0.5 units)
- Include educational objectives for both the lab and the module
- Use descriptive prefab names (e.g., 'heartPrefab', 'moleculePrefab') for the topic
- Reference plausible texture names where appropriate
- Add behavioral components (rotation, orbit, text labels) where they enhance learning

The lab should feel like a guided walkthrough where a narrator explains the topic
while 3D objects appear, move, and change to illustrate key concepts.

{lab_context}
"""

FULL_SPEC_MULTI_MODULE_INSTRUCTION = """\
Create a complete AR lab specification for the topic described below.

Structural requirements:
- Create a Lab containing one "demo" type module per scene described in the input.
  Each module owns its own set of scene objects and runs through clips that
  cover that scene's content.
- Modules should appear in the same order as the scenes appear in the input.
- Across the whole lab, include at least {min_objects} total scene objects and
  at least {min_clips} total clips, distributed across the modules.
- Each clip should use sparse object changes to animate/update the scene
- Objects should have realistic positions (within a ~3m workspace centered on the user)
- Scales should be appropriate for a tabletop AR experience (objects typically 0.02-0.5 units)
- Include educational objectives for the lab as a whole and for each module
- Use descriptive prefab names (e.g., 'heartPrefab', 'moleculePrefab') for the topic
- Reference plausible texture names where appropriate
- Add behavioral components (rotation, orbit, text labels) where they enhance learning

The lab should feel like a sequence of guided scenes, each with its own staging,
where a narrator explains the topic while 3D objects appear, move, and change.

{lab_context}
"""

FULL_SPEC_MODULE_ONLY_INSTRUCTION = """\
Return a single DemoModule (NOT wrapped in a Lab object) for the topic
described below. The module should walk through every scene in the input
as a sequence of clips on a shared set of scene objects.

Structural requirements:
- Include at least {min_objects} 3D scene objects relevant to the topic
- Create at least {min_clips} clips that form a logical educational narrative
  covering every scene described in the input
- Each clip should use sparse object changes to animate/update the scene
- Objects should have realistic positions (within a ~3m workspace centered on the user)
- Scales should be appropriate for a tabletop AR experience (objects typically 0.02-0.5 units)
- Include educational objectives for the module
- Use descriptive prefab names (e.g., 'heartPrefab', 'moleculePrefab') for the topic
- Reference plausible texture names where appropriate
- Add behavioral components (rotation, orbit, text labels) where they enhance learning

{lab_context}
"""


# ── ARLEM instruction wrappers (Workplace + Activity) ────────────────
# ARLEM has a single output shape (ARLEMScenario); the --structure modes above
# do not apply. min_things / min_places / min_actions are the structural minima.

ARLEM_FULL_INSTRUCTION = """\
Create a complete ARLEM AR training scenario for the topic described below.

Build an ARLEMScenario with two parts: a Workplace (the static environment) and an
Activity (the step-by-step logic).

Workplace requirements:
- Define at least {min_things} things (tools, models, or materials the learner interacts with)
  and at least {min_places} places (workstations, zones, or surfaces).
- Include exactly 1 person (the learner) and at least 1 device (the AR headset).
- Define detectables (markers or anchors) for spatial tracking. The workplace `origin` must
  reference one detectable, and every thing/place/person must reference a distinct detectable.
- List the primitives (supported media types: label, image, audio, video, animation) and create
  predicates (reusable instructional augmentations) that the activity steps will display.
- Any POI with id "default" must sit at the origin (zero offsets); rotation is allowed.

Activity requirements:
- Create at least {min_actions} sequential action steps with a clear pedagogical progression.
- Each action has an instruction (title + description) and enter/exit flows that activate and
  deactivate augmentations on workplace targets.
- Advance between steps with triggers (mode "click", "voice", or "detect").
- Every reference (activate/deactivate targets, augmentations, action location/device/predicate,
  POIs) must resolve to something defined in the Workplace. The `start` field and any
  action-to-action links must reference valid action ids. Language is the 2-letter ISO 639-1
  code (e.g. "en").

The scenario should read like a realistic hands-on AR training session where augmentations
guide the learner through the topic.

{lab_context}
"""

ARLEM_SIMPLE_INSTRUCTION = """\
Create a simplified ARLEM AR scenario for the topic described below.

Build an ARLEMScenario with a Workplace and a linear Activity. This is the REDUCED ARLEM
surface: there are NO persons, devices, sensors, warnings, messages, or conditional logic, and
triggers are limited to mode "click" and "detect".

Workplace requirements:
- Define at least {min_things} things and at least {min_places} places.
- Define detectables (markers or anchors). The workplace `origin` must reference one detectable,
  and every thing/place must reference a distinct detectable.
- List the primitives (label, image, audio, video, animation) and create predicates (reusable
  instructional augmentations) used by the activity steps.
- Any POI with id "default" must sit at the origin (zero offsets); rotation is allowed.

Activity requirements:
- Create at least {min_actions} sequential action steps, each with an instruction (title +
  description) and enter/exit flows that activate/deactivate augmentations on workplace targets.
- Advance between steps with triggers of mode "click" or "detect".
- Every reference (activate/deactivate targets, augmentations, POIs) must resolve to something
  defined in the Workplace. The `start` field and action-to-action links must reference valid
  action ids.

{lab_context}
"""


# ── Context rendering ────────────────────────────────────────────────

def _render_context(lab: LabDescription, level: Level) -> str:
    """Build the labelled lab-context block for the given level."""
    parts = [
        f"## Field\n{lab.field}",
        f"## Course context\n{lab.course_context}",
        f"## Topic description\n{lab.topic_description}",
    ]
    if level in (Level.L3, Level.L4):
        objectives = "\n".join(f"- {obj}" for obj in lab.learning_objectives)
        parts.append(f"## Learning objectives\n{objectives}")
    if level == Level.L4:
        parts.append(f"## Detailed script\n{lab.detailed_script}")
    return "\n\n".join(parts)


# ── Main entry point ─────────────────────────────────────────────────

def build_prompt(
    lab: LabDescription,
    level: Level,
    spec_type: str,
    *,
    structure: Structure = Structure.MULTI_MODULE,
    min_objects: int = 4,
    min_clips: int = 5,
    min_things: int = 3,
    min_places: int = 2,
    min_actions: int = 5,
) -> tuple[str, Type[BaseModel]]:
    """
    Build the prompt and pick the response model for a single run.

    Args:
        lab: parsed lab description.
        level: L1 (outline) through L4 (full input).
        spec_type: a SpecType value ("json_lab", "arlem", "arlem_simple").
            SpecType inherits from str, so callers can pass the enum directly.
        structure: output structure mode for json_lab L2-L4. Ignored for L1 and
            for ARLEM (which has a single ARLEMScenario output shape).
        min_objects, min_clips: json_lab structural minima injected into the wrapper.
        min_things, min_places, min_actions: ARLEM structural minima.

    Returns:
        (prompt_string, response_model_class)
    """
    lab_context = _render_context(lab, level)

    # L1 is a rough outline for EVERY spec: the same minimal, provider-safe
    # LabOutline model (scenes/objectives), independent of the target spec.
    if level == Level.L1:
        from lab_outline import LabOutline
        prompt = OUTLINE_INSTRUCTION.format(lab_context=lab_context)
        return prompt, LabOutline

    # L2-L4: dispatch on spec_type.
    if spec_type == "json_lab":
        # json_lab.Lab / DemoModule are provider-agnostic (plain smart-union +
        # const->enum via ConstToEnumSchemaMixin), so the same response model goes
        # to every provider. --structure selects the output shape.
        from json_lab import DemoModule, Lab

        if structure == Structure.SINGLE_MODULE:
            template = FULL_SPEC_SINGLE_MODULE_INSTRUCTION
            response_model = Lab
        elif structure == Structure.MULTI_MODULE:
            template = FULL_SPEC_MULTI_MODULE_INSTRUCTION
            response_model = Lab
        elif structure == Structure.MODULE_ONLY:
            template = FULL_SPEC_MODULE_ONLY_INSTRUCTION
            response_model = DemoModule
        else:
            raise ValueError(f"Unknown structure: {structure}")

        prompt = template.format(
            lab_context=lab_context,
            min_objects=min_objects,
            min_clips=min_clips,
        )
        return prompt, response_model

    if spec_type in ("arlem", "arlem_simple"):
        # ARLEM is provider-agnostic too (const->enum mixin; no field-level unions).
        # `structure` is ignored — ARLEMScenario is the only output shape.
        if spec_type == "arlem":
            from arlem_full import ARLEMScenario
            template = ARLEM_FULL_INSTRUCTION
        else:
            from arlem_simplified import ARLEMScenario
            template = ARLEM_SIMPLE_INSTRUCTION

        prompt = template.format(
            lab_context=lab_context,
            min_things=min_things,
            min_places=min_places,
            min_actions=min_actions,
        )
        return prompt, ARLEMScenario

    raise ValueError(f"Unknown spec_type: {spec_type!r}")
