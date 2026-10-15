"""
Prompt templates for hierarchical (module-at-a-time) JSON Lab generation.

Three LLM steps, each with its own template:
  1. PLAN_INSTRUCTION   lab description -> LabOutline (one scene per module)
  2. ASSET_PLAN_TASK    one scene -> ModuleAssetPlan (objects, components, why)
  3. MODULE_TASK        approved asset plan -> DemoModule

Prompt order is cache-friendly: every prompt *starts* with the same block (the
lab description), and the asset-plan and module prompts continue with the same
approved plan, before anything step- or module-specific. Providers with automatic
prefix caching (OpenAI, Gemini) then bill that repeated opening at the cached
rate on every call after the first. The asset-plan and module prompts are built
as shared_opening() + their task template, so the opening is the same bytes by
construction; Anthropic caches only marked prefixes, and the pipeline marks the
end of that opening as a cache breakpoint (#73).

The plan, earlier modules, and the asset registry are appended as compact
summaries rather than full JSON, so prompt size grows slowly with module count.
Structural rules mirror prompt_builder.FULL_SPEC_MULTI_MODULE_INSTRUCTION so
outputs stay comparable with one-shot runs, except that the object and clip
minimums apply per module here (they are lab-wide in the one-shot runs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

import _paths  # noqa: F401

from lab_outline import LabOutline, OutlineScene
from module_asset_plan import ModuleAssetPlan, component_key
from registry import AssetRegistry


# Shared openings. Keep these byte-identical across calls: any change here moves
# the point where the cached prefix ends.
LAB_BLOCK = """\
# Lab description
{lab_context}
"""

PLAN_BLOCK = """\
# Approved lab plan (one module per scene)
{plan}
"""

PLAN_INSTRUCTION = LAB_BLOCK + """
# Task: plan the lab
You are designing an AR lab that will be generated one module at a time. This first
step is the lab plan: a scene-by-scene outline in which EACH SCENE BECOMES ONE MODULE
of the final lab. Later steps will plan each module's assets and then write the
module itself, so do NOT specify prefabs, positions, components, or clip-level
changes here.

Requirements:
- Use {min_scenes}-{max_scenes} scenes, in teaching order. Each scene should contribute one clear
  pedagogical step toward the lab's goals.
- Together the scenes must cover every learning objective{script_clause}.
- For each scene give its purpose, the key visuals the student sees, and what the
  student does, in short prose.
- top_level_objectives are the learning objectives for the lab as a whole.
"""

# The asset-plan and module prompts are shared_opening() + one of these.
ASSET_PLAN_TASK = """
# Asset library
{assets}

# Component catalog
{catalog}

# Modules already generated
{previous}

# Task: plan the assets for module {index} of {total}
{scene}

Before module {index} is written, plan the assets it needs: its scene objects, the
components that give those objects behavior, the student interactions, and one beat
per clip. Explain WHY each object and component is needed.

Instructions:
- Fill in design_notes FIRST: reason about what the student must see and do in this
  module to achieve its purpose, and what that requires of the scene.
- List every scene object the module needs: at least {min_objects}, each with a
  purpose. Object names must be unique within the module. Prefer built-in prefabs
  and textures, or ones invented by earlier modules, over new ones; set
  is_new_asset=true when you do invent one.
- Plan the components each object needs. object_name must exactly match one of your
  planned objects, every component needs a rationale, and the same component should
  not be planned twice on one object. Prefer catalog components, and combinations of
  them, wherever they can produce the behavior.
{new_component_rule}
- interactions: each one names the planned object the student interacts with
  (object_name), what the student does (action), and what happens (effect).
- clip_beats: at least {min_beats} beats, one short line per clip in order, saying
  what changes in the scene and what is narrated.
- Keep in mind the lab is a tabletop AR experience (objects within a ~3m workspace,
  scales typically 0.02-0.5 units), but do not give positions yet.
"""

NEW_COMPONENTS_ALLOWED = """\
- A component is a small piece of C# behavior and can do almost anything. That makes
  new components powerful but costly to build, so only plan a new one when no catalog
  component, or combination of them, can provide the behavior. Keep each new
  component to ONE simple behavior with clear inputs and outputs; split complex
  behavior into several small components that later modules can reuse.
- Declare each new component exactly once in new_components: a PascalCase name of
  at least 5 letters or digits (e.g. 'PulseGlow'), a behavior_description,
  why_existing_insufficient, and its inputs: every inspector field it exposes
  (object references and values) as {name, type, description}, or an empty list if
  it has none. Then attach it by that name in components, once for every object
  that uses it. Never re-declare a component that is already in the
  catalog: attach it by its catalog name."""

NEW_COMPONENTS_DISABLED = """\
- New components are disabled for this run: leave new_components empty and use only
  components from the catalog."""

MODULE_TASK = """
# Component catalog
{catalog}

# Modules already generated
{previous}

# Task: write module {index} of {total}
{scene}

## Approved asset plan for this module
{asset_plan}

Write module {index} as a complete DemoModule, implementing its approved asset plan.

Requirements:
- moduleName: "{module_name}". Give the module a short description and educational
  objectives drawn from the lab's objectives.
- Create every planned object using exactly its planned name, prefab, and texture
  (at least {min_objects} objects). You may add minor supporting objects (such as
  text labels) if the clips need them.
- Attach each planned component to its planned object, either in the object's
  components or in a clip change if it should only take effect later.
- Built-in components use their own type (e.g. SimpleOrbitComponent). A new
  component, or one invented by an earlier module, is never its own type: write it
  as a NewComponent entry exactly as shown below, setting each input's value for
  that instance (or leaving it out to keep the default). Do not use NewComponent
  with any other scriptName.
{new_component_entries}
- Write the clips in the order of the clip beats, at least one clip per beat and at
  least {min_clips} clips in total (you may split a beat into several clips). Each
  clip uses sparse object changes, states its changeMeaning, and gives narration and
  an audioClip name when narrated.
- Make the planned interactions possible (e.g. give interactive objects a
  PointerReceiverComponent).
- Every change target must name an object defined in this module.
- Objects should have realistic positions (within a ~3m workspace centered on the
  user) and scales appropriate for a tabletop AR experience (typically 0.02-0.5 units).
"""

REVISION_INSTRUCTION = """\
A reviewer asked for changes to your previous {what}. Address ALL of the feedback
below; earlier items still apply unless a later item overrides them:

{feedback}

Return the complete revised {what}, addressing the feedback and keeping everything
the reviewer did not ask to change."""


def _length_bounds(model: type[BaseModel], field_name: str) -> tuple[Optional[int], Optional[int]]:
    """The (min_length, max_length) a schema puts on a list field."""
    lo = hi = None
    for m in model.model_fields[field_name].metadata:
        lo = getattr(m, "min_length", lo)
        hi = getattr(m, "max_length", hi)
    return lo, hi


# Read from the schemas so they can't drift: LabOutline's scene range (the plan
# step caps it further with --max-scenes) and ModuleAssetPlan's clip-beat minimum.
MIN_SCENES, MAX_SCENES_LIMIT = _length_bounds(LabOutline, "scenes")
MIN_CLIP_BEATS = _length_bounds(ModuleAssetPlan, "clip_beats")[0]


# ── Rendering helpers ────────────────────────────────────────────────

def _render_scene(scene: OutlineScene) -> str:
    return "\n".join([
        f"Scene: {scene.scene_name}",
        f"Purpose: {scene.brief_purpose}",
        "Key visuals: " + "; ".join(scene.key_visuals),
        "Student actions: " + "; ".join(scene.student_actions),
    ])


def render_plan(plan: LabOutline) -> str:
    """The approved plan as a numbered scene list. Identical for every module of a
    run (the current module is named in the task section instead), so it stays
    part of the cacheable prefix."""
    lines = [f"Lab title: {plan.lab_title}", "Lab objectives:"]
    lines += [f"- {o}" for o in plan.top_level_objectives]
    lines.append("Modules (one per scene):")
    for i, scene in enumerate(plan.scenes, start=1):
        lines.append(f"{i}. {scene.scene_name}: {scene.brief_purpose}")
    return "\n".join(lines)


# ── Prompt builders ──────────────────────────────────────────────────

def shared_opening(lab_context: str, plan: LabOutline) -> str:
    """The opening every asset-plan and module prompt starts with: the builders
    below take it and append their task. The pipeline builds it once per run and
    marks its end as Anthropic's cache breakpoint (#73)."""
    return (LAB_BLOCK.format(lab_context=lab_context) + "\n"
            + PLAN_BLOCK.format(plan=render_plan(plan)))


def build_plan_prompt(lab_context: str, *, has_script: bool, max_scenes: int) -> str:
    """The scene range comes from the run's max_scenes (validated against
    LabOutline's own 3-12 bounds in PipelineConfig), so the prompt asks for
    exactly what LabPlan will accept."""
    script_clause = " and every part of the detailed script" if has_script else ""
    return PLAN_INSTRUCTION.format(
        lab_context=lab_context, script_clause=script_clause,
        min_scenes=MIN_SCENES, max_scenes=max_scenes,
    )


def build_asset_plan_prompt(
    opening: str,
    plan: LabOutline,
    index: int,
    registry: AssetRegistry,
    *,
    allow_new_components: bool,
    min_objects: int,
    min_beats: int,
) -> str:
    """``opening`` is ``shared_opening(lab_context, plan)``."""
    return opening + ASSET_PLAN_TASK.format(
        index=index,
        total=len(plan.scenes),
        scene=_render_scene(plan.scenes[index - 1]),
        previous=registry.render_previous_modules(),
        assets=registry.render_assets(),
        catalog=registry.render_component_catalog(),
        new_component_rule=(
            NEW_COMPONENTS_ALLOWED if allow_new_components else NEW_COMPONENTS_DISABLED
        ),
        min_objects=min_objects,
        min_beats=min_beats,
    )


# Shown as each input's value in the module prompt's ready-to-copy entries; the
# module step rejects it if it comes back unchanged.
VALUE_PLACEHOLDER = "<value for this instance>"


def is_value_placeholder(value: Optional[str]) -> bool:
    """True for the placeholder, ignoring case, surrounding spaces and the angle
    brackets ('<value for this instance> ', 'Value for this instance')."""
    return value is not None and value.strip().strip("<>").strip().lower() == \
        VALUE_PLACEHOLDER.strip("<>").lower()


@dataclass
class UsableNewComponent:
    """A new component a module may use, with its canonical definition: from the
    catalog for one an earlier module invented, from this module's asset plan for
    one it declares. ``inputs`` holds {name, type, description} dicts; None leaves
    the module's input names unchecked. ``from_catalog``: the description is the
    catalog's (canonical); otherwise it is the plan's, and normalization takes the
    module's first description instead."""
    name: str
    description: str = ""
    inputs: Optional[list[dict]] = None
    from_catalog: bool = False

    def input_names(self) -> Optional[dict[str, str]]:
        """component_key(input name) -> its canonical name, or None if unchecked."""
        if self.inputs is None:
            return None
        return {component_key(i["name"]): i["name"] for i in self.inputs}


def usable_new_components(
    asset_plan: ModuleAssetPlan, registry: AssetRegistry
) -> list[UsableNewComponent]:
    """Every new component a module may use: all already in the catalog, plus the
    ones its plan declares. Computed once per module by the pipeline and passed to
    the module prompt's entries, the module step's validation scope, and
    normalization."""
    usable = [UsableNewComponent(c.name, c.description, list(c.inputs), from_catalog=True)
              for c in registry.components if not c.builtin]
    for spec in asset_plan.new_components:
        if registry.find_component(spec.name) is None:
            usable.append(UsableNewComponent(spec.name, spec.behavior_description,
                                             [i.model_dump() for i in spec.inputs]))
    return usable


def render_new_component_entries(usable: list[UsableNewComponent]) -> str:
    """One ready-to-copy NewComponent entry per component the module may use, so
    the plan's bare names don't prime ``{"type": "<name>"}`` (#77)."""
    lines = []
    for c in usable:
        literal = {"type": "NewComponent", "scriptName": c.name, "scriptDescription": c.description}
        if c.inputs:
            literal["inputs"] = [{**i, "value": VALUE_PLACEHOLDER} for i in c.inputs]
        lines.append("  " + json.dumps(literal, ensure_ascii=False))
    return "\n".join(lines) if lines else (
        "  (none: no new components are available to this module; use built-in "
        "components only)"
    )


def build_module_prompt(
    opening: str,
    plan: LabOutline,
    index: int,
    registry: AssetRegistry,
    asset_plan: ModuleAssetPlan,
    *,
    usable: list[UsableNewComponent],
    min_objects: int,
    min_clips: int,
) -> str:
    """``opening`` is ``shared_opening(lab_context, plan)``; ``usable`` is
    ``usable_new_components(asset_plan, registry)``."""
    return opening + MODULE_TASK.format(
        index=index,
        total=len(plan.scenes),
        scene=_render_scene(plan.scenes[index - 1]),
        previous=registry.render_previous_modules(),
        asset_plan=json.dumps(asset_plan.model_dump(mode="json", exclude_none=True), indent=2),
        catalog=registry.render_component_catalog(),
        module_name=asset_plan.module_name,
        new_component_entries=render_new_component_entries(usable),
        min_objects=min_objects,
        min_clips=min_clips,
    )


def build_revision_message(what: str, feedback_history: list[str]) -> str:
    """Revision request carrying every round of feedback for the step, oldest
    first, so a later round doesn't silently drop an earlier request."""
    feedback = "\n".join(f"{i}. {f}" for i, f in enumerate(feedback_history, start=1))
    return REVISION_INSTRUCTION.format(what=what, feedback=feedback)
