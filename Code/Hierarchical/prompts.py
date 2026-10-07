"""
Prompt templates for hierarchical (module-at-a-time) JSON Lab generation.

Three LLM steps, each with its own template:
  1. PLAN_INSTRUCTION         lab description -> LabOutline (one scene per module)
  2. ASSET_PLAN_INSTRUCTION   one scene -> ModuleAssetPlan (objects, components, why)
  3. MODULE_INSTRUCTION       approved asset plan -> DemoModule

Every prompt opens with the same lab-description block (prompt_builder's L3/L4
context) so the source of truth is always in view; the plan, earlier modules, and
the asset registry are appended as compact summaries rather than full JSON, so
prompt size grows slowly with module count. Structural rules mirror
prompt_builder.FULL_SPEC_MULTI_MODULE_INSTRUCTION so outputs stay comparable with
one-shot runs.
"""

from __future__ import annotations

import json
import math

import _paths  # noqa: F401

from lab_outline import LabOutline, OutlineScene
from module_asset_plan import ModuleAssetPlan
from registry import AssetRegistry


PLAN_INSTRUCTION = """\
You are designing an AR lab that will be generated one module at a time. This first
step is the lab plan: a scene-by-scene outline in which EACH SCENE BECOMES ONE MODULE
of the final lab. Later steps will plan each module's assets and then write the
module itself, so do NOT specify prefabs, positions, components, or clip-level
changes here.

Requirements:
- Aim for 3-8 scenes, in teaching order. Each scene should contribute one clear
  pedagogical step toward the lab's goals.
- Together the scenes must cover every learning objective{script_clause}.
- For each scene give its purpose, the key visuals the student sees, and what the
  student does, in short prose.
- top_level_objectives are the learning objectives for the lab as a whole.

{lab_context}
"""

ASSET_PLAN_INSTRUCTION = """\
You are generating an AR lab one module at a time. The lab plan below has been
approved. Before module {index} of {total} is written, plan the assets it needs:
its scene objects, the components that give those objects behavior, the student
interactions, and one beat per clip. Explain WHY each object and component is
needed.

{lab_context}

## Approved lab plan
{plan}

## Module to plan now: module {index} of {total}
{scene}

## Modules already generated
{previous}

## Asset library
{assets}

## Component catalog
{catalog}

Instructions:
- Fill in design_notes FIRST: reason about what the student must see and do in this
  module to achieve its purpose, and what that requires of the scene.
- List every scene object the module needs (at least {min_objects}), each with a
  purpose. Object names must be unique within the module. Prefer built-in prefabs
  and textures, or ones invented by earlier modules, over new ones; set
  is_new_asset=true when you do invent one.
- Plan the components each object needs. object_name must exactly match one of your
  planned objects, and every component needs a rationale. Prefer catalog components,
  and combinations of them, wherever they can produce the behavior.
{new_component_rule}
- interactions: what the student does in this module, with which object, and what
  happens as a result.
- clip_beats: at least {min_beats} beats, one short line per clip in order, saying
  what changes in the scene and what is narrated.
- Keep in mind the lab is a tabletop AR experience (objects within a ~3m workspace,
  scales typically 0.02-0.5 units), but do not give positions yet.
"""

NEW_COMPONENTS_ALLOWED = """\
- A component is a small piece of C# behavior and can do almost anything. That makes
  new components powerful but costly to build, so only plan a new one (is_new=true)
  when no catalog component, or combination of them, can provide the behavior. Keep
  each new component to ONE simple behavior with clear inputs and outputs; split
  complex behavior into several small components that later modules can reuse. Give
  it a lowercase script name of at least 5 characters (e.g. 'pulseglow'), a
  behavior_description, and why_existing_insufficient. Never re-invent a component
  that is already in the catalog: reference it by name with is_new=false."""

NEW_COMPONENTS_DISABLED = """\
- New components are disabled for this run: use only components from the catalog."""

MODULE_INSTRUCTION = """\
You are generating an AR lab one module at a time. Now write module {index} of
{total} as a complete DemoModule, implementing its approved asset plan.

{lab_context}

## Approved lab plan
{plan}

## Module to write now: module {index} of {total}
{scene}

## Modules already generated
{previous}

## Approved asset plan for this module
{asset_plan}

## Component catalog
{catalog}

Requirements:
- moduleName: "{module_name}". Give the module a short description and educational
  objectives drawn from the lab's objectives.
- Create every planned object using exactly its planned name, prefab, and texture.
  You may add minor supporting objects (such as text labels) if the clips need them.
- Attach each planned component to its planned object, either in the object's
  components or in a clip change if it should only take effect later.
- Built-in components use their own type (e.g. SimpleOrbitComponent). A new
  component, or one invented by an earlier module, is written as a NewComponent
  whose scriptName is exactly its planned or catalog name; for a catalog component,
  reuse its description. Allowed NewComponent scriptNames for this module:
  {allowed_new}. Do not use NewComponent with any other scriptName.
- Write the clips in the order of the clip beats, at least one clip per beat (you
  may split a beat into several clips). Each clip uses sparse object changes,
  states its changeMeaning, and gives narration and an audioClip name when narrated.
- Every change target must name an object defined in this module.
- Objects should have realistic positions (within a ~3m workspace centered on the
  user) and scales appropriate for a tabletop AR experience (typically 0.02-0.5 units).
"""

REVISION_INSTRUCTION = """\
A reviewer asked for changes to your previous {what}:

{feedback}

Return the complete revised {what}, addressing the feedback and keeping everything
the reviewer did not ask to change."""


# ── Rendering helpers ────────────────────────────────────────────────

def _render_scene(scene: OutlineScene) -> str:
    return "\n".join([
        f"Scene: {scene.scene_name}",
        f"Purpose: {scene.brief_purpose}",
        "Key visuals: " + "; ".join(scene.key_visuals),
        "Student actions: " + "; ".join(scene.student_actions),
    ])


def render_plan(plan: LabOutline, current: int | None = None) -> str:
    """The approved plan as a numbered scene list; ``current`` (1-based) is marked."""
    lines = [f"Lab title: {plan.lab_title}", "Lab objectives:"]
    lines += [f"- {o}" for o in plan.top_level_objectives]
    lines.append("Modules (one per scene):")
    for i, scene in enumerate(plan.scenes, start=1):
        marker = "  <-- this module" if i == current else ""
        lines.append(f"{i}. {scene.scene_name}: {scene.brief_purpose}{marker}")
    return "\n".join(lines)


def per_module_minimum(total_min: int, num_modules: int, floor: int) -> int:
    """Spread a lab-wide minimum across modules, never below ``floor``."""
    return max(floor, math.ceil(total_min / max(1, num_modules)))


# ── Prompt builders ──────────────────────────────────────────────────

def build_plan_prompt(lab_context: str, *, has_script: bool) -> str:
    script_clause = " and every part of the detailed script" if has_script else ""
    return PLAN_INSTRUCTION.format(lab_context=lab_context, script_clause=script_clause)


def build_asset_plan_prompt(
    lab_context: str,
    plan: LabOutline,
    index: int,
    registry: AssetRegistry,
    *,
    allow_new_components: bool,
    min_objects: int,
    min_beats: int,
) -> str:
    return ASSET_PLAN_INSTRUCTION.format(
        lab_context=lab_context,
        plan=render_plan(plan, current=index),
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


def allowed_new_component_names(
    asset_plan: ModuleAssetPlan, registry: AssetRegistry
) -> list[str]:
    """NewComponent scriptNames a module may use: the ones its plan introduces
    plus every new component already in the catalog."""
    names = list(registry.new_component_names())
    for comp in asset_plan.components:
        if comp.is_new and registry.find_component(comp.component_type) is None:
            if comp.component_type not in names:
                names.append(comp.component_type)
    return names


def build_module_prompt(
    lab_context: str,
    plan: LabOutline,
    index: int,
    registry: AssetRegistry,
    asset_plan: ModuleAssetPlan,
) -> str:
    allowed = allowed_new_component_names(asset_plan, registry)
    return MODULE_INSTRUCTION.format(
        lab_context=lab_context,
        plan=render_plan(plan, current=index),
        index=index,
        total=len(plan.scenes),
        scene=_render_scene(plan.scenes[index - 1]),
        previous=registry.render_previous_modules(),
        asset_plan=json.dumps(asset_plan.model_dump(mode="json", exclude_none=True), indent=2),
        catalog=registry.render_component_catalog(),
        module_name=asset_plan.module_name,
        allowed_new=", ".join(allowed) if allowed else "none",
    )


def build_revision_message(what: str, feedback: str) -> str:
    return REVISION_INSTRUCTION.format(what=what, feedback=feedback)
