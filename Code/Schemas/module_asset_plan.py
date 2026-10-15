"""
Module Asset Plan Pydantic Model

Response model for the hierarchical pipeline's per-module planning step
(Code/Hierarchical/). Before a DemoModule is generated, the model lists the
scene objects, components, and interactions that module needs — and *why* —
plus a one-line beat per clip. The approved plan is then fed into the module
generation step.

Plain models (no Literal tags, no discriminated unions), so the schema works on
every provider including Gemini.

Validation reports *every* problem in one error, so a bad plan costs one reask
rather than one per problem.

Catalog validation context
--------------------------
Whether a planned component may be new, which component names already exist
(built-ins plus new components invented by earlier modules), and the per-module
minimums are per-run state. The pipeline supplies them through
``asset_plan_validation(...)``, a ContextVar scope, rather than instructor's
``context=`` kwarg: passing ``context=`` also makes instructor render every
message as a Jinja template, which would mangle prompts that embed JSON or
model-written text. Without an active scope the catalog and minimum checks are
skipped, so the model stays usable on its own.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from _schema_helpers import to_pascal_case
from json_lab import Color4, ComponentInput


# NewComponent.scriptName's constraint (json_lab.py), restated for new components
# planned here so a planned name is always a valid scriptName later.
_SCRIPT_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9]{4,}$")


def component_key(name: str) -> str:
    """Comparison key for component names: casing and separators ignored, so
    'pulse_glow', 'pulseGlow' and 'PulseGlow' are the same component."""
    return to_pascal_case(name.strip()).lower()


# ── Validation context ────────────────────────────────────────────────

@dataclass(frozen=True)
class ComponentCatalogContext:
    """Per-call state for ``ModuleAssetPlan`` validation.

    catalog: names of every component that already exists — built-in component
        class names plus new-component scriptNames registered by earlier modules.
    allow_new: whether this plan may propose new components at all.
    min_objects / min_clip_beats: per-module minimums for this run.
    """
    catalog: frozenset[str] = field(default_factory=frozenset)
    allow_new: bool = True
    min_objects: int = 1
    min_clip_beats: int = 2

    def find(self, name: str) -> Optional[str]:
        """Return the catalog's spelling of ``name`` (see ``component_key``), or None."""
        key = component_key(name)
        return next((e for e in self.catalog if component_key(e) == key), None)


_CATALOG_CONTEXT: ContextVar[Optional[ComponentCatalogContext]] = ContextVar(
    "module_asset_plan_catalog_context", default=None
)


@contextmanager
def asset_plan_validation(
    catalog: Iterable[str],
    *,
    allow_new: bool = True,
    min_objects: int = 1,
    min_clip_beats: int = 2,
) -> Iterator[ComponentCatalogContext]:
    """Activate catalog and minimum checks for ``ModuleAssetPlan`` validation."""
    ctx = ComponentCatalogContext(
        catalog=frozenset(catalog), allow_new=allow_new,
        min_objects=min_objects, min_clip_beats=min_clip_beats,
    )
    token = _CATALOG_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CATALOG_CONTEXT.reset(token)


# ── Models ────────────────────────────────────────────────────────────

class PlannedObject(BaseModel):
    """A scene object this module needs."""

    name: str = Field(
        description="Unique object name within this module; the module step must reuse it exactly"
    )
    prefab: str = Field(
        description=(
            "Prefab to instantiate. Prefer a known built-in prefab or one already "
            "invented by an earlier module; otherwise invent a descriptive name."
        )
    )
    texture: Optional[str] = Field(
        default=None,
        description="Texture to apply, if any. Prefer known or previously invented textures.",
    )
    color: Optional[Color4] = Field(
        default=None, description="Optional RGBA tint/color, 0-255 each"
    )
    purpose: str = Field(
        min_length=15,
        description="Why this object is in the module: what the student should see or learn from it",
    )
    is_new_asset: bool = Field(
        description="True if the prefab or texture is not a known built-in and was not invented by an earlier module"
    )


class NewComponentSpec(BaseModel):
    """A new component this module introduces, declared once here. Every object
    that uses it attaches it by name in ``components``."""

    name: str = Field(
        description=(
            "New PascalCase script name of at least 5 letters or digits, e.g. "
            "'PulseGlow'. It must not be a catalog component."
        )
    )
    behavior_description: str = Field(
        min_length=20,
        description=(
            "The ONE behavior this new component implements, its inputs (object "
            "references, inspector values) and its outputs/effects. Keep it small "
            "and single-purpose."
        ),
    )
    why_existing_insufficient: str = Field(
        min_length=15,
        description="Why no catalog component (or combination of them) works",
    )
    inputs: list[ComponentInput] = Field(
        description=(
            "The component's inspector fields: object references and values the "
            "module sets on each instance. Use an empty list only if it needs none."
        ),
    )

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, v):
        # Same normalization as json_lab.NewComponent.scriptName, so a declared
        # name is exactly the scriptName the module step will use.
        return to_pascal_case(v)


class PlannedComponent(BaseModel):
    """A component attached to one of this module's planned objects."""

    object_name: str = Field(
        description="Name of the planned object (from `objects`) this component attaches to"
    )
    component_type: str = Field(
        description=(
            "Name of the component: a catalog component exactly as listed (e.g. "
            "'SimpleOrbitComponent', or a script name invented by an earlier module), "
            "or the name of a component declared in this plan's `new_components`."
        )
    )
    rationale: str = Field(
        min_length=15,
        description="Why this object needs this component: what it lets the student see or do",
    )


class PlannedInteraction(BaseModel):
    """Something the student does in this module, tied to one planned object."""

    object_name: str = Field(
        description="Name of the planned object (from `objects`) the student interacts with"
    )
    action: str = Field(
        min_length=5,
        description="What the student does, e.g. 'drags the Mars marker along its orbit'",
    )
    effect: str = Field(
        min_length=5,
        description="What happens as a result, e.g. 'the sight line to Mars updates'",
    )


class ModuleAssetPlan(BaseModel):
    """The asset and interaction plan for one module, written before the module itself."""

    module_name: str = Field(description="Display name of the module being planned")
    design_notes: str = Field(
        min_length=40,
        description=(
            "Reason first: what must the student see and do in this module to meet "
            "its purpose, and what does that require of the scene?"
        ),
    )
    objects: list[PlannedObject] = Field(
        min_length=1, description="Every scene object the module needs"
    )
    new_components: list[NewComponentSpec] = Field(
        default_factory=list,
        description=(
            "New components this module introduces, each declared exactly once. Leave "
            "empty when catalog components can provide every behavior."
        ),
    )
    components: list[PlannedComponent] = Field(
        default_factory=list,
        description=(
            "Every component attachment: one entry per (object, component) pair, naming "
            "a catalog component or one declared in new_components, each with a rationale"
        ),
    )
    interactions: list[PlannedInteraction] = Field(
        default_factory=list,
        description="Student interactions in this module, each tied to a planned object",
    )
    clip_beats: list[str] = Field(
        min_length=2,
        description="One short line per clip, in order: what happens in the scene and what is narrated",
    )

    @model_validator(mode="after")
    def _validate_references(self) -> "ModuleAssetPlan":
        problems: list[str] = []
        ctx = _CATALOG_CONTEXT.get()

        names: set[str] = set()
        for obj in self.objects:
            if obj.name in names:
                problems.append(f"Duplicate planned object name '{obj.name}'.")
            names.add(obj.name)
        known = sorted(names)

        if ctx is not None:
            if len(self.objects) < ctx.min_objects:
                problems.append(
                    f"Plan at least {ctx.min_objects} scene objects for this module "
                    f"(got {len(self.objects)})."
                )
            if len(self.clip_beats) < ctx.min_clip_beats:
                problems.append(
                    f"Plan at least {ctx.min_clip_beats} clip beats for this module "
                    f"(got {len(self.clip_beats)})."
                )

        for inter in self.interactions:
            if inter.object_name not in names:
                problems.append(
                    f"Interaction '{inter.action}' targets unknown object "
                    f"'{inter.object_name}'; object_name must be one of {known}."
                )

        # New components are declared once in new_components; components only
        # attaches them (or catalog components) to objects.
        declared: dict[str, str] = {}
        for spec in self.new_components:
            label = f"New component '{spec.name}'"
            key = component_key(spec.name)
            if key in declared:
                problems.append(
                    f"{label} is declared more than once in new_components; declare it "
                    "once and attach it to every object that uses it in components."
                )
                continue
            declared[key] = spec.name
            input_names = [i.name for i in spec.inputs]
            if dupes := sorted({n for n in input_names if input_names.count(n) > 1}):
                problems.append(f"{label}: input name(s) {dupes} are listed more than once.")
            if spec.name.strip() == "NewComponent":
                problems.append(
                    f"{label}: give the new component its own script name, not the "
                    "generic 'NewComponent'."
                )
            elif not _SCRIPT_NAME_RE.match(spec.name):
                problems.append(
                    f"{label}: a new component name must be PascalCase letters and "
                    "digits, start with a capital letter, and be at least 5 characters."
                )
            if ctx is None:
                continue
            if not ctx.allow_new:
                problems.append(
                    f"{label}: new components are disabled for this run; leave "
                    f"new_components empty and use catalog components: {sorted(ctx.catalog)}."
                )
            elif (existing := ctx.find(spec.name)) is not None:
                problems.append(
                    f"{label}: '{existing}' already exists in the component catalog; "
                    "remove it from new_components and attach the catalog component by name."
                )

        seen_pairs: set[tuple[str, str]] = set()
        for comp in self.components:
            label = f"'{comp.component_type}' on '{comp.object_name}'"
            if comp.object_name not in names:
                problems.append(
                    f"Component {label}: unknown object; object_name must be one of {known}."
                )
            pair = (comp.object_name, component_key(comp.component_type))
            if pair in seen_pairs:
                problems.append(f"Component {label} is planned more than once.")
            seen_pairs.add(pair)

            if comp.component_type.strip() == "NewComponent":
                problems.append(
                    f"Component {label}: component_type must not be the generic "
                    "'NewComponent'. Name a catalog component or one declared in "
                    "new_components."
                )
                continue
            if ctx is None or pair[1] in declared or ctx.find(comp.component_type):
                continue
            hint = (
                ", or declare it in new_components if no catalog component can "
                "provide the behavior" if ctx.allow_new else ""
            )
            problems.append(
                f"Component {label}: unknown component. Use a catalog component "
                f"{sorted(ctx.catalog)}{hint}."
            )

        if problems:
            raise ValueError(
                f"{len(problems)} problem(s) in this asset plan; fix all of them:\n- "
                + "\n- ".join(problems)
            )
        return self
