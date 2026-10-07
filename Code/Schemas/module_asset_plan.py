"""
Module Asset Plan Pydantic Model

Response model for the hierarchical pipeline's per-module planning step
(Code/Hierarchical/). Before a DemoModule is generated, the model lists the
scene objects, components, and interactions that module needs — and *why* —
plus a one-line beat per clip. The approved plan is then fed into the module
generation step.

Plain models (no Literal tags, no discriminated unions), so the schema works on
every provider including Gemini.

Catalog validation context
--------------------------
Whether a planned component may be new, and which component names already exist
(built-ins plus new components invented by earlier modules), is per-run state.
The pipeline supplies it through ``asset_plan_validation(...)``, a ContextVar
scope, rather than instructor's ``context=`` kwarg: passing ``context=`` also
makes instructor render every message as a Jinja template, which would mangle
prompts that embed JSON or model-written text. Without an active scope the
catalog checks are skipped, so the model stays usable on its own.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from pydantic import BaseModel, Field, model_validator

from json_lab import Color4


# NewComponent.scriptName's constraint (json_lab.py), restated for new components
# planned here so a planned name is always a valid scriptName later.
_SCRIPT_NAME_RE = re.compile(r"^\w{5,}$")


# ── Validation context ────────────────────────────────────────────────

@dataclass(frozen=True)
class ComponentCatalogContext:
    """Per-call catalog state for ``ModuleAssetPlan`` validation.

    catalog: names of every component that already exists — built-in component
        class names plus new-component scriptNames registered by earlier modules.
    allow_new: whether this plan may propose new components at all.
    """
    catalog: frozenset[str] = field(default_factory=frozenset)
    allow_new: bool = True

    def find(self, name: str) -> Optional[str]:
        """Return the catalog's spelling of ``name`` (case-insensitive), or None."""
        key = name.strip().lower()
        for entry in self.catalog:
            if entry.lower() == key:
                return entry
        return None


_CATALOG_CONTEXT: ContextVar[Optional[ComponentCatalogContext]] = ContextVar(
    "module_asset_plan_catalog_context", default=None
)


@contextmanager
def asset_plan_validation(
    catalog: Iterable[str], *, allow_new: bool = True
) -> Iterator[ComponentCatalogContext]:
    """Activate catalog checks for ``ModuleAssetPlan`` validation in this scope."""
    ctx = ComponentCatalogContext(catalog=frozenset(catalog), allow_new=allow_new)
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


class PlannedComponent(BaseModel):
    """A component to attach to one of this module's planned objects."""

    object_name: str = Field(
        description="Name of the planned object (from `objects`) this component attaches to"
    )
    component_type: str = Field(
        description=(
            "Name of the component. For an existing component use its catalog name "
            "exactly (e.g. 'SimpleOrbitComponent', or a scriptName invented by an "
            "earlier module). For a new component (is_new=true), give a new lowercase "
            "script name of at least 5 word characters, e.g. 'pulseglow'."
        )
    )
    rationale: str = Field(
        min_length=15,
        description="Why this object needs this component: what it lets the student see or do",
    )
    is_new: bool = Field(
        default=False,
        description="True only when no catalog component can provide the needed behavior",
    )
    behavior_description: Optional[str] = Field(
        default=None,
        description=(
            "Required when is_new: the ONE behavior this new component implements, "
            "its inputs (object references, inspector values) and its outputs/effects. "
            "Keep it small and single-purpose."
        ),
    )
    why_existing_insufficient: Optional[str] = Field(
        default=None,
        description="Required when is_new: why no catalog component (or combination) works",
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
    components: list[PlannedComponent] = Field(
        default_factory=list,
        description="Components to attach to the planned objects, each with a rationale",
    )
    interactions: list[str] = Field(
        default_factory=list,
        description="Student interactions in this module (what they do, with which object, and what happens)",
    )
    clip_beats: list[str] = Field(
        min_length=2,
        description="One short line per clip, in order: what happens in the scene and what is narrated",
    )

    @model_validator(mode="after")
    def _validate_references(self) -> "ModuleAssetPlan":
        names: set[str] = set()
        for obj in self.objects:
            if obj.name in names:
                raise ValueError(f"Duplicate planned object name '{obj.name}'")
            names.add(obj.name)

        ctx = _CATALOG_CONTEXT.get()
        # A new component may be attached to several objects; only one entry needs
        # is_new=true, the others may reference it by name.
        planned_new = {c.component_type.strip().lower() for c in self.components if c.is_new}
        for comp in self.components:
            if comp.object_name not in names:
                raise ValueError(
                    f"Component '{comp.component_type}' is attached to unknown object "
                    f"'{comp.object_name}'. object_name must match a planned object: "
                    f"{sorted(names)}"
                )

            if comp.component_type.strip() == "NewComponent":
                raise ValueError(
                    "component_type must not be the generic 'NewComponent'. Name an "
                    "existing catalog component, or set is_new=true and give the new "
                    "component its own script name."
                )

            if comp.is_new:
                if not _SCRIPT_NAME_RE.match(comp.component_type):
                    raise ValueError(
                        f"New component name '{comp.component_type}' must be at least 5 "
                        "word characters (letters, digits, underscore), no spaces"
                    )
                if not comp.behavior_description or not comp.why_existing_insufficient:
                    raise ValueError(
                        f"New component '{comp.component_type}' needs both "
                        "behavior_description and why_existing_insufficient"
                    )

            if ctx is None:
                continue
            existing = ctx.find(comp.component_type)
            if comp.is_new:
                if not ctx.allow_new:
                    raise ValueError(
                        f"New components are disabled for this run; replace "
                        f"'{comp.component_type}' with a catalog component: "
                        f"{sorted(ctx.catalog)}"
                    )
                if existing is not None:
                    raise ValueError(
                        f"'{comp.component_type}' already exists in the component "
                        f"catalog; reuse '{existing}' with is_new=false instead of "
                        "inventing it again"
                    )
            elif existing is None and comp.component_type.strip().lower() not in planned_new:
                hint = (
                    " or set is_new=true if no catalog component can provide the behavior"
                    if ctx.allow_new else ""
                )
                raise ValueError(
                    f"Unknown component '{comp.component_type}'. Use a catalog "
                    f"component {sorted(ctx.catalog)}{hint}."
                )
        return self
