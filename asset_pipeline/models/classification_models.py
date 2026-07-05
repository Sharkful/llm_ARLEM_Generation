from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

GeometryClass = Literal["composite", "parametric", "imported", "unclear"]

Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class CompositePartPlan(BaseModel):
    """One primitive part of a composite, with enough placement/appearance
    info that composite_baker.py can actually assemble and color it --
    added 2026-07 after discovering composite parts had never carried any
    of this: every part sat at the origin at unit scale with no material,
    so a 'composite' snowman would have rendered as N overlapping
    same-size gray spheres even after the classification itself was fixed.

    `bond_between` (added 2026-07, "methane ball-and-stick" incident):
    connecting two atoms with a stick requires rotating a cylinder to point
    from one 3D position to another -- real trigonometry, not something an
    LLM reliably gets right by guessing Euler angles (in practice it always
    returned rotation=[0,0,0], so every bond pointed the same fixed
    direction regardless of where its two atoms actually were, looking
    like a flat disc from most angles instead of a connecting rod). A part
    with bond_between set is a CONNECTOR: composite_baker.py computes its
    exact position, length, and rotation deterministically from the two
    named atoms' real final positions, so relative_position/relative_scale
    are ignored for it (only bond_thickness matters).
    """

    description: str = Field(description="Short phrase naming ONE primitive shape, e.g. 'large white sphere'.")
    color_hint: str = Field(description="Short color/material phrase for this part alone, e.g. 'matte white', 'glossy black', 'bright orange'.")
    label: str = Field(
        default="",
        description="A short stable name for THIS part ('C', 'H1', 'left_arm') so other "
        "parts can reference it via bond_between. Required on any part another part "
        "bonds to; optional otherwise.",
    )
    bond_between: Optional[list[str]] = Field(
        default=None,
        description="Set ONLY for a connector/stick/bond part: the `label`s of the "
        "exactly two OTHER parts (already listed, in either order) this part connects "
        "-- e.g. ['C', 'H1']. When set, this part is placed, sized, and rotated "
        "automatically to run exactly between those two parts' real positions; leave "
        "relative_position/relative_scale at their defaults and use bond_thickness "
        "instead. Use this for ANY strut connecting two named points -- bonds in a "
        "molecule, an axle between wheels, a strut in a truss -- never hand-compute a "
        "midpoint/rotation yourself.",
    )
    bond_thickness: float = Field(
        default=0.06,
        description="Radius of a connector part, as a fraction of the whole object's "
        "overall size. Only meaningful when bond_between is set.",
    )
    relative_position: Vec3 = Field(
        default=[0.0, 0.0, 0.0],
        description="Position as a fraction of the whole object's overall size, "
        "in the object's own local frame: [0,0,0] = center, +Y = up, +Z = forward. "
        "E.g. a hat sits around [0, 0.9, 0]; a left arm around [-0.6, 0.1, 0]. "
        "Ignored for a connector part (bond_between set).",
    )
    relative_scale: Vec3 = Field(
        default=[0.3, 0.3, 0.3],
        description="This part's size as a fraction of the whole object's overall "
        "largest dimension, per axis. A part 30% as wide/tall/deep as the whole "
        "object -> [0.3, 0.3, 0.3]. Thin parts (arms, brim) should be thin on the "
        "appropriate axis, e.g. [0.5, 0.06, 0.06] for a wide-reaching thin arm. "
        "Ignored for a connector part (bond_between set).",
    )


class GeometryClassification(BaseModel):
    """Structured instructor response for the harder routing decision --
    used only when tag/token catalog matching does not produce a confident
    catalog_match or variant (see pipeline/classifier.py).
    """

    geometry_class: GeometryClass
    reasoning: str = Field(
        description="One or two sentences explaining the classification, "
        "attached to the ResolutionRecord for human review when unclear."
    )
    composite_parts: list[CompositePartPlan] = Field(
        default_factory=list,
        description="If geometry_class is 'composite', one entry per "
        "sub-part with its own color and approximate placement/size. "
        "Empty otherwise.",
    )
    search_keywords: list[str] = Field(
        default_factory=list,
        description="If geometry_class is 'imported', 2-6 short IDENTITY "
        "phrases naming what the object actually is, for searching external "
        "3D asset libraries -- e.g. ['snowman', 'christmas figure'] or "
        "['leather armchair', 'vintage chair']. Never grammatical filler "
        "words ('friendly', 'made', 'from') and never words copied "
        "positionally from the description -- pick the words a person would "
        "actually type into a search box for this object. Empty otherwise.",
    )
