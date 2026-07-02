from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GeometryClass = Literal["composite", "parametric", "imported", "unclear"]


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
    composite_parts: list[str] = Field(
        default_factory=list,
        description="If geometry_class is 'composite', short descriptions of "
        "each sub-part (e.g. '1 large red sphere', '2 small white spheres'). "
        "Empty otherwise.",
    )
