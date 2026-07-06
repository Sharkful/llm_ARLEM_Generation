"""Stage 2.5: turn a composite classification into a composite fragment.

A composite is NOT new geometry -- it's a list of existing (or recursively
resolved) asset_ids with relative transforms, stored as a JSON fragment in
library/composites/<id>.json. Each sub-part goes back through classify()
(recursively) so a composite of composites, or a composite containing a
part that itself needs generation, is handled the same way a top-level
spec would be.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import instructor
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from llm.client_factory import get_instructor_client, resolve_provider_and_model
from models.classification_models import CompositePartPlan
from models.log_models import LLMCallEntry
from models.spec_models import AssetSpec


class CompositePart(BaseModel):
    part_id: str
    asset_spec: AssetSpec
    resolved_asset_id: str | None = None
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    # Added 2026-07 alongside CompositePartPlan: every part needs its own
    # color, or composite_baker.py has nothing to paint it with and the
    # whole assembly ends up uniformly gray/default.
    color_hint: str = ""
    material_id: str | None = None
    # Added 2026-07, "methane ball-and-stick" incident: a connector part
    # (bond_between set) is positioned/sized/rotated by composite_baker.py's
    # deterministic vector math, not by `position`/`scale` above -- an LLM
    # cannot reliably compute the rotation to point a cylinder from one 3D
    # position to another (see CompositePartPlan's docstring).
    label: str = ""
    bond_between: list[str] | None = None
    bond_thickness: float = 0.06


class CompositeFragment(BaseModel):
    composite_id: str
    display_name: str
    source_description: str
    parts: list[CompositePart]


# Primitive keyword -> semantic_type hint (added 2026-07). Composite parts
# are, by the classifier's HARD RULE, always exactly one named primitive
# shape ("large white sphere", "thin gray cylinder") -- so the word naming
# that primitive is already the identity word the catalog matcher's
# semantic_type shortcut needs for an instant, LLM-free catalog_match
# against the seed primitives (see pipeline/catalog_matcher.py score_entry).
# Without this hint, a short generic phrase like "large white sphere" barely
# overlaps sphere_basic's tag list on raw token count and falls through to a
# second LLM classification call per part -- for a 10-12 part composite
# (a snowman, say) that's 10-12 extra calls, and each one risks routing an
# ordinary sphere to 'parametric'/'unclear' instead of just reusing the seed
# primitive that already exists for exactly this purpose.
_PRIMITIVE_KEYWORDS = [
    "sphere", "cylinder", "cone", "torus", "capsule", "cube", "box", "plane", "quad",
]

# Common ways to describe a primitive without using its catalog tag word
# (a flat brim/disk has no dedicated primitive -- it's a squashed cylinder).
_PRIMITIVE_ALIASES = {
    "disk": "cylinder", "disc": "cylinder", "puck": "cylinder", "washer": "cylinder",
    "ball": "sphere", "orb": "sphere", "globe": "sphere",
    "rod": "cylinder", "tube": "cylinder", "post": "cylinder", "pipe": "cylinder",
    "block": "cube", "brick": "cube", "slab": "cube",
    "ring": "torus", "donut": "torus", "doughnut": "torus",
    "pill": "capsule",
}


def _guess_primitive_kind(part_description: str) -> str | None:
    lowered = part_description.lower()
    for keyword in _PRIMITIVE_KEYWORDS:
        if re.search(rf"\b{keyword}s?\b", lowered):  # tolerate plurals ("2 white spheres")
            return keyword
    for alias, keyword in _PRIMITIVE_ALIASES.items():
        if re.search(rf"\b{alias}s?\b", lowered):
            return keyword
    return None


def build_part_specs(parent_object_id: str, composite_parts: list[CompositePartPlan]) -> list[AssetSpec]:
    """Turn the classifier's structured part plans into AssetSpecs.

    This does not call an LLM -- part descriptions are already short,
    LLM-authored phrases (from GeometryClassification.composite_parts);
    wrapping each in a minimal AssetSpec is enough for the resolver to
    recursively classify/resolve it. When the phrase names one of the seed
    primitives, kind/semantic_type are set to that word so the part
    catalog-matches immediately (see _guess_primitive_kind above); left
    unset otherwise, since a phrase with no primitive word doesn't cleanly
    map to one without further guessing.
    """
    specs = []
    for i, plan in enumerate(composite_parts):
        primitive_kind = _guess_primitive_kind(plan.description)
        specs.append(
            AssetSpec(
                object_id=f"{parent_object_id}_part{i}",
                description=plan.description,
                kind=primitive_kind,
                semantic_type=primitive_kind,
            )
        )
    return specs


def composite_path(composite_id: str) -> Path:
    return config.LIBRARY_DIR / "composites" / f"{composite_id}.json"


def save_composite_fragment(fragment: CompositeFragment) -> Path:
    path = composite_path(fragment.composite_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fragment.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def load_composite_fragment(composite_id: str) -> CompositeFragment | None:
    path = composite_path(composite_id)
    if not path.exists():
        return None
    return CompositeFragment.model_validate(json.loads(path.read_text(encoding="utf-8")))


# ── Forced decomposition (manual redirect, added 2026-07) ────────────────
#
# Used when a human overrides a stuck 'imported'/'unclear' draft with
# "build this as a composite instead" (asset_factory.redirect_draft). The
# original classifier already said this couldn't be done -- the human is
# asserting otherwise, so this call is deliberately permissive (higher part
# budget, no "choose composite only if..." hedging) rather than re-running
# the same judgment call that already said no.

class _ForcedDecomposition(BaseModel):
    parts: list[CompositePartPlan] = Field(
        description="One entry per primitive part, each with its own color "
        "and approximate placement/size, that together approximate the "
        "object as a hand-built assembly."
    )


_FORCED_SYSTEM_PROMPT = """\
A human has decided this object should be built as an assembly of simple \
primitive shapes (spheres, cubes, cylinders, cones, tori), even though it \
may be organic or detailed -- give your best approximation, not a refusal. \
List each part as a short phrase naming exactly one primitive (e.g. "large \
white sphere", "thin black cylinder", "small orange cone"), with its own \
color_hint and an approximate relative_position/relative_scale so the \
assembled result actually resembles the object's real proportions instead \
of every part overlapping at the origin. Use as many parts as needed, up \
to the given budget, to capture the object's recognizable silhouette and \
key features.

For any strut/rod/bond that must connect two other named points (e.g. a \
molecule's bonds, an axle between wheels): give the two endpoint parts a \
short `label` each and set the connector's `bond_between: [labelA, labelB]` \
instead of guessing its position/rotation yourself -- the geometry engine \
computes the exact placement deterministically from the two labeled parts' \
real positions.
"""


def decompose_into_primitives(
    spec: AssetSpec,
    max_parts: int = 12,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[list[CompositePartPlan], LLMCallEntry]:
    resolved_provider, resolved_model = resolve_provider_and_model(provider, model)
    client = get_instructor_client(resolved_provider)

    start = time.monotonic()
    try:
        response, completion = _call_decompose(
            client, resolved_model, spec, max_parts
        )
    except Exception as exc:
        duration = time.monotonic() - start
        entry = LLMCallEntry(
            provider=resolved_provider, model=resolved_model,
            purpose="composite_decomposition", duration_seconds=round(duration, 3),
            success=False, error_message=str(exc),
        )
        raise RuntimeError(f"Forced composite decomposition failed: {exc}") from exc
    duration = time.monotonic() - start

    usage = getattr(completion, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    entry = LLMCallEntry(
        provider=resolved_provider, model=resolved_model,
        purpose="composite_decomposition", input_tokens=input_tokens,
        output_tokens=output_tokens, duration_seconds=round(duration, 3), success=True,
    )
    return response.parts[:max_parts], entry


@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(3), reraise=True)
def _call_decompose(client: instructor.Instructor, llm_model: str, spec: AssetSpec, max_parts: int):
    return client.chat.completions.create_with_completion(
        model=llm_model,
        response_model=_ForcedDecomposition,
        max_retries=2,
        messages=[
            {"role": "system", "content": _FORCED_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Object description: {spec.description}\n"
                f"Maximum parts: {max_parts}"
            )},
        ],
        # See classifier.py's identical comment -- up to 12 full
        # CompositePartPlans plus reasoning does not reliably fit in 1024.
        max_tokens=16384,
    )


# ── Revision from visual review feedback (added 2026-07, "methane sticks
#    don't connect" incident) ─────────────────────────────────────────────
#
# pipeline/visual_review.py renders the baked composite and asks a vision
# LLM whether it matches the original description; when it doesn't, this
# turns that specific complaint into a corrected part list -- the same
# structured output decompose_into_primitives produces, but grounded in
# what was ACTUALLY built (not a fresh guess) plus the reviewer's exact
# complaint, so a working part isn't discarded just because a different
# part was wrong.

def fragment_to_part_plans(fragment: CompositeFragment) -> list[CompositePartPlan]:
    """The current built state of a composite, in the same shape the
    classifier/decomposer produces, so it can be shown back to an LLM as
    'here is what currently exists' for a revision pass."""
    return [
        CompositePartPlan(
            description=part.asset_spec.description,
            color_hint=part.color_hint,
            label=part.label,
            bond_between=part.bond_between,
            bond_thickness=part.bond_thickness,
            relative_position=part.position,
            relative_scale=part.scale,
        )
        for part in fragment.parts
    ]


class _CompositeRevision(BaseModel):
    parts: list[CompositePartPlan] = Field(
        description="The corrected FULL part list (not just the changed parts) -- "
        "keep every part that the complaint doesn't implicate, fix the ones it does."
    )


_REVISION_SYSTEM_PROMPT = """\
A composite object (an assembly of simple primitive shapes) was built and \
rendered, then reviewed against its original description. The review found \
a specific problem. You are given the object's description, the CURRENT \
part list exactly as built, and the reviewer's complaint.

Output a corrected FULL part list that fixes the complaint. Keep every part \
that the complaint does not implicate unchanged (same description, \
color_hint, label, position/scale) -- do not regenerate parts that were \
already correct just because you're revising the object. Follow the same \
rules the parts were originally built under: every part is exactly one \
primitive shape with its own color_hint; a connector (bond/strut/axle) uses \
`label`/`bond_between` to reach exactly between two other named parts \
instead of a hand-computed position/rotation -- never compute a connector's \
position or rotation directly, geometry engine handles that from the two \
labeled parts' real positions.
"""


def revise_composite_parts(
    spec: AssetSpec,
    current_parts: list[CompositePartPlan],
    feedback: str,
    max_parts: int = 12,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[list[CompositePartPlan], LLMCallEntry]:
    resolved_provider, resolved_model = resolve_provider_and_model(provider, model)
    client = get_instructor_client(resolved_provider)

    start = time.monotonic()
    try:
        response, completion = _call_revise(
            client, resolved_model, spec, current_parts, feedback, max_parts
        )
    except Exception as exc:
        duration = time.monotonic() - start
        entry = LLMCallEntry(
            provider=resolved_provider, model=resolved_model,
            purpose="composite_revision", duration_seconds=round(duration, 3),
            success=False, error_message=str(exc),
        )
        raise RuntimeError(f"Composite revision failed: {exc}") from exc
    duration = time.monotonic() - start

    usage = getattr(completion, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    entry = LLMCallEntry(
        provider=resolved_provider, model=resolved_model,
        purpose="composite_revision", input_tokens=input_tokens,
        output_tokens=output_tokens, duration_seconds=round(duration, 3), success=True,
    )
    return response.parts[:max_parts], entry


@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(3), reraise=True)
def _call_revise(
    client: instructor.Instructor, llm_model: str, spec: AssetSpec,
    current_parts: list[CompositePartPlan], feedback: str, max_parts: int,
):
    current_json = json.dumps([p.model_dump(mode="json") for p in current_parts], indent=2)
    return client.chat.completions.create_with_completion(
        model=llm_model,
        response_model=_CompositeRevision,
        max_retries=2,
        messages=[
            {"role": "system", "content": _REVISION_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Object description: {spec.description}\n\n"
                f"Current parts (as built):\n{current_json}\n\n"
                f"Reviewer's complaint: {feedback}\n\n"
                f"Maximum parts: {max_parts}"
            )},
        ],
        # Output alone can be as large as the decompose call's (see above);
        # this call also echoes the current parts back in the prompt, but
        # that's input tokens, not output -- max_tokens governs the output.
        max_tokens=16384,
    )
