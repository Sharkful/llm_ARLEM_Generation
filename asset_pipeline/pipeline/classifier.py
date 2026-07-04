"""Stage 2: decide which asset class an AssetSpec belongs to.

Decision tree per offline_ar_asset_pipeline_requirements.md section 7:
  1. Confident catalog match (>= threshold)          -> catalog_match
  2. Close match but wrong color/scale/transparency   -> variant
  3. Names 2+ known primitives assembled together     -> composite
  4. Implies mechanical/geometric structure           -> parametric
  5. Implies organic/complex/realistic form            -> imported
  6. Ambiguous                                         -> flagged_for_review

Steps 1-2 are pure catalog-matcher lookups (no LLM). Steps 3-6 require
semantic judgment an LLM is better suited to than string heuristics; that
call is isolated behind `classify_geometry` (a plain function parameter),
so the branching logic in `classify()` is unit-testable with a fake
classifier and never needs a live API key to verify routing correctness.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import instructor
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from llm.client_factory import get_instructor_client, resolve_provider_and_model
from models.catalog_models import AssetCatalogEntry
from models.classification_models import GeometryClassification
from models.log_models import LLMCallEntry
from models.spec_models import AssetSpec, ResolvedAssetRef
from pipeline.catalog_matcher import CatalogMatch, find_candidates

# Below this, a catalog match is not even close enough to call a "variant".
VARIANT_FLOOR = 0.35

_SYSTEM_PROMPT = """\
You classify an AR/VR educational asset request that did NOT match anything \
in the existing asset catalog closely enough to reuse.

Choose exactly one geometry_class:
- "composite": the object is an assembly of 2+ simple recognizable primitive \
shapes (spheres, cubes, cylinders, cones, etc.) at relative positions -- e.g. \
a water molecule, coordinate axes, a solar system diagram, a simple snowman \
or robot built from stacked/attached primitives. List each sub-part briefly \
in composite_parts. HARD RULE: every composite_part must be ONE primitive \
shape describable in a short phrase naming that primitive ("large white \
sphere", "thin gray cylinder", "small orange cone") -- never itself an \
assembly, never a phrase that would need further decomposition. If the \
object cannot be expressed as 2-12 single-primitive parts, do NOT choose \
composite; choose parametric or imported instead. Prefer composite over \
imported whenever a recognizable approximation is possible with primitives \
-- a crude hand-built composite of the right identity (a snowman made of \
three spheres, stick arms, and a hat) is a far better educational AR asset \
than an unrelated real-world scan from an external library, which is \
unlikely to exist for whimsical/fictional subjects anyway.
- "parametric": the object has clear mechanical/geometric structure that a \
CSG modeling script (extrude, boolean union/subtract, simple shapes) could \
reasonably build -- e.g. a bracket, stand, dial, cutaway box, clamp.
- "imported": the object is organic, highly detailed, or realistic in a way \
neither a CSG script nor a primitive composite can reasonably approximate \
-- e.g. real furniture, realistic lab equipment, real spacecraft, plants, \
detailed characters that need more than 12 primitive parts. When you choose \
this, also fill search_keywords with the identity words a person would \
search an external 3D asset library for.
- "unclear": you cannot confidently pick one of the above from the \
description given.

Always explain your choice in reasoning (1-2 sentences) -- this is shown to \
a human reviewer when the classification is uncertain.
"""


class ClassificationError(RuntimeError):
    def __init__(self, message: str, log_entry: LLMCallEntry):
        super().__init__(message)
        self.log_entry = log_entry


def _build_user_prompt(spec: AssetSpec, top_candidates: list[CatalogMatch]) -> str:
    lines = [f"Object description: {spec.description}"]
    if spec.kind:
        lines.append(f"Guessed kind: {spec.kind}")
    if spec.semantic_type:
        lines.append(f"Semantic type: {spec.semantic_type}")
    if spec.visual_style:
        lines.append(f"Visual style: {spec.visual_style}")
    if top_candidates:
        lines.append("\nClosest catalog entries considered (none matched well enough):")
        for m in top_candidates:
            lines.append(f"  - {m.entry.asset_id} (confidence {m.confidence:.2f})")
    else:
        lines.append("\nNo catalog entries matched at all.")
    return "\n".join(lines)


def classify_geometry(
    spec: AssetSpec,
    top_candidates: list[CatalogMatch],
    provider: str | None = None,
    model: str | None = None,
) -> tuple[GeometryClassification, LLMCallEntry]:
    """Default LLM-backed classifier for steps 3-6 of the decision tree."""
    resolved_provider, resolved_model = resolve_provider_and_model(provider, model)
    client = get_instructor_client(resolved_provider)

    start = time.monotonic()
    try:
        response, completion = _call_with_usage(
            client, resolved_model, spec, top_candidates
        )
    except Exception as exc:
        duration = time.monotonic() - start
        failed_entry = LLMCallEntry(
            provider=resolved_provider,
            model=resolved_model,
            purpose="classification",
            duration_seconds=round(duration, 3),
            success=False,
            error_message=str(exc),
        )
        raise ClassificationError(str(exc), failed_entry) from exc
    duration = time.monotonic() - start

    usage = getattr(completion, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = (
        getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    )

    entry = LLMCallEntry(
        provider=resolved_provider,
        model=resolved_model,
        purpose="classification",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_seconds=round(duration, 3),
        success=True,
    )
    return response, entry


@retry(wait=wait_exponential(min=4, max=60), stop=stop_after_attempt(5), reraise=True)
def _call_with_usage(
    client: instructor.Instructor,
    llm_model: str,
    spec: AssetSpec,
    top_candidates: list[CatalogMatch],
) -> tuple[GeometryClassification, object]:
    return client.chat.completions.create_with_completion(
        model=llm_model,
        response_model=GeometryClassification,
        max_retries=3,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(spec, top_candidates)},
        ],
        max_tokens=1024,
    )


ClassifierFn = Callable[
    [AssetSpec, list[CatalogMatch]], tuple[GeometryClassification, Optional[LLMCallEntry]]
]


def classify(
    spec: AssetSpec,
    catalog: list[AssetCatalogEntry],
    classify_geometry_fn: ClassifierFn | None = None,
    match_threshold: float | None = None,
) -> tuple[ResolvedAssetRef | None, GeometryClassification | None, list[LLMCallEntry]]:
    """Route an AssetSpec through the decision tree.

    Returns (resolved_asset_ref, geometry_classification, log_entries).
    resolved_asset_ref is set for catalog_match/variant (steps 1-2).
    geometry_classification is set when steps 3-6 ran (composite/parametric/
    imported/unclear) -- the caller (resolver.py) turns "composite" into a
    ResolvedAssetRef only after composite_builder.py has actually built and
    registered the composite fragment.
    """
    threshold = match_threshold if match_threshold is not None else config.CATALOG_MATCH_CONFIDENCE_THRESHOLD
    log_entries: list[LLMCallEntry] = []

    candidates = find_candidates(spec, catalog)

    if candidates and candidates[0].confidence >= threshold:
        top = candidates[0]
        return (
            ResolvedAssetRef(
                asset_id=top.entry.asset_id,
                resolution_method="catalog_match",
                confidence=top.confidence,
                warnings=[],
            ),
            None,
            log_entries,
        )

    if candidates and candidates[0].confidence >= VARIANT_FLOOR:
        top = candidates[0]
        return (
            ResolvedAssetRef(
                asset_id=top.entry.asset_id,
                resolution_method="variant",
                confidence=top.confidence,
                warnings=[
                    f"Closest catalog match {top.entry.asset_id!r} scored "
                    f"{top.confidence:.2f}, below the {threshold:.2f} catalog_match "
                    "threshold -- treated as a variant (material/scale change, no new geometry)."
                ],
            ),
            None,
            log_entries,
        )

    # No confident catalog match or variant -- ask the geometry classifier
    # to distinguish composite / parametric / imported / unclear.
    classifier_fn = classify_geometry_fn or classify_geometry
    geometry, entry = classifier_fn(spec, candidates)
    if entry is not None:
        log_entries.append(entry)

    return None, geometry, log_entries
