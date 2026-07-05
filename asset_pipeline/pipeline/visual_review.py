"""Automated visual review of a generated/baked asset against its original
description (added 2026-07, "methane sticks don't connect" incident).

The methane ball-and-stick bug (disconnected sticks, ungraded atom colors)
was visible in a single screenshot -- a human looked at the render and
immediately knew it was wrong. Nothing in the pipeline did that check
automatically; a broken result looked exactly as "successful" to the
pipeline as a correct one, right up until a human happened to notice. This
module closes that gap: render the asset's current GLB, show it to a
vision-capable LLM alongside the original description, and ask the same
question a human reviewer would ask -- "does this actually look like what
was requested?" -- with a concrete, actionable complaint when it doesn't,
so asset_factory.review_and_repair() can feed that complaint back into a
repair pass (regenerate()'s tweak flow for parametric, a part-list revision
for composite) instead of a human having to describe the bug from scratch.

Same two-tier reasoning as source_relevance.py's candidate review (that
module judges "is this external candidate the right object"; this one
judges "does this specific render match the request"), but this is always
an image review -- there's no cheap text-only pass, since the whole point
is judging what was actually built, not what was intended.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import instructor
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from llm.client_factory import get_instructor_client, resolve_provider_and_model
from models.log_models import LLMCallEntry

_VISION_PROVIDERS = {"anthropic", "openai"}

_SYSTEM_PROMPT = """\
You review a rendered 3D AR/educational asset against the description a \
user originally asked for -- the same way a human would look at the result \
and immediately say "that's not right" or "yes, that's it."

Check specifically for:
- Structural correctness: parts that should visibly connect or touch (a \
stick between two atoms, a wheel on an axle, a handle on a door) actually \
reaching each other, not floating disconnected or overlapping wrongly.
- Color/material correctness: parts that were supposed to have distinct, \
specific colors (a snowman's black hat, a molecule's CPK atom colors) \
actually showing those colors, not uniformly gray/default or the wrong hue.
- Overall recognizability: does the silhouette/arrangement actually read as \
the requested object, at roughly the right proportions.

matches=true only if a reasonable person would say the render is a correct \
(even if simple/stylized) depiction of the request. matches=false for any \
structural defect (disconnected/misplaced parts), missing/wrong colors, or \
an arrangement that doesn't read as the requested object.

When matches=false, suggested_fix must be a specific, actionable instruction \
for regenerating this asset correctly -- e.g. "the four hydrogen-carbon \
bonds don't reach the hydrogen atoms; recompute each bond's length and \
rotation from the actual atom positions" or "the hat, eyes, and nose are all \
the same gray as the body; give each its own distinct color as originally \
requested (black hat, black eyes, orange nose)." Never vague ("looks wrong").
"""


class AssetVisualReview(BaseModel):
    matches: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="One or two sentences explaining the verdict.")
    suggested_fix: str = Field(
        default="",
        description="Only when matches=false: a specific, actionable instruction "
        "for what to change to make the render correct. Empty when matches=true.",
    )


class VisualReviewError(RuntimeError):
    pass


def _entry(provider: str, model: str, start: float, success: bool,
           error: str | None = None, usage=None) -> LLMCallEntry:
    duration = time.monotonic() - start
    input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    return LLMCallEntry(
        provider=provider, model=model, purpose="visual_review",
        input_tokens=input_tokens, output_tokens=output_tokens,
        duration_seconds=round(duration, 3), success=success, error_message=error,
    )


@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(3), reraise=True)
def _call(client: instructor.Instructor, provider: str, model: str,
          description: str, image_b64: str, media_type: str):
    if provider == "anthropic":
        image_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": image_b64},
        }
    else:  # openai
        image_block = {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
        }
    return client.chat.completions.create_with_completion(
        model=model,
        response_model=AssetVisualReview,
        max_retries=2,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Originally requested: {description}"},
                image_block,
            ]},
        ],
        max_tokens=768,
    )


def review_asset_render(
    description: str,
    image_path: Path,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[AssetVisualReview, LLMCallEntry]:
    """Raises VisualReviewError if the provider doesn't support vision or the
    image can't be read -- this is an opt-in review step, so callers should
    surface the error rather than silently skipping it."""
    resolved_provider, resolved_model = resolve_provider_and_model(provider, model)
    if resolved_provider not in _VISION_PROVIDERS:
        raise VisualReviewError(
            f"Visual review needs anthropic or openai; {resolved_provider!r} not supported here."
        )
    if not image_path.is_file():
        raise VisualReviewError(f"No render found at {image_path}.")

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    media_type = "image/png"

    client = get_instructor_client(resolved_provider)
    start = time.monotonic()
    try:
        review, completion = _call(
            client, resolved_provider, resolved_model, description, image_b64, media_type
        )
    except Exception as exc:
        raise VisualReviewError(str(exc)) from exc
    return review, _entry(resolved_provider, resolved_model, start, True,
                          usage=getattr(completion, "usage", None))
