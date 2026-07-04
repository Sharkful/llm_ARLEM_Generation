"""Automatic relevance review of external sourcing candidates (added 2026-07,
user decision after the "Frosty the Snowman" incident: a keyword search for
a snowman surfaced NASA's "Vesta - Snowman Craters" -- an asteroid crater
formation nicknamed for its shape, not a snowman model -- because it shares
exactly one word with the query. Keyword scoring cannot tell the difference
between an object and something merely NAMED after it. This module asks the
LLM instead, the same way a human would glance at a search result and think
"wait, that's not what I asked for."

Two passes, cheapest first:
  1. TEXT review -- given the object description and the candidate's own
     name/context (no network cost beyond the LLM call), does this candidate
     plausibly depict the same object? This alone catches "Snowman Craters"
     immediately, since the LLM knows craters aren't snowmen regardless of
     the shared word.
  2. IMAGE review -- for candidates that survive step 1 (typically 1-3),
     fetch the actual thumbnail and ask the same question grounded in the
     picture. Catches the inverse failure: a misleading or generic title
     that a picture immediately clarifies one way or the other.

Image review requires a vision-capable provider (anthropic or openai);
falls back to text-only (with a note) for google or on any fetch/vision
failure -- this is a quality enhancement layered on the mandatory text
gate, not a hard requirement.
"""
from __future__ import annotations

import base64
import time
from typing import Optional

import instructor
import requests
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from llm.client_factory import get_instructor_client, resolve_provider_and_model
from models.log_models import LLMCallEntry

_TIMEOUT_S = 20
_VISION_PROVIDERS = {"anthropic", "openai"}

_TEXT_SYSTEM_PROMPT = """\
You review whether a candidate 3D model from an external library actually \
depicts the object a user asked for, before it's shown to them as a match.

Keyword search is naive: it will surface anything that shares a word with \
the request even when the underlying object is completely different -- a \
crater formation nicknamed "Snowman Craters" is not a snowman toy; a \
"Base" component of satellite hardware is not the base of a snowman's \
body. Titles can also be deceptively vague or use jargon. Judge the \
ACTUAL OBJECT the candidate depicts, not just word overlap with the \
request.

matches=true only if a reasonable person would say the candidate is the \
same kind of object as requested (or a very close stand-in). matches=false \
for coincidental word overlap, a component/part of something else, or an \
unrelated real-world thing that merely shares vocabulary with the request.
"""

_IMAGE_SYSTEM_PROMPT = """\
You are shown an image of a candidate 3D model alongside the text \
description of what a user actually asked for. Judge from the image \
whether this is really the requested object -- a misleading or vague \
title can be clarified by what the picture actually shows, and a \
plausible-sounding title can also be contradicted by it.

matches=true only if the picture is consistent with the requested object. \
matches=false if the image clearly shows something else, including a \
close-up of an unrelated part/component, terrain/geology, or an object \
merely named after a visual resemblance to the requested thing.
"""


class RelevanceReview(BaseModel):
    matches: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="One sentence explaining the verdict.")


class RelevanceReviewError(RuntimeError):
    pass


def _entry(provider: str, model: str, purpose: str, start: float, success: bool,
           error: str | None = None, usage=None) -> LLMCallEntry:
    duration = time.monotonic() - start
    input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    return LLMCallEntry(
        provider=provider, model=model, purpose=purpose,
        input_tokens=input_tokens, output_tokens=output_tokens,
        duration_seconds=round(duration, 3), success=success, error_message=error,
    )


@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(3), reraise=True)
def _call_text(client: instructor.Instructor, model: str, description: str, candidate_name: str, candidate_context: str):
    return client.chat.completions.create_with_completion(
        model=model,
        response_model=RelevanceReview,
        max_retries=2,
        messages=[
            {"role": "system", "content": _TEXT_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"User requested: {description}\n\n"
                f"Candidate name: {candidate_name}\n"
                f"Candidate context (source folder/category): {candidate_context}"
            )},
        ],
        max_tokens=512,
    )


def review_candidate_text(
    description: str,
    candidate_name: str,
    candidate_context: str = "",
    provider: str | None = None,
    model: str | None = None,
) -> tuple[RelevanceReview, LLMCallEntry]:
    resolved_provider, resolved_model = resolve_provider_and_model(provider, model)
    client = get_instructor_client(resolved_provider)
    start = time.monotonic()
    try:
        review, completion = _call_text(client, resolved_model, description, candidate_name, candidate_context)
    except Exception as exc:
        entry = _entry(resolved_provider, resolved_model, "source_relevance_text", start, False, str(exc))
        raise RelevanceReviewError(str(exc)) from exc
    return review, _entry(resolved_provider, resolved_model, "source_relevance_text", start, True,
                          usage=getattr(completion, "usage", None))


@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(3), reraise=True)
def _call_image(client: instructor.Instructor, provider: str, model: str, description: str, image_b64: str, media_type: str):
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
        response_model=RelevanceReview,
        max_retries=2,
        messages=[
            {"role": "system", "content": _IMAGE_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"User requested: {description}"},
                image_block,
            ]},
        ],
        max_tokens=512,
    )


def review_candidate_image(
    description: str,
    thumbnail_url: str,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[RelevanceReview, LLMCallEntry]:
    """Raises RelevanceReviewError if the provider doesn't support vision or
    the image can't be fetched -- callers should catch this and fall back
    to the text-only verdict rather than failing the whole review."""
    resolved_provider, resolved_model = resolve_provider_and_model(provider, model)
    if resolved_provider not in _VISION_PROVIDERS:
        raise RelevanceReviewError(
            f"Image review needs anthropic or openai; {resolved_provider!r} not supported here."
        )
    try:
        response = requests.get(thumbnail_url, timeout=_TIMEOUT_S)
        response.raise_for_status()
        media_type = response.headers.get("content-type", "image/png").split(";")[0]
        image_b64 = base64.b64encode(response.content).decode("ascii")
    except requests.RequestException as exc:
        raise RelevanceReviewError(f"Could not fetch thumbnail: {exc}") from exc

    client = get_instructor_client(resolved_provider)
    start = time.monotonic()
    try:
        review, completion = _call_image(
            client, resolved_provider, resolved_model, description, image_b64, media_type
        )
    except Exception as exc:
        raise RelevanceReviewError(str(exc)) from exc
    return review, _entry(resolved_provider, resolved_model, "source_relevance_image", start, True,
                          usage=getattr(completion, "usage", None))


class CandidateReview(BaseModel):
    matches: Optional[bool] = None  # None = review could not be completed
    confidence: float = 0.0
    reasoning: str = ""
    reviewed_with_image: bool = False


def review_candidates(
    description: str,
    candidates: list,  # list[SourceCandidate] -- typed loosely to avoid a circular import
    provider: str | None = None,
    model: str | None = None,
    use_vision: bool = True,
    max_image_reviews: int = 2,
) -> tuple[list[CandidateReview], list[LLMCallEntry]]:
    """One CandidateReview per candidate, same order as input. Never raises:
    an individual candidate whose review call fails is marked matches=None
    (treated as 'unreviewed, keep but don't trust for auto-adopt') so a
    transient API error can't silently hide every candidate.
    """
    logs: list[LLMCallEntry] = []
    reviews: list[CandidateReview] = []
    image_reviews_used = 0

    for candidate in candidates:
        try:
            text_review, entry = review_candidate_text(
                description, candidate.name,
                candidate_context=getattr(candidate, "source", ""),
                provider=provider, model=model,
            )
            logs.append(entry)
        except RelevanceReviewError:
            reviews.append(CandidateReview(matches=None, reasoning="Text review failed."))
            continue

        review = CandidateReview(
            matches=text_review.matches, confidence=text_review.confidence,
            reasoning=text_review.reasoning,
        )
        if (
            use_vision
            and text_review.matches
            and image_reviews_used < max_image_reviews
            and getattr(candidate, "thumbnail_url", "")
        ):
            try:
                image_review, entry = review_candidate_image(
                    description, candidate.thumbnail_url, provider=provider, model=model,
                )
                logs.append(entry)
                image_reviews_used += 1
                review = CandidateReview(
                    matches=image_review.matches, confidence=image_review.confidence,
                    reasoning=image_review.reasoning, reviewed_with_image=True,
                )
            except RelevanceReviewError:
                pass  # keep the text-only verdict
        reviews.append(review)
    return reviews, logs
