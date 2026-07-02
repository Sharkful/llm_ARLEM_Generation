"""Stage 1: turn a free-text object description into a validated AssetSpec.

Thin, cheap, classification-style LLM call -- a good candidate for a
lower-cost provider/model override (config- or call-site-controlled) even
though the pipeline's global default is Anthropic Sonnet. Goes through
llm/client_factory.py like every other LLM call in this pipeline; never
imports a provider SDK directly.
"""
from __future__ import annotations

import time

import instructor
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from llm.client_factory import get_instructor_client, resolve_provider_and_model
from models.log_models import LLMCallEntry
from models.spec_models import AssetSpec

_SYSTEM_PROMPT = """\
Youturn a free-text description of a physical object for an AR/VR educational \
scene into a structured asset specification. Your output is a request for a \
downstream asset-resolution pipeline -- NOT the final asset.

Rules:
- object_id: a short snake_case identifier derived from the description (e.g. "moon", "lab_table")
- description: the object description, lightly cleaned up but preserving the author's intent and detail
- kind: your best guess at the base geometric form (e.g. "sphere", "cylinder", "bracket", "composite", "unknown")
- semantic_type: the real-world category the object represents (e.g. "moon", "lab_table", "molecule", "atom_marker")
- desired_size_m: the largest dimension in meters, ONLY if a size is stated or clearly implied by real-world scale; otherwise omit it (leave null) -- do not guess sizes with no basis
- visual_style: a short phrase capturing appearance/material intent if stated (e.g. "dull gray, cratered", "transparent blue"); omit if not stated
- Do NOT invent details not present or clearly implied in the description
"""


def _build_user_prompt(description: str) -> str:
    return f"Create an asset specification for this object description:\n\n{description}"


def parse_description(
    description: str,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[AssetSpec, LLMCallEntry]:
    resolved_provider, resolved_model = resolve_provider_and_model(provider, model)
    client = get_instructor_client(resolved_provider)

    start = time.monotonic()
    try:
        response, completion = _call_with_usage(client, resolved_model, description)
    except Exception as exc:
        duration = time.monotonic() - start
        failed_entry = LLMCallEntry(
            provider=resolved_provider,
            model=resolved_model,
            purpose="spec_parsing",
            duration_seconds=round(duration, 3),
            success=False,
            error_message=str(exc),
        )
        raise SpecParsingError(str(exc), failed_entry) from exc
    duration = time.monotonic() - start

    usage = getattr(completion, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = (
        getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    )
    cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

    entry = LLMCallEntry(
        provider=resolved_provider,
        model=resolved_model,
        purpose="spec_parsing",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        duration_seconds=round(duration, 3),
        success=True,
    )
    return response, entry


class SpecParsingError(RuntimeError):
    def __init__(self, message: str, log_entry: LLMCallEntry):
        super().__init__(message)
        self.log_entry = log_entry


@retry(
    wait=wait_exponential(min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_with_usage(
    client: instructor.Instructor,
    llm_model: str,
    description: str,
) -> tuple[AssetSpec, object]:
    response, completion = client.chat.completions.create_with_completion(
        model=llm_model,
        response_model=AssetSpec,
        max_retries=3,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(description)},
        ],
        max_tokens=1024,
    )
    return response, completion
