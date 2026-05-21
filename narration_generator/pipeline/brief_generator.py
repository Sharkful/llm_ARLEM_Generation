from __future__ import annotations

import time
from typing import Optional

import anthropic
import instructor
from tenacity import retry, stop_after_attempt, wait_exponential

from models.log_models import LLMCallEntry
from script_definitions import ARSceneNarrationBrief

_SYSTEM_PROMPT = """\
You are an instructional designer creating a scene narration brief for an augmented reality lab.

Your output is source information for a downstream narrator — NOT final narration text.

Rules:
- scene: describe what the learner sees in the AR scene (not what to say about it)
- objectives: 2-6 learning objectives as short imperative statements
- modules: ordered groups of clips; each clip has change and meaning
- change: describe what happens visually or interactively in the scene
- meaning: explain the instructional purpose — why this change matters for learning
- Do NOT write narration text, voice names, filenames, TTS settings, or timing
- Keep each clip focused on one distinct scene change
- Group related clips into modules (most labs have 2-4 modules)
"""


def _build_user_prompt(description: str) -> str:
    return f"Create a scene narration brief for this AR lab description:\n\n{description}"


def generate_scene_brief(
    description: str,
    llm_model: str,
    api_key: str,
) -> tuple[ARSceneNarrationBrief, LLMCallEntry]:
    client = instructor.from_anthropic(anthropic.Anthropic(api_key=api_key))

    start = time.monotonic()
    response, completion = _call_with_usage(client, llm_model, description)
    duration = time.monotonic() - start

    usage = completion.usage if hasattr(completion, "usage") else None
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    cached_tokens = getattr(usage, "cache_read_input_tokens", 0) if usage else 0

    entry = LLMCallEntry(
        model=llm_model,
        provider="anthropic",
        purpose="brief_generation",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        duration_seconds=round(duration, 3),
        success=True,
    )
    return response, entry


@retry(
    wait=wait_exponential(min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_with_usage(
    client: instructor.Instructor,
    llm_model: str,
    description: str,
) -> tuple[ARSceneNarrationBrief, object]:
    response, completion = client.chat.completions.create_with_completion(
        model=llm_model,
        response_model=ARSceneNarrationBrief,
        max_retries=3,
        messages=[
            {"role": "user", "content": _build_user_prompt(description)},
        ],
        system=_SYSTEM_PROMPT,
        max_tokens=4096,
    )
    return response, completion
