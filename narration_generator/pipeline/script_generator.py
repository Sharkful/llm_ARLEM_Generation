from __future__ import annotations

import time
from typing import Optional

import anthropic
import instructor
from tenacity import retry, stop_after_attempt, wait_exponential

from models.log_models import LLMCallEntry
from narration_definitions import ARLabNarrationScript, TTSRuntimeConfig
from script_definitions import ARSceneNarrationBrief

_SYSTEM_PROMPT = """\
You are a narration writer for augmented reality educational labs.

You receive a scene narration brief and produce a structured narration script in the TTS narration JSON format.

Rules:
- Write natural spoken narration — how a good instructor talks, not how a textbook reads
- Use pause segments (0.3–1.0 seconds) between natural breaks within a clip
- Do NOT start clips with "In this clip..." or similar meta-narration
- Keep each clip focused and concise — prefer 2–4 sentences per say segment
- Preserve the module structure and clip order from the brief exactly
- Include a clip title for each clip (used to generate audio filenames)
- Set default_voice only if instructed; otherwise omit it
- Style suggestions must be natural language, not enumerations
- Do NOT include filenames, AR anchors, timing fields, or course metadata other than pause seconds
"""


def _build_user_prompt(
    brief: ARSceneNarrationBrief,
    runtime_config: TTSRuntimeConfig,
    style_preset: Optional[str] = None,
) -> str:
    voices_str = ", ".join(runtime_config.allowed_voices) if runtime_config.allowed_voices else "none specified"
    default_voice_str = runtime_config.default_voice or "none specified"
    style_str = style_preset or "clear, friendly lecture for students"

    return (
        f"Here is the scene narration brief:\n\n"
        f"{brief.model_dump_json(indent=2)}\n\n"
        f"Available voices: {voices_str}\n"
        f"Default voice: {default_voice_str}\n"
        f"Style preference: {style_str}\n\n"
        f"Generate a complete narration script following the TTS narration JSON format."
    )


def generate_narration_script(
    brief: ARSceneNarrationBrief,
    runtime_config: TTSRuntimeConfig,
    llm_model: str,
    api_key: str,
    style_preset: Optional[str] = None,
) -> tuple[ARLabNarrationScript, LLMCallEntry]:
    client = instructor.from_anthropic(anthropic.Anthropic(api_key=api_key))

    start = time.monotonic()
    response, completion = _call_with_usage(client, llm_model, brief, runtime_config, style_preset)
    duration = time.monotonic() - start

    usage = completion.usage if hasattr(completion, "usage") else None
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    cached_tokens = getattr(usage, "cache_read_input_tokens", 0) if usage else 0

    entry = LLMCallEntry(
        model=llm_model,
        provider="anthropic",
        purpose="script_generation",
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
    brief: ARSceneNarrationBrief,
    runtime_config: TTSRuntimeConfig,
    style_preset: Optional[str],
) -> tuple[ARLabNarrationScript, object]:
    response, completion = client.chat.completions.create_with_completion(
        model=llm_model,
        response_model=ARLabNarrationScript,
        max_retries=3,
        messages=[
            {"role": "user", "content": _build_user_prompt(brief, runtime_config, style_preset)},
        ],
        system=_SYSTEM_PROMPT,
        max_tokens=8192,
    )
    return response, completion
