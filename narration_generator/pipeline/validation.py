from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from narration_definitions import ARLabNarrationScript, TTSRuntimeConfig


@dataclass
class ValidationResult:
    level: Literal["error", "warning", "info"]
    message: str
    clip_ref: Optional[str] = None


def validate_narration_script(
    script: ARLabNarrationScript,
    runtime_config: TTSRuntimeConfig,
) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    voices_to_check: list[tuple[str, Optional[str]]] = []

    if script.default_voice:
        voices_to_check.append((script.default_voice, None))

    for m_idx, module in enumerate(script.modules, start=1):
        for c_idx, clip in enumerate(module.clips, start=1):
            clip_ref = f"m{m_idx:02d}_c{c_idx:03d}"

            if clip.voice:
                voices_to_check.append((clip.voice, clip_ref))

            if not clip.title:
                results.append(ValidationResult(
                    level="warning",
                    message="Clip has no title — filename will use module/clip index only",
                    clip_ref=clip_ref,
                ))

            say_segments = [s for s in clip.segments if s.type == "say"]
            total_text = " ".join(s.text for s in say_segments)

            if len(total_text) < 20:
                results.append(ValidationResult(
                    level="warning",
                    message=f"Very short clip text ({len(total_text)} chars) — consider merging with adjacent clip",
                    clip_ref=clip_ref,
                ))

            for seg in clip.segments:
                if seg.type == "say" and len(seg.text) > 800:
                    results.append(ValidationResult(
                        level="warning",
                        message=f"Say segment is very long ({len(seg.text)} chars) — consider splitting",
                        clip_ref=clip_ref,
                    ))
                if seg.type == "pause" and seg.seconds > 3.0:
                    results.append(ValidationResult(
                        level="warning",
                        message=f"Pause of {seg.seconds}s is unusually long",
                        clip_ref=clip_ref,
                    ))

    if runtime_config.allowed_voices:
        allowed = set(runtime_config.allowed_voices)
        for voice, ref in voices_to_check:
            if voice not in allowed:
                results.append(ValidationResult(
                    level="error",
                    message=f"Voice '{voice}' not in allowed voices: {sorted(allowed)}",
                    clip_ref=ref,
                ))

    return results
