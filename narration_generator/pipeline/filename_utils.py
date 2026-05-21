from __future__ import annotations

import re
from typing import Optional

from narration_definitions import ARLabNarrationScript, NarrationClip


def make_safe_slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    text = text.strip("_")
    return text[:50]


def make_clip_filename(
    module_index: int,
    clip_index: int,
    clip_title: Optional[str] = None,
) -> str:
    base = f"m{module_index:02d}_c{clip_index:03d}"
    if clip_title:
        slug = make_safe_slug(clip_title)
        if slug:
            base = f"{base}_{slug}"
    return f"{base}.mp3"


def enumerate_clips(
    script: ARLabNarrationScript,
) -> list[tuple[int, int, NarrationClip, str]]:
    """Return (module_idx, clip_idx, clip, filename) for every clip, 1-based."""
    results = []
    for m_idx, module in enumerate(script.modules, start=1):
        for c_idx, clip in enumerate(module.clips, start=1):
            fname = make_clip_filename(m_idx, c_idx, clip.title)
            results.append((m_idx, c_idx, clip, fname))
    return results
