"""Stage 5: LLM -> MaterialDef JSON (+ optional procedural texture).

The LLM call here is cheap and structured (classification-style, like Stage
1's spec parsing) -- a good candidate for a lower-cost provider/model
override. Goes through llm/client_factory.py like every other LLM call in
this pipeline; never imports a provider SDK directly.

Procedural textures are synthesized locally with numpy + Pillow (value
noise, cellular craters, stripes, grids, gradients, rust blotches). This is
deliberately the whole texture story for v1: req. doc section 9 only
requires explicit representation of transparent/emissive/metallic/smooth
materials, and section 20.2 lists richer texturing as an open question.

Extension point: if photorealistic textures become necessary later, an
image-generation API call slots in where `synthesize_texture()` is invoked
in `generate_material()` -- no other stage needs to change, because
downstream consumers only ever see `MaterialDef.texture` as a relative path
under library/materials/.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import instructor
import numpy as np
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from llm.client_factory import get_instructor_client, resolve_provider_and_model
from models.catalog_models import MaterialDef
from models.generation_models import (
    MaterialGenerationResult,
    MaterialPlan,
    ProceduralTextureSpec,
)
from models.log_models import LLMCallEntry

_SYSTEM_PROMPT = """\
You turn a free-text description of a surface/material for an AR/VR \
educational scene into a structured material definition for the Unity URP \
render pipeline.

Rules:
- material.material_id: short snake_case identifier derived from the description (e.g. "dull_gray_rock", "transparent_glass")
- material.shader_family: keep the default "URP/Lit" unless the description clearly demands otherwise
- material.base_color: hex color like "#8a8a8a" capturing the dominant color
- material.alpha: 1.0 unless the material is transparent/translucent; pair alpha < 1.0 with transparent = true
- material.metallic: 0.0 for non-metals; 0.7-1.0 only for actual metal surfaces
- material.smoothness: 0.0-0.2 rough/matte, 0.4-0.6 typical, 0.8-1.0 polished/glassy
- material.emissive: true only if the material visibly glows or emits light
- material.texture: ALWAYS leave null -- the pipeline fills it in
- procedural_texture: set ONLY if the description implies a visible surface PATTERN
  (craters, stripes, rust, grid lines, mottling). A plain colored surface --
  even a rough or shiny one -- needs NO texture; roughness is carried by
  smoothness, not a pattern. Choose the closest supported kind:
  noise (mottled/speckled), craters, stripes, grid, gradient, rust.
- Do NOT invent details not present or clearly implied in the description
"""


class MaterialGenerationError(RuntimeError):
    def __init__(self, message: str, log_entry: LLMCallEntry):
        super().__init__(message)
        self.log_entry = log_entry


# ── Procedural texture synthesis (no LLM, no network -- pure numpy/PIL) ──

def _hex_to_rgb(hex_color: str) -> np.ndarray:
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", hex_color.strip())
    if not match:
        raise ValueError(f"Not a #rrggbb hex color: {hex_color!r}")
    value = match.group(1)
    return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def _value_noise(size: int, scale: float, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """Multi-octave value noise in [0, 1], PIL-bilinear-upsampled per octave."""
    field = np.zeros((size, size), dtype=np.float64)
    amplitude, total = 1.0, 0.0
    for octave in range(octaves):
        res = max(2, min(size, int(round(scale * (2**octave)))))
        coarse = rng.random((res, res))
        img = Image.fromarray((coarse * 255).astype(np.uint8)).resize(
            (size, size), Image.BILINEAR
        )
        field += amplitude * (np.asarray(img, dtype=np.float64) / 255.0)
        total += amplitude
        amplitude *= 0.5
    return field / total


def _blend(base: np.ndarray, accent: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Per-pixel lerp base->accent by weight in [0,1]; returns HxWx3 uint8."""
    return np.clip(
        base[None, None, :] * (1.0 - weight[..., None]) + accent[None, None, :] * weight[..., None],
        0, 255,
    ).astype(np.uint8)


def synthesize_texture(
    spec: ProceduralTextureSpec,
    out_path: Path,
    size: int | None = None,
    seed: int = 0,
) -> Path:
    """Render the requested pattern to a square RGB PNG. Deterministic per seed."""
    size = size or config.TEXTURE_SIZE
    rng = np.random.default_rng(seed)
    base = _hex_to_rgb(spec.base_color)
    accent = _hex_to_rgb(spec.accent_color)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) / size

    if spec.kind == "noise":
        weight = _value_noise(size, spec.scale, rng)
    elif spec.kind == "craters":
        # Cellular: darken radial bowls around scattered centers, with a
        # brighter rim just outside each bowl.
        weight = 0.15 * _value_noise(size, spec.scale, rng)
        count = max(3, int(spec.scale**2 / 4))
        centers = rng.random((count, 2))
        radii = rng.uniform(0.2, 1.0, count) * (0.5 / spec.scale)
        for (cy, cx), radius in zip(centers, radii):
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            bowl = np.clip(1.0 - dist / radius, 0.0, 1.0) ** 1.5
            rim = np.clip(1.0 - np.abs(dist - radius) / (radius * 0.3), 0.0, 1.0)
            weight = np.clip(weight + 0.8 * bowl - 0.3 * rim, 0.0, 1.0)
    elif spec.kind == "stripes":
        weight = (np.sin(xx * spec.scale * 2 * np.pi) > 0).astype(np.float64)
    elif spec.kind == "grid":
        period = 1.0 / spec.scale
        line = 0.08 * period
        weight = (
            ((xx % period) < line) | ((yy % period) < line)
        ).astype(np.float64)
    elif spec.kind == "gradient":
        weight = yy
    elif spec.kind == "rust":
        # Blotchy thresholded noise: mostly base metal with irregular rust
        # patches, plus fine speckle inside the patches.
        blotch = _value_noise(size, spec.scale / 2, rng)
        speckle = _value_noise(size, spec.scale * 2, rng)
        patches = np.clip((blotch - 0.45) / 0.25, 0.0, 1.0)
        weight = np.clip(patches * (0.6 + 0.4 * speckle), 0.0, 1.0)
    else:  # pragma: no cover -- Literal-typed, unreachable via validated specs
        raise ValueError(f"Unsupported texture kind: {spec.kind!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_blend(base, accent, weight), mode="RGB").save(out_path)
    return out_path


# ── LLM plan + write-through ─────────────────────────────────────────────

def generate_material(
    description: str,
    material_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    force: bool = False,
) -> tuple[MaterialGenerationResult, MaterialPlan, LLMCallEntry]:
    """Description -> MaterialPlan (LLM) -> material JSON + optional texture.

    `material_id`, when given, overrides whatever id the LLM proposed.
    Refuses to overwrite an existing material file unless `force` -- same
    no-silent-overwrite rule catalog_writer enforces for catalog entries.
    """
    resolved_provider, resolved_model = resolve_provider_and_model(provider, model)
    client = get_instructor_client(resolved_provider)

    start = time.monotonic()
    try:
        plan, completion = _call_with_usage(client, resolved_model, description)
    except Exception as exc:
        duration = time.monotonic() - start
        failed_entry = LLMCallEntry(
            provider=resolved_provider,
            model=resolved_model,
            purpose="material_generation",
            duration_seconds=round(duration, 3),
            success=False,
            error_message=str(exc),
        )
        raise MaterialGenerationError(str(exc), failed_entry) from exc
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
        purpose="material_generation",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        duration_seconds=round(duration, 3),
        success=True,
    )

    material = plan.material
    if material_id:
        material = material.model_copy(update={"material_id": material_id})
    final_id = material.material_id

    material_path = config.MATERIALS_DIR / f"{final_id}.json"
    if material_path.exists() and not force:
        return (
            MaterialGenerationResult(
                material_id=final_id,
                success=False,
                error_message=(
                    f"Material {final_id!r} already exists at {material_path}. "
                    "Pass force=True (--force) to overwrite."
                ),
            ),
            plan,
            entry,
        )

    texture_path: Path | None = None
    if plan.procedural_texture is not None:
        texture_path = config.TEXTURES_DIR / f"{final_id}.png"
        synthesize_texture(plan.procedural_texture, texture_path)
        material = material.model_copy(update={"texture": f"textures/{final_id}.png"})

    material_path.parent.mkdir(parents=True, exist_ok=True)
    material_path.write_text(
        json.dumps(material.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    return (
        MaterialGenerationResult(
            material_id=final_id,
            success=True,
            material_path=str(material_path),
            texture_path=str(texture_path) if texture_path else None,
        ),
        plan,
        entry,
    )


@retry(
    wait=wait_exponential(min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_with_usage(
    client: instructor.Instructor,
    llm_model: str,
    description: str,
) -> tuple[MaterialPlan, object]:
    response, completion = client.chat.completions.create_with_completion(
        model=llm_model,
        response_model=MaterialPlan,
        max_retries=3,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Create a material definition for this description:\n\n{description}",
            },
        ],
        max_tokens=1024,
    )
    return response, completion
