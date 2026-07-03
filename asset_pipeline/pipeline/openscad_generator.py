"""Stage 3: parametric generation via OpenSCAD.

Flow: instructor call -> OpenSCADPlan -> write .scad -> shell out to the
OpenSCAD CLI -> STL. On compile failure, feed OpenSCAD's stderr back to the
LLM in a repair prompt for up to REPAIR_ATTEMPT_LIMIT attempts, logging
every attempt (never silently retry forever). On success, measure the
actual STL bounding box and compare to the LLM's own expected_bounds_m;
divergence beyond BOUNDS_DIVERGENCE_TOLERANCE is flagged rather than
trusted, since the LLM's stated expectation is a claim, not a fact.

The compile step (compile_scad) takes no LLM dependency and is exercised
directly by tests/integration against the real OpenSCAD binary; the
generation loop (generate_parametric_asset) takes the LLM call as an
injectable function so its retry/repair branching is unit-testable without
an API key, same pattern as classifier.py.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import instructor
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from llm.client_factory import get_instructor_client, resolve_provider_and_model
from models.generation_models import CompileAttempt, GenerationResult, OpenSCADPlan
from models.log_models import LLMCallEntry
from models.spec_models import AssetSpec

REPAIR_ATTEMPT_LIMIT = 3
BOUNDS_DIVERGENCE_TOLERANCE = 0.20  # 20%, per the implementation plan
OPENSCAD_TIMEOUT_SECONDS = 60

_SYSTEM_PROMPT = """\
You write OpenSCAD scripts for simple mechanical/geometric lab-equipment \
parts for an AR/VR educational scene (brackets, stands, dials, cutaway \
boxes, clamps, etc.).

Conventions to follow:
- The model must be centered on the origin unless the object clearly stands \
on a surface (e.g. a stand/base), in which case its bottom should sit at Z=0.
- Units are meters. Keep the script's own working units as meters -- do not \
scale by 1000 for millimeters.
- +Y is up, +Z is forward, matching this pipeline's other primitive assets.
- Use only stable OpenSCAD 2021.01 language features: cube(), sphere(), \
cylinder(), polyhedron(), linear_extrude(), rotate_extrude(), \
translate/rotate/scale, union/difference/intersection, and simple \
for-loops. Do not use experimental/dev-snapshot-only features.
- Expose the object's key dimensions as top-level variables assigned from \
the `parameters` dict you return, referenced by name in the geometry (e.g. \
`width = 0.05; ... cube([width, height, depth]);`) so a human can tweak them \
later without regenerating the whole script.
- Report expected_bounds_m as your best estimate of the [x, y, z] extent in \
meters of the geometry you wrote, in the same axis order as canonical_bounds_m \
elsewhere in this pipeline.
"""

_REPAIR_SYSTEM_SUFFIX = """\

Your previous OpenSCAD script failed to compile. Fix the script so it \
compiles with OpenSCAD 2021.01. Keep the same overall design intent.
"""

_REVISION_SYSTEM_SUFFIX = """\

You previously wrote an OpenSCAD script for this object. The human reviewer \
wants a change. Apply their instruction to the existing script -- keep \
everything they did not ask to change, keep the same parameter-variable \
style, and return the full revised script with updated parameters and \
expected_bounds_m.
"""


def _build_revision_prompt(previous_source: str, instruction: str) -> str:
    return (
        "Current OpenSCAD script:\n\n"
        f"```\n{previous_source}\n```\n\n"
        f"Reviewer's change request: {instruction}\n\n"
        "Return the full revised script."
    )


class OpenSCADGenerationError(RuntimeError):
    def __init__(self, message: str, log_entry: LLMCallEntry):
        super().__init__(message)
        self.log_entry = log_entry


def _build_user_prompt(spec: AssetSpec) -> str:
    lines = [f"Object description: {spec.description}"]
    if spec.kind:
        lines.append(f"Guessed kind: {spec.kind}")
    if spec.desired_size_m:
        lines.append(f"Approximate largest dimension: {spec.desired_size_m} m")
    if spec.visual_style:
        lines.append(f"Visual style notes: {spec.visual_style}")
    return "\n".join(lines)


def _build_repair_prompt(previous_source: str, stderr: str) -> str:
    return (
        "This OpenSCAD script failed to compile:\n\n"
        f"```\n{previous_source}\n```\n\n"
        f"OpenSCAD error output:\n{stderr[-2000:]}\n\n"
        "Return a corrected version of the full script."
    )


def generate_openscad_plan(
    spec: AssetSpec,
    provider: str | None = None,
    model: str | None = None,
    repair_context: tuple[str, str] | None = None,
    revision_context: tuple[str, str] | None = None,
) -> tuple[OpenSCADPlan, LLMCallEntry]:
    """One LLM call: AssetSpec (+ optional context) -> OpenSCADPlan.

    repair_context   = (previous_source, compiler_stderr) -- fix a broken script.
    revision_context = (previous_source, human_instruction) -- apply a
                       reviewer's change request (Stage 8b iterate loop).
    """
    if repair_context and revision_context:
        raise ValueError("repair_context and revision_context are mutually exclusive")
    purpose = (
        "openscad_repair" if repair_context
        else "openscad_revision" if revision_context
        else "openscad_generation"
    )
    resolved_provider, resolved_model = resolve_provider_and_model(provider, model)
    client = get_instructor_client(resolved_provider)

    start = time.monotonic()
    try:
        response, completion = _call_with_usage(
            client, resolved_model, spec, repair_context, revision_context
        )
    except Exception as exc:
        duration = time.monotonic() - start
        failed_entry = LLMCallEntry(
            provider=resolved_provider,
            model=resolved_model,
            purpose=purpose,
            duration_seconds=round(duration, 3),
            success=False,
            error_message=str(exc),
        )
        raise OpenSCADGenerationError(str(exc), failed_entry) from exc
    duration = time.monotonic() - start

    usage = getattr(completion, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = (
        getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    )

    entry = LLMCallEntry(
        provider=resolved_provider,
        model=resolved_model,
        purpose=purpose,
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
    repair_context: tuple[str, str] | None,
    revision_context: tuple[str, str] | None = None,
) -> tuple[OpenSCADPlan, object]:
    system_prompt = _SYSTEM_PROMPT
    if repair_context is not None:
        system_prompt = _SYSTEM_PROMPT + _REPAIR_SYSTEM_SUFFIX
        previous_source, stderr = repair_context
        user_content = _build_repair_prompt(previous_source, stderr)
    elif revision_context is not None:
        system_prompt = _SYSTEM_PROMPT + _REVISION_SYSTEM_SUFFIX
        previous_source, instruction = revision_context
        user_content = (
            _build_user_prompt(spec)
            + "\n\n"
            + _build_revision_prompt(previous_source, instruction)
        )
    else:
        user_content = _build_user_prompt(spec)

    return client.chat.completions.create_with_completion(
        model=llm_model,
        response_model=OpenSCADPlan,
        max_retries=3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=4096,
    )


def compile_scad(scad_source: str, scad_path: Path, stl_path: Path) -> tuple[bool, str]:
    """Write scad_source to scad_path and compile it to stl_path.

    Returns (success, stderr). Success is determined by the STL file
    actually existing afterward, not just the subprocess return code --
    OpenSCAD's exit codes are not consistently reliable across versions,
    the same reasoning prefab_to_glb.py applies to its Blender subprocess
    calls.
    """
    if not config.OPENSCAD_BIN:
        raise RuntimeError(
            "OpenSCAD binary not found. Install OpenSCAD or set ASSET_PIPELINE_OPENSCAD_BIN."
        )

    scad_path.parent.mkdir(parents=True, exist_ok=True)
    scad_path.write_text(scad_source, encoding="utf-8")

    stl_path.parent.mkdir(parents=True, exist_ok=True)
    if stl_path.exists():
        stl_path.unlink()

    try:
        result = subprocess.run(
            [config.OPENSCAD_BIN, "-o", str(stl_path), str(scad_path)],
            capture_output=True,
            text=True,
            timeout=OPENSCAD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, f"OpenSCAD timed out after {OPENSCAD_TIMEOUT_SECONDS}s"

    if stl_path.is_file() and stl_path.stat().st_size > 0:
        return True, result.stderr
    return False, result.stderr or result.stdout


def measure_stl_bounds_m(stl_path: Path) -> list[float]:
    from stl import mesh

    m = mesh.Mesh.from_file(str(stl_path))
    extent = (m.max_ - m.min_).tolist()
    return [round(v, 6) for v in extent]


GenerateFn = Callable[
    [AssetSpec, Optional[tuple[str, str]]], tuple[OpenSCADPlan, Optional[LLMCallEntry]]
]


def generate_parametric_asset(
    spec: AssetSpec,
    asset_id: str,
    generate_plan_fn: GenerateFn | None = None,
) -> tuple[GenerationResult, list[LLMCallEntry]]:
    """Full Stage 3 loop: plan -> compile -> repair-on-failure -> bounds check.

    generate_plan_fn defaults to a real LLM call (generate_openscad_plan)
    but can be swapped for a fake in tests so the repair-loop branching is
    verified without an API key.
    """
    plan_fn = generate_plan_fn or generate_openscad_plan
    log_entries: list[LLMCallEntry] = []
    attempts: list[CompileAttempt] = []

    asset_dir = config.LIBRARY_DIR / "generated" / asset_id
    scad_path = asset_dir / "source.scad"
    stl_path = asset_dir / "source.stl"

    repair_context: tuple[str, str] | None = None
    plan: OpenSCADPlan | None = None

    for attempt_number in range(1, REPAIR_ATTEMPT_LIMIT + 1):
        plan, entry = plan_fn(spec, repair_context)
        if entry is not None:
            log_entries.append(entry)

        success, stderr = compile_scad(plan.scad_source, scad_path, stl_path)
        attempts.append(
            CompileAttempt(
                attempt_number=attempt_number,
                scad_source=plan.scad_source,
                success=success,
                stderr=stderr,
            )
        )

        if success:
            break
        repair_context = (plan.scad_source, stderr)
    else:
        return (
            GenerationResult(
                asset_id=asset_id,
                success=False,
                scad_path=str(scad_path),
                repair_attempts=attempts,
                error_message=f"OpenSCAD compile failed after {REPAIR_ATTEMPT_LIMIT} attempts.",
            ),
            log_entries,
        )

    actual_bounds = measure_stl_bounds_m(stl_path)
    bounds_diverge = _bounds_diverge(actual_bounds, plan.expected_bounds_m, BOUNDS_DIVERGENCE_TOLERANCE)

    result = GenerationResult(
        asset_id=asset_id,
        success=True,
        scad_path=str(scad_path),
        stl_path=str(stl_path),
        actual_bounds_m=actual_bounds,
        expected_bounds_m=plan.expected_bounds_m,
        bounds_diverge=bounds_diverge,
        repair_attempts=attempts,
    )
    return result, log_entries


def _bounds_diverge(actual: list[float], expected: list[float], tolerance: float) -> bool:
    for a, e in zip(actual, expected):
        if e <= 0:
            if a > tolerance:
                return True
            continue
        if abs(a - e) / e > tolerance:
            return True
    return False
