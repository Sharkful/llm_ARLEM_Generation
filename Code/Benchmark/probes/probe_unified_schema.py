"""
Probe: can each provider return a valid Lab from the unified, discriminator-free
json_lab schema — with every provider on a FREE-decode (generate + validate/retry)
mode?

Context: json_lab.py was changed to (a) drop the `Field(discriminator="type")` on
the Component union (plain Pydantic smart-union now resolves on the `type` tag) and
(b) emit single-value Literals as JSON-Schema `enum` instead of `const` (via
ConstToEnumSchemaMixin) so Gemini's function-calling schema validator accepts it.

This lets us retire the flat `json_lab_gemini.py` twin AND put Gemini on the same
free-generate-and-retry footing as the others by using GENAI_TOOLS instead of the
constrained GENAI_STRUCTURED_OUTPUTS. The modes per provider here:

    OpenAI    -> Mode.TOOLS            (from_provider default; free + retry)
    Anthropic -> Mode.ANTHROPIC_TOOLS  (free + retry)
    Gemini    -> Mode.GENAI_TOOLS      (free + retry; NOT structured outputs)

Run from the project root with the venv active and OPENAI/ANTHROPIC/GEMINI keys in
.env:
    python "Code/Benchmark/probes/probe_unified_schema.py"
"""

import os
import sys
import time
from pathlib import Path

import instructor
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "Code" / "Schemas"))
sys.path.insert(0, str(PROJECT_ROOT / "Code" / "Benchmark"))
load_dotenv(PROJECT_ROOT / ".env")

from json_lab import Lab  # the unified, discriminator-free model
from prompt_builder import LabDescription, Level, Structure, build_prompt, discover_labs
from benchmark_config import SYSTEM_PROMPT

# (provider, model_id) — small capable model from each provider.
# Edit these to match the exact models you want to spend on.
MODELS = [
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("google", "gemini-2.5-flash-lite"),
    ("openai", "gpt-5.4-nano"),
]

LAB_KEY = "heart_anatomy_and_blood_flow"
LEVEL = Level.L3
MAX_RETRIES = 3


def make_client(provider: str, model_id: str):
    """One free-decode client per provider (no constrained decoding anywhere)."""
    if provider == "anthropic":
        import anthropic
        import httpx

        # Explicit timeout disables the SDK's non-streaming guard so max_tokens
        # can exceed ~21,333 without switching to streaming (mirrors benchmark.py).
        client = anthropic.Anthropic(timeout=httpx.Timeout(900.0, connect=5.0))
        return instructor.from_anthropic(
            client, model=model_id, mode=instructor.Mode.ANTHROPIC_TOOLS, max_tokens=32000
        )
    if provider == "google":
        return instructor.from_provider(
            f"google/{model_id}",
            mode=instructor.Mode.GENAI_TOOLS,  # free + retry, the path we're validating
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        )
    if provider == "openai":
        return instructor.from_provider(f"openai/{model_id}")  # default Mode.TOOLS
    raise ValueError(provider)


def main() -> None:
    labs = discover_labs()
    key = LAB_KEY if LAB_KEY in labs else sorted(labs)[0]
    lab = LabDescription.from_yaml(labs[key])
    prompt, _model = build_prompt(
        lab, LEVEL, "json_lab", structure=Structure.MULTI_MODULE,
        min_objects=4, min_clips=5,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    print(f"instructor {instructor.__version__} | lab '{key}' | level {LEVEL.value} | "
          f"unified json_lab.Lab | max_retries={MAX_RETRIES}\n")

    rows = []
    for provider, model_id in MODELS:
        print(f"--- {provider}/{model_id} ---")
        kwargs = dict(response_model=Lab, messages=messages, max_retries=MAX_RETRIES)
        if provider == "anthropic":
            kwargs["max_tokens"] = 32000

        for attempt in range(3):  # outer loop: ride out transient 503s (not schema errors)
            t0 = time.time()
            try:
                client = make_client(provider, model_id)
                result = client.create(**kwargs)
                dt = time.time() - t0
                mods = result.modules
                n_clip = sum(len(getattr(m, "clips", []) or []) for m in mods)
                n_obj = sum(len(getattr(m, "objects", []) or []) for m in mods)
                detail = f"{len(mods)} modules / {n_clip} clips / {n_obj} objects"
                rows.append((provider, model_id, "OK", detail, dt))
                print(f"  OK  {detail} ({dt:.1f}s)\n")
                break
            except Exception as e:
                dt = time.time() - t0
                msg = str(e)
                transient = any(s in msg for s in ("503", "UNAVAILABLE", "overloaded", "high demand"))
                if transient and attempt < 2:
                    print(f"  transient {type(e).__name__}; retrying in 8s...")
                    time.sleep(8)
                    continue
                rows.append((provider, model_id, "FAIL", f"{type(e).__name__}: {msg}"[:200], dt))
                print(f"  FAIL  {type(e).__name__}: {msg[:200]} ({dt:.1f}s)\n")
                break

    print("=" * 72)
    print(f"{'PROVIDER':<10} {'MODEL':<28} {'RESULT':<6} DETAIL")
    print("-" * 72)
    for provider, model_id, status, detail, dt in rows:
        print(f"{provider:<10} {model_id:<28} {status:<6} {detail}")


if __name__ == "__main__":
    main()
