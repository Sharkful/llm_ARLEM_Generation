"""
Probe: can each provider return a valid ARLEMScenario from the unified, twin-free
ARLEM schemas (full + simplified) on a FREE-decode (generate + validate/retry) mode?

Context: arlem_full.py and arlem_simplified.py were unified to drop their Gemini-only
twins. Every model now subclasses ConstToEnumSchemaMixin (single-value Literals emit
JSON-Schema `enum` instead of `const`) and HttpUrl was replaced with `str`, so Gemini's
function-calling schema validator accepts the schema on GENAI_TOOLS — putting ARLEM on
the same free-generate-and-retry footing as json_lab. The modes per provider:

    OpenAI    -> Mode.TOOLS            (from_provider default; free + retry)
    Anthropic -> Mode.ANTHROPIC_TOOLS  (free + retry)
    Gemini    -> Mode.GENAI_TOOLS      (free + retry; NOT structured outputs)

This is the GATE for retiring arlem_full_gemini.py / arlem_simplified_gemini.py: all
three providers must come back OK for both ARLEM specs before the twins are deleted.

Run from the project root with the venv active and OPENAI/ANTHROPIC/GEMINI keys in .env:
    python "Code/Testing/probe_arlem_unified.py"
"""

import os
import sys
import time
from pathlib import Path

import instructor
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Code" / "Schemas"))
sys.path.insert(0, str(PROJECT_ROOT / "Code" / "Testing"))
load_dotenv(PROJECT_ROOT / ".env")

from prompt_builder import LabDescription, Level, build_prompt, discover_labs
from benchmark_config import SYSTEM_PROMPT

# (provider, model_id) — small capable model from each provider.
MODELS = [
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("google", "gemini-2.5-flash-lite"),
    ("openai", "gpt-5.4-nano"),
]

# Both ARLEM specs must pass for the twins to be retired.
SPECS = ["arlem", "arlem_simple"]

LAB_KEY = "heart_anatomy_and_blood_flow"
LEVEL = Level.L2
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

    print(f"instructor {instructor.__version__} | lab '{key}' | level {LEVEL.value} | "
          f"unified ARLEM schemas | max_retries={MAX_RETRIES}\n")

    rows = []
    for spec in SPECS:
        prompt, response_model = build_prompt(lab, LEVEL, spec)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for provider, model_id in MODELS:
            print(f"--- {spec} | {provider}/{model_id} ({response_model.__name__}) ---")
            kwargs = dict(response_model=response_model, messages=messages, max_retries=MAX_RETRIES)
            if provider == "anthropic":
                kwargs["max_tokens"] = 32000

            for attempt in range(3):  # outer loop: ride out transient 503s (not schema errors)
                t0 = time.time()
                try:
                    client = make_client(provider, model_id)
                    result = client.create(**kwargs)
                    dt = time.time() - t0
                    n_things = len(result.workplace.things or [])
                    n_places = len(result.workplace.places or [])
                    n_actions = len(result.activity.actions or [])
                    detail = f"{n_things} things / {n_places} places / {n_actions} actions"
                    rows.append((spec, provider, model_id, "OK", detail, dt))
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
                    rows.append((spec, provider, model_id, "FAIL", f"{type(e).__name__}: {msg}"[:200], dt))
                    print(f"  FAIL  {type(e).__name__}: {msg[:200]} ({dt:.1f}s)\n")
                    break

    print("=" * 84)
    print(f"{'SPEC':<14} {'PROVIDER':<10} {'MODEL':<28} {'RESULT':<6} DETAIL")
    print("-" * 84)
    for spec, provider, model_id, status, detail, dt in rows:
        print(f"{spec:<14} {provider:<10} {model_id:<28} {status:<6} {detail}")

    n_fail = sum(1 for r in rows if r[3] == "FAIL")
    print("-" * 84)
    if n_fail:
        print(f"{n_fail} FAILED — do NOT retire the Gemini twins yet.")
    else:
        print("All providers OK on both ARLEM specs — safe to retire the Gemini twins.")


if __name__ == "__main__":
    main()
