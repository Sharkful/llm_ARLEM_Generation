"""
Probe: does Gemini handle Pydantic Union / discriminated-union types now?

Background: the runner swaps in union-free "flat" schemas for Google models
because Gemini historically rejected Union types. Google added `anyOf` /
full-JSON-Schema support for Gemini 2.5+ in Nov 2025, so this may be obsolete.

This probe isolates the *union* question from our large, complex Lab schema by
using a tiny toy schema (a list of a discriminated union — the exact shape
`json_lab.Component` uses: a `Literal` discriminator inside a `list`). If the toy
fails, the problem is unions/mode; if the toy passes but the real Lab fails, the
problem is schema size/complexity, not unions.

It sweeps the toy across the instructor Modes reachable via `from_provider`,
with `max_retries=0` so we see the raw first-attempt outcome (no retry masking).

Run from the project root with the venv active and GEMINI_API_KEY in .env:
    python "Code/Testing/probe_gemini_unions.py"
"""

import os
import time
from typing import Annotated, Literal, Union

import instructor
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
MODEL = "gemini-2.5-flash-lite"  # cheap Gemini 2.5+ model (has anyOf support)

PROMPT = (
    "Create a zoo with exactly 3 animals: one cat and two dogs. "
    "Make up reasonable loudness values in decibels."
)


# ── Toy schemas ───────────────────────────────────────────────────────

class Cat(BaseModel):
    pet_type: Literal["cat"] = "cat"
    meow_db: int = Field(description="meow loudness in decibels")


class Dog(BaseModel):
    pet_type: Literal["dog"] = "dog"
    bark_db: int = Field(description="bark loudness in decibels")


# Discriminated union — the json_lab.Component pattern (Literal discriminator).
DiscriminatedPet = Annotated[Union[Cat, Dog], Field(discriminator="pet_type")]


class ZooDiscriminated(BaseModel):
    """A list of a discriminated union."""
    animals: list[DiscriminatedPet] = Field(description="the animals in the zoo")


class ZooPlain(BaseModel):
    """A list of a plain (non-discriminated) union, for comparison."""
    animals: list[Union[Cat, Dog]] = Field(description="the animals in the zoo")


# ── Probe ─────────────────────────────────────────────────────────────

# The only two modes from_provider("google/...") accepts in this stack
# (the new google-genai SDK). JSON / GEMINI_JSON are legacy-SDK only.
MODES = ["GENAI_STRUCTURED_OUTPUTS", "GENAI_TOOLS"]
FALLBACK_MODEL = "gemini-2.5-flash"  # used only if flash-lite returns 503


def _transient(msg: str) -> bool:
    m = msg.lower()
    return "503" in m or "unavailable" in m or "overloaded" in m


def _attempt(model_id: str, mode, response_model):
    client = instructor.from_provider(f"google/{model_id}", mode=mode, api_key=API_KEY)
    result = client.create(
        response_model=response_model,
        messages=[{"role": "user", "content": PROMPT}],
        max_retries=0,  # raw first-attempt outcome, no retry masking
    )
    return [a.pet_type for a in result.animals]


def try_one(mode_name: str, response_model, label: str) -> tuple:
    if not hasattr(instructor.Mode, mode_name):
        return (mode_name, label, "SKIP", "mode absent in this instructor build")
    mode = getattr(instructor.Mode, mode_name)
    try:  # catch ModeError / client-construction rejections early
        instructor.from_provider(f"google/{MODEL}", mode=mode, api_key=API_KEY)
    except Exception as e:
        return (mode_name, label, "FAIL(client)", f"{type(e).__name__}: {e}"[:200])

    last = ""
    for model_id in (MODEL, FALLBACK_MODEL):
        for _ in range(4):
            try:
                types = _attempt(model_id, mode, response_model)
                tag = "OK" if model_id == MODEL else f"OK (fallback {model_id})"
                return (mode_name, label, tag, f"{len(types)} animals {types}")
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                if _transient(last):
                    time.sleep(3)
                    continue
                return (mode_name, label, "FAIL", last[:220])  # deterministic failure
        # exhausted 503 retries on this model → fall through to FALLBACK_MODEL
    return (mode_name, label, "FAIL(503)", last[:200])


def main() -> None:
    print(f"instructor {instructor.__version__} | model {MODEL}\n")
    disc = ZooDiscriminated.model_json_schema()["properties"]["animals"]["items"]
    plain = ZooPlain.model_json_schema()["properties"]["animals"]["items"]
    print("discriminated union -> animals.items:", disc)
    print("plain union         -> animals.items:", plain)
    print()

    rows = []
    for m in MODES:
        rows.append(try_one(m, ZooDiscriminated, "discriminated"))
        rows.append(try_one(m, ZooPlain, "plain-union"))

    width = max(len(r[0]) for r in rows)
    print(f"{'MODE':<{width}}  {'SCHEMA':<13}  RESULT")
    print("-" * (width + 13 + 45))
    for mode_name, label, status, detail in rows:
        print(f"{mode_name:<{width}}  {label:<13}  {status}: {detail}")


if __name__ == "__main__":
    main()
