"""
Describe and route (#106; spec: PROVISIONING.md, "Routes" and "Steps per lab",
step 2): one LLM call per lab that says, for every asset the library doesn't
already have, what it should look like and how to provide it.

Routes are chosen by the asset's **form** (a surface on a simple shape, a diagram
element, a UI element, a structure from a public database, a recognizable object),
never by the lab's subject, so the tool works for STEM topics we have no labs for.
Builders are offered by their own descriptions (``builders.REGISTRY``).

The call goes through ``benchmark.create_instructor_client`` and
``benchmark.provider_create_kwargs`` (the #44 Gemini and Anthropic guards) with
``benchmark._transient_retryer`` around it, tracked by ``InstructorTracker``, like
every LLM entry point. The response is checked under ``routing_scope``: every listed
asset planned exactly once, with its own kind, an allowed form and route, and
``same_as`` merges that point at another listed asset of the same kind. All problems
come back in one error, so instructor reasks once for all of them.
"""

from __future__ import annotations

import contextvars
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field, model_validator

import builders
from audit import AssetRequest
from library import name_key

HERE = Path(__file__).resolve().parent
BENCHMARK_DIR = HERE.parent / "Benchmark"
DEFAULT_MODEL = "claude-haiku-5.5"
DEFAULT_MAX_RETRIES = 3

MODEL_FORMS = {
    "diagram_element": "a mark in a diagram: arrow, vector, axis, line, path, trail, orbit, ring, grid, "
                       "plane, marker, highlight, sight line, region",
    "ui": "a panel, board, label, readout, legend, menu, button, slider or other interface element",
    "text": "text in the scene",
    "textured_primitive": "a simple shape whose look comes from an image on it: a planet, moon or star, "
                          "a coin, a dish, a map, a screen",
    "structure": "a scientific structure that exists in a public database: a molecule, a protein, DNA "
                 "or RNA, a crystal",
    "organism": "a living thing or part of one: an organ, a cell, tissue, an animal, a plant",
    "object": "a recognizable made or natural object: an instrument, equipment, a tool, a vehicle, a "
              "building, a rock",
    "other": "none of the above",
}
TEXTURE_FORMS = {
    "real_surface": "the real look of a specific thing or place: a planet map, a star field, skin, a leaf",
    "material": "a generic material: wood, metal, stone, fabric, plastic",
    "diagram": "a picture with content: a diagram, chart, graph paper, structural formula, labelled "
               "figure, words",
    "other": "none of the above",
}
MODEL_ROUTES = {
    "kit": "A standard shape from the kit, colored by the lab: arrow, ring or torus, line or thin rod, "
           "grid, flat plane or board, dome, marker or small sphere, cube, cylinder, cone. For diagram "
           "elements in any subject: vectors, axes, orbits, paths, trails, sight lines, ground planes, "
           "highlights, regions.",
    "textured_primitive": "A simple shape (sphere, plane, cylinder) with a real image on it, found in "
                          "a texture library: planets, moons, stars, a coin, a dish, a map, a screen.",
    "ui": "Not a 3D model: a panel, board, label, readout, legend, menu, button or slider. Becomes a "
          "flat panel with text on it.",
    "search": "A recognizable real object or organism to find in free 3D model libraries: anatomy, "
              "cells, instruments, lab equipment, animals, plants, vehicles, tools, buildings.",
    "composite": "Built from a few simple colored shapes (up to about 12) when it is simple enough and "
                 "the other routes don't fit or a library is unlikely to have it.",
}
TEXTURE_ROUTES = {
    "search": "A real surface or material found in a texture library: planet and moon maps, star "
              "fields, wood, metal, stone, skin, fabric.",
    "draw": "A picture with content drawn exactly by code: a diagram, chart, graph paper, structural "
            "formula, labelled figure, words.",
    "generate": "A stylized or generic surface no library has, made by an image model. Never for real "
                "places, and never for anything with words or scientific detail.",
}
KINDS = ("model", "texture")


def forms_for(kind: str) -> dict[str, str]:
    return MODEL_FORMS if kind == "model" else TEXTURE_FORMS


def routes_for(kind: str) -> dict[str, str]:
    """The routes for a kind: the fixed ones plus every registered builder of that kind."""
    base = dict(MODEL_ROUTES if kind == "model" else TEXTURE_ROUTES)
    for b in builders.REGISTRY:
        if b.kind == kind:
            base[f"builder:{b.name}"] = f"{b.description} Needs: {b.needs}."
    return base


# ── Response model ────────────────────────────────────────────────────

# Listed assets for the call being checked: name key -> (name, kind).
_SCOPE: contextvars.ContextVar[Optional[dict[str, tuple[str, str]]]] = \
    contextvars.ContextVar("routing_scope", default=None)


@contextmanager
def routing_scope(requests: list[AssetRequest]):
    """Check ``LabAssetPlan`` responses against these assets while inside."""
    token = _SCOPE.set({r.key: (r.name, r.kind) for r in requests})
    try:
        yield
    finally:
        _SCOPE.reset(token)


class PlannedAsset(BaseModel):
    name: str = Field(description="The asset's name, exactly as listed.")
    kind: Literal["model", "texture"] = Field(description="The asset's kind, as listed.")
    description: str = Field(
        description="What it should look like in this lab, in 1 to 3 plain sentences: its shape, "
                    "parts and colors, and what it shows. Written for someone who has to find or build it.")
    must_be_right: list[str] = Field(
        default_factory=list,
        description="Up to 5 facts it must get right for the lab to be correct, as a teacher would "
                    "check them (e.g. 'four chambers visible', 'tetrahedral, 109.5 degree bond angles'). "
                    "Empty when nothing specific.")
    search_keywords: list[str] = Field(
        description="2 to 6 short phrases a person would type into a 3D model or texture library to "
                    "find it (identity words, not filler).")
    form: str = Field(description="One of the listed forms for its kind.")
    route: str = Field(description="One of the listed routes for its kind.")
    reason: str = Field(description="One sentence: why this route.")
    same_as: Optional[str] = Field(
        None,
        description="Only when this asset is clearly the same object as another listed asset of the "
                    "same kind (two names for one thing): that asset's name. It is then provided once. "
                    "Otherwise null.")


class LabAssetPlan(BaseModel):
    assets: list[PlannedAsset] = Field(description="One entry per listed asset.")

    @model_validator(mode="after")
    def _check(self):
        listed = _SCOPE.get()
        if listed is None:
            return self
        problems: list[str] = []
        seen: dict[str, PlannedAsset] = {}
        for a in self.assets:
            k = name_key(a.name)
            if k not in listed:
                problems.append(f"{a.name!r} is not one of the listed assets")
                continue
            name, kind = listed[k]
            if k in seen:
                problems.append(f"{name!r} is planned more than once")
                continue
            seen[k] = a
            a.name = name
            if a.kind != kind:
                problems.append(f"{name!r} is a {kind}, not a {a.kind}")
                continue
            if a.form not in forms_for(kind):
                problems.append(f"{name!r}: form {a.form!r} is not one of the {kind} forms "
                                f"({', '.join(forms_for(kind))})")
            if a.route not in routes_for(kind):
                problems.append(f"{name!r}: route {a.route!r} is not one of the {kind} routes "
                                f"({', '.join(routes_for(kind))})")
            if not a.description.strip():
                problems.append(f"{name!r}: the description is empty")
            if not 1 <= len([w for w in a.search_keywords if w.strip()]) <= 8:
                problems.append(f"{name!r}: give 2 to 6 search keywords")
        missing = [name for k, (name, _) in listed.items() if k not in seen]
        if missing:
            problems.append("these listed assets are missing: " + ", ".join(missing))
        for a in seen.values():
            if not a.same_as:
                continue
            t = name_key(a.same_as)
            target = seen.get(t)
            if t == name_key(a.name):
                problems.append(f"{a.name!r}: same_as can't name the asset itself")
            elif target is None:
                problems.append(f"{a.name!r}: same_as {a.same_as!r} is not a listed asset")
            elif target.kind != a.kind:
                problems.append(f"{a.name!r}: same_as {target.name!r} is a {target.kind}")
            elif target.same_as:
                problems.append(f"{a.name!r}: same_as {target.name!r}, which is itself merged into "
                                f"{target.same_as!r}; point both at the same asset")
            else:
                a.same_as = target.name
        if problems:
            raise ValueError("Fix these problems and plan every listed asset again:\n- "
                             + "\n- ".join(problems))
        return self


# ── Prompt ────────────────────────────────────────────────────────────


def _use_line(u, kind: str) -> str:
    bits = [f'object "{u.object}" in module {u.module} "{u.module_name}"']
    if u.scale is not None:
        bits.append(f"scale {u.scale}")
    if u.color is not None:
        bits.append(f"color {u.color}")
    if u.paired_with:
        bits.append(f"on model {u.paired_with}" if kind == "texture"
                    else f"textured with {u.paired_with}")
    if u.parent and u.parent != "[CURRENT_LAB]":
        bits.append(f'child of "{u.parent}"')
    line = ", ".join(bits)
    if u.text:
        line += "; text on it: " + " | ".join(f'"{t}"' for t in u.text)
    return line


def _asset_block(r: AssetRequest) -> str:
    lines = [f"[{r.kind}] {r.name}"]
    if len(r.spellings) > 1:
        lines.append(f"  also spelled: {', '.join(s for s in r.spellings if s != r.name)}")
    lines.append(f"  used {r.use_count} time(s), in module(s) {', '.join(map(str, r.modules))}")
    for u in r.uses:
        lines.append(f"  - {_use_line(u, r.kind)}")
    if r.use_count > len(r.uses):
        lines.append(f"  - ...and {r.use_count - len(r.uses)} more use(s)")
    for n in r.narration:
        lines.append(f'  narration: "{n}"')
    if r.purpose:
        lines.append(f"  planned purpose: {r.purpose}")
    return "\n".join(lines)


def _options(title: str, options: dict[str, str]) -> str:
    return title + "\n" + "\n".join(f"- {k}: {v}" for k, v in options.items())


def build_prompt(lab: dict, requests: list[AssetRequest]) -> str:
    models = sum(r.kind == "model" for r in requests)
    textures = len(requests) - models
    modules = "\n".join(
        f"{i}. {m.get('moduleName', '')}: {' '.join(str(m.get('description', '')).split())[:200]}"
        for i, m in enumerate(lab.get("modules", []), start=1))
    objectives = "\n".join(f"- {o}" for o in lab.get("objectives") or [])
    return f"""You plan how to provide the 3D models and textures for an augmented-reality science lab \
shown on a headset. The lab was generated with names for assets that don't exist yet. For each asset \
listed below, say what it should look like and how it should be provided.

LAB
id: {lab.get("labId", "")}
course: {lab.get("courseName", "")}
objectives:
{objectives}
modules:
{modules}

ASSETS TO PLAN ({models} model(s), {textures} texture(s))
Each asset shows how the lab uses it: the objects, their module, scale [x, y, z], color [r, g, b, a], \
the texture or model it is paired with, text shown on it, narration that mentions it, and its planned \
purpose when known.

{chr(10).join(_asset_block(r) for r in requests)}

{_options("FORMS FOR MODELS", MODEL_FORMS)}

{_options("FORMS FOR TEXTURES", TEXTURE_FORMS)}

{_options("ROUTES FOR MODELS", routes_for("model"))}

{_options("ROUTES FOR TEXTURES", routes_for("texture"))}

RULES
- Plan every listed asset exactly once, with its name exactly as listed and its listed kind.
- Choose the form and the route by what the asset is and how it looks, never by the lab's subject.
- Describe each asset as this lab needs it, from the lab's own context: its colors, its parts, what it \
shows. Don't add detail the lab doesn't need.
- must_be_right: only facts a teacher would check (counts, shapes, angles, colors that carry meaning).
- search_keywords: what a person would type to find it, e.g. "human heart anatomy", "jupiter texture map".
- same_as: only when two listed names are clearly one object in this lab (e.g. 'dnaHelixPrefab' and \
'dnaDoubleHelixPrefab' both showing the same double helix). Different states, cut-aways or views of a \
thing (a heart and a heart cross-section) are different assets.
- A texture is a picture wrapped onto a model. A texture with words, labels or a diagram is "draw".
"""


# ── The call ──────────────────────────────────────────────────────────

# (response_model, messages, max_retries) -> a validated response_model instance
CallFn = Callable[[type, list, int], BaseModel]


def _benchmark():
    if str(BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_DIR))
    import benchmark
    return benchmark


def model_config_for(model_key: str):
    """The ``benchmark_config.MODELS`` entry for ``model_key`` (KeyError if unknown)."""
    bm = _benchmark()
    if model_key not in bm.MODELS:
        raise KeyError(f"unknown model {model_key!r} (see benchmark.py --list-models)")
    return bm.MODELS[model_key]


API_KEY_ENV = {"anthropic": ("ANTHROPIC_API_KEY",), "openai": ("OPENAI_API_KEY",),
               "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}


def check_environment(model_key: str) -> None:
    """Exit-worthy problems before any call: unknown model, SDK missing, or no key."""
    import os
    cfg = model_config_for(model_key)
    _benchmark().require_environment([cfg])
    names = API_KEY_ENV.get(cfg.provider.value, ())
    if names and not any(os.environ.get(n) for n in names):
        raise KeyError(f"no API key for {model_key}: set {' or '.join(names)} in .env")


class Router:
    """Plans a lab's assets with one tracked LLM call (or ``call_fn``, for tests)."""

    def __init__(self, model_key: str = DEFAULT_MODEL, *, call_fn: Optional[CallFn] = None,
                 max_retries: int = DEFAULT_MAX_RETRIES):
        self.model_key = model_key
        self.max_retries = max_retries
        self.transient_retries = 0
        self.tracker = None
        if call_fn is not None:
            self.model_config = None
            self._call_fn = call_fn
            return
        bm = _benchmark()
        from tracking import InstructorTracker
        self.model_config = model_config_for(model_key)
        self.tracker = InstructorTracker(
            model=self.model_config.model_id, provider=self.model_config.provider.value,
            pricing_config=self.model_config.pricing_config)
        client = bm.create_instructor_client(self.model_config, bm.SpecType.JSON_LAB)
        self._client = self.tracker.wrap(client)
        self._call_fn = self._llm_call

    def _llm_call(self, response_model, messages, max_retries):
        bm = _benchmark()
        kwargs = dict(response_model=response_model, messages=messages, max_retries=max_retries)
        kwargs.update(bm.provider_create_kwargs(self.model_config, response_model))
        retryer = bm._transient_retryer()
        try:
            return retryer(self._client.create, **kwargs)
        finally:
            self.transient_retries += retryer.statistics.get("attempt_number", 1) - 1

    def plan(self, lab: dict, requests: list[AssetRequest]) -> tuple[Optional[LabAssetPlan], dict]:
        """(plan, record). With nothing to plan, no call is made and the plan is empty.
        On failure the plan is None and the record carries the error."""
        record: dict = {"model": self.model_key, "calls": 0}
        if self.model_config is not None:
            record.update(model_id=self.model_config.model_id,
                          provider=self.model_config.provider.value,
                          reasoning_effort=self.model_config.reasoning_effort,
                          tool_choice=_benchmark().tool_choice_for(self.model_config))
        if not requests:
            return LabAssetPlan(assets=[]), record
        prompt = build_prompt(lab, requests)
        record["prompt_characters"] = len(prompt)
        t0 = time.monotonic()
        plan, error = None, None
        try:
            with routing_scope(requests):
                plan = self._call_fn(LabAssetPlan, [{"role": "user", "content": prompt}],
                                     self.max_retries)
        except Exception as exc:  # recorded; the CLI decides the exit code
            error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:2000]}"
        record["seconds"] = round(time.monotonic() - t0, 2)
        record["transient_retries"] = self.transient_retries
        if self.tracker is not None:
            record["tracking"] = self.tracker.summary()
            record["served_models"] = self.tracker.served_models()
            record["cost_usd"] = record["tracking"]["cost"]["total_usd"]
            record["calls"] = record["tracking"]["total_calls"]
        else:
            record["calls"] = 1
        if error:
            record["error"] = error
        return plan, record
