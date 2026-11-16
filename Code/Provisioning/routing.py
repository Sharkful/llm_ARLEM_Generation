"""
Describe and route (#106; spec: PROVISIONING.md, "Routes" and "Steps per lab",
step 2): one LLM call per lab that says, for every asset the library doesn't
already have, what it should look like and how to provide it.

Routes are chosen by the asset's **form** (a surface on a simple shape, a diagram
element, a UI element, a structure from a public database, a recognizable object),
never by the lab's subject, so the tool works for STEM topics we have no labs for.
Builders are offered by their own descriptions (``builders.REGISTRY``).

The call goes through the run's ``llm.TrackedLLM`` (the repo's client factory and
``benchmark.guarded_create``, tracked). The response is checked under
``routing_scope``: every listed asset (a kind and a name: a lab can use one name for a
model and a texture) planned exactly once, an allowed form and route,
``MIN_KEYWORDS`` to ``MAX_KEYWORDS`` search keywords, and ``same_as`` merges that point
at another listed asset of the same kind.
Fields are lenient so all problems come back in one error, and instructor reasks
once for all of them.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

import builders
from audit import AssetRequest
from library import name_key


MODEL_FORMS = {
    "diagram_element": "a mark in a diagram: arrow, vector, axis, line, path, trail, orbit, ring, grid, "
                       "plane, marker, highlight, sight line, region",
    "ui": "a panel, board, label, readout, legend, menu, button, slider or other interface element",
    "text": "text in the scene",
    "textured_primitive": "a simple shape whose look comes from an image on it: a planet, moon or star, "
                          "a coin, a dish, a map, a screen",
    "effect": "something that changes, grows or moves over time or reacts to the student rather than a "
              "fixed shape: a trail drawn as an object moves, a path that depends on how something is "
              "launched, a pulse, a flow",
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
    "dynamic": "a picture that changes while the lab runs: a live graph, a video, a value display",
    "other": "none of the above",
}
# Behaviour or impossibility, not an asset: these routes are flagged for a reviewer, who
# can have the asset turned into a component (the reviewer step is #116).
REVIEW_ROUTES = {
    "component": "Not a fixed 3D model or picture but behaviour: something that changes, grows or "
                 "moves over time or reacts to the student (a trail drawn as an object moves, a path "
                 "that depends on how something is launched, a live graph). It should be a component "
                 "on an object, not an asset. Flagged for a reviewer.",
    "impossible": "Can't be provided as a 3D model or texture from what the lab gives: it needs data, "
                  "content or behaviour the lab doesn't specify. Flagged for a reviewer.",
}
MODEL_ROUTES = {
    "shape": "A simple shape, colored and scaled by the lab, optionally with a real image on it found "
             "in a texture library. The headset makes sphere, cube, plane, capsule, cylinder and quad "
             "itself; arrow, ring or torus, cone, dome, grid and rounded panel are provided once. For "
             "diagram elements in any subject (vectors, axes, orbit rings, ground planes, markers, "
             "regions, blocks) and for things whose look is an image on a simple shape (planets, moons, "
             "stars, a coin, a map, a screen).",
    "ui": "Not a 3D model: a panel, board, label, readout, legend, menu, button or slider. Becomes a "
          "flat panel with text on it.",
    "search": "A recognizable real object or organism to find in free 3D model libraries: anatomy, "
              "cells, instruments, lab equipment, animals, plants, vehicles, tools, buildings.",
    "composite": "Built from a few simple colored shapes (up to about 12) when it is simple enough and "
                 "the other routes don't fit or a library is unlikely to have it. One shape alone is "
                 "\"shape\", not composite.",
    **REVIEW_ROUTES,
}
TEXTURE_ROUTES = {
    "search": "A real surface or material found in a texture library: planet and moon maps, star "
              "fields, wood, metal, stone, skin, fabric.",
    "draw": "A picture with content drawn exactly by code: a diagram, chart, graph paper, structural "
            "formula, labelled figure, words.",
    "generate": "A stylized or generic surface no library has, made by an image model. Never for real "
                "places, and never for anything with words or scientific detail.",
    **REVIEW_ROUTES,
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

# Listed assets for the call being checked: (kind, name key) -> name. Keyed by kind
# too, because a lab can use one name for a model and a texture (two assets).
_SCOPE: contextvars.ContextVar[Optional[dict[tuple[str, str], str]]] = \
    contextvars.ContextVar("routing_scope", default=None)
MIN_KEYWORDS, MAX_KEYWORDS = 2, 6


@contextmanager
def routing_scope(requests: list[AssetRequest]):
    """Check ``LabAssetPlan`` responses against these assets while inside."""
    token = _SCOPE.set({(r.kind, r.key): r.name for r in requests})
    try:
        yield
    finally:
        _SCOPE.reset(token)


def match_listed(items, listed: dict[tuple[str, str], str], verb: str):
    """Match a response's items (each with ``name`` and ``kind``) to the listed assets.
    Returns ``(seen, problems)``: ``seen`` maps (kind, name key) to its item, with the
    item's name set to the lab's spelling; problems are unknown names, wrong kinds,
    duplicates and missing assets. An item with the wrong kind is reported once, not
    also as missing, and never makes the correct item look duplicated. Shared by every
    per-lab check (routing, reuse)."""
    problems: list[str] = []
    seen: dict = {}
    wrong_kind: set = set()   # answered, but with the wrong kind
    kinds_of: dict[str, set] = {}
    for kind, k in listed:
        kinds_of.setdefault(k, set()).add(kind)
    for a in items:
        k = name_key(a.name)
        if k not in kinds_of:
            problems.append(f"{a.name!r} is not one of the listed assets")
            continue
        if a.kind not in kinds_of[k]:
            listed_as = " and a ".join(sorted(kinds_of[k]))
            problems.append(f"{a.name!r} is listed as a {listed_as}, not a {a.kind or 'kind-less'} asset")
            if len(kinds_of[k]) == 1:   # clearly that asset: don't also call it missing
                wrong_kind.add((next(iter(kinds_of[k])), k))
            continue
        key = (a.kind, k)
        if key in seen:
            problems.append(f"{a.kind} {listed[key]!r} is {verb} more than once")
            continue
        seen[key] = a
        a.name = listed[key]
    missing = [f"{name} ({kind})" for (kind, k), name in listed.items()
               if (kind, k) not in seen and (kind, k) not in wrong_kind]
    if missing:
        problems.append("these listed assets are missing: " + ", ".join(missing))
    return seen, problems


def strict_schema(optional: set[str]):
    """A ``json_schema_extra`` that keeps the tool schema the LLM sees strict while the
    model validates leniently: every field not in ``optional`` is required, and none of
    them shows its lenient default."""
    def apply(schema: dict) -> None:
        props = schema.get("properties", {})
        schema["required"] = [k for k in props if k not in optional]
        for k, prop in props.items():
            if k not in optional:
                prop.pop("default", None)
    return apply


class PlannedAsset(BaseModel):
    # Only name is required when validating: everything else is checked in
    # LabAssetPlan._check, so all problems in a response come back in one error (field
    # errors would stop validation before that check runs, and cost a reask each). The
    # schema sent to the LLM still marks them required (strict_schema).
    model_config = ConfigDict(json_schema_extra=strict_schema({"must_be_right", "same_as"}))

    name: str = Field(description="The asset's name, exactly as listed.")
    kind: str = Field("", description="The asset's kind, as listed.",
                      json_schema_extra={"enum": list(KINDS)})
    description: str = Field(
        "",
        description="What it should look like in this lab, in 1 to 3 plain sentences: its shape, "
                    "parts and colors, and what it shows. Written for someone who has to find or build it.")
    must_be_right: list[str] = Field(
        default_factory=list,
        description="Up to 5 facts it must get right for the lab to be correct, as a teacher would "
                    "check them (e.g. 'four chambers visible', 'tetrahedral, 109.5 degree bond angles'). "
                    "Empty when nothing specific.")
    search_keywords: list[str] = Field(
        default_factory=list,
        description=f"{MIN_KEYWORDS} to {MAX_KEYWORDS} short phrases a person would type into a 3D "
                    "model or texture library to "
                    "find it (identity words, not filler).")
    form: str = Field("", description="One of the listed forms for its kind.")
    route: str = Field("", description="One of the listed routes for its kind.")
    reason: str = Field("", description="One sentence: why this route.")
    same_as: Optional[str] = Field(
        None,
        description="Only when this asset is clearly the same object as another listed asset of the "
                    "same kind (two names for one thing): that asset's name. It is then provided once. "
                    "Otherwise null.")


class LabAssetPlan(BaseModel):
    assets: list[PlannedAsset] = Field(default_factory=list, description="One entry per listed asset.")

    @model_validator(mode="after")
    def _check(self):
        listed = _SCOPE.get()
        if listed is None:
            return self
        seen, problems = match_listed(self.assets, listed, "planned")
        forms = {kind: forms_for(kind) for kind in KINDS}     # once per validation
        routes = {kind: routes_for(kind) for kind in KINDS}
        for a in seen.values():
            name = a.name
            if a.form not in forms[a.kind]:
                problems.append(f"{name!r}: form {a.form!r} is not one of the {a.kind} forms "
                                f"({', '.join(forms[a.kind])})")
            if a.route not in routes[a.kind]:
                problems.append(f"{name!r}: route {a.route!r} is not one of the {a.kind} routes "
                                f"({', '.join(routes[a.kind])})")
            if not a.description.strip():
                problems.append(f"{name!r}: the description is empty")
            if not a.reason.strip():
                problems.append(f"{name!r}: the reason is empty")
            n = len([w for w in a.search_keywords if w.strip()])
            if not MIN_KEYWORDS <= n <= MAX_KEYWORDS:
                problems.append(f"{name!r}: give {MIN_KEYWORDS} to {MAX_KEYWORDS} search keywords "
                                f"(got {n})")
        for (kind, k), a in seen.items():
            if not a.same_as:
                continue
            t = name_key(a.same_as)
            target = seen.get((kind, t))
            if t == k:
                problems.append(f"{a.name!r}: same_as can't name the asset itself")
            elif target is None:
                other = next((x for (kk, tt), x in seen.items() if tt == t), None)
                problems.append(f"{a.name!r}: same_as {a.same_as!r} is not a listed {kind}"
                                + (f" (it is a {other.kind})" if other is not None else ""))
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
    if getattr(u, "components", None):
        bits.append("components " + ", ".join(u.components))
    line = ", ".join(bits)
    if u.text:
        line += "; text on it: " + " | ".join(f'"{t}"' for t in u.text)
    return line


def asset_block(r: AssetRequest) -> str:
    lines = [f"[{r.kind}] {r.name}"]
    if len(r.spellings) > 1:
        lines.append(f"  also spelled: {', '.join(s for s in r.spellings if s != r.name)}")
    lines.append(f"  used {r.use_count} time(s), in module(s) {', '.join(map(str, r.modules))}")
    for u in r.uses:
        lines.append(f"  - {_use_line(u, r.kind)}")
    if r.use_count > len(r.uses):
        lines.append(f"  - ...and {r.use_count - len(r.uses)} more use(s)")
    if r.components:
        lines.append("  components on its objects: " + ", ".join(
            f"{c} on {n} of {r.use_count}" for c, n in r.components.items()))
    for n in r.narration:
        lines.append(f'  narration: "{n}"')
    if r.purpose:
        lines.append(f"  planned purpose: {r.purpose}")
    return "\n".join(lines)


def _options(title: str, options: dict[str, str]) -> str:
    return title + "\n" + "\n".join(f"- {k}: {v}" for k, v in options.items())


def lab_summary(lab: dict) -> str:
    """The LAB block every provisioning prompt opens with."""
    modules = "\n".join(
        f"{i}. {m.get('moduleName', '')}: {' '.join(str(m.get('description', '')).split())[:200]}"
        for i, m in enumerate(lab.get("modules", []), start=1))
    objectives = "\n".join(f"- {o}" for o in lab.get("objectives") or [])
    return f"""LAB
id: {lab.get("labId", "")}
course: {lab.get("courseName", "")}
objectives:
{objectives}
modules:
{modules}"""


def build_prompt(lab: dict, requests: list[AssetRequest]) -> str:
    models = sum(r.kind == "model" for r in requests)
    textures = len(requests) - models
    return f"""You plan how to provide the 3D models and textures for an augmented-reality science lab \
shown on a headset. The lab was generated with names for assets that don't exist yet. For each asset \
listed below, say what it should look like and how it should be provided.

{lab_summary(lab)}

ASSETS TO PLAN ({models} model(s), {textures} texture(s))
Each asset shows how the lab uses it: the objects, their module, scale [x, y, z], color [r, g, b, a], \
the texture or model it is paired with, text shown on it, its components (what students do with it), narration that mentions it, and its planned \
purpose when known.

{chr(10).join(asset_block(r) for r in requests)}

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
- Use "component" for behaviour the lab named as an asset (something that changes, grows, moves or \
reacts), and "impossible" only when nothing can be made from what the lab gives. Say why in reason: \
a reviewer reads it.
"""


# ── The call ──────────────────────────────────────────────────────────


def plan_routes(llm, lab: dict, requests: list[AssetRequest]) -> tuple[Optional[LabAssetPlan], dict]:
    """(plan, record) from one call through ``llm`` (an ``llm.TrackedLLM``). With
    nothing to plan, no call is made and the plan is empty. On failure the plan is None
    and the record carries the error."""
    if not requests:
        return LabAssetPlan(assets=[]), {**llm.describe(), "calls": 0}
    with routing_scope(requests):
        return llm.call(LabAssetPlan, build_prompt(lab, requests))
