"""
Reuse (#108; spec: PROVISIONING.md, "Reuse"): before anything is planned or provided,
check whether the library already has each asset, by meaning, not just by name.

1. **Name or alias hit** (``LocalFolderStore.find``, in ``provision_lab``): available,
   no LLM call.
2. **Shortlist** (``shortlist``): the whole library while it has at most
   ``SHORTLIST_LIMIT`` reusable entries; above that, for each asset the
   ``SHORTLIST_SIZE`` entries that share the most words with it (an embedding shortlist
   can replace this once libraries are that big).
3. **One LLM judge per lab** (``ReusePlan``), for all the assets step 1 didn't match.
   Per asset, one of:
   - ``reuse``: a library entry of the same kind *is* this asset (the lab's color and
     scale adapt it); a model may also take a library texture (a built-in sphere with a
     planet map);
   - ``textured_shape``: a model that is a simple shape with a library texture on it
     (a planet map on a sphere), the shape made by the headset;
   - ``none``: nothing in the library is this asset; it is planned and provided.
4. **Aliases** (``provision_lab --save-aliases``): an accepted ``reuse`` of a single
   entry (no texture) that is not built-in adds the lab's name to that entry's aliases,
   so the next lab with that name hits step 1 with no call. Decisions that combine two
   entries, and stand-ins for a generic built-in (``marsPrefab`` as ``tiny_sphere``),
   are not saved as aliases; the judge decides them again each time.

The response is checked under ``reuse_scope``: every listed asset decided exactly once
(``routing.match_listed``), entries that exist in the shortlist with the right kind,
and a shape the headset makes. All problems come back in one error (one reask).
"""

from __future__ import annotations

import contextvars
import re
from contextlib import contextmanager
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audit import AssetRequest, name_words
from library import LibraryEntry
from routing import KINDS, asset_block, lab_summary, match_listed, strict_schema

SHORTLIST_LIMIT = 200   # send the whole library up to this many reusable entries
SHORTLIST_SIZE = 20     # above it, this many per asset
DECISIONS = {
    "reuse": "a library entry of the same kind IS this asset: the same object, though a different "
             "color or size is fine (the lab's color and scale adapt it). A model may also name a "
             "library texture to put on it (texture_id), e.g. a built-in sphere with a planet map.",
    "textured_shape": "(models only) the asset is a simple shape whose look is a library texture: a "
                      "planet, moon or star map on a sphere, a map on a plane. Give the shape and the "
                      "texture_id.",
    "none": "nothing in the library is this asset; it will be planned and provided. Related is not "
            "enough: a heart is not a lung, an Earth map is not a Mars map.",
}
# The simple shapes the headset makes itself (PROVISIONING.md, "Simple shapes";
# Sharkful/Quest3_CyberAR#39).
SHAPES = ("sphere", "cube", "plane", "capsule", "cylinder", "quad",
          "arrow", "torus", "cone", "dome", "grid", "panel")

_WORD = re.compile(r"[a-z0-9]+")


def candidates(entries: list[LibraryEntry]) -> list[LibraryEntry]:
    """The entries the judge may reuse: every reusable one (not system objects)."""
    return [e for e in entries if e.reusable]


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def _asset_words(r: AssetRequest) -> set[str]:
    words = _words(name_words(r.name))
    for u in r.uses:
        words |= _words(name_words(u.object))
    for line in r.narration:
        words |= _words(line)
    return words


def _entry_words(e: LibraryEntry) -> set[str]:
    return (_words(name_words(e.name)) | _words(" ".join(e.tags)) | _words(e.description)
            | {w for a in e.aliases for w in _words(name_words(a))})


def shortlist(requests: list[AssetRequest], entries: list[LibraryEntry]) -> list[LibraryEntry]:
    """The entries to show the judge: all of them while the library is small, else the
    best word matches for each asset (in library order)."""
    if len(entries) <= SHORTLIST_LIMIT:
        return list(entries)
    words = {e.asset_id: _entry_words(e) for e in entries}
    keep: set[str] = set()
    for r in requests:
        aw = _asset_words(r)
        ranked = sorted(entries, key=lambda e: -len(aw & words[e.asset_id]))
        keep |= {e.asset_id for e in ranked[:SHORTLIST_SIZE]}
    return [e for e in entries if e.asset_id in keep]


# ── Response model ────────────────────────────────────────────────────

# (listed assets: (kind, name key) -> name, offered entries: asset_id -> entry)
_SCOPE: contextvars.ContextVar = contextvars.ContextVar("reuse_scope", default=None)


@contextmanager
def reuse_scope(requests: list[AssetRequest], entries: list[LibraryEntry]):
    """Check ``ReusePlan`` responses against these assets and offered entries."""
    token = _SCOPE.set(({(r.kind, r.key): r.name for r in requests},
                        {e.asset_id: e for e in entries}))
    try:
        yield
    finally:
        _SCOPE.reset(token)


class ReuseDecision(BaseModel):
    # Lenient fields, checked in ReusePlan._check (all problems in one error); the
    # schema the LLM sees stays strict (routing.strict_schema).
    model_config = ConfigDict(json_schema_extra=strict_schema({"asset_id", "texture_id", "shape", "adapt"}))

    name: str = Field(description="The asset's name, exactly as listed.")
    kind: str = Field("", description="The asset's kind, as listed.",
                      json_schema_extra={"enum": list(KINDS)})
    decision: str = Field("", description="reuse, textured_shape or none.",
                          json_schema_extra={"enum": list(DECISIONS)})
    asset_id: str = Field("", description="reuse: the library entry's id (same kind as the asset).")
    texture_id: str = Field("", description="A library texture's id: required for textured_shape; "
                                            "optional for reusing a model.")
    shape: str = Field("", description="textured_shape: the simple shape.",
                       json_schema_extra={"enum": list(SHAPES)})
    adapt: str = Field("", description="How the lab's color or scale changes the reused entry, if at "
                                       "all (e.g. 'tinted red by the lab'); else empty.")
    reason: str = Field("", description="One sentence a reviewer can check: why this decision.")


class ReusePlan(BaseModel):
    decisions: list[ReuseDecision] = Field(default_factory=list,
                                           description="One decision per listed asset.")

    @model_validator(mode="after")
    def _check(self):
        scope = _SCOPE.get()
        if scope is None:
            return self
        listed, offered = scope
        seen, problems = match_listed(self.decisions, listed, "decided")

        def entry(asset_id: str, kind: str, what: str) -> bool:
            e = offered.get(asset_id)
            if e is None:
                problems.append(f"{name!r}: {what} {asset_id!r} is not a library id listed above")
            elif e.kind != kind:
                problems.append(f"{name!r}: {what} {asset_id!r} is a {e.kind}, not a {kind}")
            return e is not None and e.kind == kind

        for d in seen.values():
            name = d.name
            if d.decision not in DECISIONS:
                problems.append(f"{name!r}: decision {d.decision!r} is not one of {', '.join(DECISIONS)}")
                continue
            if not d.reason.strip():
                problems.append(f"{name!r}: the reason is empty")
            if d.decision == "none":
                d.asset_id = d.texture_id = d.shape = d.adapt = ""
                continue
            if d.decision == "reuse":
                d.shape = ""
                if not d.asset_id:
                    problems.append(f"{name!r}: reuse needs the asset_id of the entry to reuse")
                else:
                    entry(d.asset_id, d.kind, "asset_id")
                if d.texture_id:
                    if d.kind != "model":
                        problems.append(f"{name!r}: only a reused model can take a texture_id")
                    else:
                        entry(d.texture_id, "texture", "texture_id")
            else:  # textured_shape
                d.asset_id = ""
                if d.kind != "model":
                    problems.append(f"{name!r}: textured_shape is for models; for a texture use "
                                    f"reuse or none")
                    continue
                if d.shape not in SHAPES:
                    problems.append(f"{name!r}: shape {d.shape!r} is not one of {', '.join(SHAPES)}")
                if not d.texture_id:
                    problems.append(f"{name!r}: textured_shape needs the texture_id of a library texture")
                else:
                    entry(d.texture_id, "texture", "texture_id")
        if problems:
            raise ValueError("Fix these problems and decide every listed asset again:\n- "
                             + "\n- ".join(problems))
        return self


# ── Prompt and call ───────────────────────────────────────────────────


def _entry_line(e: LibraryEntry) -> str:
    bits = [e.kind] + (["built-in"] if e.builtin else [])
    line = f"- {e.asset_id} [{', '.join(bits)}] {e.name}: {' '.join(e.description.split())}"
    extra = []
    if e.mapping:
        extra.append(f"mapping {e.mapping}")
    if e.tags:
        extra.append("tags " + ", ".join(e.tags))
    return line + (f" ({'; '.join(extra)})" if extra else "")


def build_prompt(lab: dict, requests: list[AssetRequest], entries: list[LibraryEntry]) -> str:
    return f"""You check whether the assets an augmented-reality science lab needs already exist in our \
asset library, so they are reused instead of being made again. The lab named its assets itself, so a \
name rarely matches exactly: judge by what each asset is and how the lab uses it.

{lab_summary(lab)}

LIBRARY ({len(entries)} entries: id [kind] name: description)
{chr(10).join(_entry_line(e) for e in entries)}

ASSETS TO CHECK ({len(requests)})
Each asset shows how the lab uses it: the objects, their module, scale [x, y, z], color [r, g, b, a], \
the texture or model it is paired with, text shown on it, narration that mentions it, and its planned \
purpose when known.

{chr(10).join(asset_block(r) for r in requests)}

DECISIONS
{chr(10).join(f"- {k}: {v}" for k, v in DECISIONS.items())}

SHAPES the headset makes itself: {", ".join(SHAPES)}

RULES
- Decide every listed asset exactly once, with its name exactly as listed and its listed kind.
- Reuse only what the library entry really is. When unsure, choose none: the asset is then planned and \
provided, which costs little; a wrong reuse shows the wrong thing in the lab.
- A plain shape (a built-in sphere with only the lab's color) stands in only for something generic: a \
marker, a ball, an atom drawn as a colored ball. A specific real object with its own look (Mars, a heart, \
a named molecule) is not a plain shape tinted: reuse an entry that is that object, put its library \
texture on a sphere (texture_id, or textured_shape), or choose none so it gets the right look. Mars with \
no Mars map in the library is none, not a red sphere.
- Prefer a built-in prefab when the lab uses it the way it works (a sphere students grab); otherwise \
prefer textured_shape for a surface on a simple shape.
- adapt: how the lab's color or scale changes the reused entry, if at all; else empty.
- reason: one sentence a reviewer can check.
"""


def judge_reuse(llm, lab: dict, requests: list[AssetRequest],
                entries: list[LibraryEntry]) -> tuple[Optional[ReusePlan], dict]:
    """(plan, record) from one call through ``llm`` (an ``llm.TrackedLLM``). With no
    assets or no reusable entries, no call is made and the plan is empty. On failure the
    plan is None and the record carries the error."""
    pool = shortlist(requests, candidates(entries))
    if not requests or not pool:
        return ReusePlan(decisions=[]), {**llm.describe(), "calls": 0, "shortlist": len(pool)}
    with reuse_scope(requests, pool):
        plan, record = llm.call(ReusePlan, build_prompt(lab, requests, pool))
    record["shortlist"] = len(pool)
    return plan, record
