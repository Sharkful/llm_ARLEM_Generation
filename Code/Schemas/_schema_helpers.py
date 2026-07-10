"""
Shared schema helpers for the provider-agnostic Pydantic models.

These are imported by ``json_lab.py`` and the ARLEM schemas (``arlem_full.py`` /
``arlem_simplified.py``) so a single shared schema can be sent to every provider
(OpenAI, Anthropic, Gemini) without per-provider "twin" files.

Contents:
  * ``ConstToEnumSchemaMixin`` - rewrites single-value ``Literal`` JSON-Schema
    ``const`` into a one-element ``enum`` (Gemini's GENAI_TOOLS path rejects
    ``const``).
  * ``_coerce_number_list`` - BeforeValidator that normalizes LLM-emitted numeric
    vectors (comma-joined strings, stringified coordinates).
  * ``clamp_number`` - factory returning a BeforeValidator that coerces numeric
    strings / int<->float and silently clamps into a range. Use for *subjective*
    bounded fields (sizes, positions, rotations, volume); do NOT use where an
    out-of-range value is semantically critical and should force a retry instead.
"""

from __future__ import annotations

from pydantic import BaseModel


# ── const -> enum (Gemini GENAI_TOOLS compatibility) ──────────────────

class ConstToEnumSchemaMixin(BaseModel):
    """Emit single-value ``Literal`` fields as JSON-Schema ``enum`` instead of ``const``.

    This conversion exists to satisfy Gemini. Pydantic v2 renders a one-value
    ``Literal`` (e.g. a ``type`` tag) as ``{"const": "X"}``. Google's google-genai
    SDK builds function-calling tool schemas through a strict ``types.Schema`` model
    that forbids the ``const`` keyword, so a ``const`` field makes Gemini's
    ``GENAI_TOOLS`` path reject the whole schema with "Extra inputs are not
    permitted". The one-element form ``{"enum": ["X"]}`` is semantically identical
    and is what Gemini accepts.

    Because OpenAI and Anthropic accept ``enum`` and ``const`` interchangeably, doing
    the swap on the model itself (rather than per provider) is safe and lets a single
    shared schema go to every provider unchanged. The hook below is provider-agnostic:
    it runs whenever this model's JSON schema is built, so all providers receive the
    ``enum`` form.
    """

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        # 1. Run Pydantic's normal schema generation. Returns this model's JSON
        #    Schema dict exactly as it would be emitted by default, with any
        #    single-value Literal rendered as {"const": "<value>"}.
        schema = handler(core_schema)
        # 2. Rewrite any single-value const on this model's own fields into the
        #    equivalent one-element enum. The value is read out of the existing
        #    `const`, never restated, so it cannot drift from the `Literal`.
        for prop in schema.get("properties", {}).values():
            if "const" in prop:
                prop["enum"] = [prop.pop("const")]
        # 3. Return the modified schema. This fires once per model that inherits
        #    the mixin, as the root model recurses into it.
        return schema


# ── Optional-list None-safety ─────────────────────────────────────────

def none_to_empty_list(v):
    """BeforeValidator: coerce an explicit ``None`` into ``[]``.

    Optional list fields (``sensors``, ``pois``, ``activates``, ``deactivate``,
    ``messages``, ``if_logic`` …) accept an explicit ``null`` from the LLM, which
    lands as ``None`` and makes any validator that iterates the field raise a raw
    ``TypeError`` unless it remembers a per-site ``or []`` guard. Those guards
    leaked twice — issue #49, then review F1 on ``sensors``, a field the earlier
    guards missed. Attaching this coercer to the field fixes it once: ``None``
    becomes ``[]`` before any validator runs, so a newly added Optional list field
    can't reintroduce the crash (review F14).

    Only ``None`` is remapped; every other value (a real list, or an invalid
    scalar) is passed through untouched so the list validator still raises a clean
    error. This normalizes ``model_dump`` of an explicit null to ``[]``; that is
    safe here because every consumer treats absent / null / ``[]`` as empty —
    outputs are saved with ``exclude_none=True`` and ``analyze_arlem`` reads each
    list via ``.get(key, [])``.
    """
    return [] if v is None else v


# ── numeric coercion helpers ──────────────────────────────────────────

def _coerce_number_list(v):
    """Normalize an LLM-emitted numeric vector before list parsing.

    Handles two failure modes seen in the sweeps:
      * a whole vector crammed into one comma-joined string
        (``"-1.05,2.0,-1.4"`` -> ``[-1.05, 2.0, -1.4]``), and
      * individual coordinates emitted as numeric strings
        (``"-0.82"`` -> ``-0.82``), which Pydantic's smart ``Union[int, float]``
        rejects rather than coerces.

    Anything it can't parse is passed through untouched so the normal list
    validator still raises a clean error."""
    if isinstance(v, str):
        v = v.split(",")
    elif (
        isinstance(v, (list, tuple))
        and len(v) == 1
        and isinstance(v[0], str)
        and "," in v[0]
    ):
        v = v[0].split(",")
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            if isinstance(x, str):
                try:
                    x = float(x.strip())
                except ValueError:
                    pass
            out.append(x)
        return out
    return v


def clamp_number(lo: float, hi: float, *, as_int: bool = False):
    """Return a ``mode="before"`` validator that coerces and silently clamps.

    The returned callable:
      * parses numeric strings and accepts int<->float (``"30"`` -> ``30.0``),
      * optionally rounds to int (``as_int=True``) for integer fields,
      * clamps the result into ``[lo, hi]`` instead of raising.

    Non-numeric and ``None`` input is passed through untouched so the field's
    normal validator emits a clean error.

    Use this ONLY for *subjective* bounded fields where a quiet clamp is
    preferable to a paid retry (sizes, positions, rotations, scale, volume).
    For *semantically critical* magnitudes - where a far-too-large value signals
    a real misunderstanding - leave the hard ``ge``/``le`` constraint in place so
    the model is forced to retry instead.

    Generalizes ``json_lab``'s original ``_coerce_fontsize`` clamp.
    """

    def _clamp(v):
        if v is None:
            return v
        try:
            v = float(v)
        except (TypeError, ValueError):
            return v
        v = max(lo, min(hi, v))
        # Cast to int *after* clamping: clamping against float bounds can yield a
        # float boundary (e.g. 100.0), which would then fail an `int` field — so an
        # integer field must always come back as a Python int.
        return int(round(v)) if as_int else v

    return _clamp
