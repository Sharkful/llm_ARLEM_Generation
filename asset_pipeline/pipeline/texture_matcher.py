"""Stage 6c.3 step 2: archive-first texture matching.

Same cheap token/fuzzy scoring philosophy as catalog_matcher, applied to
the texture registry, with one addition geometry matching doesn't have: a
candidate must be *geometrically* compatible (its mapping must fit the
target object's UV info) before its semantic score counts at all. This is
the "download once, reuse forever" guarantee: a hit here means zero LLM,
zero download, zero synthesis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from models.catalog_models import UVInfo
from models.texture_models import TextureAsset, mapping_fits
from pipeline.texture_index import load_index

# A match below this is not trusted for silent reuse -- same conservative
# stance as the geometry catalog matcher.
MATCH_THRESHOLD = 0.5

# Identity words too generic to prove a texture belongs to an object
# ("surface" matching everything is how Mars nearly got the Earth daymap).
_GENERIC_IDENTITY = {
    "surface", "map", "maps", "texture", "daymap", "tileable",
    "equirectangular", "procedural", "material", "albedo", "skin",
}


@dataclass
class TextureMatch:
    texture: TextureAsset
    score: float


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 1]


def score_texture(query_tokens: list[str], texture: TextureAsset) -> float:
    """Fraction of query tokens found in the texture's searchable text,
    weighted: id/name/semantic_type hits count double tag hits."""
    if not query_tokens:
        return 0.0
    strong = _tokens(
        f"{texture.texture_id} {texture.display_name} {texture.semantic_type or ''}"
    )
    weak = _tokens(" ".join(texture.tags))
    total = 0.0
    for token in query_tokens:
        if any(token in s or s in token for s in strong):
            total += 1.0
        elif any(token in w or w in token for w in weak):
            total += 0.5
    return total / len(query_tokens)


def identity_relates(texture: TextureAsset, description: str) -> bool:
    """Does the object's own description mention this texture's identity?

    Guard against LLM search-query drift (a texture_query that says "earth"
    for a Mars object): the texture's specific identity words (texture_id /
    semantic_type / display_name, minus generic surface words) must
    intersect the description. Textures with no specific identity words
    (purely generic names) pass -- there is nothing to contradict.
    """
    identity = {
        t for t in _tokens(
            f"{texture.texture_id} {texture.semantic_type or ''} {texture.display_name}"
        )
        if t not in _GENERIC_IDENTITY
    }
    if not identity:
        return True
    desc_tokens = _tokens(description)
    return any(
        i in d or d in i for i in identity for d in desc_tokens
    )


def match_texture(
    query: str,
    target_uv: UVInfo | None = None,
    prefer_authentic: bool = False,
    threshold: float = MATCH_THRESHOLD,
    must_relate_to: str | None = None,
) -> TextureMatch | None:
    """Best compatible texture for `query`, or None if nothing clears the bar.

    target_uv: the destination object's UV info -- geometrically incompatible
    textures are excluded outright (an equirect Earth map must not "match"
    onto an unwrapped bracket no matter how good the words look).

    must_relate_to: the object's own description; candidates whose specific
    identity words don't appear in it are excluded (see identity_relates).
    """
    query_tokens = _tokens(query)
    best: TextureMatch | None = None
    for texture in load_index():
        if target_uv is not None and not mapping_fits(texture.mapping, target_uv):
            continue
        if must_relate_to is not None and not identity_relates(texture, must_relate_to):
            continue
        score = score_texture(query_tokens, texture)
        if prefer_authentic and texture.authentic:
            score *= 1.25
        if score >= threshold and (best is None or score > best.score):
            best = TextureMatch(texture=texture, score=score)
    return best
