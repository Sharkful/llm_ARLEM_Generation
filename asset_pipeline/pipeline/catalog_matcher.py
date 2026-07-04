"""Stage 2: cheap-first matching of an AssetSpec against the asset catalog.

No embeddings in this pass -- exact/fuzzy tag, semantic_type, and keyword
token matching against catalog entries. Sufficient for a hand-curated
catalog of primitives/variants; if the catalog grows large enough that
string matching starts missing real matches, an embedding-based ranker can
be added alongside this without changing its call signature (it already
returns ranked candidates with confidence scores, which is what a future
embedding ranker would also need to produce).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from models.catalog_models import AssetCatalogEntry
from models.spec_models import AssetSpec

_WORD_RE = re.compile(r"[a-z0-9]+")

# Common filler words that shouldn't drive a match on their own.
_STOPWORDS = {
    "a", "an", "the", "with", "of", "and", "or", "is", "are", "to", "for",
    "on", "in", "at", "small", "large", "basic", "simple",
}

# Broad *category* words that must never trigger the semantic_type identity
# shortcut below: "furniture" matching a table's 'furniture' tag is not
# evidence the spec IS that table (found the hard way: 'worn leather
# armchair' -> semantic_type 'furniture' -> confidence 1.0 against
# small_wooden_table_01). Identity-level types ("moon", "table", "beaker")
# still shortcut; category-level ones fall through to token scoring.
_GENERIC_SEMANTIC_TYPES = {
    "furniture", "equipment", "apparatus", "object", "item", "prop",
    "decor", "decoration", "tool", "hardware", "container", "device",
    "structure", "shape", "primitive", "model", "asset",
}


@dataclass
class CatalogMatch:
    entry: AssetCatalogEntry
    confidence: float
    matched_on: list[str]


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _spec_tokens(spec: AssetSpec) -> set[str]:
    tokens: set[str] = set()
    for field in (spec.description, spec.kind, spec.semantic_type, spec.visual_style):
        tokens |= _tokenize(field)
    return tokens


def _entry_tokens(entry: AssetCatalogEntry) -> set[str]:
    tokens = set(t.lower() for t in entry.tags)
    tokens |= _tokenize(entry.display_name)
    tokens |= _tokenize(entry.asset_class)
    return tokens


def score_entry(spec: AssetSpec, entry: AssetCatalogEntry) -> CatalogMatch:
    spec_tokens = _spec_tokens(spec)
    entry_tokens = _entry_tokens(entry)

    matched = sorted(spec_tokens & entry_tokens)

    # Exact semantic_type-to-tag hit is the strongest possible signal --
    # but only for identity-level types, never broad categories (see
    # _GENERIC_SEMANTIC_TYPES).
    semantic_type = (spec.semantic_type or "").strip().lower()
    if (
        semantic_type
        and semantic_type not in _GENERIC_SEMANTIC_TYPES
        and semantic_type in entry_tokens
    ):
        return CatalogMatch(entry=entry, confidence=1.0, matched_on=[semantic_type, *matched])

    if not spec_tokens or not entry_tokens:
        return CatalogMatch(entry=entry, confidence=0.0, matched_on=[])

    # Token overlap (Jaccard-ish, weighted toward the spec side since specs
    # are short and every matched word is meaningful).
    overlap = len(spec_tokens & entry_tokens)
    token_score = overlap / max(len(spec_tokens), 1)

    # Fuzzy string similarity as a secondary, softer signal (catches near
    # misses like "cratered" vs "crater").
    fuzzy_score = difflib.SequenceMatcher(
        None, " ".join(sorted(spec_tokens)), " ".join(sorted(entry_tokens))
    ).ratio()

    confidence = round(min(1.0, 0.7 * token_score + 0.3 * fuzzy_score), 4)
    return CatalogMatch(entry=entry, confidence=confidence, matched_on=matched)


def find_candidates(
    spec: AssetSpec, catalog: list[AssetCatalogEntry], top_k: int = 5
) -> list[CatalogMatch]:
    scored = [score_entry(spec, entry) for entry in catalog]
    scored.sort(key=lambda m: m.confidence, reverse=True)
    return [m for m in scored if m.confidence > 0][:top_k]
