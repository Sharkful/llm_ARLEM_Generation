"""Sourcing assist for the `imported` route (policy amended 2026-07).

Original Stage 6b policy required a human to run every search and approve
every download. Amended by user decision: the pipeline itself searches the
verified source allowlist and

  1. AUTO-DOWNLOADS when all three hold: the top hit matches confidently,
     its license is machine-verifiable unrestricted (Poly Haven == CC0),
     and the real-world size is known (spec.desired_size_m) -- so intake
     can complete without a human.
  2. Otherwise presents CANDIDATES -- name, thumbnail, license, source URL
     -- for one-click approval in the review app (POST /api/adopt).
  3. With no usable hits, reports SPECIFIC search URLs for verified
     sources so the user clicks a link instead of hunting.

Unchanged guardrails: only allowlisted sources are ever contacted; nothing
is fetched from a source whose licensing a human hasn't verified; every
download still flows through the normal intake gate (license check,
normalization, provenance).

License display policy (2026-07, user decision): the author selects a
license tier in the review app -- see LICENSE_TIERS (CC0-only / +CC-BY
default / +CC-BY-SA; NC/ND/unknown are never offered). Out-of-tier hits
are hidden but counted, and the
count is reported in the note. Link-only browse sources (no usable API, so
they appear as search URLs, never as candidates): Smithsonian 3D,
Kenney (all CC0), Quaternius (all CC0, no site search), and OpenGameArt
(mixed licenses -- its link is pre-filtered to CC0 3D art).

Sources considered and deferred (2026-07 research, see the planning
session that added Thingiverse): TurboSquid's public API is seller-side
publishing only, no buyer search/download; CGTrader has a real buyer API
but needs the user to register and its license/cost terms weren't
confirmed; Sketchfab/Fab is mid-migration after Epic folded Sketchfab into
Fab, with no confirmed stable public replacement API yet; Free3D has no
public API and inconsistent per-item license labeling. Re-evaluate any of
these before adding them -- don't assume this research is still current.
"""
from __future__ import annotations

import re
from typing import Literal, Optional
from urllib.parse import quote_plus

from pydantic import BaseModel, Field

from models.log_models import LLMCallEntry
from models.spec_models import AssetSpec
from pipeline.external_intake import IntakeError, IntakeResult, intake_asset
from pipeline.nasa3d import Nasa3DError, fetch_nasa3d_to_intake, search_nasa3d
from pipeline.polyhaven import PolyHavenSearchError, fetch_to_intake, search_models
from pipeline.source_relevance import review_candidates
from pipeline.thingiverse import (
    ThingiverseError,
    fetch_thingiverse_to_intake,
    map_license,
    search_thingiverse,
)

# Top-hit token coverage (0..1) required to download without asking.
# (The >=2 identity-word guard below is the second condition.)
AUTO_ADOPT_CONFIDENCE = 0.5
# Candidates below this are noise (moon rocks offered for a lunar lander --
# found the hard way); show only the search links instead.
MIN_CANDIDATE_CONFIDENCE = 0.25
MAX_CANDIDATES = 5

# Candidate license display policy (2026-07, user decisions): the author
# picks a TIER in the review app (Create/Worklist pull-down); each tier is
# cumulative and only includes licenses whose obligations this pipeline can
# actually discharge (attribution is recorded at intake and flagged on the
# catalog entry). NC/ND/unknown licenses are never offered at any tier --
# NC restricts commercial contexts and ND forbids the derivative meshes
# this pipeline exists to make. Default is cc-by: a survey of Thingiverse's
# top "human heart" hits found 0/30 CC0 but 15/30 CC-BY, so CC0-only
# starves organic/anatomy subjects. Hidden hits are COUNTED and reported in
# the sourcing note so a license-filtered search never masquerades as "no
# results" (learned from the missing-Thingiverse-token episode, where a
# swallowed condition made a working search look empty).
_CC0_SET = frozenset({"cc0", "cc0-1.0", "public-domain"})
_CC_BY_SET = _CC0_SET | {"cc-by", "cc-by-3.0", "cc-by-4.0"}
_CC_BY_SA_SET = _CC_BY_SET | {"cc-by-sa", "cc-by-sa-3.0", "cc-by-sa-4.0"}
LICENSE_TIERS: dict[str, dict] = {
    "cc0": {
        "label": "CC0 / public domain only",
        "description": "No conditions at all: free for any use, no credit required.",
        "licenses": _CC0_SET,
    },
    "cc-by": {
        "label": "CC0 + CC-BY (credit the creator)",
        "description": (
            "Also allows attribution licenses: free for any use as long as the "
            "original creator is credited. The pipeline records the author at "
            "intake and marks the asset so bundles can include attribution."
        ),
        "licenses": _CC_BY_SET,
    },
    "cc-by-sa": {
        "label": "CC0 + CC-BY + CC-BY-SA (credit + share-alike)",
        "description": (
            "Also allows share-alike: credit the creator, AND any modified "
            "version you distribute (the pipeline's scaled/decimated copies "
            "count) must be shared under the same CC-BY-SA license."
        ),
        "licenses": _CC_BY_SA_SET,
    },
}
DEFAULT_LICENSE_TIER = "cc-by"

# Verified places a human can look when the API allowlist has nothing.
# (Search-URL templates only -- the pipeline never scrapes these.)
# Kenney and Quaternius are blanket-CC0 catalogs with no usable search/API
# (Kenney's ?q= filter runs client-side; Quaternius has no search at all,
# hence no {q}); OpenGameArt's licensing is mixed, so its link pre-filters
# to 3D art (type tid 10) + CC0 (license tid 4) -- IDs read from its
# advanced-search form, 2026-07.
_SEARCH_URL_TEMPLATES = [
    ("Poly Haven (CC0)", "https://polyhaven.com/all?s={q}"),
    ("NASA 3D Resources (public domain)", "https://nasa3d.arc.nasa.gov/search?q={q}"),
    ("Smithsonian 3D (check per-item license)", "https://3d.si.edu/explore?search={q}"),
    ("Thingiverse (check per-item license)", "https://www.thingiverse.com/search?q={q}"),
    ("Kenney (CC0)", "https://kenney.nl/assets?q={q}"),
    ("Quaternius (CC0, browse packs)", "https://quaternius.com/"),
    ("OpenGameArt (pre-filtered to CC0 3D art)",
     "https://opengameart.org/art-search-advanced?keys={q}"
     "&field_art_type_tid%5B%5D=10&field_art_licenses_tid%5B%5D=4"),
]


def _license_allowed(license_str: str, allowed: frozenset[str]) -> bool:
    return license_str.strip().lower().replace(" ", "-") in allowed


class SourceCandidate(BaseModel):
    source: Literal["polyhaven", "nasa3d", "thingiverse"] = "polyhaven"
    source_id: str  # polyhaven asset id, nasa3d repo file path, or thingiverse thing id
    name: str
    license: str
    url: str
    thumbnail_url: str = ""
    download_count: int = 0
    confidence: float = 0.0  # query-token coverage, 0..1
    auto_downloadable: bool = True  # CC0/PD source; still needs size + confidence
    # Set by the relevance-review gate (pipeline/source_relevance.py) --
    # None means review wasn't run or failed, not "no verdict reached".
    relevance_match: Optional[bool] = None
    relevance_reasoning: str = ""
    relevance_reviewed_with_image: bool = False


class SourcingAssist(BaseModel):
    query: str
    candidates: list[SourceCandidate] = Field(default_factory=list)
    search_urls: list[str] = Field(default_factory=list)
    auto_adopted: Optional[IntakeResult] = None
    note: str = ""
    # Hits dropped by the CANDIDATE_LICENSES display policy -- surfaced in
    # the note so "0 candidates" is distinguishable from "hits existed but
    # their licenses are restricted".
    license_filtered: int = 0


# Classifier-ish / generic kind values that are noise in a search query.
_GENERIC_QUERY_WORDS = {
    "composite", "parametric", "imported", "unclear", "unknown", "primitive",
    "object", "model", "asset", "scale", "realistic", "simple",
    "a", "an", "the", "with", "about", "and", "of",
}


def build_query(spec: AssetSpec, keywords: list[str] | None = None) -> str:
    """Short, identity-focused search terms.

    Prefers LLM-authored `keywords` (GeometryClassification.search_keywords,
    carried on ResolutionRecord) when given -- these name what the object
    actually IS ("snowman", "carrot nose"), unlike positional extraction
    from a full sentence, which surfaces grammatical filler for anything
    longer than a terse description (found the hard way: 'a friendly
    snowman made from...' -> 'snowman friendly made from'). Falls back to
    the mechanical semantic_type/kind/description extraction only when no
    keywords were supplied (e.g. classify_geometry_fn was overridden with a
    stub in a test, or this is called directly without going through the
    classifier).
    """
    if keywords:
        words: list[str] = []
        for phrase in keywords:
            words.extend(re.split(r"[^a-z0-9]+", phrase.lower()))
        cleaned = [w for w in dict.fromkeys(words) if len(w) > 2]
        if cleaned:
            return " ".join(cleaned[:6])

    words = []
    for field in (spec.semantic_type, spec.kind):
        if field:
            words.extend(re.split(r"[^a-z0-9]+", field.lower()))
    words.extend(re.split(r"[^a-z0-9]+", spec.description.lower()))
    cleaned = [
        w for w in dict.fromkeys(words)
        if len(w) > 2 and w not in _GENERIC_QUERY_WORDS and not any(ch.isdigit() for ch in w)
    ]
    return " ".join(cleaned[:4])


def _confidence(score: int, query: str) -> float:
    tokens = [t for t in query.split() if t]
    return round(score / (2 * max(len(tokens), 1)), 3)  # strong hit = 2/token


def search_urls_for(query: str) -> list[str]:
    q = quote_plus(query)
    return [f"{label}: {template.format(q=q)}" for label, template in _SEARCH_URL_TEMPLATES]


def assist_imported(
    spec: AssetSpec,
    intake_id: str,
    keywords: list[str] | None = None,
    manual_query: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    license_tier: str | None = None,
) -> tuple[SourcingAssist, list[LLMCallEntry]]:
    """Run the sourcing ladder for an `imported`-classified spec.

    keywords: LLM-authored identity words (ResolutionRecord.search_keywords)
              -- preferred over mechanical extraction.
    manual_query: a human-typed override (from the review app's "search
              again" box), used verbatim, highest priority.
    license_tier: a LICENSE_TIERS key from the review app's pull-down;
              None/unknown falls back to DEFAULT_LICENSE_TIER.

    Never raises for network trouble -- a failed search degrades to
    search URLs, since the caller's fallback (flag for human) must survive
    offline operation.
    """
    tier = license_tier if license_tier in LICENSE_TIERS else DEFAULT_LICENSE_TIER
    allowed = LICENSE_TIERS[tier]["licenses"]
    query = manual_query.strip() if manual_query and manual_query.strip() else build_query(spec, keywords=keywords)
    result = SourcingAssist(query=query, search_urls=search_urls_for(query))
    review_logs: list[LLMCallEntry] = []

    candidates: list[SourceCandidate] = []
    errors: list[str] = []
    license_filtered = 0
    try:
        for hit in search_models(query, limit=MAX_CANDIDATES, asset_type="models"):
            if not _license_allowed(hit.license, allowed):  # always CC0 today; guards a future change
                license_filtered += 1
                continue
            candidates.append(SourceCandidate(
                source="polyhaven",
                source_id=hit.asset_id,
                name=hit.name,
                license=hit.license,
                url=f"https://polyhaven.com/a/{hit.asset_id}",
                thumbnail_url=hit.thumbnail_url,
                download_count=hit.download_count,
                confidence=_confidence(hit.score, query),
            ))
    except PolyHavenSearchError as exc:
        errors.append(str(exc))
    try:
        for hit in search_nasa3d(query, limit=MAX_CANDIDATES):
            candidates.append(SourceCandidate(
                source="nasa3d",
                source_id=hit["path"],
                name=hit["name"],
                license="public-domain",
                url=hit["html_url"],
                thumbnail_url=hit["thumbnail_url"],
                confidence=_confidence(hit["score"], query),
            ))
    except Nasa3DError as exc:
        errors.append(str(exc))
    try:
        # Thingiverse licensing is per-item, and restricted licenses dominate
        # its top results (0/30 CC0 for "human heart"), so the license filter
        # runs INSIDE the search with a deeper hit pool -- otherwise filtering
        # the top 5 leaves one or zero candidates even when in-tier models
        # exist a few ranks down (found the hard way).
        tv_hits, tv_hidden = search_thingiverse(
            query, limit=MAX_CANDIDATES, allowed_licenses=allowed,
        )
        license_filtered += tv_hidden
        for hit in tv_hits:
            candidates.append(SourceCandidate(
                source="thingiverse",
                source_id=hit.thing_id,
                name=hit.name,
                license=map_license(hit.license),
                url=hit.public_url,
                thumbnail_url=hit.thumbnail_url,
                download_count=hit.download_count,
                confidence=_confidence(hit.score, query),
                # Never auto-adopt (2026-07 policy) -- Thingiverse's mixed
                # licensing hasn't earned the same trust as Poly Haven's
                # blanket CC0 catalog. Also enforced below in the auto-adopt
                # condition itself; this is the second, independent guard.
                auto_downloadable=False,
            ))
    except ThingiverseError as exc:
        errors.append(str(exc))

    result.license_filtered = license_filtered
    # Appended to every note so a license-filtered search is never mistaken
    # for an empty one.
    hidden = (
        f" ({license_filtered} hit(s) hidden by the license policy: showing "
        f"{LICENSE_TIERS[tier]['label']}.)"
        if license_filtered else ""
    )

    if errors and not candidates:
        result.note = (
            f"Source search unavailable ({'; '.join(errors)}); use the search links.{hidden}"
        )
        return result, review_logs

    # Junk suppression: a low-scoring hit is worse than no suggestion.
    candidates = [c for c in candidates if c.confidence >= MIN_CANDIDATE_CONFIDENCE]
    candidates.sort(key=lambda c: -c.confidence)
    candidates = candidates[:MAX_CANDIDATES]

    if not candidates:
        result.note = (
            "No confident allowlisted-source match. Try the search links; download "
            f"the file into intake/{intake_id}/ and run `cli.py intake {intake_id}`.{hidden}"
        )
        return result, review_logs

    # Relevance review (pipeline/source_relevance.py): keyword overlap alone
    # cannot tell a real match from something merely NAMED after the query
    # ("Snowman Craters" is geology, not a snowman). This is the gate that
    # actually checks intent, not just words.
    reviews, review_logs = review_candidates(spec.description, candidates, provider=provider, model=model)
    for candidate, review in zip(candidates, reviews):
        candidate.relevance_match = review.matches
        candidate.relevance_reasoning = review.reasoning
        candidate.relevance_reviewed_with_image = review.reviewed_with_image
    # Keep non-matches out entirely (matches=False); keep unreviewed
    # (matches=None, e.g. a transient API error) since hiding everything on
    # a review hiccup would be worse than showing an unvetted candidate.
    candidates = [c for c in candidates if c.relevance_match is not False]
    result.candidates = candidates

    if not result.candidates:
        result.note = (
            f"Found candidate(s) for {query!r} but an automatic relevance review "
            "determined none actually depict the requested object -- see reasoning "
            "in the log. Try 'search again' with different keywords, or the search "
            f"links below.{hidden}"
        )
        return result, review_logs

    top = result.candidates[0]
    # Auto-adopt safety: >=2 distinct query words in the candidate's own
    # id/name (so one generic token can't trigger a wrong unattended
    # download) AND an explicit relevance-review pass (matches=True, not
    # just "not rejected") -- auto-adopt is the one place we act without a
    # human, so it holds to a higher bar than merely being shown.
    name_text = f"{top.source_id} {top.name}".lower()
    identity_hits = sum(1 for t in set(query.split()) if t and t in name_text)
    if (
        top.confidence >= AUTO_ADOPT_CONFIDENCE
        and identity_hits >= 2
        and top.relevance_match is True
        and top.auto_downloadable
        # Belt-and-suspenders alongside auto_downloadable=False above: only
        # Poly Haven's blanket-CC0 catalog is trusted enough to auto-adopt
        # (2026-07 policy) -- new sources' license metadata must prove
        # reliable in practice before this is ever relaxed for them too.
        and top.source == "polyhaven"
        and spec.desired_size_m
    ):
        try:
            result.auto_adopted = adopt_candidate(
                top, intake_id, target_size_m=spec.desired_size_m
            )
            result.note = (
                f"Auto-downloaded {top.source_id!r} ({top.license}, confidence "
                f"{top.confidence:.2f}, relevance-reviewed) and intook it as {intake_id!r}."
            )
        except (PolyHavenSearchError, Nasa3DError, ThingiverseError, IntakeError, ValueError) as exc:
            result.note = f"Auto-download of {top.source_id!r} failed ({exc}); approve manually."
    else:
        missing = []
        if top.confidence < AUTO_ADOPT_CONFIDENCE:
            missing.append(f"match confidence {top.confidence:.2f} < {AUTO_ADOPT_CONFIDENCE}")
        if identity_hits < 2:
            missing.append("match not specific enough for unattended download")
        if top.relevance_match is not True:
            missing.append("relevance review did not confirm a match")
        if top.source != "polyhaven":
            missing.append(f"{top.source} candidates always require manual approval")
        if not spec.desired_size_m:
            missing.append("no real-world size stated")
        result.note = (
            f"{len(result.candidates)} candidate(s) found -- approve one in the review "
            f"app ({'; '.join(missing)}).{hidden}"
        )
    return result, review_logs


def adopt_candidate(
    candidate: SourceCandidate,
    intake_id: str,
    target_size_m: float,
) -> IntakeResult:
    """Fetch a chosen candidate and run the normal 6a intake gate on it."""
    if candidate.source == "nasa3d":
        fetch_nasa3d_to_intake(
            candidate.source_id, intake_id=intake_id, target_size_m=target_size_m
        )
    elif candidate.source == "thingiverse":
        fetch_thingiverse_to_intake(
            candidate.source_id, intake_id=intake_id, target_size_m=target_size_m
        )
    else:
        fetch_to_intake(
            candidate.source_id, intake_id=intake_id, target_size_m=target_size_m
        )
    # Every fetcher pre-fills source.json; the full license/provenance/
    # normalization gate runs here as always.
    return intake_asset(intake_id)
