"""Thingiverse source (api.thingiverse.com), added 2026-07.

Third allowlisted model source after Poly Haven (all-CC0) and NASA 3D
Resources (all public-domain per NASA guidelines). Unlike those two,
Thingiverse's per-item license genuinely varies (CC0, CC-BY, CC-BY-SA,
CC-BY-NC, "All Rights Reserved", ...), so map_license() below is
deliberately fail-closed: only labels we can map with confidence become an
allowlisted string (cc0/cc-by/cc-by-sa); anything else passes through UNCHANGED so
pipeline/external_intake.py's `_LICENSE_ALLOWLIST` gate rejects it at
intake time rather than this module silently laundering an ambiguous or
restrictive license into something permissive.

Content skew: Thingiverse is a 3D-printing repository, so hits are STL/OBJ
geometry only -- no baked color or PBR textures. Fine for mechanical/parts
objects (this pipeline's OpenSCAD "parametric" route's actual subject
matter), less useful for richly-colored/textured subjects.

Policy (2026-07, confirmed with the user when this source was added):
Thingiverse candidates are NEVER auto-adopted by pipeline/source_assist.py
regardless of confidence/license/size -- its license metadata hasn't earned
the same trust as Poly Haven's blanket CC0 catalog. See that module's
auto-adopt condition and each SourceCandidate's `auto_downloadable=False`
below (belt-and-suspenders: two independent places block it).

Requires config.THINGIVERSE_APP_TOKEN (register an app at
https://www.thingiverse.com/developers) -- Poly Haven/NASA3D need no key;
Thingiverse's API does.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

import requests
from pydantic import BaseModel

import config

API_BASE = "https://api.thingiverse.com"
SITE = "thingiverse.com"
_TIMEOUT_S = 30

# Model files mesh_processor.py can actually normalize. Thingiverse things
# often ship .3mf/.gcode/.zip alongside -- skip anything we can't intake
# rather than downloading a file that will fail Stage 4 normalization.
_MESH_EXTS = {".stl", ".obj", ".glb"}

# Thingiverse's own license labels (verbatim strings its API returns),
# lowercased, -> our internal allowlist vocabulary. Deliberately short: see
# module docstring for why unmapped labels must NOT be guessed.
_LICENSE_MAP = {
    "creative commons - public domain dedication": "cc0",
    "public domain": "cc0",
    "creative commons - attribution": "cc-by",
    "creative commons - attribution - share alike": "cc-by-sa",
}


class ThingiverseError(RuntimeError):
    pass


class ThingiverseResult(BaseModel):
    thing_id: str
    name: str
    license: str  # raw Thingiverse label -- NOT yet mapped; see map_license
    creator: str = ""
    thumbnail_url: str = ""
    public_url: str = ""
    download_count: int = 0
    score: int = 0


def _require_token() -> str:
    if not config.THINGIVERSE_APP_TOKEN:
        raise ThingiverseError(
            "THINGIVERSE_APP_TOKEN is not set -- register an app at "
            "https://www.thingiverse.com/developers and add the token to .env."
        )
    return config.THINGIVERSE_APP_TOKEN


def _get_json(path: str, **params):
    token = _require_token()
    url = f"{API_BASE}{path}"
    try:
        response = requests.get(
            url, params={"access_token": token, **params}, timeout=_TIMEOUT_S
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise ThingiverseError(f"Thingiverse API request failed ({path}): {exc}") from exc


def map_license(raw_license: str) -> str:
    """Fail-closed license mapping -- see module docstring. Returns the
    normalized allowlist string when confidently mappable, otherwise the
    raw label unchanged (so the intake license gate rejects it). Public
    because source_assist.py also maps candidate licenses through this
    before applying its unrestricted-license display policy."""
    return _LICENSE_MAP.get((raw_license or "").strip().lower(), raw_license)


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2]


def _score(query_tokens: list[str], name: str, tags: list[str]) -> int:
    """Client-side query/name/tag overlap score, computed the same way as
    nasa3d.py's _tokens()-based scoring -- Thingiverse's own /search ranks
    server-side, but source_assist.py's confidence math (_confidence())
    expects a comparable integer score across every source so candidates
    from different sites can be judged on the same scale."""
    name_tokens = _tokens(name)
    tag_tokens = _tokens(" ".join(tags))
    return sum(
        2 if any(t in h or h in t for h in name_tokens) else
        1 if any(t in h or h in t for h in tag_tokens) else 0
        for t in query_tokens
    )


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(
            url, params={"access_token": _require_token()}, timeout=_TIMEOUT_S, stream=True,
        ) as response:
            response.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    except requests.RequestException as exc:
        raise ThingiverseError(f"Download failed ({url}): {exc}") from exc


def search_thingiverse(
    query: str,
    limit: int = 5,
    allowed_licenses: frozenset[str] | set[str] | None = None,
    search_depth: int = 30,
) -> tuple[list[ThingiverseResult], int]:
    """Search Thingiverse and fetch each hit's authoritative per-thing
    detail record (license, creator) so a human reviewing candidates sees
    the real current license -- search-result summaries aren't trusted for
    this since Thingiverse's search endpoint doesn't reliably echo it.

    Returns (results, license_hidden). When `allowed_licenses` is given
    (mapped labels, e.g. {"cc0", "cc-by"}), out-of-set hits are skipped and
    counted in license_hidden instead of returned. The search page is
    fetched `search_depth` deep (one API call) because restricted licenses
    dominate Thingiverse's top results (a "human heart" survey found 0/30
    CC0, 15/30 CC-BY) -- filtering only the top `limit` hits would starve
    the candidate list even when in-tier models exist a few ranks down.
    Detail calls stop as soon as `limit` results are collected, so the
    extra depth costs nothing when the top hits already qualify."""
    query_tokens = [t for t in query.lower().split() if t]
    if not query_tokens:
        raise ThingiverseError("Empty search query.")

    payload = _get_json(f"/search/{quote(query)}/", per_page=max(limit, search_depth))
    hits = payload.get("hits", []) if isinstance(payload, dict) else payload

    results: list[ThingiverseResult] = []
    license_hidden = 0
    for hit in hits:
        if len(results) >= limit:
            break
        thing_id = str(hit.get("id"))
        if not thing_id or thing_id == "None":
            continue
        detail = _get_json(f"/things/{thing_id}/")
        raw_license = detail.get("license", "unknown")
        if allowed_licenses is not None and map_license(raw_license) not in allowed_licenses:
            license_hidden += 1
            continue
        name = detail.get("name") or hit.get("name") or thing_id
        tags = [t.get("name", "") for t in detail.get("tags", []) if isinstance(t, dict)]
        results.append(ThingiverseResult(
            thing_id=thing_id,
            name=name,
            license=raw_license,
            creator=((detail.get("creator") or {}).get("name")) or "",
            thumbnail_url=hit.get("thumbnail") or detail.get("thumbnail", "") or "",
            public_url=detail.get("public_url") or f"https://www.thingiverse.com/thing:{thing_id}",
            download_count=detail.get("download_count", 0) or 0,
            score=_score(query_tokens, name, tags),
        ))
    results.sort(key=lambda r: (-r.score, -r.download_count))
    return results, license_hidden


def fetch_thingiverse_to_intake(
    thing_id: str,
    intake_id: str | None = None,
    target_size_m: float | None = None,
) -> Path:
    """Download one human-chosen thing's first usable mesh file into
    intake/<intake_id>/ with a pre-filled source.json, ready for the normal
    Stage 6a `intake` command. The license gate there is the real
    enforcement point -- this only records what Thingiverse reports."""
    intake_id = intake_id or f"thing_{thing_id}"
    intake_dir = config.INTAKE_DIR / intake_id
    if intake_dir.exists() and any(intake_dir.iterdir()):
        raise ThingiverseError(
            f"Intake folder {intake_dir} already exists and is not empty -- "
            "pick a different intake_id or clear it first."
        )

    detail = _get_json(f"/things/{thing_id}/")
    files = _get_json(f"/things/{thing_id}/files/")
    if not isinstance(files, list):
        files = []
    usable = [f for f in files if Path(f.get("name", "")).suffix.lower() in _MESH_EXTS]
    if not usable:
        found = ", ".join(f.get("name", "?") for f in files) or "none"
        raise ThingiverseError(
            f"No .stl/.obj/.glb file found for thing {thing_id!r} (found: {found})."
        )
    chosen = usable[0]

    intake_dir.mkdir(parents=True, exist_ok=True)
    dest = intake_dir / chosen["name"]
    _download_file(chosen["download_url"], dest)

    raw_license = detail.get("license", "") or ""
    mapped_license = map_license(raw_license)
    was_mapped = raw_license.strip().lower() in _LICENSE_MAP
    creator = ((detail.get("creator") or {}).get("name")) or None

    notes = (
        "" if target_size_m
        else "FILL IN target_size_m (real-world largest dimension, meters) before running intake."
    )
    if not was_mapped:
        notes = (
            f"{notes} NOTE: raw Thingiverse license was {raw_license!r}, which does not map to "
            "an allowlisted license -- `cli.py intake` will reject this unless you have "
            "independently verified the terms and corrected this field."
        ).strip()

    source = {
        "display_name": detail.get("name", thing_id),
        "license": mapped_license,
        "source_site": SITE,
        "source_url": detail.get("public_url") or f"https://www.thingiverse.com/thing:{thing_id}",
        "original_author": creator,
        "target_size_m": target_size_m,
        "pivot": "base_center",
        "tags": [
            t.get("name") for t in detail.get("tags", [])
            if isinstance(t, dict) and t.get("name")
        ][:8],
        "notes": notes,
    }
    (intake_dir / "source.json").write_text(
        json.dumps(source, indent=2), encoding="utf-8"
    )
    return intake_dir
