"""Stage 6b: semi-automated external asset search assist -- Poly Haven only.

The allowlist is deliberately this one source: Poly Haven is confirmed
all-CC0, has a public no-auth API, and the requirements doc (section 3.2)
explicitly excludes unrestricted external retrieval. Do NOT add Sketchfab
or other sources here until a human has verified that source's API terms
and per-asset license metadata -- that verification, not code, is the
gating step.

Human-in-the-loop by construction: `search_models()` is read-only;
`fetch_to_intake()` only runs when a human explicitly names the asset to
download (the CLI `fetch-external <id>` invocation *is* the confirmation).
Nothing in this module is called automatically by the resolver or any
other stage. Downloads land in intake/<asset_id>/ with a pre-filled
source.json, and Stage 6a's normal `intake` flow -- license gate included
-- still runs afterwards; the search assist never bypasses it.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests
from pydantic import BaseModel, Field

import config

API_BASE = "https://api.polyhaven.com"
SITE = "polyhaven.com"
LICENSE = "CC0"  # every Poly Haven asset; the API has no per-asset license field
_TIMEOUT_S = 30
_MODEL_TYPE = "models"


class PolyHavenSearchError(RuntimeError):
    pass


class PolyHavenResult(BaseModel):
    asset_id: str
    name: str
    license: str = LICENSE
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    download_count: int = 0
    polycount: int | None = None
    thumbnail_url: str = ""
    score: int = 0


class PolyHavenFile(BaseModel):
    file_format: str  # e.g. "gltf", "fbx", "blend"
    resolution: str  # e.g. "1k", "2k"
    url: str
    size_bytes: int = 0
    include_count: int = 0  # companion files (textures/bins) downloaded alongside


def _get_json(path: str) -> dict:
    url = f"{API_BASE}{path}"
    try:
        response = requests.get(url, timeout=_TIMEOUT_S)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise PolyHavenSearchError(f"Poly Haven API request failed ({url}): {exc}") from exc


def search_models(query: str, limit: int = 10, asset_type: str = _MODEL_TYPE) -> list[PolyHavenResult]:
    """Token-match `query` against Poly Haven's listing (client-side: the
    API has no search endpoint). Read-only -- downloads nothing.
    asset_type: "models" or "textures" (Stage 6c.5) -- same API shape."""
    assets = _get_json(f"/assets?t={asset_type}")
    tokens = [t for t in query.lower().split() if t]
    if not tokens:
        raise PolyHavenSearchError("Empty search query.")

    results: list[PolyHavenResult] = []
    for asset_id, info in assets.items():
        haystack_strong = f"{asset_id} {info.get('name', '')}".lower()
        haystack_weak = " ".join(
            [*info.get("tags", []), *info.get("categories", [])]
        ).lower()
        score = sum(
            (2 if token in haystack_strong else 0) + (1 if token in haystack_weak else 0)
            for token in tokens
        )
        if score == 0:
            continue
        results.append(
            PolyHavenResult(
                asset_id=asset_id,
                name=info.get("name", asset_id),
                authors=sorted(info.get("authors", {})),
                categories=info.get("categories", []),
                tags=info.get("tags", []),
                download_count=info.get("download_count", 0),
                polycount=info.get("polycount"),
                thumbnail_url=(
                    info.get("thumbnail_url")
                    or f"https://cdn.polyhaven.com/asset_img/thumbs/{asset_id}.png?width=256"
                ),
                score=score,
            )
        )

    results.sort(key=lambda r: (-r.score, -r.download_count))
    return results[:limit]


def list_model_files(asset_id: str) -> list[PolyHavenFile]:
    """List downloadable format/resolution options for one model."""
    tree = _get_json(f"/files/{asset_id}")
    options: list[PolyHavenFile] = []
    for file_format, res_map in tree.items():
        if not isinstance(res_map, dict):
            continue
        for resolution, entries in res_map.items():
            if not isinstance(entries, dict):
                continue
            # Each resolution holds one-or-more named entries; the payload
            # with a "url" is the main file, its "include" map lists
            # companion files (textures, .bin) with library-relative paths.
            for payload in entries.values():
                if isinstance(payload, dict) and "url" in payload:
                    options.append(
                        PolyHavenFile(
                            file_format=file_format,
                            resolution=resolution,
                            url=payload["url"],
                            size_bytes=payload.get("size", 0),
                            include_count=len(payload.get("include", {})),
                        )
                    )
                    break
    return options


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, timeout=_TIMEOUT_S, stream=True) as response:
            response.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    except requests.RequestException as exc:
        raise PolyHavenSearchError(f"Download failed ({url}): {exc}") from exc


def fetch_to_intake(
    asset_id: str,
    intake_id: str | None = None,
    file_format: str = "gltf",
    resolution: str = "1k",
    target_size_m: float | None = None,
) -> tuple[Path, list[Path]]:
    """Download one human-chosen model (main file + includes) into
    intake/<intake_id>/ and pre-fill its source.json.

    Only ever invoked by an explicit human action naming `asset_id` --
    never called by the resolver or any automated stage. After this, the
    human reviews/completes source.json (target_size_m especially, if not
    passed here) and runs the normal Stage 6a `intake` command.
    """
    intake_id = intake_id or asset_id
    assets = _get_json(f"/assets?t={_MODEL_TYPE}")
    if asset_id not in assets:
        raise PolyHavenSearchError(
            f"{asset_id!r} is not a Poly Haven model id -- use search-external first."
        )
    info = assets[asset_id]

    tree = _get_json(f"/files/{asset_id}")
    res_map = tree.get(file_format)
    if not isinstance(res_map, dict) or resolution not in res_map:
        available = ", ".join(
            f"{fmt}/{res}"
            for fmt, rm in tree.items() if isinstance(rm, dict)
            for res in rm
        )
        raise PolyHavenSearchError(
            f"No {file_format}/{resolution} download for {asset_id!r}. Available: {available}"
        )
    payload = next(
        (p for p in res_map[resolution].values() if isinstance(p, dict) and "url" in p),
        None,
    )
    if payload is None:
        raise PolyHavenSearchError(
            f"Unrecognized file listing shape for {asset_id!r} {file_format}/{resolution}."
        )

    intake_dir = config.INTAKE_DIR / intake_id
    if intake_dir.exists() and any(intake_dir.iterdir()):
        raise PolyHavenSearchError(
            f"Intake folder {intake_dir} already exists and is not empty -- "
            "pick a different --intake-id or clear it first."
        )
    main_name = payload["url"].rsplit("/", 1)[-1]
    downloaded: list[Path] = []
    _download_file(payload["url"], intake_dir / main_name)
    downloaded.append(intake_dir / main_name)
    for rel_path, inc in payload.get("include", {}).items():
        if isinstance(inc, dict) and "url" in inc:
            dest = intake_dir / rel_path
            # Guard against path escape from a hostile/odd include entry.
            if not dest.resolve().is_relative_to(intake_dir.resolve()):
                raise PolyHavenSearchError(f"Include path escapes intake dir: {rel_path!r}")
            _download_file(inc["url"], dest)
            downloaded.append(dest)

    source = {
        "display_name": info.get("name", asset_id),
        "license": LICENSE,
        "source_site": SITE,
        "source_url": f"https://polyhaven.com/a/{asset_id}",
        "original_author": ", ".join(sorted(info.get("authors", {}))) or None,
        "target_size_m": target_size_m,
        "pivot": "base_center",
        "tags": list(dict.fromkeys([*info.get("categories", []), *info.get("tags", [])])),
        "semantic_type": None,
        "pedagogically_critical": False,
        "notes": (
            ""
            if target_size_m
            else "FILL IN target_size_m (real-world largest dimension, meters) before running intake."
        ),
    }
    source_path = intake_dir / "source.json"
    source_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
    return source_path, downloaded


# ── Stage 6c.5: texture fetch (same human-approval contract) ─────────────

# Poly Haven map-kind keys worth keeping (skip blend/gltf/mtlx bundles).
_TEXTURE_MAP_KINDS = ["Diffuse", "nor_gl", "nor_dx", "Rough", "AO", "arm", "Displacement"]


def fetch_texture_to_intake(
    asset_id: str,
    intake_id: str | None = None,
    resolution: str = "1k",
    target_size_m: float | None = None,  # unused; textures have no size -- kept for CLI symmetry
) -> tuple[Path, list[Path]]:
    """Download one human-chosen texture's map set into intake/<intake_id>/
    with a pre-filled source.json, ready for `cli.py intake-texture`.

    Downloads every available PBR map at the chosen resolution (albedo,
    normals, roughness, AO...) -- v1 only wires albedo into materials, but
    keeping the rest costs disk while re-downloading costs a human.
    """
    intake_id = intake_id or asset_id
    assets = _get_json("/assets?t=textures")
    if asset_id not in assets:
        raise PolyHavenSearchError(
            f"{asset_id!r} is not a Poly Haven texture id -- use "
            "search-external --type texture first."
        )
    info = assets[asset_id]
    tree = _get_json(f"/files/{asset_id}")

    intake_dir = config.INTAKE_DIR / intake_id
    if intake_dir.exists() and any(intake_dir.iterdir()):
        raise PolyHavenSearchError(
            f"Intake folder {intake_dir} already exists and is not empty -- "
            "pick a different --intake-id or clear it first."
        )

    downloaded: list[Path] = []
    for kind in _TEXTURE_MAP_KINDS:
        res_map = tree.get(kind)
        if not isinstance(res_map, dict) or resolution not in res_map:
            continue
        # Prefer jpg over png for size; take whatever single format exists.
        entries = res_map[resolution]
        payload = entries.get("jpg") or entries.get("png") or next(
            (p for p in entries.values() if isinstance(p, dict) and "url" in p), None
        )
        if not (isinstance(payload, dict) and "url" in payload):
            continue
        ext = payload["url"].rsplit(".", 1)[-1].lower()
        dest = intake_dir / f"{asset_id}_{kind}_{resolution}.{ext}"
        _download_file(payload["url"], dest)
        downloaded.append(dest)
    if not downloaded:
        raise PolyHavenSearchError(
            f"No downloadable maps at resolution {resolution!r} for {asset_id!r}."
        )

    # Physical tile size: the API's `dimensions` field is millimeters.
    dims_mm = info.get("dimensions")
    tile_size_m = (
        [round(dims_mm[0] / 1000, 4), round(dims_mm[1] / 1000, 4)]
        if isinstance(dims_mm, list) and len(dims_mm) == 2
        else None
    )
    source = {
        "display_name": info.get("name", asset_id),
        "license": LICENSE,
        "mapping": "tileable",
        "source_site": SITE,
        "source_url": f"https://polyhaven.com/a/{asset_id}",
        "original_author": ", ".join(sorted(info.get("authors", {}))) or None,
        "semantic_type": None,
        "tags": list(dict.fromkeys([*info.get("categories", []), *info.get("tags", [])])),
        "tile_size_m": tile_size_m,
        "authentic": True,
        "notes": "",
    }
    source_path = intake_dir / "source.json"
    source_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
    return source_path, downloaded
