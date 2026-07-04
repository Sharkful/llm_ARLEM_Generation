"""NASA 3D Resources source (github.com/nasa/NASA-3D-Resources).

Second allowlisted model source (verified 2026-07): NASA's official 3D
model library, public domain per NASA media usage guidelines, published as
a GitHub repository. Machine-accessible without scraping: one Git tree API
call lists every file (1.6k entries, untruncated), and files download from
raw.githubusercontent.com. Each model folder ships a same-name .glb plus a
.png/.jpg thumbnail -- ideal for the candidate cards.

This is the source that covers spacecraft/NASA hardware, which Poly Haven
(props/furniture/nature) does not -- the gap the Apollo scene exposed.

The tree is cached locally (library/cache/nasa3d_tree.json, TTL 7 days) so
repeated searches cost zero API calls -- the unauthenticated GitHub API
allows only 60 requests/hour.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

import config

_REPO = "nasa/NASA-3D-Resources"
_BRANCH = "master"
_TREE_URL = f"https://api.github.com/repos/{_REPO}/git/trees/{_BRANCH}?recursive=1"
_RAW_BASE = f"https://raw.githubusercontent.com/{_REPO}/{_BRANCH}/"
_HTML_BASE = f"https://github.com/{_REPO}/tree/{_BRANCH}/"
_CACHE_TTL_S = 7 * 24 * 3600
_TIMEOUT_S = 60

# Model files we can actually intake (mesh_processor formats).
_MODEL_EXTS = {".glb", ".obj", ".fbx", ".stl"}
_THUMB_EXTS = {".png", ".jpg", ".jpeg"}


class Nasa3DError(RuntimeError):
    pass


def _cache_path() -> Path:
    return config.LIBRARY_DIR / "cache" / "nasa3d_tree.json"


def _load_tree(force_refresh: bool = False) -> list[dict]:
    cache = _cache_path()
    if not force_refresh and cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8-sig"))
        if time.time() - payload.get("fetched_at", 0) < _CACHE_TTL_S:
            return payload["tree"]
    try:
        response = requests.get(_TREE_URL, timeout=_TIMEOUT_S)
        response.raise_for_status()
        tree = response.json().get("tree", [])
    except requests.RequestException as exc:
        if cache.is_file():  # stale cache beats no data
            return json.loads(cache.read_text(encoding="utf-8-sig"))["tree"]
        raise Nasa3DError(f"NASA 3D Resources tree fetch failed: {exc}") from exc
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"fetched_at": time.time(), "tree": tree}), encoding="utf-8"
    )
    return tree


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2]


class Nasa3DHit(dict):
    """path, name, size, score, thumbnail_path (dict keeps this module light)."""


def search_nasa3d(query: str, limit: int = 5) -> list[dict]:
    """Token-match query words against model file paths. Returns dicts:
    {path, name, folder, size, score, thumbnail_url, html_url, raw_url}."""
    query_tokens = [t for t in query.lower().split() if t]
    if not query_tokens:
        return []
    tree = _load_tree()

    models = [
        e for e in tree
        if e.get("type") == "blob" and Path(e["path"]).suffix.lower() in _MODEL_EXTS
    ]
    thumbs_by_folder: dict[str, str] = {}
    for e in tree:
        p = Path(e["path"])
        if e.get("type") == "blob" and p.suffix.lower() in _THUMB_EXTS:
            thumbs_by_folder.setdefault(str(p.parent), e["path"])

    hits = []
    for entry in models:
        path = Path(entry["path"])
        haystack = _tokens(str(path))
        score = sum(
            2 if any(t in h or h in t for h in _tokens(path.stem)) else
            1 if any(t in h or h in t for h in haystack) else 0
            for t in query_tokens
        )
        if score == 0:
            continue
        thumb = thumbs_by_folder.get(str(path.parent))
        hits.append({
            "path": entry["path"],
            "name": path.stem,
            "folder": str(path.parent),
            "size": entry.get("size", 0),
            "score": score,
            "thumbnail_url": _RAW_BASE + quote(thumb) if thumb else "",
            "html_url": _HTML_BASE + quote(str(path.parent)),
            "raw_url": _RAW_BASE + quote(entry["path"]),
        })
    hits.sort(key=lambda h: (-h["score"], h["size"]))
    return hits[:limit]


def fetch_nasa3d_to_intake(
    path: str,
    intake_id: str,
    target_size_m: float | None = None,
) -> Path:
    """Download one chosen model file into intake/<intake_id>/ with a
    pre-filled public-domain source.json, ready for the normal 6a intake."""
    intake_dir = config.INTAKE_DIR / intake_id
    if intake_dir.exists() and any(intake_dir.iterdir()):
        raise Nasa3DError(
            f"Intake folder {intake_dir} already exists and is not empty -- "
            "pick a different id or clear it first."
        )
    intake_dir.mkdir(parents=True, exist_ok=True)
    dest = intake_dir / Path(path).name
    try:
        with requests.get(_RAW_BASE + quote(path), timeout=_TIMEOUT_S, stream=True) as response:
            response.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    except requests.RequestException as exc:
        raise Nasa3DError(f"Download failed ({path}): {exc}") from exc

    source = {
        "display_name": Path(path).stem,
        "license": "public-domain",
        "source_site": "github.com/nasa/NASA-3D-Resources",
        "source_url": _HTML_BASE + quote(str(Path(path).parent)),
        "original_author": "NASA",
        "target_size_m": target_size_m,
        "pivot": "base_center",
        "tags": [t for t in _tokens(Path(path).stem)][:8],
        "notes": "NASA 3D Resources, public domain per NASA media usage guidelines.",
    }
    (intake_dir / "source.json").write_text(
        json.dumps(source, indent=2), encoding="utf-8"
    )
    return intake_dir
