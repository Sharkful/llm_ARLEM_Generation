"""Stage 6c.4: manual texture intake -- image file(s) + source.json ->
library/textures/<texture_id>/ + registry entry with full provenance.

Mirrors 6a's mesh intake: same drop-folder workflow, same license gate
(shared check_license_fields), same never-silently-skip-provenance rule.
Downloaded once, indexed forever -- the archive matcher (texture_matcher)
finds it on every later request.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

import config
from models.catalog_models import ProvenanceInfo
from models.texture_models import MapKind, MappingKind, TextureAsset
from pipeline.external_intake import IntakeError, check_license_fields
from pipeline.texture_index import append_texture, find_texture

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Filename-fragment -> map kind, checked in order (first hit wins).
# Explicit `maps` in source.json overrides all of this.
_KIND_PATTERNS: list[tuple[str, MapKind]] = [
    (r"nor_?gl", "normal_gl"),
    (r"nor_?dx", "normal_dx"),
    (r"normal", "normal_gl"),
    (r"rough", "roughness"),
    (r"(?:^|_)ao(?:_|\.|$)", "ao"),
    (r"(?:^|_)arm(?:_|\.|$)", "arm"),
    (r"disp|height", "displacement"),
    (r"diff|albedo|color|day|surface", "albedo"),
]

EQUIRECT_ASPECT_TOLERANCE = 0.05  # equirect maps must be ~2:1


class TextureIntakeSource(BaseModel):
    """Hand-filled source.json for a texture intake (6c.4)."""

    display_name: str
    license: str
    mapping: MappingKind
    source_site: Optional[str] = None
    source_url: Optional[str] = None
    original_author: Optional[str] = None
    semantic_type: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    tile_size_m: Optional[list[float]] = None  # tileables: real-world meters per tile
    authentic: bool = True  # manual intakes are usually real-world maps
    maps: Optional[dict[MapKind, str]] = None  # explicit filename->kind override
    notes: str = ""


class TextureIntakeResult(BaseModel):
    texture_id: str
    success: bool
    texture: Optional[TextureAsset] = None
    warnings: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None


def detect_map_kind(filename: str) -> MapKind | None:
    name = filename.lower()
    for pattern, kind in _KIND_PATTERNS:
        if re.search(pattern, name):
            return kind
    return None


def _discover_maps(intake_dir: Path, source: TextureIntakeSource) -> dict[MapKind, Path]:
    images = [
        p for p in sorted(intake_dir.iterdir())
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]
    if not images:
        raise IntakeError(
            f"No image file found in {intake_dir} -- expected at least one of: "
            f"{', '.join(sorted(_IMAGE_EXTS))}"
        )
    if source.maps:
        by_name = {p.name: p for p in images}
        missing = [f for f in source.maps.values() if f not in by_name]
        if missing:
            raise IntakeError(f"source.json maps reference missing file(s): {missing}")
        return {kind: by_name[fname] for kind, fname in source.maps.items()}

    if len(images) == 1:
        return {"albedo": images[0]}

    detected: dict[MapKind, Path] = {}
    unlabeled: list[str] = []
    for image in images:
        kind = detect_map_kind(image.name)
        if kind is None:
            unlabeled.append(image.name)
        elif kind in detected:
            raise IntakeError(
                f"Two files both look like the {kind!r} map "
                f"({detected[kind].name}, {image.name}) -- disambiguate with an "
                "explicit `maps` dict in source.json."
            )
        else:
            detected[kind] = image
    if unlabeled:
        raise IntakeError(
            f"Could not classify {unlabeled} by filename -- add an explicit "
            "`maps` dict to source.json (filename -> one of albedo/normal_gl/"
            "normal_dx/roughness/ao/arm/displacement)."
        )
    if "albedo" not in detected:
        raise IntakeError("No albedo map identified -- every texture needs one.")
    return detected


def _ingest_image(src: Path, dest_dir: Path, kind: MapKind, warnings: list[str]) -> tuple[str, list[int]]:
    """Copy (and downscale if oversized) one map into the payload folder.
    Returns (LIBRARY_DIR-relative path of the runtime file, [w, h])."""
    with Image.open(src) as img:
        width, height = img.size
        dest = dest_dir / f"{kind}{src.suffix.lower()}"
        if max(width, height) > config.TEXTURE_MAX_DIM:
            scale = config.TEXTURE_MAX_DIM / max(width, height)
            resized = img.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
            resized.save(dest)
            shutil.copy2(src, dest_dir / f"original_{src.name}")
            warnings.append(
                f"{kind}: {width}x{height} exceeds TEXTURE_MAX_DIM "
                f"({config.TEXTURE_MAX_DIM}); wrote a downscaled runtime copy and "
                "kept the original alongside."
            )
        else:
            shutil.copy2(src, dest)
    # POSIX separators: these paths become browser URLs and JSON references.
    return (Path("textures") / dest_dir.name / dest.name).as_posix(), [width, height]


def intake_texture(texture_id: str) -> TextureIntakeResult:
    intake_dir = config.INTAKE_DIR / texture_id
    source_json = intake_dir / "source.json"
    if not intake_dir.is_dir():
        raise IntakeError(
            f"No intake folder for {texture_id!r}. Create {intake_dir} containing the "
            "image file(s) plus a source.json (see intake/README.md)."
        )
    if not source_json.is_file():
        raise IntakeError(
            f"Missing {source_json}. Every texture intake needs a hand-filled "
            "source.json recording display_name, license, and mapping."
        )
    try:
        source = TextureIntakeSource.model_validate_json(
            source_json.read_text(encoding="utf-8-sig")
        )
    except ValidationError as exc:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise IntakeError(f"source.json for {texture_id!r} is invalid -- {detail}") from exc

    normalized_license = check_license_fields(source.license, source.original_author)
    if find_texture(texture_id) is not None:
        raise IntakeError(
            f"texture_id {texture_id!r} already exists in the texture index; pick a "
            "different id or remove the existing entry explicitly."
        )
    map_files = _discover_maps(intake_dir, source)

    warnings: list[str] = []
    dest_dir = config.LIBRARY_DIR / "textures" / texture_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    maps: dict[MapKind, str] = {}
    albedo_resolution: list[int] | None = None
    for kind, src_path in map_files.items():
        try:
            rel_path, resolution = _ingest_image(src_path, dest_dir, kind, warnings)
        except OSError as exc:
            raise IntakeError(f"{src_path.name} is not a readable image: {exc}") from exc
        maps[kind] = rel_path
        if kind == "albedo":
            albedo_resolution = resolution

    if source.mapping == "equirectangular" and albedo_resolution:
        w, h = albedo_resolution
        if h == 0 or abs(w / h - 2.0) > 2.0 * EQUIRECT_ASPECT_TOLERANCE:
            warnings.append(
                f"claimed equirectangular but albedo is {w}x{h} (not ~2:1) -- "
                "check the mapping."
            )

    license_lines = [
        f"License: {source.license}",
        f"Author: {source.original_author or 'unknown'}",
        f"Source site: {source.source_site or 'unknown'}",
        f"Source URL: {source.source_url or 'unknown'}",
    ]
    if normalized_license.startswith("cc-by"):
        license_lines.append(
            f"Attribution REQUIRED: credit {source.original_author!r} in any distribution."
        )
    if source.notes:
        license_lines.append(f"Notes: {source.notes}")
    (dest_dir / "LICENSE.txt").write_text("\n".join(license_lines) + "\n", encoding="utf-8")

    texture = TextureAsset(
        texture_id=texture_id,
        display_name=source.display_name,
        mapping=source.mapping,
        maps=maps,
        semantic_type=source.semantic_type,
        tags=source.tags,
        resolution=albedo_resolution,
        tile_size_m=source.tile_size_m,
        authentic=source.authentic,
        provenance=ProvenanceInfo(
            source_type="external_approved",
            source_site=source.source_site,
            source_url=source.source_url,
            original_author=source.original_author,
            license=source.license,
            license_status="approved",
            attribution_required=normalized_license.startswith("cc-by"),
            date_imported_or_generated=date.today().isoformat(),
        ),
    )
    append_texture(texture)
    return TextureIntakeResult(
        texture_id=texture_id, success=True, texture=texture, warnings=warnings
    )
