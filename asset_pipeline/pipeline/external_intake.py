"""Stage 6a: normalize a human-provided model file into the library.

Manual-first by design (implementation plan Stage 6 / req. doc section 3.2):
a human downloads a file from an approved source, drops it in
intake/<asset_id>/ next to a hand-filled source.json (see IntakeSource and
intake/README.md), and this module does the rest -- license gate, Stage 4
mesh normalization, provenance record, catalog entry. Nothing here
downloads anything; the semi-automated search assist (Stage 6b, Poly Haven
allowlist) comes later and still funnels through this same intake path.

Every rejection carries the specific missing-field/violation reason --
never silently skip provenance (req. doc section 11).
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ValidationError

import config
from models.catalog_models import AssetCatalogEntry, IntakeSource, ProvenanceInfo
from pipeline.catalog_writer import append_entry, find_by_id
from pipeline.mesh_processor import (
    MeshNormalizationResult,
    MeshProcessingError,
    SourceFormat,
    normalize_mesh,
)

# Licenses that may enter the library (req. doc section 11). CC-BY variants
# additionally require original_author so attribution is actually capturable.
# Add new licenses only after a human verifies the terms -- this list is the
# guardrail, not a suggestion.
_LICENSE_ALLOWLIST = {
    "cc0",
    "cc0-1.0",
    "cc-by",
    "cc-by-3.0",
    "cc-by-4.0",
    "internal",
    "proprietary-cleared",
}

_MESH_FORMATS: dict[str, SourceFormat] = {
    ".stl": "stl",
    ".obj": "obj",
    ".fbx": "fbx",
    ".glb": "glb",
    ".gltf": "glb",  # Blender's gltf importer handles both containers
}


class IntakeError(RuntimeError):
    """Raised for any intake rejection; message states the specific reason."""


class IntakeResult(BaseModel):
    asset_id: str
    success: bool
    glb_path: str | None = None
    original_path: str | None = None
    meta_path: str | None = None
    license_path: str | None = None
    catalog_entry: AssetCatalogEntry | None = None
    normalization: MeshNormalizationResult | None = None
    error_message: str | None = None


def _normalize_license(license_str: str) -> str:
    return license_str.strip().lower().replace(" ", "-")


def load_intake_source(asset_id: str) -> tuple[IntakeSource, Path]:
    intake_dir = config.INTAKE_DIR / asset_id
    source_json = intake_dir / "source.json"
    if not intake_dir.is_dir():
        raise IntakeError(
            f"No intake folder for {asset_id!r}. Create {intake_dir} containing the "
            "downloaded model file plus a source.json (see intake/README.md)."
        )
    if not source_json.is_file():
        raise IntakeError(
            f"Missing {source_json}. Every intake needs a hand-filled source.json "
            "recording display_name, license, source URL/author, and target_size_m."
        )
    try:
        # utf-8-sig: tolerate the BOM that Notepad/PowerShell prepend, since
        # source.json is authored by hand on whatever editor the human has.
        source = IntakeSource.model_validate_json(source_json.read_text(encoding="utf-8-sig"))
    except ValidationError as exc:
        missing = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise IntakeError(f"source.json for {asset_id!r} is invalid -- {missing}") from exc
    return source, intake_dir


def check_license(source: IntakeSource) -> str:
    normalized = _normalize_license(source.license)
    if normalized not in _LICENSE_ALLOWLIST:
        raise IntakeError(
            f"License {source.license!r} is not in the allowlist "
            f"({', '.join(sorted(_LICENSE_ALLOWLIST))}). Verify the terms and either "
            "correct source.json or do not intake this asset."
        )
    if normalized.startswith("cc-by") and not (source.original_author or "").strip():
        raise IntakeError(
            "License is CC-BY but source.json has no original_author -- attribution "
            "must be captured before a CC-BY asset can enter the library."
        )
    return normalized


def find_mesh_file(intake_dir: Path) -> tuple[Path, SourceFormat]:
    candidates = [
        p for p in sorted(intake_dir.iterdir())
        if p.is_file() and p.suffix.lower() in _MESH_FORMATS
    ]
    if not candidates:
        raise IntakeError(
            f"No model file found in {intake_dir} -- expected exactly one of: "
            f"{', '.join(sorted(_MESH_FORMATS))}"
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise IntakeError(
            f"Multiple model files in {intake_dir} ({names}) -- keep exactly one per "
            "asset_id so there is no ambiguity about what was approved."
        )
    mesh = candidates[0]
    return mesh, _MESH_FORMATS[mesh.suffix.lower()]


def _write_license_file(out_dir: Path, source: IntakeSource, normalized_license: str) -> Path:
    lines = [
        f"License: {source.license}",
        f"Author: {source.original_author or 'unknown'}",
        f"Source site: {source.source_site or 'unknown'}",
        f"Source URL: {source.source_url or 'unknown'}",
    ]
    if normalized_license.startswith("cc-by"):
        lines.append(
            f"Attribution REQUIRED: credit {source.original_author!r} in any distribution."
        )
    if source.notes:
        lines.append(f"Notes: {source.notes}")
    license_path = out_dir / "LICENSE.txt"
    license_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return license_path


def intake_asset(asset_id: str) -> IntakeResult:
    """Run the full 6a flow: gate, normalize, record provenance, catalog.

    License and duplicate checks run before Blender is invoked, so a
    rejected intake never leaves half-processed files in library/imported/.
    """
    source, intake_dir = load_intake_source(asset_id)
    normalized_license = check_license(source)
    if find_by_id(asset_id) is not None:
        raise IntakeError(
            f"asset_id {asset_id!r} already exists in the catalog; pick a different id "
            "or remove the existing entry explicitly."
        )
    mesh_path, mesh_format = find_mesh_file(intake_dir)

    out_dir = config.LIBRARY_DIR / "imported" / asset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / "model.glb"

    try:
        normalization = normalize_mesh(
            asset_id=asset_id,
            source_path=mesh_path,
            source_format=mesh_format,
            glb_path=glb_path,
            target_size_m=source.target_size_m,
            pivot=source.pivot,
        )
    except MeshProcessingError as exc:
        raise IntakeError(f"Mesh normalization could not run: {exc}") from exc
    if not normalization.success:
        return IntakeResult(
            asset_id=asset_id,
            success=False,
            normalization=normalization,
            error_message=f"Mesh normalization failed: {normalization.error_message}",
        )

    original_path = out_dir / f"original{mesh_path.suffix.lower()}"
    shutil.copy2(mesh_path, original_path)
    license_path = _write_license_file(out_dir, source, normalized_license)
    meta_path = out_dir / "meta.json"
    meta_path.write_text(
        json.dumps(normalization.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    provenance = ProvenanceInfo(
        source_type="external_approved",
        source_site=source.source_site,
        source_url=source.source_url,
        original_author=source.original_author,
        license=source.license,
        license_status="approved",
        attribution_required=normalized_license.startswith("cc-by"),
        modified=True,
        modifications=["normalized scale/pivot (Stage 4 mesh processor)"],
        date_imported_or_generated=date.today().isoformat(),
    )
    entry = AssetCatalogEntry(
        asset_id=asset_id,
        display_name=source.display_name,
        asset_class="imported",
        address=f"library/imported/{asset_id}/model.glb",
        canonical_bounds_m=normalization.final_bounds_m,
        pivot=source.pivot,
        tags=source.tags,
        provenance=provenance,
        # Implementation plan section 4: imported assets default to review
        # level 3, or 4 when tagged pedagogically critical.
        review_level=4 if source.pedagogically_critical else 3,
    )
    append_entry(entry)

    return IntakeResult(
        asset_id=asset_id,
        success=True,
        glb_path=str(glb_path),
        original_path=str(original_path),
        meta_path=str(meta_path),
        license_path=str(license_path),
        catalog_entry=entry,
        normalization=normalization,
    )
