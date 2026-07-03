"""Stage 7: executable validation of the catalog + library, per req. doc
section 12 (12.1 schema, 12.2 completeness, 12.3 scale/geometry,
12.4 licensing/provenance, 12.5 runtime readiness).

Report format mirrors asset_audit.py's established convention: a colorized
screen mode and a machine-readable JSON mode over the same data. Errors
mean the library is not deployable (CLI exits non-zero); warnings are
recorded but allowed, per req. doc section 17's fallback-must-be-recorded
rule.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

import config
from models.catalog_models import AssetCatalogEntry, MaterialDef
from models.spec_models import ResolutionRecord
from pipeline.catalog_writer import load_catalog
from pipeline.composite_builder import CompositeFragment

Category = Literal["schema", "completeness", "geometry", "licensing", "runtime"]
CATEGORY_ORDER: list[Category] = [
    "schema", "completeness", "geometry", "licensing", "runtime",
]

# 12.3 "object scale is within acceptable limits": outside this range is
# almost certainly a unit mistake (mm-as-m or a building-sized prop).
MIN_SANE_DIM_M = 0.001
MAX_SANE_DIM_M = 50.0


class ValidationIssue(BaseModel):
    level: Literal["error", "warning"]
    category: Category
    asset_id: str | None = None
    message: str


class ValidationReport(BaseModel):
    issues: list[ValidationIssue] = Field(default_factory=list)
    checked_assets: int = 0
    checked_materials: int = 0
    checked_composites: int = 0
    checked_records: int = 0

    @property
    def errors(self) -> int:
        return sum(1 for i in self.issues if i.level == "error")

    @property
    def warnings(self) -> int:
        return sum(1 for i in self.issues if i.level == "warning")

    @property
    def passed(self) -> bool:
        return self.errors == 0


def _err(category: Category, message: str, asset_id: str | None = None) -> ValidationIssue:
    return ValidationIssue(level="error", category=category, asset_id=asset_id, message=message)


def _warn(category: Category, message: str, asset_id: str | None = None) -> ValidationIssue:
    return ValidationIssue(level="warning", category=category, asset_id=asset_id, message=message)


def _resolve_address(address: str) -> Path | None:
    """Map a 'library/...' address to a real path; None for non-file addresses
    (Unity built-ins etc.), which have nothing on disk to check."""
    prefix = "library/"
    if address.startswith(prefix):
        return config.LIBRARY_DIR / address[len(prefix):]
    return None


# ── Per-category checks (each independently testable) ────────────────────

def check_catalog_schema() -> tuple[list[AssetCatalogEntry], list[ValidationIssue]]:
    """12.1: the catalog file parses and validates; asset_ids are unique."""
    issues: list[ValidationIssue] = []
    try:
        entries = load_catalog()
    except (json.JSONDecodeError, ValidationError, OSError) as exc:
        return [], [_err("schema", f"catalog.json failed to load/validate: {exc}")]

    seen: set[str] = set()
    for entry in entries:
        if entry.asset_id in seen:
            issues.append(_err("schema", "duplicate asset_id in catalog", entry.asset_id))
        seen.add(entry.asset_id)
    return entries, issues


def check_materials() -> tuple[list[MaterialDef], list[ValidationIssue]]:
    """12.1 + 12.2 for materials: every material JSON validates, its id
    matches its filename, and any texture it references exists."""
    issues: list[ValidationIssue] = []
    materials: list[MaterialDef] = []
    for path in sorted(config.MATERIALS_DIR.glob("*.json")):
        try:
            material = MaterialDef.model_validate_json(path.read_text(encoding="utf-8-sig"))
        except (ValidationError, ValueError) as exc:
            issues.append(_err("schema", f"material {path.name} invalid: {exc}"))
            continue
        materials.append(material)
        if material.material_id != path.stem:
            issues.append(
                _warn(
                    "schema",
                    f"material_id {material.material_id!r} does not match filename {path.name}",
                )
            )
        if material.texture:
            if not (config.MATERIALS_DIR / material.texture).is_file():
                issues.append(
                    _err(
                        "completeness",
                        f"material {material.material_id!r} references missing texture "
                        f"{material.texture!r}",
                    )
                )
    return materials, issues


def check_composites(catalog_ids: set[str]) -> tuple[int, list[ValidationIssue]]:
    """12.1 + 12.2 for composite fragments: they parse, and every resolved
    sub-part points at a real catalog entry."""
    issues: list[ValidationIssue] = []
    count = 0
    for path in sorted((config.LIBRARY_DIR / "composites").glob("*.json")):
        try:
            fragment = CompositeFragment.model_validate_json(
                path.read_text(encoding="utf-8-sig")
            )
        except (ValidationError, ValueError) as exc:
            issues.append(_err("schema", f"composite {path.name} invalid: {exc}"))
            continue
        count += 1
        for part in fragment.parts:
            if part.resolved_asset_id and part.resolved_asset_id not in catalog_ids:
                issues.append(
                    _err(
                        "completeness",
                        f"composite part {part.part_id!r} references unknown asset_id "
                        f"{part.resolved_asset_id!r}",
                        fragment.composite_id,
                    )
                )
            elif not part.resolved_asset_id:
                issues.append(
                    _warn(
                        "runtime",
                        f"composite part {part.part_id!r} is not resolved yet",
                        fragment.composite_id,
                    )
                )
    return count, issues


def check_entry_files(entry: AssetCatalogEntry) -> list[ValidationIssue]:
    """12.2: the files a catalog entry promises actually exist."""
    issues: list[ValidationIssue] = []
    address_path = _resolve_address(entry.address)
    if address_path is not None:
        if not address_path.is_file():
            issues.append(
                _err("completeness", f"address {entry.address!r} does not exist on disk",
                     entry.asset_id)
            )
        elif address_path.stat().st_size == 0:
            issues.append(
                _err("geometry", f"address {entry.address!r} is a zero-byte file (mesh "
                     "import/export failed?)", entry.asset_id)
            )
        elif not (address_path.parent / "meta.json").is_file():
            issues.append(
                _warn("completeness", "no meta.json next to the model (normalization "
                      "record missing)", entry.asset_id)
            )
    return issues


def check_entry_geometry(entry: AssetCatalogEntry) -> list[ValidationIssue]:
    """12.3: sane canonical bounds and triangle count under threshold."""
    issues: list[ValidationIssue] = []
    largest = max(entry.canonical_bounds_m, default=0)
    if any(v < 0 for v in entry.canonical_bounds_m) or largest <= 0:
        issues.append(
            _err("geometry", f"canonical_bounds_m {entry.canonical_bounds_m} has a "
                 "negative dimension or no positive extent", entry.asset_id)
        )
    else:
        if 0 in entry.canonical_bounds_m and entry.asset_class != "primitive":
            # Zero thickness is by design for flat built-ins (plane/quad);
            # on a mesh-backed asset it usually means a failed import.
            issues.append(
                _warn("geometry", f"canonical_bounds_m {entry.canonical_bounds_m} has a "
                      "zero dimension on a mesh-backed asset", entry.asset_id)
            )
        if largest > MAX_SANE_DIM_M or largest < MIN_SANE_DIM_M:
            issues.append(
                _warn("geometry", f"largest dimension {largest:g} m is outside the sane "
                      f"range [{MIN_SANE_DIM_M}, {MAX_SANE_DIM_M}] -- unit mistake?",
                      entry.asset_id)
            )

    address_path = _resolve_address(entry.address)
    if address_path is not None:
        meta_path = address_path.parent / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, OSError) as exc:
                issues.append(_err("schema", f"meta.json unreadable: {exc}", entry.asset_id))
                return issues
            tri = meta.get("triangle_count_after")
            if isinstance(tri, int) and tri > config.MAX_TRIANGLE_COUNT:
                issues.append(
                    _err("geometry", f"triangle count {tri} exceeds MAX_TRIANGLE_COUNT "
                         f"({config.MAX_TRIANGLE_COUNT})", entry.asset_id)
                )
    return issues


def check_entry_licensing(entry: AssetCatalogEntry) -> list[ValidationIssue]:
    """12.4: imported assets are approved, attributed, and licensed."""
    issues: list[ValidationIssue] = []
    prov = entry.provenance
    is_external = entry.asset_class == "imported" or prov.source_type == "external_approved"
    if is_external:
        if prov.license_status != "approved":
            issues.append(
                _err("licensing", f"imported asset has license_status "
                     f"{prov.license_status!r} (must be 'approved')", entry.asset_id)
            )
        if not (prov.license or "").strip():
            issues.append(
                _err("licensing", "imported asset has no license recorded", entry.asset_id)
            )
        if prov.attribution_required and not (prov.original_author or "").strip():
            issues.append(
                _err("licensing", "attribution required but no original_author recorded",
                     entry.asset_id)
            )
        address_path = _resolve_address(entry.address)
        if address_path is not None and not (address_path.parent / "LICENSE.txt").is_file():
            issues.append(
                _warn("licensing", "no LICENSE.txt next to the imported model",
                      entry.asset_id)
            )
    elif prov.license_status == "unknown" and prov.source_type not in ("core", "generated", "composite"):
        issues.append(
            _warn("licensing", f"asset from source_type {prov.source_type!r} has unknown "
                  "license status", entry.asset_id)
        )
    return issues


def check_resolution_records(records_path: Path) -> tuple[int, list[ValidationIssue]]:
    """12.5: no unresolved AssetSpecs remain in a scene being finalized.
    `records_path` holds one ResolutionRecord or a list of them."""
    issues: list[ValidationIssue] = []
    try:
        raw = json.loads(records_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError) as exc:
        return 0, [_err("runtime", f"resolution records file unreadable: {exc}")]
    raw_list = raw if isinstance(raw, list) else [raw]

    count = 0
    for item in raw_list:
        try:
            record = ResolutionRecord.model_validate(item)
        except ValidationError as exc:
            issues.append(_err("schema", f"resolution record invalid: {exc}"))
            continue
        count += 1
        if record.requires_author_review:
            reason = record.review_reason or "flagged for author review"
            issues.append(_err("runtime", f"unresolved: {reason}", record.object_id))
    return count, issues


# ── Orchestrator ──────────────────────────────────────────────────────────

def validate_library(records_path: Path | None = None) -> ValidationReport:
    report = ValidationReport()

    entries, issues = check_catalog_schema()
    report.issues.extend(issues)
    report.checked_assets = len(entries)
    catalog_ids = {e.asset_id for e in entries}

    materials, issues = check_materials()
    report.issues.extend(issues)
    report.checked_materials = len(materials)

    count, issues = check_composites(catalog_ids)
    report.issues.extend(issues)
    report.checked_composites = count

    for entry in entries:
        report.issues.extend(check_entry_files(entry))
        report.issues.extend(check_entry_geometry(entry))
        report.issues.extend(check_entry_licensing(entry))

    if records_path is not None:
        count, issues = check_resolution_records(records_path)
        report.issues.extend(issues)
        report.checked_records = count

    return report


# ── Output formatting (mirrors asset_audit.py's screen/json split) ────────

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _colorize(text: str, color: str, use_color: bool) -> str:
    return f"{color}{text}{_RESET}" if use_color else text


def format_screen_text(report: ValidationReport, use_color: bool = True) -> str:
    lines = [_colorize("=== Asset Pipeline Validation ===", _BOLD, use_color)]
    lines.append(
        f"Assets: {report.checked_assets}  |  Materials: {report.checked_materials}  |  "
        f"Composites: {report.checked_composites}"
        + (f"  |  Records: {report.checked_records}" if report.checked_records else "")
        + f"  |  Errors: {report.errors}  |  Warnings: {report.warnings}"
    )
    lines.append("")

    for category in CATEGORY_ORDER:
        group = [i for i in report.issues if i.category == category]
        label = _colorize(category.upper(), _BOLD, use_color)
        if not group:
            lines.append(f"{label}  {_colorize('[OK]', _GREEN, use_color)}")
            continue
        lines.append(label)
        for issue in group:
            tag = (
                _colorize("[ERROR]", _RED, use_color)
                if issue.level == "error"
                else _colorize("[WARN] ", _YELLOW, use_color)
            )
            asset = f"{issue.asset_id}: " if issue.asset_id else ""
            lines.append(f"  {tag} {asset}{issue.message}")
    lines.append("")
    verdict = (
        _colorize("PASSED", _GREEN, use_color)
        if report.passed
        else _colorize("FAILED", _RED, use_color)
    )
    lines.append(f"Result: {verdict} ({report.errors} error(s), {report.warnings} warning(s))")
    return "\n".join(lines)


def format_json_output(report: ValidationReport) -> str:
    return json.dumps(
        {
            "summary": {
                "checked_assets": report.checked_assets,
                "checked_materials": report.checked_materials,
                "checked_composites": report.checked_composites,
                "checked_records": report.checked_records,
                "errors": report.errors,
                "warnings": report.warnings,
                "passed": report.passed,
            },
            "issues": [i.model_dump(mode="json") for i in report.issues],
        },
        indent=2,
    )
