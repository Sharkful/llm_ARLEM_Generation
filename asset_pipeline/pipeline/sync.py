"""Stage 9: copy a resolved subset of library/ into a consumer assets folder.

Copy-only, one direction: library/ stays the source of truth; the consumer
(the three.js viewer today, a Unity Assets/ folder later) never becomes a
build dependency. Target layout matches what asset_audit.py already scans:

    <target>/models/<asset_id>.glb
    <target>/materials/<material_id>.json
    <target>/textures/<file>

Default target is the viewer's asset folder
(viewer/arlem_preview_toolkit/arlem_preview_toolkit/assets).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel, Field

import config
from models.catalog_models import MaterialDef
from pipeline.catalog_writer import load_catalog

DEFAULT_SYNC_TARGET = (
    config.PACKAGE_DIR.parent
    / "viewer" / "arlem_preview_toolkit" / "arlem_preview_toolkit" / "assets"
)


class SyncResult(BaseModel):
    target: str
    models_copied: list[str] = Field(default_factory=list)
    materials_copied: list[str] = Field(default_factory=list)
    textures_copied: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)  # non-file addresses (Unity built-ins)


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def sync_assets(
    asset_ids: list[str],
    target: Path | None = None,
    material_ids: list[str] | None = None,
) -> SyncResult:
    """Copy the given assets' GLBs plus materials and any texture files
    those materials reference.

    material_ids: explicit material names to sync in addition to the
    per-asset `<asset_id>_mat` convention. Scene objects name their material
    after the *object* (`moon_mat`), not the catalog asset they resolved to
    (`sphere_basic`), so the scene resolver passes those here.
    """
    target = Path(target) if target else DEFAULT_SYNC_TARGET
    result = SyncResult(target=str(target))
    catalog = {e.asset_id: e for e in load_catalog()}

    for asset_id in dict.fromkeys(asset_ids):  # de-dupe, keep order
        entry = catalog.get(asset_id)
        if entry is None:
            result.skipped.append(f"{asset_id} (not in catalog)")
            continue
        if not entry.address.startswith("library/"):
            result.skipped.append(f"{asset_id} (built-in: {entry.address})")
        else:
            src = config.LIBRARY_DIR / entry.address[len("library/"):]
            if src.is_file():
                _copy(src, target / "models" / f"{asset_id}.glb")
                result.models_copied.append(asset_id)
            else:
                result.skipped.append(f"{asset_id} (missing file: {entry.address})")

    wanted_materials = dict.fromkeys(
        [f"{asset_id}_mat" for asset_id in asset_ids] + list(material_ids or [])
    )
    for material_id in wanted_materials:
        material_path = config.MATERIALS_DIR / f"{material_id}.json"
        if material_path.is_file():
            material = MaterialDef.model_validate_json(
                material_path.read_text(encoding="utf-8-sig")
            )
            texture_dest_name: str | None = None
            if material.texture:
                texture_src = config.LIBRARY_DIR / material.texture
                if texture_src.is_file():
                    # Flatten into textures/ with a unique name: registry
                    # payloads are all called albedo.<ext>, so use the
                    # texture folder's name (the texture_id) instead.
                    texture_dest_name = (
                        texture_src.name
                        if texture_src.parent == config.TEXTURES_DIR
                        else f"{texture_src.parent.name}{texture_src.suffix}"
                    )
                    _copy(texture_src, target / "textures" / texture_dest_name)
                    result.textures_copied.append(texture_dest_name)
            # The copied material references the flattened texture path;
            # the library copy keeps its LIBRARY_DIR-relative path.
            synced = material.model_copy(
                update={"texture": f"textures/{texture_dest_name}" if texture_dest_name else material.texture}
            )
            dest = target / "materials" / material_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                synced.model_dump_json(indent=2), encoding="utf-8"
            )
            result.materials_copied.append(material.material_id)

    return result
