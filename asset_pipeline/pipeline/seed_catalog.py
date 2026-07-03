"""Seed library/catalog.json with core primitive assets.

Bounds/pivot/axis conventions match the Unity built-in primitive meshes
already procedurally generated in
viewer/arlem_preview_toolkit/arlem_preview_toolkit/prefab_to_glb.py
(_sphere_mesh, _box_mesh, _quad_mesh, _cylinder_mesh, _capsule_mesh all use
radius=0.5 / unit dimensions, center pivot), so scale stays consistent with
what the existing viewer already renders.
"""
from __future__ import annotations

from datetime import datetime, timezone

import config
from models.catalog_models import AssetCatalogEntry, ProvenanceInfo, UVInfo
from pipeline.catalog_writer import save_catalog

_NOW = datetime.now(timezone.utc).isoformat()


def _core_provenance() -> ProvenanceInfo:
    return ProvenanceInfo(
        source_type="core",
        license="N/A (built-in Unity primitive)",
        license_status="approved",
        date_imported_or_generated=_NOW,
    )


def build_seed_entries() -> list[AssetCatalogEntry]:
    entries = [
        AssetCatalogEntry(
            asset_id="sphere_basic",
            display_name="Basic Sphere",
            asset_class="primitive",
            address="Primitives/Sphere",
            material_slots=["surface"],
            canonical_bounds_m=[1.0, 1.0, 1.0],
            pivot="center",
            tags=["sphere", "primitive", "planet", "atom", "marker", "moon"],
            provenance=_core_provenance(),
            review_level=0,
        ),
        AssetCatalogEntry(
            asset_id="cube_basic",
            display_name="Basic Cube",
            asset_class="primitive",
            address="Primitives/Cube",
            material_slots=["surface"],
            canonical_bounds_m=[1.0, 1.0, 1.0],
            pivot="center",
            tags=["cube", "box", "primitive", "block"],
            provenance=_core_provenance(),
            review_level=0,
        ),
        AssetCatalogEntry(
            asset_id="cylinder_basic",
            display_name="Basic Cylinder",
            asset_class="primitive",
            address="Primitives/Cylinder",
            material_slots=["surface"],
            canonical_bounds_m=[1.0, 1.0, 1.0],
            pivot="center",
            tags=["cylinder", "primitive", "rod", "post"],
            provenance=_core_provenance(),
            review_level=0,
        ),
        AssetCatalogEntry(
            asset_id="cone_basic",
            display_name="Basic Cone",
            asset_class="primitive",
            address="Primitives/Cone",
            material_slots=["surface"],
            canonical_bounds_m=[1.0, 1.0, 1.0],
            pivot="base_center",
            tags=["cone", "primitive", "pointer", "arrow_head"],
            provenance=_core_provenance(),
            review_level=0,
        ),
        AssetCatalogEntry(
            asset_id="torus_basic",
            display_name="Basic Torus",
            asset_class="primitive",
            address="Primitives/Torus",
            material_slots=["surface"],
            canonical_bounds_m=[1.0, 0.3, 1.0],
            pivot="center",
            tags=["torus", "ring", "primitive", "orbit"],
            provenance=_core_provenance(),
            review_level=0,
        ),
        AssetCatalogEntry(
            asset_id="plane_basic",
            display_name="Basic Plane",
            asset_class="primitive",
            address="Primitives/Plane",
            material_slots=["surface"],
            canonical_bounds_m=[10.0, 0.0, 10.0],
            pivot="center",
            default_up_axis="+Y",
            tags=["plane", "ground", "primitive", "floor"],
            provenance=_core_provenance(),
            review_level=0,
        ),
        AssetCatalogEntry(
            asset_id="quad_basic",
            display_name="Basic Quad",
            asset_class="primitive",
            address="Primitives/Quad",
            material_slots=["surface"],
            canonical_bounds_m=[1.0, 1.0, 0.0],
            pivot="center",
            default_forward_axis="+Z",
            tags=["quad", "flat", "primitive", "billboard", "label_backing"],
            provenance=_core_provenance(),
            review_level=0,
        ),
        AssetCatalogEntry(
            asset_id="capsule_basic",
            display_name="Basic Capsule",
            asset_class="primitive",
            address="Primitives/Capsule",
            material_slots=["surface"],
            canonical_bounds_m=[1.0, 2.0, 1.0],
            pivot="center",
            tags=["capsule", "primitive", "pill", "person_marker"],
            provenance=_core_provenance(),
            review_level=0,
        ),
        AssetCatalogEntry(
            asset_id="arrow_basic",
            display_name="Basic Arrow",
            asset_class="composite",
            address="Composites/Arrow",
            material_slots=["surface"],
            canonical_bounds_m=[0.2, 0.2, 1.0],
            pivot="base_center",
            default_forward_axis="+Z",
            tags=["arrow", "primitive", "direction", "vector"],
            capabilities={"scalable": True, "supports_material_override": True, "has_collider": False},
            provenance=_core_provenance(),
            review_level=0,
        ),
        AssetCatalogEntry(
            asset_id="label_basic",
            display_name="Basic Text Label",
            asset_class="primitive",
            address="Primitives/Label",
            material_slots=["text", "backing"],
            canonical_bounds_m=[0.5, 0.15, 0.01],
            pivot="center",
            tags=["label", "text", "textmeshpro", "annotation"],
            capabilities={"scalable": True, "supports_material_override": True, "has_collider": False},
            provenance=_core_provenance(),
            review_level=0,
        ),
    ]
    # Stage 6c: built-in primitives ship with known UV conventions. The
    # sphere family is equirectangular -- which is exactly why a planet map
    # binds to sphere_basic with zero mesh work. Everything else gets
    # "generic" (usable UVs exist; tileables fit, equirect maps do not).
    for entry in entries:
        entry.uv = UVInfo(
            status="builtin",
            convention="equirect" if entry.asset_id == "sphere_basic" else "generic",
        )
    return entries


def main() -> None:
    entries = build_seed_entries()
    save_catalog(entries)
    print(f"Seeded catalog with {len(entries)} core primitives -> {config.CATALOG_PATH}")


if __name__ == "__main__":
    main()
