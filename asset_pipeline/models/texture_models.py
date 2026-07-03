"""Stage 6c texture-library models (see implementation plan section 6c.1).

A TextureAsset is a first-class, reusable surface: once a texture enters
library/textures/, later requests that match it semantically AND
geometrically (mapping compatibility) reuse it -- never re-download,
never re-generate.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from models.catalog_models import ProvenanceInfo, UVInfo

# The reuse contract: what UV layout does this texture assume?
#   equirectangular -- planet maps / ball skins; fits any standard UV sphere.
#   tileable        -- wood/rust/asphalt; fits anything with UVs, scaled by
#                      uv_tiling computed from tile_size_m.
#   atlas           -- painted against ONE mesh's UV islands; fits only the
#                      asset whose uv_hash it records.
MappingKind = Literal["equirectangular", "tileable", "atlas"]

# v1 consumes 'albedo' only, but stores whatever the source provides --
# keeping a normal map costs disk; re-downloading costs a human.
MapKind = Literal[
    "albedo", "normal_gl", "normal_dx", "roughness", "ao", "arm", "displacement"
]


class TextureAsset(BaseModel):
    texture_id: str
    display_name: str
    mapping: MappingKind
    maps: dict[MapKind, str]  # map kind -> path relative to LIBRARY_DIR
    semantic_type: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    resolution: Optional[list[int]] = None  # [w, h] of the albedo map
    tile_size_m: Optional[list[float]] = None  # [u, v] real-world meters per tile
    authentic: bool = False  # photographic/measured vs synthesized
    bound_asset_id: Optional[str] = None  # atlas only
    uv_hash: Optional[str] = None  # atlas only
    provenance: ProvenanceInfo


def mapping_fits(texture_mapping: MappingKind, uv: UVInfo | None) -> bool:
    """Can a texture with this mapping be bound to an object with this UV info?

    Unknown UV info (None) is treated permissively for tileables (most
    imported meshes have usable UVs) and strictly for equirect/atlas
    (those genuinely require a known layout).
    """
    if texture_mapping == "tileable":
        return uv is None or uv.status != "none"
    if uv is None:
        return False
    if texture_mapping == "equirectangular":
        return uv.convention == "equirect"
    # atlas: needs a hash match, checked separately by the caller/validator --
    # here we only require that a specific layout exists.
    return uv.convention == "atlas" or uv.uv_hash is not None
