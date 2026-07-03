"""Stage 6c.2: UV inspection tools -- read a GLB's UV layer with pygltflib,
hash it, and rasterize the island wireframe to a PNG with Pillow.

No Blender involved: the GLB Stage 4 already exported contains the UV
coordinates, so a second Blender run would be pure overhead. The layout
image is the human/LLM-facing template for authoring atlas textures; the
hash is what lets an atlas texture assert which layout it was painted for
(validator checks it after regenerations).
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pygltflib
from PIL import Image, ImageDraw

_COMPONENT_STRUCT = {
    5121: ("B", 1),  # unsigned byte
    5123: ("H", 2),  # unsigned short
    5125: ("I", 4),  # unsigned int
    5126: ("f", 4),  # float
}


def _read_accessor(gltf: pygltflib.GLTF2, accessor_index: int) -> np.ndarray:
    accessor = gltf.accessors[accessor_index]
    view = gltf.bufferViews[accessor.bufferView]
    blob = gltf.binary_blob()
    fmt, size = _COMPONENT_STRUCT[accessor.componentType]
    components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor.type]
    start = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    count = accessor.count * components
    data = struct.unpack_from(f"<{count}{fmt}", blob, start)
    return np.array(data, dtype=np.float64).reshape(accessor.count, components)


def read_uvs(glb_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(uv_coords[N,2], triangle_indices[M,3]) from the first primitive with
    a TEXCOORD_0 attribute, or None if the mesh has no UV layer."""
    gltf = pygltflib.GLTF2().load(str(glb_path))
    for mesh in gltf.meshes:
        for primitive in mesh.primitives:
            texcoord = getattr(primitive.attributes, "TEXCOORD_0", None)
            if texcoord is None:
                continue
            uvs = _read_accessor(gltf, texcoord)[:, :2]
            if primitive.indices is not None:
                indices = _read_accessor(gltf, primitive.indices).astype(np.int64)
                triangles = indices.reshape(-1, 3)
            else:
                triangles = np.arange(len(uvs), dtype=np.int64).reshape(-1, 3)
            return uvs, triangles
    return None


def uv_hash(glb_path: Path) -> str | None:
    data = read_uvs(glb_path)
    if data is None:
        return None
    uvs, _ = data
    return hashlib.md5(np.ascontiguousarray(uvs.astype(np.float32)).tobytes()).hexdigest()[:16]


def export_uv_layout(glb_path: Path, out_path: Path, size: int = 512) -> Path | None:
    """Rasterize the UV island wireframe (white bg, dark edges). Returns None
    if the mesh has no UVs."""
    data = read_uvs(glb_path)
    if data is None:
        return None
    uvs, triangles = data

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    def to_px(uv):
        # UV origin is bottom-left; image origin is top-left.
        return (uv[0] * (size - 1), (1.0 - uv[1]) * (size - 1))

    for tri in triangles:
        pts = [to_px(uvs[i]) for i in tri]
        draw.line([pts[0], pts[1], pts[2], pts[0]], fill=(60, 70, 90), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
