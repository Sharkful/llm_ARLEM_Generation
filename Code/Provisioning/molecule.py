"""
The ``molecule`` builder (#111; spec: PROVISIONING.md, "Builders"): any small molecule,
given its name or formula, as a ball-and-stick model from PubChem data.

1. **Look up** the compound on PubChem (PUG REST, no key): its name, else its formula.
2. **Coordinates**, best first:
   - PubChem's 3D conformer (``pubchem_3d``);
   - an **idealized VSEPR geometry** for a molecule with one central atom and every
     other atom bonded to it (CH4, NH3, H2O, CO2, BF3, PF5, SF6, XeF4...), from
     PubChem's 2D record: electron domains = bonded atoms + lone pairs, placed at the
     ideal positions (lone pairs take the equatorial positions of a trigonal
     bipyramid first, the axial ones of an octahedron first), bond lengths from
     covalent radii (``idealized_vsepr``). PubChem has no 3D record for hypervalent
     molecules such as XeF4, PF5 or SF6, and these are exactly what VSEPR labs show;
   - RDKit, if installed, embedding PubChem's 2D connectivity (``rdkit``);
   - a single atom (argon) is one ball (``single_atom``);
   - otherwise "can't build", and the asset falls through to search.
   PubChem bond types other than single, double and triple (quadruple, dative,
   complex, unknown) are drawn as one stick and reported in the details; their atoms get
   no lone pairs, and an idealized geometry refuses them. Ionic bonds are not drawn.
3. **Model**: CPK (Jmol) colors, balls at a fifth of each atom's van der Waals
   radius, bonds split in two halves colored by their atoms, double and triple bonds
   as two or three sticks, one material per element. With the ``lone_pairs`` option,
   each central atom's lone pairs (valence electrons minus bond orders minus charge,
   halved) are translucent lobes in the free electron-domain directions.
4. **Model convention** (PROVISIONING.md): GLB, +Y up, centered on its bounding box,
   scaled so the largest side is 1.0, within ``library.MAX_MODEL_TRIANGLES``: a model
   over it is rebuilt with coarser balls and sticks, and refused if still over. A
   thumbnail is drawn with matplotlib, to check the result by eye.

PubChem replies are cached for the process, so building one compound for two assets
asks PubChem once.

Deterministic: the same PubChem data gives byte-identical models, and thumbnails from
the same view on every machine (equal spreads take the world axes). PubChem's data is in
the public domain (NCBI); the CID, source and geometry method go into the provenance.

    python "Code/Provisioning/molecule.py" "xenon tetrafluoride" --lone-pairs --out <folder>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import library  # noqa: E402
from builders import BuilderInfo, BuildRequest, BuildResult  # noqa: E402

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"   # tests point this at a mock server
PUBCHEM_PAGE = "https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
PUBCHEM_LICENSE_URL = "https://www.ncbi.nlm.nih.gov/home/about/policies/"
USE_RDKIT = True        # tests turn the optional fallback off

# Look and size, in ångström before the model is scaled to fit a unit cube.
BALL_FACTOR = 0.2       # ball radius = BALL_FACTOR * van der Waals radius
BOND_RADIUS = 0.08
# Sticks of a double or triple bond: thinner, and spaced (center to center) wider than
# a stick, so they read as two or three, not one thick stick.
MULTI_BOND_RADIUS = {2: 0.05, 3: 0.04}
MULTI_BOND_GAP = {2: 0.16, 3: 0.12}
BOND_SHORTENING = {1: 1.0, 2: 0.87, 3: 0.78}   # idealized bond length factor by order
LONE_PAIR_COLOR = (180, 200, 255, 90)
LONE_PAIR_SIZE = (0.3, 0.55)   # lobe radius across and along its direction
# (ball subdivisions, stick sections): standard, then coarser when over the budget
DETAIL = {"standard": (2, 16), "reduced": (1, 8)}   # 320 / 80 triangles per ball
VERSION = 2   # BuilderInfo.version: bump when the output changes
# PubChem bond types (PC_BondType) that aren't single, double or triple
UNUSUAL_BONDS = {4: "quadruple", 5: "dative", 6: "complex", 7: "ionic", 255: "unknown"}

# Z: (symbol, covalent radius (Cordero 2008), van der Waals radius (Bondi/Alvarez),
#     Jmol CPK color, valence electrons for lone pairs or None for transition metals)
ELEMENTS: dict[int, tuple[str, float, float, tuple[int, int, int], Optional[int]]] = {
    1: ("H", 0.31, 1.20, (255, 255, 255), 1),
    2: ("He", 0.28, 1.40, (217, 255, 255), 2),
    3: ("Li", 1.28, 1.82, (204, 128, 255), 1),
    4: ("Be", 0.96, 1.53, (194, 255, 0), 2),
    5: ("B", 0.84, 1.92, (255, 181, 181), 3),
    6: ("C", 0.76, 1.70, (144, 144, 144), 4),
    7: ("N", 0.71, 1.55, (48, 80, 248), 5),
    8: ("O", 0.66, 1.52, (255, 13, 13), 6),
    9: ("F", 0.57, 1.47, (144, 224, 80), 7),
    10: ("Ne", 0.58, 1.54, (179, 227, 245), 8),
    11: ("Na", 1.66, 2.27, (171, 92, 242), 1),
    12: ("Mg", 1.41, 1.73, (138, 255, 0), 2),
    13: ("Al", 1.21, 1.84, (191, 166, 166), 3),
    14: ("Si", 1.11, 2.10, (240, 200, 160), 4),
    15: ("P", 1.07, 1.80, (255, 128, 0), 5),
    16: ("S", 1.05, 1.80, (255, 255, 48), 6),
    17: ("Cl", 1.02, 1.75, (31, 240, 31), 7),
    18: ("Ar", 1.06, 1.88, (128, 209, 227), 8),
    19: ("K", 2.03, 2.75, (143, 64, 212), 1),
    20: ("Ca", 1.76, 2.31, (61, 255, 0), 2),
    22: ("Ti", 1.60, 2.15, (191, 194, 199), None),
    25: ("Mn", 1.39, 2.05, (156, 122, 199), None),
    26: ("Fe", 1.32, 2.04, (224, 102, 51), None),
    27: ("Co", 1.26, 2.00, (240, 144, 160), None),
    28: ("Ni", 1.24, 1.97, (80, 208, 80), None),
    29: ("Cu", 1.32, 1.96, (200, 128, 51), None),
    30: ("Zn", 1.22, 2.01, (125, 128, 176), None),
    32: ("Ge", 1.20, 2.11, (102, 143, 143), 4),
    33: ("As", 1.19, 1.85, (189, 128, 227), 5),
    34: ("Se", 1.20, 1.90, (255, 161, 0), 6),
    35: ("Br", 1.20, 1.85, (166, 41, 41), 7),
    36: ("Kr", 1.16, 2.02, (92, 184, 209), 8),
    47: ("Ag", 1.45, 2.03, (192, 192, 192), None),
    50: ("Sn", 1.39, 2.17, (102, 128, 128), 4),
    53: ("I", 1.39, 1.98, (148, 0, 148), 7),
    54: ("Xe", 1.40, 2.16, (66, 158, 176), 8),
    79: ("Au", 1.36, 2.14, (255, 209, 35), None),
    80: ("Hg", 1.32, 2.23, (184, 184, 208), None),
    82: ("Pb", 1.46, 2.02, (87, 89, 97), 4),
}
DEFAULT_ELEMENT = ("X", 1.50, 2.00, (255, 20, 147), None)   # Jmol's "unknown" pink

_FORMULA = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")

INFO_OPTIONS = (("lone_pairs", "show each central atom's lone pairs as translucent lobes (for "
                               "VSEPR, electron-domain or molecular-geometry labs)"),)


class BuildError(Exception):
    """The molecule can't be built; the message says why (it goes to the manifest)."""


# ── PubChem ───────────────────────────────────────────────────────────


_CACHE: dict[str, Optional[dict]] = {}   # url -> reply (None for 404), for this process


def _get_json(path: str) -> Optional[dict]:
    """GET ``PUBCHEM + path`` as JSON with the library's download (shared retry
    policy, size limit, network-failure classification). None on 404. Cached per URL
    (404s included) for the process; failures aren't cached."""
    url = PUBCHEM + path
    if url not in _CACHE:
        _CACHE[url] = _fetch_json(url)
    return _CACHE[url]


def _fetch_json(url: str) -> Optional[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "reply.json"
        try:
            library.download(url, dest)
        except library.DownloadError as exc:
            if exc.status == 404:
                return None
            raise BuildError(f"PubChem request failed: {exc}") from exc
        try:
            return json.loads(dest.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise BuildError(f"PubChem sent a reply that isn't JSON ({url})") from exc


def lookup(query: str) -> dict:
    """{cid, title, formula, matched_by} for a compound name, or a formula."""
    query = " ".join(query.split())
    if not query:
        raise BuildError("no compound name was given")
    matched_by = "name"
    reply = _get_json(f"/compound/name/{urllib.parse.quote(query, safe='')}/cids/JSON")
    if reply is None and _FORMULA.match(query):
        matched_by = "formula"
        reply = _get_json(f"/compound/fastformula/{urllib.parse.quote(query)}/cids/JSON")
    cids = ((reply or {}).get("IdentifierList") or {}).get("CID") or []
    if not cids:
        raise BuildError(f"PubChem has no compound named {query!r}")
    cid = int(cids[0])
    props = _get_json(f"/compound/cid/{cid}/property/Title,MolecularFormula/JSON") or {}
    row = ((props.get("PropertyTable") or {}).get("Properties") or [{}])[0]
    return {"cid": cid, "title": row.get("Title") or query, "formula": row.get("MolecularFormula") or "",
            "matched_by": matched_by, "query": query}


@dataclass
class Molecule:
    elements: list[int]                      # atomic numbers
    charges: list[int]
    bonds: list[tuple[int, int, int]]        # (atom index, atom index, order)
    coords: list[tuple[float, float, float]] = field(default_factory=list)
    # bonds PubChem types as other than single/double/triple: (atom, atom, type name)
    unusual: list[tuple[int, int, str]] = field(default_factory=list)

    def unusual_atoms(self) -> set[int]:
        return {i for a, b, _ in self.unusual for i in (a, b)}

    def neighbors(self, i: int) -> list[int]:
        return [b if a == i else a for a, b, _ in self.bonds if i in (a, b)]

    def bond_order_sum(self, i: int) -> int:
        return sum(o for a, b, o in self.bonds if i in (a, b))


def _parse_record(record: dict, with_coords: bool) -> Molecule:
    try:
        comp = record["PC_Compounds"][0]
        aids = comp["atoms"]["aid"]
        index = {aid: k for k, aid in enumerate(aids)}
        elements = [int(z) for z in comp["atoms"]["element"]]
        charges = [0] * len(aids)
        for c in comp["atoms"].get("charge") or []:
            charges[index[c["aid"]]] = int(c["value"])
        bonds_raw = comp.get("bonds") or {}
        bonds = []
        unusual = []
        for a, b, o in zip(bonds_raw.get("aid1", []), bonds_raw.get("aid2", []), bonds_raw.get("order", [])):
            o = int(o)
            if o in (1, 2, 3):
                bonds.append((index[a], index[b], o))
                continue
            kind = UNUSUAL_BONDS.get(o, "unknown")
            unusual.append((index[a], index[b], kind))
            if kind != "ionic":   # drawn as one stick; an ionic bond isn't drawn
                bonds.append((index[a], index[b], 1))
        mol = Molecule(elements, charges, bonds, unusual=unusual)
        if with_coords:
            c = comp["coords"][0]
            conf = c["conformers"][0]
            pos = {aid: (conf["x"][k], conf["y"][k], conf["z"][k]) for k, aid in enumerate(c["aid"])}
            mol.coords = [tuple(map(float, pos[aid])) for aid in aids]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise BuildError(f"PubChem's record couldn't be read ({type(exc).__name__}: {exc})") from exc
    return mol


def fetch(cid: int) -> tuple[Molecule, bool]:
    """The molecule, and whether its coordinates are PubChem's 3D conformer."""
    rec3d = _get_json(f"/compound/cid/{cid}/record/JSON?record_type=3d")
    if rec3d is not None:
        return _parse_record(rec3d, with_coords=True), True
    rec2d = _get_json(f"/compound/cid/{cid}/record/JSON?record_type=2d")
    if rec2d is None:
        raise BuildError(f"PubChem has no record for CID {cid}")
    return _parse_record(rec2d, with_coords=False), False


# ── Geometry ──────────────────────────────────────────────────────────


def _element(z: int):
    return ELEMENTS.get(z, DEFAULT_ELEMENT)


def lone_pair_count(mol: Molecule, i: int) -> int:
    """Lone pairs on atom ``i``: (valence electrons - bond orders - charge) / 2."""
    valence = _element(mol.elements[i])[4]
    if valence is None or i in mol.unusual_atoms():   # can't count through a dative bond
        return 0
    return max(0, (valence - mol.bond_order_sum(i) - mol.charges[i]) // 2)


def _unit(v):
    import numpy as np
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _ideal_slots(domains: int):
    """Ideal electron-domain directions, ordered so lone pairs fill the first slots."""
    import numpy as np
    s3 = math.sqrt(3) / 2
    table = {
        1: [(0, 0, 1)],
        2: [(1, 0, 0), (-1, 0, 0)],
        3: [(0, 1, 0), (-s3, -0.5, 0), (s3, -0.5, 0)],
        4: [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
        # trigonal bipyramid: equatorial first (lone pairs go there), then axial
        5: [(1, 0, 0), (-0.5, s3, 0), (-0.5, -s3, 0), (0, 0, 1), (0, 0, -1)],
        # octahedron: one axis first (two lone pairs end up opposite), then the plane
        6: [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
    }
    if domains not in table:
        raise BuildError(f"no idealized geometry for {domains} electron domains")
    return [_unit(np.array(v, dtype=float)) for v in table[domains]]


VSEPR_SHAPES = {  # (bonded atoms, lone pairs) -> VSEPR molecular geometry
    (2, 0): "linear", (3, 0): "trigonal planar", (2, 1): "bent", (4, 0): "tetrahedral",
    (3, 1): "trigonal pyramidal", (2, 2): "bent", (5, 0): "trigonal bipyramidal",
    (4, 1): "seesaw", (3, 2): "T-shaped", (2, 3): "linear", (6, 0): "octahedral",
    (5, 1): "square pyramidal", (4, 2): "square planar", (1, 0): "linear", (1, 1): "linear",
    (1, 2): "linear", (1, 3): "linear",
}


def central_atom(mol: Molecule) -> Optional[int]:
    """The atom every other atom is bonded to, with no other bonds; None otherwise."""
    n = len(mol.elements)
    if n < 2:
        return None
    for c in range(n):
        if sorted(mol.neighbors(c)) == [i for i in range(n) if i != c] and \
                all(c in (a, b) for a, b, _ in mol.bonds):
            return c
    return None


def idealize(mol: Molecule) -> tuple[dict[int, list], str]:
    """Idealized VSEPR coordinates for a one-center molecule (set on ``mol``). Returns
    the lone-pair directions on the center and the molecular geometry's name."""
    import numpy as np
    c = central_atom(mol)
    if c is None:
        raise BuildError("PubChem has no 3D coordinates, and the molecule has more than one "
                         "central atom (an idealized geometry needs exactly one)")
    if mol.unusual:
        kinds = ", ".join(sorted({k for _, _, k in mol.unusual}))
        raise BuildError(f"PubChem has no 3D coordinates, and the molecule has {kinds} bonds, "
                         "which an idealized VSEPR geometry can't place")
    neighbors = mol.neighbors(c)
    pairs = lone_pair_count(mol, c)
    slots = _ideal_slots(len(neighbors) + pairs)
    coords = [np.zeros(3)] * len(mol.elements)
    orders = {(a if a != c else b): o for a, b, o in mol.bonds}
    for slot, atom in zip(slots[pairs:], neighbors):
        length = (_element(mol.elements[c])[1] + _element(mol.elements[atom])[1]) \
            * BOND_SHORTENING[orders[atom]]
        coords[atom] = slot * length
    mol.coords = [tuple(float(x) for x in p) for p in coords]
    shape = VSEPR_SHAPES.get((len(neighbors), pairs), f"{len(neighbors)} bonded atoms, {pairs} lone pairs")
    return {c: slots[:pairs]} if pairs else {}, shape


def free_directions(bonds: list, count: int):
    """``count`` directions as far as possible from the bond directions (unit vectors)
    and from each other: lone pairs around an atom with real coordinates.
    Deterministic: a fixed start and a fixed number of repulsion steps."""
    import numpy as np
    bonds = [_unit(b) for b in bonds]
    base = -np.sum(bonds, axis=0) if bonds else np.array([0.0, 0.0, 1.0])
    if np.linalg.norm(base) < 1e-6:
        base = np.cross(bonds[0], [0.0, 0.0, 1.0])
        if np.linalg.norm(base) < 1e-6:
            base = np.cross(bonds[0], [1.0, 0.0, 0.0])
    base = _unit(base)
    ref = bonds[0] if bonds else np.array([1.0, 0.0, 0.0])
    perp = np.cross(base, ref)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(base, [1.0, 0.0, 0.0]) if abs(base[0]) < 0.9 else np.cross(base, [0.0, 1.0, 0.0])
    perp = _unit(perp)
    perp2 = np.cross(base, perp)
    pts = []
    for k in range(count):
        angle = 2 * math.pi * k / max(count, 1)
        offset = 0.0 if count == 1 else 0.6
        pts.append(_unit(base + offset * (math.cos(angle) * perp + math.sin(angle) * perp2)))
    for _ in range(400):
        moved = []
        for i, p in enumerate(pts):
            force = np.zeros(3)
            for q in bonds + [x for j, x in enumerate(pts) if j != i]:
                d = p - q
                dist = max(np.linalg.norm(d), 1e-3)
                force += d / dist ** 3
            force -= np.dot(force, p) * p   # along the sphere
            moved.append(_unit(p + 0.02 * force))
        pts = moved
    return pts


# ── Mesh ──────────────────────────────────────────────────────────────


def _material(name: str, rgba):
    from trimesh.visual.material import PBRMaterial
    return PBRMaterial(name=name, baseColorFactor=list(rgba), metallicFactor=0.0, roughnessFactor=0.55,
                       alphaMode="BLEND" if rgba[3] < 255 else "OPAQUE", doubleSided=rgba[3] < 255)


def _bond_offset(mol: Molecule, a: int, b: int, pos):
    """A unit vector across the bond a-b, in the plane of a neighbouring atom when there
    is one (so a double bond lies flat, as drawn in chemistry)."""
    import numpy as np
    axis = _unit(pos[b] - pos[a])
    for atom, other in ((a, b), (b, a)):
        for n in mol.neighbors(atom):
            if n == other:
                continue
            v = pos[n] - pos[atom]
            v = v - np.dot(v, axis) * axis
            if np.linalg.norm(v) > 1e-6:
                return _unit(v)
    for ref in ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]):
        v = np.cross(axis, ref)
        if np.linalg.norm(v) > 1e-6:
            return _unit(v)
    return np.array([1.0, 0.0, 0.0])


def build_scene(mol: Molecule, lone_pairs: dict[int, list], detail: str = "standard"):
    """The ball-and-stick ``trimesh.Scene`` (one mesh per material), unscaled, and the
    centers of its atoms and lone-pair lobes (for the thumbnail's view)."""
    import numpy as np
    import trimesh

    subdivisions, sections = DETAIL[detail]
    pos = [np.asarray(p, dtype=float) for p in mol.coords]
    centers = list(pos)
    parts: dict[str, list] = {}       # material name -> meshes, in first-seen order
    colors: dict[str, tuple] = {}

    def add(name, rgba, mesh):
        parts.setdefault(name, []).append(mesh)
        colors[name] = rgba

    def symbol(i):
        return _element(mol.elements[i])[0]

    for i, z in enumerate(mol.elements):
        sym, _, vdw, rgb, _ = _element(z)
        ball = trimesh.creation.icosphere(subdivisions=subdivisions, radius=BALL_FACTOR * vdw)
        ball.apply_translation(pos[i])
        add(sym, (*rgb, 255), ball)
    for a, b, order in mol.bonds:
        across = _bond_offset(mol, a, b, pos) if order > 1 else np.zeros(3)
        radius = MULTI_BOND_RADIUS.get(order, BOND_RADIUS)
        gap = MULTI_BOND_GAP.get(order, 0.0)
        offsets = [0.0] if order == 1 else ([-gap / 2, gap / 2] if order == 2 else [-gap, 0.0, gap])
        mid = (pos[a] + pos[b]) / 2
        for off in offsets:
            shift = across * off
            for atom, end in ((a, pos[a]), (b, pos[b])):
                stick = trimesh.creation.cylinder(radius=radius, segment=[end + shift, mid + shift],
                                                  sections=sections)
                add(symbol(atom), (*_element(mol.elements[atom])[3], 255), stick)
    across_r, along_r = LONE_PAIR_SIZE
    for i, directions in lone_pairs.items():
        ball_r = BALL_FACTOR * _element(mol.elements[i])[2]
        for d in directions:
            lobe = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
            lobe.apply_scale([across_r, across_r, along_r])
            # rotate the lobe's z axis onto d, then move it just outside the ball
            lobe.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], d))
            lobe.apply_translation(pos[i] + d * (ball_r + along_r * 0.85))
            centers.append(pos[i] + d * (ball_r + along_r * 0.85))
            add("lone_pair", LONE_PAIR_COLOR, lobe)

    scene = trimesh.Scene()
    for name, meshes in parts.items():
        mesh = trimesh.util.concatenate(meshes)
        mesh.visual = trimesh.visual.TextureVisuals(material=_material(name, colors[name]))
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    return scene, centers


def normalize(scene) -> tuple[list[float], int, "np.ndarray"]:
    """Center the scene on its bounding box and scale its largest side to 1.0 (the
    model convention). Returns the bounds [x, y, z], the triangle count and the 4x4
    transform applied."""
    import numpy as np
    lo, hi = scene.bounds
    size = float(np.max(hi - lo))
    if size <= 0:
        raise BuildError("the model has no size")
    import trimesh
    matrix = trimesh.transformations.scale_matrix(1.0 / size) @ \
        trimesh.transformations.translation_matrix(-(lo + hi) / 2)
    for geom in scene.geometry.values():
        geom.apply_transform(matrix)
    lo, hi = scene.bounds
    bounds = [round(float(x), 6) for x in (hi - lo)]
    return bounds, int(sum(len(g.faces) for g in scene.geometry.values())), matrix


def main_axes(points):
    """The points' main axes as rows (largest spread first), each with a fixed sign,
    forming a right-handed frame: a left-handed one would show the mirror image (the
    other enantiomer of a chiral molecule) and turn every triangle inside out."""
    import numpy as np
    pts = np.asarray(points, dtype=float)
    pts = pts - pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(pts.T @ pts)       # ascending spread
    vals, axes = vals[::-1], vecs[:, ::-1].T.copy()  # rows: largest, middle, smallest
    # Equal spreads (a tetrahedron, an octahedron, around a linear molecule's axis) leave
    # the axes free within their subspace, and LAPACK builds pick them differently; there
    # the world axes, projected into the subspace, choose them, so every machine draws
    # the same view.
    tol = 1e-6 * max(float(vals[0]), 1e-12)
    k = 0
    while k < 3:
        j = k
        while j + 1 < 3 and abs(vals[j + 1] - vals[k]) <= tol:
            j += 1
        if j > k:
            basis = axes[k:j + 1]
            chosen = []
            for w in np.eye(3):
                v = basis.T @ (basis @ w)          # the world axis projected into the subspace
                for c in chosen:
                    v = v - np.dot(v, c) * c
                if np.linalg.norm(v) > 1e-6 and len(chosen) < j - k + 1:
                    chosen.append(v / np.linalg.norm(v))
            axes[k:j + 1] = chosen
        k = j + 1
    for k in range(2):                              # a fixed sign for the first two
        if axes[k][np.argmax(np.abs(axes[k]))] < 0:
            axes[k] = -axes[k]
    axes[2] = np.cross(axes[0], axes[1])            # right-handed, never a reflection
    return axes


def _rotation(rot_y: float, rot_x: float):
    import numpy as np
    ry = np.array([[math.cos(rot_y), 0, math.sin(rot_y)], [0, 1, 0], [-math.sin(rot_y), 0, math.cos(rot_y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(rot_x), -math.sin(rot_x)], [0, math.sin(rot_x), math.cos(rot_x)]])
    return rx @ ry


def draw_thumbnail(scene, dest: Path, points=None) -> None:
    """A small render to check the model by eye: each mesh's triangles, flat-shaded
    in its material color, back to front (painter's order). The view is a 3/4 view of
    the model's own main axes (largest spread across, smallest toward the viewer). With
    ``points`` (atom and lone-pair centers), it keeps that view unless two points come
    close on screen (one hides another; a tetrahedron's main axes are arbitrary): then
    it takes the nearest turn, from a fixed set, that keeps them reasonably apart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PolyCollection

    axes = main_axes(np.vstack([g.vertices for g in scene.geometry.values()]))
    default = (30, -35)
    turns = sorted({(y, x) for x in (-20, -35, -50) for y in range(0, 360, 15)} | {default},
                   key=lambda t: (abs(t[1] - default[1]) / 15 + min(abs(t[0] - default[0]),
                                                                   360 - abs(t[0] - default[0])) / 15, t))
    views = [_rotation(math.radians(y), math.radians(x)) @ axes for y, x in turns]
    view = views[0]
    if points is not None and len(points) >= 2:
        pts2 = np.asarray(points, dtype=float)
        upper = np.triu_indices(len(pts2), k=1)

        def spread(v):   # the closest two points on screen (rounded: no float-noise picks)
            flat = (pts2 @ v.T)[:, :2]
            gaps = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)[upper]
            return round(float(gaps.min()), 4)
        scores = [spread(v) for v in views]
        enough = 0.6 * max(scores)
        view = next(v for v, sc in zip(views, scores) if sc >= enough)
    light = _unit([0.4, 0.6, 1.0])
    polys, faces_rgba, depth = [], [], []
    for geom in scene.geometry.values():
        mat = geom.visual.material
        base = np.asarray(mat.baseColorFactor, dtype=float) / 255.0
        verts = geom.vertices @ view.T
        tris = verts[geom.faces]
        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norms > 0, norms, 1)
        keep = normals[:, 2] > 0 if base[3] >= 1 else np.ones(len(tris), dtype=bool)
        shade = 0.35 + 0.65 * np.clip(normals @ light, 0, 1)
        for tri, s in zip(tris[keep], shade[keep]):
            polys.append(tri[:, :2])
            faces_rgba.append((*(base[:3] * s), base[3]))
            depth.append(tri[:, 2].mean())
    order = np.argsort(depth)
    fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.add_collection(PolyCollection([polys[k] for k in order], facecolors=[faces_rgba[k] for k in order],
                                     edgecolors="none", antialiased=False))
    flat = np.vstack(polys) if polys else np.zeros((1, 2))
    center = (flat.min(axis=0) + flat.max(axis=0)) / 2
    half = max(float(np.max(flat.max(axis=0) - flat.min(axis=0))) / 2, 1e-6) * 1.08
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.savefig(dest, format="png", facecolor="white")
    plt.close(fig)


# ── The builder ───────────────────────────────────────────────────────


def _rdkit_coords(mol: Molecule) -> Optional[str]:
    """Embed the molecule with RDKit (optional). None when it worked, else why not."""
    if not USE_RDKIT:
        return "the RDKit fallback is turned off"
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog("rdApp.*")   # its reason goes into the result, not stderr
    except ImportError:
        return "RDKit isn't installed to embed it"
    rw = Chem.RWMol()
    for z, q in zip(mol.elements, mol.charges):
        atom = Chem.Atom(z)
        atom.SetFormalCharge(q)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    kinds = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for a, b, o in mol.bonds:
        rw.AddBond(a, b, kinds[o])
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        if AllChem.EmbedMolecule(m, randomSeed=0xA51) != 0:
            return "RDKit couldn't embed it"
        AllChem.MMFFOptimizeMolecule(m)
    except Exception as exc:
        return f"RDKit couldn't embed it ({type(exc).__name__}: {' '.join(str(exc).split())[:200]})"
    conf = m.GetConformer()
    mol.coords = [tuple(float(x) for x in conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())]
    return None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "molecule"


def build(request: BuildRequest, out_dir: Path) -> BuildResult:
    """The ``molecule`` builder: a ball-and-stick GLB and a thumbnail in ``out_dir``."""
    import numpy as np
    try:
        import trimesh  # noqa: F401
    except ImportError:
        return BuildResult(False, "the molecule builder needs trimesh (pip install -r requirements.txt)")
    want_pairs = "lone_pairs" in request.options
    try:
        found = lookup(request.input)
        mol, has_3d = fetch(found["cid"])
        shape = None
        center_pairs: dict[int, list] = {}
        if has_3d:
            geometry = "pubchem_3d"
        elif len(mol.elements) == 1:
            mol.coords, geometry = [(0.0, 0.0, 0.0)], "single_atom"
        elif central_atom(mol) is not None and not mol.unusual:
            center_pairs, shape = idealize(mol)
            geometry = "idealized_vsepr"
        else:
            rdkit_problem = _rdkit_coords(mol)
            if rdkit_problem:
                if mol.unusual:
                    kinds = ", ".join(sorted({k for _, _, k in mol.unusual}))
                    why = f"it has {kinds} bonds, which an idealized geometry can't place"
                else:
                    why = "it has more than one central atom (no idealized geometry)"
                raise BuildError(f"PubChem has no 3D coordinates for it, {why}, and {rdkit_problem}")
            geometry = "rdkit"
        lone_pairs: dict[int, list] = {}
        pairs_per_atom = {}
        for i in range(len(mol.elements)):
            n = lone_pair_count(mol, i)
            if n and len(mol.neighbors(i)) >= 2:   # central atoms only, as VSEPR draws them
                pairs_per_atom[i] = n
        if shape is None:
            c = central_atom(mol)
            if c is not None:
                shape = VSEPR_SHAPES.get((len(mol.neighbors(c)), lone_pair_count(mol, c)))
        if want_pairs:
            pos = [np.asarray(p, dtype=float) for p in mol.coords]
            for i, n in pairs_per_atom.items():
                lone_pairs[i] = center_pairs.get(i) or free_directions(
                    [pos[j] - pos[i] for j in mol.neighbors(i)], n)
        for detail in DETAIL:
            scene, centers = build_scene(mol, lone_pairs, detail)
            bounds, triangles, matrix = normalize(scene)
            if triangles <= library.MAX_MODEL_TRIANGLES:
                break
        else:
            raise BuildError(f"the model has {triangles} triangles even at reduced detail, more "
                             f"than the headset budget of {library.MAX_MODEL_TRIANGLES}")
    except BuildError as exc:
        return BuildResult(False, str(exc))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, thumb = out_dir / library.ROLE_FILES["model"], out_dir / library.ROLE_FILES["thumbnail"]
    model.write_bytes(scene.export(file_type="glb"))
    points = np.c_[np.asarray(centers, dtype=float), np.ones(len(centers))] @ matrix.T
    draw_thumbnail(scene, thumb, points[:, :3])

    title, cid, formula = found["title"], found["cid"], found["formula"]
    suffix = "_lone_pairs" if lone_pairs else ""
    method = {"pubchem_3d": "PubChem's 3D conformer",
              "idealized_vsepr": "an idealized VSEPR geometry from PubChem's 2D record",
              "rdkit": "RDKit embedding of PubChem's 2D record",
              "single_atom": "PubChem's record of a single atom"}[geometry]
    look = "ball-and-stick, CPK colors" + (", lone pairs as translucent lobes" if lone_pairs else "")
    entry = {
        "asset_id": f"pubchem_{cid}{suffix}",
        "kind": "model",
        "name": f"{_slug(title)}_molecule{suffix}",
        "form": "structure",
        "description": f"Ball-and-stick model of {title} ({formula})"
                       + (f", {shape}" if shape else "") + f": {look}. PubChem CID {cid}.",
        "tags": sorted({*_slug(title).split("_"), formula.lower(), "molecule", "ball and stick"} - {""}),
        "bounds": bounds,
        "triangles": triangles,
        "provenance": {
            "method": "builder", "source_site": "PubChem", "page_url": PUBCHEM_PAGE.format(cid=cid),
            "record_id": str(cid), "title": title, "license": "public-domain",
            "license_url": PUBCHEM_LICENSE_URL,
            "modifications": [f"{look}, built from {method}", "scaled to fit a unit cube"],
            "notes": builder_note(), "builder": "molecule", "builder_version": VERSION,
        },
    }
    details = {
        "cid": cid, "title": title, "formula": formula, "matched_by": found["matched_by"],
        "geometry": geometry, "shape": shape, "atoms": len(mol.elements), "bonds": len(mol.bonds),
        "bond_orders": sorted(o for _, _, o in mol.bonds),
        "lone_pairs": {f"{_element(mol.elements[i])[0]}{i + 1}": n for i, n in pairs_per_atom.items()},
        "lone_pairs_drawn": bool(lone_pairs), "bounds": bounds, "triangles": triangles,
        "detail": detail, "unusual_bonds": sorted(k for _, _, k in mol.unusual),
        "builder_version": VERSION,
    }
    return BuildResult(True, files={"model": model, "thumbnail": thumb}, entry=entry, details=details)


def builder_note() -> str:
    """The provenance note naming the builder version that made an entry."""
    return f"built by the molecule builder, version {VERSION} (#111)"


INFO = BuilderInfo(
    name="molecule",
    kind="model",
    description="A ball-and-stick model of any small molecule (one compound: water, methane, glucose, "
                "caffeine, sulfur hexafluoride), with real or ideal VSEPR geometry and standard CPK "
                "colors, from PubChem. Not proteins, DNA or crystals.",
    needs="the compound's common or IUPAC name in English (e.g. 'ammonia', 'xenon tetrafluoride'), "
          "or its formula when there is no name",
    options=INFO_OPTIONS,
    build=build,
    version=VERSION,
)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build one molecule from PubChem (a check by eye).")
    ap.add_argument("compound", help="a compound name or formula, e.g. 'xenon tetrafluoride'")
    ap.add_argument("--lone-pairs", action="store_true", help="show the central atoms' lone pairs")
    ap.add_argument("--out", required=True, help="folder for model.glb and thumbnail.png")
    args = ap.parse_args(argv)
    result = build(BuildRequest(name=args.compound, kind="model", input=args.compound,
                                options=("lone_pairs",) if args.lone_pairs else ()), Path(args.out))
    if not result.ok:
        print(f"can't build {args.compound!r}: {result.reason}", file=sys.stderr)
        return 1
    print(json.dumps(result.details, indent=2))
    print(f"wrote {result.files['model']} and {result.files['thumbnail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
