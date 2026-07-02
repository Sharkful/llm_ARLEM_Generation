"""Integration tier: exercises the real Blender binary. Self-skips if
Blender isn't installed/detected, per the implementation plan's Stage 1.1
testing design (same pattern as test_openscad_compile.py for OpenSCAD).
"""
from pathlib import Path

import pygltflib
import pytest

import config
from pipeline.mesh_processor import normalize_mesh
from pipeline.openscad_generator import compile_scad

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not config.BLENDER_BIN, reason="Blender binary not found on this machine"
)


@pytest.fixture
def bracket_stl(tmp_path):
    """Compile the Stage 3 fixture .scad into a real STL to feed Stage 4."""
    if not config.OPENSCAD_BIN:
        pytest.skip("OpenSCAD binary not found on this machine (needed to produce the STL fixture)")
    source = (FIXTURES_DIR / "bracket_good.scad").read_text(encoding="utf-8")
    scad_path = tmp_path / "bracket.scad"
    stl_path = tmp_path / "bracket.stl"
    success, stderr = compile_scad(source, scad_path, stl_path)
    assert success, f"fixture STL failed to compile: {stderr}"
    return stl_path


def test_normalize_mesh_centers_and_scales_stl(tmp_path, bracket_stl):
    glb_path = tmp_path / "model.glb"
    result = normalize_mesh(
        asset_id="bracket_test",
        source_path=bracket_stl,
        source_format="stl",
        glb_path=glb_path,
        target_size_m=0.10,
        pivot="center",
    )

    assert result.success is True, result.error_message
    assert glb_path.exists()
    # Fixture bracket's largest original dimension (~0.05m) scaled to a
    # 0.10m target -> final bounds should be roughly double the original.
    assert result.scale_applied == pytest.approx(2.0, rel=0.01)
    assert max(result.final_bounds_m) == pytest.approx(0.10, abs=0.001)
    assert result.decimated is False
    assert result.triangle_count_before == result.triangle_count_after


def test_normalize_mesh_base_center_pivot_sits_on_origin(tmp_path, bracket_stl):
    glb_path = tmp_path / "model.glb"
    result = normalize_mesh(
        asset_id="bracket_test_base",
        source_path=bracket_stl,
        source_format="stl",
        glb_path=glb_path,
        target_size_m=0.05,
        pivot="base_center",
    )
    assert result.success is True

    gltf = pygltflib.GLTF2().load(str(glb_path))
    pos_accessor_index = gltf.meshes[0].primitives[0].attributes.POSITION
    accessor = gltf.accessors[pos_accessor_index]

    # base_center pivot: the mesh's lowest Y (up axis) should sit at 0,
    # not be centered around 0.
    assert accessor.min[1] == pytest.approx(0.0, abs=0.001)
    assert accessor.max[1] > 0.0


def test_normalize_mesh_center_pivot_is_symmetric(tmp_path, bracket_stl):
    glb_path = tmp_path / "model.glb"
    result = normalize_mesh(
        asset_id="bracket_test_center",
        source_path=bracket_stl,
        source_format="stl",
        glb_path=glb_path,
        target_size_m=0.05,
        pivot="center",
    )
    assert result.success is True

    gltf = pygltflib.GLTF2().load(str(glb_path))
    pos_accessor_index = gltf.meshes[0].primitives[0].attributes.POSITION
    accessor = gltf.accessors[pos_accessor_index]

    # center pivot: bounds should be roughly symmetric around 0 on all axes.
    for lo, hi in zip(accessor.min, accessor.max):
        assert lo == pytest.approx(-hi, abs=0.001)


def test_normalize_mesh_decimates_high_poly_meshes(tmp_path, monkeypatch):
    if not config.OPENSCAD_BIN:
        pytest.skip("OpenSCAD binary not found on this machine (needed to produce a high-poly STL)")

    # A high-$fn sphere produces far more triangles than the default
    # threshold; lower the threshold so this test doesn't need a huge mesh.
    monkeypatch.setattr(config, "MAX_TRIANGLE_COUNT", 500)

    scad_path = tmp_path / "hires.scad"
    stl_path = tmp_path / "hires.stl"
    success, stderr = compile_scad("sphere(r=0.05, $fn=200);", scad_path, stl_path)
    assert success, f"high-poly fixture failed to compile: {stderr}"

    glb_path = tmp_path / "model.glb"
    result = normalize_mesh(
        asset_id="hires_test", source_path=stl_path, source_format="stl",
        glb_path=glb_path, target_size_m=0.05, pivot="center",
    )

    assert result.success is True
    assert result.decimated is True
    assert result.triangle_count_before > 500
    assert result.triangle_count_after < result.triangle_count_before
    # Bounds should still be correctly scaled despite decimation.
    assert max(result.final_bounds_m) == pytest.approx(0.05, abs=0.002)


def test_normalize_mesh_raises_for_missing_source(tmp_path):
    with pytest.raises(Exception):
        normalize_mesh(
            asset_id="missing",
            source_path=tmp_path / "does_not_exist.stl",
            source_format="stl",
            glb_path=tmp_path / "out.glb",
            target_size_m=0.05,
        )
