"""Integration tier per asset_pipeline/README.md §Testing: exercises the
real OpenSCAD binary. Self-skips if OpenSCAD isn't installed/detected so
`pytest tests` degrades gracefully on a machine without it, per the
implementation plan's Stage 1.1 testing design.
"""
from pathlib import Path

import pytest

import config
from pipeline.openscad_generator import compile_scad, measure_stl_bounds_m

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not config.OPENSCAD_BIN, reason="OpenSCAD binary not found on this machine"
)


def test_compile_scad_succeeds_on_valid_fixture(tmp_path):
    source = (FIXTURES_DIR / "bracket_good.scad").read_text(encoding="utf-8")
    scad_path = tmp_path / "bracket.scad"
    stl_path = tmp_path / "bracket.stl"

    success, stderr = compile_scad(source, scad_path, stl_path)

    assert success is True
    assert stl_path.exists()
    assert stl_path.stat().st_size > 0
    assert scad_path.read_text(encoding="utf-8") == source


def test_compile_scad_fails_on_broken_fixture(tmp_path):
    source = (FIXTURES_DIR / "bracket_broken.scad").read_text(encoding="utf-8")
    scad_path = tmp_path / "broken.scad"
    stl_path = tmp_path / "broken.stl"

    success, stderr = compile_scad(source, scad_path, stl_path)

    assert success is False
    assert stderr  # OpenSCAD should report a parse error
    assert not stl_path.exists() or stl_path.stat().st_size == 0


def test_measure_stl_bounds_matches_known_fixture_geometry(tmp_path):
    # bracket_good.scad is two overlapping cubes:
    #   cube([0.05, 0.005, 0.02])  and  cube([0.005, 0.05, 0.02])
    # so the union's bounding box should be [0.05, 0.05, 0.02].
    source = (FIXTURES_DIR / "bracket_good.scad").read_text(encoding="utf-8")
    scad_path = tmp_path / "bracket.scad"
    stl_path = tmp_path / "bracket.stl"

    success, _ = compile_scad(source, scad_path, stl_path)
    assert success is True

    bounds = measure_stl_bounds_m(stl_path)
    assert bounds == pytest.approx([0.05, 0.05, 0.02], abs=0.001)


def test_generate_parametric_asset_end_to_end_with_real_openscad(tmp_path, monkeypatch):
    """Full loop (no LLM) -- a fake plan_fn returns the real fixture source,
    but compile_scad and measure_stl_bounds_m are the real implementations
    shelling out to openscad.exe, proving the plumbing in
    generate_parametric_asset works against the actual binary, not just
    mocked stand-ins.
    """
    from models.spec_models import AssetSpec
    from pipeline.openscad_generator import generate_parametric_asset

    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path)

    source = (FIXTURES_DIR / "bracket_good.scad").read_text(encoding="utf-8")

    def fake_plan_fn(spec, repair_context):
        from models.generation_models import OpenSCADPlan

        plan = OpenSCADPlan(
            parameters={"width": 0.05, "height": 0.05, "depth": 0.02},
            scad_source=source,
            expected_bounds_m=[0.05, 0.05, 0.02],
            notes="fixture bracket",
        )
        return plan, None

    spec = AssetSpec(object_id="bracket_fixture", description="a test L-bracket")
    result, logs = generate_parametric_asset(spec, "bracket_fixture", generate_plan_fn=fake_plan_fn)

    assert result.success is True
    assert result.bounds_diverge is False
    assert Path(result.stl_path).exists()
    assert logs == []
