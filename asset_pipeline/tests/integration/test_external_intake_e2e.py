"""Integration tier: full Stage 6a intake against the real Blender binary,
using an OpenSCAD-compiled STL as the stand-in for a manually downloaded
file. Self-skips without Blender/OpenSCAD, per the Stage 1.1 testing design.
"""
import json

import pytest

import config
from pipeline.catalog_writer import load_catalog
from pipeline.external_intake import intake_asset
from pipeline.openscad_generator import compile_scad

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not (config.BLENDER_BIN and config.OPENSCAD_BIN),
    reason="Blender and OpenSCAD are both required for the intake e2e test",
)


def test_intake_e2e_normalizes_and_catalogs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")
    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(config, "CATALOG_PATH", tmp_path / "library" / "catalog.json")

    # Stand-in for the human's downloaded file: compile the bracket fixture.
    intake_dir = tmp_path / "intake" / "bracket_ext"
    intake_dir.mkdir(parents=True)
    source = (FIXTURES_DIR / "bracket_good.scad").read_text(encoding="utf-8")
    success, stderr = compile_scad(source, tmp_path / "b.scad", intake_dir / "bracket.stl")
    assert success, f"fixture STL failed to compile: {stderr}"

    (intake_dir / "source.json").write_text(
        json.dumps(
            {
                "display_name": "External bracket",
                "license": "CC0",
                "source_site": "example-cc0-library.org",
                "target_size_m": 0.2,
                "pivot": "base_center",
                "tags": ["bracket", "hardware"],
            }
        ),
        encoding="utf-8",
    )

    result = intake_asset("bracket_ext")

    assert result.success is True, result.error_message
    out_dir = tmp_path / "library" / "imported" / "bracket_ext"
    assert (out_dir / "model.glb").exists()
    assert (out_dir / "original.stl").exists()
    assert (out_dir / "LICENSE.txt").exists()
    assert (out_dir / "meta.json").exists()

    [entry] = load_catalog()
    assert entry.asset_id == "bracket_ext"
    assert entry.asset_class == "imported"
    assert entry.review_level == 3
    assert entry.provenance.license_status == "approved"
    # Normalized scale actually enforced: largest bound == target_size_m.
    assert max(entry.canonical_bounds_m) == pytest.approx(0.2, abs=0.002)
