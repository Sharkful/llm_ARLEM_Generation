import pytest

from pipeline.mesh_processor import (
    MeshNormalizationResult,
    MeshProcessingError,
    normalize_mesh,
    save_mesh_meta,
)


def test_normalize_mesh_raises_when_blender_missing(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "BLENDER_BIN", None)
    with pytest.raises(MeshProcessingError, match="Blender binary not found"):
        normalize_mesh(
            asset_id="x", source_path=tmp_path / "nope.stl", source_format="stl",
            glb_path=tmp_path / "out.glb", target_size_m=0.05,
        )


def test_normalize_mesh_raises_when_source_missing(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "BLENDER_BIN", "fake_blender_path_for_test")
    with pytest.raises(MeshProcessingError, match="Source mesh not found"):
        normalize_mesh(
            asset_id="x", source_path=tmp_path / "does_not_exist.stl", source_format="stl",
            glb_path=tmp_path / "out.glb", target_size_m=0.05,
        )


def test_save_mesh_meta_writes_json(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path)

    result = MeshNormalizationResult(
        asset_id="bracket_01",
        success=True,
        source_path="source.stl",
        glb_path="model.glb",
        original_bounds_m=[1.0, 1.0, 1.0],
        final_bounds_m=[0.05, 0.05, 0.05],
        pivot="center",
        scale_applied=0.05,
        triangle_count_before=100,
        triangle_count_after=100,
        decimated=False,
    )
    meta_path = save_mesh_meta("bracket_01", result)
    assert meta_path == tmp_path / "generated" / "bracket_01" / "meta.json"
    assert meta_path.exists()

    import json

    loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    assert loaded["asset_id"] == "bracket_01"
    assert loaded["final_bounds_m"] == [0.05, 0.05, 0.05]
