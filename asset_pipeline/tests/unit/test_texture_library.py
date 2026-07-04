import json

import pytest
from PIL import Image

import config
from models.catalog_models import ProvenanceInfo, UVInfo
from pipeline import texture_intake, texture_matcher
from pipeline.external_intake import IntakeError
from models.texture_models import TextureAsset, mapping_fits
from pipeline.texture_index import append_texture, find_texture, load_index


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")
    (tmp_path / "library" / "textures").mkdir(parents=True)
    (tmp_path / "intake").mkdir()
    return tmp_path


def _texture(texture_id="earth_daymap", mapping="equirectangular", **overrides):
    defaults = dict(
        texture_id=texture_id,
        display_name=texture_id,
        mapping=mapping,
        maps={"albedo": f"textures/{texture_id}/albedo.jpg"},
        semantic_type="earth_surface",
        tags=["earth", "planet", "daymap"],
        authentic=True,
        provenance=ProvenanceInfo(
            source_type="external_approved", license="CC-BY",
            license_status="approved", original_author="X",
        ),
    )
    defaults.update(overrides)
    return TextureAsset(**defaults)


# ── index ─────────────────────────────────────────────────────────────────

def test_index_roundtrip_and_uniqueness(sandbox):
    append_texture(_texture())
    assert find_texture("earth_daymap") is not None
    with pytest.raises(ValueError, match="already exists"):
        append_texture(_texture())
    assert len(load_index()) == 1


# ── mapping compatibility ─────────────────────────────────────────────────

def test_equirect_fits_only_equirect_targets():
    sphere = UVInfo(status="builtin", convention="equirect")
    bracket = UVInfo(status="generated", convention="generic")
    assert mapping_fits("equirectangular", sphere) is True
    assert mapping_fits("equirectangular", bracket) is False
    assert mapping_fits("equirectangular", None) is False


def test_tileable_fits_anything_with_uvs():
    assert mapping_fits("tileable", UVInfo(status="generated", convention="generic")) is True
    assert mapping_fits("tileable", UVInfo(status="none", convention="none")) is False
    assert mapping_fits("tileable", None) is True  # permissive when unknown


# ── matcher ───────────────────────────────────────────────────────────────

def test_match_texture_finds_semantic_hit_with_compatible_mapping(sandbox):
    append_texture(_texture())
    sphere = UVInfo(status="builtin", convention="equirect")

    match = texture_matcher.match_texture("earth daymap surface", target_uv=sphere)

    assert match is not None
    assert match.texture.texture_id == "earth_daymap"


def test_match_texture_excludes_geometric_mismatch(sandbox):
    append_texture(_texture())
    bracket = UVInfo(status="generated", convention="generic")

    assert texture_matcher.match_texture("earth daymap", target_uv=bracket) is None


def test_match_texture_returns_none_below_threshold(sandbox):
    append_texture(_texture())
    assert texture_matcher.match_texture("wooden crate slats") is None


def test_identity_guard_blocks_llm_query_drift(sandbox):
    """Regression: the LLM's texture_query said 'earth daymap' (parroting a
    prompt example) for a MARS object -- the object's own description must
    gate the bind."""
    append_texture(_texture("earth_daymap", semantic_type="earth_surface",
                            tags=["earth", "planet", "daymap"]))
    append_texture(_texture("mars_surface", semantic_type="mars_surface",
                            tags=["mars", "planet", "surface", "red"]))
    sphere = UVInfo(status="builtin", convention="equirect")

    # Drifted query, honest description: earth must be excluded, mars can win.
    drifted = texture_matcher.match_texture(
        "earth daymap equirectangular", target_uv=sphere,
        must_relate_to="the planet mars with its red dusty surface",
    )
    assert drifted is None or drifted.texture.texture_id != "earth_daymap"

    honest = texture_matcher.match_texture(
        "mars red surface", target_uv=sphere,
        must_relate_to="the planet mars with its red dusty surface",
    )
    assert honest is not None
    assert honest.texture.texture_id == "mars_surface"


def test_prefer_authentic_breaks_ties(sandbox):
    append_texture(_texture("earth_proc", authentic=False, tags=["earth", "planet"]))
    append_texture(_texture("earth_real", authentic=True, tags=["earth", "planet"]))

    match = texture_matcher.match_texture("earth planet", prefer_authentic=True)
    assert match.texture.texture_id == "earth_real"


# ── intake ────────────────────────────────────────────────────────────────

def _write_texture_intake(sandbox, texture_id="earth_daymap", size=(400, 200), source_overrides=None):
    intake_dir = sandbox / "intake" / texture_id
    intake_dir.mkdir(parents=True)
    Image.new("RGB", size, (40, 80, 160)).save(intake_dir / f"{texture_id}.jpg")
    source = {
        "display_name": "Earth daymap",
        "license": "CC-BY-4.0",
        "mapping": "equirectangular",
        "original_author": "Solar System Scope",
        "source_site": "solarsystemscope.com",
        "semantic_type": "earth_surface",
        "tags": ["earth", "planet"],
    }
    source.update(source_overrides or {})
    for key in [k for k, v in source.items() if v is None]:
        del source[key]
    (intake_dir / "source.json").write_text(json.dumps(source), encoding="utf-8")
    return intake_dir


def test_intake_texture_indexes_with_provenance(sandbox):
    _write_texture_intake(sandbox)

    result = texture_intake.intake_texture("earth_daymap")

    assert result.success is True
    texture = find_texture("earth_daymap")
    assert texture.mapping == "equirectangular"
    assert texture.resolution == [400, 200]
    assert texture.provenance.license_status == "approved"
    assert texture.provenance.attribution_required is True
    payload = sandbox / "library" / "textures" / "earth_daymap"
    assert (payload / "albedo.jpg").exists()
    assert "Attribution REQUIRED" in (payload / "LICENSE.txt").read_text(encoding="utf-8")


def test_intake_texture_warns_on_bad_equirect_aspect(sandbox):
    _write_texture_intake(sandbox, size=(300, 300))  # square, not 2:1

    result = texture_intake.intake_texture("earth_daymap")

    assert any("not ~2:1" in w for w in result.warnings)


def test_intake_texture_downscales_oversized_and_keeps_original(sandbox, monkeypatch):
    monkeypatch.setattr(config, "TEXTURE_MAX_DIM", 128)
    _write_texture_intake(sandbox, size=(512, 256))

    result = texture_intake.intake_texture("earth_daymap")

    payload = sandbox / "library" / "textures" / "earth_daymap"
    with Image.open(payload / "albedo.jpg") as img:
        assert max(img.size) == 128
    assert (payload / "original_earth_daymap.jpg").exists()
    assert any("downscaled" in w for w in result.warnings)


def test_intake_texture_rejects_disallowed_license(sandbox):
    _write_texture_intake(sandbox, source_overrides={"license": "CC-BY-NC"})
    with pytest.raises(IntakeError, match="allowlist"):
        texture_intake.intake_texture("earth_daymap")


def test_intake_texture_rejects_duplicate_id(sandbox):
    _write_texture_intake(sandbox)
    texture_intake.intake_texture("earth_daymap")
    with pytest.raises(IntakeError, match="already exists"):
        texture_intake.intake_texture("earth_daymap")


def test_intake_multi_map_set_detected_by_filename(sandbox):
    intake_dir = sandbox / "intake" / "rusty_metal"
    intake_dir.mkdir(parents=True)
    for name in ("rusty_metal_diff_1k.jpg", "rusty_metal_nor_gl_1k.jpg", "rusty_metal_rough_1k.jpg"):
        Image.new("RGB", (64, 64)).save(intake_dir / name)
    (intake_dir / "source.json").write_text(json.dumps({
        "display_name": "Rusty metal", "license": "CC0", "mapping": "tileable",
        "tile_size_m": [2.0, 2.0],
    }), encoding="utf-8")

    result = texture_intake.intake_texture("rusty_metal")

    texture = result.texture
    assert set(texture.maps) == {"albedo", "normal_gl", "roughness"}
    assert texture.tile_size_m == [2.0, 2.0]


def test_intake_unclassifiable_file_asks_for_explicit_maps(sandbox):
    intake_dir = sandbox / "intake" / "mystery"
    intake_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(intake_dir / "a.jpg")
    Image.new("RGB", (64, 64)).save(intake_dir / "b.jpg")
    (intake_dir / "source.json").write_text(json.dumps({
        "display_name": "Mystery", "license": "CC0", "mapping": "tileable",
    }), encoding="utf-8")

    with pytest.raises(IntakeError, match="explicit"):
        texture_intake.intake_texture("mystery")
