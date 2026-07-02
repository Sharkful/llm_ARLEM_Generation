"""Integration tier: one real (cheap, small) material-generation LLM call
per provider, parametrized so whichever provider has no key on this machine
self-skips -- the multi-provider seam gets exercised wherever keys exist,
per the implementation plan's Stage 1.1 testing design.
"""
import pytest

import config
from models.catalog_models import MaterialDef
from pipeline.material_generator import generate_material

_PROVIDER_KEYS = {
    "anthropic": config.ANTHROPIC_API_KEY,
    "openai": config.OPENAI_API_KEY,
    "google": config.GEMINI_API_KEY,
}


@pytest.mark.parametrize("provider", list(_PROVIDER_KEYS))
def test_generate_material_real_call(provider, tmp_path, monkeypatch):
    if not _PROVIDER_KEYS[provider]:
        pytest.skip(f"no API key for provider {provider!r} on this machine")

    monkeypatch.setattr(config, "MATERIALS_DIR", tmp_path / "materials")
    monkeypatch.setattr(config, "TEXTURES_DIR", tmp_path / "materials" / "textures")

    result, plan, entry = generate_material(
        "rough red rust with visible surface texture", provider=provider
    )

    assert result.success is True
    assert entry.success is True
    written = MaterialDef.model_validate_json(
        (tmp_path / "materials" / f"{result.material_id}.json").read_text(encoding="utf-8")
    )
    # A rust description should read as a rough, non-glowing surface with a
    # visible pattern -- the texture is the point of this exit-test example.
    assert written.emissive is False
    assert plan.procedural_texture is not None
    assert result.texture_path is not None
