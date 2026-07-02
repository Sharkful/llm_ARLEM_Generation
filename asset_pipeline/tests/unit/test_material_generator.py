from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import config
from models.catalog_models import MaterialDef
from models.generation_models import MaterialPlan, ProceduralTextureSpec
from pipeline import material_generator


# ── Texture synthesis (pure numpy/PIL, no LLM) ───────────────────────────

ALL_KINDS = ["noise", "craters", "stripes", "grid", "gradient", "rust"]


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_synthesize_texture_writes_rgb_png_of_configured_size(kind, tmp_path):
    spec = ProceduralTextureSpec(kind=kind, base_color="#8a8a8a", accent_color="#3a2a1a")
    out = material_generator.synthesize_texture(spec, tmp_path / f"{kind}.png", size=64)

    img = Image.open(out)
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_synthesize_texture_is_deterministic_per_seed(tmp_path):
    spec = ProceduralTextureSpec(kind="rust", base_color="#7a7a7a", accent_color="#8b3a1a")
    a = material_generator.synthesize_texture(spec, tmp_path / "a.png", size=64, seed=7)
    b = material_generator.synthesize_texture(spec, tmp_path / "b.png", size=64, seed=7)
    c = material_generator.synthesize_texture(spec, tmp_path / "c.png", size=64, seed=8)

    a_px, b_px, c_px = (np.asarray(Image.open(p)) for p in (a, b, c))
    assert np.array_equal(a_px, b_px)
    assert not np.array_equal(a_px, c_px)


def test_synthesize_texture_uses_both_colors(tmp_path):
    spec = ProceduralTextureSpec(kind="stripes", base_color="#ff0000", accent_color="#0000ff", scale=4)
    out = material_generator.synthesize_texture(spec, tmp_path / "s.png", size=64)

    pixels = np.asarray(Image.open(out)).reshape(-1, 3)
    assert any(np.array_equal(p, [255, 0, 0]) for p in pixels)
    assert any(np.array_equal(p, [0, 0, 255]) for p in pixels)


def test_hex_to_rgb_rejects_garbage():
    with pytest.raises(ValueError):
        material_generator._hex_to_rgb("not-a-color")


# ── generate_material with a mocked instructor client ────────────────────

class _FakeUsage:
    def __init__(self, input_tokens=10, output_tokens=5, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _FakeCompletions:
    def __init__(self, response, completion):
        self._response = response
        self._completion = completion
        self.last_call_kwargs = None

    def create_with_completion(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._response, self._completion


class _FakeClient:
    def __init__(self, response, completion):
        self.chat = SimpleNamespace(completions=_FakeCompletions(response, completion))


def _plan(material_id="dull_gray_rock", texture=None):
    return MaterialPlan(
        material=MaterialDef(material_id=material_id, base_color="#8a8a8a", smoothness=0.1),
        procedural_texture=texture,
        reasoning="test",
    )


@pytest.fixture
def materials_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MATERIALS_DIR", tmp_path / "materials")
    monkeypatch.setattr(config, "TEXTURES_DIR", tmp_path / "materials" / "textures")
    return tmp_path / "materials"


def _patch_client(monkeypatch, plan):
    fake = _FakeClient(plan, SimpleNamespace(usage=_FakeUsage(input_tokens=42, output_tokens=17)))
    monkeypatch.setattr(material_generator, "get_instructor_client", lambda provider: fake)
    return fake


def test_generate_material_writes_json_without_texture(materials_in_tmp, monkeypatch):
    _patch_client(monkeypatch, _plan())

    result, plan, entry = material_generator.generate_material(
        "dull gray rock", provider="anthropic", model="claude-sonnet-4-6"
    )

    assert result.success is True
    assert result.texture_path is None
    written = MaterialDef.model_validate_json(
        (materials_in_tmp / "dull_gray_rock.json").read_text(encoding="utf-8")
    )
    assert written.base_color == "#8a8a8a"
    assert written.texture is None
    assert entry.purpose == "material_generation"
    assert entry.input_tokens == 42


def test_generate_material_synthesizes_texture_and_links_relative_path(
    materials_in_tmp, monkeypatch
):
    texture = ProceduralTextureSpec(kind="rust", base_color="#7a7a7a", accent_color="#8b3a1a")
    _patch_client(monkeypatch, _plan(material_id="rusty_panel", texture=texture))

    result, plan, entry = material_generator.generate_material(
        "rough red rust", provider="anthropic", model="claude-sonnet-4-6"
    )

    assert result.success is True
    assert (materials_in_tmp / "textures" / "rusty_panel.png").exists()
    written = MaterialDef.model_validate_json(
        (materials_in_tmp / "rusty_panel.json").read_text(encoding="utf-8")
    )
    assert written.texture == "textures/rusty_panel.png"


def test_generate_material_id_override_wins(materials_in_tmp, monkeypatch):
    _patch_client(monkeypatch, _plan(material_id="llm_proposed_id"))

    result, _, _ = material_generator.generate_material(
        "dull gray rock", material_id="my_id", provider="anthropic", model="claude-sonnet-4-6"
    )

    assert result.material_id == "my_id"
    assert (materials_in_tmp / "my_id.json").exists()
    assert not (materials_in_tmp / "llm_proposed_id.json").exists()


def test_generate_material_refuses_overwrite_without_force(materials_in_tmp, monkeypatch):
    _patch_client(monkeypatch, _plan())
    material_generator.generate_material("dull gray rock", provider="anthropic", model="m")

    result, _, _ = material_generator.generate_material(
        "dull gray rock", provider="anthropic", model="m"
    )
    assert result.success is False
    assert "already exists" in result.error_message

    result, _, _ = material_generator.generate_material(
        "dull gray rock", provider="anthropic", model="m", force=True
    )
    assert result.success is True


def test_generate_material_wraps_failure_with_log_entry(materials_in_tmp, monkeypatch):
    # Patch the retry-wrapped call directly so this test doesn't burn through
    # tenacity's real exponential backoff (5 attempts, up to 60s each).
    def _always_raise(client, llm_model, description):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        material_generator, "get_instructor_client", lambda provider: _FakeClient(None, None)
    )
    monkeypatch.setattr(material_generator, "_call_with_usage", _always_raise)

    with pytest.raises(material_generator.MaterialGenerationError) as exc_info:
        material_generator.generate_material("anything", provider="anthropic", model="m")

    assert exc_info.value.log_entry.success is False
    assert "boom" in exc_info.value.log_entry.error_message


def test_generate_material_sends_material_plan_response_model(materials_in_tmp, monkeypatch):
    fake = _patch_client(monkeypatch, _plan())

    material_generator.generate_material(
        "brushed metal", provider="openai", model="gpt-4o-mini"
    )

    kwargs = fake.chat.completions.last_call_kwargs
    assert kwargs["response_model"] is MaterialPlan
    assert kwargs["model"] == "gpt-4o-mini"
    assert any(
        "brushed metal" in m["content"] for m in kwargs["messages"] if m["role"] == "user"
    )
