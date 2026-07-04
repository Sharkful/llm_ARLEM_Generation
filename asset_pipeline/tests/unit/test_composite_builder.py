from types import SimpleNamespace

import pytest

from models.spec_models import AssetSpec
from pipeline import composite_builder
from pipeline.composite_builder import (
    CompositeFragment,
    CompositePart,
    _ForcedDecomposition,
    build_part_specs,
    composite_path,
    decompose_into_primitives,
    load_composite_fragment,
    save_composite_fragment,
)


def test_build_part_specs_creates_one_spec_per_description():
    specs = build_part_specs("water_molecule", ["1 red sphere (oxygen)", "2 white spheres (hydrogen)"])
    assert len(specs) == 2
    assert specs[0].object_id == "water_molecule_part0"
    assert specs[0].description == "1 red sphere (oxygen)"
    assert specs[1].object_id == "water_molecule_part1"


def test_build_part_specs_hints_primitive_kind_for_catalog_matching():
    """Regression: composite sub-parts must get a semantic_type hint so they
    catalog-match the seed primitives without an extra LLM classify() call
    per part -- found the hard way with a 12-part snowman composite where
    every 'large white sphere'-style part needed its own classification."""
    specs = build_part_specs("snowman", [
        "large white sphere (base)", "2 white spheres (arms attachment)",
        "thin dark brown cylinder (stick arm)", "long orange cone (carrot nose)",
        "a vaguely snowman-shaped blob with no clear primitive",
    ])
    assert specs[0].kind == "sphere" and specs[0].semantic_type == "sphere"
    assert specs[1].kind == "sphere"  # plural "spheres" still matches
    assert specs[2].kind == "cylinder"
    assert specs[3].kind == "cone"
    assert specs[4].kind is None and specs[4].semantic_type is None


def test_build_part_specs_recognizes_primitive_aliases():
    """A flat disk (e.g. a hat brim) has no dedicated primitive -- it's a
    squashed cylinder; found the hard way when this was the one part of a
    13-part snowman composite that failed to catalog-match."""
    specs = build_part_specs("snowman", ["flat black disk (top hat brim)"])
    assert specs[0].kind == "cylinder"


def test_save_and_load_composite_fragment_round_trips(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path)

    fragment = CompositeFragment(
        composite_id="water_molecule_composite",
        display_name="Water Molecule",
        source_description="a water molecule model",
        parts=[
            CompositePart(part_id="p0", asset_spec=build_part_specs("water_molecule", ["oxygen"])[0]),
        ],
    )
    path = save_composite_fragment(fragment)
    assert path == composite_path("water_molecule_composite")
    assert path.exists()

    loaded = load_composite_fragment("water_molecule_composite")
    assert loaded == fragment


def test_load_composite_fragment_returns_none_when_missing(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path)
    assert load_composite_fragment("does_not_exist") is None


class _FakeCompletions:
    def __init__(self, response):
        self._response = response
        self.last_call_kwargs = None

    def create_with_completion(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._response, SimpleNamespace(usage=None)


class _FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=_FakeCompletions(response))


def test_decompose_into_primitives_returns_parts_and_log(monkeypatch):
    fake = _FakeClient(_ForcedDecomposition(parts=[
        "large white sphere", "medium white sphere", "small white sphere",
        "thin black cylinder", "small black cone",
    ]))
    monkeypatch.setattr(composite_builder, "get_instructor_client", lambda provider: fake)

    spec = AssetSpec(object_id="snowman", description="a friendly snowman")
    parts, entry = decompose_into_primitives(spec, max_parts=12)

    assert len(parts) == 5
    assert "large white sphere" in parts
    assert entry.purpose == "composite_decomposition"
    assert entry.success is True
    kwargs = fake.chat.completions.last_call_kwargs
    assert kwargs["response_model"] is _ForcedDecomposition


def test_decompose_into_primitives_truncates_to_max_parts(monkeypatch):
    fake = _FakeClient(_ForcedDecomposition(parts=[f"part {i}" for i in range(20)]))
    monkeypatch.setattr(composite_builder, "get_instructor_client", lambda provider: fake)

    spec = AssetSpec(object_id="x", description="x")
    parts, entry = decompose_into_primitives(spec, max_parts=5)

    assert len(parts) == 5


def test_decompose_into_primitives_wraps_failure(monkeypatch):
    class _AlwaysFails:
        chat = SimpleNamespace(completions=SimpleNamespace(
            create_with_completion=lambda **k: (_ for _ in ()).throw(RuntimeError("boom"))
        ))

    monkeypatch.setattr(composite_builder, "get_instructor_client", lambda provider: _AlwaysFails())

    spec = AssetSpec(object_id="x", description="x")
    with pytest.raises(RuntimeError, match="Forced composite decomposition failed"):
        decompose_into_primitives(spec)
