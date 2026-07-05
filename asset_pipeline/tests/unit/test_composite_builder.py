from types import SimpleNamespace

import pytest

from models.classification_models import CompositePartPlan
from models.spec_models import AssetSpec
from pipeline import composite_builder
from pipeline.composite_builder import (
    CompositeFragment,
    CompositePart,
    _CompositeRevision,
    _ForcedDecomposition,
    build_part_specs,
    composite_path,
    decompose_into_primitives,
    fragment_to_part_plans,
    load_composite_fragment,
    revise_composite_parts,
    save_composite_fragment,
)


def _part(description, color="gray", position=(0.0, 0.0, 0.0), scale=(0.3, 0.3, 0.3)):
    return CompositePartPlan(
        description=description, color_hint=color,
        relative_position=list(position), relative_scale=list(scale),
    )


def test_build_part_specs_creates_one_spec_per_description():
    specs = build_part_specs("water_molecule", [
        _part("1 red sphere (oxygen)", "red"), _part("2 white spheres (hydrogen)", "white"),
    ])
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
        _part("large white sphere (base)"), _part("2 white spheres (arms attachment)"),
        _part("thin dark brown cylinder (stick arm)"), _part("long orange cone (carrot nose)"),
        _part("a vaguely snowman-shaped blob with no clear primitive"),
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
    specs = build_part_specs("snowman", [_part("flat black disk (top hat brim)")])
    assert specs[0].kind == "cylinder"


def test_save_and_load_composite_fragment_round_trips(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path)

    fragment = CompositeFragment(
        composite_id="water_molecule_composite",
        display_name="Water Molecule",
        source_description="a water molecule model",
        parts=[
            CompositePart(
                part_id="p0", asset_spec=build_part_specs("water_molecule", [_part("oxygen", "red")])[0],
                color_hint="red",
            ),
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
        _part("large white sphere", "white"), _part("medium white sphere", "white"),
        _part("small white sphere", "white"), _part("thin black cylinder", "black"),
        _part("small black cone", "black"),
    ]))
    monkeypatch.setattr(composite_builder, "get_instructor_client", lambda provider: fake)

    spec = AssetSpec(object_id="snowman", description="a friendly snowman")
    parts, entry = decompose_into_primitives(spec, max_parts=12)

    assert len(parts) == 5
    assert any(p.description == "large white sphere" for p in parts)
    assert entry.purpose == "composite_decomposition"
    assert entry.success is True
    kwargs = fake.chat.completions.last_call_kwargs
    assert kwargs["response_model"] is _ForcedDecomposition


def test_decompose_into_primitives_truncates_to_max_parts(monkeypatch):
    fake = _FakeClient(_ForcedDecomposition(parts=[_part(f"part {i}") for i in range(20)]))
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


# ── revise_composite_parts (added 2026-07, visual-review auto-repair) ─────

def test_fragment_to_part_plans_carries_current_geometry_and_bonds():
    fragment = CompositeFragment(
        composite_id="molecule_composite", display_name="Molecule", source_description="x",
        parts=[
            CompositePart(
                part_id="C", asset_spec=AssetSpec(object_id="C", description="carbon sphere"),
                resolved_asset_id="sphere_basic", label="C", position=[0.0, 0.0, 0.0],
                scale=[0.3, 0.3, 0.3], color_hint="dark gray",
            ),
            CompositePart(
                part_id="bond1", asset_spec=AssetSpec(object_id="bond1", description="bond"),
                bond_between=["C", "H1"], bond_thickness=0.07, color_hint="white",
            ),
        ],
    )

    plans = fragment_to_part_plans(fragment)

    assert len(plans) == 2
    assert plans[0].description == "carbon sphere"
    assert plans[0].relative_position == [0.0, 0.0, 0.0]
    assert plans[0].relative_scale == [0.3, 0.3, 0.3]
    assert plans[0].label == "C"
    assert plans[1].bond_between == ["C", "H1"]
    assert plans[1].bond_thickness == 0.07


def test_revise_composite_parts_sends_current_state_and_feedback(monkeypatch):
    current = [
        _part("carbon sphere", "dark gray"),
        CompositePartPlan(description="bond", color_hint="white", label="", bond_between=["C", "H1"], bond_thickness=0.06),
    ]
    fake = _FakeClient(_CompositeRevision(parts=[
        _part("carbon sphere", "dark gray"),
        CompositePartPlan(description="bond", color_hint="white", label="", bond_between=["C", "H1"], bond_thickness=0.09),
    ]))
    monkeypatch.setattr(composite_builder, "get_instructor_client", lambda provider: fake)

    spec = AssetSpec(object_id="methane", description="a methane molecule")
    revised, entry = revise_composite_parts(
        spec, current, feedback="the bond is too thin to see clearly", max_parts=12,
    )

    assert len(revised) == 2
    assert revised[1].bond_thickness == 0.09
    assert entry.purpose == "composite_revision"
    assert entry.success is True
    kwargs = fake.chat.completions.last_call_kwargs
    assert kwargs["response_model"] is _CompositeRevision
    user_content = kwargs["messages"][1]["content"]
    assert "too thin to see clearly" in user_content
    assert "carbon sphere" in user_content  # current state included, not just feedback


def test_revise_composite_parts_wraps_failure(monkeypatch):
    class _AlwaysFails:
        chat = SimpleNamespace(completions=SimpleNamespace(
            create_with_completion=lambda **k: (_ for _ in ()).throw(RuntimeError("boom"))
        ))

    monkeypatch.setattr(composite_builder, "get_instructor_client", lambda provider: _AlwaysFails())

    spec = AssetSpec(object_id="x", description="x")
    with pytest.raises(RuntimeError, match="Composite revision failed"):
        revise_composite_parts(spec, [_part("a sphere")], feedback="wrong")
