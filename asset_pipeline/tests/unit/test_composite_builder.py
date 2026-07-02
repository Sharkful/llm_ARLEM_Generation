from pipeline.composite_builder import (
    CompositeFragment,
    CompositePart,
    build_part_specs,
    composite_path,
    load_composite_fragment,
    save_composite_fragment,
)


def test_build_part_specs_creates_one_spec_per_description():
    specs = build_part_specs("water_molecule", ["1 red sphere (oxygen)", "2 white spheres (hydrogen)"])
    assert len(specs) == 2
    assert specs[0].object_id == "water_molecule_part0"
    assert specs[0].description == "1 red sphere (oxygen)"
    assert specs[1].object_id == "water_molecule_part1"


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
