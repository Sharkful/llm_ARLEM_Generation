from pipeline.catalog_writer import save_catalog
from pipeline.seed_catalog import build_seed_entries


def test_seed_entries_are_internally_unique_and_nonempty(tmp_path):
    entries = build_seed_entries()
    assert len(entries) >= 8
    save_catalog(entries, tmp_path / "catalog.json")  # raises on duplicate asset_id


def test_every_seed_entry_has_approved_provenance():
    for entry in build_seed_entries():
        assert entry.provenance.license_status == "approved"


def test_every_seed_entry_has_nonzero_bounds():
    for entry in build_seed_entries():
        # plane's Y bound is intentionally 0 (a flat ground plane); everything
        # else should have all three dimensions set.
        nonzero = [v for v in entry.canonical_bounds_m if v > 0]
        assert len(nonzero) >= 2, entry.asset_id
