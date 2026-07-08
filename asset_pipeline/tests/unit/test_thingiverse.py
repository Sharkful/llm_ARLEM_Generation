import json

import pytest

import config
from pipeline import thingiverse

_FAKE_SEARCH = {
    "hits": [
        {"id": 111, "name": "Gear (search summary)", "thumbnail": "thumb111.png"},
        {"id": 222, "name": "Wall Bracket NC (search summary)", "thumbnail": "thumb222.png"},
    ]
}

_FAKE_THINGS = {
    "111": {
        "name": "Involute Gear",
        "license": "Creative Commons - Public Domain Dedication",
        "creator": {"name": "Alice"},
        "public_url": "https://www.thingiverse.com/thing:111",
        "download_count": 500,
        "tags": [{"name": "gear"}, {"name": "mechanical"}],
    },
    "222": {
        "name": "Wall Bracket",
        "license": "Creative Commons - Attribution - Non Commercial",
        "creator": {"name": "Bob"},
        "public_url": "https://www.thingiverse.com/thing:222",
        "download_count": 10,
        "tags": [{"name": "bracket"}],
    },
}

_FAKE_FILES = {
    "111": [
        {"name": "gear.stl", "download_url": "https://cdn.example/gear.stl"},
        {"name": "gear.3mf", "download_url": "https://cdn.example/gear.3mf"},
    ],
    "222": [{"name": "bracket.stl", "download_url": "https://cdn.example/bracket.stl"}],
}


@pytest.fixture
def fake_token(monkeypatch):
    monkeypatch.setattr(config, "THINGIVERSE_APP_TOKEN", "fake-token")


@pytest.fixture
def fake_api(monkeypatch, fake_token):
    def fake_get_json(path, **params):
        if path.startswith("/search/"):
            return _FAKE_SEARCH
        if path.endswith("/files/"):
            thing_id = path.split("/")[2]
            return _FAKE_FILES[thing_id]
        if path.startswith("/things/"):
            thing_id = path.split("/")[2]
            return _FAKE_THINGS[thing_id]
        raise AssertionError(f"unexpected API path {path}")

    monkeypatch.setattr(thingiverse, "_get_json", fake_get_json)


def _fake_download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"stl-bytes")


# ── search ──────────────────────────────────────────────────────────────

def test_search_ranks_name_matches_and_carries_raw_unmapped_license(fake_api):
    results, hidden = thingiverse.search_thingiverse("gear mechanical")

    assert hidden == 0  # no license filter given
    ids = [r.thing_id for r in results]
    assert ids[0] == "111"
    top = results[0]
    # Raw Thingiverse label, deliberately NOT mapped at search time (see
    # module docstring) -- mapping only happens at fetch time.
    assert top.license == "Creative Commons - Public Domain Dedication"
    assert top.creator == "Alice"
    assert top.download_count == 500
    assert top.public_url == "https://www.thingiverse.com/thing:111"


def test_search_uses_thing_name_not_search_summary_name(fake_api):
    [top, *_], _hidden = thingiverse.search_thingiverse("gear mechanical")
    assert top.name == "Involute Gear"  # from /things/{id}/, not the search hit


def test_search_license_filter_hides_and_counts_out_of_set_hits(fake_api):
    """allowed_licenses filtering happens inside the search (against the
    authoritative per-thing license, mapped) so a deeper hit pool can be
    walked until `limit` in-tier results are found."""
    results, hidden = thingiverse.search_thingiverse(
        "gear bracket", allowed_licenses={"cc0", "cc-by"},
    )

    assert [r.thing_id for r in results] == ["111"]  # PD-dedication gear kept
    assert hidden == 1  # the NC bracket was examined and hidden


def test_search_stops_fetching_details_once_limit_reached(fake_api, monkeypatch):
    """Early exit: with limit=1 and the first hit qualifying, the second
    hit's detail record must never be fetched."""
    fetched = []
    real_get_json = thingiverse._get_json

    def counting_get_json(path, **params):
        fetched.append(path)
        return real_get_json(path, **params)

    monkeypatch.setattr(thingiverse, "_get_json", counting_get_json)

    results, hidden = thingiverse.search_thingiverse(
        "gear bracket", limit=1, allowed_licenses={"cc0"},
    )

    assert [r.thing_id for r in results] == ["111"]
    assert hidden == 0
    assert "/things/222/" not in fetched


def test_search_rejects_empty_query(fake_api):
    with pytest.raises(thingiverse.ThingiverseError, match="Empty search query"):
        thingiverse.search_thingiverse("   ")


def test_search_without_token_raises_clear_error(monkeypatch):
    monkeypatch.setattr(config, "THINGIVERSE_APP_TOKEN", None)
    with pytest.raises(thingiverse.ThingiverseError, match="THINGIVERSE_APP_TOKEN"):
        thingiverse.search_thingiverse("gear")


# ── license mapping (fail-closed) ────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("Creative Commons - Public Domain Dedication", "cc0"),
    ("  Public Domain  ", "cc0"),
    ("Creative Commons - Attribution", "cc-by"),
    ("Creative Commons - Attribution - Share Alike", "cc-by-sa"),
])
def test_map_license_known_labels(raw, expected):
    assert thingiverse.map_license(raw) == expected


@pytest.mark.parametrize("raw", [
    "Creative Commons - Attribution - Non Commercial",
    "All Rights Reserved",
    "GNU - GPL",
    "",
])
def test_map_license_unmapped_labels_pass_through_unchanged(raw):
    """Fail-closed: an unrecognized label must come back verbatim so
    external_intake.py's allowlist gate rejects it -- never coerced into
    something permissive."""
    assert thingiverse.map_license(raw) == raw


# ── fetch ─────────────────────────────────────────────────────────────────

def test_fetch_downloads_stl_and_prefills_cc0_source_json(fake_api, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")
    monkeypatch.setattr(thingiverse, "_download_file", _fake_download)

    intake_dir = thingiverse.fetch_thingiverse_to_intake("111", target_size_m=0.05)

    assert (intake_dir / "gear.stl").exists()
    source = json.loads((intake_dir / "source.json").read_text(encoding="utf-8"))
    assert source["license"] == "cc0"
    assert source["original_author"] == "Alice"
    assert source["source_site"] == "thingiverse.com"
    assert source["target_size_m"] == 0.05
    assert source["tags"] == ["gear", "mechanical"]
    assert "FILL IN" not in source["notes"]
    assert "NOTE" not in source["notes"]


def test_fetch_prefers_stl_over_3mf(fake_api, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")
    monkeypatch.setattr(thingiverse, "_download_file", _fake_download)

    intake_dir = thingiverse.fetch_thingiverse_to_intake("111", target_size_m=0.05)

    assert (intake_dir / "gear.stl").exists()
    assert not (intake_dir / "gear.3mf").exists()


def test_fetch_restrictive_license_is_not_laundered_and_is_flagged(fake_api, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")
    monkeypatch.setattr(thingiverse, "_download_file", _fake_download)

    intake_dir = thingiverse.fetch_thingiverse_to_intake("222", target_size_m=0.1)

    source = json.loads((intake_dir / "source.json").read_text(encoding="utf-8"))
    # Verbatim, unmapped -- external_intake._LICENSE_ALLOWLIST will reject it.
    assert source["license"] == "Creative Commons - Attribution - Non Commercial"
    assert "does not map to an allowlisted license" in source["notes"]


def test_fetch_without_size_flags_missing_field(fake_api, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")
    monkeypatch.setattr(thingiverse, "_download_file", _fake_download)

    intake_dir = thingiverse.fetch_thingiverse_to_intake("111")

    source = json.loads((intake_dir / "source.json").read_text(encoding="utf-8"))
    assert source["target_size_m"] is None
    assert "FILL IN target_size_m" in source["notes"]


def test_fetch_refuses_nonempty_intake_folder(fake_api, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")
    existing = tmp_path / "intake" / "thing_111"
    existing.mkdir(parents=True)
    (existing / "pending.stl").write_bytes(b"x")

    with pytest.raises(thingiverse.ThingiverseError, match="not empty"):
        thingiverse.fetch_thingiverse_to_intake("111")


def test_fetch_no_usable_mesh_file_raises(monkeypatch, tmp_path, fake_token):
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")

    def fake_get_json(path, **params):
        if path.endswith("/files/"):
            return [{"name": "model.gcode", "download_url": "https://cdn.example/model.gcode"}]
        return {"name": "Unprintable Thing", "license": "cc0"}

    monkeypatch.setattr(thingiverse, "_get_json", fake_get_json)

    with pytest.raises(thingiverse.ThingiverseError, match=r"No \.stl/\.obj/\.glb file"):
        thingiverse.fetch_thingiverse_to_intake("999")
