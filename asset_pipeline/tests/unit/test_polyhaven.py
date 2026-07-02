import json

import pytest

import config
from pipeline import polyhaven

_FAKE_ASSETS = {
    "wooden_table_01": {
        "name": "Wooden Table 01",
        "categories": ["furniture"],
        "tags": ["table", "wood", "dining"],
        "authors": {"Jane Modeler": "all"},
        "download_count": 5000,
    },
    "marble_bust_01": {
        "name": "Marble Bust 01",
        "categories": ["decor"],
        "tags": ["statue", "marble"],
        "authors": {"Bob Sculptor": "all"},
        "download_count": 9000,
    },
    "side_table_tall": {
        "name": "Side Table Tall",
        "categories": ["furniture"],
        "tags": ["table"],
        "authors": {},
        "download_count": 100,
    },
}

_FAKE_FILES = {
    "gltf": {
        "1k": {
            "gltf": {
                "url": "https://dl.example/wooden_table_01.gltf",
                "size": 1234,
                "include": {
                    "textures/wood_diff_1k.png": {"url": "https://dl.example/wood_diff_1k.png"},
                    "wooden_table_01.bin": {"url": "https://dl.example/wooden_table_01.bin"},
                },
            }
        }
    },
    "blend": {"1k": {"blend": {"url": "https://dl.example/t.blend", "size": 99}}},
}


@pytest.fixture
def fake_api(monkeypatch):
    def fake_get_json(path):
        if path.startswith("/assets"):
            return _FAKE_ASSETS
        if path.startswith("/files/"):
            return _FAKE_FILES
        raise AssertionError(f"unexpected API path {path}")

    monkeypatch.setattr(polyhaven, "_get_json", fake_get_json)


def test_search_ranks_name_matches_above_tag_matches(fake_api):
    results = polyhaven.search_models("wooden table")

    ids = [r.asset_id for r in results]
    assert ids[0] == "wooden_table_01"  # both tokens in id/name
    assert "side_table_tall" in ids  # 'table' matches
    assert "marble_bust_01" not in ids  # no token matches


def test_search_results_carry_cc0_license_and_provenance_fields(fake_api):
    [top, *_] = polyhaven.search_models("wooden table")

    assert top.license == "CC0"
    assert top.authors == ["Jane Modeler"]
    assert top.thumbnail_url  # always populated for the human's visual check
    assert top.download_count == 5000


def test_search_respects_limit(fake_api):
    assert len(polyhaven.search_models("table", limit=1)) == 1


def test_list_model_files_flattens_formats_and_counts_includes(fake_api):
    files = polyhaven.list_model_files("wooden_table_01")

    by_format = {f.file_format: f for f in files}
    assert by_format["gltf"].include_count == 2
    assert by_format["gltf"].url.endswith(".gltf")
    assert by_format["blend"].include_count == 0


def test_fetch_downloads_main_and_includes_and_prefills_source_json(
    fake_api, tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")
    downloads = []

    def fake_download(url, dest):
        downloads.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")

    monkeypatch.setattr(polyhaven, "_download_file", fake_download)

    source_path, downloaded = polyhaven.fetch_to_intake("wooden_table_01", target_size_m=0.8)

    intake_dir = tmp_path / "intake" / "wooden_table_01"
    assert (intake_dir / "wooden_table_01.gltf").exists()
    assert (intake_dir / "textures" / "wood_diff_1k.png").exists()
    assert (intake_dir / "wooden_table_01.bin").exists()
    assert len(downloads) == 3

    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert source["license"] == "CC0"
    assert source["source_site"] == "polyhaven.com"
    assert source["source_url"] == "https://polyhaven.com/a/wooden_table_01"
    assert source["original_author"] == "Jane Modeler"
    assert source["target_size_m"] == 0.8
    assert "FILL IN" not in source["notes"]


def _fake_download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"x")


def test_fetch_without_size_flags_the_missing_field(fake_api, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")
    monkeypatch.setattr(polyhaven, "_download_file", _fake_download)

    source_path, _ = polyhaven.fetch_to_intake("wooden_table_01")

    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert source["target_size_m"] is None
    assert "FILL IN target_size_m" in source["notes"]


def test_fetch_unknown_id_is_rejected(fake_api, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")
    with pytest.raises(polyhaven.PolyHavenSearchError, match="not a Poly Haven model id"):
        polyhaven.fetch_to_intake("no_such_model")


def test_fetch_refuses_nonempty_intake_folder(fake_api, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")
    existing = tmp_path / "intake" / "wooden_table_01"
    existing.mkdir(parents=True)
    (existing / "pending.glb").write_bytes(b"x")

    with pytest.raises(polyhaven.PolyHavenSearchError, match="not empty"):
        polyhaven.fetch_to_intake("wooden_table_01")


def test_fetch_rejects_path_escaping_include(fake_api, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTAKE_DIR", tmp_path / "intake")
    monkeypatch.setattr(polyhaven, "_download_file", _fake_download)
    evil_files = {
        "gltf": {
            "1k": {
                "gltf": {
                    "url": "https://dl.example/m.gltf",
                    "include": {"..\\..\\evil.png": {"url": "https://dl.example/evil.png"}},
                }
            }
        }
    }
    monkeypatch.setattr(
        polyhaven,
        "_get_json",
        lambda path: _FAKE_ASSETS if path.startswith("/assets") else evil_files,
    )

    with pytest.raises(polyhaven.PolyHavenSearchError, match="escapes intake dir"):
        polyhaven.fetch_to_intake("wooden_table_01")
