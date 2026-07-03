import json

import pytest

import config
from models.catalog_models import AssetCatalogEntry, ProvenanceInfo
from pipeline.catalog_writer import save_catalog
from webapp import app as webapp_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    monkeypatch.setattr(config, "LIBRARY_DIR", lib)
    monkeypatch.setattr(config, "CATALOG_PATH", lib / "catalog.json")
    (lib / "previews").mkdir(parents=True)
    monkeypatch.setattr(webapp_app, "WORKLIST_PATH", tmp_path / "worklist.json")
    webapp_app.app.config["TESTING"] = True
    return webapp_app.app.test_client()


def _seed_catalog():
    save_catalog([
        AssetCatalogEntry(
            asset_id="sphere_basic", display_name="Sphere", asset_class="primitive",
            address="Primitives/Sphere", canonical_bounds_m=[1, 1, 1],
            provenance=ProvenanceInfo(source_type="core"),
        )
    ])


def test_catalog_endpoint_includes_preview_url_when_png_exists(client):
    _seed_catalog()
    (config.LIBRARY_DIR / "previews" / "sphere_basic.png").write_bytes(b"png")

    data = client.get("/api/catalog").get_json()

    [asset] = data["assets"]
    assert asset["asset_id"] == "sphere_basic"
    assert asset["preview_url"] == "/library/previews/sphere_basic.png"


def test_generate_requires_description(client):
    res = client.post("/api/generate", json={})
    assert res.status_code == 400
    assert "description" in res.get_json()["error"]


def test_generate_reports_pipeline_errors_cleanly(client, monkeypatch):
    def boom(description, asset_id=None, provider=None, model=None):
        raise RuntimeError("no API key configured")

    monkeypatch.setattr(webapp_app, "create_from_description", boom)
    res = client.post("/api/generate", json={"description": "a bracket"})

    assert res.status_code == 500
    assert "no API key" in res.get_json()["error"]


def test_regenerate_validates_parameters_shape(client):
    res = client.post(
        "/api/regenerate",
        json={"asset_id": "x", "parameters": {"width": "not-a-number"}},
    )
    assert res.status_code == 400
    assert "numbers" in res.get_json()["error"]


def test_approve_unknown_asset_404s(client):
    _seed_catalog()
    res = client.post("/api/approve", json={"asset_id": "ghost"})
    assert res.status_code == 404


def test_approve_records_stamp(client):
    _seed_catalog()
    res = client.post("/api/approve", json={"asset_id": "sphere_basic"})

    assert res.status_code == 200
    entry = res.get_json()["entry"]
    assert any("approved by" in m for m in entry["provenance"]["modifications"])


def test_worklist_roundtrip_persists(client):
    payload = {
        "items": [{"description": "a bracket", "status": "pending"}],
        "cursor": 0,
    }
    assert client.post("/api/worklist", json=payload).status_code == 200

    data = client.get("/api/worklist").get_json()
    assert data["items"][0]["description"] == "a bracket"


def test_worklist_rejects_malformed_payload(client):
    assert client.post("/api/worklist", json={"nope": 1}).status_code == 400


def test_library_static_is_scoped(client):
    _seed_catalog()
    (config.LIBRARY_DIR / "previews" / "sphere_basic.png").write_bytes(b"png")

    ok = client.get("/library/previews/sphere_basic.png")
    assert ok.status_code == 200
    escape = client.get("/library/../cli.py")
    assert escape.status_code in (403, 404)
