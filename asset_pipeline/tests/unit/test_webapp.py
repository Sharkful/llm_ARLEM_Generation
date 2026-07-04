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


def _poll_job(client, job_id, timeout_s=5.0):
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = client.get(f"/api/job/{job_id}").get_json()
        if job["done"]:
            return job
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_generate_runs_as_job_with_progress_events(client, monkeypatch):
    from models.spec_models import AssetSpec
    from pipeline.asset_factory import DraftAsset

    def fake_create(description, asset_id=None, provider=None, model=None, progress=None):
        progress("Checking local library for possible matches...")
        progress("Matched existing catalog asset 'sphere_basic'...")
        return DraftAsset(
            asset_id="moon", status="existing", resolution_method="catalog_match",
            description=description, spec=AssetSpec(object_id="moon", description=description),
            matched_asset_id="sphere_basic",
        ), []

    monkeypatch.setattr(webapp_app, "create_from_description", fake_create)
    res = client.post("/api/generate", json={"description": "the moon"})

    assert res.status_code == 202
    job = _poll_job(client, res.get_json()["job_id"])
    assert job["error"] is None
    assert any("Checking local library" in e for e in job["events"])
    assert job["draft"]["matched_asset_id"] == "sphere_basic"


def test_generate_reports_pipeline_errors_cleanly(client, monkeypatch):
    def boom(description, asset_id=None, provider=None, model=None, progress=None):
        raise RuntimeError("no API key configured")

    monkeypatch.setattr(webapp_app, "create_from_description", boom)
    res = client.post("/api/generate", json={"description": "a bracket"})

    assert res.status_code == 202
    job = _poll_job(client, res.get_json()["job_id"])
    assert "no API key" in job["error"]


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


def test_drafts_endpoint_lists_persisted_drafts(client):
    from models.spec_models import AssetSpec
    from pipeline.asset_factory import DraftAsset, save_draft_state

    (config.LIBRARY_DIR / "generated").mkdir(parents=True, exist_ok=True)
    save_draft_state(DraftAsset(
        asset_id="moon", status="existing", resolution_method="catalog_match",
        description="the moon", spec=AssetSpec(object_id="moon", description="the moon"),
        matched_asset_id="sphere_basic",
    ))

    data = client.get("/api/drafts").get_json()

    [draft] = data["drafts"]
    assert draft["asset_id"] == "moon"
    assert draft["matched_asset_id"] == "sphere_basic"


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


def test_redirect_requires_valid_target(client):
    res = client.post("/api/redirect", json={"asset_id": "x", "target": "nonsense"})
    assert res.status_code == 400
    assert "target" in res.get_json()["error"]


def test_redirect_runs_as_job_and_forwards_manual_query(client, monkeypatch):
    from models.spec_models import AssetSpec
    from pipeline.asset_factory import DraftAsset

    captured = {}

    def fake_redirect(asset_id, target, manual_query=None, provider=None, model=None, progress=None):
        captured.update(asset_id=asset_id, target=target, manual_query=manual_query)
        progress("Searching allowlisted sources again with your keywords...")
        return DraftAsset(
            asset_id=asset_id, status="needs_human", resolution_method="imported",
            description="a snowman", spec=AssetSpec(object_id=asset_id, description="a snowman"),
        ), []

    monkeypatch.setattr(webapp_app, "redirect_draft", fake_redirect)
    res = client.post("/api/redirect", json={
        "asset_id": "frosty", "target": "imported", "manual_query": "snowman christmas figure",
    })

    assert res.status_code == 202
    job = _poll_job(client, res.get_json()["job_id"])
    assert job["error"] is None
    assert captured == {
        "asset_id": "frosty", "target": "imported", "manual_query": "snowman christmas figure",
    }
    assert any("Searching allowlisted sources again" in e for e in job["events"])
