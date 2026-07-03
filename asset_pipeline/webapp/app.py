"""Stage 8b: local Flask review/creation app.

Thin HTTP layer over pipeline/asset_factory.py -- no pipeline logic lives
here. Serves:
  - the single-page UI (index.html) with its Library / Create / Worklist modes
  - webapp/asset_preview.html as the live three.js viewport (same page Stage
    8's batch renderer drives headlessly)
  - library/ as scoped static files so the browser can fetch GLBs/previews

Local-only by design (127.0.0.1): the regenerate endpoint shells out to
OpenSCAD/Blender and must never be reachable beyond localhost.

Worklist state persists to webapp/worklist.json so a production authoring
session (load a list, step through, iterate, save each) survives restarts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from pipeline.asset_factory import (
    AssetFactoryError,
    approve_asset,
    create_from_description,
    load_draft_state,
    regenerate,
    save_draft,
)
from pipeline.catalog_writer import load_catalog
from pipeline.mesh_processor import MeshProcessingError
from pipeline.openscad_generator import OpenSCADGenerationError
from pipeline.spec_parser import SpecParsingError

WEBAPP_DIR = Path(__file__).parent
WORKLIST_PATH = WEBAPP_DIR / "worklist.json"

app = Flask(__name__)


def _error(message: str, code: int = 400):
    return jsonify({"error": message}), code


# ── Static: UI, viewport, library files ───────────────────────────────────

@app.get("/")
def index():
    return send_from_directory(WEBAPP_DIR, "index.html")


@app.get("/webapp/<path:name>")
def webapp_static(name: str):
    return send_from_directory(WEBAPP_DIR, name)


@app.get("/library/<path:name>")
def library_static(name: str):
    # Scoped to library/ only -- never the whole repo.
    return send_from_directory(config.LIBRARY_DIR, name)


# ── API ───────────────────────────────────────────────────────────────────

@app.get("/api/catalog")
def api_catalog():
    entries = []
    for e in load_catalog():
        d = e.model_dump(mode="json")
        d["preview_url"] = (
            f"/library/previews/{e.asset_id}.png"
            if (config.LIBRARY_DIR / "previews" / f"{e.asset_id}.png").exists()
            else None
        )
        entries.append(d)
    return jsonify({"assets": entries})


@app.post("/api/generate")
def api_generate():
    payload = request.get_json(silent=True) or {}
    description = (payload.get("description") or "").strip()
    if not description:
        return _error("description is required")
    try:
        draft, logs = create_from_description(
            description,
            asset_id=(payload.get("asset_id") or "").strip() or None,
            provider=payload.get("provider"),
            model=payload.get("model"),
        )
    except (SpecParsingError, OpenSCADGenerationError, MeshProcessingError,
            AssetFactoryError, RuntimeError) as exc:
        return _error(str(exc), 500)
    return jsonify({
        "draft": draft.model_dump(mode="json"),
        "llm_calls": [entry.model_dump(mode="json") for entry in logs],
    })


@app.post("/api/regenerate")
def api_regenerate():
    payload = request.get_json(silent=True) or {}
    asset_id = (payload.get("asset_id") or "").strip()
    if not asset_id:
        return _error("asset_id is required")
    parameters = payload.get("parameters") or None
    if parameters is not None:
        try:
            parameters = {str(k): float(v) for k, v in parameters.items()}
        except (TypeError, ValueError):
            return _error("parameters must map names to numbers")
    try:
        draft, logs = regenerate(
            asset_id,
            parameters=parameters,
            tweak=(payload.get("tweak") or "").strip() or None,
            provider=payload.get("provider"),
            model=payload.get("model"),
        )
    except (OpenSCADGenerationError, MeshProcessingError, AssetFactoryError,
            RuntimeError) as exc:
        return _error(str(exc), 500)
    return jsonify({
        "draft": draft.model_dump(mode="json"),
        "llm_calls": [entry.model_dump(mode="json") for entry in logs],
    })


@app.get("/api/draft/<asset_id>")
def api_draft(asset_id: str):
    draft = load_draft_state(asset_id)
    if draft is None:
        return _error(f"no draft for {asset_id!r}", 404)
    return jsonify({"draft": draft.model_dump(mode="json")})


@app.post("/api/save")
def api_save():
    payload = request.get_json(silent=True) or {}
    asset_id = (payload.get("asset_id") or "").strip()
    if not asset_id:
        return _error("asset_id is required")
    try:
        entry = save_draft(
            asset_id,
            display_name=(payload.get("display_name") or "").strip() or None,
            tags=payload.get("tags"),
        )
    except (AssetFactoryError, ValueError) as exc:
        return _error(str(exc), 400)
    return jsonify({"entry": entry.model_dump(mode="json")})


@app.post("/api/approve")
def api_approve():
    payload = request.get_json(silent=True) or {}
    asset_id = (payload.get("asset_id") or "").strip()
    if not asset_id:
        return _error("asset_id is required")
    try:
        entry = approve_asset(asset_id)
    except AssetFactoryError as exc:
        return _error(str(exc), 404)
    return jsonify({"entry": entry.model_dump(mode="json")})


# ── Worklist persistence (client drives, server stores) ──────────────────

@app.get("/api/worklist")
def api_worklist_get():
    if WORKLIST_PATH.is_file():
        return jsonify(json.loads(WORKLIST_PATH.read_text(encoding="utf-8-sig")))
    return jsonify({"items": [], "cursor": 0})


@app.post("/api/worklist")
def api_worklist_set():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return _error("expected {items: [...], cursor: int}")
    WORKLIST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return jsonify(payload)


def main(port: int = 5173, open_browser: bool = True) -> None:
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
    # 127.0.0.1 only -- see module docstring.
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
