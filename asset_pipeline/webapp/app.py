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
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from pipeline.asset_factory import (
    AssetFactoryError,
    adopt_draft_candidate,
    approve_asset,
    create_from_description,
    discard_draft,
    edit_composite,
    get_composite_fragment,
    load_draft_state,
    redirect_draft,
    regenerate,
    review_and_repair,
    save_draft,
    update_catalog_metadata,
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


# ── Background jobs: long pipeline calls (LLM + OpenSCAD + Blender) run in
#    a thread; the UI polls /api/job/<id> and shows each progress line live
#    ("checking local library for possible matches...") ───────────────────

_JOBS: dict[str, dict] = {}


def _start_job(fn) -> str:
    """fn(progress) -> (draft, llm_call_entries). Returns the job id."""
    job_id = uuid.uuid4().hex[:12]
    job = {"events": [], "done": False, "error": None, "draft": None, "llm_calls": []}
    _JOBS[job_id] = job

    def target():
        try:
            draft, logs = fn(lambda message: job["events"].append(message))
            job["draft"] = draft.model_dump(mode="json")
            job["llm_calls"] = [entry.model_dump(mode="json") for entry in logs]
        except Exception as exc:  # surfaced to the UI, never a silent thread death
            job["error"] = str(exc)
        job["done"] = True

    threading.Thread(target=target, daemon=True).start()
    return job_id


@app.get("/api/job/<job_id>")
def api_job(job_id: str):
    job = _JOBS.get(job_id)
    if job is None:
        return _error(f"unknown job {job_id!r}", 404)
    return jsonify(job)


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
    job_id = _start_job(lambda progress: create_from_description(
        description,
        asset_id=(payload.get("asset_id") or "").strip() or None,
        provider=payload.get("provider"),
        model=payload.get("model"),
        progress=progress,
    ))
    return jsonify({"job_id": job_id}), 202


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
    tweak = (payload.get("tweak") or "").strip() or None
    job_id = _start_job(lambda progress: regenerate(
        asset_id,
        parameters=parameters,
        tweak=tweak,
        provider=payload.get("provider"),
        model=payload.get("model"),
        progress=progress,
    ))
    return jsonify({"job_id": job_id}), 202


@app.get("/api/draft/<asset_id>")
def api_draft(asset_id: str):
    draft = load_draft_state(asset_id)
    if draft is None:
        return _error(f"no draft for {asset_id!r}", 404)
    return jsonify({"draft": draft.model_dump(mode="json")})


@app.get("/api/drafts")
def api_drafts():
    """All persisted drafts -- scene-object bindings (moon = sphere + moon
    texture), works in progress, and items awaiting human sourcing. These
    are not catalog entries, so the Library grid alone would hide them."""
    drafts = []
    for draft_file in sorted((config.LIBRARY_DIR / "generated").glob("*/draft.json")):
        draft = load_draft_state(draft_file.parent.name)
        if draft is not None:
            drafts.append(draft.model_dump(mode="json"))
    drafts.sort(key=lambda d: d.get("updated_at") or "", reverse=True)
    return jsonify({"drafts": drafts})


@app.post("/api/adopt")
def api_adopt():
    """One-click approval of a sourcing candidate: download + intake + resolve."""
    payload = request.get_json(silent=True) or {}
    asset_id = (payload.get("asset_id") or "").strip()
    source_id = (payload.get("source_id") or "").strip()
    if not asset_id or not source_id:
        return _error("asset_id and source_id are required")
    try:
        size = float(payload.get("size") or 0)
    except (TypeError, ValueError):
        return _error("size must be a number (meters)")
    if size <= 0:
        return _error("size (real-world largest dimension, meters) is required")
    def run(progress):
        progress(f"Downloading {source_id!r} and running the intake gate...")
        draft = adopt_draft_candidate(asset_id, source_id, target_size_m=size)
        progress("Intake complete -- cataloged with provenance.")
        return draft, []

    return jsonify({"job_id": _start_job(run)}), 202


@app.post("/api/redirect")
def api_redirect():
    """Manual override: force a stuck draft down a specific path
    (parametric/composite/imported), bypassing the original classification."""
    payload = request.get_json(silent=True) or {}
    asset_id = (payload.get("asset_id") or "").strip()
    target = (payload.get("target") or "").strip()
    if not asset_id or target not in ("parametric", "composite", "imported"):
        return _error("asset_id and target ('parametric'|'composite'|'imported') are required")
    manual_query = (payload.get("manual_query") or "").strip() or None
    job_id = _start_job(lambda progress: redirect_draft(
        asset_id, target, manual_query=manual_query,
        provider=payload.get("provider"), model=payload.get("model"), progress=progress,
    ))
    return jsonify({"job_id": job_id}), 202


@app.get("/api/composite/<asset_id>")
def api_composite_get(asset_id: str):
    """Current parts list of a composite draft, for the editor panel.
    Each part gets its resolved base_color inlined (fragment.json only
    stores a material_id) so the editor can show a real color swatch
    instead of guessing from color_hint text."""
    try:
        fragment = get_composite_fragment(asset_id)
    except AssetFactoryError as exc:
        return _error(str(exc), 404)
    data = fragment.model_dump(mode="json")
    for part in data["parts"]:
        part["current_base_color"] = None
        if part.get("material_id"):
            mat_path = config.MATERIALS_DIR / f"{part['material_id']}.json"
            if mat_path.is_file():
                part["current_base_color"] = json.loads(
                    mat_path.read_text(encoding="utf-8-sig")
                ).get("base_color")
    return jsonify({"fragment": data})


@app.post("/api/composite/<asset_id>/edit")
def api_composite_edit(asset_id: str):
    """Edit part transforms/colors/bonds directly and rebake -- no LLM call
    unless regenerate_materials_for is given (see edit_composite docstring)."""
    payload = request.get_json(silent=True) or {}
    part_edits = payload.get("part_edits") or {}
    if not isinstance(part_edits, dict):
        return _error("part_edits must be an object of {part_id: {field: value}}")
    regenerate_materials_for = payload.get("regenerate_materials_for") or None
    job_id = _start_job(lambda progress: edit_composite(
        asset_id, part_edits=part_edits,
        regenerate_materials_for=regenerate_materials_for,
        provider=payload.get("provider"), model=payload.get("model"),
        progress=progress,
    ))
    return jsonify({"job_id": job_id}), 202


@app.post("/api/review/<asset_id>")
def api_review(asset_id: str):
    """Render the draft, ask a vision LLM whether it matches the original
    description, and (auto_repair, default True) feed a real mismatch back
    into a repair pass. The verdict is embedded in the returned draft
    (draft.visual_review) either way."""
    payload = request.get_json(silent=True) or {}
    auto_repair = payload.get("auto_repair", True)

    def run(progress):
        draft, review, logs = review_and_repair(
            asset_id, auto_repair=auto_repair,
            provider=payload.get("provider"), model=payload.get("model"),
            progress=progress,
        )
        return draft, logs

    return jsonify({"job_id": _start_job(run)}), 202


@app.post("/api/draft/<asset_id>/discard")
def api_discard_draft(asset_id: str):
    """Delete an in-progress draft and everything it produced -- the
    'start over' escape hatch for a stuck/wrong/stale draft. Refuses a
    draft that's already saved/approved (see discard_draft's docstring)."""
    try:
        discard_draft(asset_id)
    except AssetFactoryError as exc:
        return _error(str(exc), 400)
    return jsonify({"ok": True})


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


@app.post("/api/catalog/<asset_id>/metadata")
def api_catalog_metadata(asset_id: str):
    """Rename/re-tag a catalog entry so it's easier to find later --
    metadata only, never geometry/provenance/scale."""
    payload = request.get_json(silent=True) or {}
    display_name = payload.get("display_name")
    tags = payload.get("tags")
    if tags is not None and not isinstance(tags, list):
        return _error("tags must be a list of strings")
    try:
        entry = update_catalog_metadata(asset_id, display_name=display_name, tags=tags)
    except AssetFactoryError as exc:
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
