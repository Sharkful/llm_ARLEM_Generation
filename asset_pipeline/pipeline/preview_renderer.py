"""Stage 8: batch static preview thumbnails for catalog assets.

Reuses the Playwright + three.js pattern proven by
viewer/arlem_preview_toolkit/arlem_preview_toolkit/screenshot_cli.py --
same local http.server + headless Chromium + capture-hook approach --
against webapp/asset_preview.html, a single-asset companion page to
viewer2.html (plan Stage 8 option (b): viewer2.html is module/clip-shaped,
so a dedicated one-object page is less code than teaching it a special
mode). asset_preview.html doubles as the Stage 8b interactive viewport, so
this stays the project's one rendering path.

Requires `playwright` + its chromium browser (playwright install chromium)
and network access for the three.js CDN the page imports (same CDN
viewer2.html already uses). preview() raises PreviewError with a clear
message when either is missing; the CLI reports it rather than tracebacking.
"""
from __future__ import annotations

import base64
import http.server
import threading
from html import escape
from pathlib import Path

import config
from models.catalog_models import AssetCatalogEntry

_PAGE_ROUTE = "/webapp/asset_preview.html"


class PreviewError(RuntimeError):
    pass


def spec_for_entry(entry: AssetCatalogEntry) -> dict:
    """Map a catalog entry to an asset_preview.html load spec."""
    if entry.address.startswith("library/"):
        return {"kind": "glb", "url": f"/{entry.address}"}
    # Unity-style addresses ("Primitives/Sphere", "Composites/Arrow"):
    # the page builds an equivalent three.js primitive scaled to bounds.
    name = entry.address.rsplit("/", 1)[-1].lower()
    return {"kind": "primitive", "name": name, "bounds": list(entry.canonical_bounds_m)}


def preview_path(asset_id: str) -> Path:
    return config.LIBRARY_DIR / "previews" / f"{asset_id}.png"


def render_glb_to_png(glb_address: str, out_path: Path, size: int = 512) -> Path:
    """One-off render of an arbitrary library/... GLB that isn't (yet) a
    catalog entry -- e.g. a draft mid-iteration, for the visual review step
    (pipeline/visual_review.py) to grab a fresh screenshot of exactly what
    the human/reviewer would currently see, without requiring a save first.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise PreviewError(
            "playwright is not installed. Run: pip install playwright && playwright install chromium"
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    httpd, port = _start_server(config.PACKAGE_DIR)
    try:
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(args=["--enable-webgl"])
            except Exception as exc:
                raise PreviewError(
                    f"Chromium failed to launch ({exc}). Run: playwright install chromium"
                ) from exc
            page = browser.new_page(viewport={"width": size, "height": size})
            page.goto(f"http://127.0.0.1:{port}{_PAGE_ROUTE}")
            try:
                page.wait_for_function("window._previewReady === true", timeout=20000)
            except Exception as exc:
                raise PreviewError(
                    "asset_preview.html never initialized -- no network access to the "
                    f"three.js CDN? ({exc})"
                ) from exc
            try:
                page.evaluate(
                    "(spec) => window._previewLoad(spec)", {"kind": "glb", "url": f"/{glb_address}"}
                )
                b64 = page.evaluate("() => window._previewCapture()")
            except Exception as exc:
                raise PreviewError(f"Preview failed for {glb_address!r}: {exc}") from exc
            out_path.write_bytes(base64.b64decode(b64))
            browser.close()
    finally:
        httpd.shutdown()
    return out_path


def _start_server(root: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, *_):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def render_previews(
    entries: list[AssetCatalogEntry],
    force: bool = False,
    size: int = 512,
) -> tuple[list[Path], list[str]]:
    """Render one PNG per entry into library/previews/.

    Returns (written_paths, skipped_asset_ids). Skips assets whose preview
    already exists unless `force` (matching prefab_to_glb.py's --force
    convention). One browser/page serves the whole batch -- the page's
    _previewLoad() hot-swaps assets, so per-asset cost is load+render only.
    """
    to_render = [e for e in entries if force or not preview_path(e.asset_id).exists()]
    skipped = [e.asset_id for e in entries if e not in to_render]
    if not to_render:
        return [], skipped

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise PreviewError(
            "playwright is not installed. Run: pip install playwright && playwright install chromium"
        ) from exc

    (config.LIBRARY_DIR / "previews").mkdir(parents=True, exist_ok=True)
    # Serve the package dir so both /webapp/... and /library/... resolve.
    # Note: GLB previews assume the default in-package LIBRARY_DIR; with an
    # ASSET_PIPELINE_LIBRARY_DIR override pointing elsewhere, primitive
    # previews still work but /library/ GLB fetches would 404 -- acceptable
    # for a local dev tool, revisit if an external library becomes the norm.
    httpd, port = _start_server(config.PACKAGE_DIR)

    written: list[Path] = []
    try:
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(args=["--enable-webgl"])
            except Exception as exc:
                raise PreviewError(
                    f"Chromium failed to launch ({exc}). Run: playwright install chromium"
                ) from exc
            page = browser.new_page(viewport={"width": size, "height": size})
            page.goto(f"http://127.0.0.1:{port}{_PAGE_ROUTE}")
            try:
                page.wait_for_function("window._previewReady === true", timeout=20000)
            except Exception as exc:
                raise PreviewError(
                    "asset_preview.html never initialized -- no network access to the "
                    f"three.js CDN? ({exc})"
                ) from exc

            for entry in to_render:
                spec = spec_for_entry(entry)
                try:
                    page.evaluate("(spec) => window._previewLoad(spec)", spec)
                    b64 = page.evaluate("() => window._previewCapture()")
                except Exception as exc:
                    raise PreviewError(
                        f"Preview failed for {entry.asset_id!r}: {exc}"
                    ) from exc
                out = preview_path(entry.asset_id)
                out.write_bytes(base64.b64decode(b64))
                written.append(out)
            browser.close()
    finally:
        httpd.shutdown()
    return written, skipped


def write_contact_sheet(entries: list[AssetCatalogEntry]) -> Path:
    """Static contact-sheet HTML over library/previews/ (req. doc section 13)."""
    previews_dir = config.LIBRARY_DIR / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for entry in sorted(entries, key=lambda e: e.asset_id):
        png = preview_path(entry.asset_id)
        img = (
            f'<img src="{escape(entry.asset_id)}.png" alt="{escape(entry.asset_id)}">'
            if png.exists()
            else '<div class="missing">no preview</div>'
        )
        bounds = " × ".join(f"{v:g}" for v in entry.canonical_bounds_m)
        cards.append(
            f'<figure>{img}<figcaption><b>{escape(entry.asset_id)}</b><br>'
            f'{escape(entry.asset_class)} · review {entry.review_level}<br>'
            f'{bounds} m</figcaption></figure>'
        )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Asset Catalog Contact Sheet</title>
<style>
  body {{ font-family: sans-serif; background: #f4f5f7; margin: 24px; }}
  h1 {{ font-size: 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }}
  figure {{ background: #fff; border-radius: 8px; padding: 10px; margin: 0;
            box-shadow: 0 1px 3px rgba(0,0,0,.15); text-align: center; }}
  img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: #e8eaed; border-radius: 4px; }}
  .missing {{ width: 100%; aspect-ratio: 1; display: flex; align-items: center;
              justify-content: center; background: #eee; color: #999; border-radius: 4px; }}
  figcaption {{ font-size: 12px; color: #444; margin-top: 6px; line-height: 1.5; }}
</style></head><body>
<h1>Asset Catalog — {len(entries)} asset(s)</h1>
<div class="grid">
{chr(10).join(cards)}
</div></body></html>
"""
    sheet = previews_dir / "contact_sheet.html"
    sheet.write_text(html, encoding="utf-8")
    return sheet
