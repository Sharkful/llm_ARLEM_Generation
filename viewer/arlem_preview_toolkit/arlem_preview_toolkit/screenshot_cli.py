#!/usr/bin/env python3
"""
ARLEM Screenshot CLI
====================
Capture screenshots of ARLEM lab scenes from the command line.
Drives viewer2.html headlessly via Playwright.

Usage examples:
  python screenshot_cli.py --list
  python screenshot_cli.py --module 1 --clip 5
  python screenshot_cli.py --module "Synchronous" --clips 0 8
  python screenshot_cli.py --module 1 --clips 0 8 --camera 0 5 0 0 0 0 --cleanup
  python screenshot_cli.py --json path/to/my_lab.json --module 0 --clip -1
  python screenshot_cli.py --cleanup-all
  python screenshot_cli.py --cleanup-all --out path/to/screenshots
"""

import argparse
import asyncio
import base64
import http.server
import json
import os
import re
import threading
from pathlib import Path

VIEWER_DIR = Path(__file__).parent
DEFAULT_JSON = VIEWER_DIR / "moon_lab_with_buttons.json"
DEFAULT_CAMERA_POS = [4.0, 3.0, 7.0]
DEFAULT_CAMERA_TGT = [0.0, 0.4, 1.5]


# ── Helpers ───────────────────────────────────────────────────────────

def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")


def resolve_module(lab: dict, spec: str) -> int:
    if spec.lstrip("-").isdigit():
        idx = int(spec)
        if idx < 0 or idx >= len(lab["modules"]):
            raise ValueError(f"Module index {idx} out of range (0–{len(lab['modules'])-1})")
        return idx
    spec_lower = spec.lower()
    for i, m in enumerate(lab["modules"]):
        if spec_lower in m.get("moduleName", "").lower():
            return i
    raise ValueError(f"No module name contains '{spec}'")


def build_filename(prefix: str, module_idx: int, module_name: str,
                   clip_idx: int, clip_name: str) -> str:
    m_slug = sanitize(module_name)
    if clip_idx < 0:
        c_slug = "base"
        c_num  = "base"
    else:
        c_slug = sanitize(clip_name) if clip_name else f"clip{clip_idx}"
        c_num  = f"c{clip_idx:03d}"
    parts = [p for p in [prefix, f"m{module_idx:02d}", m_slug, c_num, c_slug] if p]
    return "_".join(parts) + ".png"


def list_modules(lab: dict) -> None:
    print(f"{'Index':>5}  {'Clips':>5}  Module Name")
    print("-" * 50)
    for i, m in enumerate(lab["modules"]):
        clips = len(m.get("clips", []))
        print(f"{i:>5}  {clips:>5}  {m.get('moduleName', '(untitled)')}")


# ── HTTP server (needed so fetch() works for GLBs/textures) ──────────

def start_server(directory: Path) -> tuple[http.server.HTTPServer, int]:
    os.chdir(directory)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass  # suppress request logs

    httpd = http.server.HTTPServer(("localhost", 0), QuietHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port


# ── Core capture logic ────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    from playwright.async_api import async_playwright

    lab_path = Path(args.json).resolve()

    # --cleanup-all: wipe the screenshots directory and exit
    if args.cleanup_all:
        if args.out:
            out_dir = Path(args.out).resolve()
        else:
            out_dir = lab_path.parent / "screenshots"
        if not out_dir.exists():
            print(f"Directory does not exist: {out_dir}")
            return
        files = list(out_dir.glob("*.png"))
        for f in files:
            f.unlink()
        print(f"Deleted {len(files)} file(s) from {out_dir}")
        return

    lab = json.loads(lab_path.read_text(encoding="utf-8"))

    # --list: just print the module table and exit
    if args.list:
        list_modules(lab)
        return

    # Resolve module
    module_idx  = resolve_module(lab, str(args.module))
    module_info = lab["modules"][module_idx]
    module_name = module_info.get("moduleName", f"module{module_idx}")
    clips       = module_info.get("clips", [])

    # Build clip list
    if args.clips is not None:
        lo, hi = args.clips
        clip_indices = list(range(lo, hi + 1))
    elif args.clip is not None:
        clip_indices = [args.clip]
    else:
        clip_indices = [-1]  # base layout

    # Camera
    cam_args = args.camera or (DEFAULT_CAMERA_POS + DEFAULT_CAMERA_TGT)
    camera = {
        "position": cam_args[0:3],
        "target":   cam_args[3:6],
    }

    # Output directory
    if args.out:
        out_dir = Path(args.out).resolve()
    else:
        out_dir = lab_path.parent / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefix for filenames and cleanup scoping
    prefix = args.prefix or ""

    # --cleanup: delete only files matching this module's prefix pattern
    if args.cleanup:
        m_slug  = sanitize(module_name)
        pattern = f"*m{module_idx:02d}_{m_slug}_*.png"
        deleted = 0
        for f in out_dir.glob(pattern):
            f.unlink()
            deleted += 1
        if deleted:
            print(f"Cleaned {deleted} file(s) matching '{pattern}' in {out_dir}")

    # Start local HTTP server so fetch()/GLB loads work
    httpd, port = start_server(VIEWER_DIR)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--enable-webgl"])
        page = await browser.new_page(
            viewport={"width": args.width, "height": args.height}
        )

        await page.goto(f"http://localhost:{port}/viewer2.html")

        # Wait for the viewer JS module to finish initialising
        await page.wait_for_function("typeof window._arlemCapture === 'function'",
                                     timeout=15000)

        # If a custom JSON was supplied (not the viewer's default), inject it.
        # _setRawLab() is exposed by viewer2.html and calls populateModules/buildScene.
        default_json_name = DEFAULT_JSON.name
        if lab_path.name != default_json_name or lab_path.parent != VIEWER_DIR:
            await page.wait_for_function("typeof window._setRawLab === 'function'",
                                         timeout=10000)
            await page.evaluate("(lab) => window._setRawLab(lab)", lab)
        else:
            # Wait for the viewer's auto-fetched default JSON to load
            await page.wait_for_function("window.rawLab !== null", timeout=10000)

        print(f"Module {module_idx}: {module_name}  |  {len(clip_indices)} clip(s)  ->  {out_dir}")

        for clip_idx in clip_indices:
            clip_name = clips[clip_idx].get("clipName", f"clip{clip_idx}") \
                        if 0 <= clip_idx < len(clips) else "base"

            filename = build_filename(prefix, module_idx, module_name, clip_idx, clip_name)
            out_path = out_dir / filename

            b64 = await page.evaluate(
                "async (p) => window._arlemCapture(p)",
                {"moduleIndex": module_idx, "clipIndex": clip_idx, "camera": camera}
            )
            out_path.write_bytes(base64.b64decode(b64))
            print(f"  [{clip_idx:>4}] {filename}")

        await browser.close()

    httpd.shutdown()
    print("Done.")


# ── CLI argument parsing ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture screenshots of ARLEM lab scenes via headless Chromium.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--json", default=str(DEFAULT_JSON), metavar="PATH",
                   help="Lab JSON file path (default: moon_lab_with_buttons.json in viewer dir)")
    p.add_argument("--module", default="0", metavar="INT|STR",
                   help="Module index (0-based) or substring of module name")
    p.add_argument("--clip", type=int, default=None, metavar="INT",
                   help="Single clip index (-1 = base layout)")
    p.add_argument("--clips", type=int, nargs=2, metavar=("FROM", "TO"),
                   help="Inclusive clip range, e.g. --clips 0 5")
    p.add_argument("--camera", type=float, nargs=6,
                   metavar=("PX", "PY", "PZ", "TX", "TY", "TZ"),
                   help="Camera position (px py pz) then look-at target (tx ty tz)")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="Output directory (default: screenshots/ next to the JSON file)")
    p.add_argument("--prefix", default="", metavar="STR",
                   help="Filename prefix for all output images")
    p.add_argument("--width", type=int, default=1280, metavar="INT",
                   help="Viewport width in pixels (default: 1280)")
    p.add_argument("--height", type=int, default=720, metavar="INT",
                   help="Viewport height in pixels (default: 720)")
    p.add_argument("--cleanup", action="store_true",
                   help="Delete existing images for this module/prefix before capturing")
    p.add_argument("--cleanup-all", action="store_true",
                   help="Delete all PNG files in the screenshots directory, then exit")
    p.add_argument("--list", action="store_true",
                   help="Print all modules and clip counts, then exit (no screenshots)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
