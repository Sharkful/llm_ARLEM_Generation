"""Integration tier: real headless Chromium render of catalog assets via
webapp/asset_preview.html. Self-skips when playwright/chromium/network is
unavailable, per the Stage 1.1 testing design.

Exit test for Stage 8: seed primitives render to recognizable, non-blank
thumbnails (non-blank asserted here via pixel variance; "recognizable" is
the human check on the contact sheet).
"""
import numpy as np
import pytest
from PIL import Image

import config
from pipeline.catalog_writer import load_catalog
from pipeline.preview_renderer import PreviewError, preview_path, render_previews

pytest.importorskip("playwright.sync_api", reason="playwright not installed")


def _render_or_skip(entries, **kwargs):
    try:
        return render_previews(entries, **kwargs)
    except PreviewError as exc:
        pytest.skip(f"preview rendering unavailable on this machine: {exc}")


def test_seed_primitives_render_nonblank_previews():
    catalog = load_catalog()
    wanted = ["sphere_basic", "cube_basic", "torus_basic", "arrow_basic"]
    entries = [e for e in catalog if e.asset_id in wanted]
    assert entries, "seed catalog missing -- run `cli.py catalog seed`"

    written, _ = _render_or_skip(entries, force=True)

    assert len(written) == len(entries)
    for path in written:
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
        assert img.shape[0] >= 256
        # A blank/failed render is one flat color; a real thumbnail has the
        # object against the background -> meaningful pixel variance.
        assert img.std() > 5.0, f"{path.name} looks blank (std={img.std():.2f})"


def test_imported_glb_renders_nonblank_preview():
    catalog = load_catalog()
    entries = [e for e in catalog if e.asset_id == "small_wooden_table_01"]
    if not entries:
        pytest.skip("small_wooden_table_01 not in catalog on this machine")

    written, _ = _render_or_skip(entries, force=True)

    [path] = written
    img = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    assert img.std() > 5.0, "GLB preview looks blank"
