"""Integration tier: live, read-only Poly Haven API calls (no downloads).

Skips itself if the network/API is unreachable, per the Stage 1.1 design.
The Stage 6b exit test from the implementation plan: search 'wooden table'
returns real results with the license field populated.
"""
import pytest

from pipeline import polyhaven


def _live_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except polyhaven.PolyHavenSearchError as exc:
        pytest.skip(f"Poly Haven API unreachable from this machine: {exc}")


def test_search_wooden_table_returns_real_cc0_results():
    results = _live_or_skip(polyhaven.search_models, "wooden table")

    assert results, "expected at least one Poly Haven model matching 'wooden table'"
    for r in results:
        assert r.license == "CC0"
        assert r.asset_id
        assert r.thumbnail_url.startswith("https://")


def test_top_result_has_downloadable_gltf():
    results = _live_or_skip(polyhaven.search_models, "wooden table", limit=1)
    assert results
    files = _live_or_skip(polyhaven.list_model_files, results[0].asset_id)

    gltf_options = [f for f in files if f.file_format == "gltf"]
    assert gltf_options, f"no gltf downloads listed for {results[0].asset_id!r}"
    assert all(f.url.startswith("https://") for f in gltf_options)
