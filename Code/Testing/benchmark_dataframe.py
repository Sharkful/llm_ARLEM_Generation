"""
Load every benchmark run into one tidy pandas DataFrame.

Each ``*_metrics.json`` in ``Artifacts/Data/Benchmark/`` is one run. This module
flattens the nested ``tracking`` (tokens/cost/timing/retries) and ``lab_metrics``
(modules/clips/objects/components/prefabs) blocks into a single flat row, adds a
handful of derived ratios, and returns them as a DataFrame you can slice by model,
provider, or specificity level.

Usage:
    from benchmark_dataframe import load_runs
    df = load_runs()                       # L1-L4 formative runs only
    df = load_runs(levels_only=False)      # include legacy --topic runs (level=None)

Run directly for a quick sanity dump:
    python "Code/Testing/benchmark_dataframe.py"
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmark_config import MODELS
from lab_metrics import analyze_assets

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "Artifacts" / "Data" / "Benchmark"

# model_id (as recorded in the metrics file) -> rough capability/cost tier.
SIZE_BY_MODEL = {cfg.model_id: cfg.size.value for cfg in MODELS.values()}

# Per-run ``*_metrics.json`` files are written only for successful runs; failures
# survive only in the per-sweep ``suite_results_*.json`` arrays (a clean superset:
# every success in a suite also has a standalone metrics file). To include failed
# runs, union both sources and dedupe.

LEVELS = ["L1", "L2", "L3", "L4"]
PROVIDERS = ["openai", "anthropic", "google"]


def fail_mode(err: str | None) -> str:
    """Classify a terminal error string into a coarse failure mode.

    Kept identical to ``build_summary_report.fail_mode`` so both reports agree.
    """
    if not err:
        return ""
    e = err.lower()
    if "incomplete" in e or "max_tokens length" in e:
        return "truncation (max_tokens)"
    # Check schema validation before the 404 signature: discriminated-union
    # failures contain "union_tag_not_found", whose "not_found" must NOT be read
    # as a model-not-found 404. Use the specific 404 signature, not a substring.
    if "validation error" in e:
        return "schema validation"
    if "not_found_error" in e or "error code: 404" in e:
        return "404 model-not-found"
    return "other"


def _ratio(numer, denom):
    """Safe division: NaN when the denominator is missing or zero."""
    try:
        if not denom:
            return float("nan")
        return numer / denom
    except (TypeError, ZeroDivisionError):
        return float("nan")


def flatten_record(rec: dict) -> dict:
    """Flatten one metrics record into a single flat row of scalars."""
    t = rec.get("tracking") or {}
    toks = t.get("tokens") or {}
    timing = t.get("timing") or {}
    retries = t.get("retries") or {}
    errs = t.get("errors") or {}
    lm = rec.get("lab_metrics") or {}
    am = rec.get("asset_metrics") or {}

    prompt_tokens = toks.get("prompt", 0) or 0
    completion_tokens = toks.get("completion", 0) or 0
    total_tokens = toks.get("total", 0) or 0
    cost_usd = (t.get("cost") or {}).get("total_usd", 0.0) or 0.0

    # Gemini bills the response-schema tokens as input but excludes them from the
    # reported usage; prefer the adjusted figures when the runner recorded them.
    # (see memory: project_gemini_schema_token_billing)
    adj_prompt = rec.get("prompt_tokens_adjusted")
    adj_cost = rec.get("cost_adjusted_usd")
    effective_prompt = adj_prompt if adj_prompt is not None else prompt_tokens
    effective_total = (
        effective_prompt + completion_tokens
        if adj_prompt is not None
        else total_tokens
    )
    effective_cost = adj_cost if adj_cost is not None else cost_usd

    duration_ms = timing.get("total_duration_ms", 0.0) or 0.0
    wall_s = rec.get("wall_time_seconds", 0.0) or 0.0

    num_modules = lm.get("num_modules", 0) or 0
    num_objects = lm.get("num_objects", 0) or 0
    num_clips = lm.get("num_clips", 0) or 0
    num_components = lm.get("num_components", 0) or 0
    num_object_changes = lm.get("num_object_changes", 0) or 0
    num_unique_objects = len(lm.get("unique_object_names") or [])

    # Novel-asset counts come from re-reading the saved output JSON (see
    # load_runs); absent for failed runs and L1 outlines, which stay all-zero.
    novel_textures = am.get("novel_textures", 0) or 0
    novel_prefabs = am.get("novel_prefabs", 0) or 0
    audio_refs = am.get("audio_refs", 0) or 0

    return {
        # ── identity ──────────────────────────────────────────────
        "model": rec.get("model"),
        "display_name": rec.get("display_name", rec.get("model")),
        "provider": rec.get("provider"),
        "size": SIZE_BY_MODEL.get(rec.get("model")),
        "level": rec.get("level"),
        "structure": rec.get("structure"),
        "lab_name": rec.get("lab_name"),
        "spec_type": rec.get("spec_type"),
        "timestamp": rec.get("timestamp"),
        "success": bool(rec.get("success")),
        "fail_mode": fail_mode(rec.get("error")),
        # had_usage splits failures that burned tokens/time (truncation, schema
        # retries) from no-op failures (404 / config errors) that never reached
        # the model and so carry no meaningful cost/token/time stats.
        "had_usage": total_tokens > 0,
        # ── cost / tokens / timing ────────────────────────────────
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "effective_prompt_tokens": effective_prompt,
        "effective_total_tokens": effective_total,
        "effective_cost_usd": effective_cost,
        "duration_ms": duration_ms,
        "wall_s": wall_s,
        "retries": retries.get("total", 0) or 0,
        "parse_errors": errs.get("parse_errors", 0) or 0,
        "api_errors": errs.get("api_errors", 0) or 0,
        # ── structural output (0 for failed runs and L1 outlines) ──
        "num_objectives": lm.get("num_objectives", 0) or 0,
        "num_modules": num_modules,
        "num_objects": num_objects,
        "num_clips": num_clips,
        "num_components": num_components,
        "num_object_changes": num_object_changes,
        "num_text_labels": lm.get("num_text_labels", 0) or 0,
        "num_educational_objectives": lm.get("num_educational_objectives", 0) or 0,
        "num_unique_prefabs": lm.get("num_unique_prefabs", 0) or 0,
        "num_unique_objects": num_unique_objects,
        # ── novel assets (0 for failed runs and L1 outlines) ──────
        # Distinct assets the model invented (not in the moon-lab library) and
        # that we'd have to author; *_refs count object-instances using one.
        "novel_textures": novel_textures,
        "novel_prefabs": novel_prefabs,
        "novel_texture_refs": am.get("novel_texture_refs", 0) or 0,
        "novel_prefab_refs": am.get("novel_prefab_refs", 0) or 0,
        "texture_refs": am.get("texture_refs", 0) or 0,
        "prefab_refs": am.get("prefab_refs", 0) or 0,
        # Audio has no known library — volume/density, not novelty.
        "audio_refs": audio_refs,
        "audio_unique": am.get("audio_unique", 0) or 0,
        # ── derived ratios ────────────────────────────────────────
        "clips_per_module": _ratio(num_clips, num_modules),
        "objects_per_module": _ratio(num_objects, num_modules),
        "components_per_object": _ratio(num_components, num_objects),
        "changes_per_clip": _ratio(num_object_changes, num_clips),
        "tokens_per_sec": _ratio(completion_tokens, duration_ms / 1000.0),
        # Novel assets normalized: prefabs/textures attach to objects (per-object
        # and per-module are the honest denominators); audio attaches to clips.
        "novel_prefabs_per_module": _ratio(novel_prefabs, num_modules),
        "novel_textures_per_module": _ratio(novel_textures, num_modules),
        "novel_prefabs_per_object": _ratio(novel_prefabs, num_objects),
        "novel_textures_per_object": _ratio(novel_textures, num_objects),
        "audio_per_clip": _ratio(audio_refs, num_clips),
    }


def load_runs(
    benchmark_dir: Path = BENCH,
    levels_only: bool = True,
    include_failures: bool = False,
) -> pd.DataFrame:
    """Flatten benchmark runs into one DataFrame.

    Args:
        benchmark_dir: directory holding the run JSON.
        levels_only: keep only runs whose internal ``level`` is L1-L4 (the
            structured formative sweep); drops legacy ``--topic`` runs.
        include_failures: also pull failed runs from ``suite_results_*.json``.
            Default reads only the success-only ``*_metrics.json`` files. Use the
            ``success`` / ``had_usage`` columns to slice the result.
    """
    raw = []
    out_dir = benchmark_dir / "Outputs"
    for p in sorted((benchmark_dir / "Metrics").glob("*_metrics.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        # Pair the run with its saved output JSON (same stem, _output suffix) and
        # compute novel-asset counts — no LLM calls, just re-reading what we kept.
        out_path = out_dir / (p.name[: -len("_metrics.json")] + "_output.json")
        if out_path.exists():
            rec["asset_metrics"] = analyze_assets(
                json.loads(out_path.read_text(encoding="utf-8"))
            )
        raw.append(rec)
    if include_failures:
        for p in sorted(benchmark_dir.glob("suite_results_*.json")):
            raw.extend(json.loads(p.read_text(encoding="utf-8")))

    df = pd.DataFrame(flatten_record(r) for r in raw)
    if df.empty:
        return df

    # Defensive dedupe: one row per unique run (suite successes also have a
    # standalone metrics file, so the union overlaps).
    df = df.drop_duplicates(
        subset=["model", "level", "structure", "lab_name", "timestamp"]
    ).reset_index(drop=True)

    if levels_only:
        df = df[df["level"].isin(LEVELS)].reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_runs(include_failures=True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(f"rows: {df.shape[0]}  cols: {df.shape[1]}")
    print(f"levels present: {sorted(df['level'].dropna().unique())}")
    print(f"providers: {sorted(df['provider'].dropna().unique())}")
    print(f"models: {df['model'].nunique()}  labs: {sorted(df['lab_name'].dropna().unique())}")
    print(f"success: {int(df['success'].sum())}/{len(df)}")
    fails = df[~df["success"]]
    print(f"failures: {len(fails)}  (billable={int(fails['had_usage'].sum())}, "
          f"no-op={int((~fails['had_usage']).sum())})")
    print("fail modes:")
    print(fails["fail_mode"].value_counts().to_string())
    print("\ncolumns:")
    print(list(df.columns))
