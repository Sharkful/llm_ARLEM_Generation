"""
Recompute per-run ``cost_usd`` against the current price table, non-destructively.

Each run's cost is baked into its ``*_metrics.json`` at generation time from the
``DEFAULT_PRICING`` entry in effect *then*. When a provider changes a published
rate after the fact, every downstream artifact keeps quoting the stale number and
there is no way to notice — cost is stored, never derived.

This script re-derives cost from the token counts a run already recorded
(``prompt_tokens``/``completion_tokens`` x ``tracking.pricing.DEFAULT_PRICING``)
and writes a *new* CSV alongside the original. Nothing is overwritten: the input
CSV, and the ``*_metrics.json`` files behind it, are read-only here. The output is
a strict superset of the input — the recomputed figure lands in ``cost_usd`` and
the original is preserved in ``cost_usd_recorded``.

It works off the run CSV rather than ``benchmark_dataframe.load_runs()`` on
purpose: the per-run ``*_metrics.json`` are gitignored, so for every sweep older
than the current one the CSV export is the only surviving record. Repricing has to
work on what survives.

Integrity check: models whose rate did not change must reproduce their recorded
cost to the cent. Any mismatch means the stored cost and the stored tokens
disagree for a reason unrelated to this reprice, so the script reports it and
exits non-zero rather than quietly publishing a number it can't account for.

Usage:
    python "Code/Analysis/reprice_runs.py" \
        "Artifacts/Data/Benchmark/Reports/benchmark_runs_wave2.csv"

    # custom destination / dry run
    python "Code/Analysis/reprice_runs.py" <csv> --out <path>
    python "Code/Analysis/reprice_runs.py" <csv> --dry-run
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.pricing import DEFAULT_PRICING  # noqa: E402

# A recorded cost and a recomputed one may differ in the last float bit purely
# from summation order, so "unchanged" is a tolerance, not equality. Half a cent
# per run is far below any figure we report and far above float noise.
UNCHANGED_TOL_USD = 0.005


def cost_for(model: str, prompt_tokens: float, completion_tokens: float) -> float:
    price = DEFAULT_PRICING[model]
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


def reprice(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(repriced_df, per_model_delta_report)``.

    ``df`` is expected to be the *raw text* of the source CSV (every column read
    as ``str``). Parsing a float column and re-serializing it can shift the
    decimal text by one unit in the last place — harmless numerically, but it
    would mean the output is not a byte-exact superset of the input, and the
    whole point here is that nothing except cost moves. Keeping untouched
    columns as the strings they already were sidesteps that entirely.

    Raises ``KeyError`` if any model in the CSV has no pricing entry — silently
    pricing an unknown model at zero would understate a total with no warning.
    """
    unknown = sorted(set(df["model"]) - set(DEFAULT_PRICING))
    if unknown:
        raise KeyError(
            f"no DEFAULT_PRICING entry for: {unknown}. Add them to "
            "tracking/pricing.py before repricing — pricing them at 0 would "
            "silently understate the totals."
        )

    prompt = pd.to_numeric(df["prompt_tokens"])
    completion = pd.to_numeric(df["completion_tokens"])

    out = df.copy()
    # Preserve the recorded figure as the exact text the source carried.
    out["cost_usd_recorded"] = df["cost_usd"]
    out["cost_usd"] = [
        cost_for(m, p, c) for m, p, c in zip(df["model"], prompt, completion)
    ]

    report = (
        pd.DataFrame({
            "model": df["model"],
            "display_name": df["display_name"],
            "recorded": pd.to_numeric(df["cost_usd"]),
            "repriced": out["cost_usd"],
        })
        .groupby(["model", "display_name"], as_index=False)
        .agg(
            runs=("recorded", "size"),
            recorded_usd=("recorded", "sum"),
            repriced_usd=("repriced", "sum"),
        )
    )
    report["delta_usd"] = report["repriced_usd"] - report["recorded_usd"]
    report["delta_pct"] = (
        report["delta_usd"] / report["recorded_usd"].where(report["recorded_usd"] != 0)
    ) * 100
    return out, report.sort_values("delta_usd")


def check_unchanged_models(report: pd.DataFrame) -> list[str]:
    """Models that shifted by less than a rounding tolerance but not exactly zero.

    Returns the list of models whose reprice moved them by more than
    ``UNCHANGED_TOL_USD`` *per run* — the caller decides whether that was
    expected. A model whose published rate is unchanged should land at ~0.
    """
    per_run = (report["delta_usd"].abs() / report["runs"]).fillna(0)
    return report.loc[per_run > UNCHANGED_TOL_USD, "model"].tolist()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs_csv", type=Path, help="Run-level CSV to reprice (read-only).")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: <input stem>_repriced.csv beside the input).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the delta report without writing anything.",
    )
    args = ap.parse_args(argv)

    src = args.runs_csv if args.runs_csv.is_absolute() else PROJECT_ROOT / args.runs_csv
    if not src.exists():
        print(f"error: no such CSV: {src}", file=sys.stderr)
        return 2

    # Every column as raw text: only cost is recomputed, so nothing else should
    # be re-serialized (see reprice()). keep_default_na keeps blanks blank
    # rather than turning them into the string "nan" on the way back out.
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    if "cost_usd_recorded" in df.columns:
        print(
            f"error: {src.name} already carries a 'cost_usd_recorded' column — it "
            "looks repriced already. Reprice the original CSV, not the output.",
            file=sys.stderr,
        )
        return 2

    repriced, report = reprice(df)

    moved = check_unchanged_models(report)
    print(f"Source : {src.relative_to(PROJECT_ROOT)}  ({len(df)} runs)")
    print(f"Priced against tracking/pricing.py DEFAULT_PRICING\n")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(
            report.to_string(
                index=False,
                columns=["display_name", "runs", "recorded_usd", "repriced_usd",
                         "delta_usd", "delta_pct"],
                float_format=lambda v: f"{v:,.4f}",
            )
        )
    tot_rec = report["recorded_usd"].sum()
    tot_new = report["repriced_usd"].sum()
    print(
        f"\nTOTAL  recorded ${tot_rec:,.4f} -> repriced ${tot_new:,.4f} "
        f"({tot_new - tot_rec:+,.4f}, {(tot_new / tot_rec - 1) * 100:+.2f}%)"
    )
    print(f"Models whose cost moved: {moved or 'none'}")

    # Integrity: every model that did NOT move must reproduce its recorded cost,
    # which is what makes the moved ones trustworthy.
    unmoved = report[~report["model"].isin(moved)]
    bad = unmoved[(unmoved["delta_usd"].abs() / unmoved["runs"]) > 1e-6]
    if not bad.empty:
        print(
            "\nerror: models with an unchanged rate did not reproduce their "
            "recorded cost — stored cost and stored tokens disagree:",
            file=sys.stderr,
        )
        print(bad.to_string(index=False), file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    dest = args.out or src.with_name(f"{src.stem}_repriced.csv")
    if not dest.is_absolute():
        dest = PROJECT_ROOT / dest
    repriced.to_csv(dest, index=False)
    print(f"\nWrote {dest.relative_to(PROJECT_ROOT)} ({len(repriced)} rows, "
          f"{len(repriced.columns)} cols; original untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
