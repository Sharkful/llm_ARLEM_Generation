"""
Recompute per-run ``cost_usd`` against the current price table, non-destructively.

Each run's cost is baked into its ``*_metrics.json`` at generation time from the
``DEFAULT_PRICING`` entry in effect *then*. When a provider changes a published
rate after the fact, every downstream artifact keeps quoting the stale number and
there is no way to notice — cost is stored, never derived.

This script re-derives cost from the token counts a run already recorded, through
the same ``tracking.pricing.PricingCalculator`` the runs use (uncached input, cache
reads and cache writes each at their own rate, when the CSV carries
``cached_tokens`` / ``cache_write_tokens``; 0 otherwise), and writes a *new* CSV
alongside the original.

Gemini thinking (#84): records made before the fix left Gemini's thinking tokens
out of ``completion_tokens``, but they are still in ``total_tokens``, so the gap
``total - prompt - completion`` is exactly the unbilled thinking (0 on records
after the fix, and on every other provider). It is billed at the output rate and
reported in its own ``thinking_cost_usd`` column. Nothing is overwritten: the input
CSV, and the ``*_metrics.json`` files behind it, are read-only here. The output is
a strict superset of the input — the recomputed figure lands in ``cost_usd`` and
the original is preserved in ``cost_usd_recorded``.

It works off the run CSV rather than ``benchmark_dataframe.load_runs()`` on
purpose: the per-run ``*_metrics.json`` are gitignored, so for every sweep older
than the current one the CSV export is the only surviving record. Repricing has to
work on what survives.

Integrity check: models whose rate did not change must reproduce their recorded
cost to the cent. It compares the rate-only re-cost (without the thinking add-on),
so a Gemini run's thinking doesn't count as a rate change. Any mismatch means the stored cost and the stored tokens
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

from tracking.metrics import TokenUsage  # noqa: E402
from tracking.pricing import DEFAULT_PRICING, PricingCalculator  # noqa: E402

# A recorded cost and a recomputed one may differ in the last float bit purely
# from summation order, so "unchanged" is a tolerance, not equality. Half a cent
# per run is far below any figure we report and far above float noise.
UNCHANGED_TOL_USD = 0.005


def cost_for(model: str, prompt_tokens: float, completion_tokens: float,
             cached_tokens: float = 0, cache_write_tokens: float = 0) -> float:
    """Cost at the current rates, by the calculator the runs use."""
    return PricingCalculator(model=model).calculate_cost(TokenUsage(
        prompt_tokens=int(prompt_tokens), completion_tokens=int(completion_tokens),
        cached_tokens=int(cached_tokens), cache_write_tokens=int(cache_write_tokens)))


def thinking_gap(provider: str, prompt: float, completion: float, total: float) -> float:
    """Gemini thinking left out of completion_tokens (records before #84); 0 otherwise."""
    return max(0.0, total - prompt - completion) if provider == "google" else 0.0


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

    def col(name):
        # Older CSVs lack the cache columns; their runs cached nothing.
        return pd.to_numeric(df[name]) if name in df.columns else pd.Series(0, index=df.index)

    prompt, completion, total = col("prompt_tokens"), col("completion_tokens"), col("total_tokens")
    cached, written = col("cached_tokens"), col("cache_write_tokens")

    rate_only = [cost_for(m, p, c, ca, w) for m, p, c, ca, w
                 in zip(df["model"], prompt, completion, cached, written)]
    thinking = [thinking_gap(pv, p, c, t) * DEFAULT_PRICING[m]["output"] / 1_000_000
                for m, pv, p, c, t in zip(df["model"], df["provider"], prompt, completion, total)]

    out = df.copy()
    # Preserve the recorded figure as the exact text the source carried.
    out["cost_usd_recorded"] = df["cost_usd"]
    out["cost_usd"] = [r + t for r, t in zip(rate_only, thinking)]
    out["thinking_cost_usd"] = thinking

    report = (
        pd.DataFrame({
            "model": df["model"],
            "display_name": df["display_name"],
            "recorded": pd.to_numeric(df["cost_usd"]),
            "rate_only": rate_only,
            "thinking": thinking,
            "repriced": out["cost_usd"],
        })
        .groupby(["model", "display_name"], as_index=False)
        .agg(
            runs=("recorded", "size"),
            recorded_usd=("recorded", "sum"),
            rate_only_usd=("rate_only", "sum"),
            thinking_usd=("thinking", "sum"),
            repriced_usd=("repriced", "sum"),
        )
    )
    # A rate change moves rate_only; thinking is reported apart (see the docstring).
    report["rate_delta_usd"] = report["rate_only_usd"] - report["recorded_usd"]
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
    per_run = (report["rate_delta_usd"].abs() / report["runs"]).fillna(0)
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
                columns=["display_name", "runs", "recorded_usd", "rate_only_usd",
                         "thinking_usd", "repriced_usd", "delta_usd", "delta_pct"],
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
    bad = unmoved[(unmoved["rate_delta_usd"].abs() / unmoved["runs"]) > 1e-6]
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
