"""
Build the publication figure for wave-2 token usage and cost per model.

Two stacked panels sharing one x-axis of models: tokens on top, USD cost below.
Models are grouped by provider (small -> large within each group) and coloured by
provider. Two aggregations are emitted, since they answer different questions:

    --agg mean    mean tokens / cost per *successful* run  (per-lab economics)
    --agg total   totals across all 35 wave-2 runs         (what the sweep cost)
    --agg both    both files (default)

Deliberately NOT a dual-axis chart. Plotting tokens and cost against two y-scales
on one set of bars invents a crossover point that is an artifact of where the two
scales are pinned, not a fact about the data; stacked panels carry the same two
measures against one shared x with no such artifact.

Provider colours are the validated categorical slots blue / aqua-green / orange
(worst all-pairs CVD deltaE 9.2, above the 8.0 target). A literal green for
OpenAI was tested first and hard-failed: green #008300 against orange #eb6834 is
deltaE 3.2 under protanopia -- the classic red-green confusion pair.

Usage:
    python "Code/Analysis/figures/cost_figures.py"
    python "Code/Analysis/figures/cost_figures.py" --agg mean --formats pdf png

    # Same figures against a repriced run table, written as _repriced siblings
    # so the originals stay put (see Code/Analysis/reprice_runs.py):
    python "Code/Analysis/figures/cost_figures.py" \
        --runs-csv Artifacts/Data/Benchmark/Reports/benchmark_runs_wave2_repriced.csv \
        --suffix _repriced

Outputs (under Artifacts/Paper/figures/):
    wave2_tokens_cost_mean{suffix}.{pdf,png}
    wave2_tokens_cost_total{suffix}.{pdf,png}
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from figure_style import (
    BAR_WIDTH,
    GRID,
    INK,
    INK_MUTED,
    INK_SOFT,
    MODEL_ORDER,
    PROVIDER_COLOR,
    PROVIDER_LABEL,
    SURFACE,
    draw_provider_brackets,
    rounded_bar,
    set_style,
)

ROOT = Path(__file__).resolve().parents[3]
RUNS_CSV = ROOT / "Artifacts" / "Data" / "Benchmark" / "Reports" / "benchmark_runs_wave2.csv"
OUT_DIR = ROOT / "Artifacts" / "Paper" / "figures"

# ── Data ─────────────────────────────────────────────────────────────

def load_table(agg: str, runs_csv: Path = RUNS_CSV) -> pd.DataFrame:
    """One row per model in display order, with the aggregate for `agg`."""
    df = pd.read_csv(runs_csv)
    rows = []
    for display, tick, provider in MODEL_ORDER:
        runs = df[df["display_name"] == display]
        if runs.empty:
            raise SystemExit(f"No wave-2 runs found for {display!r} in {runs_csv}")
        ok = runs[runs["success"] == True]  # noqa: E712 -- pandas mask, not identity
        source = ok if agg == "mean" else runs
        rows.append({
            "display": display,
            "tick": tick,
            "provider": provider,
            "n_ok": int(len(ok)),
            "n_runs": int(len(runs)),
            "tokens": source["total_tokens"].mean() if agg == "mean" else source["total_tokens"].sum(),
            "cost": source["cost_usd"].mean() if agg == "mean" else source["cost_usd"].sum(),
        })
    return pd.DataFrame(rows)


# ── Formatting ───────────────────────────────────────────────────────

def fmt_tokens(v: float, agg: str) -> str:
    return f"{v / 1e6:.2f}M" if agg == "total" else f"{v / 1e3:.1f}k"


def fmt_cost(v: float) -> str:
    if v >= 1:
        return f"${v:,.2f}"
    if v >= 0.01:
        return f"${v:.3f}"
    return f"${v:.4f}"


def draw_panel(ax, tbl, values, labels, ylabel, headroom=1.16):
    xs = range(len(tbl))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.xaxis.grid(False)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)

    ax.set_xlim(-0.62, len(tbl) - 0.38)
    ax.set_ylim(0, max(values) * headroom)
    # Patches need final limits before the points->data conversion is meaningful.
    ax.figure.canvas.draw()
    for x, v, row in zip(xs, values, tbl.itertuples()):
        rounded_bar(ax, x, v, BAR_WIDTH, PROVIDER_COLOR[row.provider])

    for x, v, lab in zip(xs, values, labels):
        ax.annotate(lab, (x, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_ylabel(ylabel, fontsize=9.5, color=INK, labelpad=6)
    ax.tick_params(axis="y", labelsize=8.5, pad=2)
    ax.set_xticks(list(xs))


def build_figure(agg: str, runs_csv: Path = RUNS_CSV, rate_note: str = "") -> plt.Figure:
    tbl = load_table(agg, runs_csv)

    fig, (ax_tok, ax_cost) = plt.subplots(
        2, 1, figsize=(7.4, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.13},
    )
    # Fix the axes geometry before any bars are drawn: rounded_bar converts a
    # point radius through transData, so a later subplots_adjust would silently
    # rescale corners that were computed against the old axes box.
    fig.subplots_adjust(left=0.115, right=0.985, top=0.885, bottom=0.165)

    if agg == "mean":
        tok_label = "Mean tokens per successful run\n(thousands)"
        cost_label = "Mean cost per successful run\n(USD)"
        title = "Token usage and cost per generated lab, by model"
        sub = ("Wave-2 benchmark, 2026-07-10 · 35 runs per model across "
               "7 prompt-level × spec cells · means over successful runs only")
    else:
        tok_label = "Total tokens across sweep\n(millions)"
        cost_label = "Total cost across sweep\n(USD)"
        title = "Total token usage and cost of the wave-2 sweep, by model"
        sub = ("Wave-2 benchmark, 2026-07-10 · 35 runs per model across "
               "7 prompt-level × spec cells · totals include failed runs")
    sub += rate_note

    draw_panel(ax_tok, tbl, list(tbl["tokens"]),
               [fmt_tokens(v, agg) for v in tbl["tokens"]], tok_label)
    draw_panel(ax_cost, tbl, list(tbl["cost"]),
               [fmt_cost(v) for v in tbl["cost"]], cost_label)

    if agg == "total":
        ax_tok.yaxis.set_major_formatter(lambda v, _: f"{v / 1e6:.1f}")
    else:
        ax_tok.yaxis.set_major_formatter(lambda v, _: f"{v / 1e3:.0f}")
    # Dollar ticks land on whole units in the totals panel and on cents in the
    # per-run panel; a fixed precision pads one of them with dead zeros.
    cost_dp = 0 if tbl["cost"].max() >= 2 else 2
    ax_cost.yaxis.set_major_formatter(lambda v, _: f"{v:,.{cost_dp}f}" if v else "0")

    ax_tok.set_xticklabels([])
    ax_cost.set_xticklabels(list(tbl["tick"]), fontsize=9, color=INK)
    ax_cost.tick_params(axis="x", pad=4)

    # Runs-OK row, then the provider brackets, in axes-fraction space below the
    # tick labels. Tick labels are 1-2 lines, so the row sits clear of both.
    trans = ax_cost.get_xaxis_transform()
    y_n = -0.235
    for x, row in enumerate(tbl.itertuples()):
        weak = row.n_ok < row.n_runs
        ax_cost.text(x, y_n, f"{row.n_ok}/{row.n_runs}", transform=trans,
                     ha="center", va="top", fontsize=7.5,
                     color=INK_SOFT if weak else INK_MUTED,
                     fontweight="bold" if weak else "normal", clip_on=False)
    # Right-aligned clear of the first column, in the empty margin below the
    # y tick labels -- centred under x=0 it sat on top of the first value.
    ax_cost.text(-0.78, y_n, "runs OK", transform=trans, ha="right", va="top",
                 fontsize=7.5, color=INK_MUTED, style="italic", clip_on=False)

    draw_provider_brackets(ax_cost, tbl, y_rule=-0.30, y_label=-0.325)

    fig.suptitle(title, fontsize=11.5, fontweight="bold", color=INK,
                 x=0.055, ha="left", y=0.985)
    fig.text(0.055, 0.945, sub, fontsize=8.5, color=INK_SOFT, ha="left", va="top")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agg", choices=["mean", "total", "both"], default="both")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument(
        "--runs-csv", type=Path, default=RUNS_CSV,
        help="Run-level CSV to plot (default: the wave-2 export).",
    )
    ap.add_argument(
        "--suffix", default="",
        help="Appended to output filenames, e.g. --suffix _repriced writes "
             "wave2_tokens_cost_mean_repriced.pdf and leaves the original alone.",
    )
    args = ap.parse_args()

    runs_csv = args.runs_csv if args.runs_csv.is_absolute() else ROOT / args.runs_csv
    # A repriced table carries the as-recorded figure alongside the recomputed
    # one; say so on the figure, since the cost axis then reflects the current
    # price table rather than what was billed at run time.
    rate_note = ""
    if "cost_usd_recorded" in pd.read_csv(runs_csv, nrows=0).columns:
        rate_note = "\nCosts recomputed at current published rates (see tracking/pricing.py)"

    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    aggs = ["mean", "total"] if args.agg == "both" else [args.agg]
    for agg in aggs:
        fig = build_figure(agg, runs_csv, rate_note)
        for ext in args.formats:
            out = OUT_DIR / f"wave2_tokens_cost_{agg}{args.suffix}.{ext}"
            fig.savefig(out, dpi=400 if ext == "png" else None,
                        bbox_inches="tight", pad_inches=0.06)
            print(f"wrote {out.relative_to(ROOT)}")
        plt.close(fig)


if __name__ == "__main__":
    main()
