"""
Publication-ready figures: mean token consumption and generation cost per
run, by model, coloured and patterned by provider.

Drawn in the shared house style (``figure_style``) so it sits alongside the
``wave2_tokens_cost_*`` and timing figures. Bars also carry a per-provider fill
pattern (``PROVIDER_HATCH``) so provider identity survives greyscale print.

Level names are Outline / Objectives / Script throughout the paper; the run CSVs
still store them as ``L1`` / ``L3`` / ``L4``, so those remain the lookup keys in
code (and in the ``_L1`` output filename, kept stable for the LaTeX that already
includes it). ``figure_style.LEVEL_NAME`` is the one place the mapping lives.

Produces two figures over two different slices of the same run table:

  - ``cost_tokens_by_model.pdf/png`` — the Objectives/Script base dataset
    (excludes the ``level == "L1"`` rows tagged ``spec_type == "json_lab"``:
    the Outline level always returns the spec-agnostic ``LabOutline``, a much
    shorter/cheaper task than a full spec, so it isn't a fair "cost of
    generating a lab" data point — same exclusion used throughout the wave-two
    failure tables). Means include failed runs, which still burn tokens.

  - ``cost_tokens_by_model_L1.pdf/png`` — Outline runs only, to compare against
    the base figure and decide whether its (cheap) cost is worth folding into
    an overall cost figure or reporting separately.

Outputs: Artifacts/Paper/figures/

Usage:
    python "Code/Analysis/figures/cost_by_model_figures.py"

    # Same figures against a repriced run table, written as _repriced siblings
    # so the originals stay put (see Code/Analysis/reprice_runs.py):
    python "Code/Analysis/figures/cost_by_model_figures.py" \
        --runs-csv Artifacts/Data/Benchmark/Reports/benchmark_runs_wave2_repriced.csv \
        --suffix _repriced
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
    LEVEL_NAME,
    MODEL_ORDER,
    PROVIDER_COLOR,
    PROVIDER_HATCH,
    draw_provider_brackets,
    rounded_bar,
    set_style,
)

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "Artifacts" / "Data" / "Benchmark" / "Reports"
FIGURES = ROOT / "Artifacts" / "Paper" / "figures"
CSV_PATH = REPORTS / "benchmark_runs_wave2.csv"


# ── Data ─────────────────────────────────────────────────────────────

def base_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Objectives/Script runs only — excludes the Outline rows (spec_type='json_lab')."""
    is_l1_json_lab = (df["spec_type"] == "json_lab") & (df["level"] == "L1")
    return df[~is_l1_json_lab].copy()


def l1_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["level"] == "L1"].copy()


def load_table(df_slice: pd.DataFrame) -> pd.DataFrame:
    """One row per model in display order: mean tokens / cost over all runs."""
    rows = []
    for display, tick, provider in MODEL_ORDER:
        runs = df_slice[df_slice["display_name"] == display]
        if runs.empty:
            raise SystemExit(f"No runs found for {display!r} in this slice")
        rows.append({
            "display": display,
            "tick": tick,
            "provider": provider,
            "n_ok": int(runs["success"].sum()),
            "n_runs": int(len(runs)),
            "tokens": runs["total_tokens"].mean(),
            "cost": runs["cost_usd"].mean(),
        })
    return pd.DataFrame(rows)


# ── Formatting ───────────────────────────────────────────────────────

def fmt_cost(v: float) -> str:
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
        rounded_bar(ax, x, v, BAR_WIDTH, PROVIDER_COLOR[row.provider],
                    hatch=PROVIDER_HATCH[row.provider])

    for x, v, lab in zip(xs, values, labels):
        ax.annotate(lab, (x, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_ylabel(ylabel, fontsize=9.5, color=INK, labelpad=6)
    ax.tick_params(axis="y", labelsize=8.5, pad=2)
    ax.set_xticks(list(xs))


def build_figure(df_slice: pd.DataFrame, *, title: str, sub: str) -> plt.Figure:
    tbl = load_table(df_slice)

    fig, (ax_tok, ax_cost) = plt.subplots(
        2, 1, figsize=(7.4, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.13},
    )
    # Fix the axes geometry before any bars are drawn: rounded_bar converts a
    # point radius through transData, so a later subplots_adjust would silently
    # rescale corners that were computed against the old axes box.
    fig.subplots_adjust(left=0.115, right=0.985, top=0.885, bottom=0.165)

    draw_panel(ax_tok, tbl, list(tbl["tokens"]),
               [f"{v / 1e3:.1f}k" for v in tbl["tokens"]],
               "Mean tokens per run\n(thousands)")
    draw_panel(ax_cost, tbl, list(tbl["cost"]),
               [fmt_cost(v) for v in tbl["cost"]],
               "Mean cost per run\n(USD)")

    ax_tok.yaxis.set_major_formatter(lambda v, _: f"{v / 1e3:.0f}")
    cost_dp = 3 if tbl["cost"].max() < 0.05 else 2
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
    ax_cost.text(-0.78, y_n, "runs OK", transform=trans, ha="right", va="top",
                 fontsize=7.5, color=INK_MUTED, style="italic", clip_on=False)

    draw_provider_brackets(ax_cost, tbl, y_rule=-0.30, y_label=-0.325)

    fig.suptitle(title, fontsize=11.5, fontweight="bold", color=INK,
                 x=0.055, ha="left", y=0.985)
    fig.text(0.055, 0.945, sub, fontsize=8.5, color=INK_SOFT, ha="left", va="top")
    return fig


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument(
        "--runs-csv", type=Path, default=CSV_PATH,
        help="Run-level CSV to plot (default: the wave-2 export).",
    )
    ap.add_argument(
        "--suffix", default="",
        help="Appended to output filenames, e.g. --suffix _repriced writes "
             "cost_tokens_by_model_repriced.pdf and leaves the original alone.",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    csv_path = args.runs_csv if args.runs_csv.is_absolute() else ROOT / args.runs_csv
    df = pd.read_csv(csv_path)

    # A repriced table carries the as-recorded figure alongside the recomputed
    # one; say so on the figure, since the cost axis then reflects the current
    # price table rather than what was billed at run time.
    rate_note = ""
    if "cost_usd_recorded" in df.columns:
        rate_note = "\nCosts recomputed at current published rates (see tracking/pricing.py)"

    set_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    base_df = base_slice(df)
    n_base = base_df.groupby("display_name").size()
    l1_df = l1_slice(df)
    n_l1 = l1_df.groupby("display_name").size()

    specs = [
        (base_df, f"cost_tokens_by_model{args.suffix}",
         "Mean token usage and cost per run, by model",
         f"Wave-2 benchmark · {LEVEL_NAME['L3']} and {LEVEL_NAME['L4']} levels · "
         f"{int(n_base.iloc[0])} runs per model, {int(n_base.sum())} total · "
         "means include failed runs, which still consume tokens"),
        (l1_df, f"cost_tokens_by_model_L1{args.suffix}",
         f"Mean token usage and cost per run, by model ({LEVEL_NAME['L1']} level only)",
         f"Wave-2 benchmark · {LEVEL_NAME['L1']} level · {int(n_l1.iloc[0])} runs per model, "
         f"{int(n_l1.sum())} total · means include failed runs"),
    ]
    for df_slice, stem, title, sub in specs:
        fig = build_figure(df_slice, title=title, sub=sub + rate_note)
        for ext in args.formats:
            out = FIGURES / f"{stem}.{ext}"
            fig.savefig(out, dpi=400 if ext == "png" else None,
                        bbox_inches="tight", pad_inches=0.06)
            print(f"wrote {out.relative_to(ROOT)}")
        plt.close(fig)


if __name__ == "__main__":
    main()
