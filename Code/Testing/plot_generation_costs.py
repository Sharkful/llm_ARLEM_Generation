"""
Publication-ready figures: mean token consumption and generation cost per
run, by model, colored by provider.

Reads the deduped run-level CSV directly (``benchmark_runs_wave2.csv``) rather
than ``benchmark_dataframe.load_runs()``, since the per-run ``*_metrics.json``
files that loader depends on are gitignored and not present in this checkout —
the CSV export is what survives.

Produces two figures over two different slices of the same data:

  - ``cost_tokens_by_model.pdf/png`` — the L3/L4 base dataset (excludes the 55
    ``level == "L1"`` rows tagged ``spec_type == "json_lab"``: L1 always
    returns the spec-agnostic ``LabOutline``, a much shorter/cheaper task than
    L3/L4's full spec, so it isn't a fair "cost of generating a lab" data
    point — same exclusion used throughout the wave-two failure tables).
    30 runs/model, 330 total, includes failed runs (which still burn tokens).

  - ``cost_tokens_by_model_L1.pdf/png`` — L1 runs only (5 runs/model, 55
    total). L1 never failed in this dataset, so this slice exists to compare
    against the base figure and decide whether L1's (cheap, always-succeeds)
    cost is worth folding into an overall cost figure or reporting separately.

Figures are sized at 8x7in so that scaling down to a ~6.5in \\textwidth (a
typical single-column article) only shrinks text by ~0.8x; font sizes below
are chosen so they stay legible (roughly caption-sized or larger) after that
shrink, not just at native resolution.

Outputs: Artifacts/Figures/

Usage:
    python "Code/Testing/plot_generation_costs.py"
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "Artifacts" / "Data" / "Benchmark" / "Reports"
FIGURES = ROOT / "Artifacts" / "Figures"
CSV_PATH = REPORTS / "benchmark_runs_wave2.csv"

# Dataviz-skill categorical hues (validated for adjacent-pair CVD separation
# in this exact left-to-right order — see conversation notes; orange and
# green must stay separated by blue, which this model order guarantees).
PROVIDER_COLOR = {
    "anthropic": "#eb6834",  # orange
    "google":    "#2a78d6",  # blue
    "openai":    "#008300",  # green
}
PROVIDER_LABEL = {"anthropic": "Anthropic", "google": "Google", "openai": "OpenAI"}

# Fixed model order: alphabetical by provider, then small -> medium -> large
# within a provider (matches generation_cost_tables.tex's billing-rate table).
MODEL_ORDER = [
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5",      "anthropic"),
    ("claude-sonnet-5",           "Claude Sonnet 5",        "anthropic"),
    ("claude-opus-4-8",           "Claude Opus 4.8",        "anthropic"),
    ("gemini-2.5-flash-lite",     "Gemini 2.5 Flash-Lite",  "google"),
    ("gemini-3.1-flash-lite",     "Gemini 3.1 Flash-Lite",  "google"),
    ("gemini-3.5-flash",          "Gemini 3.5 Flash",       "google"),
    ("gemini-3.1-pro-preview",    "Gemini 3.1 Pro Preview", "google"),
    ("gpt-5.4-nano",              "GPT-5.4 Nano",           "openai"),
    ("gpt-5.4-mini",              "GPT-5.4 Mini",           "openai"),
    ("gpt-5.4",                   "GPT-5.4",                "openai"),
    ("gpt-5.5",                   "GPT-5.5",                "openai"),
]

GROUP_GAP = 0.9  # extra x-spacing inserted between provider clusters

# Font sizes target legibility at ~0.8x (8in design width -> ~6.5in textwidth),
# not at native resolution — see module docstring.
FS_TITLE = 17
FS_LEGEND = 13
FS_AXIS_LABEL = 14
FS_TICK = 12
FS_VALUE = 10.5
FS_FOOTNOTE = 9.5


def load_df() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def base_slice(df: pd.DataFrame) -> pd.DataFrame:
    """L3/L4 runs only — excludes L1 rows (tagged spec_type='json_lab')."""
    is_l1_json_lab = (df["spec_type"] == "json_lab") & (df["level"] == "L1")
    return df[~is_l1_json_lab].copy()


def l1_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["level"] == "L1"].copy()


def compute_positions() -> list[float]:
    positions, pos, prev_provider = [], 0.0, None
    for _, _, provider in MODEL_ORDER:
        if prev_provider is not None and provider != prev_provider:
            pos += GROUP_GAP
        positions.append(pos)
        pos += 1.0
        prev_provider = provider
    return positions


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#d5d3ca", linewidth=1.1, zorder=0)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8a877e")
        ax.spines[side].set_linewidth(1.3)
    ax.tick_params(colors="#3a3835", labelsize=FS_TICK, width=1.1, length=4.5)


def bar_labels(ax, bars, values, fmt):
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt(v),
            ha="center", va="bottom",
            fontsize=FS_VALUE, color="#0b0b0b",
        )


def make_figure(df_slice: pd.DataFrame, *, title: str, footnote: str, out_stem: str):
    means = df_slice.groupby("model")[["prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"]].mean()

    order_ids = [m for m, _, _ in MODEL_ORDER]
    missing = set(order_ids) - set(means.index)
    if missing:
        raise ValueError(f"models in MODEL_ORDER not found in this slice: {missing}")

    token_vals = [means.loc[m, "total_tokens"] for m in order_ids]
    cost_vals = [means.loc[m, "cost_usd"] for m in order_ids]
    colors = [PROVIDER_COLOR[p] for _, _, p in MODEL_ORDER]
    labels = [d for _, d, _ in MODEL_ORDER]
    x = compute_positions()

    fig, (ax_tok, ax_cost) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.1},
    )

    bars_tok = ax_tok.bar(x, token_vals, width=0.62, color=colors, zorder=3)
    bars_cost = ax_cost.bar(x, cost_vals, width=0.62, color=colors, zorder=3)

    style_axis(ax_tok)
    style_axis(ax_cost)
    ax_tok.tick_params(axis="x", bottom=False)
    ax_tok.set_ylim(0, max(token_vals) * 1.18)  # headroom so value labels clear the top spine
    ax_cost.set_ylim(0, max(cost_vals) * 1.18)

    ax_tok.set_ylabel("Mean tokens / run", fontsize=FS_AXIS_LABEL, color="#0b0b0b")
    ax_cost.set_ylabel("Mean cost / run (USD)", fontsize=FS_AXIS_LABEL, color="#0b0b0b")
    ax_tok.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_cost.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.3f}"))

    bar_labels(ax_tok, bars_tok, token_vals, lambda v: f"{v:,.0f}")
    bar_labels(ax_cost, bars_cost, cost_vals, lambda v: f"${v:.4f}")

    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(labels, rotation=32, ha="right", fontsize=FS_TICK, color="#0b0b0b")
    ax_cost.set_xlim(min(x) - 0.7, max(x) + 0.7)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=PROVIDER_COLOR[p], label=PROVIDER_LABEL[p])
        for p in ("anthropic", "google", "openai")
    ]
    # Outside the axes (a row between title and plot) so it never competes
    # with a bar or its value label for space.
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.93),
        ncol=3, frameon=False, fontsize=FS_LEGEND, handlelength=1.2, handleheight=1.2,
        columnspacing=1.8,
    )

    fig.suptitle(title, fontsize=FS_TITLE, y=0.99)
    fig.text(0.5, 0.015, footnote, ha="center", fontsize=FS_FOOTNOTE, color="#3a3835")

    fig.subplots_adjust(left=0.13, right=0.97, top=0.85, bottom=0.24, hspace=0.1)

    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES / f"{out_stem}.pdf"
    png_path = FIGURES / f"{out_stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def main():
    plt.rcParams["font.family"] = ["serif"]
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Georgia", "serif"]

    df = load_df()

    base_df = base_slice(df)
    n_base = base_df.groupby("model").size()
    make_figure(
        base_df,
        title="Mean Token Consumption and Generation Cost per Run, by Model",
        footnote=(
            f"n = {int(n_base.iloc[0])} runs/model, {int(n_base.sum())} total "
            "(L3/L4 only; includes failed runs, which still consume tokens). "
            "Source: benchmark_runs_wave2.csv."
        ),
        out_stem="cost_tokens_by_model",
    )

    l1_df = l1_slice(df)
    n_l1 = l1_df.groupby("model").size()
    n_l1_failed = int((~l1_df["success"]).sum())
    make_figure(
        l1_df,
        title="Mean Token Consumption and Generation Cost per Run, by Model (L1 only)",
        footnote=(
            f"n = {int(n_l1.iloc[0])} runs/model, {int(n_l1.sum())} total "
            f"(L1 outline generation only; {n_l1_failed} of {int(n_l1.sum())} failed). "
            "Source: benchmark_runs_wave2.csv."
        ),
        out_stem="cost_tokens_by_model_L1",
    )


if __name__ == "__main__":
    main()
