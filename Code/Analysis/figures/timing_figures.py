"""
Build the publication figures for wave-2 *generation time*, plus the LaTeX
snippets (figure blocks + booktabs tables) that drop into the paper.

Four figures, each answering a different question:

  1. wave2_latency_throughput   -- how long one request takes, and how fast the
                                   model emits tokens once it is going
  2. wave2_time_decomposition   -- where the sweep's hours actually went:
                                   clean runs / schema-repair runs / failed runs
  3. wave2_time_vs_tokens       -- small multiples showing that time is a linear
                                   function of output tokens with a per-model
                                   slope, which is what licenses (1)'s tok/s
  4. wave2_time_by_level_spec   -- how prompt level and target spec move the cost

What the time number *is*
-------------------------
``duration_ms`` is start -> end of the whole instructor ``create()`` call,
including every schema-repair retry, so it measures **time to a valid
structured output**, not single-call latency. (``wall_s`` is the same number in
seconds -- the two agree to within 0.03% across all 385 runs -- so only one of
them is used here.) The wave-2 sweep ran strictly serially: checking every
consecutive pair of run timestamps finds zero overlap, and the 6.80 h of summed
run time sits inside a 6.86 h span. That is what makes cross-model wall-clock
comparison legitimate at all, and it belongs in the caption.

Slice
-----
Figures 1-3 use the **Objectives/Script slice** (30 runs per model), matching
the exclusion in ``cost_by_model_figures.py``: the Outline level returns the
spec-agnostic ``LabOutline``, a much shorter task (15 s mean vs 56 s / 88 s), so
folding it in dilutes any per-request number. Figure 4 is the deliberate
exception -- comparing levels is its entire subject, so it runs over all 385
rows and says so on the figure.

Level names are Outline / Objectives / Script throughout the paper; the run
CSVs still store them as ``L1`` / ``L3`` / ``L4``, so those remain the lookup
keys in code. ``figure_style.LEVEL_NAME`` is the one place the mapping lives.

Why output tokens, never total tokens, as the time denominator
--------------------------------------------------------------
Prompt tokens are 47-85% of the totals here and are prefilled far faster than
they are decoded, so "total tokens per second" mostly measures prompt length:
Gemini 3.1 Flash Lite reads 2184 total tok/s against 321 output tok/s purely
because 85% of its traffic is prompt. Every throughput number below is output
tokens only.

Retry counting
--------------
The ``retries`` column double-counts -- it maps to ``parse_errors`` as
0->0, 1->2, 2->4, 3->6, 4->7, i.e. 2x up to a cap. The clean/not-clean
*partition* is identical either way, so these figures split on
``parse_errors > 0`` and never quote a raw retry count.

Colour
------
Provider hues are the validated categorical slots blue / aqua-green / orange
shared via ``figure_style`` (worst all-pairs CVD deltaE 9.2). Figure 2's
stacked segments need their own ordered ramp; the literal status trio
good/warning/critical hard-failed (green #0ca30c vs red #d03b3b is deltaE 4.1
under deuteranopia -- the red-green trap), so it uses violet/amber/red, which
passes all-pairs at deltaE 19.8 and shares no hue with the provider palette,
keeping the two figures from cross-talking. Figure 2's provider brackets are
drawn in muted ink for the same reason.

Usage:
    python "Code/Analysis/figures/timing_figures.py"
    python "Code/Analysis/figures/timing_figures.py" --figures latency scatter
    python "Code/Analysis/figures/timing_figures.py" \
        --runs-csv Artifacts/Data/Benchmark/Reports/benchmark_runs_wave2.csv \
        --suffix _asrun

Outputs:
    Artifacts/Paper/figures/wave2_latency_throughput{suffix}.{pdf,png}
    Artifacts/Paper/figures/wave2_time_decomposition{suffix}.{pdf,png}
    Artifacts/Paper/figures/wave2_time_vs_tokens{suffix}.{pdf,png}
    Artifacts/Paper/figures/wave2_time_by_level_spec{suffix}.{pdf,png}
    Artifacts/Paper/tables/generation_time_tables{suffix}.tex
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath

from figure_style import (
    BAR_WIDTH,
    CORNER_PT,
    GRID,
    HATCH_COLOR,
    INK,
    INK_MUTED,
    INK_SOFT,
    LEVEL_GLOSS,
    LEVEL_NAME,
    MODEL_ORDER,
    PROVIDER_COLOR,
    PROVIDER_HATCH,
    PROVIDER_LABEL,
    SURFACE,
    draw_provider_brackets,
    rounded_bar,
    set_style,
)  # noqa: F401  -- CORNER_PT re-exported for figures that draw their own marks

ROOT = Path(__file__).resolve().parents[3]
RUNS_CSV = (ROOT / "Artifacts" / "Data" / "Benchmark" / "Reports"
            / "benchmark_runs_wave2_repriced.csv")
FIG_DIR = ROOT / "Artifacts" / "Paper" / "figures"
TEX_DIR = ROOT / "Artifacts" / "Paper" / "tables"

# Outcome ramp for the stacked decomposition -- ordered calm -> alarming.
# Validated all-pairs on white: worst CVD deltaE 19.8, normal-vision 24.1.
OUTCOME_COLOR = {
    "clean": "#4a3aa7",     # violet -- valid first try
    "repair": "#eda100",    # amber  -- valid after >=1 schema repair
    "failed": "#d03b3b",    # red    -- never produced a valid spec
}
# Redundant patterns so the bands survive greyscale print (see PROVIDER_HATCH).
OUTCOME_HATCH = {
    "clean": "",
    "repair": "///",
    "failed": "xxx",
}
OUTCOME_LABEL = {
    "clean": "Valid on first attempt",
    "repair": "Valid after schema repair",
    "failed": "Never valid (run failed)",
}

# Display name over the prompt contents. Keys stay L1/L3/L4 -- that is what the
# `level` column holds -- while the printed name comes from figure_style.
LEVEL_LABEL = {k: f"{LEVEL_NAME[k]}\n{LEVEL_GLOSS[k]}" for k in ("L1", "L3", "L4")}
SPEC_LABEL = {
    "json_lab": "json_lab\n(Lab JSON)",
    "arlem_simple": "arlem_simple\n(reduced ARLEM)",
    "arlem": "arlem\n(full ARLEM)",
}

SERIAL_NOTE = ("runs executed serially, so wall-clock time is free of "
               "concurrency contention")


# ---- Data -----------------------------------------------------------

def load_runs(runs_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(runs_csv)
    df["dur_s"] = df["duration_ms"] / 1000.0
    # parse_errors, not retries -- see module docstring.
    df["repaired"] = df["parse_errors"] > 0
    df["ok"] = df["success"] == True  # noqa: E712 -- pandas mask, not identity
    return df


def base_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Objectives/Script only: Outline's LabOutline is a much cheaper task."""
    return df[df["level"] != "L1"]


def per_model(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model in display order, with every timing aggregate."""
    rows = []
    for display, tick, provider in MODEL_ORDER:
        runs = df[df["display_name"] == display]
        if runs.empty:
            raise SystemExit(f"No runs found for {display!r}")
        ok = runs[runs["ok"]]
        clean = ok[~ok["repaired"]]
        rep = ok[ok["repaired"]]
        fail = runs[~runs["ok"]]
        rows.append({
            "display": display, "tick": tick, "provider": provider,
            "n_runs": len(runs), "n_ok": len(ok),
            "n_clean": len(clean), "n_repair": len(rep), "n_fail": len(fail),
            "med_s": ok["dur_s"].median(),
            "mean_s": ok["dur_s"].mean(),
            "q1_s": ok["dur_s"].quantile(0.25),
            "q3_s": ok["dur_s"].quantile(0.75),
            # Pooled throughput: total output tokens over total time. This is the
            # rate the sweep actually ran at, and it inverts cleanly to s/1k.
            "out_tps": ok["completion_tokens"].sum() / ok["dur_s"].sum(),
            "s_per_1k": ok["dur_s"].sum() / (ok["completion_tokens"].sum() / 1000.0),
            "clean_min": clean["dur_s"].sum() / 60.0,
            "repair_min": rep["dur_s"].sum() / 60.0,
            "fail_min": fail["dur_s"].sum() / 60.0,
            "total_min": runs["dur_s"].sum() / 60.0,
        })
    return pd.DataFrame(rows)


# ---- Shared chrome --------------------------------------------------

def style_panel(ax, n_cols):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.xaxis.grid(False)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.set_xlim(-0.62, n_cols - 0.38)
    ax.set_xticks(range(n_cols))
    ax.tick_params(axis="y", labelsize=8.5, pad=2)


def counts_row(ax, tbl, y, label, numer, weak_when):
    """A `n/N` row under the x-axis, bolded where the count is notable.

    `numer` names the column to show over `n_runs`; `weak_when` decides which
    entries get the bold treatment, so each figure can foreground the count
    its own story turns on.
    """
    trans = ax.get_xaxis_transform()
    for x, row in enumerate(tbl.itertuples()):
        weak = weak_when(row)
        ax.text(x, y, f"{getattr(row, numer)}/{row.n_runs}", transform=trans,
                ha="center", va="top", fontsize=7.5,
                color=INK_SOFT if weak else INK_MUTED,
                fontweight="bold" if weak else "normal", clip_on=False)
    # Right-aligned into the empty margin under the y tick labels; centred on
    # x=0 it would sit on top of the first column's value.
    ax.text(-0.78, y, label, transform=trans, ha="right", va="top",
            fontsize=7.5, color=INK_MUTED, style="italic", clip_on=False)


def titles(fig, title, sub):
    fig.suptitle(title, fontsize=11.5, fontweight="bold", color=INK,
                 x=0.055, ha="left", y=0.985)
    fig.text(0.055, 0.945, sub, fontsize=8.5, color=INK_SOFT, ha="left", va="top")


# ---- Figure 1: latency + throughput ---------------------------------

def fig_latency_throughput(df: pd.DataFrame) -> plt.Figure:
    tbl = per_model(base_slice(df))

    fig, (ax_lat, ax_tps) = plt.subplots(
        2, 1, figsize=(7.4, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.13},
    )
    # Geometry before marks: rounded_bar converts a point radius through
    # transData, so a later subplots_adjust would rescale finished corners.
    fig.subplots_adjust(left=0.115, right=0.985, top=0.865, bottom=0.165)

    n = len(tbl)
    style_panel(ax_lat, n)
    ax_lat.set_ylim(0, tbl["q3_s"].max() * 1.18)
    fig.canvas.draw()
    for x, row in enumerate(tbl.itertuples()):
        rounded_bar(ax_lat, x, row.med_s, BAR_WIDTH, PROVIDER_COLOR[row.provider],
                    hatch=PROVIDER_HATCH[row.provider])
    # IQR whisker. A 2px surface ring keeps the below-median half legible where
    # it crosses its own bar, per the overlapping-mark spec.
    ring = [pe.withStroke(linewidth=3.1, foreground=SURFACE)]
    for x, row in enumerate(tbl.itertuples()):
        ax_lat.plot([x, x], [row.q1_s, row.q3_s], color=INK, linewidth=1.1,
                    solid_capstyle="butt", zorder=4, path_effects=ring)
        for yv in (row.q1_s, row.q3_s):
            ax_lat.plot([x - 0.13, x + 0.13], [yv, yv], color=INK, linewidth=1.1,
                        solid_capstyle="butt", zorder=4, path_effects=ring)
        ax_lat.annotate(f"{row.med_s:.0f}s", (x, row.q3_s), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=INK)
    ax_lat.set_ylabel("Time to a valid spec\n(seconds per request)",
                      fontsize=9.5, color=INK, labelpad=6)
    ax_lat.legend(
        handles=[Line2D([0], [0], color=INK, lw=1.1)],
        labels=["interquartile range (bar = median)"],
        loc="upper left", frameon=False, fontsize=8, handlelength=1.4,
        labelcolor=INK_SOFT, borderpad=0.1,
    )

    style_panel(ax_tps, n)
    ax_tps.set_ylim(0, tbl["out_tps"].max() * 1.16)
    fig.canvas.draw()
    for x, row in enumerate(tbl.itertuples()):
        rounded_bar(ax_tps, x, row.out_tps, BAR_WIDTH, PROVIDER_COLOR[row.provider],
                    hatch=PROVIDER_HATCH[row.provider])
        ax_tps.annotate(f"{row.out_tps:.0f}", (x, row.out_tps), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=INK)
    ax_tps.set_ylabel("Output throughput\n(output tokens per second)",
                      fontsize=9.5, color=INK, labelpad=6)

    ax_lat.set_xticklabels([])
    ax_tps.set_xticklabels(list(tbl["tick"]), fontsize=9, color=INK)
    ax_tps.tick_params(axis="x", pad=4)
    counts_row(ax_tps, tbl, y=-0.235, label="runs OK", numer="n_ok",
               weak_when=lambda r: r.n_ok < r.n_runs)
    draw_provider_brackets(ax_tps, tbl, y_rule=-0.30, y_label=-0.325)

    titles(fig,
           "Generation time and output throughput per request, by model",
           "Wave-2 benchmark, 2026-07-10 \u00b7 30 Objectives/Script runs per "
           "model \u00b7 successful runs only\nTime is measured end to end and includes every "
           "schema-repair retry, so it is time to a valid spec,\nnot single-call "
           f"latency \u00b7 {SERIAL_NOTE}")
    return fig


# ---- Figure 2: where the time went ----------------------------------

def stack_segment(ax, x, y0, y1, width, color, round_top, hatch=None):
    """One segment of a stacked column, square at its base."""
    if y1 <= y0:
        return
    left, right = x - width / 2, x + width / 2
    if not round_top:
        verts = [(left, y0), (left, y1), (right, y1), (right, y0), (left, y0)]
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO,
                 MplPath.CLOSEPOLY]
    else:
        inv = ax.transData.inverted()
        r_px = CORNER_PT * ax.figure.dpi / 72
        p0, p1 = inv.transform((0, 0)), inv.transform((r_px, r_px))
        rx = min(abs(p1[0] - p0[0]), width / 2)
        ry = min(abs(p1[1] - p0[1]), (y1 - y0) / 2)
        verts = [(left, y0), (left, y1 - ry), (left, y1), (left + rx, y1),
                 (right - rx, y1), (right, y1), (right, y1 - ry),
                 (right, y0), (left, y0)]
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
                 MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
                 MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                           hatch=hatch or None,
                           edgecolor=HATCH_COLOR if hatch else "none",
                           linewidth=0, zorder=3))


# Every text size in the figure, in one place for tuning legibility.
D_FONT = {"suptitle": 18, "legend": 14, "ylabel": 15, "yticks": 13,
          "xticks": 13, "values": 12, "brackets": 14}


def fig_time_decomposition(df: pd.DataFrame) -> plt.Figure:
    tbl = per_model(base_slice(df))

    # Same width as the structure figures, so the same point sizes read the
    # same once LaTeX scales both to \linewidth.
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.85, bottom=0.17)

    style_panel(ax, len(tbl))
    ax.set_ylim(0, tbl["total_min"].max() * 1.17)
    ax.tick_params(axis="y", labelsize=D_FONT["yticks"])
    fig.canvas.draw()

    # 2px surface gap between touching segments, expressed in data units.
    inv = ax.transData.inverted()
    gap = abs(inv.transform((0, 2 * fig.dpi / 96))[1] - inv.transform((0, 0))[1])

    for x, row in enumerate(tbl.itertuples()):
        parts = [("clean", row.clean_min), ("repair", row.repair_min),
                 ("failed", row.fail_min)]
        drawn = [(k, v) for k, v in parts if v > 0]
        y = 0.0
        for i, (key, v) in enumerate(drawn):
            top = y + v
            last = i == len(drawn) - 1
            stack_segment(ax, x, y, top - (0 if last else gap), BAR_WIDTH,
                          OUTCOME_COLOR[key], round_top=last,
                          hatch=OUTCOME_HATCH[key])
            y = top
        ax.annotate(f"{row.total_min:.0f} min", (x, row.total_min), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=D_FONT["values"], color=INK)

    ax.set_ylabel("Wall-clock time spent\n(minutes, 30 runs per model)",
                  fontsize=D_FONT["ylabel"], color=INK, labelpad=6)
    ax.set_xticklabels(list(tbl["tick"]), fontsize=D_FONT["xticks"], color=INK)
    ax.tick_params(axis="x", pad=4)
    # Brackets stay muted ink here: the fills already spend the colour channel.
    draw_provider_brackets(ax, tbl, y_rule=-0.145, y_label=-0.17,
                           rule_color=INK_MUTED, fontsize=D_FONT["brackets"])

    fig.legend(
        handles=[Patch(facecolor=OUTCOME_COLOR[k], hatch=OUTCOME_HATCH[k] or None,
                       edgecolor=HATCH_COLOR, linewidth=0)
                 for k in ("clean", "repair", "failed")],
        labels=[OUTCOME_LABEL[k] for k in ("clean", "repair", "failed")],
        loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, frameon=False,
        fontsize=D_FONT["legend"], handlelength=1.8, handleheight=1.2,
        labelcolor=INK, columnspacing=2.4, borderpad=0.0,
    )

    fig.suptitle("Where the sweep's compute time went, by model",
                 fontsize=D_FONT["suptitle"], fontweight="bold", color=INK,
                 x=0.5, ha="center", y=0.99)
    return fig


# ---- Figure 3: time vs output tokens, small multiples ---------------

def fig_time_vs_tokens(df: pd.DataFrame) -> plt.Figure:
    ok = base_slice(df)
    ok = ok[ok["ok"]]

    fig, axes = plt.subplots(3, 4, figsize=(7.4, 6.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.088, right=0.985, top=0.825, bottom=0.10,
                        hspace=0.34, wspace=0.12)
    flat = axes.ravel()

    xmax = ok["completion_tokens"].max() * 1.08 / 1000
    ymax = ok["dur_s"].max() * 1.10

    for i, (display, _tick, provider) in enumerate(MODEL_ORDER):
        ax = flat[i]
        g = ok[ok["display_name"] == display]
        x = g["completion_tokens"].to_numpy() / 1000.0
        y = g["dur_s"].to_numpy()

        ax.set_axisbelow(True)
        ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)

        r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
        # A fit line on a relationship this weak would assert a slope the data
        # does not support, so it is withheld rather than drawn faint.
        if len(x) > 2 and r >= 0.80:
            slope, icept = np.polyfit(x, y, 1)
            # Drawn only across the tokens this model was actually observed at.
            # Running it out to the shared xmax would extrapolate a model like
            # 3.1 Flash Lite (max 6k output tokens) five times past its data.
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, slope * xs + icept, color=INK_SOFT, linewidth=1.2,
                    zorder=4, solid_capstyle="round")
            note = f"{slope:.1f} s/1k \u00b7 r={r:.2f}"
        else:
            note = f"no fit \u00b7 r={r:.2f}" if len(x) > 2 else "no fit"

        ax.scatter(x, y, s=17, facecolor=PROVIDER_COLOR[provider],
                   edgecolor=SURFACE, linewidth=0.9, zorder=5)
        short = (display.replace("Claude ", "").replace("Gemini ", "")
                 .replace("GPT-", "GPT "))
        ax.set_title(short, fontsize=8.5, color=INK, pad=12, loc="left")
        ax.text(0.0, 1.01, f"n={len(x)} \u00b7 {note}", transform=ax.transAxes,
                fontsize=7.2, color=INK_MUTED, ha="left", va="bottom")
        ax.tick_params(labelsize=7.5, length=2, pad=1.5)

    # The last cell holds the legend rather than a model, so the panel directly
    # above it is the bottom of its column and needs its x tick labels back --
    # sharex only puts them on the literal bottom row.
    ncols = axes.shape[1]
    flat[len(flat) - 1 - ncols].tick_params(labelbottom=True)

    flat[-1].axis("off")
    flat[-1].legend(
        handles=[Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                        markerfacecolor=PROVIDER_COLOR[p], markeredgecolor=SURFACE,
                        markeredgewidth=0.9)
                 for p in ("anthropic", "google", "openai")]
        + [Line2D([0], [0], color=INK_SOFT, lw=1.2)],
        labels=[PROVIDER_LABEL[p] for p in ("anthropic", "google", "openai")]
        + ["least-squares fit"],
        loc="center left", frameon=False, fontsize=8, handlelength=1.3,
        labelcolor=INK_SOFT, borderpad=0.0, labelspacing=0.8,
    )

    fig.supxlabel("Output tokens (thousands)", fontsize=9.5, color=INK, y=0.028)
    fig.supylabel("Time to a valid spec (seconds)", fontsize=9.5, color=INK, x=0.014)

    titles(fig,
           "Generation time is linear in output tokens, at a rate set by the model",
           "Wave-2 benchmark, 2026-07-10 \u00b7 successful Objectives/Script runs \u00b7 "
           "shared axes, so a steeper line is a slower model\nThe per-model slope is what "
           "licenses summarising a model's speed as one tokens-per-second number")
    return fig


# ---- Figure 4: level and spec ---------------------------------------

def _box_panel(ax, df, key, order, labels, xlabel):
    data = [df[df[key] == k]["dur_s"].to_numpy() for k in order]
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    ax.boxplot(data, widths=0.42, showfliers=False, patch_artist=True,
               medianprops=dict(color=INK, linewidth=1.4),
               boxprops=dict(facecolor="#f2f1ee", edgecolor=GRID, linewidth=0.9),
               whiskerprops=dict(color=INK_MUTED, linewidth=0.9),
               capprops=dict(color=INK_MUTED, linewidth=0.9), zorder=2)

    # Per-model medians on top: shows the pooled box is not one model's artefact.
    rng = np.random.default_rng(7)
    for i, k in enumerate(order, start=1):
        sub = df[df[key] == k]
        for display, _tick, provider in MODEL_ORDER:
            v = sub[sub["display_name"] == display]["dur_s"]
            if v.empty:
                continue
            ax.scatter(i + rng.uniform(-0.14, 0.14), v.median(), s=15,
                       facecolor=PROVIDER_COLOR[provider], edgecolor=SURFACE,
                       linewidth=0.8, zorder=5)
        ax.text(i, -0.045, f"n={len(sub)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.2, color=INK_MUTED, clip_on=False)

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(labels, fontsize=8.5, color=INK)
    ax.tick_params(axis="x", pad=15, length=0)
    ax.tick_params(axis="y", labelsize=8.5, pad=2)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK, labelpad=8)


def fig_time_by_level_spec(df: pd.DataFrame) -> plt.Figure:
    ok = df[df["ok"]]
    # The two panels deliberately run over different slices. Levels need the
    # Outline level -- comparing levels is the panel's subject. Specs must NOT
    # have it: Outline exists only under json_lab (every level returns
    # LabOutline there), so pooling it in drags json_lab's median from 68 s down
    # to 37 s and makes the most expensive target spec look like the cheapest.
    ok_spec = ok[ok["level"] != "L1"]

    fig, (ax_l, ax_s) = plt.subplots(1, 2, figsize=(7.4, 4.8), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.735, bottom=0.215, wspace=0.08)

    _box_panel(ax_l, ok, "level", ["L1", "L3", "L4"],
               [LEVEL_LABEL[k] for k in ("L1", "L3", "L4")],
               "Prompt specificity level (all runs)")
    _box_panel(ax_s, ok_spec, "spec_type", ["json_lab", "arlem_simple", "arlem"],
               [SPEC_LABEL[k] for k in ("json_lab", "arlem_simple", "arlem")],
               "Target specification (Objectives/Script)")
    ax_l.set_ylabel("Time to a valid spec (seconds)", fontsize=9.5, color=INK,
                    labelpad=6)

    # Legend above the axes, as in the decomposition figure: inside the panels
    # it lands on the tall arlem whisker.
    leg = ax_l.legend(
        handles=[Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                        markerfacecolor=PROVIDER_COLOR[p], markeredgecolor=SURFACE,
                        markeredgewidth=0.8)
                 for p in ("anthropic", "google", "openai")],
        labels=[PROVIDER_LABEL[p] for p in ("anthropic", "google", "openai")],
        loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3, frameon=False,
        fontsize=8, handlelength=1.0, labelcolor=INK_SOFT, borderpad=0.0,
        columnspacing=1.6, title="Dots: per-model median", title_fontsize=8,
    )
    leg.get_title().set_color(INK_MUTED)
    leg.get_title().set_ha("left")

    titles(fig,
           "Generation time by prompt specificity and target specification",
           "Wave-2 benchmark, 2026-07-10 \u00b7 successful runs \u00b7 boxes pool "
           "all 11 models, whiskers to 1.5\u00d7IQR, outliers omitted\nThe Outline "
           "level appears only in the left panel: it exists solely under json_lab, "
           "so pooling it into the spec comparison would bias it")
    return fig


# ---- LaTeX ----------------------------------------------------------

PROV_TEX = {"anthropic": "Anthropic", "google": "Google", "openai": "OpenAI"}

FIGURE_BLOCKS = [
    ("wave2_latency_throughput", "fig:generation-time",
     "Generation time and output throughput per request, by model. Bars are "
     "medians over the 30 Objectives and Script runs per model; whiskers give "
     "the interquartile "
     "range. Time is measured end to end and includes every schema-repair retry, "
     "so it is time to a valid specification rather than single-call latency. "
     "Runs executed serially, so wall-clock time carries no concurrency "
     "contention. Throughput counts output tokens only."),
    ("wave2_time_decomposition", "fig:time-decomposition",
     "Where each model's compute time went. Runs are bucketed whole into those "
     "valid on the first attempt, those that succeeded only after at least one "
     "schema repair, and those that never produced a valid specification. Retry "
     "burden, not raw model speed, dominates the time budget: all three "
     "GPT-5.4 variants were valid on the first attempt in just 1 of 30 runs "
     "each, and Gemini~2.5~Flash-Lite spent 83\\% of its time on runs that "
     "never succeeded."),
    ("wave2_time_vs_tokens", "fig:time-vs-tokens",
     "Time to a valid specification against output tokens, one panel per model "
     "on shared axes, so a steeper line is a slower model. The relationship is "
     "close to linear within every model except Gemini~2.5~Flash-Lite, whose "
     "seven successful runs are too few and too retry-contaminated to fit. The "
     "per-model slope is what licenses summarising speed as a single "
     "tokens-per-second figure."),
    ("wave2_time_by_level_spec", "fig:time-by-level-spec",
     "Generation time by prompt specificity level and by target specification. "
     "Boxes pool all eleven models; dots mark each model's median. Median time "
     "rises from 15~s at the Outline level to 44~s at Objectives and 63~s at "
     "Script, and the reduced ARLEM "
     "specification generates in 39~s against 67~s for full ARLEM and 68~s for "
     "the Lab JSON. The panels use different slices by necessity: Outline "
     "runs exist only under \\texttt{json\\_lab}, so including them in the "
     "right-hand panel would drag that spec's median down to 37~s and make the "
     "most expensive target look like the cheapest."),
]


def write_tex(df: pd.DataFrame, suffix: str, out: Path) -> None:
    tbl = per_model(base_slice(df))
    slice_h = base_slice(df)["dur_s"].sum() / 3600.0
    all_h = df["dur_s"].sum() / 3600.0

    L = [
        "% Generation time tables and figures -- wave-2 benchmark.",
        "% GENERATED by Code/Analysis/figures/timing_figures.py -- edit that, not this.",
        "%",
        "% The timing source column is duration_ms: start -> end of the whole",
        "% instructor create() call, INCLUDING every schema-repair retry. It is",
        "% therefore time to a valid structured output, not single-call latency.",
        "% The sweep ran strictly serially (zero overlap between consecutive runs),",
        "% so wall-clock time carries no concurrency contention.",
        "%",
        "% Numbers below are the Objectives/Script slice, 30 runs per model: the",
        "% Outline level returns the spec-agnostic LabOutline, a much cheaper task,",
        "% and is excluded for the",
        "% same reason as in generation_cost_tables.tex. Throughput uses OUTPUT",
        "% tokens only -- prompt tokens are 47-85% of totals here and are prefilled",
        "% far faster than they are decoded, so a total-token rate would mostly",
        "% measure prompt length.",
        "%",
        "% Required packages:",
        "%   \\usepackage{booktabs}",
        "%   \\usepackage{graphicx}",
        "%",
        "% Regenerate with:",
        "%   python \"Code/Analysis/figures/timing_figures.py\"",
        "",
        "% " + "-" * 73,
        "% Per-model generation time",
        "% " + "-" * 73,
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Generation time by model. Median and interquartile range are "
        "over the successful runs of 30 Objectives and Script requests per model; "
        "throughput and total are pooled over the same runs.}",
        "  \\label{tab:generation-time}",
        "  \\small",
        "  \\begin{tabular}{llrrrrrr}",
        "    \\toprule",
        "    & & & \\multicolumn{2}{c}{Time per request (s)} & "
        "\\multicolumn{2}{c}{Output throughput} & Total \\\\",
        "    \\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "    Provider & Model & OK & Median & IQR & tok/s & s/1k tok & (min) \\\\",
        "    \\midrule",
    ]
    prev = None
    for row in tbl.itertuples():
        if prev is not None and row.provider != prev:
            L.append("    \\cmidrule(lr){1-8}")
        prov = PROV_TEX[row.provider] if row.provider != prev else ""
        L.append(
            f"    {prov} & {row.display} & {row.n_ok}/{row.n_runs} & "
            f"{row.med_s:.0f} & {row.q1_s:.0f}--{row.q3_s:.0f} & "
            f"{row.out_tps:.0f} & {row.s_per_1k:.1f} & {row.total_min:.0f} \\\\"
        )
        prev = row.provider
    L += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
        "% " + "-" * 73,
        "% Where the time went",
        "% " + "-" * 73,
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Share of each model's wall-clock time spent on runs that were "
        "valid on the first attempt, needed at least one schema repair, or never "
        "produced a valid specification. Runs are bucketed whole.}",
        "  \\label{tab:time-decomposition}",
        "  \\small",
        "  \\begin{tabular}{llrrrrr}",
        "    \\toprule",
        "    & & \\multicolumn{3}{c}{Share of wall-clock time (\\%)} & "
        "First-try & Total \\\\",
        "    \\cmidrule(lr){3-5}",
        "    Provider & Model & First try & Repaired & Failed & runs & (min) \\\\",
        "    \\midrule",
    ]
    prev = None
    for row in tbl.itertuples():
        if prev is not None and row.provider != prev:
            L.append("    \\cmidrule(lr){1-7}")
        prov = PROV_TEX[row.provider] if row.provider != prev else ""
        tot = row.total_min or 1.0
        L.append(
            f"    {prov} & {row.display} & {100 * row.clean_min / tot:.0f} & "
            f"{100 * row.repair_min / tot:.0f} & {100 * row.fail_min / tot:.0f} & "
            f"{row.n_clean}/{row.n_runs} & {row.total_min:.0f} \\\\"
        )
        prev = row.provider
    L += [
        "    \\midrule",
        f"    \\multicolumn{{7}}{{l}}{{\\footnotesize Sweep total: "
        f"{slice_h:.2f}~h over the Objectives and Script slice; {all_h:.2f}~h "
        f"including the Outline level.}} \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
        "% " + "-" * 73,
        "% Figures. \\includegraphics paths assume Figures/ sits next to the",
        "% compiled main.tex, per the document's actual layout -- same convention",
        "% as generation_cost_tables.tex.",
        "% " + "-" * 73,
    ]
    for stem, label, cap in FIGURE_BLOCKS:
        L += [
            "",
            "\\begin{figure}[htbp]",
            "  \\centering",
            f"  \\includegraphics[width=\\linewidth]{{Figures/{stem}{suffix}.pdf}}",
            f"  \\caption{{{cap}}}",
            f"  \\label{{{label}}}",
            "\\end{figure}",
        ]

    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


# ---- CLI ------------------------------------------------------------

BUILDERS = {
    "latency": ("wave2_latency_throughput", fig_latency_throughput),
    "decomposition": ("wave2_time_decomposition", fig_time_decomposition),
    "scatter": ("wave2_time_vs_tokens", fig_time_vs_tokens),
    "levels": ("wave2_time_by_level_spec", fig_time_by_level_spec),
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", nargs="+", default=["all"],
                    choices=["all", *BUILDERS])
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--runs-csv", type=Path, default=RUNS_CSV,
                    help="Run-level CSV to plot (default: the repriced wave-2 "
                         "export; pricing is irrelevant to timing, but keeping "
                         "one source keeps the run set identical to the cost "
                         "figures).")
    ap.add_argument("--suffix", default="",
                    help="Appended to output filenames, e.g. --suffix _asrun.")
    ap.add_argument("--no-tex", action="store_true",
                    help="Skip the LaTeX snippet file.")
    args = ap.parse_args()

    runs_csv = args.runs_csv if args.runs_csv.is_absolute() else ROOT / args.runs_csv
    df = load_runs(runs_csv)

    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    wanted = list(BUILDERS) if "all" in args.figures else args.figures
    for key in wanted:
        stem, builder = BUILDERS[key]
        fig = builder(df)
        for ext in args.formats:
            path = FIG_DIR / f"{stem}{args.suffix}.{ext}"
            fig.savefig(path, dpi=400 if ext == "png" else None,
                        bbox_inches="tight", pad_inches=0.06)
            print(f"wrote {path.relative_to(ROOT)}")
        plt.close(fig)

    if not args.no_tex:
        write_tex(df, args.suffix,
                  TEX_DIR / f"generation_time_tables{args.suffix}.tex")


if __name__ == "__main__":
    main()
