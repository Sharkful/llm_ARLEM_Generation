"""
Output-structure figures: what the generated labs actually contain.

Everything upstream of this file measures the *cost* of generation -- tokens,
dollars, seconds. These five figures measure the *artefact*: how big the
generated specifications are, how much new authoring work they imply, and
whether they would function at all.

Five figures, two LaTeX tables, written to Artifacts/Paper/.

    wave2_json_lab_size          size against the human-authored moon lab
    wave2_authoring_burden       novel textures invented, per model
    wave2_arlem_validity         the OpenAI zero-activate defect
    wave2_specificity_elasticity what the Script level adds, format against format
    wave2_arlem_floor            how often models stop at the stated minimum

Regenerate:
    python "Code/Analysis/figures/structure_figures.py"


Slices, and why they are not optional
-------------------------------------
Level names are Outline / Objectives / Script throughout the paper; the run CSVs
still store them as ``L1`` / ``L3`` / ``L4``, so those remain the lookup keys in
code. ``figure_style.LEVEL_NAME`` is the one place the mapping lives.

**The Outline level is excluded from every structural figure.** It returns the
spec-agnostic ``LabOutline``, which has no modules, clips, objects or components
-- so all 55 Outline rows record structural zeros. They are 35% of the json_lab
rows and would halve every median. The json_lab slice is therefore the 104
successful Objectives and Script runs, matching the cost and timing figures.

**arlem and arlem_simple are pooled only where the schemas agree.** The
simplified schema has no persons, sensors, devices, apps, warnings, messages,
or voice/sensor triggers, so a zero in those columns is a schema fact, not a
model behaviour. Figures 3 and 5 pool the two specs because the fields they
read (``activates``, things/places/actions) exist identically in both, and the
two prompts carry identical minimum-count wording. Figure 4 uses full ARLEM
alone, to keep one schema per panel.

**Gemini 2.5 Flash-Lite is kept, with an explicit n.** It has 0 successful full-
ARLEM runs, 3 arlem_simple and 4 json_lab Objectives/Script. Dropping it would hide the
fact that it barely produced anything; plotting it unmarked would present a
4-run median as equal evidence to a 10-run one. Every figure carries a counts
row so the thin support is visible where the mark is.


The moon-lab reference, and its one limit
-----------------------------------------
``Artifacts/Data/Original Moon Lab/Processed/`` holds the hand-authored Unity
lab the schema was reverse-engineered from: 8 modules, 76 clips, 60 objects, 48
components, and 60 objects covered by just 6 distinct prefabs. It is a fair
baseline for *size* and *reuse*, and figures 1 and 2 draw it as a reference rule.

It is NOT a baseline for novelty. ``lab_metrics.KNOWN_PREFABS`` and
``KNOWN_TEXTURES`` were derived from this very file, so its novel-asset count is
0 by construction. No figure draws a "0 novel assets" reference line.


Why share, not count, is the headline on novel assets
-----------------------------------------------------
Absolute novel-asset counts track lab size: GPT-5.5 invents 16.5 prefabs while
building 51 objects; Gemini 3.1 Flash-Lite invents 1 while building 6. Ranking
models on the raw count mostly ranks them on verbosity. The share (novel /
distinct prefabs referenced) removes that confound, and is the actual finding --
it is 0.88 median, IQR 0.78-0.92, and near-flat across models, levels and
topics. Figure 2 leads with the share and keeps the count as the texture strip.

Novel *textures* are 46% zeros with a tail to 41, so panel 2b is a dot strip
rather than a box: a box would report a median of 0 for five models while GPT-5.5
sits at 12.5, and would hide that the zeros are a mode, not a floor.


Palettes (validated, do not hand-edit)
--------------------------------------
Provider hues come from ``figure_style`` unchanged. Two new ramps, both run
through the dataviz six-checks validator on white:

* level ordinal ``#6b9ed8 -> #123a75`` -- PASS (monotone L, light end 2.72:1)
* floor ordinal ``#86afe0 / #3b76bf / #14396f`` -- PASS (light end 2.22:1)
* validity status ``#4a3aa7 / #d03b3b`` -- PASS all-pairs, worst CVD dE 21.5

Figures whose fills already spend colour on level or status draw their provider
brackets in muted ink instead of provider hue, so the bracket stays grouping
chrome rather than implying a second encoding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath

from figure_style import (
    BAR_WIDTH, CORNER_PT, GRID, HATCH_COLOR, INK, INK_MUTED, INK_SOFT, LEVEL_NAME,
    MODEL_ORDER, PROVIDER_COLOR, PROVIDER_HATCH, SURFACE, draw_provider_brackets,
    rounded_bar,
    set_style,
)

ROOT = Path(__file__).resolve().parents[3]
RUNS_CSV = (ROOT / "Artifacts" / "Data" / "Benchmark" / "Reports"
            / "benchmark_runs_wave2_repriced.csv")
MOON_LAB = (ROOT / "Artifacts" / "Data" / "Original Moon Lab" / "Processed"
            / "Optomized_moon_lab_final.json")
FIG_DIR = ROOT / "Artifacts" / "Paper" / "figures"
TEX_DIR = ROOT / "Artifacts" / "Paper" / "tables"

# Ordered ramps -- see module docstring for validator results.
LEVEL_COLOR = {"L3": "#6b9ed8", "L4": "#123a75"}
# Pattern on the light step, so the two levels survive greyscale print
# independent of how the printer renders the lightness gap (see PROVIDER_HATCH).
LEVEL_HATCH = {"L3": "///", "L4": ""}
# Keys stay L1/L3/L4 -- that is what the `level` column holds. The printed name
# comes from figure_style.LEVEL_NAME; the gloss says what the prompt carried,
# since these two figures turn entirely on the difference between them.
LEVEL_LABEL = {"L3": f"{LEVEL_NAME['L3']}  — objectives, no script",
               "L4": f"{LEVEL_NAME['L4']}  — objectives + full script"}
# Three ordinal blues for degrees of compliance, plus the status red for an
# outright violation -- a run BELOW the stated minimum is a different kind of
# thing from a run that merely met it, so it does not belong on the ramp.
# Red separates from all three steps (worst all-pairs CVD dE 18.5).
FLOOR_COLOR = {"below": "#d03b3b", "at": "#14396f", "plus1": "#3b76bf",
               "above": "#86afe0"}
FLOOR_LABEL = {"below": "below the minimum", "at": "exactly the minimum",
               "plus1": "one above", "above": "two or more above"}
VALID_COLOR = {"renderable": "#4a3aa7", "empty": "#d03b3b"}
VALID_LABEL = {"renderable": "Activates \u22651 augmentation",
               "empty": "Activates nothing (renders empty)"}

# The minimum counts stated in every Objectives/Script ARLEM prompt, full and simplified
# alike (prompt_builder.build_prompt defaults; verified in the saved prompt
# artefacts under Artifacts/Data/Benchmark/prompts/).
ARLEM_FLOORS = [("num_things", 3, "things"),
                ("num_places", 2, "places"),
                ("num_actions", 5, "actions")]

REF_LINE = "hand-authored moon lab"

# Narrower than figure_style.BAR_WIDTH: these panels are full-width with only
# eleven slots, where the shared width renders as a very heavy bar.
SLIM_WIDTH = 0.50


# ---- Data -----------------------------------------------------------

def load_runs(runs_csv: Path) -> pd.DataFrame:
    """Every run in the CSV, successes and failures alike.

    Failures are kept so the counts rows can divide by attempts rather than by
    successes -- an earlier version filtered here and every ratio read `20/20`.
    Aggregation is always over ``df[df["ok"]]``.
    """
    df = pd.read_csv(runs_csv)
    df["ok"] = df["success"] == True  # noqa: E712 -- pandas mask, not identity
    return df


def json_lab_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Successful json_lab Objectives/Script runs, with the derived asset ratios.

    The Outline level is dropped here, not filtered per figure: LabOutline has no
    structure to measure, so every structural column is 0 for those runs.
    """
    jl = df[(df["spec_type"] == "json_lab") & (df["level"] != "L1")].copy()
    uniq = jl["num_unique_prefabs"].replace(0, np.nan)
    # Share of distinct prefabs the model invented, and how many objects each
    # distinct prefab had to cover. Both are undefined for a lab that
    # references no prefabs at all, hence the NaN rather than a 0.
    jl["novel_share"] = jl["novel_prefabs"] / uniq
    jl["reuse"] = jl["num_objects"] / uniq
    return jl


def arlem_slice(df: pd.DataFrame, full_only: bool = False) -> pd.DataFrame:
    specs = ["arlem"] if full_only else ["arlem", "arlem_simple"]
    return df[df["spec_type"].isin(specs)].copy()


def moon_lab_reference() -> dict:
    """Structural counts for the hand-authored lab, computed not hardcoded.

    Reaches into Code/Benchmark for the same analyser the benchmark runs, so
    the reference and the measurements can never drift apart.
    """
    sys.path.insert(0, str(ROOT / "Code" / "Benchmark"))
    from lab_metrics import analyze_json_lab  # noqa: E402

    m = analyze_json_lab(json.loads(MOON_LAB.read_text(encoding="utf-8")))
    ref = {k: m[k] for k in ("num_modules", "num_clips", "num_objects",
                             "num_components", "num_unique_prefabs")}
    ref["reuse"] = ref["num_objects"] / ref["num_unique_prefabs"]
    return ref


def model_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model in display order: identity, attempts, successes.

    `df` is an unfiltered slice, so `n_runs` counts attempts and `n_ok` counts
    the runs the figures actually aggregate over.
    """
    rows = []
    for display, tick, provider in MODEL_ORDER:
        runs = df[df["display_name"] == display]
        rows.append({"display": display, "tick": tick, "provider": provider,
                     "n_runs": len(runs), "n_ok": int(runs["ok"].sum())})
    return pd.DataFrame(rows)


# ---- Shared chrome --------------------------------------------------

def style_panel(ax, n_cols, grid_axis="y"):
    ax.set_axisbelow(True)
    if grid_axis == "y":
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.xaxis.grid(False)
    else:
        ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.yaxis.grid(False)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.set_xlim(-0.62, n_cols - 0.38)
    ax.set_xticks(range(n_cols))
    ax.tick_params(axis="y", labelsize=8.5, pad=2)


def counts_row(ax, tbl, y, label, values, weak_when, denom=None):
    """An `n/N` row under the x-axis, bolded where the count is notable.

    `values` is a list aligned to `tbl`; `weak_when` takes (value, n_runs) and
    decides which entries earn the bold treatment, so each figure foregrounds
    the count its own story turns on. `denom` defaults to attempts; pass
    `n_ok` where the question is about the runs that did produce output.
    """
    trans = ax.get_xaxis_transform()
    if denom is None:
        denom = tbl["n_runs"]
    for x, (val, n) in enumerate(zip(values, denom)):
        weak = weak_when(val, n)
        ax.text(x, y, f"{val}/{n}", transform=trans, ha="center", va="top",
                fontsize=7.5, color=INK_SOFT if weak else INK_MUTED,
                fontweight="bold" if weak else "normal", clip_on=False)
    # Right-aligned into the empty margin under the y tick labels; centred on
    # x=0 it would sit on top of the first column's value.
    ax.text(-0.78, y, label, transform=trans, ha="right", va="top",
            fontsize=7.5, color=INK_MUTED, style="italic", clip_on=False)


def reference_rule(ax, value, text, fontsize=7.5):
    """Dashed rule for the hand-authored lab, labelled just above its right end.

    The label sits inside the axes: hung off the right margin it overran the
    neighbouring panel in a multi-column layout.
    """
    ax.axhline(value, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=2)
    if text is None:  # the caller names the rule in a legend instead
        return
    ax.text(-0.55, value, " " + text, ha="left", va="bottom",
            fontsize=fontsize, color=INK_SOFT, zorder=6,
            path_effects=[pe.withStroke(linewidth=3.0, foreground=SURFACE)])


def titles(fig, title, sub, y=0.985, ysub=0.945):
    fig.suptitle(title, fontsize=11.5, fontweight="bold", color=INK,
                 x=0.055, ha="left", y=y)
    fig.text(0.055, ysub, sub, fontsize=8.5, color=INK_SOFT,
             ha="left", va="top")


def level_legend(fig, y=0.945):
    handles = [Patch(facecolor=LEVEL_COLOR[lv], hatch=LEVEL_HATCH[lv] or None,
                     edgecolor=HATCH_COLOR, linewidth=0, label=LEVEL_LABEL[lv])
               for lv in ("L3", "L4")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, y),
               ncols=2, frameon=False, fontsize=8.5, handletextpad=0.5,
               handlelength=1.8, handleheight=1.2,
               columnspacing=1.6)


def rounded_top_rect(ax, x, y0, y1, width, color):
    """A rectangle from y0 to y1 with rounded top corners.

    ``figure_style.rounded_bar`` is anchored to the baseline; a stacked column's
    top segment starts partway up, so it needs the same corner treatment at an
    arbitrary bottom. Radius is in points and converted per-axes, and clamped so
    a thin segment degrades to a square rather than a lozenge.
    """
    if y1 <= y0:
        return
    inv = ax.transData.inverted()
    r_px = CORNER_PT * ax.figure.dpi / 72
    p0, p1 = inv.transform((0, 0)), inv.transform((r_px, r_px))
    rx = min(abs(p1[0] - p0[0]), width / 2)
    ry = min(abs(p1[1] - p0[1]), (y1 - y0) / 2)

    left, right = x - width / 2, x + width / 2
    verts = [(left, y0), (left, y1 - ry),
             (left, y1), (left + rx, y1),
             (right - rx, y1),
             (right, y1), (right, y1 - ry),
             (right, y0), (left, y0)]
    codes = [MplPath.MOVETO, MplPath.LINETO,
             MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO,
             MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                           edgecolor="none", zorder=3))


def stacked_column(ax, x, parts, width, gap_data):
    """Draw a stacked column bottom-up with a 2px surface gap between fills.

    ``parts`` is [(value, color), ...] bottom to top. Segment tops stay on the
    true cumulative value; the gap is shaved off the visible fill only, so the
    stack still reads against the axis. Only the topmost non-empty segment gets
    rounded corners -- a rounded interior segment would read as a bar end.
    """
    top_i = max((i for i, (v, _) in enumerate(parts) if v > 0), default=None)
    running = 0.0
    for i, (val, color) in enumerate(parts):
        if val <= 0:
            continue
        y0, y1 = running, running + val
        running = y1
        if i == top_i:
            rounded_top_rect(ax, x, y0, y1, width, color)
        else:
            ax.bar(x, max(y1 - gap_data, y0) - y0, bottom=y0, width=width,
                   color=color, edgecolor="none", zorder=3)


def gap_in_data_units(ax, px=2.0):
    """Convert the 2px surface gap between stacked fills into y data units."""
    inv = ax.transData.inverted()
    dpi_px = px * ax.figure.dpi / 96
    return abs(inv.transform((0, dpi_px))[1] - inv.transform((0, 0))[1])


# ---- Figure 1: json_lab size vs the hand-authored lab ----------------

A_PANELS = [("num_modules", "Modules (scenes)"),
            ("num_clips", "Clips (steps)"),
            ("num_objects", "Scene objects"),
            ("num_components", "Behaviour components")]

# Every text size in the figure, in one place for tuning legibility.
A_FONT = {"suptitle": 18, "legend": 14, "ylabel": 15, "yticks": 13,
          "xticks": 13, "zero": 11, "brackets": 14}


def fig_json_lab_size(df: pd.DataFrame) -> plt.Figure:
    jl = json_lab_slice(df)
    tbl = model_rows(jl)
    ref = moon_lab_reference()
    n = len(tbl)

    # Full-width stacked panels, not a 2x2 grid: eleven two-line model labels
    # plus three provider brackets do not fit under a half-width axes.
    fig, axes = plt.subplots(len(A_PANELS), 1, figsize=(10.4, 11.0), sharex=True)
    for ax, (col, label) in zip(axes, A_PANELS):
        style_panel(ax, n)
        # Paired bars: pooling the two levels would report the midpoint of a
        # bimodal distribution, since Script roughly doubles every count here.
        for x, row in enumerate(tbl.itertuples()):
            for k, lv in enumerate(("L3", "L4")):
                runs = jl[jl["ok"] & (jl["display_name"] == row.display)
                          & (jl["level"] == lv)]
                if runs.empty:
                    continue
                med = runs[col].median()
                bx = x + (k - 0.5) * 0.34
                rounded_bar(ax, bx, med, 0.31, LEVEL_COLOR[lv],
                            hatch=LEVEL_HATCH[lv])
                if med == 0:
                    # No bar can be drawn at zero; say so rather than leaving a
                    # gap that reads as a failed run.
                    # Ink, not the series colour: position already says which
                    # level it is, and text never carries series identity.
                    ax.text(bx, 0, "0", ha="center", va="bottom",
                            fontsize=A_FONT["zero"], color=INK_MUTED)
        # Named in the legend: an in-panel label collided with the bars.
        reference_rule(ax, ref[col], None)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))
        ax.set_ylabel(label, fontsize=A_FONT["ylabel"])
        ax.tick_params(axis="y", labelsize=A_FONT["yticks"])
        top = max(ref[col],
                  jl.groupby(["display_name", "level"])[col].median().max())
        ax.set_ylim(0, top * 1.2)

    axes[-1].set_xticklabels(tbl["tick"], fontsize=A_FONT["xticks"])
    draw_provider_brackets(axes[-1], tbl, -0.27, -0.32, rule_color=INK_MUTED,
                           fontsize=A_FONT["brackets"])

    fig.suptitle("Generated lab size against the hand-authored reference",
                 fontsize=A_FONT["suptitle"], fontweight="bold", color=INK,
                 x=0.5, ha="center", y=0.995)
    handles = [Patch(facecolor=LEVEL_COLOR[lv], hatch=LEVEL_HATCH[lv] or None,
                     edgecolor=HATCH_COLOR, linewidth=0, label=LEVEL_LABEL[lv])
               for lv in ("L3", "L4")]
    handles.append(plt.Line2D([], [], color=INK_SOFT, linewidth=1.4,
                              linestyle=(0, (4, 3)), label=REF_LINE))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncols=3, frameon=False, fontsize=A_FONT["legend"],
               handletextpad=0.5, handlelength=1.8, handleheight=1.2,
               columnspacing=2.4)
    # Set by hand so the legend sits just above the first panel.
    fig.subplots_adjust(left=0.1, right=0.99, top=0.925, bottom=0.09,
                        hspace=0.18)
    return fig


# ---- Figure 2: authoring burden -------------------------------------

# Every text size in the figure, in one place for tuning legibility.
B_FONT = {"suptitle": 18, "ylabel": 15, "yticks": 13, "xticks": 13,
          "values": 12.5, "brackets": 14}


def fig_authoring_burden(df: pd.DataFrame) -> plt.Figure:
    # Novel textures only. The prefab share is near-flat across models (the
    # paper text carries it), and prefab reuse was cut for legibility.
    # Bars are means, not medians: 46% of runs invent no texture, so a median
    # reads 0 for five models and flattens the difference between them.
    jl = json_lab_slice(df)
    tbl = model_rows(jl)
    n = len(tbl)

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    style_panel(ax, n)
    means = []
    for x, row in enumerate(tbl.itertuples()):
        vals = jl[jl["ok"] & (jl["display_name"] == row.display)]["novel_textures"]
        means.append(vals.mean() if len(vals) else np.nan)
        if not len(vals):
            continue
        rounded_bar(ax, x, vals.mean(), SLIM_WIDTH,
                    PROVIDER_COLOR[row.provider],
                    hatch=PROVIDER_HATCH[row.provider])
    top = np.nanmax(means)
    for x, m in enumerate(means):
        if m == m:
            ax.text(x, m + top * 0.02, f"{m:.1f}", ha="center", va="bottom",
                    fontsize=B_FONT["values"], color=INK)
    ax.set_ylim(0, top * 1.14)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylabel("Textures invented\n(mean per run)", fontsize=B_FONT["ylabel"])
    ax.tick_params(axis="y", labelsize=B_FONT["yticks"])
    ax.set_xticklabels(tbl["tick"], fontsize=B_FONT["xticks"])
    draw_provider_brackets(ax, tbl, -0.14, -0.165,
                           fontsize=B_FONT["brackets"])

    fig.suptitle("Amount of New Texture Names Created",
                 fontsize=B_FONT["suptitle"], fontweight="bold", color=INK,
                 x=0.5, ha="center", y=0.99)
    # Set by hand: tight_layout left a wide band under the title.
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.2)
    return fig


# ---- Figure 3: ARLEM activity validity ------------------------------

def fig_arlem_validity(df: pd.DataFrame) -> plt.Figure:
    ar = arlem_slice(df)
    tbl = model_rows(ar)
    n = len(tbl)

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    style_panel(ax, n)
    ax.set_ylim(0, 1.0)
    gap = gap_in_data_units(ax)

    rend_counts, med_act = [], []
    for x, row in enumerate(tbl.itertuples()):
        runs = ar[ar["ok"] & (ar["display_name"] == row.display)]
        rend = int((runs["total_activates"] > 0).sum())
        rend_counts.append(rend)
        med_act.append(runs["total_activates"].median() if len(runs) else np.nan)
        if not len(runs):
            continue
        frac = rend / len(runs)
        stacked_column(ax, x, [(frac, VALID_COLOR["renderable"]),
                               (1 - frac, VALID_COLOR["empty"])],
                       SLIM_WIDTH, gap)

    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Share of successful ARLEM runs", fontsize=9)
    ax.set_xticklabels(tbl["tick"], fontsize=8.5)

    counts_row(ax, tbl, -0.155, "runs activating ≥1",
               rend_counts, lambda v, n_: v == 0, denom=tbl["n_ok"])
    trans = ax.get_xaxis_transform()
    for x, v in enumerate(med_act):
        ax.text(x, -0.225, "\u2014" if not v == v else f"{v:.0f}", transform=trans,
                ha="center", va="top", fontsize=7.5,
                color=INK_SOFT if v == 0 else INK_MUTED,
                fontweight="bold" if v == 0 else "normal", clip_on=False)
    ax.text(-0.78, -0.225, "median activated", transform=trans,
            ha="right", va="top", fontsize=7.5, color=INK_MUTED, style="italic",
            clip_on=False)
    draw_provider_brackets(ax, tbl, -0.305, -0.345, rule_color=INK_MUTED)

    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=7,
                          markerfacecolor=VALID_COLOR[k], markeredgecolor="none",
                          label=VALID_LABEL[k]) for k in ("renderable", "empty")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 1.0),
               ncols=2, frameon=False, fontsize=8.5, handletextpad=0.5,
               columnspacing=1.6)

    titles(fig, "ARLEM activities that would render nothing",
           "All successful runs of both ARLEM specifications. A scenario whose "
           "action flows never activate an augmentation is schema-valid and "
           "functionally empty.\nEvery OpenAI model does this in every run "
           "(74/74, both schema variants); every other provider fills the same "
           "field.",
           ysub=0.955)
    fig.tight_layout(rect=(0.03, 0.155, 0.99, 0.885))
    return fig


# ---- Figure 4: specificity elasticity -------------------------------

# The two formats name their parts differently, so the figure compares only
# the parts that correspond, row-aligned across the panels. ``None`` holds a
# slot with no counterpart.
#
# ARLEM rows count what the actions *show*, not what the workplace declares.
# The workplace ``primitives`` list is a menu of five media types that every
# run fills near-completely, so it cannot register growth; an augmentation
# reaches the learner only when an action's enter/exit flow activates it, as
# either a predicate (a wrapped, reusable asset) or a primitive used directly.
# Both routes are counted, by the media type they resolve to. These counts are
# not in the run CSV -- see ``add_activation_counts``.
D_ROWS = [
    # (json_lab column, label)            (arlem column, label)
    (("num_objects", "objects"),         ("shown_all", "augmentations shown")),
    (None,                               ("shown_animation", "  animations")),
    (("num_clips", "clips"),             ("num_actions", "action steps")),
]

# Every text size in the figure, in one place for tuning legibility.
D_FONT = {"suptitle": 16, "legend": 12, "panel": 13, "rows": 12,
          "ratio": 12, "ratio_header": 9.5, "xticks": 10.5, "xlabel": 11}

METRICS_DIR = ROOT / "Artifacts" / "Data" / "Benchmark" / "Metrics"
OUTPUTS_DIR = ROOT / "Artifacts" / "Data" / "Benchmark" / "Outputs"


def add_activation_counts(ar: pd.DataFrame) -> pd.DataFrame:
    """Per-run counts of what ARLEM actions activate, read from saved outputs.

    ``shown_all`` counts every activation of a predicate, primitive or warning
    (activations of type ``action`` launch another step and show nothing).
    ``shown_animation`` splits out those whose media type is animation (how 3D
    models are displayed): a predicate resolves through its workplace ``type``,
    a direct primitive activation is its own type. Runs are matched to their
    output file through the metrics record, on (model, timestamp).
    """
    counts = {}
    for m in METRICS_DIR.glob("*_arlem*_metrics.json"):
        rec = json.loads(m.read_text(encoding="utf-8"))
        out = OUTPUTS_DIR / (m.name[: -len("_metrics.json")] + "_output.json")
        if not rec.get("success") or not out.exists():
            continue
        spec = json.loads(out.read_text(encoding="utf-8"))
        pred_type = {p.get("id"): p.get("type") for p in
                     (spec.get("workplace") or {}).get("predicates") or []}
        n = {"shown_all": 0, "shown_animation": 0}
        for action in (spec.get("activity") or {}).get("actions") or []:
            for flow in ("enter", "exit"):
                for a in (action.get(flow) or {}).get("activates") or []:
                    kind = a.get("type")
                    if kind == "action":
                        continue
                    n["shown_all"] += 1
                    media = (pred_type.get(a.get("augmentation"))
                             if kind == "predicate" else a.get("augmentation"))
                    if media == "animation":
                        n["shown_animation"] += 1
        counts[(rec["model"], rec["timestamp"])] = n
    keys = list(zip(ar["model"], ar["timestamp"]))
    for col in ("shown_all", "shown_animation"):
        ar[col] = [counts.get(k, {}).get(col, np.nan) for k in keys]
    return ar


def _level_mean(data, level, col):
    return data[data["ok"] & (data["level"] == level)][col].mean()


def _elasticity_panel(ax, data, rows, title):
    """Objectives-to-Script dumbbells on a linear axis, one per row slot."""
    for y, row in enumerate(rows):
        if row is None:
            continue
        col, _ = row
        lo, hi = _level_mean(data, "L3", col), _level_mean(data, "L4", col)
        ax.plot([lo, hi], [y, y], color=INK_MUTED, linewidth=1.6, zorder=2,
                solid_capstyle="round")
        for val, lv in ((lo, "L3"), (hi, "L4")):
            ax.plot(val, y, marker="o", markersize=10, zorder=3,
                    markerfacecolor=LEVEL_COLOR[lv], markeredgecolor=SURFACE,
                    markeredgewidth=1.5, linestyle="none")
        ratio = hi / lo if lo else np.nan
        ax.text(1.03, y, f"{ratio:.2f}×", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=D_FONT["ratio"], color=INK,
                clip_on=False)
    ax.text(1.03, -0.55, "change\nObjectives\n→ Script",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=D_FONT["ratio_header"], color=INK_MUTED, style="italic",
            clip_on=False)

    ax.set_yticks([y for y, r in enumerate(rows) if r is not None])
    ax.set_yticklabels([r[1] for r in rows if r is not None],
                       fontsize=D_FONT["rows"])
    ax.tick_params(axis="y", length=0, labelleft=True)
    ax.tick_params(axis="x", labelsize=D_FONT["xticks"])
    ax.set_xlabel("mean count across generations", fontsize=D_FONT["xlabel"])
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.yaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(title, fontsize=D_FONT["panel"], fontweight="bold", color=INK, loc="left",
                 pad=10)


def fig_specificity_elasticity(df: pd.DataFrame) -> plt.Figure:
    # Not sharey: a shared y axis shares tick labels too, and each panel names
    # its rows in its own format's terms. The limits are set on both instead.
    fig, (ax_j, ax_a) = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    _elasticity_panel(ax_j, json_lab_slice(df), [r[0] for r in D_ROWS],
                      "Lab JSON")
    ar = add_activation_counts(arlem_slice(df, full_only=True))
    ok = ar[ar["ok"] & ar["level"].isin(["L3", "L4"])]
    assert ok["shown_all"].notna().all(), "ARLEM run without a matching output"
    _elasticity_panel(ax_a, ar,
                      [r[1] for r in D_ROWS], "ARLEM")
    # Shared linear axis, so a bar's length compares across the two formats.
    ax_j.set_xlim(0, 45)
    for ax in (ax_j, ax_a):
        ax.set_ylim(len(D_ROWS) - 0.4, -0.6)
        # Rule off the sequence row from the scene-content rows above it.
        ax.axhline(len(D_ROWS) - 1.5, color=GRID, linewidth=0.8, zorder=1)

    fig.suptitle("Changes in Output Caused by Increased Detail",
                 fontsize=D_FONT["suptitle"], fontweight="bold", color=INK,
                 x=0.5, ha="center", y=0.99)
    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=12,
                          markerfacecolor=LEVEL_COLOR[lv],
                          markeredgecolor=SURFACE, label=LEVEL_LABEL[lv])
               for lv in ("L3", "L4")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncols=2, frameon=False, fontsize=D_FONT["legend"],
               handletextpad=0.4, columnspacing=2.4)
    fig.tight_layout(rect=(0.0, 0.0, 0.95, 0.86), w_pad=1.0)
    return fig


# ---- Figure 5: ARLEM floor-hugging ----------------------------------

def fig_arlem_floor(df: pd.DataFrame) -> plt.Figure:
    ar = arlem_slice(df)
    tbl = model_rows(ar)
    n = len(tbl)

    fig, axes = plt.subplots(3, 1, figsize=(10.4, 8.8), sharex=True)
    for ax, (col, floor, name) in zip(axes, ARLEM_FLOORS):
        style_panel(ax, n)
        ax.set_ylim(0, 1.0)
        gap = gap_in_data_units(ax)
        for x, row in enumerate(tbl.itertuples()):
            runs = ar[ar["ok"] & (ar["display_name"] == row.display)][col]
            if runs.empty:
                continue
            # The four buckets must partition the runs -- an earlier version
            # omitted "below" and its columns did not reach 100%.
            parts = [((runs < floor).mean(), FLOOR_COLOR["below"]),
                     ((runs == floor).mean(), FLOOR_COLOR["at"]),
                     ((runs == floor + 1).mean(), FLOOR_COLOR["plus1"]),
                     ((runs > floor + 1).mean(), FLOOR_COLOR["above"])]
            assert abs(sum(v for v, _ in parts) - 1.0) < 1e-9, (col, row.display)
            stacked_column(ax, x, parts, SLIM_WIDTH, gap)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "50%", "100%"])
        ax.set_ylabel(f"{name}\nprompt asks for ≥{floor}", fontsize=9)
        ok = ar[ar["ok"]]
        overall = (ok[col] == floor).mean()
        ax.text(n - 0.42, 0.5, f"  {overall:.0%} of runs\n  stop at {floor}",
                ha="left", va="center", fontsize=7.5, color=INK_SOFT,
                clip_on=False)

    axes[-1].set_xticklabels(tbl["tick"], fontsize=8.5)
    counts_row(axes[-1], tbl, -0.155, "runs succeeded", list(tbl["n_ok"]),
               lambda v, n_: v < n_)
    draw_provider_brackets(axes[-1], tbl, -0.30, -0.34, rule_color=INK_MUTED)

    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=7,
                          markerfacecolor=FLOOR_COLOR[k], markeredgecolor="none",
                          label=FLOOR_LABEL[k])
               for k in ("below", "at", "plus1", "above")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 1.0),
               ncols=4, frameon=False, fontsize=8.5, handletextpad=0.5,
               columnspacing=1.6)

    titles(fig, "ARLEM output stops at the number the prompt asks for",
           "All successful runs of both ARLEM specifications, which carry "
           "identical minimum-count wording.\n"
           "Models satisfy the constraint and stop: 78% of runs produce exactly "
           "two places and 56% exactly three things. The json_lab prompt states "
           "its minimums the same way and gets no such effect.",
           y=0.99, ysub=0.955)
    fig.tight_layout(rect=(0.03, 0.07, 0.99, 0.90))
    return fig


# ---- LaTeX ----------------------------------------------------------

FIGURE_BLOCKS = [
    ("wave2_json_lab_size", "fig:json-lab-size",
     "Size of the generated Lab JSON against the hand-authored moon lab the "
     "schema was derived from. Bars are per-model medians over the successful "
     "Objectives and Script runs, shown separately because the detailed script "
     "roughly doubles every count. The dashed rule is the reference lab. Models "
     "come closest on objects and modules and fall furthest short on clips: the "
     "reference lab's 76 steps are more than double the best model median of "
     "35. The Outline level is excluded throughout \\textemdash{} it returns a "
     "\\texttt{LabOutline}, "
     "which has no modules, clips, objects or components to count."),
    ("wave2_authoring_burden", "fig:authoring-burden",
     "Novel textures per generated lab: textures a run references that do not "
     "exist in the documented moon-lab library, and which someone would have "
     "to paint. Bars are per-model means over the successful Objectives and "
     "Script Lab JSON runs; means rather than medians because 46\\% of runs "
     "invent none, and a median would read zero for five models. "
     "Gemini~2.5~Flash-Lite has only 4 successful runs."),
    ("wave2_arlem_validity", "fig:arlem-validity",
     "ARLEM scenarios whose action flows never activate an augmentation. Such "
     "a scenario passes schema validation and cross-validation, and renders "
     "nothing. Every OpenAI model produces this in every run \\textemdash{} 74 of "
     "74 across both the full and the simplified ARLEM schema, with "
     "\\texttt{enter.activates} and \\texttt{exit.activates} empty on every "
     "action \\textemdash{} while Anthropic and Google models populate the same "
     "field, at a median of 5 to 26 activations per run. The defect is "
     "invisible to success rate, token cost and generation time alike."),
    ("wave2_specificity_elasticity", "fig:specificity-elasticity",
     "Growth from the Objectives level (learning objectives, no script) to the "
     "Script level (plus the full lesson "
     "script), pooled over all models, for the parts of each format that "
     "correspond. ARLEM rows count what the action flows activate \\textemdash{} "
     "predicates and directly used primitives alike, split by media type "
     "\\textemdash{} rather than what the workplace declares. Counts are means "
     "on a shared linear axis. The script roughly doubles every Lab JSON count "
     "(objects $2.05\\times$, clips $1.98\\times$). In ARLEM it grows the "
     "augmentations shown by $1.67\\times$, driven by animations, the route "
     "for 3D models ($1.86\\times$), while action steps grow only "
     "$1.33\\times$. ARLEM means include the OpenAI runs, "
     "which activate nothing (34 of 90), so they understate what the other "
     "models show but not how it grows."),
    ("wave2_arlem_floor", "fig:arlem-floor",
     "How often a generated ARLEM scenario contains exactly the minimum the "
     "prompt asked for. Both ARLEM specifications carry identical wording "
     "(\\emph{at least 3 things}, \\emph{at least 2 places}, \\emph{at least 5 "
     "sequential actions}), so the runs are pooled. Models satisfy the "
     "constraint and stop: 78\\% of runs produce exactly two places and 56\\% "
     "exactly three things, against 23\\% at the action floor. Red marks runs "
     "that fall \\emph{below} the stated minimum, which happens only for "
     "Gemini~3.1~Flash-Lite (7 runs with fewer than three things, one with a "
     "single place). The Lab JSON prompt states its minimums the same way and "
     "produces no comparable effect, with 2\\% of runs at the object floor and "
     "9\\% at the clip floor \\textemdash{} so the behaviour tracks the target "
     "schema, not the prompt style."),
]


def _tex_table(caption, label, colspec, header, body_rows, note=None):
    L = ["\\begin{table}[htbp]", "  \\centering", "  \\small",
         f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
         f"  \\begin{{tabular}}{{{colspec}}}", "    \\toprule",
         "    " + header, "    \\midrule"]
    L += ["    " + r for r in body_rows]
    L += ["    \\bottomrule", "  \\end{tabular}"]
    if note:
        L.append(f"  \\par\\smallskip\\footnotesize {note}")
    L.append("\\end{table}")
    return L


def write_tex(df: pd.DataFrame, suffix: str, out: Path) -> None:
    jl = json_lab_slice(df)
    ar = arlem_slice(df)
    ref = moon_lab_reference()
    tbl = model_rows(jl)
    ar_tbl = model_rows(ar)

    def esc(s):
        return s.replace("&", "\\&")

    # -- json_lab structure table --
    jl_rows, prev = [], None
    for row in tbl.itertuples():
        if prev is not None and row.provider != prev:
            jl_rows.append("\\addlinespace")
        prev = row.provider
        r = jl[jl["ok"] & (jl["display_name"] == row.display)]
        if r.empty:
            jl_rows.append(f"{esc(row.display)} & 0/{row.n_runs} & \\multicolumn{{7}}{{c}}{{"
                           "\\itshape no successful runs} \\\\")
            continue
        jl_rows.append(
            f"{esc(row.display)} & {len(r)}/{row.n_runs} & {r['num_modules'].median():.0f} & "
            f"{r['num_clips'].median():.0f} & {r['num_objects'].median():.0f} & "
            f"{r['num_components'].median():.0f} & "
            f"{r['novel_prefabs'].median():.0f} & "
            f"{r['novel_share'].median() * 100:.0f}\\% & "
            f"{r['reuse'].median():.1f} \\\\")
    jl_rows.append("\\addlinespace")
    jl_rows.append(
        f"\\itshape {REF_LINE} & \\itshape --- & \\itshape {ref['num_modules']} & "
        f"\\itshape {ref['num_clips']} & \\itshape {ref['num_objects']} & "
        f"\\itshape {ref['num_components']} & \\itshape 0 & \\itshape 0\\% & "
        f"\\itshape {ref['reuse']:.1f} \\\\")

    # -- ARLEM structure table --
    ar_rows, prev = [], None
    for row in ar_tbl.itertuples():
        if prev is not None and row.provider != prev:
            ar_rows.append("\\addlinespace")
        prev = row.provider
        r = ar[ar["ok"] & (ar["display_name"] == row.display)]
        if r.empty:
            ar_rows.append(f"{esc(row.display)} & 0/{row.n_runs} & \\multicolumn{{7}}{{c}}{{"
                           "\\itshape no successful runs} \\\\")
            continue
        rend = (r["total_activates"] > 0).mean() * 100
        ar_rows.append(
            f"{esc(row.display)} & {len(r)}/{row.n_runs} & {r['num_things'].median():.0f} & "
            f"{r['num_places'].median():.0f} & {r['num_detectables'].median():.0f} & "
            f"{r['num_actions'].median():.0f} & {r['total_triggers'].median():.0f} & "
            f"{r['total_pois'].median():.0f} & {rend:.0f}\\% \\\\")

    L = [
        "% Output-structure tables and figures -- wave-2 benchmark.",
        "% GENERATED by Code/Analysis/figures/structure_figures.py -- edit that, not this.",
        "%",
        "% LEVELS. Outline / Objectives / Script are the paper-facing names for the",
        "% L1 / L3 / L4 keys the run CSVs still store.",
        "%",
        "% SLICES. json_lab numbers are the 104 successful Objectives and Script",
        "% runs. The Outline level is",
        "% excluded because it returns the spec-agnostic LabOutline, which has no",
        "% modules, clips, objects or components -- all 55 Outline rows record",
        "% structural zeros and would halve every median. ARLEM numbers pool the",
        "% full and simplified schemas (192 successful runs), which is valid only",
        "% for fields both schemas define; persons, sensors, devices, apps,",
        "% warnings, messages and voice/sensor triggers exist in the full schema",
        "% only, so a zero there is a schema fact and no table reports one.",
        "%",
        "% REFERENCE. The moon-lab row is the hand-authored Unity lab the schema",
        "% was reverse-engineered from, measured with the same analyser. Its 0",
        "% novel assets are true BY CONSTRUCTION -- lab_metrics.KNOWN_PREFABS and",
        "% KNOWN_TEXTURES were derived from this file -- so the row is a valid",
        "% baseline for size and reuse but NOT for novelty. Do not cite it as",
        "% evidence that a human author invents fewer assets.",
        "%",
        "% Gemini 2.5 Flash-Lite has 0 successful full-ARLEM runs, 3 simplified",
        "% and 4 json_lab Objectives/Script. Its n column is the caveat; medians over 3-4",
        "% runs are not comparable to medians over 10.",
        "%",
        "% Required packages:",
        "%   \\usepackage{booktabs}",
        "%   \\usepackage{graphicx}",
        "%",
        "% Regenerate with:",
        "%   python \"Code/Analysis/figures/structure_figures.py\"",
        "",
        "% " + "-" * 73,
        "% Per-model Lab JSON structure",
        "% " + "-" * 73,
    ]
    L += _tex_table(
        "Structure of the generated Lab JSON, by model. Values are medians over "
        "the successful Objectives and Script runs; \\emph{novel} counts prefabs "
        "absent from the "
        "documented asset library, and \\emph{reuse} is scene objects per distinct "
        "prefab.",
        "tab:json-lab-structure",
        "lrrrrrrrr",
        "\\textbf{Model} & \\textbf{Runs} & \\textbf{Mod.} & \\textbf{Clips} & "
        "\\textbf{Obj.} & \\textbf{Comp.} & \\textbf{Novel} & "
        "\\textbf{Novel \\%} & \\textbf{Reuse} \\\\",
        jl_rows,
        note="The reference lab's 0 novel assets are true by construction: the "
             "known-asset lists were derived from it. Compare the size and reuse "
             "columns, not the novelty columns.")
    L += [
        "",
        "% " + "-" * 73,
        "% Per-model ARLEM structure",
        "% " + "-" * 73,
    ]
    L += _tex_table(
        "Structure of the generated ARLEM scenarios, by model, pooling the full "
        "and simplified schemas. Values are medians over successful runs. "
        "\\emph{Renders} is the share of runs whose action flows activate at "
        "least one augmentation; a run at 0\\% is schema-valid and displays "
        "nothing.",
        "tab:arlem-structure",
        "lrrrrrrrr",
        "\\textbf{Model} & \\textbf{Runs} & \\textbf{Things} & \\textbf{Places} & "
        "\\textbf{Detect.} & \\textbf{Actions} & \\textbf{Trig.} & "
        "\\textbf{POIs} & \\textbf{Renders} \\\\",
        ar_rows)

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
    "size": ("wave2_json_lab_size", fig_json_lab_size),
    "assets": ("wave2_authoring_burden", fig_authoring_burden),
    "validity": ("wave2_arlem_validity", fig_arlem_validity),
    "elasticity": ("wave2_specificity_elasticity", fig_specificity_elasticity),
    "floor": ("wave2_arlem_floor", fig_arlem_floor),
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", nargs="+", default=["all"],
                    choices=["all", *BUILDERS])
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--runs-csv", type=Path, default=RUNS_CSV,
                    help="Run-level CSV to plot (default: the repriced wave-2 "
                         "export; pricing is irrelevant to structure, but one "
                         "source keeps the run set identical to the cost and "
                         "timing figures).")
    ap.add_argument("--suffix", default="",
                    help="Appended to output filenames, e.g. --suffix _alt.")
    ap.add_argument("--no-tex", action="store_true",
                    help="Skip the LaTeX snippet file.")
    args = ap.parse_args()

    runs_csv = args.runs_csv if args.runs_csv.is_absolute() else ROOT / args.runs_csv
    df = load_runs(runs_csv)

    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)

    wanted = list(BUILDERS) if "all" in args.figures else args.figures
    for key in wanted:
        stem, builder = BUILDERS[key]
        fig = builder(df)
        for ext in args.formats:
            path = FIG_DIR / f"{stem}{args.suffix}.{ext}"
            fig.savefig(path, dpi=400 if ext == "png" else None,
                        bbox_inches="tight")
            print(f"wrote {path.relative_to(ROOT)}")
        plt.close(fig)

    if not args.no_tex:
        write_tex(df, args.suffix, TEX_DIR / f"structure_tables{args.suffix}.tex")


if __name__ == "__main__":
    main()
