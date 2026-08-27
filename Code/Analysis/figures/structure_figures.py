"""
Output-structure figures: what the generated labs actually contain.

Everything upstream of this file measures the *cost* of generation -- tokens,
dollars, seconds. These five figures measure the *artefact*: how big the
generated specifications are, how much new authoring work they imply, and
whether they would function at all.

Five figures, two LaTeX tables, written to Artifacts/Paper/.

    wave2_json_lab_size          size against the human-authored moon lab
    wave2_authoring_burden       novel assets + prefab reuse collapse
    wave2_arlem_validity         the OpenAI zero-activate defect
    wave2_specificity_elasticity how much L4's script buys, per spec family
    wave2_arlem_floor            how often models stop at the stated minimum

Regenerate:
    python "Code/Analysis/figures/structure_figures.py"


Slices, and why they are not optional
-------------------------------------
**L1 is excluded from every structural figure.** L1 returns the spec-agnostic
``LabOutline``, which has no modules, clips, objects or components -- so all 55
L1 rows record structural zeros. They are 35% of the json_lab rows and would
halve every median. The json_lab slice is therefore the 104 successful L3/L4
runs, matching the cost and timing figures.

**arlem and arlem_simple are pooled only where the schemas agree.** The
simplified schema has no persons, sensors, devices, apps, warnings, messages,
or voice/sensor triggers, so a zero in those columns is a schema fact, not a
model behaviour. Figures 3 and 5 pool the two specs because the fields they
read (``activates``, things/places/actions) exist identically in both, and the
two prompts carry identical minimum-count wording. Figure 4 uses full ARLEM
alone, since it walks fields the simplified schema drops.

**Gemini 2.5 Flash-Lite is kept, with an explicit n.** It has 0 successful full-
ARLEM runs, 3 arlem_simple and 4 json_lab L3/L4. Dropping it would hide the
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

from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from figure_style import (
    BAR_WIDTH, CORNER_PT, GRID, INK, INK_MUTED, INK_SOFT, MODEL_ORDER,
    PROVIDER_COLOR, SURFACE, draw_provider_brackets, rounded_bar, set_style,
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
LEVEL_LABEL = {"L3": "L3  \u2014 objectives only", "L4": "L4  \u2014 + full script"}
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

# The minimum counts stated in every L3/L4 ARLEM prompt, full and simplified
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
    """Successful json_lab L3/L4 runs, with the two derived asset ratios.

    L1 is dropped here, not filtered per figure: LabOutline has no structure to
    measure, so every structural column is 0 for those runs.
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


def reference_rule(ax, value, text):
    """Dashed rule for the hand-authored lab, labelled just above its right end.

    The label sits inside the axes: hung off the right margin it overran the
    neighbouring panel in a multi-column layout.
    """
    ax.axhline(value, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=2)
    ax.text(-0.55, value, " " + text, ha="left", va="bottom",
            fontsize=7.5, color=INK_SOFT, zorder=6,
            path_effects=[pe.withStroke(linewidth=3.0, foreground=SURFACE)])


def titles(fig, title, sub, y=0.985, ysub=0.945):
    fig.suptitle(title, fontsize=11.5, fontweight="bold", color=INK,
                 x=0.055, ha="left", y=y)
    fig.text(0.055, ysub, sub, fontsize=8.5, color=INK_SOFT,
             ha="left", va="top")


def level_legend(fig, y=0.945):
    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=7,
                          markerfacecolor=LEVEL_COLOR[lv], markeredgecolor="none",
                          label=LEVEL_LABEL[lv]) for lv in ("L3", "L4")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, y),
               ncols=2, frameon=False, fontsize=8.5, handletextpad=0.5,
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
        # Paired bars: pooling L3 and L4 would report the midpoint of a
        # bimodal distribution, since L4 roughly doubles every count here.
        for x, row in enumerate(tbl.itertuples()):
            for k, lv in enumerate(("L3", "L4")):
                runs = jl[jl["ok"] & (jl["display_name"] == row.display)
                          & (jl["level"] == lv)]
                if runs.empty:
                    continue
                med = runs[col].median()
                bx = x + (k - 0.5) * 0.34
                rounded_bar(ax, bx, med, 0.31, LEVEL_COLOR[lv])
                if med == 0:
                    # No bar can be drawn at zero; say so rather than leaving a
                    # gap that reads as a failed run.
                    # Ink, not the series colour: position already says which
                    # level it is, and text never carries series identity.
                    ax.text(bx, 0, "0", ha="center", va="bottom", fontsize=7,
                            color=INK_MUTED)
        reference_rule(ax, ref[col], REF_LINE)
        ax.set_ylabel(label, fontsize=9)
        top = max(ref[col],
                  jl.groupby(["display_name", "level"])[col].median().max())
        ax.set_ylim(0, top * 1.16)

    axes[-1].set_xticklabels(tbl["tick"], fontsize=8.5)
    counts_row(axes[-1], tbl, -0.155, "runs succeeded", list(tbl["n_ok"]),
               lambda v, n_: v < n_)
    draw_provider_brackets(axes[-1], tbl, -0.30, -0.34, rule_color=INK_MUTED)

    titles(fig, "Generated lab size against the hand-authored reference",
           "Median over successful L3/L4 runs per model. The reference lab is "
           "the hand-authored Unity moon lab the schema was derived from.\n"
           "Models come closest on objects and modules and fall furthest short "
           "on clips — the reference lab's 76 steps are more than double the "
           "best model median.",
           y=0.99, ysub=0.965)
    level_legend(fig, y=0.998)
    fig.tight_layout(rect=(0.03, 0.055, 0.99, 0.935))
    return fig


# ---- Figure 2: authoring burden -------------------------------------

def fig_authoring_burden(df: pd.DataFrame) -> plt.Figure:
    jl = json_lab_slice(df)
    tbl = model_rows(jl)
    ref = moon_lab_reference()
    n = len(tbl)
    rng = np.random.default_rng(20260826)  # fixed: the strip must not move between runs

    fig, (ax_share, ax_tex, ax_reuse) = plt.subplots(
        3, 1, figsize=(10.4, 9.6), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.05, 1.0]})

    # -- 2a: share of distinct prefabs that are novel --------------------
    style_panel(ax_share, n)
    for x, row in enumerate(tbl.itertuples()):
        runs = jl[jl["ok"] & (jl["display_name"] == row.display)]["novel_share"].dropna()
        if runs.empty:
            continue
        rounded_bar(ax_share, x, runs.median(), SLIM_WIDTH,
                    PROVIDER_COLOR[row.provider])
        ring = [pe.withStroke(linewidth=3.1, foreground=SURFACE)]
        ax_share.plot([x, x], [runs.quantile(0.25), runs.quantile(0.75)],
                      color=INK, linewidth=1.1, solid_capstyle="butt",
                      zorder=4, path_effects=ring)
    pooled = jl[jl["ok"]]["novel_share"].median()
    ax_share.axhline(pooled, color=INK_SOFT, linewidth=1.0,
                     linestyle=(0, (4, 3)), zorder=2)
    ax_share.text(n - 0.42, pooled, f"  pooled median {pooled:.0%}",
                  ha="left", va="center", fontsize=7.5, color=INK_SOFT,
                  clip_on=False)
    ax_share.set_ylim(0, 1.06)
    ax_share.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_share.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax_share.set_ylabel("Prefabs invented\n(share of distinct prefabs)", fontsize=9)

    # -- 2b: novel textures, one dot per run ----------------------------
    # 46% of runs invent none, so a box would report a median of 0 for five
    # models and hide that the zeros are a mode. Ten runs per model is exactly
    # the size a strip reads well at.
    style_panel(ax_tex, n)
    for x, row in enumerate(tbl.itertuples()):
        vals = jl[jl["ok"] & (jl["display_name"] == row.display)]["novel_textures"].to_numpy()
        if not len(vals):
            continue
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax_tex.plot(x + jitter, vals, linestyle="none", marker="o",
                    markersize=5.5, markerfacecolor=PROVIDER_COLOR[row.provider],
                    markeredgecolor=SURFACE, markeredgewidth=1.5, alpha=0.95,
                    zorder=3)
        ax_tex.plot([x - 0.26, x + 0.26], [np.median(vals)] * 2, color=INK,
                    linewidth=1.4, solid_capstyle="butt", zorder=4,
                    path_effects=[pe.withStroke(linewidth=3.4, foreground=SURFACE)])
    ax_tex.set_ylim(-1.6, jl["novel_textures"].max() * 1.1)
    ax_tex.set_ylabel("Textures invented\n(one dot per run, bar = median)", fontsize=9)

    # -- 2c: objects covered per distinct prefab ------------------------
    style_panel(ax_reuse, n)
    for x, row in enumerate(tbl.itertuples()):
        runs = jl[jl["ok"] & (jl["display_name"] == row.display)]["reuse"].dropna()
        if runs.empty:
            continue
        rounded_bar(ax_reuse, x, runs.median(), SLIM_WIDTH,
                    PROVIDER_COLOR[row.provider])
    reference_rule(ax_reuse, ref["reuse"],
                   f"{REF_LINE}: {ref['reuse']:.0f}\u00d7")
    ax_reuse.set_ylim(0, ref["reuse"] * 1.16)
    ax_reuse.set_ylabel("Prefab reuse\n(objects per distinct prefab)", fontsize=9)

    ax_reuse.set_xticklabels(tbl["tick"], fontsize=8.5)
    counts_row(ax_reuse, tbl, -0.22, "runs succeeded", list(tbl["n_ok"]),
               lambda v, n_: v < n_)
    draw_provider_brackets(ax_reuse, tbl, -0.34, -0.375)

    titles(fig, "Authoring burden left behind by each generated lab",
           "Successful L3/L4 json_lab runs. \u201cInvented\u201d means the asset is "
           "not in the moon-lab library the schema documents \u2014 someone would "
           "have to model or paint it.\nThe share is near-flat across models, "
           "levels and topics: library grounding fails uniformly, not just on "
           "the weak models. Gemini 3.1 Flash-Lite is low only because its labs "
           "are nearly empty.",
           ysub=0.955)
    fig.tight_layout(rect=(0.03, 0.06, 0.99, 0.905))
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

D_JSON = [("num_modules", "modules"), ("num_clips", "clips"),
          ("num_objects", "objects"), ("num_components", "components"),
          ("num_object_changes", "object changes"),
          ("num_text_labels", "text labels"), ("novel_prefabs", "novel prefabs")]
D_ARLEM = [("num_predicates", "predicates"), ("num_actions", "actions"),
           ("total_triggers", "triggers"), ("total_pois", "POIs"),
           ("num_things", "things"), ("num_places", "places"),
           ("num_detectables", "detectables")]


def _elasticity_panel(ax, data, metrics, title):
    """Slope chart on a log x-axis, so equal slopes are equal growth ratios."""
    rows = []
    for col, label in metrics:
        lo = data[data["ok"] & (data["level"] == "L3")][col].mean()
        hi = data[data["ok"] & (data["level"] == "L4")][col].mean()
        rows.append((label, lo, hi, hi / lo if lo else np.nan))
    rows.sort(key=lambda r: r[3])

    for y, (label, lo, hi, ratio) in enumerate(rows):
        ax.plot([lo, hi], [y, y], color=INK_MUTED, linewidth=1.4, zorder=2,
                solid_capstyle="round")
        for val, lv in ((lo, "L3"), (hi, "L4")):
            ax.plot(val, y, marker="o", markersize=8, zorder=3,
                    markerfacecolor=LEVEL_COLOR[lv], markeredgecolor=SURFACE,
                    markeredgewidth=1.5, linestyle="none")
        ax.text(1.02, y, f"{ratio:.2f}\u00d7", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8.5, color=INK,
                fontweight="bold" if ratio >= 2 else "normal", clip_on=False)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xscale("log")
    ax.set_xlim(1.6, 72)
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_xticklabels(["2", "5", "10", "20", "50"], fontsize=8.5)
    ax.set_xlabel("mean count per generated specification (log scale)", fontsize=8.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.yaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)


def fig_specificity_elasticity(df: pd.DataFrame) -> plt.Figure:
    fig, (ax_j, ax_a) = plt.subplots(1, 2, figsize=(11.0, 5.6))
    _elasticity_panel(ax_j, json_lab_slice(df), D_JSON,
                      "json_lab  \u2014  Lab JSON")
    _elasticity_panel(ax_a, arlem_slice(df, full_only=True), D_ARLEM,
                      "arlem  \u2014  full ARLEM scenario")

    titles(fig, "What the detailed script buys, by target specification",
           "Mean count per specification at L3 (objectives only) and L4 (plus "
           "the full script), pooled over all models. On a log axis the slope "
           "is the growth ratio, printed at the right.\nThe script roughly "
           "doubles a Lab JSON but barely moves an ARLEM workplace: it lands "
           "almost entirely in predicates and actions, leaving the environment "
           "untouched.",
           y=0.99, ysub=0.95)
    level_legend(fig, y=0.998)
    fig.tight_layout(rect=(0.02, 0.02, 0.955, 0.87), w_pad=6.0)
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
     "L3 and L4 runs, shown separately because the detailed script roughly "
     "doubles every count. The dashed rule is the reference lab. Models come "
     "closest on objects and modules and fall furthest short on clips: the "
     "reference lab's 76 steps are more than double the best model median of "
     "35. L1 is excluded throughout \\textemdash{} it returns a \\texttt{LabOutline}, "
     "which has no modules, clips, objects or components to count."),
    ("wave2_authoring_burden", "fig:authoring-burden",
     "Authoring work implied by each generated lab. Top: the share of distinct "
     "prefabs a run references that do not exist in the documented moon-lab "
     "library, bars median and whiskers interquartile range. Middle: novel "
     "textures, one dot per run, drawn as a strip because 46\\% of runs invent "
     "none and a box plot would report a median of zero for five models. "
     "Bottom: how many scene objects each distinct prefab covers, against the "
     "reference lab's 10. The invention rate is 88\\% pooled and near-flat "
     "across models, levels and topics, so library grounding fails uniformly "
     "rather than only on the weaker models; Gemini~3.1~Flash-Lite scores low "
     "only because its labs are nearly empty. Novel counts are reported as a "
     "share because absolute counts track lab size."),
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
     "Growth from L3 (learning objectives only) to L4 (plus the full lesson "
     "script), pooled over all models. Counts are means; the axis is "
     "logarithmic, so an equal slope is an equal growth ratio and the printed "
     "multiplier reads directly off the line. The Lab JSON is elastic to "
     "specificity \\textemdash{} modules $2.4\\times$, components $2.3\\times$, "
     "objects $2.1\\times$, clips $2.0\\times$ \\textemdash{} while ARLEM is "
     "nearly inelastic. The script's detail lands in ARLEM's predicates "
     "($1.6\\times$) and actions ($1.3\\times$) and leaves the workplace "
     "environment essentially unchanged (detectables $1.04\\times$)."),
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
        "% SLICES. json_lab numbers are the 104 successful L3/L4 runs. L1 is",
        "% excluded because it returns the spec-agnostic LabOutline, which has no",
        "% modules, clips, objects or components -- all 55 L1 rows record",
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
        "% and 4 json_lab L3/L4. Its n column is the caveat; medians over 3-4",
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
        "the successful L3/L4 runs; \\emph{novel} counts prefabs absent from the "
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
