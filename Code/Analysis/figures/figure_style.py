"""
Shared house style for every publication figure in this folder.

The design tokens, the `set_style()` rcParams, and the two custom marks
(`rounded_bar`, `draw_provider_brackets`) live here rather than in whichever
figure script happened to define them first, so a palette or ordering change
lands in one place and every figure moves together.

Provider colours are the validated categorical slots blue / aqua-green /
orange (worst all-pairs CVD deltaE 9.2, above the 8.0 target). A literal green
for OpenAI was tested first and hard-failed: green #008300 against orange
#eb6834 is deltaE 3.2 under protanopia -- the classic red-green confusion pair.

Not a runnable script -- import it:

    from figure_style import MODEL_ORDER, PROVIDER_COLOR, set_style
"""

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


# Provider hues: validated categorical slots 1 / 3 / 2. See module docstring.
PROVIDER_COLOR = {
    "google": "#2a78d6",     # blue
    "openai": "#1baf7a",     # aqua-green
    "anthropic": "#eb6834",  # orange
}
PROVIDER_LABEL = {
    "google": "Google — Gemini",
    "openai": "OpenAI — GPT",
    "anthropic": "Anthropic — Claude",
}
# Redundant fill pattern per provider, for figures that must survive greyscale
# print (the three hues collapse to similar greys). Solid / diagonal /
# cross-hatch are the most distinct trio at bar widths this narrow. Opt-in via
# rounded_bar(..., hatch=PROVIDER_HATCH[provider]).
PROVIDER_HATCH = {
    "google": "///",
    "openai": "xxx",
    "anthropic": "",
}
HATCH_COLOR = "#ffffff"  # drawn over the fill; reads on both the hue and its grey
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#7a7975"
GRID = "#e2e2df"
SURFACE = "#ffffff"

# Display order: providers alphabetical, models small -> large within provider.
# Ties inside a size tier break by generation / tier name (nano < mini, 2.5 < 3.1).
MODEL_ORDER = [
    # (display_name in CSV, short tick label, provider)
    ("Claude Haiku 4.5", "Haiku 4.5", "anthropic"),
    ("Claude Sonnet 5", "Sonnet 5", "anthropic"),
    ("Claude Opus 4.8", "Opus 4.8", "anthropic"),
    ("Gemini 2.5 Flash Lite", "2.5 Flash\nLite", "google"),
    ("Gemini 3.1 Flash Lite", "3.1 Flash\nLite", "google"),
    ("Gemini 3.5 Flash", "3.5 Flash", "google"),
    ("Gemini 3.1 Pro", "3.1 Pro", "google"),
    ("GPT-5.4 Nano", "5.4 Nano", "openai"),
    ("GPT-5.4 Mini", "5.4 Mini", "openai"),
    ("GPT-5.4", "5.4", "openai"),
    ("GPT-5.5", "5.5", "openai"),
]

# Prompt specificity levels. Run artefacts -- and the `level` column of every
# exported CSV -- still carry the original L1/L3/L4 keys, so those stay the
# lookup keys everywhere; these are the paper-facing names, and no figure
# should ever print a bare "L3" at a reader. Rename here and every figure,
# legend and generated caption moves together.
LEVEL_NAME = {"L1": "Outline", "L3": "Objectives", "L4": "Script"}
# What each level's prompt carries, phrased as what it *adds* to the one before.
# Kept terse: these sit under a three-column tick row, where the spelled-out
# "+ learning objectives" / "+ full lesson script" run into each other.
LEVEL_GLOSS = {
    "L1": "description only",
    "L3": "+ objectives",
    "L4": "+ full script",
}

BAR_WIDTH = 0.66
CORNER_PT = 2.0  # rounded data-end radius, in points


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": INK_SOFT,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.major.size": 0,
        "ytick.major.size": 3,
        "ytick.major.width": 0.8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "hatch.linewidth": 1.0,
        "pdf.fonttype": 42,   # embed TrueType, not Type3 -- most venues require it
        "ps.fonttype": 42,
    })


# ── Marks ────────────────────────────────────────────────────────────

def rounded_bar(ax, x, height, width, color, hatch=None):
    """A bar with rounded top corners, square-anchored to the baseline.

    Radius is specified in points and converted per-axes, so it stays visually
    constant across panels with very different data ranges. Clamped on short
    bars so a hairline never turns into a lozenge.

    ``hatch`` overlays a pattern in HATCH_COLOR (matplotlib draws hatches in the
    edgecolor; linewidth=0 keeps the outline itself invisible).
    """
    if height <= 0:
        return
    # points -> display px -> data units, separately per axis
    inv = ax.transData.inverted()
    r_px = CORNER_PT * ax.figure.dpi / 72
    p0 = inv.transform((0, 0))
    p1 = inv.transform((r_px, r_px))
    rx = min(abs(p1[0] - p0[0]), width / 2)
    ry = min(abs(p1[1] - p0[1]), height / 2)

    left, right, top = x - width / 2, x + width / 2, height
    verts = [
        (left, 0), (left, top - ry),
        (left, top), (left + rx, top),          # top-left curve
        (right - rx, top),
        (right, top), (right, top - ry),        # top-right curve
        (right, 0), (left, 0),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO,
        MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                           hatch=hatch or None,
                           edgecolor=HATCH_COLOR if hatch else "none",
                           linewidth=0, zorder=3))


def draw_provider_brackets(ax, tbl, y_rule, y_label, rule_color=None):
    """Colour rule + ink label under each provider group.

    This is the identity encoding -- it sits directly beneath the marks it
    names, so hue is never the only cue and a separate legend box would only
    repeat it further away.

    ``rule_color`` overrides the provider hue for charts where the fills
    already spend the colour channel on something else (e.g. the stacked
    clean/repair/failed decomposition), so the brackets stay pure grouping
    chrome instead of implying a second encoding.
    """
    trans = ax.get_xaxis_transform()  # x in data, y in axes fraction
    start = 0
    provs = list(tbl["provider"])
    for i in range(len(provs) + 1):
        if i == len(provs) or provs[i] != provs[start]:
            lo, hi = start - BAR_WIDTH / 2, i - 1 + BAR_WIDTH / 2
            prov = provs[start]
            ax.plot([lo, hi], [y_rule, y_rule], transform=trans, clip_on=False,
                    color=rule_color or PROVIDER_COLOR[prov], linewidth=2.4,
                    solid_capstyle="round", zorder=5)
            ax.text((lo + hi) / 2, y_label, PROVIDER_LABEL[prov], transform=trans,
                    ha="center", va="top", fontsize=9.5, color=INK, clip_on=False)
            start = i
