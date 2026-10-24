"""
Shared publication figure style (Nature-Communications taste).

Conventions:
  - Arial sans-serif, modest font sizes
  - top / right spines hidden, ticks pointing outward
  - no gridlines
  - editable vector text (pdf.fonttype = 42) so figures can be tweaked in
    Illustrator/Inkscape at submission
  - figures saved as both vector PDF and high-DPI PNG
"""

import matplotlib as mpl


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_spines(ax):
    """Hide top/right spines and set outward ticks on a single axis."""
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1.0)


def panel_label(ax, letter, dx=-0.08, dy=1.02, fontsize=13):
    """Bold panel identifier at the top-left corner (outside the axes), matching Fig 1.

    Use this instead of embedding '(A)' in the panel title, so every figure carries a
    consistent, professional panel-letter scheme (one distinct letter per panel).
    """
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="bottom", ha="right")


def save_fig(fig, stem, formats=("pdf", "png")):
    """Save a figure to <stem>.<ext> for each requested format."""
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")


# ---------------------------------------------------------------------------
# Colour: Okabe & Ito colour-blind-safe palette.
# RULE: ONE COLOUR = ONE CONCEPT across the whole figure, and colour is used
# ONLY where marker shape, fill, line style or position cannot carry the meaning. Everything else is
# black ink: axes, reference lines (thin black dashed, with an inline label, so they cannot be read as data) and uncoded markers.
# A black marker must mean the same thing in every panel (e.g. a filled dot = "front propagated"
# wherever it appears). Never pair red against green.
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "reddish_purple": "#CC79A7",
}

PALETTE = {
    # edge-detection criterion (paper C panel F only)
    "criterion_i": OKABE_ITO["bluish_green"],      # upper edge, criterion (i)
    "criterion_ii": OKABE_ITO["reddish_purple"],   # upper edge, criterion (ii)
    # the lower (weak-coupling) edge, and nothing else: no grey reference lines or bars of other quantities
    "lower_edge": "#7F7F7F",                       # neutral grey
    # neutral ink for everything uncoded
    "neutral": OKABE_ITO["black"],
    # hue for one sequential series per figure (paper C: lattice refinement h, panel A): see sequential()
    "sequential_base": OKABE_ITO["orange"],
}
# Paper C codes the release rule (original vs settling) by marker, not colour: filled circle + solid
# line vs open square + solid line, black (v6d: the dashed line is reserved for reference lines).


def palette(name):
    """Colour for a quantity by its meaning, e.g. palette('lower_edge'). KeyError if undefined."""
    return PALETTE[name]


def sequential(n, base=None, light=0.35, dark=0.55):
    """n shades of one hue, light -> dark (first = lightest), for an ordered series such as
    refinement h = 1, 1/2, 1/4, 1/8. Shades run from `base` mixed `light` with white to
    `base` mixed `dark` with black. Default hue PALETTE['sequential_base'] (Okabe-Ito orange)."""
    import numpy as np
    from matplotlib.colors import to_rgb
    b = np.array(to_rgb(base or PALETTE["sequential_base"]))
    lo, hi = b + light * (1 - b), b * (1 - dark)
    return [tuple(lo + (hi - lo) * t) for t in np.linspace(0, 1, n)] if n > 1 else [tuple(b)]


def apply_nature_sizes(font=None):
    """Nature-like small sizes for a dense multi-panel figure (16 cm wide): 6 pt ticks and
    legends, 7 pt labels, thin lines. Call after apply_style().

    font: None keeps the house default (Arial sans-serif, Arial maths). A serif name such as
    "Times New Roman" (a journal rule) switches text to that font and maths to STIX."""
    if font:
        mpl.rcParams.update({"font.family": "serif", "font.serif": [font, "STIXGeneral"]})
    mpl.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.6, "lines.linewidth": 0.8, "lines.markersize": 3,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })
    if font:
        mpl.rcParams["mathtext.fontset"] = "stix"
