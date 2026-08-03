"""Shared figure style for the manuscript.

Every figure script imports from here. Nothing in this module recomputes a
statistic: constants are read back from results/*.json and asserted against the
values reported in the manuscript, so a drifted input file fails loudly instead
of silently redrawing a different figure.

Backend is matplotlib only (drawing, preview, export, visual QA).
"""

from __future__ import annotations

import json
import os
from math import erfc, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"
SRCDIR = FIGDIR / "source_data"

# --------------------------------------------------------------------------- #
# geometry (Nature-family column widths)
# --------------------------------------------------------------------------- #
MM = 1.0 / 25.4
W1 = 89 * MM        # single column
W15 = 120 * MM      # 1.5 column
W2 = 183 * MM       # double column
HMAX = 225 * MM     # usable canvas height, leaves room for the caption


# --------------------------------------------------------------------------- #
# typography
# --------------------------------------------------------------------------- #
def _register_arial() -> str:
    """Arial is required by the style; fall back to DejaVu Sans with a warning."""
    candidates = [
        Path("/mnt/c/Windows/Fonts"),
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts/truetype/msttcorefonts"),
    ]
    found = False
    for d in candidates:
        if not d.exists():
            continue
        for name in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
            f = d / name
            if f.exists():
                fm.fontManager.addfont(str(f))
                found = True
        if found:
            break
    if not found:
        print("[style] WARNING: Arial not found, falling back to DejaVu Sans")
        return "DejaVu Sans"
    return "Arial"


FONT = _register_arial()

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    # spines and ticks carry the printed structure; 0.5 pt reads anaemic at
    # 183 mm, so keep them at the lower end of the journal-final 0.7-1.2 range
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.minor.size": 1,
    "ytick.minor.size": 1,
    "lines.linewidth": 0.9,
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.35,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "savefig.dpi": 600,
    # panels are placed at millimetre coordinates, so the canvas must not be
    # cropped to content: a tight bounding box would break the column widths
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.0,
    "axes.unicode_minus": True,
}
plt.rcParams.update(RC)

FS_MIN = 5          # smallest permitted annotation size
FS_PANEL = 8        # panel letters


# --------------------------------------------------------------------------- #
# colour
# --------------------------------------------------------------------------- #
# signal family: diverging AUC scale, neutral at 0.5, fixed 0.30-0.70 everywhere
AUC_COLORS = ["#2166AC", "#67A9CF", "#D1E5F0", "#FFFFFF",
              "#FDDBC7", "#EF8A62", "#B2182B"]
CMAP_AUC = LinearSegmentedColormap.from_list("auc_rdbu", AUC_COLORS, N=256)
CMAP_AUC.set_bad("#F2F2F2")
AUC_VMIN, AUC_VMAX = 0.30, 0.70
NORM_AUC = Normalize(vmin=AUC_VMIN, vmax=AUC_VMAX, clip=True)

# neutral family
GREY_FILL = "#F2F2F2"
GREY_RULE = "#BDBDBD"
GREY_TEXT = "#525252"
INK = "#000000"

# accent family: three meanings only, never used decoratively
ACC_FAIL = "#D7301F"    # failed pre-registered prediction, retraction, reversal
ACC_POWER = "#F0B429"   # underpowered, uninterpretable
# the negative-control accent was #6A51A3, which is dE 11 from the cognitive
# family purple: the same colour to the eye, for the opposite meaning, and the
# two sat in adjacent panels of Figure 3. This wine is dE 35 from its nearest
# neighbour in normal vision and 17 under protanopia, at the same chroma as the
# trait families so it still belongs to the set.
ACC_NEG = "#9E1F5B"     # negative control

# trait families: row bands and tick marks only, never fills
FAMILY_COLOR = {
    "cognitive": "#5E3C99",
    "vagal": "#1B7837",
    "conduction": "#4393C3",
    "atrial": "#B8B8B8",
    "ventricular": "#E08214",
    "control": "#737373",
}
# the same families at text contrast: the row-band greys above are deliberately
# pale and would not survive being set as 5 pt type
FAMILY_INK = {
    "cognitive": "#5E3C99",
    "vagal": "#1B7837",
    "conduction": "#2E6E9E",
    "atrial": "#6B6B6B",
    "ventricular": "#B35F0A",
    "control": "#737373",
}

# supporting fills for categorical panels that carry no AUC meaning. Low chroma
# so a panel using all six still reads as one figure, and separated in luminance
# so it survives greyscale. Never used where AUC_COLORS or an accent applies.
SUPPORT = ["#2E4A62", "#4E7A8A", "#7FA0A0", "#A9B5A6", "#C8BCA6", "#E2DCD1"]

# Figure S5 only: the mouse cluster classes. Declared here because no figure
# script should hold a colour of its own. The two teals are the scored class and
# the panel-identified class excluded from scoring; the rest is the SUPPORT
# ladder. The scored class and the comparator are the two the mouse analysis
# turns on, so they must stay apart in greyscale as well as in colour, which
# 399_qa.py enforces.
MOUSE_CLASS = {
    "nodal": "#013D36",
    "non_myocyte_nodal": "#5AB4AC",
    "non_myocyte_purkinje": SUPPORT[3],
    "working_CM": SUPPORT[0],
    "fibroblast": SUPPORT[1],
    "endothelium": SUPPORT[2],
    "immune": SUPPORT[4],
    "erythroid": SUPPORT[5],
}
MOUSE_CLASS_HATCH = {"non_myocyte_nodal": "////",
                     "non_myocyte_purkinje": "\\\\"}


FAMILY_LABEL = {
    "cognitive": "Cognitive",
    "vagal": "Vagal / heart rate",
    "conduction": "Conduction",
    "atrial": "Atrial",
    "ventricular": "Ventricular",
    "control": "Control",
}

# anatomical positions along the conduction axis, reused in every figure
AXIS_COLOR = {
    "SAN_P_cell": "#B2182B",
    "AVN_P_cell": "#EF8A62",
    "AVN_bundle_cell": "#92C5DE",
    "Purkinje": "#2166AC",
}
# the same four at text contrast. #92C5DE and #EF8A62 are legitimate fills but
# fall to about 1.8:1 and 2.6:1 against white, which no 5-6 pt label survives;
# these are the same hues darkened until they do.
AXIS_INK = {
    "SAN_P_cell": "#8C1420",
    "AVN_P_cell": "#C1532E",
    "AVN_bundle_cell": "#3A7CA8",
    "Purkinje": "#1A4E80",
}


# --------------------------------------------------------------------------- #
# constants read back from results, then checked against the manuscript
# --------------------------------------------------------------------------- #
def load(rel: str):
    with open(RESULTS / rel) as fh:
        return json.load(fh)


_NEG = load("negative_audit.json")

SE = _NEG["se_by_state"]                # SE of the within-stratum AUC
MDE = _NEG["mde"]                       # minimum detectable AUC at 80% power
ALPHA = _NEG["alpha"]
POWER = _NEG["power"]
REFERENCE_RG = _NEG["reference_rg"]     # equivalence-test reference effect

AXIS_STATES = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
STATE_LABEL = {
    "SAN_P_cell": "SAN\npacemaker",
    "AVN_P_cell": "AVN\npacemaker",
    "AVN_bundle_cell": "AV\nbundle",
    "Purkinje": "Purkinje",
}
STATE_LABEL_FLAT = {
    "SAN_P_cell": "SAN pacemaker",
    "AVN_P_cell": "AVN pacemaker",
    "AVN_bundle_cell": "AV bundle",
    "Purkinje": "Purkinje",
}
N_CELLS = {"SAN_P_cell": 245, "AVN_P_cell": 155,
           "AVN_bundle_cell": 37, "Purkinje": 82}
N_STRATA = {"SAN_P_cell": 6, "AVN_P_cell": 3,
            "AVN_bundle_cell": 3, "Purkinje": 10}

UNDERPOWERED = "AVN_bundle_cell"        # n = 37, MDE 0.637: supports no inference

# guard against a drifted input file
for _s, _se, _mde in [("SAN_P_cell", 0.0209, 0.558), ("AVN_P_cell", 0.0260, 0.573),
                      ("Purkinje", 0.0348, 0.598), ("AVN_bundle_cell", 0.0488, 0.637)]:
    assert abs(SE[_s] - _se) < 5e-5, f"SE drift for {_s}: {SE[_s]}"
    assert abs(MDE[_s] - _mde) < 5e-4, f"MDE drift for {_s}: {MDE[_s]}"
assert abs(REFERENCE_RG - 0.234) < 5e-4, f"reference rg drift: {REFERENCE_RG}"


# --------------------------------------------------------------------------- #
# trait ordering and naming, matching Table 2 of the manuscript
# --------------------------------------------------------------------------- #
TRAIT_ORDER = [
    ("EducationalAttainment", "Educational attainment", "cognitive"),
    ("Intelligence", "Intelligence", "cognitive"),
    ("ReactionTime", "Reaction time", "cognitive"),
    ("HRV_SDNNc", "HRV, SDNN corrected", "vagal"),
    ("HRV_SDNN", "HRV, SDNN", "vagal"),
    ("HRV_RMSSDc", "HRV, RMSSD corrected", "vagal"),
    ("HRV_RMSSD", "HRV, RMSSD", "vagal"),
    ("RestingHeartRate", "Resting heart rate", "vagal"),
    ("PRinterval", "PR interval", "conduction"),
    ("QRSduration", "QRS duration", "conduction"),
    ("BundleBranchBlock", "Bundle branch block", "conduction"),
    ("AVblock", "Atrioventricular block", "conduction"),
    ("AtrialFlutter", "Atrial flutter", "atrial"),
    ("AtrialFibrillation", "Atrial fibrillation", "atrial"),
    ("HeartFailure", "Heart failure", "ventricular"),
    ("Brugada", "Brugada syndrome", "ventricular"),
    ("QTinterval", "QT interval", "ventricular"),
    ("RheumatoidArthritis", "Rheumatoid arthritis (control)", "control"),
    ("RheumatoidArthritis_noMHC", "Rheumatoid arthritis, MHC-free", "control"),
]
TRAIT_LABEL = {k: lab for k, lab, _ in TRAIT_ORDER}
TRAIT_FAMILY = {k: fam for k, _, fam in TRAIT_ORDER}


# --------------------------------------------------------------------------- #
# multiple testing: reproduce the stored Benjamini-Hochberg correction
# --------------------------------------------------------------------------- #
def bh_flags():
    """Return {(trait, state): 'fdr' | 'nominal' | None} over all 140 tests.

    The q values are read from the stored per-test table. They are cross-checked
    against a re-derivation from the stored z values and against the counts in
    results/corrections.json, so a drifted input fails loudly. No new statistic.
    """
    import pandas as pd

    tab = pd.read_csv(RESULTS / "scdrs_axis" / "cell_level_fdr.tsv", sep="\t")
    q = {(r.trait, r.cell): r.fdr for r in tab.itertuples()}
    p_stored = {(r.trait, r.cell): r.p for r in tab.itertuples()}

    stored = load("corrections.json")["fdr"]
    assert len(q) == stored["n_tests"], f"{len(q)} tests"
    n_sig = sum(v < ALPHA for v in q.values())
    assert n_sig == stored["n_survive"], f"{n_sig} survive"
    lost = sorted(f"{k[0]}/{k[1]}" for k in q
                  if p_stored[k] < ALPHA and q[k] >= ALPHA)
    assert lost == sorted(stored["lost"]), f"lost set differs: {lost}"

    # independent re-derivation from the stored z values
    sa = load("scdrs_axis/axis_interaction.json")["stratified_auc"]
    keys = [(t, s) for t, per in sa.items() for s in per]
    p = np.asarray([erfc(abs(sa[t][s]["z"]) / sqrt(2)) for t, s in keys])
    order = np.argsort(p)
    m = len(p)
    passing = p[order] <= ALPHA * np.arange(1, m + 1) / m
    kmax = int(np.max(np.where(passing)[0]) + 1) if passing.any() else 0
    redrived = {keys[i] for i in order[:kmax]}
    assert redrived == {k for k in q if q[k] < ALPHA}, "stored FDR disagrees"

    return {k: ("fdr" if q[k] < ALPHA
                else ("nominal" if p_stored[k] < ALPHA else None)) for k in q}


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
def new_figure(width_mm, height_mm):
    """Canvas of exact physical size; panels are then placed in millimetres."""
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    fig._mm = (width_mm, height_mm)
    return fig


def ax_mm(fig, x, y, w, h, **kw):
    """Axes placed by millimetre rectangle, origin at the lower-left corner."""
    W, H = fig._mm
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def text_mm(fig, x, y, s, **kw):
    W, H = fig._mm
    return fig.text(x / W, y / H, s, **kw)


def letter_mm(fig, x, y, letter):
    text_mm(fig, x, y, letter, fontsize=FS_PANEL, fontweight="bold",
            va="top", ha="left")


def panel_letter(ax, letter, x=-0.09, y=1.06, **kw):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", va="bottom", ha="left", **kw)


def fig_letter(fig, letter, x, y):
    """Panel letter placed in figure coordinates, for schematic panels."""
    fig.text(x, y, letter, fontsize=FS_PANEL, fontweight="bold",
             va="bottom", ha="left")


def hatch_underpowered(ax, x0, x1, y0=None, y1=None, label=None,
                       orientation="vertical", fontsize=FS_MIN):
    """45-degree mask marking the AV bundle column or any underpowered region."""
    if orientation == "vertical":
        lo, hi = ax.get_ylim()
        y0 = lo if y0 is None else y0
        y1 = hi if y1 is None else y1
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   hatch="////", edgecolor=ACC_POWER,
                                   linewidth=0, zorder=4, clip_on=True))
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=ACC_POWER, linewidth=0.6, zorder=5,
                                   clip_on=True))
    if label:
        ax.text((x0 + x1) / 2, y1, label, ha="center", va="bottom",
                fontsize=fontsize, color=ACC_POWER, zorder=6)


def sig_square(ax, col, row, flag, zorder=6, inset=(0.09, 0.09)):
    """Outline a heatmap cell that reaches significance.

    Follows the convention of the single-cell heritability literature: the fill
    carries the effect and an outlined cell carries the discrete decision, so no
    number has to be printed inside the cell. `inset` is given in cell fractions
    and must be set per figure when cells are not square, or the outline shows a
    fat margin on the long side and none on the short one.
    """
    ix, iy = inset
    common = dict(fill=False, edgecolor=INK, zorder=zorder)
    rect = ((col + ix, row + iy), 1 - 2 * ix, 1 - 2 * iy)
    if flag == "fdr":
        ax.add_patch(plt.Rectangle(*rect, linewidth=0.9, **common))
    elif flag == "nominal":
        ax.add_patch(plt.Rectangle(*rect, linewidth=0.5,
                                   linestyle=(0, (1.4, 1.2)), **common))


def sig_marker(ax, x, y, flag, size=6, color=INK, zorder=6):
    """Filled dot = survives FDR 0.05; open dot = nominal only; nothing = ns."""
    if flag == "fdr":
        ax.scatter([x], [y], s=size, c=color, marker="o", linewidths=0,
                   zorder=zorder, clip_on=False)
    elif flag == "nominal":
        ax.scatter([x], [y], s=size, facecolors="none", edgecolors=color,
                   marker="o", linewidths=0.4, zorder=zorder, clip_on=False)


def auc_colorbar(fig, ax, label="AUC vs myocyte lineage", orientation="vertical",
                 **kw):
    sm = plt.cm.ScalarMappable(norm=NORM_AUC, cmap=CMAP_AUC)
    cb = fig.colorbar(sm, cax=ax, orientation=orientation, **kw)
    cb.set_label(label, fontsize=6)
    cb.outline.set_linewidth(0.4)
    if orientation == "vertical":
        cb.ax.tick_params(labelsize=5, width=0.4, length=1.8)
        cb.set_ticks([0.30, 0.40, 0.50, 0.60, 0.70])
        cb.ax.set_yticklabels(["\u22640.30", "0.40", "0.50", "0.60", "\u22650.70"])
    else:
        cb.ax.tick_params(labelsize=5, width=0.4, length=1.8)
        cb.set_ticks([0.30, 0.40, 0.50, 0.60, 0.70])
        cb.ax.set_xticklabels(["\u22640.30", "0.40", "0.50", "0.60", "\u22650.70"])
    return cb


def zero_line(ax, value=0.5, axis="y", **kw):
    style = dict(color=GREY_RULE, linewidth=0.5, linestyle=(0, (3, 2)), zorder=1)
    style.update(kw)
    (ax.axhline if axis == "y" else ax.axvline)(value, **style)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def allow_offaxis(ax, *which):
    """Declare that this axes deliberately draws labels outside its own limits.

    Row-label columns and bracket ticks are placed in data coordinates with
    clip_on=False, which makes them indistinguishable from truncated data. Mark
    the axis they run off, and only that one, so a genuine clip on the other
    axis is still caught.
    """
    ax._allow_offaxis = set(getattr(ax, "_allow_offaxis", set())) | set(which)


def clipping_report(fig, tol=0.004):
    """Name every axes whose plotted data runs past its own limits.

    Truncated data is the failure a figure cannot show you: the axis looks fine
    and the tail is simply gone. Compares each axes' dataLim against its
    viewLim, ignoring shifts smaller than `tol` of the view range.
    """
    out = []
    for i, ax in enumerate(fig.axes):
        if not ax.has_data():
            continue
        dl, vl = ax.dataLim, ax.viewLim
        if not np.isfinite([dl.x0, dl.x1, dl.y0, dl.y1]).all():
            continue
        exempt = getattr(ax, "_allow_offaxis", set())
        for lo, hi, dlo, dhi, axis in [
                (min(vl.x0, vl.x1), max(vl.x0, vl.x1), dl.x0, dl.x1, "x"),
                (min(vl.y0, vl.y1), max(vl.y0, vl.y1), dl.y0, dl.y1, "y")]:
            if axis in exempt:
                continue
            slack = tol * (hi - lo)
            if dlo < lo - slack:
                out.append(f"axes[{i}] {axis} data reaches {dlo:.4g}, "
                           f"axis starts {lo:.4g}")
            if dhi > hi + slack:
                out.append(f"axes[{i}] {axis} data reaches {dhi:.4g}, "
                           f"axis ends {hi:.4g}")
    return out


def save_figure(fig, name, source_data=None):
    """Write PDF, SVG and 600-dpi TIFF, plus one xlsx of per-panel source data."""
    FIGDIR.mkdir(exist_ok=True)
    (FIGDIR / "qa").mkdir(parents=True, exist_ok=True)
    clipped = clipping_report(fig)
    for line in clipped:
        print(f"[{name}] CLIPPED  {line}")
    with open(FIGDIR / "qa" / f"_clip_{name}.json", "w") as fh:
        json.dump(clipped, fh)
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.svg")
    # the raster deliverable must be flat RGB on white: an alpha channel is
    # rendered as black by some production pipelines
    tif = FIGDIR / f"{name}.tif"
    fig.savefig(tif, dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    from PIL import Image
    im = Image.open(tif)
    if im.mode != "RGB":
        flat = Image.new("RGB", im.size, "white")
        flat.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
        im.close()
        flat.save(tif, compression="tiff_lzw", dpi=(600, 600))
    else:
        im.close()
    written = [f"{name}.pdf", f"{name}.svg", f"{name}.tif"]
    (FIGDIR / "preview").mkdir(exist_ok=True)          # screen check only
    fig.savefig(FIGDIR / "preview" / f"{name}.png", dpi=300)
    if source_data:
        import pandas as pd
        SRCDIR.mkdir(parents=True, exist_ok=True)
        path = SRCDIR / f"{name}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as xl:
            for sheet, df in source_data.items():
                pd.DataFrame(df).to_excel(xl, sheet_name=sheet[:31], index=False)
        written.append(f"source_data/{name}.xlsx")
    size = fig.get_size_inches()
    print(f"[{name}] {size[0]*25.4:.0f} x {size[1]*25.4:.0f} mm -> " + ", ".join(written))
    plt.close(fig)


if __name__ == "__main__":
    flags = bh_flags()
    n_fdr = sum(1 for v in flags.values() if v == "fdr")
    print(f"font={FONT}  tests={len(flags)}  survive_fdr={n_fdr}")
    print("SE ", {k: round(v, 4) for k, v in SE.items()})
    print("MDE", {k: round(v, 3) for k, v in MDE.items()})
