"""Figure 6. The population level gives the same shape.

Renumbered from 4: the population-level section moved after replication,
so this figure is now cited sixth. The file name is unchanged.

a  Genetic correlation of three cognitive traits with six cardiovascular traits.
b  The strongest row, with its confidence intervals and the correction threshold.
c  The null as an established result: equivalence against a reference effect.
d  Why a self-implemented estimator can be believed.

Reads results/ only; computes no new statistic. The r_g colour scale is fixed
at ±0.30 and is a different quantity from the AUC scale used elsewhere.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from _style import (ACC_FAIL, CMAP_AUC, FAMILY_COLOR, FS_MIN, GREY_FILL,
                    GREY_RULE, GREY_TEXT, INK, REFERENCE_RG, ax_mm, letter_mm,
                    load, new_figure, save_figure, zero_line)

W, H = 120.0, 132.0
fig = new_figure(W, H)

RG = load("ldsc_rg.json")["rg"]
NEG = load("negative_audit.json")

COG = [("EducationalAttainment", "Educational attainment"),
       ("Intelligence", "Intelligence"),
       ("ReactionTime", "Reaction time")]
CARD = [("HRV_RMSSD", "HRV, RMSSD", "vagal"),
        ("HRV_SDNN", "HRV, SDNN", "vagal"),
        ("RestingHeartRate", "Resting heart rate", "vagal"),
        ("PRinterval", "PR interval", "conduction"),
        ("AtrialFibrillation", "Atrial fibrillation", "atrial"),
        ("QTinterval", "QT interval", "ventricular")]
BONF = 0.05 / (len(COG) * len(CARD))

NORM_RG = Normalize(vmin=-0.30, vmax=0.30, clip=True)
# the atrial band colour is too light to set type in
TEXT_COLOR = dict(FAMILY_COLOR, atrial=GREY_TEXT)


def get(a, b):
    return RG.get(f"{a}|{b}") or RG[f"{b}|{a}"]


# --------------------------------------------------------------------------- #
# a  the shape
# --------------------------------------------------------------------------- #
HX, HW, HY, HH = 42.0, 66.0, 97.0, 19.5
axa = ax_mm(fig, HX, HY, HW, HH)
axa.set_xlim(0, len(CARD))
axa.set_ylim(len(COG), 0)
axa.set_xticks([])
axa.set_yticks([])
for sp in axa.spines.values():
    sp.set_visible(False)

rows_a = []
for r, (ck, clab) in enumerate(COG):
    for c, (dk, dlab, fam) in enumerate(CARD):
        v = get(ck, dk)
        rgba = CMAP_AUC(NORM_RG(v["rg"]))
        axa.add_patch(Rectangle((c, r), 1, 1, facecolor=rgba,
                                edgecolor="white", linewidth=0.4))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        fg = "white" if lum < 0.45 else INK
        axa.text(c + 0.5, r + 0.40, f"{v['rg']:+.3f}".replace("-", "−"),
                 ha="center", va="center", fontsize=FS_MIN, color=fg)
        axa.text(c + 0.5, r + 0.72, f"({v['se']:.3f})", ha="center",
                 va="center", fontsize=FS_MIN, color=fg)
        if v["p"] < BONF:
            axa.add_patch(Rectangle((c + 0.04, r + 0.04), 0.92, 0.92,
                                    fill=False, edgecolor=INK, linewidth=0.8,
                                    zorder=5))
        rows_a.append(dict(cognitive=clab, cardiovascular=dlab,
                           rg=round(v["rg"], 4), SE=round(v["se"], 4),
                           P=v["p"], passes_bonferroni=bool(v["p"] < BONF)))
    axa.text(-0.08, r + 0.5, clab, ha="right", va="center", fontsize=6,
             color=FAMILY_COLOR["cognitive"], clip_on=False)
for c, (dk, dlab, fam) in enumerate(CARD):
    axa.text(c + 0.5, -0.12, dlab, rotation=40, rotation_mode="anchor",
             ha="left", va="bottom", fontsize=FS_MIN, color=TEXT_COLOR[fam])

cax = ax_mm(fig, HX, 91.0, 26.0, 1.6)
sm = plt.cm.ScalarMappable(norm=NORM_RG, cmap=CMAP_AUC)
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("genetic correlation r$_g$", fontsize=6, labelpad=1.5)
cb.outline.set_linewidth(0.4)
cb.ax.tick_params(labelsize=FS_MIN, width=0.4, length=1.8)
cb.set_ticks([-0.3, 0, 0.3])
cb.ax.set_xticklabels(["≤−0.30", "0", "≥0.30"])
fig.text((HX + 30) / W, 88.6 / H,
         f"black outline, P < {BONF:.4f} (Bonferroni over {len(COG) * len(CARD)} pairs)\n"
         "value in cell, r$_g$ (SE)", fontsize=FS_MIN, color=GREY_TEXT,
         va="bottom", ha="left", linespacing=1.6)
letter_mm(fig, 1.5, H - 2.0, "A")

# --------------------------------------------------------------------------- #
# b  the strongest row
# --------------------------------------------------------------------------- #
axb = ax_mm(fig, 42.0, 58.0, 66.0, 24.0)
axb.set_xlim(-0.20, 0.42)
axb.set_ylim(len(CARD) - 0.4, -0.6)
axb.set_yticks([])
axb.spines["left"].set_visible(False)
axb.tick_params(labelsize=FS_MIN)
axb.set_xlabel("r$_g$ with educational attainment (95% CI)", fontsize=6,
               labelpad=1.5)
zero_line(axb, 0.0, axis="x")

rows_b = []
for r, (dk, dlab, fam) in enumerate(CARD):
    v = get("EducationalAttainment", dk)
    lo, hi = v["rg"] - 1.96 * v["se"], v["rg"] + 1.96 * v["se"]
    sig = v["p"] < BONF
    col = TEXT_COLOR[fam]
    axb.plot([lo, hi], [r, r], color=col, linewidth=0.7, solid_capstyle="butt")
    axb.scatter([v["rg"]], [r], s=13 if sig else 9, c=col if sig else "white",
                edgecolors=col, linewidths=0.5, zorder=5)
    axb.text(-0.215, r, dlab, ha="right", va="center", fontsize=6, color=col,
             clip_on=False)
    rows_b.append(dict(cardiovascular=dlab, rg=round(v["rg"], 4),
                       SE=round(v["se"], 4), CI_low=round(lo, 4),
                       CI_high=round(hi, 4), P=v["p"]))
axb.text(0.41, -0.55, "filled, passes Bonferroni", fontsize=FS_MIN,
         ha="right", va="bottom", color=GREY_TEXT)
letter_mm(fig, 1.5, 86.0, "B")

# --------------------------------------------------------------------------- #
# c  the null, as an established result
# --------------------------------------------------------------------------- #
EQ = NEG["rg_equivalence"]
axc = ax_mm(fig, 42.0, 30.0, 66.0, 18.0)
axc.set_xlim(0, 0.28)
axc.set_ylim(len(EQ) - 0.4, -0.9)
axc.set_yticks([])
axc.spines["left"].set_visible(False)
axc.tick_params(labelsize=FS_MIN)
axc.set_xlabel("|r$_g$| and its 95% upper bound", fontsize=6, labelpad=1.5)
axc.axvspan(REFERENCE_RG, 0.28, facecolor=GREY_FILL, edgecolor="none", zorder=0)
axc.axvline(REFERENCE_RG, color=INK, linewidth=0.5, linestyle=(0, (2, 1.6)),
            zorder=1)
# horizontal, above row 0: the rotated form was taller than the 18 mm panel
axc.text(0.0, -0.72, f"grey, at or above the reference effect {REFERENCE_RG:.3f}",
         fontsize=FS_MIN, ha="left", va="center", color=INK)

rows_c = []
LBL = {"EducationalAttainment": "Educational attainment",
       "Intelligence": "Intelligence", "ReactionTime": "Reaction time",
       "AtrialFibrillation": "atrial fibrillation", "QTinterval": "QT interval"}
for r, (k, v) in enumerate(EQ.items()):
    a, b = k.split("|")
    axc.plot([abs(v["rg"]), v["upper"]], [r, r], color=INK, linewidth=0.7,
             solid_capstyle="butt", zorder=3)
    axc.scatter([abs(v["rg"])], [r], s=9, c=INK, linewidths=0, zorder=4)
    axc.scatter([v["upper"]], [r], s=9, facecolors="white", edgecolors=INK,
                linewidths=0.5, zorder=4)
    axc.text(-0.006, r, f"{LBL[a]} × {LBL[b]}", ha="right", va="center",
             fontsize=FS_MIN, clip_on=False)
    rows_c.append(dict(pair=f"{LBL[a]} x {LBL[b]}", rg=round(v["rg"], 4),
                       SE=round(v["se"], 4),
                       upper_bound=round(v["upper"], 4),
                       excludes_reference=bool(v["excludes_reference"])))
letter_mm(fig, 1.5, 51.0, "C")

# --------------------------------------------------------------------------- #
# d  why the estimator can be believed
# --------------------------------------------------------------------------- #
axd = ax_mm(fig, 42.0, 9.0, 66.0, 12.0)
v = get("RestingHeartRate", "HRV_RMSSD")
lo, hi = v["rg"] - 1.96 * v["se"], v["rg"] + 1.96 * v["se"]
axd.set_xlim(-1.20, 0.10)
axd.set_ylim(-1.9, 1.5)
axd.set_yticks([])
axd.spines["left"].set_visible(False)
axd.tick_params(labelsize=FS_MIN)
axd.set_xlabel("genetic correlation r$_g$", fontsize=6, labelpad=1.5)
axd.text(-1.235, 0, "resting heart rate ×\nHRV, RMSSD", ha="right",
         va="center", fontsize=FS_MIN, linespacing=1.5, clip_on=False)
axd.axvspan(-0.74, -0.55, facecolor=GREY_FILL, edgecolor="none", zorder=0)
axd.text(-0.645, 1.45, "published\n−0.74 to −0.55", fontsize=FS_MIN,
         ha="center", va="top", color=GREY_TEXT, linespacing=1.5)
axd.plot([lo, hi], [0, 0], color=INK, linewidth=0.7, solid_capstyle="butt",
         zorder=3)
axd.scatter([v["rg"]], [0], s=14, c=INK, linewidths=0, zorder=4)
axd.text(v["rg"], -0.30, f"{v['rg']:.3f} (SE {v['se']:.3f})".replace("-", "−"),
         fontsize=FS_MIN, ha="center", va="top", color=INK)
GHOST = -0.121
axd.scatter([GHOST], [0], s=14, facecolors="white", edgecolors=ACC_FAIL,
            linewidths=0.6, zorder=4)
axd.text(GHOST, 1.45, "−0.121 before\nallele alignment", fontsize=FS_MIN,
         ha="center", va="top", color=ACC_FAIL, linespacing=1.5)
rows_d = [dict(estimate="after allele alignment", rg=round(v["rg"], 4),
               SE=round(v["se"], 4), CI_low=round(lo, 4), CI_high=round(hi, 4)),
          dict(estimate="before allele alignment", rg=GHOST, SE=None,
               CI_low=None, CI_high=None),
          dict(estimate="published range", rg=None, SE=None, CI_low=-0.74,
               CI_high=-0.55)]
letter_mm(fig, 1.5, 23.0, "D")

save_figure(fig, "Fig6", source_data={
    "a_rg_matrix": rows_a, "b_educational_attainment": rows_b,
    "c_equivalence": rows_c, "d_implementation_check": rows_d})
