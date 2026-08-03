"""Figure 1. Design, comparator, positive control, and a mechanistic closure.

Panels run in the order the Methods and Results discuss them.

a  Where the cells come from.
b  The annotations are independently supported by fixed marker panels.
c  What each population is compared against, and how the strata combine.
d  The sequencing chemistry that forces the comparison to be made within strata.
e  It uses the right genes: the score's largest contributors in sinoatrial
   pacemaker cells, and what deleting each gene set costs each trait.
f  And the pipeline returns the known answer: atrial fibrillation finds muscle.

Reads results/ only. The marker-panel z scores in b are the stored panel means
standardized across the seven groups, and are asserted against the values
reported in the manuscript.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, PathPatch,
                                Rectangle)

from _style import (AXIS_COLOR, AXIS_INK, AXIS_STATES, CMAP_AUC, FAMILY_COLOR,
                    FS_MIN, GREY_FILL, GREY_RULE, GREY_TEXT, INK, RESULTS,
                    SUPPORT, TRAIT_LABEL, ax_mm, letter_mm, load, new_figure,
                    save_figure, zero_line)

W, H = 183.0, 106.0
fig = new_figure(W, H)

TOP_ROW_LETTER = 105.0
BOT_ROW_LETTER = 46.0

# --------------------------------------------------------------------------- #
# a  where the cells come from
# --------------------------------------------------------------------------- #
N_CELLS_ANNOT = {"SAN_P_cell": (245, 6), "AVN_P_cell": (155, 3),
                 "AVN_bundle_cell": (38, 4), "Purkinje": (110, 12)}

axa = ax_mm(fig, 1.0, 57.0, 46.0, 46.0)
axa.set_xlim(0, 46)
axa.set_ylim(14, 60)
axa.set_aspect("equal")
axa.axis("off")


def bezier(anchors, segments, closed=False, **kw):
    """anchors[0] then (control, control, anchor) triples."""
    verts = [anchors[0]]
    codes = [MplPath.MOVETO]
    for seg in segments:
        verts += list(seg)
        codes += [MplPath.CURVE4] * 3
    if closed:
        verts.append(anchors[0])
        codes.append(MplPath.CLOSEPOLY)
    return PathPatch(MplPath(verts, codes), **kw)


APEX = (14.0, 17.0)
VENTRICLE = bezier(
    [APEX],
    [((10.4, 20.4), (6.4, 25.0), (5.6, 32.0)),      # right free wall, downward
     ((5.0, 37.2), (5.8, 41.4), (7.6, 43.0)),       # right shoulder
     ((14.0, 44.6), (23.0, 44.4), (29.8, 42.6)),    # atrioventricular groove
     ((31.6, 38.0), (31.8, 34.0), (30.4, 30.0)),    # left shoulder
     ((28.6, 25.0), (21.0, 19.4), APEX)],           # left free wall to apex
    closed=True, facecolor=GREY_FILL, edgecolor=GREY_RULE, linewidth=0.7,
    joinstyle="round", zorder=2)
axa.add_patch(Ellipse((11.6, 47.6), 12.4, 9.6, angle=9, facecolor="white",
                      edgecolor=GREY_RULE, linewidth=0.7, zorder=1))
axa.add_patch(Ellipse((25.4, 47.8), 11.6, 9.0, angle=-9, facecolor="white",
                      edgecolor=GREY_RULE, linewidth=0.7, zorder=1))
axa.add_patch(VENTRICLE)
axa.add_patch(bezier([(18.6, 43.4)],
                     [((17.8, 35.0), (16.6, 28.0), (15.4, 21.6))],
                     facecolor="none", edgecolor=GREY_RULE, linewidth=0.6,
                     linestyle=(0, (2.5, 1.8)), zorder=3))
for x, y, lab in [(10.2, 46.4, "RA"), (26.4, 49.6, "LA"),
                  (11.4, 33.0, "RV"), (24.8, 34.0, "LV")]:
    axa.text(x, y, lab, fontsize=FS_MIN, ha="center", va="center",
             color=GREY_TEXT, zorder=4)

# the conduction axis
axa.add_patch(Circle((6.9, 51.4), 1.3, facecolor=AXIS_COLOR["SAN_P_cell"],
                     edgecolor="none", zorder=5))
axa.plot([7.4, 16.2], [50.6, 45.0], color=AXIS_COLOR["SAN_P_cell"],
         linewidth=0.6, linestyle=(0, (1.5, 1.2)), zorder=4)
axa.add_patch(Circle((17.4, 44.2), 1.2, facecolor=AXIS_COLOR["AVN_P_cell"],
                     edgecolor="none", zorder=5))
axa.plot([17.7, 18.4], [43.1, 36.4], color=AXIS_INK["AVN_bundle_cell"],
         linewidth=1.5, solid_capstyle="round", zorder=4)
# each bundle branch forks into a small fan, so the network reads as branching
# fibres rather than as an arrow
BRANCHES = [(((17.2, 32.2), (15.6, 30.6), (14.8, 29.4)),
             [(11.4, 25.4), (13.3, 24.0), (15.7, 24.9)]),
            (((20.6, 32.2), (21.4, 30.6), (22.2, 29.4)),
             [(25.6, 25.4), (23.7, 24.0), (21.3, 24.9)])]
for (ctrl1, ctrl2, node), tips in BRANCHES:
    axa.add_patch(bezier([(18.4, 36.4)], [(ctrl1, ctrl2, node)],
                         facecolor="none", edgecolor=AXIS_COLOR["Purkinje"],
                         linewidth=0.9, capstyle="round", zorder=4))
    for tip in tips:
        axa.plot([node[0], tip[0]], [node[1], tip[1]],
                 color=AXIS_COLOR["Purkinje"], linewidth=0.55,
                 solid_capstyle="round", zorder=4)

# name at 6 pt in the darkened axis ink, count at 5 pt in grey underneath, and a
# leader from the left edge of the name block to the structure it names
LABELS = [("SAN_P_cell", "Sinoatrial node", 1.0, 58.4, "left", (4.6, 54.3),
           (6.5, 52.6)),
          ("AVN_P_cell", "Atrioventricular node", 33.0, 46.6, "left",
           (32.4, 45.4), (18.7, 44.2)),
          ("AVN_bundle_cell", "AV bundle", 33.0, 39.4, "left", (32.4, 38.2),
           (18.6, 39.0)),
          ("Purkinje", "Purkinje network", 33.0, 29.6, "left", (32.4, 28.4),
           (26.4, 25.6))]
for key, lab, xt, yt, ha, lead0, lead1 in LABELS:
    n, d = N_CELLS_ANNOT[key]
    axa.text(xt, yt, lab, fontsize=6, ha=ha, va="center",
             color=AXIS_INK[key], zorder=6)
    axa.text(xt, yt - 2.6, f"{n} cells, {d} donors", fontsize=FS_MIN, ha=ha,
             va="center", color=GREY_TEXT, zorder=6)
    axa.plot([lead0[0], lead1[0]], [lead0[1], lead1[1]],
             color=AXIS_INK[key], linewidth=0.4, zorder=3)
letter_mm(fig, 0.5, TOP_ROW_LETTER, "A")

# --------------------------------------------------------------------------- #
# b  the annotations hold up against fixed marker panels
# --------------------------------------------------------------------------- #
PM = load("purkinje_identity.json")["panel_means"]
GROUPS = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje", "vCM*",
          "aCM*", "all other cells"]
GLAB = {"SAN_P_cell": "SAN pacemaker", "AVN_P_cell": "AVN pacemaker",
        "AVN_bundle_cell": "AV bundle", "Purkinje": "Purkinje",
        "vCM*": "ventricular myocytes", "aCM*": "atrial myocytes",
        "all other cells": "all other cells"}
PANELS = ["pacemaker / nodal", "ventricular conduction", "ventricular identity",
          "atrial identity"]

Z = {}
for pan in PANELS:
    v = np.array([PM[g][pan] for g in GROUPS])
    Z[pan] = (v - v.mean()) / v.std(ddof=1)
for pan, g, want in [("ventricular conduction", "Purkinje", 0.89),
                     ("pacemaker / nodal", "Purkinje", -0.72),
                     ("pacemaker / nodal", "SAN_P_cell", 1.81),
                     ("ventricular identity", "Purkinje", 0.94),
                     ("atrial identity", "Purkinje", -0.86),
                     ("ventricular conduction", "AVN_bundle_cell", 0.86),
                     ("ventricular conduction", "vCM*", 0.45)]:
    got = Z[pan][GROUPS.index(g)]
    assert abs(got - want) < 0.006, f"{pan}/{g}: {got:.3f} vs {want}"

axd = ax_mm(fig, 78.0, 59.0, 44.0, 44.0)
axd.set_xlim(-0.6, len(PANELS) - 0.4)
axd.set_ylim(len(GROUPS) - 0.35, -1.05)
axd.set_yticks([])
axd.set_xticks(range(len(PANELS)))
axd.set_xticklabels(["pacemaker /\nnodal", "ventricular\nconduction",
                     "ventricular\nidentity", "atrial\nidentity"],
                    fontsize=FS_MIN, linespacing=1.4)
axd.tick_params(labelsize=FS_MIN, bottom=False, pad=2)
axd.spines["left"].set_visible(False)
axd.spines["bottom"].set_visible(False)

# dot size and colour both carry z, following the marker dot-plot convention of
# the single-cell heart literature: size = magnitude, colour = signed value
NORM_Z = Normalize(vmin=-2.0, vmax=2.0, clip=True)
DOT = lambda z: 7 + 52 * min(abs(z), 2.0) / 2.0
rows_b = []
for r, g in enumerate(GROUPS):
    for c, pan in enumerate(PANELS):
        z = Z[pan][r]
        axd.scatter([c], [r], s=DOT(z), c=[CMAP_AUC(NORM_Z(z))],
                    edgecolors=GREY_RULE, linewidths=0.3, zorder=3)
        rows_b.append(dict(cell_group=GLAB[g], panel=pan, z=round(z, 3),
                           panel_mean=round(PM[g][pan], 4)))
    axd.text(-0.74, r, GLAB[g], fontsize=6, ha="right", va="center",
             color=INK if g in AXIS_STATES else GREY_TEXT, clip_on=False)
# the four conduction populations sit above the rule, the reference groups below
axd.plot([-0.6, len(PANELS) - 0.4], [3.5, 3.5], color=GREY_RULE, linewidth=0.4,
         zorder=1)

for j, z in enumerate([0.5, 1.0, 2.0]):
    axd.scatter([0.16 + j * 0.60], [-0.92], s=DOT(z), c=GREY_TEXT, linewidths=0,
                clip_on=False)
    axd.text(0.30 + j * 0.60, -0.92, f"{z:.1f}", fontsize=FS_MIN, va="center",
             color=GREY_TEXT, clip_on=False)
axd.text(-0.56, -0.92, "|z|", fontsize=FS_MIN, va="center", ha="right",
         color=GREY_TEXT, clip_on=False)

cbz = ax_mm(fig, 126.0, 65.0, 1.8, 20.0)
smz = plt.cm.ScalarMappable(norm=NORM_Z, cmap=CMAP_AUC)
cb = fig.colorbar(smz, cax=cbz)
cb.set_label("z across the seven groups", fontsize=6, labelpad=2)
cb.outline.set_linewidth(0.4)
cb.ax.tick_params(labelsize=FS_MIN, width=0.4, length=1.8)
cb.set_ticks([-2, 0, 2])
cb.ax.set_yticklabels(["≤−2", "0", "≥2"])
letter_mm(fig, 51.0, TOP_ROW_LETTER, "B")

# --------------------------------------------------------------------------- #
# c  what each population is compared against
# --------------------------------------------------------------------------- #
axs = ax_mm(fig, 138.0, 59.0, 43.0, 44.0)
axs.set_xlim(0, 43)
axs.set_ylim(0, 44)
axs.axis("off")
axs.text(21.5, 43.4, "one stratum = donor × region × chemistry",
         fontsize=FS_MIN, ha="center", va="top", color=INK)

axs.scatter([2.4], [39.0], s=4.0, c=AXIS_COLOR["SAN_P_cell"], linewidths=0)
axs.text(3.8, 39.0, "conduction cells", fontsize=FS_MIN, va="center", color=INK)
axs.scatter([23.0], [39.0], s=2.6, facecolors="none", edgecolors=GREY_RULE,
            linewidths=0.35)
axs.text(24.4, 39.0, "myocyte lineage", fontsize=FS_MIN, va="center",
         color=GREY_TEXT)

rng = np.random.default_rng(3)
for i, (x0, foc) in enumerate([(1.5, 5), (15.5, 3), (29.5, 4)]):
    axs.add_patch(Rectangle((x0, 21.0), 12.0, 14.5, facecolor="white",
                            edgecolor=GREY_RULE, linewidth=0.5))
    pts = rng.uniform([x0 + 1.4, 22.4], [x0 + 10.6, 34.1], size=(26, 2))
    axs.scatter(pts[foc:, 0], pts[foc:, 1], s=2.6, facecolors="none",
                edgecolors=GREY_RULE, linewidths=0.35)
    axs.scatter(pts[:foc, 0], pts[:foc, 1], s=4.0,
                c=AXIS_COLOR["SAN_P_cell"], linewidths=0)
    axs.add_patch(FancyArrowPatch((x0 + 6.0, 20.2), (x0 + 6.0, 17.4),
                                  arrowstyle="-|>", mutation_scale=3.5,
                                  linewidth=0.5, color=GREY_TEXT))
    axs.text(x0 + 6.0, 16.4, f"AUC$_{i + 1}$", fontsize=FS_MIN, ha="center",
             va="top", color=INK)

# the three per-stratum values are gathered on one rule, then combined once
axs.plot([7.5, 35.5], [12.4, 12.4], color=GREY_TEXT, linewidth=0.5, zorder=2)
for x0 in (1.5, 15.5, 29.5):
    axs.plot([x0 + 6.0, x0 + 6.0], [13.8, 12.4], color=GREY_TEXT,
             linewidth=0.5, zorder=2)
axs.add_patch(FancyArrowPatch((21.5, 12.4), (21.5, 9.0), arrowstyle="-|>",
                              mutation_scale=3.5, linewidth=0.5,
                              color=GREY_TEXT))
axs.text(21.5, 8.0, "van Elteren weighted mean", fontsize=FS_MIN, ha="center",
         va="top", color=INK)
letter_mm(fig, 133.0, TOP_ROW_LETTER, "C")

# --------------------------------------------------------------------------- #
# d  the chemistry that forces this design
# --------------------------------------------------------------------------- #
axch = ax_mm(fig, 5.0, 18.0, 34.0, 24.0)
axch.set_xlim(0, 100)
axch.set_ylim(-1.75, 2.05)
axch.set_yticks([])
axch.set_xticks([0, 50, 100])
axch.set_xticklabels(["0", "50", "100%"], fontsize=FS_MIN)
axch.spines["left"].set_visible(False)
axch.tick_params(labelsize=FS_MIN, pad=1.5)
# a categorical assay variable, so it takes the low-chroma support fills: the
# diverging AUC ramp would invite the reader to read blue and red as effect
CHEM = [("10x multiome", SUPPORT[0], "white"), ("3′ v3", SUPPORT[2], INK),
        ("3′ v2", SUPPORT[4], INK)]
rows_chem = []
for r, (name, parts) in enumerate([("conduction-axis background",
                                    [34.3, 33.5, 32.2]),
                                   ("nodal conduction cells", [100.0, 0, 0])]):
    left = 0.0
    for (clab, col, txt), frac in zip(CHEM, parts):
        if frac <= 0:
            continue
        axch.add_patch(Rectangle((left, r - 0.21), frac, 0.42, facecolor=col,
                                 edgecolor="white", linewidth=0.4))
        if frac > 12:
            axch.text(left + frac / 2, r, f"{frac:.1f}", fontsize=FS_MIN,
                      ha="center", va="center", color=txt)
        rows_chem.append(dict(group=name, chemistry=clab, percent=frac))
        left += frac
    axch.text(0, r + 0.30, name, fontsize=FS_MIN, ha="left", va="bottom",
              color=INK)
x = 0.0
for clab, col, _ in CHEM:
    axch.add_patch(Rectangle((x, -1.06), 3.4, 0.34, facecolor=col,
                             edgecolor="none"))
    axch.text(x + 4.6, -0.89, clab, fontsize=FS_MIN, ha="left", va="center",
              color=GREY_TEXT)
    x += 9.6 + 2.9 * len(clab)
letter_mm(fig, 0.5, BOT_ROW_LETTER, "D")

# --------------------------------------------------------------------------- #
# e  and it uses the right genes
# --------------------------------------------------------------------------- #
CD = load("scdrs/control_dissection.json")
DRIVERS = CD["drivers"]["HRV_RMSSD"][:10]
VAGAL_HIT = {"RGS6", "CHRM2"}
assert DRIVERS[0] == "RGS6" and DRIVERS[2] == "CHRM2", DRIVERS[:3]

axf1 = ax_mm(fig, 46.0, 16.0, 24.0, 26.0)
axf1.set_xlim(0, 2)
axf1.set_ylim(4.7, -2.5)
axf1.set_xticks([])
axf1.set_yticks([])
for sp in axf1.spines.values():
    sp.set_visible(False)
axf1.text(1.02, -2.4, "HRV score contribution\nSAN pacemaker cells",
          fontsize=FS_MIN, ha="center", va="top", color=GREY_TEXT,
          linespacing=1.45)
rows_e = []
for r, gene in enumerate(DRIVERS):
    col, row_i = divmod(r, 5)
    hit = gene in VAGAL_HIT
    axf1.text(col + 0.30, row_i, f"{r + 1}", fontsize=FS_MIN, ha="right",
              va="center", color=GREY_TEXT)
    axf1.text(col + 0.40, row_i, gene, fontsize=6, ha="left", va="center",
              color=FAMILY_COLOR["vagal"] if hit else INK,
              fontweight="bold" if hit else "normal")
    rows_e.append(dict(panel="e, contributing genes", rank=r + 1, gene=gene,
                       trait="HRV, RMSSD", cell_state="SAN pacemaker",
                       vagal_panel=hit))

# Both deletions are scored against their own size- and weight-matched random
# deletions, so the 300-gene programme and the 23-gene panel are each compared
# with a null of their own size rather than with each other's. Rheumatoid
# arthritis is excluded: its full z is -0.195, so a percentage of z is a ratio
# to a number near zero, and its vagal arm deletes no genes at all.
KO = CD["knockout"]
LADDER = [("HRV_RMSSD", 0.0), ("RestingHeartRate", 1.0), ("PRinterval", 2.0),
          ("AtrialFibrillation", 3.0), ("EducationalAttainment", 4.3)]
ARMS = [("neuronal", "neural programme", SUPPORT[1]),
        ("vagal", "vagal panel", FAMILY_COLOR["vagal"])]
axf2 = ax_mm(fig, 98.0, 10.0, 29.0, 32.0)
axf2.set_xlim(0, 14.5)
axf2.set_ylim(4.85, -2.25)
axf2.set_yticks([])
axf2.spines["left"].set_visible(False)
axf2.set_xticks([0, 5, 10])
axf2.tick_params(labelsize=FS_MIN, pad=1.5)
axf2.set_xlabel("% of the score's z lost", fontsize=6, labelpad=1.5)
axf2.text(-0.15, -0.60, "genes", fontsize=FS_MIN, ha="right", va="center",
          color=GREY_TEXT, clip_on=False)
for r, (t_key, y) in enumerate(LADDER):
    full = KO[t_key]["full"]["z"]
    axf2.text(-1.80, y, TRAIT_LABEL[t_key], fontsize=FS_MIN, ha="right",
              va="center", color=INK, clip_on=False)
    for j, (arm, _, col) in enumerate(ARMS):
        a = KO[t_key][arm]
        pct = abs(a["delta"]) / abs(full) * 100
        yy = y + (-0.20 if j == 0 else 0.20)
        axf2.barh([yy], [pct], height=0.34, color=col, linewidth=0, zorder=3)
        axf2.text(-0.15, yy, str(a["n_dropped"]), fontsize=FS_MIN, ha="right",
                  va="center", color=GREY_TEXT, clip_on=False)
        txt = f"{pct:.1f}"
        axf2.text(pct + 0.30, yy, txt, fontsize=FS_MIN, va="center",
                  ha="left", color=col)
        if a["p"] < 0.05:
            axf2.scatter([pct + 0.65 + 0.485 * len(txt)], [yy], s=2.4, c=col,
                         linewidths=0, zorder=4)
        rows_e.append(dict(panel="e, gene-set deletion", trait=TRAIT_LABEL[t_key],
                           deleted=arm, z_full=round(full, 4),
                           z_after=round(a["z"], 4), percent_lost=round(pct, 2),
                           P=a["p"], genes_deleted=a["n_dropped"],
                           matched_random_mean_z=round(a["null_mean"], 4)))
# the key rows run in the same order as the two bars of every pair
for (_, lab, col), ky in zip(ARMS, (-1.95, -1.30)):
    axf2.add_patch(Rectangle((0.0, ky - 0.14), 0.5, 0.28, facecolor=col,
                             edgecolor="none"))
    axf2.text(0.75, ky, lab, fontsize=FS_MIN, ha="left", va="center",
              color=GREY_TEXT)
# the significance key rides on the vagal key line: on its own line it read as
# a continuation of the "genes" column header
axf2.scatter([7.0], [-1.30], s=2.4, c=GREY_TEXT, linewidths=0)
axf2.text(7.5, -1.30, "P < 0.05", fontsize=FS_MIN, ha="left", va="center",
          color=GREY_TEXT)

WANT = {("HRV, RMSSD", "vagal"): 11.18, ("HRV, RMSSD", "neuronal"): 1.01,
        ("Educational attainment", "vagal"): 5.04,
        ("Educational attainment", "neuronal"): 2.90,
        ("Atrial fibrillation", "vagal"): 1.01}
got = {(r["trait"], r["deleted"]): r["percent_lost"] for r in rows_e
       if r["panel"] == "e, gene-set deletion"}
for k, v in WANT.items():
    assert abs(got[k] - v) < 0.02, (k, got[k], v)
assert got[("Educational attainment", "vagal")] > got[("Educational attainment",
                                                       "neuronal")]
letter_mm(fig, 42.0, BOT_ROW_LETTER, "E")

# --------------------------------------------------------------------------- #
# f  the pipeline returns the known answer
# --------------------------------------------------------------------------- #
g = pd.read_csv(RESULTS / "scdrs" / "group_analysis_fdr.tsv", sep="\t")
af = (g[g.trait == "AtrialFibrillation"]
      .sort_values("assoc_mcz", ascending=False).reset_index(drop=True))
axe = ax_mm(fig, 134.0, 18.0, 46.0, 24.0)
axe.set_xlim(-4.8, 17.6)
axe.set_ylim(0, 1.24)
axe.set_yticks([])
axe.spines["left"].set_visible(False)
axe.tick_params(labelsize=FS_MIN, pad=1.5)
axe.set_xlabel("scDRS association z across 62 cell states", fontsize=6,
               labelpad=1.5)
zero_line(axe, 0.0, axis="x")

IS_CM = af.cell_state.str.match(r"^(aCM|vCM)").to_numpy()
rows_f = []
for i, row in af.iterrows():
    z, cm = row.assoc_mcz, bool(IS_CM[i])
    axe.plot([z, z], [0.06, 0.06 + (0.54 if cm else 0.36)],
             color=INK if cm else GREY_RULE, linewidth=1.0 if cm else 0.6,
             solid_capstyle="butt", zorder=4 if cm else 2)
    rows_f.append(dict(rank=i + 1, cell_state=row.cell_state,
                       n_cells=int(row.n_cell), z=round(row.assoc_mcz, 3),
                       P=row.assoc_mcp, FDR=round(row.assoc_fdr, 4),
                       cardiomyocyte=cm))
# labels are ordered by z so no two leaders cross, and the top-ranked state is
# named inside the gap its own outlying z creates
for i, lab, ty in [(1, "vCM4, ventricular  #2", 0.94),
                   (2, "aCM4, atrial  #3", 0.74)]:
    z = af.assoc_mcz[i]
    axe.text(7.0, ty, lab, fontsize=FS_MIN, ha="right", va="center", color=INK)
    axe.plot([7.3, z], [ty, 0.64], color=GREY_TEXT, linewidth=0.4, zorder=1)
axe.text(15.4, 0.88, "unclassified  #1", fontsize=FS_MIN, ha="right",
         va="center", color=GREY_TEXT)
axe.plot([15.6, af.assoc_mcz[0]], [0.88, 0.64], color=GREY_TEXT, linewidth=0.4,
         zorder=1)

axe.text(-4.7, 1.22, "atrial fibrillation", fontsize=6, ha="left", va="top",
         color=INK)
for xk, col, lab in [(4.0, INK, "cardiomyocyte"), (11.0, GREY_RULE, "other")]:
    axe.plot([xk, xk], [1.08, 1.22], color=col, linewidth=1.0 if col == INK
             else 0.6, solid_capstyle="butt")
    axe.text(xk + 0.4, 1.15, lab, fontsize=FS_MIN, ha="left", va="center",
             color=GREY_TEXT)
assert af.cell_state[0] == "unclassified" and af.cell_state[1] == "vCM4", \
    af.cell_state[:3].tolist()
letter_mm(fig, 129.0, BOT_ROW_LETTER, "F")

save_figure(fig, "Fig1", source_data={
    "b_marker_panels": rows_b, "d_chemistry": rows_chem,
    "e_genes": rows_e, "f_atrial_fibrillation": rows_f})
