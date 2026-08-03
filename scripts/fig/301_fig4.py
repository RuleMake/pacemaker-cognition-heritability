"""Figure 5. Boundaries: a gradient along the axis, a reversal, and a floor.

a  Effect along the conduction axis with the undetectable band of each position.
b  Every interpretable population significantly below its own myocyte comparator.
c  The interaction test cannot decide, and the reason is arithmetic.
d  Detection floor against observed effect, all traits and all four positions.
e  The internal control: one population, one comparator, opposite signs.

Reads results/ only; computes no new statistic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from _style import (ACC_FAIL, ACC_POWER, ALPHA, AXIS_STATES,
                    FAMILY_COLOR, FAMILY_INK, FS_MIN, GREY_FILL, GREY_RULE,
                    GREY_TEXT, INK,
                    MDE, N_CELLS, RESULTS, SE, STATE_LABEL, TRAIT_FAMILY,
                    TRAIT_LABEL, TRAIT_ORDER, UNDERPOWERED, ax_mm,
                    letter_mm, load,
                    new_figure, save_figure, zero_line)

W, H = 183.0, 116.0
fig = new_figure(W, H)

SA = load("scdrs_axis/axis_interaction.json")
STRAT = SA["stratified_auc"]
NEG = load("negative_audit.json")

# --------------------------------------------------------------------------- #
# a  gradient along the axis
# --------------------------------------------------------------------------- #
axa = ax_mm(fig, 13.0, 60.0, 74.0, 44.0)
axa.set_xlim(-0.55, 3.55)
axa.set_ylim(0.24, 0.85)          # headroom so the note clears the hatch,
                                  # floor low enough for the QT lower bar
axa.set_ylabel("AUC vs myocyte lineage", fontsize=6, labelpad=2)
axa.tick_params(labelsize=FS_MIN)
axa.set_xticks(range(4))
axa.set_xticklabels([STATE_LABEL[s].replace("\n", " ") + f"\nn = {N_CELLS[s]}"
                     + ("\nunderpowered" if s == UNDERPOWERED else "")
                     for s in AXIS_STATES], fontsize=FS_MIN, linespacing=1.5)
axa.get_xticklabels()[AXIS_STATES.index(UNDERPOWERED)].set_color(ACC_POWER)

rows_a = []
for i, s in enumerate(AXIS_STATES):
    lo, hi = 1 - MDE[s], MDE[s]
    axa.add_patch(Rectangle((i - 0.42, lo), 0.84, hi - lo, facecolor=GREY_FILL,
                            edgecolor="none", zorder=0))
zero_line(axa, 0.5)

SHOW = [("ReactionTime", "cognitive", "o"),
        ("EducationalAttainment", "cognitive", "^"),
        ("HRV_SDNN", "vagal", "D"),
        ("QTinterval", "ventricular", "v"),
        ("RheumatoidArthritis", "control", "s")]
for key, fam, mk in SHOW:
    y = [STRAT[key][s]["auc"] for s in AXIS_STATES]
    col = FAMILY_COLOR[fam]
    axa.plot(range(4), y, color=col, linewidth=0.6, linestyle=(0, (2, 1.6)),
             zorder=2)
    axa.scatter(range(4), y, s=11, c=col, marker=mk, linewidths=0, zorder=3)
    for i, s in enumerate(AXIS_STATES):
        axa.plot([i, i], [y[i] - 1.96 * SE[s], y[i] + 1.96 * SE[s]], color=col,
                 linewidth=0.5, zorder=2)
    rows_a.append(dict(trait=TRAIT_LABEL[key], family=fam,
                       **{STATE_LABEL[s].replace("\n", " "): round(STRAT[key][s]["auc"], 4)
                          for s in AXIS_STATES}))

for _fill, _lw in [(False, 0), (False, 0.6)]:
    axa.add_patch(Rectangle((1.58, 0.24), 0.84, 0.61, fill=_fill,
                            hatch="/////" if _lw == 0 else None,
                            edgecolor=ACC_POWER, linewidth=_lw,
                            zorder=1 if _lw == 0 else 2, alpha=0.7 if _lw == 0 else 1))
for j, (key, fam, mk) in enumerate(SHOW):
    yv = 0.375 - j * 0.021
    axa.scatter([-0.42], [yv], s=11, c=FAMILY_COLOR[fam], marker=mk,
                linewidths=0)
    lab = TRAIT_LABEL[key].replace(" (control)", "")
    axa.text(-0.32, yv, lab, fontsize=FS_MIN, color=FAMILY_COLOR[fam],
             va="center", ha="left")
axa.text(-0.5, 0.845, "grey band, effects too small to detect at 80% power",
         fontsize=FS_MIN, color=GREY_TEXT, ha="left", va="top")
letter_mm(fig, 1.5, H - 2.0, "A")

# --------------------------------------------------------------------------- #
# b  every significant depletion
# --------------------------------------------------------------------------- #
axb = ax_mm(fig, 104.0, 60.0, 72.0, 44.0)
# every significant depletion the paper is allowed to read: the bundle column is
# excluded here exactly as it is in the text. Filtering stored q values by sign
# is not a new statistic.
fdr = pd.read_csv(RESULTS / "scdrs_axis" / "cell_level_fdr.tsv", sep="\t")
sel = fdr[(fdr.fdr < ALPHA) & (fdr.auc < 0.5)
          & (~fdr.trait.str.endswith("_lenadj"))
          & (fdr.cell != UNDERPOWERED)].sort_values("auc")
assert len(sel) == 5, sel[["trait", "cell"]].to_dict("records")
axb.set_xlim(-0.6, len(sel) - 0.4)
axb.set_ylim(0.30, 0.55)
axb.set_ylabel("AUC vs myocyte lineage", fontsize=6, labelpad=2)
axb.tick_params(labelsize=FS_MIN)
zero_line(axb, 0.5)

rows_b, xlab = [], []
for i, r in enumerate(sel.itertuples()):
    col = FAMILY_INK[TRAIT_FAMILY[r.trait]]
    axb.plot([i, i], [0.5, r.auc], color=col, linewidth=0.9, zorder=2)
    axb.scatter([i], [r.auc], s=16, c=col, zorder=3, linewidths=0)
    axb.text(i, r.auc - 0.007, f"{r.auc:.3f}\nz {r.z:+.2f}".replace("-", "−"),
             fontsize=FS_MIN, ha="center", va="top", color=col, linespacing=1.4)
    xlab.append(f"{TRAIT_LABEL[r.trait]}\n{STATE_LABEL[r.cell]}"
                .replace("Brugada syndrome", "Brugada")
                .replace("HRV, SDNN corrected", "HRV, SDNNc")
                .replace("QT interval", "QT"))
    rows_b.append(dict(trait=TRAIT_LABEL[r.trait],
                       cell_state=STATE_LABEL[r.cell].replace("\n", " "),
                       AUC=round(r.auc, 4), z=round(r.z, 3),
                       FDR=r.fdr))
axb.set_xticks(range(len(sel)))
axb.set_xticklabels(xlab, fontsize=FS_MIN, linespacing=1.4)
letter_mm(fig, 92.0, H - 2.0, "B")

# --------------------------------------------------------------------------- #
# c  the internal control
# --------------------------------------------------------------------------- #
axc = ax_mm(fig, 140.0, 13.0, 38.0, 30.0)
axc.set_xlim(0.28, 0.85)
axc.set_ylim(-0.9, 1.20)
axc.set_yticks([])
axc.spines["left"].set_visible(False)
axc.tick_params(labelsize=FS_MIN)
axc.set_xticks([0.35, 0.5, 0.65, 0.8])
axc.set_xlabel("AUC vs myocyte lineage", fontsize=6, labelpad=1.5)
zero_line(axc, 0.5, axis="x")

rt = STRAT["ReactionTime"]["AVN_P_cell"]
qt = STRAT["QTinterval"]["AVN_P_cell"]
se = SE["AVN_P_cell"]
axc.plot([qt["auc"], rt["auc"]], [0, 0], color=GREY_RULE, linewidth=0.7,
         zorder=1)
# family colours, so a trait keeps one colour across every panel of the figure
for v, lab, col, dy in [(qt, "QT interval", FAMILY_COLOR["ventricular"], -0.30),
                        (rt, "Reaction time", FAMILY_COLOR["cognitive"], 0.30)]:
    axc.plot([v["auc"] - 1.96 * se, v["auc"] + 1.96 * se], [0, 0], color=col,
             linewidth=0.7, zorder=2)
    axc.scatter([v["auc"]], [0], s=20, c=col, zorder=4, linewidths=0)
    axc.text(v["auc"], dy, f"{lab}\n{v['auc']:.3f}  (z {v['z']:+.2f})".replace("-", "−"),
             fontsize=FS_MIN, color=col, ha="center",
             va="bottom" if dy > 0 else "top", linespacing=1.5)
axc.text(0.565, 1.17, "155 atrioventricular pacemaker cells,\none comparator",
         fontsize=FS_MIN, color=GREY_TEXT, ha="center", va="top",
         linespacing=1.5)
rows_c = [dict(trait="Reaction time", AUC=round(rt["auc"], 4),
               z=round(rt["z"], 3), n_cells=155),
          dict(trait="QT interval", AUC=round(qt["auc"], 4),
               z=round(qt["z"], 3), n_cells=155)]
letter_mm(fig, 130.0, 48.0, "E")

# --------------------------------------------------------------------------- #
# d  detection floor against observed effect
# --------------------------------------------------------------------------- #
axd = ax_mm(fig, 76.0, 13.0, 50.0, 30.0)
axd.set_xlim(-0.6, 3.6)
axd.set_ylim(0.0, 0.36)           # headroom so the note clears the hatch
axd.set_ylabel("|AUC − 0.5|", fontsize=6, labelpad=2)
axd.tick_params(labelsize=FS_MIN)
axd.set_xticks(range(4))
axd.set_xticklabels([STATE_LABEL[s] for s in AXIS_STATES], fontsize=FS_MIN,
                    linespacing=1.4)

rows_d = []
rng = np.random.default_rng(0)
for i, s in enumerate(AXIS_STATES):
    floor = MDE[s] - 0.5
    axd.plot([i - 0.45, i + 0.45], [floor, floor], color=INK, linewidth=0.9,
             zorder=4, solid_capstyle="butt")
    axd.add_patch(Rectangle((i - 0.45, 0), 0.9, floor, facecolor=GREY_FILL,
                            edgecolor="none", zorder=0))
    # jitter spans i +- 0.22, so the group's left edge is always clear of dots
    axd.text(i - 0.44, floor + 0.006, f"{MDE[s]:.3f}", fontsize=FS_MIN,
             ha="left", va="bottom", color=INK)
    for key, _, fam in TRAIT_ORDER:
        d = abs(STRAT[key][s]["auc"] - 0.5)
        axd.scatter([i + rng.uniform(-0.22, 0.22)], [d], s=4,
                    c=FAMILY_COLOR[fam], linewidths=0, alpha=0.85, zorder=3)
        rows_d.append(dict(cell_state=STATE_LABEL[s].replace("\n", " "),
                           trait=TRAIT_LABEL[key],
                           abs_effect=round(d, 4), MDE_effect=round(floor, 4)))
i_und = AXIS_STATES.index(UNDERPOWERED)
axd.add_patch(Rectangle((i_und - 0.45, 0), 0.9, 0.36, fill=False,
                        hatch="/////", edgecolor=ACC_POWER, linewidth=0,
                        zorder=1, alpha=0.7))
axd.add_patch(Rectangle((i_und - 0.45, 0), 0.9, 0.36, fill=False,
                        edgecolor=ACC_POWER, linewidth=0.6, zorder=2))
axd.text(-0.55, 0.357, "line, minimum detectable effect at 80% power;"
         "  dots, all 19 traits", fontsize=FS_MIN, color=GREY_TEXT, ha="left",
         va="top")
letter_mm(fig, 64.0, 48.0, "D")

# --------------------------------------------------------------------------- #
# e  the interaction test cannot decide
# --------------------------------------------------------------------------- #
axe = ax_mm(fig, 15.0, 13.0, 48.0, 30.0)
n_perm = SA["perm_n"]
grid = np.arange(1, n_perm + 1) / n_perm
axe.set_xlim(0.0, 0.60)
axe.set_ylim(-1.05, 1.55)
axe.set_yticks([])
axe.spines["left"].set_visible(False)
axe.tick_params(labelsize=FS_MIN)
axe.set_xticks([0.0, 0.2, 0.4, 0.6])
axe.set_xlabel("attainable P, exhaustive permutation", fontsize=6, labelpad=1.5)

for g in grid:
    if g > 0.60:
        continue
    axe.plot([g, g], [-0.30, 0.30], color=GREY_RULE, linewidth=0.6,
             solid_capstyle="butt", zorder=2)
axe.axvline(0.05, color=INK, linewidth=0.5, linestyle=(0, (3, 2)), zorder=1)
axe.text(0.062, 1.50, "α = 0.05", fontsize=FS_MIN, ha="left", va="top",
         color=INK)

floor_p = 1.0 / n_perm
axe.plot([floor_p, floor_p], [-0.42, 0.42], color=ACC_POWER, linewidth=1.2,
         solid_capstyle="butt", zorder=5)
axe.annotate(f"smallest P this test\ncan return, 1/{n_perm} = {floor_p:.3f}",
             xy=(floor_p, -0.44), xytext=(0.115, -0.80), fontsize=FS_MIN,
             color=ACC_POWER, ha="left", va="center", linespacing=1.5,
             arrowprops=dict(arrowstyle="-", linewidth=0.4, color=ACC_POWER,
                             shrinkA=0, shrinkB=1))
obs_p = SA["perm_p"]
axe.plot([obs_p, obs_p], [-0.42, 0.42], color=INK, linewidth=1.2,
         solid_capstyle="butt", zorder=5)
axe.annotate(f"observed D = {SA['D']:+.3f}\nranked 3rd of {n_perm}, P = {obs_p:.3f}",
             xy=(obs_p, 0.44), xytext=(0.185, 1.08), fontsize=FS_MIN, color=INK,
             ha="left", va="center", linespacing=1.5,
             arrowprops=dict(arrowstyle="-", linewidth=0.4, color=GREY_TEXT,
                             shrinkA=0, shrinkB=1))
rows_e = [dict(quantity="interaction D", value=round(SA["D"], 4)),
          dict(quantity="sinoatrial term", value=round(SA["term_san"], 4)),
          dict(quantity="apex term", value=round(SA["term_ax"], 4)),
          dict(quantity="permutation P", value=round(obs_p, 4)),
          dict(quantity="number of arm assignments", value=n_perm),
          dict(quantity="smallest attainable P", value=round(floor_p, 4))]
letter_mm(fig, 1.5, 48.0, "C")

save_figure(fig, "Fig4", source_data={
    "a_axis_gradient": rows_a, "b_depletion": rows_b,
    "c_internal_control": rows_c, "d_detection_floor": rows_d,
    "e_interaction": rows_e})
