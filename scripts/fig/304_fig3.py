"""Figure 3. Alternative explanations, taken one at a time.

a  An immune control with more genome-wide-significant variants than any
   cardiovascular trait here is flat.
b  Gene length accounts for part of the excess, not for its existence.
c  Deleting a neural programme removes a small fraction of the score.
d  A glutamatergic programme is not distinguishable from chance.
e  The signal is not carried by a few genes, and not by the size threshold.

Reads results/ only; computes no new statistic. Panels c and d report the
stored summaries of their null distributions, not re-simulated draws.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import numpy as np
from matplotlib.patches import Rectangle

from _style import (ACC_NEG, ACC_POWER, AXIS_STATES, FAMILY_COLOR, FAMILY_INK,
                    FS_MIN,
                    GREY_FILL, GREY_RULE, GREY_TEXT, INK, RESULTS, SE,
                    STATE_LABEL, TRAIT_LABEL, UNDERPOWERED, ax_mm, letter_mm,
                    load, new_figure, save_figure, zero_line)

W, H = 183.0, 96.0
fig = new_figure(W, H)

STRAT = load("scdrs_axis/axis_interaction.json")["stratified_auc"]

# --------------------------------------------------------------------------- #
# a  the immune control
# --------------------------------------------------------------------------- #
axa = ax_mm(fig, 17.0, 58.0, 44.0, 28.0)
axa.set_xlim(-0.6, 3.6)
axa.set_ylim(0.25, 0.66)
axa.set_ylabel("AUC vs myocyte lineage", fontsize=6, labelpad=2)
axa.tick_params(labelsize=FS_MIN)
axa.set_xticks(range(4))
axa.set_xticklabels([STATE_LABEL[s] for s in AXIS_STATES], fontsize=FS_MIN,
                    linespacing=1.4)
zero_line(axa, 0.5)

rows_a = []
for j, (key, lab, off, filled) in enumerate([
        ("RheumatoidArthritis", "34,164 variants", -0.17, True),
        ("RheumatoidArthritis_noMHC", "MHC-free rebuild", 0.17, False)]):
    for i, s in enumerate(AXIS_STATES):
        v = STRAT[key][s]
        x = i + off
        axa.plot([x, x], [v["auc"] - 1.96 * SE[s], v["auc"] + 1.96 * SE[s]],
                 color=ACC_NEG, linewidth=0.6, zorder=2)
        axa.scatter([x], [v["auc"]], s=11, zorder=4, linewidths=0.5,
                    marker="o" if filled else "s",
                    c=ACC_NEG if filled else "white", edgecolors=ACC_NEG)
        rows_a.append(dict(gene_set=lab,
                           cell_state=STATE_LABEL[s].replace("\n", " "),
                           AUC=round(v["auc"], 4), z=round(v["z"], 3)))
    axa.scatter([-0.45], [0.648 - j * 0.024], s=11, linewidths=0.5,
                marker="o" if filled else "s",
                c=ACC_NEG if filled else "white", edgecolors=ACC_NEG)
    axa.text(-0.34, 0.648 - j * 0.024, lab, fontsize=FS_MIN, va="center",
             color=ACC_NEG)
i_und = AXIS_STATES.index(UNDERPOWERED)
axa.add_patch(Rectangle((i_und - 0.45, 0.25), 0.9, 0.41, fill=False,
                        hatch="/////", edgecolor=ACC_POWER, linewidth=0,
                        zorder=1, alpha=0.7))
axa.add_patch(Rectangle((i_und - 0.45, 0.25), 0.9, 0.41, fill=False,
                        edgecolor=ACC_POWER, linewidth=0.6, zorder=2))
axa.get_xticklabels()[i_und].set_color(ACC_POWER)
letter_mm(fig, 1.5, H - 2.0, "A")

# --------------------------------------------------------------------------- #
# b  gene length
# --------------------------------------------------------------------------- #
axb = ax_mm(fig, 75.0, 58.0, 30.0, 28.0)
axb.set_xlim(-0.35, 1.55)
axb.set_ylim(-0.06, 0.215)
axb.set_ylabel("excess over the myocyte\nlineage, AUC − 0.5", fontsize=6,
               labelpad=2, linespacing=1.4)
axb.tick_params(labelsize=FS_MIN)
axb.set_xticks([0, 1])
axb.set_xticklabels(["original\ngene set", "length\nadjusted"],
                    fontsize=FS_MIN, linespacing=1.4)
# stop the zero rule short of the endpoint labels, one of which sits on it
axb.plot([-0.35, 1.02], [0.0, 0.0], color=GREY_RULE, linewidth=0.5,
         linestyle=(0, (3, 2)), zorder=1)

rows_b = []
for s in AXIS_STATES:
    a0 = STRAT["EducationalAttainment"][s]["auc"] - 0.5
    a1 = STRAT["EducationalAttainment_lenadj"][s]["auc"] - 0.5
    is_san = s == "SAN_P_cell"
    col = FAMILY_COLOR["cognitive"] if is_san else GREY_TEXT
    axb.plot([0, 1], [a0, a1], color=col, linewidth=0.9 if is_san else 0.5,
             zorder=3 if is_san else 2)
    axb.scatter([0, 1], [a0, a1], s=11 if is_san else 6, c=col, linewidths=0,
                zorder=4)
    axb.text(1.06, a1, STATE_LABEL[s].replace("\n", " "), fontsize=FS_MIN,
             va="center", ha="left", color=col)
    rows_b.append(dict(cell_state=STATE_LABEL[s].replace("\n", " "),
                       excess_original=round(a0, 4),
                       excess_length_adjusted=round(a1, 4)))
san0 = STRAT["EducationalAttainment"]["SAN_P_cell"]["auc"] - 0.5
san1 = STRAT["EducationalAttainment_lenadj"]["SAN_P_cell"]["auc"] - 0.5
share = (san0 - san1) / san0
assert abs(share - 0.17) < 0.01, share
axb.annotate("", xy=(0.02, san1), xytext=(0.02, san0),
             arrowprops=dict(arrowstyle="<->", linewidth=0.4, color=INK,
                             shrinkA=0, shrinkB=0))
# below the line, not beside it: the sinoatrial line descends across exactly the
# band the label used to sit in
axb.text(0.02, san1 - 0.012, f"{share * 100:.0f}% removed", fontsize=FS_MIN,
         va="center", ha="left", color=INK)
axb.text(-0.32, 0.212, "educational attainment", fontsize=FS_MIN, va="top",
         ha="left", color=FAMILY_COLOR["cognitive"])
letter_mm(fig, 65.0, H - 2.0, "B")

# --------------------------------------------------------------------------- #
# c  deleting the neural programme
# --------------------------------------------------------------------------- #
KO = load("scdrs/control_dissection.json")["knockout"]
KO_ORDER = ["EducationalAttainment", "HRV_RMSSD", "PRinterval",
            "AtrialFibrillation", "RestingHeartRate", "RheumatoidArthritis"]
axc = ax_mm(fig, 136.0, 58.0, 30.0, 28.0)
axc.set_xlim(0, 0.082)
axc.set_ylim(len(KO_ORDER) - 0.4, -0.6)
axc.set_yticks([])
axc.spines["left"].set_visible(False)
axc.tick_params(labelsize=FS_MIN)
axc.set_xlabel("loss in the score's z when the\nneural programme is deleted",
               fontsize=6, labelpad=1.5, linespacing=1.4)
axc.set_xticks([0, 0.02, 0.04, 0.06])

rows_c = []
for r, t in enumerate(KO_ORDER):
    full = KO[t]["full"]["z"]
    n = KO[t]["neuronal"]
    pct = abs(n["delta"]) / abs(full) * 100
    sig = n["p"] < 0.05
    col = INK if sig else GREY_TEXT
    axc.barh([r], [abs(n["delta"])], height=0.5, color=col, linewidth=0,
             zorder=3)
    tag = f"P = {n['p']:.3f}".replace("0.000", "< 0.001")
    if t == "EducationalAttainment":
        tag = f"{pct:.1f}% of z,  " + tag
    axc.text(abs(n["delta"]) + 0.0015, r, tag, fontsize=FS_MIN, va="center",
             ha="left", color=col)
    axc.text(-0.002, r, TRAIT_LABEL[t].replace(" (control)", ""),
             fontsize=6, va="center", ha="right",
             color=col, clip_on=False)
    rows_c.append(dict(trait=TRAIT_LABEL[t], z_full=round(full, 4),
                       z_after_deletion=round(n["z"], 4),
                       percent_removed=round(pct, 2), P=n["p"],
                       genes_deleted=n["n_dropped"],
                       matched_random_mean_z=round(n["null_mean"], 4)))
assert abs(rows_c[0]["percent_removed"] - 2.9) < 0.05, rows_c[0]
letter_mm(fig, 118.0, H - 2.0, "C")

# --------------------------------------------------------------------------- #
# d  the glutamatergic programme
# --------------------------------------------------------------------------- #
GL = load("audit_cognitive.json")["glutamatergic_vs_null"]
axd = ax_mm(fig, 6.0, 12.0, 46.0, 26.0)
axd.set_xlim(0.46, 0.68)
axd.set_ylim(-1.9, 1.5)
axd.set_yticks([])
axd.spines["left"].set_visible(False)
axd.tick_params(labelsize=FS_MIN)
axd.set_xlabel("AUC of the glutamatergic gene set\nin conduction cells",
               fontsize=6, labelpad=1.5, linespacing=1.4)
axd.axvspan(0.46, GL["null_p95"], facecolor=GREY_FILL, edgecolor="none",
            zorder=0)
axd.axvline(GL["null_p95"], color=GREY_RULE, linewidth=0.5,
            linestyle=(0, (2, 1.6)), zorder=1)
axd.plot([GL["null_mean"], GL["null_mean"]], [-0.3, 0.3], color=GREY_TEXT,
         linewidth=0.9, solid_capstyle="butt", zorder=3)
axd.text(GL["null_mean"], -0.42, f"null mean\n{GL['null_mean']:.3f}",
         fontsize=FS_MIN, ha="center", va="top", color=GREY_TEXT,
         linespacing=1.5)
axd.text(GL["null_p95"], -0.42, f"95th percentile\n{GL['null_p95']:.3f}",
         fontsize=FS_MIN, ha="center", va="top", color=GREY_TEXT,
         linespacing=1.5)
axd.scatter([GL["observed"]], [0], s=20, c=GREY_TEXT, linewidths=0, zorder=5)
axd.annotate(f"observed {GL['observed']:.3f}\nP = {GL['p']:.3f}",
             xy=(GL["observed"], 0.14), xytext=(GL["observed"] + 0.012, 0.95),
             fontsize=FS_MIN, ha="left", va="center", color=GREY_TEXT,
             linespacing=1.5,
             arrowprops=dict(arrowstyle="-", linewidth=0.4, color=GREY_RULE,
                             shrinkA=0, shrinkB=1))
axd.text(0.620, -1.72, f"grey, {GL['n_null']} matched random sets",
         fontsize=FS_MIN, ha="right", va="bottom", color=GREY_TEXT)
rows_d = [dict(quantity="observed", value=round(GL["observed"], 4)),
          dict(quantity="null mean", value=round(GL["null_mean"], 4)),
          dict(quantity="null 95th percentile", value=round(GL["null_p95"], 4)),
          dict(quantity="P", value=GL["p"]),
          dict(quantity="random gene sets", value=GL["n_null"])]
letter_mm(fig, 1.5, 46.0, "D")

# --------------------------------------------------------------------------- #
# e  a few strong genes, and the size threshold
# --------------------------------------------------------------------------- #
AUDIT = json.loads((RESULTS / "scdrs" / "audit.json").read_text())
FRAG = [x for x in AUDIT if x.get("check") == "fragility"]
axe1 = ax_mm(fig, 84.0, 12.0, 32.0, 26.0)
axe1.set_xlim(0.5, 1.25)
axe1.set_ylim(len(FRAG) - 0.4, -0.6)
axe1.set_yticks([])
axe1.spines["left"].set_visible(False)
axe1.tick_params(labelsize=FS_MIN)
axe1.set_xlabel("z retained after removing the\n50 highest-weighted genes",
                fontsize=6, labelpad=1.5, linespacing=1.4)
axe1.axvline(1.0, color=GREY_RULE, linewidth=0.5, linestyle=(0, (3, 2)))

rows_e = []
for r, x in enumerate(FRAG):
    trait, state = x["item"].split("/")
    fam = "cognitive" if trait == "EducationalAttainment" else (
        "vagal" if trait in ("HRV_RMSSD", "RestingHeartRate") else "conduction")
    col = FAMILY_COLOR[fam]
    axe1.scatter([x["value"]], [r], s=11, c=col, linewidths=0, zorder=4,
                 marker="o" if state == "SAN_P_cell" else "^")
    axe1.text(0.49, r, f"{TRAIT_LABEL[trait]}, "
              f"{'SAN' if state == 'SAN_P_cell' else 'AVN'}", fontsize=FS_MIN,
              va="center", ha="right", color=col, clip_on=False)
    rows_e.append(dict(panel="e1 fragility", trait=TRAIT_LABEL[trait],
                       cell_state=STATE_LABEL[state].replace("\n", " "),
                       fraction_retained=round(x["value"], 4)))
axe1.text(1.24, -0.55, "above 1, the effect grows", fontsize=FS_MIN,
          ha="right", va="bottom", color=GREY_TEXT)

GS = load("scdrs/audit_robustness.json")["gene_set_size"]
SIZES = ["250", "500", "1000", "2000"]
axe2 = ax_mm(fig, 140.0, 12.0, 34.0, 26.0)
axe2.set_xlim(-0.3, 3.3)
axe2.set_ylim(20, 0.2)
axe2.set_ylabel("rank among cell states", fontsize=6, labelpad=2)
axe2.tick_params(labelsize=FS_MIN)
axe2.set_xticks(range(4))
axe2.set_xticklabels(SIZES, fontsize=FS_MIN)
axe2.set_xlabel("genes in the set", fontsize=6, labelpad=1.5)

FAM_OF = {"EducationalAttainment": "cognitive", "HRV_RMSSD": "vagal",
          "RestingHeartRate": "vagal", "PRinterval": "conduction",
          "AtrialFibrillation": "atrial"}
for state, mk in [("SAN_P_cell", "o"), ("AVN_P_cell", "^")]:
    for trait, per_size in GS[state].items():
        col = FAMILY_COLOR[FAM_OF[trait]]
        if FAM_OF[trait] == "atrial":
            col = FAMILY_INK["atrial"]
        y = [per_size[s] for s in SIZES]
        axe2.plot(range(4), y, color=col, linewidth=0.5, alpha=0.9, zorder=2)
        axe2.scatter(range(4), y, s=6, c=col, marker=mk, linewidths=0, zorder=3)
        rows_e.append(dict(panel="e2 gene-set size", trait=TRAIT_LABEL[trait],
                           cell_state=STATE_LABEL[state].replace("\n", " "),
                           **{f"top_{s}": per_size[s] for s in SIZES}))
for yv, mk, lab in [(13.0, "o", "sinoatrial"), (15.6, "^", "atrioventricular")]:
    axe2.scatter([2.35], [yv], s=6, c=GREY_TEXT, marker=mk, linewidths=0)
    axe2.text(2.5, yv, lab, fontsize=FS_MIN, va="center", ha="left",
              color=GREY_TEXT)
letter_mm(fig, 66.0, 46.0, "E")
letter_mm(fig, 126.0, 46.0, "F")

save_figure(fig, "Fig3", source_data={
    "a_immune_control": rows_a, "b_gene_length": rows_b,
    "c_neural_programme": rows_c, "d_glutamatergic": rows_d,
    "e_fragility_and_size": rows_e})
