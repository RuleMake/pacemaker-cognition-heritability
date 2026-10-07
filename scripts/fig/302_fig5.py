"""Figure 6. The result does not rest on one donor, and what crosses species.

a  Leave-one-donor-out range around each estimate.
b  Direction in every donor separately.
c  The one donor that shows nothing, and its cell identity.
d  Pre-registered mouse scorecard, including the prediction that failed.
e  Whether the cross-species signal is the cardiovascular signal relabelled.

Reads results/ only; computes no new statistic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.patches import Rectangle

from _style import (ACC_FAIL, ACC_NEG, ACC_POWER, CMAP_AUC, FAMILY_COLOR,
                    FS_MIN, GREY_FILL, GREY_RULE, GREY_TEXT, INK, NORM_AUC,
                    allow_offaxis, auc_colorbar, ax_mm, letter_mm, load,
                    new_figure,
                    save_figure, text_mm, zero_line)

W, H = 183.0, 122.0
fig = new_figure(W, H)

LODO = load("leave_one_donor_out.json")
QC = load("donor_identity_qc.json")
MREP = load("mouse_replication.json")
MPRED = load("mouse_predictions.json")
MAUD = load("mouse_audit.json")
STRAT = load("scdrs_axis/axis_interaction.json")["stratified_auc"]

ROWS = [("ReactionTime", "Reaction time", "cognitive"),
        ("Intelligence", "Intelligence", "cognitive"),
        ("EducationalAttainment", "Educational attainment", "cognitive"),
        ("HRV_SDNN", "HRV, SDNN", "vagal"),
        ("RestingHeartRate", "Resting heart rate", "vagal"),
        ("QTinterval", "QT interval", "ventricular"),
        ("RheumatoidArthritis", "Rheumatoid arthritis", "control")]

# --------------------------------------------------------------------------- #
# a  leave-one-donor-out
# --------------------------------------------------------------------------- #
axa = ax_mm(fig, 12.0, 68.0, 30.0, 45.0)
axa.set_xlim(0.29, 0.82)
axa.set_ylim(len(ROWS), -2.4)   # same row grid as b, so the labels serve both
axa.set_yticks([])
axa.spines["left"].set_visible(False)
axa.tick_params(labelsize=FS_MIN)
axa.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
axa.set_xlabel("AUC vs myocyte lineage", fontsize=6, labelpad=1.5)
# stop the 0.5 rule below the key, which it otherwise runs through
axa.plot([0.5, 0.5], [-0.45, 7.0], color=GREY_RULE, linewidth=0.5,
         linestyle=(0, (3, 2)), zorder=1)

# the label column is centred in the 28 mm gutter between the two panels
GUTTER = 0.82 + (14.0 / 30.0) * (0.82 - 0.29)

rows_a = []
for r, (key, lab, fam) in enumerate(ROWS):
    col = FAMILY_COLOR[fam]
    for state, dy, mk in [("SAN_P_cell", 0.30, "o"), ("AVN_P_cell", 0.70, "^")]:
        d = LODO["lodo"][f"{key}|{state}"]
        axa.plot([d["lodo_min"], d["lodo_max"]], [r + dy, r + dy], color=col,
                 linewidth=0.6, solid_capstyle="butt", zorder=2)
        axa.scatter([d["full"]], [r + dy], s=9, c=col, marker=mk, zorder=4,
                    linewidths=0)
        rows_a.append(dict(trait=lab, cell_state=state, full=round(d["full"], 4),
                           lodo_min=round(d["lodo_min"], 4),
                           lodo_max=round(d["lodo_max"], 4),
                           donor_dropped_at_min=d["worst_donor"]))
    axa.text(GUTTER, r + 0.5, lab, ha="center", va="center", fontsize=6,
             color=col, clip_on=False)
axa.scatter([0.305], [-1.70], s=9, c=INK, marker="o", linewidths=0)
axa.text(0.323, -1.70, "sinoatrial", fontsize=FS_MIN, va="center", color=INK)
axa.scatter([0.545], [-1.70], s=9, c=INK, marker="^", linewidths=0)
axa.text(0.563, -1.70, "atrioventricular", fontsize=FS_MIN, va="center",
         color=INK)
axa.text(0.295, -1.00, "line, leave-one-donor-out range",
         fontsize=FS_MIN, va="center", ha="left", color=GREY_TEXT)
letter_mm(fig, 1.5, H - 2.0, "A")

# --------------------------------------------------------------------------- #
# b  every donor separately
# --------------------------------------------------------------------------- #
COLS = ([("SAN_P_cell", d) for d in LODO["donors_of"]["SAN_P_cell"]]
        + [("AVN_P_cell", d) for d in LODO["donors_of"]["AVN_P_cell"]])
axb = ax_mm(fig, 70.0, 68.0, 46.0, 45.0)
axb.set_xlim(0, len(COLS))
axb.set_ylim(len(ROWS), -2.4)
allow_offaxis(axb, "y")
axb.set_xticks([])
axb.set_yticks([])
for sp in axb.spines.values():
    sp.set_visible(False)

rows_b = []
for r, (key, lab, fam) in enumerate(ROWS):
    n_above = 0
    for c, (state, donor) in enumerate(COLS):
        v = LODO["per_donor"][f"{key}|{state}"]["per_donor"][donor]
        axb.add_patch(Rectangle((c, r), 1, 1, facecolor=CMAP_AUC(NORM_AUC(v)),
                                edgecolor="white", linewidth=0.4))
        if v > 0.5:
            n_above += 1
        rows_b.append(dict(trait=lab, cell_state=state, donor=donor,
                           AUC=round(v, 4)))
    axb.text(len(COLS) + 0.25, r + 0.5, f"{n_above}/{len(COLS)}", fontsize=FS_MIN,
             va="center", ha="left", clip_on=False,
             color=GREY_TEXT if key == "RheumatoidArthritis" else INK)
axb.text(len(COLS) + 0.25, -1.45, "above 0.5", fontsize=FS_MIN, va="bottom",
         ha="left", color=GREY_TEXT, clip_on=False)

for c, (state, donor) in enumerate(COLS):
    axb.text(c + 0.5, -0.15, donor, rotation=90, fontsize=FS_MIN, va="bottom",
             ha="center", color=GREY_TEXT)
axb.plot([0.05, 5.95], [-1.35, -1.35], color=GREY_RULE, linewidth=0.5,
         clip_on=False)
axb.plot([6.05, 8.95], [-1.35, -1.35], color=GREY_RULE, linewidth=0.5,
         clip_on=False)
axb.text(3.0, -1.45, "sinoatrial", fontsize=FS_MIN, ha="center", va="bottom",
         color=GREY_TEXT)
axb.text(7.5, -1.45, "atrioventricular", fontsize=FS_MIN, ha="center",
         va="bottom", color=GREY_TEXT)
axb.text(0.5, len(ROWS) + 0.55, "donor A61", fontsize=FS_MIN, color=ACC_POWER,
         ha="center", va="top", rotation=0, clip_on=False)
axb.plot([0.5, 0.5], [len(ROWS) + 0.05, len(ROWS) + 0.3], color=ACC_POWER,
         linewidth=0.5, clip_on=False)
cb = ax_mm(fig, 82.0, 61.5, 22.0, 1.8)
auc_colorbar(fig, cb, label="AUC in that donor", orientation="horizontal")
letter_mm(fig, 66.0, H - 2.0, "B")

# --------------------------------------------------------------------------- #
# c  the donor that shows nothing
# --------------------------------------------------------------------------- #
axc = ax_mm(fig, 138.0, 68.0, 39.0, 45.0)
axc.set_xlim(0.58, 0.99)
axc.set_ylim(0.36, 0.90)
axc.tick_params(labelsize=FS_MIN)
axc.set_xlabel("pacemaker marker-panel AUC\n(cell identity, per donor)",
               fontsize=6, labelpad=1.5, linespacing=1.4)
axc.set_ylabel("educational attainment AUC", fontsize=6, labelpad=2)
zero_line(axc, 0.5)

OFFSET = {"A61": (0.0, -0.020, "center", "top"),
          "AH1": (0.008, 0.0, "left", "center"),
          "AH2": (-0.008, 0.0, "right", "center"),
          "AV10": (0.0, 0.014, "center", "bottom"),
          "AV14": (0.0, 0.014, "center", "bottom"),
          "AV3": (0.0, -0.016, "center", "top")}
rows_c = []
for donor, d in QC["per_donor"].items():
    x, y = d["panel"]["auc"], d["edu"]
    is61 = donor == "A61"
    axc.scatter([x], [y], s=16, c=ACC_POWER if is61 else INK, zorder=4,
                linewidths=0)
    dx, dy, ha, va = OFFSET[donor]
    axc.text(x + dx, y + dy, donor, fontsize=FS_MIN, ha=ha, va=va,
             color=ACC_POWER if is61 else GREY_TEXT)
    rows_c.append(dict(donor=donor, panel_AUC=round(x, 4),
                       panel_z=round(d["panel"]["z"], 3),
                       n_cells=d["panel"]["n_cells"],
                       educational_attainment_AUC=round(y, 4),
                       identity_ok=d["identity_ok"]))
axc.text(0.595, 0.895, "suggestive only, not counted", fontsize=FS_MIN,
         va="top", ha="left", color=GREY_TEXT)
letter_mm(fig, 127.0, H - 2.0, "C")

# --------------------------------------------------------------------------- #
# d  pre-registered mouse scorecard
# --------------------------------------------------------------------------- #
BAR = MPRED["bar"]
GROUPS = [("P1", "gate", "P1_gate", "AUC > 0.55 in mouse"),
          ("P2", "test", "P2_test", "AUC > 0.55 in mouse"),
          ("P3", "negative control", "P3_negative", "0.45 < AUC < 0.55"),
          ("P4", "signed prediction", "P4_signed", "AUC ≤ 0.55")]
LAB = {"RestingHeartRate": "Resting heart rate", "HRV_SDNN": "HRV, SDNN",
       "HRV_RMSSD": "HRV, RMSSD", "EducationalAttainment": "Educational attainment",
       "Intelligence": "Intelligence", "ReactionTime": "Reaction time",
       "RheumatoidArthritis": "Rheumatoid arthritis", "QTinterval": "QT interval"}

order, group_of = [], {}
for tag, _, key, _ in GROUPS:
    for t in MPRED[key]["traits"]:
        order.append(t)
        group_of[t] = tag
n = len(order)

axh = ax_mm(fig, 39.0, 13.0, 30.0, 33.0)
axm = ax_mm(fig, 74.0, 13.0, 30.0, 33.0)
for a, lo, hi, ttl in [(axh, 0.44, 0.74, "human\nsinoatrial pacemaker"),
                       (axm, 0.44, 0.90, "mouse\nnodal, embryonic day 16.5")]:
    a.set_xlim(lo, hi)
    a.set_ylim(n - 0.4, -0.6)
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.tick_params(labelsize=FS_MIN)
    a.set_title(ttl, fontsize=6, pad=2.5, linespacing=1.4)
    a.set_xlabel("AUC vs working myocardium", fontsize=6, labelpad=1.5)
    zero_line(a, 0.5, axis="x")
axh.set_xticks([0.5, 0.6, 0.7])
axm.set_xticks([0.5, 0.7, 0.9])
allow_offaxis(axh, "x")
axm.axvline(BAR, color=INK, linewidth=0.5, linestyle=(0, (2, 1.6)), zorder=1)
axm.text(BAR + 0.018, 3.0, "pre-registered bar", fontsize=FS_MIN, ha="center",
         va="center", color=INK, rotation=90)

rows_d = []
for r, t in enumerate(order):
    tag = group_of[t]
    mouse = MREP["results"][t]["auc"]
    human = STRAT[t]["SAN_P_cell"]["auc"]
    if tag == "P3":
        passed = 0.45 < mouse < 0.55
    elif tag == "P4":
        passed = mouse <= BAR
    else:
        passed = mouse > BAR
    col = ACC_FAIL if not passed else (ACC_NEG if tag == "P3" else INK)
    axh.scatter([human], [r], s=12, c=GREY_TEXT, linewidths=0, zorder=4)
    axm.scatter([mouse], [r], s=14, c=col, linewidths=0, zorder=4)
    # outside the axis: the largest mouse AUC is 0.856, so nothing reaches here
    axm.text(0.925, r, "pass" if passed else "fail", fontsize=FS_MIN,
             ha="left", va="center", color=col, clip_on=False,
             fontweight="normal" if passed else "bold")
    axh.text(0.420, r, LAB[t], fontsize=6, ha="right", va="center",
             color=col if not passed else INK, clip_on=False)
    rows_d.append(dict(prediction=tag, trait=LAB[t],
                       human_SAN_AUC=round(human, 4),
                       mouse_AUC=round(mouse, 4),
                       mouse_z=round(MREP["results"][t]["z"], 3),
                       outcome="pass" if passed else "fail"))

for tag, name, key, claim in GROUPS:
    idx = [i for i, t in enumerate(order) if group_of[t] == tag]
    y0, y1 = min(idx) - 0.32, max(idx) + 0.32
    col = ACC_FAIL if tag == "P4" else GREY_TEXT
    axh.plot([0.153, 0.145, 0.145, 0.153], [y0, y0, y1, y1], color=col,
             linewidth=0.5, clip_on=False)
    axh.text(0.137, (y0 + y1) / 2, tag, fontsize=FS_MIN, ha="right",
             va="center", color=col, clip_on=False)

text_mm(fig, 71.5, 4.5, "independent axes: pattern comparable, magnitude not",
        fontsize=FS_MIN, ha="center", va="bottom", color=GREY_TEXT)
letter_mm(fig, 1.5, 57.0, "D")

# --------------------------------------------------------------------------- #
# e  is the cross-species signal the cardiovascular signal relabelled?
# --------------------------------------------------------------------------- #
pairs = list(MAUD["overlap"].keys())
axe1 = ax_mm(fig, 122.0, 13.0, 13.0, 33.0)
axe1.set_ylim(0, 0.10)
axe1.set_xlim(-0.6, 0.6)
axe1.set_xticks([])
axe1.tick_params(labelsize=FS_MIN)
axe1.set_ylabel("gene-set Jaccard", fontsize=6, labelpad=2)
# the only panel of e with no categorical x, so the strip names itself
axe1.set_xlabel("9 trait pairs", fontsize=FS_MIN, labelpad=6)
axe1.set_title("pairwise\ngene-set overlap", fontsize=6, pad=2.5,
               linespacing=1.4)
rng_x = [-0.34, 0.17, 0.0, -0.17, 0.34, -0.26, -0.09, 0.09, 0.26]
rows_e = []
for i, p in enumerate(pairs):
    j = MAUD["overlap"][p]["jaccard"]
    axe1.scatter([rng_x[i]], [j], s=8, c=INK, linewidths=0, zorder=3)
    rows_e.append(dict(panel="e1 gene-set overlap", pair=p, jaccard=round(j, 4)))
axe1.axhline(MAUD["mean_jaccard"], color=GREY_RULE, linewidth=0.5,
             linestyle=(0, (3, 2)))
# two lines, so the label needs a clear lane only right of x = 0.16
axe1.text(0.58, MAUD["mean_jaccard"] + 0.0014,
          f"mean\n{MAUD['mean_jaccard']:.3f}", fontsize=FS_MIN, ha="right",
          va="bottom", color=GREY_TEXT, linespacing=1.4)

axe2 = ax_mm(fig, 146.0, 13.0, 13.0, 33.0)
axe2.set_ylim(0, 0.40)
axe2.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
axe2.set_xlim(-0.9, 1.9)
axe2.set_xticks([0, 1])
axe2.set_xticklabels(["mouse", "human"], fontsize=FS_MIN, rotation=30,
                     ha="right", rotation_mode="anchor")
axe2.tick_params(labelsize=FS_MIN)
axe2.set_ylabel("score correlation", fontsize=6, labelpad=2)
axe2.set_title("within-pacemaker\nscore correlation", fontsize=6, pad=2.5,
               linespacing=1.4)
for i, p in enumerate(pairs):
    c = MAUD["correlation"][p]
    axe2.plot([0, 1], [c["mouse"], c["human"]], color=GREY_RULE, linewidth=0.4,
              zorder=1)
    axe2.scatter([0, 1], [c["mouse"], c["human"]], s=6, c=INK, linewidths=0,
                 zorder=3)
    rows_e.append(dict(panel="e2 score correlation", pair=p,
                       mouse=round(c["mouse"], 4), human=round(c["human"], 4)))

axe3 = ax_mm(fig, 167.0, 13.0, 13.0, 33.0)
axe3.set_ylim(0.46, 0.94)
axe3.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
axe3.set_xlim(-0.7, 1.7)
axe3.set_xticks([0, 1])
axe3.set_xticklabels(["full set", "disjoint"], fontsize=FS_MIN, rotation=30,
                     ha="right", rotation_mode="anchor")
axe3.tick_params(labelsize=FS_MIN)
axe3.set_ylabel("mouse AUC", fontsize=6, labelpad=2)
axe3.set_title("cardiovascular\ngenes removed", fontsize=6, pad=2.5,
               linespacing=1.4)
zero_line(axe3, 0.5)
SHORT = {"ReactionTime": ("React. time", "o"),
         "EducationalAttainment": ("Educ. attain.", "^"),
         "Intelligence": ("Intelligence", "s")}
col = FAMILY_COLOR["cognitive"]
for j, t in enumerate(SHORT):
    d = MAUD["subtraction"][t]
    lab, mk = SHORT[t]
    axe3.plot([0, 1], [d["full_auc"], d["disjoint_auc"]], color=col,
              linewidth=0.6, zorder=2)
    axe3.scatter([0, 1], [d["full_auc"], d["disjoint_auc"]], s=8, c=col,
                 marker=mk, linewidths=0, zorder=3)
    yv = 0.905 - j * 0.028
    axe3.scatter([-0.48], [yv], s=8, c=col, marker=mk, linewidths=0)
    axe3.text(-0.28, yv, lab, fontsize=FS_MIN, va="center", ha="left",
              color=col)
    rows_e.append(dict(panel="e3 gene subtraction", trait=LAB[t],
                       full_AUC=round(d["full_auc"], 4),
                       disjoint_AUC=round(d["disjoint_auc"], 4),
                       full_z=round(d["full_z"], 3),
                       disjoint_z=round(d["disjoint_z"], 3),
                       genes_removed=d["n_removed"], genes_total=d["n_genes"]))
letter_mm(fig, 112.0, 57.0, "E")

save_figure(fig, "Fig5", source_data={
    "a_leave_one_donor_out": rows_a, "b_per_donor": rows_b,
    "c_donor_identity": rows_c, "d_mouse_scorecard": rows_d,
    "e_circularity_audit": rows_e})
