"""Figure 2. Cognitive-trait heritability is enriched in nodal pacemaker cells.

a  19 traits x 4 conduction populations, within-stratum AUC against the myocyte
   lineage of the same donor, region and chemistry.
b  Effect sizes with 1.96 SE and the minimum detectable effect of each column.
c  Ambient-RNA median split of the 245 sinoatrial pacemaker cells.
d  Where the result came from: educational attainment across the 62 cell states
   of the single-chemistry nodal dataset.

Reads results/ only; computes no new statistic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from matplotlib.patches import Rectangle

from _style import (ACC_POWER, AXIS_STATES, CMAP_AUC, FAMILY_COLOR,
                    FAMILY_LABEL, FS_MIN, GREY_FILL, GREY_RULE, GREY_TEXT, INK, MDE,
                    N_CELLS, NORM_AUC, RESULTS, SE, STATE_LABEL, TRAIT_ORDER,
                    UNDERPOWERED, auc_colorbar, ax_mm, bh_flags, letter_mm,
                    allow_offaxis, load, new_figure, save_figure, text_mm,
                    sig_square, zero_line)

W, H = 183.0, 178.0
fig = new_figure(W, H)

SA = load("scdrs_axis/axis_interaction.json")["stratified_auc"]
FLAG = bh_flags()

# --------------------------------------------------------------------------- #
# a  heatmap
# --------------------------------------------------------------------------- #
HM_X, HM_W = 37.0, 46.0
ROW_MM = 3.1
HM_Y, HM_H = 102.0, ROW_MM * 22.7   # 19 trait rows plus the annotation zone

ax = ax_mm(fig, HM_X, HM_Y, HM_W, HM_H)
n_row = len(TRAIT_ORDER)
ax.set_xlim(0, len(AXIS_STATES))
ax.set_ylim(n_row, -3.7)
allow_offaxis(ax, "x")
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)

rows_a = []
for r, (key, label, fam) in enumerate(TRAIT_ORDER):
    for c, state in enumerate(AXIS_STATES):
        v = SA[key][state]
        auc, z = v["auc"], v["z"]
        rgba = CMAP_AUC(NORM_AUC(auc))
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=rgba,
                               edgecolor="white", linewidth=0.4))
        if state != UNDERPOWERED:   # this column carries no inference either way
            flag = FLAG[(key, state)]
            sig_square(ax, c, r, flag, inset=(0.035, 0.115))
            # a diverging fill loses its sign in greyscale, so every cell that
            # carries an inference also prints its signed z
            if flag:
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(c + 0.5, r + 0.54, f"{z:+.1f}".replace("-", "−"),
                        ha="center", va="center", fontsize=FS_MIN,
                        color="white" if lum < 0.45 else INK, zorder=7)
        rows_a.append(dict(trait=label, family=fam,
                           cell_state=STATE_LABEL[state].replace("\n", " "),
                           n_cells=N_CELLS[state], AUC=round(auc, 4),
                           z=round(z, 3), significance=FLAG[(key, state)] or "ns"))

# trait labels and family band
BAND_X, BAND_W = -0.42, 0.30
for r, (key, label, fam) in enumerate(TRAIT_ORDER):
    style = dict(fontsize=6, color=INK)
    if fam == "cognitive":
        style.update(fontweight="bold")
    if fam == "control":
        style.update(style="italic", color=GREY_TEXT)
    ax.text(BAND_X - 0.10, r + 0.5, label, ha="right", va="center",
            clip_on=False, **style)

fams = [f for _, _, f in TRAIT_ORDER]
start = 0
for i in range(1, len(fams) + 1):
    if i == len(fams) or fams[i] != fams[start]:
        ax.add_patch(Rectangle((BAND_X, start + 0.12), BAND_W, i - start - 0.24,
                               facecolor=FAMILY_COLOR[fams[start]],
                               edgecolor="none", clip_on=False))
        start = i

# column headers
for c, state in enumerate(AXIS_STATES):
    ax.text(c + 0.5, -1.05, STATE_LABEL[state], ha="center", va="bottom",
            fontsize=6, linespacing=1.35)
    ax.text(c + 0.5, -0.22, f"n = {N_CELLS[state]}", ha="center", va="bottom",
            fontsize=FS_MIN, color=GREY_TEXT)

# the two nodal columns cannot be told apart
ax.add_patch(Rectangle((0.98, 0), 0.04, n_row, facecolor="white",
                       edgecolor="none", zorder=7))
ax.plot([0.08, 0.08, 1.92, 1.92], [-2.62, -2.86, -2.86, -2.62],
        color=GREY_RULE, linewidth=0.5, clip_on=False, zorder=8)
ax.text(1.0, -2.96, "not resolvable", ha="center", va="bottom",
        fontsize=FS_MIN, color=GREY_TEXT, clip_on=False)

# the AV bundle column supports no inference
ax.add_patch(Rectangle((2, 0), 1, n_row, fill=False, hatch="/////",
                       edgecolor=ACC_POWER, linewidth=0, zorder=5, alpha=0.85))
ax.add_patch(Rectangle((2, 0), 1, n_row, fill=False, edgecolor=ACC_POWER,
                       linewidth=0.6, zorder=6))
ax.plot([2.08, 2.08, 2.92, 2.92], [-2.62, -2.86, -2.86, -2.62],
        color=ACC_POWER, linewidth=0.5, clip_on=False, zorder=8)
ax.text(2.5, -2.96, "underpowered", ha="center", va="bottom", fontsize=FS_MIN,
        color=ACC_POWER, clip_on=False)

cax = ax_mm(fig, HM_X + HM_W + 2.8, HM_Y + 14, 2.0, 26)
auc_colorbar(fig, cax)
letter_mm(fig, 1.5, H - 2.0, "A")

# --------------------------------------------------------------------------- #
# shared key strip between the rows
# --------------------------------------------------------------------------- #
axk = ax_mm(fig, 2.0, 90.0, 112.0, 8.0)
axk.set_xlim(0, 112)
axk.set_ylim(0, 8)
axk.axis("off")
x = 0.0
for fam in ["cognitive", "vagal", "conduction", "atrial", "ventricular",
            "control"]:
    axk.add_patch(Rectangle((x, 5.0), 1.6, 2.2, facecolor=FAMILY_COLOR[fam],
                            edgecolor="none"))
    axk.text(x + 2.4, 6.1, FAMILY_LABEL[fam], fontsize=FS_MIN, va="center",
             ha="left", color=GREY_TEXT)
    # 0.97 mm per character is measured Arial 5 pt; the old 0.84 left the
    # longest label almost touching the next swatch at 50% reduction
    x += 3.4 + 0.97 * len(FAMILY_LABEL[fam])
axk.add_patch(Rectangle((0.3, 1.05), 2.0, 1.6, fill=False, edgecolor=INK,
                        linewidth=0.9))
axk.text(3.1, 1.85, "FDR < 0.05", fontsize=FS_MIN, va="center",
         color=GREY_TEXT)
axk.add_patch(Rectangle((17.0, 1.05), 2.0, 1.6, fill=False, edgecolor=INK,
                        linewidth=0.5, linestyle=(0, (1.4, 1.2))))
axk.text(19.8, 1.85, "nominal only", fontsize=FS_MIN, va="center",
         color=GREY_TEXT)
axk.text(36.0, 1.85, "error bars, ± 1.96 SE", fontsize=FS_MIN, va="center",
         color=GREY_TEXT)
axk.add_patch(Rectangle((59.0, 1.15), 1.6, 1.5, facecolor=GREY_FILL,
                        edgecolor="none"))
axk.text(61.5, 1.85, "below the minimum detectable effect", fontsize=FS_MIN,
         va="center", color=GREY_TEXT)

# --------------------------------------------------------------------------- #
# b  forest, effect size against detectability
# --------------------------------------------------------------------------- #
FOREST = ["EducationalAttainment", "Intelligence", "ReactionTime",
          "HRV_SDNN", "RheumatoidArthritis"]
FOREST_LAB = {"EducationalAttainment": "Educational\nattainment",
              "Intelligence": "Intelligence",
              "ReactionTime": "Reaction time",
              "HRV_SDNN": "HRV, SDNN",
              "RheumatoidArthritis": "Rheumatoid\narthritis"}

rows_b = []
for i, state in enumerate(["SAN_P_cell", "AVN_P_cell"]):
    axf = ax_mm(fig, 116.0 + i * 33.0, 106.0, 29.0, 52.0)
    axf.set_xlim(0.28, 0.82)
    axf.set_ylim(len(FOREST) - 0.45, -0.55)
    axf.axvspan(0.5, MDE[state], facecolor=GREY_FILL, edgecolor="none",
                zorder=0)
    axf.axvline(MDE[state], color=GREY_RULE, linewidth=0.5,
                linestyle=(0, (1, 1.6)), zorder=1)
    zero_line(axf, 0.5, axis="x")
    axf.set_xticks([0.3, 0.5, 0.7])
    axf.tick_params(labelsize=FS_MIN)
    axf.set_yticks([])
    axf.spines["left"].set_visible(False)
    axf.set_xlabel("AUC vs myocyte lineage", fontsize=6, labelpad=1.5)
    axf.set_title(f"{STATE_LABEL[state].replace(chr(10), ' ')}, n = {N_CELLS[state]}",
                  fontsize=6, pad=3.0)
    se = SE[state]
    for r, key in enumerate(FOREST):
        v = SA[key][state]
        auc = v["auc"]
        lo, hi = auc - 1.96 * se, auc + 1.96 * se
        is_ctrl = key == "RheumatoidArthritis"
        col = GREY_TEXT if is_ctrl else INK
        axf.plot([lo, hi], [r, r], color=col, linewidth=0.6,
                 solid_capstyle="butt")
        axf.scatter([auc], [r], s=9, c=col, zorder=5,
                    marker="s" if is_ctrl else "o", linewidths=0)
        if i == 0:
            axf.text(0.25, r, FOREST_LAB[key], ha="right", va="center",
                     fontsize=6, clip_on=False, linespacing=1.35,
                     color=GREY_TEXT if is_ctrl else INK,
                     fontweight="bold" if key in ("EducationalAttainment",
                                                  "Intelligence",
                                                  "ReactionTime") else "normal")
        rows_b.append(dict(trait=FOREST_LAB[key].replace("\n", " "),
                           cell_state=STATE_LABEL[state].replace("\n", " "),
                           AUC=round(auc, 4), SE=round(se, 4),
                           CI_low=round(lo, 4), CI_high=round(hi, 4),
                           MDE=round(MDE[state], 4)))

letter_mm(fig, 99.0, H - 2.0, "B")

# --------------------------------------------------------------------------- #
# c  the distributions the within-stratum statistic is computed from
# --------------------------------------------------------------------------- #
obs = pd.read_csv(RESULTS / "obs_axis.tsv", sep="\t", index_col=0)
obs["stratum"] = (obs.donor_id.astype(str) + "|" + obs.region.astype(str)
                  + "|" + obs.assay.astype(str))
is_cm = obs.cell_state.str.match(r"^(aCM|vCM)")
pool = is_cm | obs.cell_state.isin(AXIS_STATES)          # the comparator pool
san = obs.cell_state == "SAN_P_cell"
strata = sorted(obs.loc[san, "stratum"].unique(),
                key=lambda s: -int(san[obs.stratum == s].sum()))
assert len(strata) == 6 and int(san.sum()) == 245, (len(strata), int(san.sum()))

rows_c = []
PANEL_C = [("EducationalAttainment", "Educational attainment", (-3.1, 5.6)),
           ("RheumatoidArthritis", "Rheumatoid arthritis", (-5.9, 3.7))]
for i, (key, lab, xlim) in enumerate(PANEL_C):
    sc = pd.read_csv(RESULTS / "scdrs_axis" / f"{key}.score.tsv", sep="\t",
                     index_col=0)["norm_score"].reindex(obs.index)
    axv = ax_mm(fig, 30.0 + i * 74.0, 56.0, 68.0, 26.0)
    axv.set_xlim(*xlim)
    axv.set_ylim(len(strata) - 0.35, -1.95)
    axv.set_yticks([])
    axv.spines["left"].set_visible(False)
    axv.tick_params(labelsize=FS_MIN)
    axv.set_xlabel("scDRS score of the individual cell", fontsize=6,
                   labelpad=1.5)
    axv.set_title(lab, fontsize=6, pad=2.5,
                  color=FAMILY_COLOR["cognitive"] if i == 0 else GREY_TEXT)
    zero_line(axv, 0.0, axis="x")
    for r, st in enumerate(strata):
        here = obs.stratum == st
        comp = sc[here & pool & ~san].dropna().to_numpy()
        foc = sc[here & san].dropna().to_numpy()
        parts = axv.violinplot([comp], positions=[r], vert=False,
                               widths=0.82, showextrema=False, showmedians=False)
        for b in parts["bodies"]:
            b.set_facecolor(GREY_FILL)
            b.set_edgecolor(GREY_RULE)
            b.set_linewidth(0.4)
            b.set_alpha(1.0)
        col = FAMILY_COLOR["cognitive"] if i == 0 else GREY_TEXT
        axv.scatter(foc, [r - 0.22] * len(foc), s=1.6, c=col, linewidths=0,
                    alpha=0.85, zorder=4)
        if i == 0:
            axv.text(-3.35, r, f"{st.split('|')[0]},  {len(foc)} vs {len(comp):,}",
                     fontsize=FS_MIN, ha="right", va="center", color=GREY_TEXT,
                     clip_on=False)
        rows_c.append(dict(trait=lab, stratum=st, n_pacemaker=len(foc),
                           n_comparator=len(comp),
                           median_pacemaker=round(float(pd.Series(foc).median()), 4),
                           median_comparator=round(float(pd.Series(comp).median()), 4)))
    if i == 0:
        axv.add_patch(Rectangle((-3.0, -1.75), 0.8, 0.34, facecolor=GREY_FILL,
                                edgecolor=GREY_RULE, linewidth=0.4))
        axv.text(-1.95, -1.58, "myocyte lineage, same stratum",
                 fontsize=FS_MIN, va="center", ha="left", color=GREY_TEXT)
        axv.scatter([2.05], [-1.58], s=1.6 * 6, c=FAMILY_COLOR["cognitive"],
                    linewidths=0)
        axv.text(2.4, -1.58, "pacemaker cells", fontsize=FS_MIN,
                 va="center", ha="left", color=GREY_TEXT)
text_mm(fig, 101.0, 47.5, "independent axes: within-violin position is the "
        "comparison", fontsize=FS_MIN, ha="center", va="bottom", color=GREY_TEXT)
letter_mm(fig, 1.5, 86.0, "C")

# --------------------------------------------------------------------------- #
# d  ambient-RNA median split
# --------------------------------------------------------------------------- #
AMB = load("ambient_stratified.json")
axc = ax_mm(fig, 37.0, 11.0, 58.0, 28.0)
axc.set_xlim(0.42, 0.80)
axc.set_ylim(len(FOREST) + 0.85, -0.55)
zero_line(axc, 0.5, axis="x")
axc.set_yticks([])
axc.spines["left"].set_visible(False)
axc.tick_params(labelsize=FS_MIN)
axc.set_xticks([0.45, 0.55, 0.65, 0.75])
axc.set_xlabel("AUC vs myocyte lineage, sinoatrial pacemaker cells",
               fontsize=6, labelpad=1.5)

rows_d = []
for r, key in enumerate(FOREST):
    d = AMB[key]
    col = GREY_TEXT if key == "RheumatoidArthritis" else INK
    axc.plot([d["carrying"], d["all"]], [r, r], color=GREY_RULE, linewidth=0.6,
             zorder=1, solid_capstyle="butt")
    axc.plot([d["clean"], d["all"]], [r, r], color=GREY_RULE, linewidth=0.6,
             zorder=1, solid_capstyle="butt")
    # three series, three markers: two open circles were not tellable apart
    axc.scatter([d["all"]], [r], s=9, facecolors="white", edgecolors=col,
                marker="o", linewidths=0.5, zorder=4)
    axc.scatter([d["carrying"]], [r], s=10, facecolors="white", edgecolors=col,
                marker="s", linewidths=0.5, zorder=4)
    axc.scatter([d["clean"]], [r], s=13, c=col, zorder=5, linewidths=0)
    axc.text(0.408, r, FOREST_LAB[key].replace("\n", " "), ha="right",
             va="center", fontsize=6, color=col, clip_on=False)
    rows_d.append(dict(trait=FOREST_LAB[key].replace("\n", " "),
                       all_245=round(d["all"], 4), z_all=round(d["z_all"], 3),
                       low_load_125=round(d["clean"], 4),
                       z_low_load=round(d["z_clean"], 3),
                       high_load_120=round(d["carrying"], 4),
                       z_high_load=round(d["z_carrying"], 3)))

for j, (lab, mk) in enumerate([
        ("low-neural-load half, n = 125", None),
        ("all sinoatrial pacemaker cells, n = 245", "o"),
        ("high-neural-load half, n = 120", "s")]):
    yv = 4.62 + j * 0.42
    if mk is None:
        axc.scatter([0.432], [yv], s=13, c=INK, linewidths=0, clip_on=False)
    else:
        axc.scatter([0.432], [yv], s=9 if mk == "o" else 10, facecolors="white",
                    edgecolors=INK, marker=mk, linewidths=0.5, clip_on=False)
    axc.text(0.443, yv, lab, fontsize=FS_MIN, va="center", ha="left",
             color=GREY_TEXT, clip_on=False)
letter_mm(fig, 1.5, 44.0, "D")

# --------------------------------------------------------------------------- #
# d  origin: educational attainment across the 62 nodal cell states
# --------------------------------------------------------------------------- #
g = pd.read_csv(RESULTS / "scdrs" / "group_analysis_fdr.tsv", sep="\t")
edu = g[g.trait == "EducationalAttainment"].sort_values("assoc_mcz",
                                                        ascending=False)
axd = ax_mm(fig, 116.0, 11.0, 62.0, 28.0)
axd.set_xlim(-2.7, 6.4)
axd.set_ylim(0, 1.28)
axd.set_yticks([])
axd.spines["left"].set_visible(False)
axd.tick_params(labelsize=FS_MIN)
axd.set_xlabel("scDRS association z across 62 cell states,\nnodal dataset",
               fontsize=6, labelpad=1.5, linespacing=1.4)
zero_line(axd, 0.0, axis="x")

HL = {"NC2_glial_NGF+": ("NC2 glial", 3, 0.70),
      "SAN_P_cell": ("SAN pacemaker", 2, 0.90),
      "NC1_glial": ("NC1 glial", 1, 1.10)}
rows_e = []
for _, row in edu.iterrows():
    z, name = row.assoc_mcz, row.cell_state
    if name in HL:
        continue
    sig = row.assoc_fdr < 0.05
    axd.plot([z, z], [0.08, 0.08 + (0.34 if sig else 0.24)],
             color=INK if sig else GREY_RULE, linewidth=0.6,
             solid_capstyle="butt", zorder=2)
for name, (lab, rank, ty) in HL.items():
    z = float(edu.loc[edu.cell_state == name, "assoc_mcz"].iloc[0])
    is_san = name == "SAN_P_cell"
    axd.plot([z, z], [0.08, 0.54], color=INK, linewidth=1.1 if is_san else 0.7,
             solid_capstyle="butt", zorder=4)
    axd.text(1.4, ty, f"{lab}  #{rank}", fontsize=FS_MIN, ha="right",
             va="center", color=INK,
             fontweight="bold" if is_san else "normal")
    axd.plot([1.6, z], [ty, 0.54], color=GREY_TEXT, linewidth=0.4, zorder=1)
for _, row in edu.iterrows():
    rows_e.append(dict(cell_state=row.cell_state, n_cells=int(row.n_cell),
                       z=round(row.assoc_mcz, 3), p=row.assoc_mcp,
                       FDR=round(row.assoc_fdr, 4)))
axd.text(-2.6, 1.26, "black ticks, FDR < 0.05", fontsize=FS_MIN,
         color=GREY_TEXT, ha="left", va="top")
letter_mm(fig, 99.0, 44.0, "E")

save_figure(fig, "Fig2", source_data={
    "a_heatmap": rows_a, "b_forest": rows_b,
    "c_within_stratum": rows_c, "d_ambient_split": rows_d,
    "e_nodal_ranking": rows_e})
