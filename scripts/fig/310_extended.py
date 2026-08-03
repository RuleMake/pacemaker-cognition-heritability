"""Extended Data figures 1 to 7.

Numbered in order of first citation in the manuscript.

S1  gene-length adjustment across every gene set that was rebuilt
S2  locus structure of the gene sets
S3  the spatial route and the arithmetic that closed it
S4  sensitivity to the comparator definition, against sampling noise
S5  how the mouse clusters were annotated
S6  every cell-level test, its direction, and the correction itself
S7  the second Mendelian randomization analysis, drawn by 311_figS7_mr_v2.py

Reads results/ only; computes no new statistic. Counting stored values by sign,
and reading a second stored comparator, are not new statistics.

Display rule followed throughout: a panel carries data, encodings and units.
Anything that is a sentence belongs in the legend, not on the canvas.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from _style import (ACC_FAIL, ACC_NEG, ACC_POWER, ALPHA, AXIS_COLOR,
                    AXIS_STATES, FAMILY_INK, FS_MIN, GREY_FILL, GREY_RULE,
                    GREY_TEXT, INK, MOUSE_CLASS, MOUSE_CLASS_HATCH,
                    RESULTS, SE, STATE_LABEL, SUPPORT,
                    TRAIT_FAMILY, TRAIT_LABEL, UNDERPOWERED, allow_offaxis,
                    ax_mm, letter_mm,
                    load, new_figure, save_figure, text_mm, zero_line)

PRETTY = dict(TRAIT_LABEL)
PRETTY.update({"HRV_SDNNc": "HRV, SDNN corrected",
               "HRV_RMSSDc": "HRV, RMSSD corrected"})
SLATE = SUPPORT[0]


def nice(key):
    base = key.replace("_lenadj", "").replace("_noMHC", "")
    lab = PRETTY.get(base, base)
    if key.endswith("_lenadj"):
        lab += ", length adjusted"
    if key.endswith("_noMHC"):
        lab += ", MHC-free"
    return lab


def fam_ink(key):
    base = key.replace("_lenadj", "").replace("_noMHC", "")
    return FAMILY_INK.get(TRAIT_FAMILY.get(base, "control"), INK)


def state_key(ax, states, x, y0, dy, ha="left", size=FS_MIN):
    """The four-population key, drawn identically wherever it appears."""
    for i, s in enumerate(states):
        under = s == UNDERPOWERED
        ax.scatter([x], [y0 - i * dy], s=10, c=AXIS_COLOR[s], linewidths=0.35,
                   edgecolors=ACC_POWER if under else "white",
                   marker="D" if under else "o", clip_on=False, zorder=6)
        ax.text(x + 0.018 * (1 if ha == "left" else -1), y0 - i * dy,
                STATE_LABEL[s].replace("\n", " ")
                + (", no inference" if under else ""),
                fontsize=size, va="center", ha=ha, clip_on=False,
                color=ACC_POWER if under else AXIS_COLOR[s])


def shift_strip(ax, per_state, label, xmax, annotate=False):
    """Signed shift of every stored AUC, one row per population, against that
    population's own +-1.96 SE. Makes the 'smaller than sampling noise' claim
    visible instead of asserting it in a sentence."""
    ax.set_ylim(len(AXIS_STATES) - 0.45, -0.75)
    ax.set_xlim(-xmax, xmax)
    ax.set_yticks(range(len(AXIS_STATES)))
    ax.set_yticklabels([STATE_LABEL[s].replace("\n", " ") for s in AXIS_STATES],
                       fontsize=FS_MIN)
    ax.get_yticklabels()[AXIS_STATES.index(UNDERPOWERED)].set_color(ACC_POWER)
    ax.tick_params(labelsize=FS_MIN, length=1.8)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([t / 100 for t in range(-10, 11, 5) if t / 100 <= xmax])
    ax.set_xlabel(label, fontsize=6, labelpad=1.5)
    zero_line(ax, 0.0, axis="x")
    for r, s in enumerate(AXIS_STATES):
        band = 1.96 * SE[s]
        ax.add_patch(Rectangle((-band, r - 0.34), 2 * band, 0.68,
                               facecolor=GREY_FILL, edgecolor="none", zorder=0))
        v = np.asarray(per_state[s])
        under = s == UNDERPOWERED
        ax.scatter(v, np.full(v.shape, r), s=9, c=AXIS_COLOR[s],
                   linewidths=0.25,
                   edgecolors=ACC_POWER if under else "white",
                   marker="D" if under else "o", alpha=0.9, zorder=3)
        out = v[np.abs(v) > band]
        if out.size:
            ax.scatter(out, np.full(out.shape, r), s=36,
                       facecolors="none", edgecolors=INK, linewidths=0.5,
                       zorder=5)
        ax.plot([-band, -band], [r - 0.34, r + 0.34], color=GREY_RULE,
                linewidth=0.5, zorder=1)
        ax.plot([band, band], [r - 0.34, r + 0.34], color=GREY_RULE,
                linewidth=0.5, zorder=1)
        if annotate:
            ax.text(band + xmax * 0.035, r, f"|shift| ≤ {np.abs(v).max():.3f}",
                    fontsize=FS_MIN, va="center", ha="left", color=GREY_TEXT)


# =========================================================================== #
# ED 6  every cell-level test, its direction, and the correction
# =========================================================================== #
def ed6():
    tab = pd.read_csv(RESULTS / "scdrs_axis" / "cell_level_fdr.tsv", sep="\t")
    fig = new_figure(183.0, 88.0)

    axa = ax_mm(fig, 16.0, 15.0, 82.0, 65.0)
    axa.set_xlim(-0.30, 0.30)          # symmetric and wide enough for all 140
    axa.set_ylim(-0.5, 24.0)
    axa.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    axa.tick_params(labelsize=FS_MIN)
    axa.set_xlabel("effect, AUC − 0.5      (left, depleted;  right, enriched)",
                   fontsize=6, labelpad=1.5)
    axa.set_ylabel("−log$_{10}$ FDR", fontsize=6, labelpad=2)
    axa.axhline(-np.log10(ALPHA), color=INK, linewidth=0.5,
                linestyle=(0, (3, 2)))
    axa.text(0.295, -np.log10(ALPHA) + 0.4, "FDR 0.05", fontsize=FS_MIN,
             color=INK, ha="right")
    zero_line(axa, 0.0, axis="x")
    drawn = 0
    for state in AXIS_STATES:
        s = tab[tab.cell == state]
        under = state == UNDERPOWERED
        drawn += len(s)
        axa.scatter(s.auc - 0.5, -np.log10(s.fdr.clip(lower=1e-22)),
                    s=8, c=AXIS_COLOR[state], linewidths=0.3,
                    edgecolors=ACC_POWER if under else "white",
                    marker="D" if under else "o", alpha=0.9, zorder=3)
    assert drawn == len(tab) == 140, drawn
    x = tab.auc - 0.5
    assert x.min() > -0.30 and x.max() < 0.30, (x.min(), x.max())
    state_key(axa, AXIS_STATES, -0.272, 22.9, 1.65)
    top = tab.sort_values("fdr").head(4)
    for i, (_, r) in enumerate(top.iterrows()):
        axa.annotate(f"{nice(r.trait)}\n{STATE_LABEL[r.cell].replace(chr(10), ' ')}",
                     xy=(r.auc - 0.5, -np.log10(max(r.fdr, 1e-22))),
                     xytext=(r.auc - 0.5 - 0.055, 23.0 - 2.6 * i),
                     fontsize=FS_MIN, ha="right", va="center", color=GREY_TEXT,
                     linespacing=1.4,
                     arrowprops=dict(arrowstyle="-", linewidth=0.4,
                                     color=GREY_RULE, shrinkA=0, shrinkB=1.5))
    letter_mm(fig, 1.5, 86.0, "A")

    axb = ax_mm(fig, 122.0, 51.0, 56.0, 27.0)
    p = np.sort(tab.p.to_numpy())
    m = len(p)
    k = np.arange(1, m + 1)
    axb.plot(k, p, color=INK, linewidth=0.8, zorder=3)
    axb.plot(k, ALPHA * k / m, color=INK, linewidth=0.6,
             linestyle=(0, (3, 2)), zorder=2)
    n_sig = int((tab.fdr < ALPHA).sum())
    axb.axvline(n_sig, color=GREY_RULE, linewidth=0.5)
    axb.set_yscale("log")
    axb.set_xlim(0, m)
    axb.set_ylim(1e-24, 1.5)
    axb.set_xticks([0, 35, 70, 105, 140])
    axb.tick_params(labelsize=FS_MIN)
    axb.set_xlabel("test rank", fontsize=6, labelpad=1.5)
    axb.set_ylabel("P", fontsize=6, labelpad=2)
    axb.text(n_sig + 4, 1e-21, f"{n_sig} of {m}\nsurvive", fontsize=FS_MIN,
             color=INK, va="bottom", linespacing=1.4)
    # the empty quadrant is below the threshold and right of the crossing
    axb.text(136, 2e-6, "Benjamini–Hochberg\nthreshold", fontsize=FS_MIN,
             color=INK, ha="right", va="top", linespacing=1.4)
    letter_mm(fig, 106.0, 86.0, "B")

    # direction is the point: the distal populations have no enrichments at all
    axc = ax_mm(fig, 142.0, 15.0, 36.0, 22.0)
    rows_c = []
    for i, s in enumerate(AXIS_STATES):
        sig = tab[(tab.cell == s) & (tab.fdr < ALPHA)]
        up = int((sig.auc > 0.5).sum())
        dn = int((sig.auc < 0.5).sum())
        under = s == UNDERPOWERED
        axc.barh([i], [up], height=0.58, color=AXIS_COLOR[s], linewidth=0,
                 zorder=3)
        axc.barh([i], [dn], left=up, height=0.58, color=AXIS_COLOR[s],
                 alpha=0.30, edgecolor=INK, linewidth=0.4, zorder=3)
        axc.text(up + dn + 1.2, i, f"{up} ↑   {dn} ↓", fontsize=FS_MIN,
                 va="center", color=ACC_POWER if under else INK)
        rows_c.append(dict(cell_state=STATE_LABEL[s].replace("\n", " "),
                           tests=int((tab.cell == s).sum()),
                           significant=up + dn, enriched=up, depleted=dn,
                           interpreted=not under))
    axc.set_yticks(range(4))
    axc.set_yticklabels([STATE_LABEL[s].replace("\n", " ") for s in AXIS_STATES],
                        fontsize=FS_MIN)
    axc.get_yticklabels()[AXIS_STATES.index(UNDERPOWERED)].set_color(ACC_POWER)
    axc.invert_yaxis()
    axc.set_xlim(0, 34)
    allow_offaxis(axc, "x")
    axc.set_xticks([0, 10, 20, 30])
    axc.tick_params(labelsize=FS_MIN, length=1.8)
    axc.spines["left"].set_visible(False)
    axc.set_xlabel("tests at FDR < 0.05, of 35", fontsize=6, labelpad=1.5)
    # the bracket clears the tick labels, which run back to about x = -12
    axc.plot([-13.4, -14.7, -14.7, -13.4], [-0.30, -0.30, 1.30, 1.30],
             color=GREY_RULE, linewidth=0.5, clip_on=False)
    axc.text(-15.8, 0.5, "not\nresolvable", fontsize=FS_MIN, ha="right",
             va="center", color=GREY_TEXT, linespacing=1.4, clip_on=False)
    letter_mm(fig, 106.0, 43.0, "C")

    save_figure(fig, "FigureS6", source_data={
        "a_all_140_tests": tab.assign(trait=[nice(t) for t in tab.trait]),
        "c_direction": rows_c})


# =========================================================================== #
# ED 4  comparator sensitivity, against sampling noise
# =========================================================================== #
def ed4():
    strat = load("scdrs_axis/axis_interaction.json")["stratified_auc"]
    cm = load("audit_cognitive.json")["cognitive_cm_referenced"]
    fig = new_figure(183.0, 80.0)

    ax = ax_mm(fig, 17.0, 15.0, 58.0, 58.0)
    # the lowest point is 0.4027 in both axes: at a 0.40 limit a quarter of
    # that marker was cut away by its own corner
    ax.set_xlim(0.385, 0.80)
    ax.set_ylim(0.385, 0.80)
    ax.set_aspect("equal")
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.tick_params(labelsize=FS_MIN)
    ax.set_xlabel("AUC, myocyte-lineage comparator (main text)", fontsize=6,
                  labelpad=1.5)
    ax.set_ylabel("AUC, cardiomyocyte-only comparator", fontsize=6, labelpad=2)
    ax.plot([0.385, 0.80], [0.385, 0.80], color=GREY_RULE, linewidth=0.5,
            linestyle=(0, (3, 2)), zorder=1)
    zero_line(ax, 0.5)
    zero_line(ax, 0.5, axis="x")

    rows, by_state = [], {s: [] for s in AXIS_STATES}
    for trait, per in cm.items():
        for state, v in per.items():
            a, b = strat[trait][state]["auc"], v["auc"]
            under = state == UNDERPOWERED
            ax.scatter([a], [b], s=13, c=AXIS_COLOR[state], linewidths=0.4,
                       edgecolors=ACC_POWER if under else "white",
                       marker="D" if under else "o", zorder=4)
            by_state[state].append(b - a)
            rows.append(dict(trait=PRETTY.get(trait, trait),
                             cell_state=STATE_LABEL[state].replace("\n", " "),
                             AUC_myocyte_lineage=round(a, 4),
                             AUC_cardiomyocyte_only=round(b, 4),
                             shift=round(b - a, 4)))
    state_key(ax, AXIS_STATES, 0.424, 0.778, 0.023)
    letter_mm(fig, 1.5, 78.0, "A")

    axb = ax_mm(fig, 98.0, 15.0, 70.0, 58.0)
    shift_strip(axb, by_state, "shift in AUC when the comparator is changed",
                0.115, annotate=True)
    outside = [(s, v) for s, vs in by_state.items() for v in vs
               if abs(v) > 1.96 * SE[s]]
    assert outside == [], outside
    letter_mm(fig, 82.0, 78.0, "B")
    save_figure(fig, "FigureS4", source_data={"comparator_shift": rows})


# =========================================================================== #
# ED 7  every Mendelian randomization test, and the criterion they failed
# =========================================================================== #
def ed7():
    mr = load("mr_bidirectional.json")
    COG = ("EducationalAttainment", "Intelligence", "ReactionTime")
    VAG = ("HRV_RMSSD", "HRV_SDNN", "RestingHeartRate")
    NEGOUT = ("AtrialFibrillation", "QTinterval")
    fwd_v = [k for k in mr if k.split("->")[0] in COG and k.split("->")[1] in VAG]
    fwd_m = [k for k in mr if k.split("->")[1] in NEGOUT]
    rev = [k for k in mr if k.split("->")[1] in COG]
    groups = [("cognitive → vagal and heart rate", fwd_v, INK),
              ("cognitive → the pre-declared negative-control outcomes",
               fwd_m, ACC_NEG),
              ("cardiac → cognitive", rev, INK)]

    fig = new_figure(172.0, 112.0)
    order, group_of = [], {}
    for title, keys, _ in groups:
        for k in keys:
            order.append(k)
            group_of[k] = title
        order.append(None)
    order = order[:-1]

    axf = ax_mm(fig, 54.0, 15.0, 60.0, 88.0)
    axf.set_ylim(len(order) + 0.6, -1.6)
    axf.set_xlim(-0.22, 0.26)
    axf.set_yticks([])
    axf.spines["left"].set_visible(False)
    axf.tick_params(labelsize=FS_MIN)
    axf.set_xlabel("IVW estimate (95% CI)", fontsize=6, labelpad=1.5)
    zero_line(axf, 0.0, axis="x")

    axe = ax_mm(fig, 128.0, 15.0, 40.0, 88.0)
    axe.set_ylim(len(order) + 0.6, -1.6)
    axe.set_xlim(-0.0055, 0.0138)      # the stored intercepts span -0.004..0.007
    axe.set_yticks([])
    axe.spines["left"].set_visible(False)
    axe.set_xticks([-0.004, 0.000, 0.004, 0.008])
    axe.tick_params(labelsize=FS_MIN)
    axe.set_xlabel("MR-Egger intercept", fontsize=6, labelpad=1.5)
    zero_line(axe, 0.0, axis="x")

    rows, triggered = [], []
    for r, k in enumerate(order):
        if k is None:
            continue
        v = mr[k]
        a, b = k.split("->")
        weak = v["n_iv"] < 20
        trips = b in NEGOUT and v["ivw_p"] < 0.05     # the pre-declared criterion
        if trips:
            triggered.append(k)
        col = ACC_FAIL if trips else (ACC_POWER if weak else INK)
        lo, hi = v["ivw"] - 1.96 * v["ivw_se"], v["ivw"] + 1.96 * v["ivw_se"]
        axf.plot([lo, hi], [r, r], color=col, linewidth=0.6)
        axf.scatter([v["ivw"]], [r], s=10, c=col, linewidths=0, zorder=4)
        axf.text(-0.235, r, f"{PRETTY.get(a, a)} → {PRETTY.get(b, b)}",
                 fontsize=FS_MIN, ha="right", va="center", color=col,
                 clip_on=False)
        axf.text(0.255, r, f"{v['n_iv']}", fontsize=FS_MIN, ha="right",
                 va="center", color=GREY_TEXT)
        alarm = v["egger_intercept_p"] < 0.05
        ecol = ACC_FAIL if alarm else GREY_TEXT
        axe.scatter([v["egger_intercept"]], [r], s=10, c=ecol, linewidths=0,
                    zorder=4)
        if alarm:
            axe.text(v["egger_intercept"] + 0.0006, r,
                     f"P = {v['egger_intercept_p']:.4f}", fontsize=FS_MIN,
                     va="center", color=ACC_FAIL)
        rows.append(dict(test=f"{PRETTY.get(a, a)} -> {PRETTY.get(b, b)}",
                         group=group_of[k], n_instruments=v["n_iv"],
                         IVW=round(v["ivw"], 4), IVW_SE=round(v["ivw_se"], 4),
                         IVW_P=v["ivw_p"], MR_Egger=round(v["egger"], 4),
                         weighted_median=round(v["wmedian"], 4),
                         Egger_intercept=round(v["egger_intercept"], 5),
                         Egger_intercept_P=v["egger_intercept_p"],
                         trips_negative_control=bool(
                             b in NEGOUT and v["ivw_p"] < 0.05)))
    assert len(triggered) == 3, triggered
    y = 0
    for title, keys, tcol in groups:
        axf.text(-0.235, y - 0.62, title, fontsize=FS_MIN, ha="right",
                 va="center", color=tcol, fontweight="bold", clip_on=False)
        y += len(keys) + 1
    axf.text(0.255, -0.62, "instruments", fontsize=FS_MIN, ha="right",
             va="center", color=GREY_TEXT)

    # the instruments column occupies the right edge of every row, so the two
    # keys go to the empty strip below the last label rather than beside it
    text_mm(fig, 6.0, 10.0, "red, trips the pre-declared negative control",
            fontsize=FS_MIN, ha="left", va="center", color=ACC_FAIL)
    text_mm(fig, 6.0, 5.5, "amber, fewer than 20 instruments", fontsize=FS_MIN,
            ha="left", va="center", color=ACC_POWER)
    letter_mm(fig, 1.5, 110.0, "A")
    letter_mm(fig, 120.0, 110.0, "B")
    save_figure(fig, "FigureS7", source_data={"mr_tests": rows})


# =========================================================================== #
# ED 3  the spatial route
# =========================================================================== #
def ed3():
    dil = load("dilution.json")
    lib = load("library_size_confound.json")["per_trait"]
    ver = load("depthmatched_verdict.json")
    scen = load("corrections.json")["spatial_power"]
    fig = new_figure(183.0, 98.0)
    COMP = {"vs myocytes (wrong)": "myocyte comparator (superseded)",
            "vs all cells (right)": "all-cell comparator (used in the text)"}

    axa = ax_mm(fig, 34.0, 60.0, 58.0, 28.0)
    keys = list(dil["purity"])
    axa.set_ylim(len(keys) - 0.4, -0.9)
    axa.set_xlim(0, 96)
    axa.set_yticks([])
    axa.spines["left"].set_visible(False)
    axa.tick_params(labelsize=FS_MIN)
    axa.set_xlabel("pacemaker cells as a share of the spot (%)", fontsize=6,
                   labelpad=1.5)
    rows_a = []
    for r, k in enumerate(keys):
        d = dil["purity"][k]
        axa.plot([d["median"], d["max"]], [r, r], color=GREY_RULE, linewidth=0.6)
        axa.plot([d["max"], d["max"]], [r - 0.16, r + 0.16], color=GREY_RULE,
                 linewidth=0.7)
        axa.scatter([d["p90"], d["p99"]], [r, r], s=8, facecolors="white",
                    edgecolors=SLATE, linewidths=0.5, zorder=4)
        axa.scatter([d["median"]], [r], s=13, c=SLATE, linewidths=0, zorder=5)
        axa.text(-1.5, r, k.replace("`", ""), fontsize=FS_MIN, ha="right",
                 va="center", color=INK, clip_on=False)
        axa.text(d["max"] + 1.5, r, f"n = {d['n']:,}", fontsize=FS_MIN,
                 va="center", color=GREY_TEXT)
        rows_a.append(dict(panel="a purity", scenario=k, n=d["n"],
                           median=round(d["median"], 3), p90=round(d["p90"], 3),
                           p99=round(d["p99"], 3), max=round(d["max"], 3)))
    axa.text(95, -0.85, "filled, median;  open, 90th and 99th;  bar, maximum",
             fontsize=FS_MIN, ha="right", va="center", color=GREY_TEXT)
    letter_mm(fig, 1.5, 96.0, "A")

    axb = ax_mm(fig, 124.0, 60.0, 54.0, 28.0)
    order = sorted(lib, key=lambda t: -lib[t]["total_umi"])
    axb.set_ylim(len(order) - 0.4, -0.9)
    axb.set_xlim(0, 1.0)
    axb.set_yticks([])
    axb.spines["left"].set_visible(False)
    axb.tick_params(labelsize=FS_MIN)
    axb.set_xlabel("Spearman ρ between the spot score and total UMI",
                   fontsize=6, labelpad=1.5)
    rows_b = []
    for r, t in enumerate(order):
        v = lib[t]
        col = fam_ink(t)
        axb.plot([0, max(v["total_umi"], v["n_genes"])], [r, r], color=col,
                 linewidth=0.7)
        axb.scatter([v["n_genes"]], [r], s=16, facecolors="white",
                    edgecolors=col, linewidths=0.5, zorder=4)
        axb.scatter([v["total_umi"]], [r], s=5, c=col, linewidths=0,
                    zorder=5)
        axb.text(-0.015, r, PRETTY.get(t, t), fontsize=FS_MIN, ha="right",
                 va="center", color=col, clip_on=False)
        rows_b.append(dict(panel="b depth", trait=PRETTY.get(t, t),
                           rho_total_umi=round(v["total_umi"], 3),
                           rho_n_genes=round(v["n_genes"], 3),
                           rho_mito_frac=round(v["mito_frac"], 3)))
    axb.text(0.99, -0.85, "filled, total UMI;  open, genes detected",
             fontsize=FS_MIN, ha="right", va="center", color=GREY_TEXT)
    letter_mm(fig, 98.0, 96.0, "B")

    # the load-bearing depth result is the correlation, not the score level
    axc = ax_mm(fig, 34.0, 15.0, 58.0, 26.0)
    q1, q2 = ver["Q1"], ver["Q2"]
    tr = list(q1)
    axc.set_ylim(len(tr) - 0.4, -0.9)
    axc.set_xlim(-0.02, 0.92)
    axc.set_yticks([])
    axc.spines["left"].set_visible(False)
    axc.tick_params(labelsize=FS_MIN)
    axc.set_xlabel("Spearman ρ with depth, before and after depth matching",
                   fontsize=6, labelpad=1.5)
    rows_c = []
    for r, t in enumerate(tr):
        o, n = q2[t]["old"], q2[t]["new"]
        col = fam_ink(t)
        axc.annotate("", xy=(n, r), xytext=(o, r),
                     arrowprops=dict(arrowstyle="-|>", color=col,
                                     linewidth=0.7, shrinkA=2.0, shrinkB=0,
                                     mutation_scale=4))
        axc.scatter([o], [r], s=9, facecolors="white", edgecolors=col,
                    linewidths=0.55, zorder=4)
        axc.text(-0.03, r, PRETTY.get(t, t), fontsize=FS_MIN, ha="right",
                 va="center", color=col, clip_on=False)
        rows_c.append(dict(panel="c depth matching", trait=PRETTY.get(t, t),
                           rho_before=round(o, 4), rho_after=round(n, 4),
                           median_score_before=round(q1[t]["old_median"], 3),
                           median_score_after=round(q1[t]["median"], 3)))
    axc.text(0.91, -0.85, "open, before;  arrowhead, after matching",
             fontsize=FS_MIN,
             ha="right", va="center", color=GREY_TEXT)
    letter_mm(fig, 1.5, 50.0, "C")

    axd = ax_mm(fig, 124.0, 15.0, 54.0, 26.0)
    axd.set_xlim(0, 32)
    axd.set_ylim(0, 1.28)
    axd.set_xticks([0, 10, 20, 30])
    axd.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axd.tick_params(labelsize=FS_MIN)
    axd.set_xlabel("spot purity (%)", fontsize=6, labelpad=1.5)
    axd.set_ylabel("power per section", fontsize=6, labelpad=2)
    axd.axhline(0.8, color=GREY_RULE, linewidth=0.5, linestyle=(0, (3, 2)))
    axd.text(0.8, 0.82, "80%", fontsize=FS_MIN, ha="left", color=GREY_TEXT)
    PURITY = {"all node spots": dil["purity"]["annotated `node` only"]["median"],
              "top 1% abundance": dil["purity"]["top 1% by pacemaker abundance"]["median"]}
    rows_d = []
    for k, v in scen.items():
        comp, spots = k.split("|")
        used = comp.startswith("vs all cells")
        axd.scatter([PURITY[spots]], [v["power"]], s=16,
                    c=SLATE if used else "white", edgecolors=SLATE,
                    linewidths=0.6, marker="o" if used else "s", zorder=4)
        rows_d.append(dict(panel="d power", comparator=COMP[comp], spots=spots,
                           purity_percent=round(PURITY[spots], 2),
                           d_cell=round(v["d_cell"], 4),
                           d_spot=round(v["d_spot"], 4),
                           power=round(v["power"], 4),
                           P_six_of_eight=v["p_6of8"],
                           used_in_the_text=used))
    # the keys sit above 1.0, which power cannot reach, so neither can be
    # read as a scenario
    axd.scatter([1.5], [1.22], s=16, c=SLATE, linewidths=0, zorder=4)
    axd.text(3.2, 1.22, "all-cell comparator", fontsize=FS_MIN, va="center",
             color=INK)
    axd.scatter([1.5], [1.10], s=16, facecolors="white", edgecolors=SLATE,
                linewidths=0.6, marker="s", zorder=4)
    axd.text(3.2, 1.10, "myocyte comparator", fontsize=FS_MIN, va="center",
             color=GREY_TEXT)
    letter_mm(fig, 98.0, 50.0, "D")

    save_figure(fig, "FigureS3", source_data={
        "a_purity": rows_a, "b_depth": rows_b, "c_depth_matching": rows_c,
        "d_power": rows_d})


# =========================================================================== #
# ED 5  how the mouse clusters were annotated
# =========================================================================== #
def ed5():
    mp = load("mouse_prepare.json")
    labels, classes = mp["cluster_labels"], mp["classes"]
    fig = new_figure(183.0, 55.0)

    order = ["nodal", "non_myocyte_nodal", "non_myocyte_purkinje", "working_CM",
             "fibroblast", "endothelium", "immune", "erythroid"]
    CLS = {k: (v, MOUSE_CLASS_HATCH.get(k)) for k, v in MOUSE_CLASS.items()}
    NICE = {"nodal": "nodal, scored",
            "non_myocyte_nodal": "nodal panel, non-myocyte, excluded",
            "non_myocyte_purkinje": "Purkinje panel, non-myocyte, excluded",
            "working_CM": "working cardiomyocyte, the comparator",
            "fibroblast": "fibroblast", "endothelium": "endothelium",
            "immune": "immune", "erythroid": "erythroid"}
    ZONE = {"atrioventricular": "atrioventricular",
            "purkinje_left": "Purkinje, left",
            "purkinje_right": "Purkinje, right",
            "sinoatrial": "sinoatrial"}

    axa = ax_mm(fig, 30.0, 40.0, 102.0, 7.5)
    ids = sorted(labels, key=lambda c: (order.index(labels[c]), int(c)))
    axa.set_xlim(-0.6, len(ids) - 0.4)
    axa.set_ylim(-0.5, 0.5)
    axa.set_yticks([])
    for sp in axa.spines.values():
        sp.set_visible(False)
    axa.set_xticks(range(len(ids)))
    axa.set_xticklabels(ids, fontsize=FS_MIN)
    axa.tick_params(labelsize=FS_MIN, length=0, pad=1.5)
    axa.set_xlabel(f"Leiden cluster, resolution {mp['leiden_res']}", fontsize=6,
                   labelpad=1.5)
    rows_a = []
    for i, c in enumerate(ids):
        fc, hatch = CLS[labels[c]]
        axa.add_patch(Rectangle((i - 0.42, -0.32), 0.84, 0.64, facecolor=fc,
                                edgecolor="white", linewidth=0.4, hatch=hatch))
        rows_a.append(dict(cluster=int(c), assigned_class=NICE[labels[c]]))
    axa.text(-1.4, 0, f"{mp['n_cells']:,} cells\n{mp['n_genes']:,} genes",
             fontsize=FS_MIN, va="center", ha="right", color=GREY_TEXT,
             linespacing=1.4, clip_on=False)
    letter_mm(fig, 1.5, 53.0, "A")

    axb = ax_mm(fig, 30.0, 12.0, 102.0, 16.0)
    zones = list(classes)
    axb.set_ylim(len(zones) - 0.4, -0.6)
    axb.set_xlim(0, 100)
    axb.set_yticks([])
    axb.spines["left"].set_visible(False)
    axb.tick_params(labelsize=FS_MIN)
    axb.set_xlabel("cells in the zone (%)", fontsize=6, labelpad=1.5)
    rows_b = []
    for r, z in enumerate(zones):
        tot = sum(classes[z].values())
        left = 0.0
        for cls in order:
            v = classes[z].get(cls, 0)
            if not v:
                continue
            frac = 100.0 * v / tot
            fc, hatch = CLS[cls]
            axb.add_patch(Rectangle((left, r - 0.3), frac, 0.6, facecolor=fc,
                                    edgecolor="white", linewidth=0.4,
                                    hatch=hatch))
            if frac > 11:
                lum = int(fc[1:3], 16) * 0.299 + int(fc[3:5], 16) * 0.587 \
                    + int(fc[5:7], 16) * 0.114
                axb.text(left + frac / 2, r, f"{frac:.0f}", fontsize=FS_MIN,
                         ha="center", va="center",
                         color="white" if lum < 130 else INK)
            left += frac
            rows_b.append(dict(zone=ZONE[z], cell_class=NICE[cls], n=v,
                               percent=round(frac, 2)))
        axb.text(-1.5, r, f"{ZONE[z]}, {tot:,}", fontsize=FS_MIN,
                 ha="right", va="center", color=INK, clip_on=False)
    letter_mm(fig, 1.5, 34.0, "B")

    axl = ax_mm(fig, 137.5, 12.0, 45.5, 36.0)
    axl.axis("off")
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    for i, cls in enumerate(order):
        fc, hatch = CLS[cls]
        axl.add_patch(Rectangle((0.0246, 0.905 - i * 0.115), 0.0923, 0.072,
                                facecolor=fc, edgecolor="white",
                                linewidth=0.4, hatch=hatch))
        axl.text(0.1538, 0.941 - i * 0.115, NICE[cls], fontsize=FS_MIN,
                 va="center", color=INK)
    save_figure(fig, "FigureS5", source_data={
        "a_cluster_assignment": rows_a, "b_zone_composition": rows_b})


# =========================================================================== #
# ED 1  gene-length adjustment, over the sets that were rebuilt
# =========================================================================== #
def ed1():
    strat = load("scdrs_axis/axis_interaction.json")["stratified_auc"]
    pairs = sorted(k for k in strat if f"{k}_lenadj" in strat)
    missing = sorted(k for k in strat
                     if not k.endswith("_lenadj") and f"{k}_lenadj" not in strat)
    assert len(pairs) == 16 and len(missing) == 3, (len(pairs), missing)
    fig = new_figure(183.0, 80.0)

    ax = ax_mm(fig, 17.0, 15.0, 58.0, 58.0)
    LO, HI = 0.22, 0.72
    ax.set_xlim(LO, HI)
    ax.set_ylim(LO, HI)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=FS_MIN)
    ax.set_xlabel("AUC, original gene set", fontsize=6, labelpad=1.5)
    ax.set_ylabel("AUC, gene length regressed out", fontsize=6, labelpad=2)
    ax.plot([LO, HI], [LO, HI], color=GREY_RULE, linewidth=0.5,
            linestyle=(0, (3, 2)), zorder=1)
    zero_line(ax, 0.5)
    zero_line(ax, 0.5, axis="x")
    rows, by_state, n_drawn = [], {s: [] for s in AXIS_STATES}, 0
    for key in pairs:
        for state in AXIS_STATES:
            a = strat[key][state]["auc"]
            b = strat[f"{key}_lenadj"][state]["auc"]
            under = state == UNDERPOWERED
            assert LO < a < HI and LO < b < HI, (key, state, a, b)
            ax.scatter([a], [b], s=10, c=AXIS_COLOR[state], linewidths=0.25,
                       edgecolors=ACC_POWER if under else "white",
                       marker="D" if under else "o", alpha=0.9, zorder=3)
            by_state[state].append(b - a)
            n_drawn += 1
            rows.append(dict(trait=nice(key),
                             cell_state=STATE_LABEL[state].replace("\n", " "),
                             AUC_original=round(a, 4),
                             AUC_length_adjusted=round(b, 4),
                             shift=round(b - a, 4)))
    assert n_drawn == 64, n_drawn
    edu = strat["EducationalAttainment"]["SAN_P_cell"]["auc"]
    edu_adj = strat["EducationalAttainment_lenadj"]["SAN_P_cell"]["auc"]
    # label placed below-right of its point, inside the empty lower-right corner
    ax.annotate("educational attainment,\nsinoatrial pacemaker",
                xy=(edu, edu_adj), xytext=(0.712, 0.462), fontsize=FS_MIN,
                ha="right", va="center", color=INK, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", linewidth=0.4, color=GREY_RULE,
                                shrinkA=0, shrinkB=2))
    state_key(ax, AXIS_STATES, 0.238, 0.700, 0.024)
    letter_mm(fig, 1.5, 78.0, "A")

    axb = ax_mm(fig, 98.0, 15.0, 70.0, 58.0)
    shift_strip(axb, by_state, "shift in AUC when gene length is regressed out",
                0.105, annotate=True)
    # one of the 64 shifts leaves its own band: resting heart rate in
    # sinoatrial pacemaker cells, -0.045 against +-0.041, which takes that
    # AUC from 0.583 to 0.538 and its z from +4.00 to +1.83. The earlier
    # form of this check compared every population against the widest of the
    # four bands and so could not fail.
    outside = [(s, v) for s, vs in by_state.items() for v in vs
               if abs(v) > 1.96 * SE[s]]
    assert len(outside) == 1 and outside[0][0] == "SAN_P_cell", outside
    assert abs(outside[0][1] + 0.0452) < 5e-4, outside
    letter_mm(fig, 82.0, 78.0, "B")
    save_figure(fig, "FigureS1", source_data={"length_adjusted": rows})


# =========================================================================== #
# ED 2  locus structure of the gene sets
# =========================================================================== #
def ed2():
    gs = load("scdrs/geneset_spread.json")
    keys = sorted(gs, key=lambda k: -gs[k]["n_loci"])
    fig = new_figure(144.0, 74.0)
    ROW = 3.6
    H = ROW * len(keys)

    axa = ax_mm(fig, 34.0, 14.0, 49.0, H)
    axb = ax_mm(fig, 89.0, 14.0, 49.0, H)
    for ax in (axa, axb):
        ax.set_ylim(len(keys) - 0.4, -1.0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=FS_MIN)
    axa.set_xlim(0, 470)
    axb.set_xlim(-0.012, 0.40)
    axa.set_xticks([0, 100, 200, 300, 400])
    # pinned: the wider panel would otherwise pick up a 0.05 locator
    axb.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    axa.set_xlabel("independent loci after 1 Mb merging", fontsize=6,
                   labelpad=1.5)
    axb.set_xlabel("share of the gene set's weight", fontsize=6, labelpad=1.5)

    rows = []
    for r, k in enumerate(keys):
        d = gs[k]
        ctrl = k.startswith("Rheumatoid")
        cog = k in ("EducationalAttainment", "Intelligence", "ReactionTime")
        col = ACC_NEG if ctrl else SLATE
        axa.barh([r], [d["n_loci"]], height=0.62, color=col, linewidth=0)
        axa.text(d["n_loci"] + 8, r, str(d["n_loci"]), fontsize=FS_MIN,
                 va="center", color=col)
        axa.text(-10, r, nice(k), fontsize=FS_MIN, ha="right", va="center",
                 color=col, clip_on=False,
                 fontweight="bold" if cog else "normal")
        axb.scatter([d["largest_locus"]], [r], s=9, c=SLATE, linewidths=0,
                    zorder=4)
        axb.scatter([d["top_chr_share"]], [r], s=9, facecolors="white",
                    edgecolors=SLATE, linewidths=0.5, zorder=4)
        zero_mhc = d["mhc_share"] <= 0.0
        axb.scatter([d["mhc_share"]], [r], s=11,
                    c="white" if zero_mhc else ACC_NEG, marker="D",
                    edgecolors=ACC_NEG, linewidths=0.5, zorder=5)
        rows.append(dict(trait=nice(k), n_genes=d["n_genes"],
                         n_chromosomes=d["n_chr"], n_loci=d["n_loci"],
                         largest_locus_share=round(d["largest_locus"], 4),
                         top_chromosome_share=round(d["top_chr_share"], 4),
                         MHC_share=round(d["mhc_share"], 4)))
    axb.axvline(0.0, color=GREY_RULE, linewidth=0.5)

    # the top three rows carry nothing beyond x = 0.22, so the key goes there
    for j, ((c, mk, filled), lab) in enumerate([
            ((SLATE, "o", True), "largest locus"),
            ((SLATE, "o", False), "largest chromosome"),
            ((ACC_NEG, "D", True), "MHC")]):
        axb.scatter([0.242], [0.15 + j * 1.05], s=10, marker=mk,
                    c=c if filled else "white", edgecolors=c, linewidths=0.5)
        axb.text(0.255, 0.15 + j * 1.05, lab, fontsize=FS_MIN, va="center",
                 color=INK)
    i_ctrl = keys.index("RheumatoidArthritis")
    # the two rows above the control carry nothing beyond x = 0.15
    mhc = gs["RheumatoidArthritis"]["mhc_share"]
    axb.annotate(f"{mhc * 100:.0f}% of the weight\nsits in the MHC",
                 xy=(mhc, i_ctrl), xytext=(0.335, i_ctrl - 1.7),
                 fontsize=FS_MIN, ha="right", va="center", color=ACC_NEG,
                 linespacing=1.5,
                 arrowprops=dict(arrowstyle="-", linewidth=0.4, color=ACC_NEG,
                                 shrinkA=0, shrinkB=2))
    letter_mm(fig, 1.5, 72.0, "A")
    letter_mm(fig, 85.5, 72.0, "B")
    save_figure(fig, "FigureS2", source_data={"locus_structure": rows})


# ed7 is not run: 311_figS7_mr_v2.py supersedes it and writes FigureS7 from
# the second Mendelian randomization analysis. Rendering the supplement used
# to overwrite that figure with the first one.
for f in (ed1, ed2, ed4, ed6, ed5, ed3):
    f()
