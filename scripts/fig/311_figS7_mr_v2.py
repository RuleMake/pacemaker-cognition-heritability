"""Figure S7, rebuilt for the second Mendelian randomization analysis.

Panel A  every primary-specification test, grouped by what it asks, with the
         re-specified negative-control family marked.
Panel B  the rheumatoid-arthritis control that separates the two instrument
         sets, which is what makes the two directions non-comparable.

Reads results/mr_v2_estimates.tsv and results/mr_v2_verdict.json only.
Supersedes the ed7() block of 310_extended.py, which read the first analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from _style import (ACC_FAIL, ACC_NEG, ACC_POWER, FS_MIN, GREY_RULE, GREY_TEXT,
                    INK, RESULTS, SUPPORT, TRAIT_LABEL, ax_mm, letter_mm,
                    load, new_figure, save_figure, text_mm, zero_line)

SLATE = SUPPORT[0]
PRETTY = dict(TRAIT_LABEL)
PRETTY.update({"RheumatoidArthritis": "Rheumatoid arthritis",
               "HRV_RMSSD": "HRV, RMSSD", "HRV_SDNN": "HRV, SDNN"})

COG = ("EducationalAttainment", "Intelligence", "ReactionTime")
AUT = ("HRV_RMSSD", "HRV_SDNN", "RestingHeartRate")
NEG = ("AtrialFibrillation", "QTinterval")

import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.default": "regular"})
BONF_N1 = 0.05 / 6          # negative-control family threshold
BONF_ALL = 0.05 / 24        # threshold across the hypothesis tests



def sci(p: float) -> str:
    """P as 'm.m × 10^e' with a mathtext exponent, set in the figure's Arial."""
    m, e = f"{p:.1e}".split("e")
    return f"{m} × 10$^{{{int(e)}}}$"

def main():
    df = pd.read_csv(RESULTS / "mr_v2_estimates.tsv", sep="\t")
    prim = df[df["spec"] == "primary"].copy()
    sens = df[df["spec"] != "primary"].set_index(["exposure", "outcome"])
    prim["key"] = list(zip(prim["exposure"], prim["outcome"]))
    P = {k: r for k, r in zip(prim["key"], (row for _, row in prim.iterrows()))}

    groups = [
        ("cognitive → autonomic",
         [(a, b) for a in COG for b in AUT], INK),
        ("cognitive → the re-specified negative-control outcomes",
         [(a, b) for a in COG for b in NEG], ACC_NEG),
        ("autonomic → cognitive",
         [(a, b) for a in AUT for b in COG], SLATE),
    ]

    order, group_of = [], {}
    for title, keys, _ in groups:
        for k in keys:
            if k in P:
                order.append(k)
                group_of[k] = title
        order.append(None)
    order = order[:-1]

    fig = new_figure(172.0, 140.0)

    axf = ax_mm(fig, 56.0, 42.0, 60.0, 90.0)
    axf.set_ylim(len(order) + 0.6, -1.8)
    axf.set_xlim(-0.30, 0.50)
    axf.set_yticks([])
    axf.spines["left"].set_visible(False)
    axf.tick_params(labelsize=FS_MIN)
    axf.set_xlabel("IVW estimate (95% CI), SD per SD", fontsize=6, labelpad=1.5)
    zero_line(axf, 0.0, axis="x")

    axz = ax_mm(fig, 126.0, 42.0, 34.0, 90.0)
    axz.set_ylim(len(order) + 0.6, -1.8)
    axz.set_xscale("log")
    axz.set_xlim(0.8, 60.0)
    axz.set_yticks([])
    axz.set_xticks([1, 10])
    axz.set_xticklabels(["1", "10"])
    axz.spines["left"].set_visible(False)
    axz.tick_params(labelsize=FS_MIN)
    axz.set_xlabel("implied Steiger bound, |Z|", fontsize=6, labelpad=1.5)
    axz.axvline(1.96, color=GREY_RULE, linewidth=0.5, linestyle=(0, (2, 2)),
                zorder=1)

    rows = []
    for r, k in enumerate(order):
        if k is None:
            continue
        v = P[k]
        a, b = k
        neg_family = b in NEG and a in COG
        trips = neg_family and v["ivw_p"] < BONF_N1
        weak = v["n_iv"] < 10
        col = ACC_FAIL if trips else (ACC_POWER if weak else
                                      (ACC_NEG if neg_family else INK))
        lo = v["ivw"] - 1.96 * v["ivw_se"]
        hi = v["ivw"] + 1.96 * v["ivw_se"]
        axf.plot([lo, hi], [r, r], color=col, linewidth=0.6)
        axf.scatter([v["ivw"]], [r], s=10, c=col, linewidths=0, zorder=4)
        axf.text(-0.315, r, f"{PRETTY.get(a, a)} → {PRETTY.get(b, b)}",
                 fontsize=FS_MIN, ha="right", va="center", color=col,
                 clip_on=False)
        axf.text(0.495, r, f"{int(v['n_iv'])}", fontsize=FS_MIN, ha="right",
                 va="center", color=GREY_TEXT)
        if trips:
            axf.text(hi + 0.012, r, f"P = {v['ivw_p']:.4f}", fontsize=FS_MIN,
                     va="center", color=ACC_FAIL)

        zb = float(v["steiger_z_bound"])
        zcol = ACC_FAIL if zb < 1.96 else GREY_TEXT
        axz.scatter([zb], [r], s=10, c=zcol, linewidths=0, zorder=4)

        s = sens.loc[k] if k in sens.index else None
        rows.append(dict(test=f"{PRETTY.get(a, a)} -> {PRETTY.get(b, b)}",
                         group=group_of[k], n_instruments=int(v["n_iv"]),
                         F_mean=round(float(v["F_mean"]), 1),
                         I2_gx=round(float(v["I2_gx"]), 3),
                         IVW=round(float(v["ivw"]), 4),
                         IVW_SE=round(float(v["ivw_se"]), 4),
                         IVW_P=float(v["ivw_p"]),
                         weighted_median=round(float(v["wmedian"]), 4),
                         weighted_mode=round(float(v["wmode"]), 4),
                         RAPS=round(float(v["raps"]), 4),
                         Q_over_df=round(float(v["Q_over_df"]), 2),
                         MDE_80=round(float(v["mde_80"]), 3),
                         implied_Steiger_Z_bound=round(zb, 2),
                         sensitivity_IVW_P=(None if s is None
                                            else float(s["ivw_p"])),
                         trips_negative_control=bool(trips)))

    y = 0
    for title, keys, tcol in groups:
        n = sum(1 for k in keys if k in P)
        axf.text(-0.315, y - 0.75, title, fontsize=FS_MIN, ha="right",
                 va="center", color=tcol, fontweight="bold", clip_on=False)
        y += n + 1
    axf.text(0.495, -0.75, "instruments", fontsize=FS_MIN, ha="right",
             va="center", color=GREY_TEXT)

    # ---- panel B: the control that separates the two instrument sets --------
    axr = ax_mm(fig, 56.0, 13.0, 60.0, 18.0)
    ra = [(e, P[(e, "RheumatoidArthritis")])
          for e in COG + AUT if (e, "RheumatoidArthritis") in P]
    axr.set_ylim(len(ra) - 0.5, -0.5)
    # widened 2026-10-01: with the RA sample size corrected to 97,173, EA -> RA is -0.31
    axr.set_xlim(-0.45, 0.42)
    axr.set_yticks([])
    axr.spines["left"].set_visible(False)
    axr.tick_params(labelsize=FS_MIN)
    axr.set_xlabel("effect on rheumatoid arthritis (95% CI)", fontsize=6,
                   labelpad=1.5)
    zero_line(axr, 0.0, axis="x")
    for r, (e, v) in enumerate(ra):
        col = ACC_FAIL if v["ivw_p"] < BONF_ALL else INK
        lo = v["ivw"] - 1.96 * v["ivw_se"]
        hi = v["ivw"] + 1.96 * v["ivw_se"]
        axr.plot([lo, hi], [r, r], color=col, linewidth=0.6)
        axr.scatter([v["ivw"]], [r], s=10, c=col, linewidths=0, zorder=4)
        axr.text(-0.465, r, f"{PRETTY.get(e, e)} instruments", fontsize=FS_MIN,
                 ha="right", va="center", color=col, clip_on=False)
        if col == ACC_FAIL:
            axr.text(hi + 0.012, r, f"P = {sci(v['ivw_p'])}", fontsize=FS_MIN,
                     ha="left", va="center", color=ACC_FAIL, clip_on=False)
        rows.append(dict(test=f"{PRETTY.get(e, e)} -> Rheumatoid arthritis",
                         group="instrument-quality control",
                         n_instruments=int(v["n_iv"]),
                         F_mean=round(float(v["F_mean"]), 1),
                         I2_gx=round(float(v["I2_gx"]), 3),
                         IVW=round(float(v["ivw"]), 4),
                         IVW_SE=round(float(v["ivw_se"]), 4),
                         IVW_P=float(v["ivw_p"]),
                         weighted_median=round(float(v["wmedian"]), 4),
                         weighted_mode=round(float(v["wmode"]), 4),
                         RAPS=round(float(v["raps"]), 4),
                         Q_over_df=round(float(v["Q_over_df"]), 2),
                         MDE_80=round(float(v["mde_80"]), 3),
                         implied_Steiger_Z_bound=round(
                             float(v["steiger_z_bound"]), 2),
                         sensitivity_IVW_P=None,
                         trips_negative_control=False))

    text_mm(fig, 5.0, 3.5,
            "magenta, the negative-control outcome family; red, trips the negative-control "
            "family or the instrument control; amber, fewer than 10 instruments",
            fontsize=FS_MIN, ha="left", va="center", color=GREY_TEXT)
    letter_mm(fig, 1.5, 138.0, "A")
    letter_mm(fig, 1.5, 33.0, "B")
    save_figure(fig, "FigureS7", source_data={"mr_v2_tests": rows})


if __name__ == "__main__":
    main()
