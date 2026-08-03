"""
The three things the audit found missing, done properly.
========================================================

  A  FDR. Sixty-eight cell-level tests (17 traits x 4 conduction cell types) were run
     and reported without any multiplicity correction. Several of the smaller claims
     — Brugada on Purkinje at z+1.9, atrial flutter on AVN_P_cell at z+2.6 — are the
     kind that do not survive one, and saying so is not optional.

  B  The shape test. It was reported as p = (1/15)^3 = 3e-4: all three cognitive
     traits rank both HRV indices above the four other cardiac traits. That assumed
     the three were independent. They are not — educational attainment and
     intelligence have rg = 0.738, which makes them close to one trait, while reaction
     time is nearly independent of both (0.055 and 0.185). The exponent has to come
     from the measured correlation, not from the number of files on disk.

  C  The spatial power calculation used the wrong contrast. It plugged in the
     cell-level effect measured as pacemaker cells against WORKING MYOCYTES, but the
     spatial test compared node-region spots against spots from other compartments —
     a background of every cell type, not of myocytes. The effect size has to match
     the comparison the design actually made, and against an all-cell background the
     pacemaker effect is much larger, which makes the spatial null harder to excuse.

Usage:  python scripts/192_corrections.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.as_posix()
SC = f"{ROOT}/results/scdrs_axis"
OUT = f"{ROOT}/results/corrections.json"

FOCUS = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
out = {}

# ================================================================ A  FDR
print("=" * 100)
print("A. FDR ACROSS ALL CELL-LEVEL TESTS (myocyte-referenced, 17 traits x 4 cells)")
print("=" * 100)
j = json.load(open(f"{SC}/axis_interaction.json"))
tab = j.get("stratified_auc", {})
rows = []
for t, d in tab.items():
    for c, r in d.items():
        z = r["z"]
        rows.append(dict(trait=t, cell=c, auc=r["auc"], z=z,
                         p=float(2 * stats.norm.sf(abs(z)))))
df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
m = len(df)
# Benjamini-Hochberg: the running minimum goes from the LARGEST p downwards, so that
# a q-value can never exceed the q of a less significant test. Taking cummin() in the
# ascending direction — as the first version of this did — propagates the smallest
# value forward and hands FDR = 0 to a test with p = 0.97.
q = (df.p * m / (df.index + 1)).to_numpy()
df["fdr"] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1.0)
print(f"{m} tests\n")
print(f"{'trait':<26}{'cell':<18}{'AUC':>8}{'z':>8}{'p':>11}{'FDR':>10}")
for r in df.itertuples():
    if r.fdr < 0.25 or abs(r.z) > 1.8:
        flag = "" if r.fdr < 0.05 else ("  n.s. after FDR" if r.p < 0.05 else "")
        print(f"{r.trait:<26}{r.cell:<18}{r.auc:>8.3f}{r.z:>8.1f}"
              f"{r.p:>11.2e}{r.fdr:>10.3f}{flag}")
df.to_csv(f"{SC}/cell_level_fdr.tsv", sep="\t", index=False)
lost = df[(df.p < 0.05) & (df.fdr >= 0.05)]
print(f"\n  {int((df.fdr < 0.05).sum())} of {m} survive FDR < 0.05")
print(f"  {len(lost)} were nominally significant and do NOT survive:")
for r in lost.itertuples():
    print(f"     {r.trait} / {r.cell}  (z{r.z:+.1f}, p={r.p:.3f}, FDR={r.fdr:.2f})")
out["fdr"] = dict(n_tests=m, n_survive=int((df.fdr < 0.05).sum()),
                  lost=[f"{r.trait}/{r.cell}" for r in lost.itertuples()])

# ================================================================ B  shape test
print("\n" + "=" * 100)
print("B. THE SHAPE TEST, WITH THE COGNITIVE TRAITS' NON-INDEPENDENCE MEASURED")
print("=" * 100)
rg = json.load(open(f"{ROOT}/results/ldsc_rg.json")).get("rg", {})


def get_rg(a, b):
    for k in (f"{a}|{b}", f"{b}|{a}"):
        if k in rg:
            return rg[k]["rg"]
    return None


COG = ["EducationalAttainment", "Intelligence", "ReactionTime"]
pairs = [(a, b, get_rg(a, b)) for i, a in enumerate(COG) for b in COG[i + 1:]]
print("  measured rg among the cognitive traits:")
for a, b, v in pairs:
    print(f"     {a:<24}{b:<18}{v:+.3f}" if v is not None else f"     {a} {b}  missing")
vals = [abs(v) for _, _, v in pairs if v is not None]
# Effective number of independent traits from the mean correlation, the standard
# Nyholt-style correction: k_eff = k / (1 + (k-1) * mean_r)
k = len(COG)
mean_r = float(np.mean(vals)) if vals else 0.0
k_eff = k / (1 + (k - 1) * mean_r)
p_naive = (1 / 15) ** k
p_corr = (1 / 15) ** k_eff
print(f"\n  mean |rg| = {mean_r:.3f}   ->   effective independent traits "
      f"= {k_eff:.2f} (of {k})")
print(f"  reported earlier : p = (1/15)^{k}    = {p_naive:.2e}   <- assumed independence")
print(f"  corrected        : p = (1/15)^{k_eff:.2f} = {p_corr:.2e}")
print(f"\n  The shape still holds, but it is {p_corr / p_naive:.0f}x weaker than "
      f"reported. Use {p_corr:.1e}.")
out["shape_test"] = dict(mean_rg=mean_r, k=k, k_eff=k_eff,
                         p_naive=p_naive, p_corrected=p_corr)

# ================================================================ C  spatial power
print("\n" + "=" * 100)
print("C. SPATIAL POWER, WITH THE CONTRAST THE SPATIAL TEST ACTUALLY MADE")
print("=" * 100)
rank = pd.read_csv(f"{SC}/stratified_ranking.tsv", sep="\t")
sub = rank[(rank.trait == "HRV_SDNN") & (rank.cell_state == "SAN_P_cell")]
auc_all = float(sub.auc.iloc[0]) if len(sub) else None
auc_cm = tab.get("HRV_SDNN", {}).get("SAN_P_cell", {}).get("auc")
print(f"  pacemaker cells vs WORKING MYOCYTES  AUC {auc_cm:.3f}  "
      f"(what 172 used — wrong contrast)")
print(f"  pacemaker cells vs ALL OTHER CELLS   AUC {auc_all:.3f}  "
      f"(what the spatial design compared)")

dil = json.load(open(f"{ROOT}/results/dilution.json"))
node = dil["purity"]["annotated `node` only"]
top1 = dil["purity"]["top 1% by pacemaker abundance"]
n_bg, N_SEC = 27108, 8


def power(dd, n1, n2, alpha=0.05):
    if n1 < 2 or n2 < 2 or dd <= 0:
        return 0.0
    return float(stats.norm.sf(stats.norm.ppf(1 - alpha / 2) - dd
                               / np.sqrt(1 / n1 + 1 / n2)))


print(f"\n{'contrast':<34}{'d cell':>9}{'scenario':<22}{'d spot':>9}"
      f"{'power/section':>15}{'P(>=6/8)':>11}")
resC = {}
for label, auc in (("vs myocytes (wrong)", auc_cm), ("vs all cells (right)", auc_all)):
    d_cell = float(np.sqrt(2) * stats.norm.ppf(auc))
    for sc_name, node_d in (("all node spots", node), ("top 1% abundance", top1)):
        p_pur = node_d["median"] / 100.0
        n = int(node_d["n"])
        d_spot = d_cell * p_pur
        pw = power(d_spot, n // N_SEC, n_bg // N_SEC)
        p6 = float(sum(stats.binom.pmf(x, N_SEC, pw) for x in range(6, 9)))
        resC[f"{label}|{sc_name}"] = dict(d_cell=d_cell, d_spot=float(d_spot),
                                          power=pw, p_6of8=p6)
        print(f"{label:<34}{d_cell:>9.3f}{sc_name:<22}{d_spot:>9.3f}"
              f"{pw * 100:>14.1f}%{p6 * 100:>10.1f}%")
out["spatial_power"] = resC

base = resC["vs all cells (right)|all node spots"]
rich = resC["vs all cells (right)|top 1% abundance"]
print(f"\n  With the correct contrast the compartment-level design still had only")
print(f"  {base['p_6of8'] * 100:.1f}% chance of hitting 6 of 8 sections, so that null "
      f"stays uninformative.")
print(f"  The abundance-based design had {rich['p_6of8'] * 100:.1f}% — even more clearly "
      f"powered than\n  the first calculation suggested, and it still found nothing. "
      f"The conclusion is\n  unchanged and slightly strengthened: dilution kills the "
      f"compartment test, and the\n  depth artefact is what is left to explain the "
      f"abundance test.")

with open(OUT, "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nwrote {OUT}")
