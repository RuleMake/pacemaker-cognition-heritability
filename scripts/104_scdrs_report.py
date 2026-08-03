"""
Present the scDRS results with each trait's GWAS power alongside them.
=====================================================================

A null from an underpowered trait is not a biological finding, and the two are
indistinguishable in a table that reports only p-values. This project has already been
burned twice by that: a schizophrenia control chosen on headline sample size that held
one genome-wide significant SNP, and four heart-rate-variability indices whose spatial
results turned out to track GWAS power at r = +0.95 rather than biology.

The atrioventricular-block GWAS makes the point again. It has 617,488 participants and
436 genome-wide significant SNPs, with lambda_GC = 1.010 — essentially no polygenic
inflation. Whatever it shows about the AV node, "nothing" would be the expected result
even if the AV node were the whole story.

So every trait is reported next to:
    gw-sig      genome-wide significant SNPs — the honest power summary
    lambda_GC   genomic inflation; near 1.00 means little polygenic signal
    max Z       the strongest gene, from MAGMA
    median Z    how much signal is spread across all genes
and traits below the reproducibility floor this project measured (between 920 and
1,585 genome-wide significant SNPs, script 51) are marked so a null from them is not
read as evidence of absence.

Usage:  python scripts/104_scdrs_report.py
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.as_posix()
SC = f"{ROOT}/results/scdrs"
GWAS = f"{ROOT}/data/gwas_gsmap"
MAGMA = os.path.expanduser("~/cardio/magma/genes")
GENELOC = os.path.expanduser("~/cardio/tools/NCBI37.3.gene.loc")

CONDUCTION = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
FLOOR = 1585          # measured reproducibility floor, script 51 TEST B

g = f"{SC}/group_analysis.tsv"
if not os.path.exists(g):
    raise SystemExit(f"{g} not found — scDRS has not finished")
grp = pd.read_csv(g, sep="\t")
key = "cell_state" if "cell_state" in grp.columns else grp.columns[0]

# ---------------------------------------------------------------- power per trait
power = {}
for p in sorted(glob.glob(f"{GWAS}/*.sumstats.gz")):
    t = os.path.basename(p).replace(".sumstats.gz", "")
    z = pd.read_csv(p, sep="\t", usecols=["Z"]).Z.to_numpy()
    z = z[np.isfinite(z)]
    power[t] = dict(n_snp=int(z.size), n_gws=int((np.abs(z) > 5.45).sum()),
                    lambda_gc=float(np.median(z ** 2) / 0.4549))
for p in sorted(glob.glob(f"{MAGMA}/*.genes.out")):
    t = os.path.basename(p).replace(".genes.out", "")
    if ".chr" in t or t not in power:
        continue
    d = pd.read_csv(p, sep=r"\s+")
    power[t]["max_z"] = float(d.ZSTAT.max())
    power[t]["median_z"] = float(d.ZSTAT.median())

traits = [t for t in grp.trait.unique() if t in power]
print("=" * 104)
print("GWAS POWER — read every result below against this table")
print("=" * 104)
print(f"{'trait':<24}{'SNPs':>13}{'gw-sig':>9}{'lambda_GC':>11}"
      f"{'max Z':>8}{'median Z':>10}   power")
for t in sorted(traits, key=lambda x: -power[x]["n_gws"]):
    d = power[t]
    tag = ("adequate" if d["n_gws"] >= FLOOR else
           "MARGINAL" if d["n_gws"] >= 920 else "BELOW FLOOR — nulls uninformative")
    print(f"{t:<24}{d['n_snp']:>13,}{d['n_gws']:>9,}{d['lambda_gc']:>11.3f}"
          f"{d.get('max_z', float('nan')):>8.2f}{d.get('median_z', float('nan')):>10.2f}"
          f"   {tag}")

# ---------------------------------------------------------------- multiple testing
# 62 cell states are tested per trait. With 1,000 control sets the Monte Carlo p-value
# cannot go below ~0.001, so Bonferroni at 0.05/62 = 0.0008 is past the resolution of
# the test and would reject everything by construction. Benjamini-Hochberg within each
# trait is the appropriate correction.
def bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


grp["assoc_fdr"] = np.nan
for t in grp.trait.unique():
    m = grp.trait == t
    grp.loc[m, "assoc_fdr"] = bh(grp.loc[m, "assoc_mcp"].to_numpy())
grp.to_csv(f"{SC}/group_analysis_fdr.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- conduction cells
print("\n" + "=" * 104)
print("CONDUCTION-SYSTEM CELL STATES  (z, with FDR across the 62 states in each trait)")
print("=" * 104)
print(f"{'cell state':<20}{'n':>7}" + "".join(f"{t[:15]:>17}" for t in traits))
for cs in CONDUCTION:
    sub = grp[grp[key] == cs]
    if sub.empty:
        print(f"{cs:<20}{'absent':>7}")
        continue
    n = int(sub["n_cell"].iloc[0]) if "n_cell" in sub.columns else 0
    line = f"{cs:<20}{n:>7,}"
    for t in traits:
        r = sub[sub.trait == t]
        if r.empty:
            line += f"{'-':>17}"
            continue
        z, q = float(r.assoc_mcz.iloc[0]), float(r.assoc_fdr.iloc[0])
        star = "**" if q < 0.05 else "*" if q < 0.10 else ""
        line += f"{z:>+12.2f}{star:<5}"
    print(line + ("   (n<50, unreliable)" if 0 < n < 50 else ""))
print("\n(* FDR<0.10, ** FDR<0.05, within trait across 62 cell states)")

# ---------------------------------------------------------------- relative ranking
# Absolute z is not comparable across traits — a better-powered GWAS lifts every cell
# type at once, which is exactly how the spatial analysis was misled (node affinity
# tracked mean chi-square at r = +0.95). Rank within trait is immune to that: GWAS
# power scales all cell types together and leaves their order unchanged.
print("\n" + "=" * 104)
print("RANK OF EACH CONDUCTION CELL STATE WITHIN ITS TRAIT  (1 = strongest of 62)")
print("=" * 104)
print(f"{'cell state':<20}" + "".join(f"{t[:15]:>17}" for t in traits))
rank_tbl = {}
for cs in CONDUCTION:
    line = f"{cs:<20}"
    for t in traits:
        sub = grp[grp.trait == t].sort_values("assoc_mcz", ascending=False)
        lst = list(sub[key])
        if cs not in lst:
            line += f"{'-':>17}"
            continue
        r = lst.index(cs) + 1
        rank_tbl.setdefault(cs, {})[t] = r
        line += f"{f'{r}/{len(lst)}':>17}"
    print(line)
print("\nCompare each cardiac trait against EducationalAttainment in the same row:")
print("a conduction cell state that ranks high for cardiac traits AND for a")
print("non-cardiac control is reflecting data structure, not biology — that is")
print("precisely the pattern that sank the spatial analysis.")

# ---------------------------------------------------------------- where traits land
print("\n" + "=" * 104)
print("WHERE EACH TRAIT LANDS  (top 5 cell states by association z)")
print("=" * 104)
out = {}
for t in traits:
    sub = grp[grp.trait == t].sort_values("assoc_mcz", ascending=False)
    top = [f"{r[key]}({r.assoc_mcz:+.1f})" for _, r in sub.head(5).iterrows()]
    ranks = {cs: (list(sub[key]).index(cs) + 1) for cs in CONDUCTION
             if cs in list(sub[key])}
    flag = "" if power[t]["n_gws"] >= 920 else "   [underpowered]"
    print(f"\n{t:<24}{flag}")
    print(f"   top: {', '.join(top)}")
    print("   " + "  ".join(f"{cs} {r}/{len(sub)}" for cs, r in ranks.items()))
    out[t] = dict(top5=[r[key] for _, r in sub.head(5).iterrows()],
                  conduction_ranks=ranks, n_groups=int(len(sub)),
                  **power[t])

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 104)
print("VERDICT")
print("=" * 104)
CTRL = "EducationalAttainment"
acm = [c for c in grp[key].unique() if str(c).startswith("aCM")]
if "AtrialFibrillation" in traits and acm:
    sub = grp[grp.trait == "AtrialFibrillation"].sort_values("assoc_mcz",
                                                             ascending=False)
    lst = list(sub[key])
    best = min(lst.index(c) + 1 for c in acm if c in lst)
    ok = bool((grp[(grp.trait == "AtrialFibrillation") & (grp[key].isin(acm))]
               .assoc_fdr < 0.05).any())
    print(f"positive control  atrial fibrillation -> atrial cardiomyocytes: "
          f"best aCM rank {best}/{len(lst)}, FDR<0.05: {ok}")
    print(f"   -> pipeline {'VALIDATED' if ok and best <= 15 else 'QUESTIONABLE'}")

for cs in ["SAN_P_cell", "AVN_P_cell"]:
    if cs not in rank_tbl:
        continue
    r = rank_tbl[cs]
    card = {t: v for t, v in r.items() if t != CTRL}
    ctrl_rank = r.get(CTRL)
    print(f"\n{cs}:")
    for t, v in sorted(card.items(), key=lambda kv: kv[1]):
        pw = "" if power[t]["n_gws"] >= 920 else "  [underpowered]"
        print(f"   {t:<24}rank {v}{pw}")
    if ctrl_rank:
        print(f"   {CTRL:<24}rank {ctrl_rank}   <- non-cardiac control")
        better = [t for t, v in card.items() if v < ctrl_rank
                  and power[t]["n_gws"] >= 920]
        print(f"   => {len(better)} adequately-powered cardiac trait(s) rank this "
              f"cell state above the control: {', '.join(better) if better else 'none'}")

with open(f"{SC}/scdrs_report.json", "w") as f:
    json.dump(dict(traits=out, conduction_ranks=rank_tbl), f, indent=2, default=float)
print(f"\nwrote {SC}/scdrs_report.json and group_analysis_fdr.tsv")
