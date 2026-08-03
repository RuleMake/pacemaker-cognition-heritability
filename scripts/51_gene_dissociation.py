"""
Formalise the two things the gene diagnostic revealed.
======================================================

Test A — the node/myocyte dissociation
--------------------------------------
Script 50 showed by eye that canonical node-specifying genes (HCN4, SHOX2, TBX3,
TBX18, ISL1, BMP4) rank WORSE under cardiac traits than under a non-cardiac control,
while pan-cardiomyocyte genes rank far better. That needs a test, not an eyeball.

The clean design is a contrast of contrasts, because absolute PCC is not comparable
across traits (different GWAS power rescales everything):

    delta(trait) = percentile(node genes) - percentile(atrial myocyte genes)

computed within each trait, so the trait-wide scaling cancels. If heart-rate genetics
were pacemaker-executed, delta should be more negative (node ranked better) for
resting heart rate than for the control. The prediction is directional and testable.

Two gene-set definitions are used and must agree:
  curated   hand-picked canonical markers — mechanistically interpretable, small n
  empirical gsMap's own `Annotation` column, i.e. the compartment where each gene's
            specificity score peaks — unbiased and large n

Test B — how much GWAS power does gsMap need?
---------------------------------------------
The four HRV indices share one sample (n=46,075) but differ sharply in signal:
2,459 / 1,585 / 920 / 362 genome-wide significant SNPs. Their gene-level outputs
degraded in exactly that order. That makes them an accidental dose-response
experiment on gsMap itself: how much signal must a GWAS carry before gsMap's
gene-level output becomes reproducible rather than noise? Nobody has measured this,
and it is the kind of usage guidance a methods reader wants.

Usage:  python scripts/51_gene_dissociation.py
"""

from pathlib import Path
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, wilcoxon

ROOT = str(Path(__file__).resolve().parent.parent)
DIAG = f"{ROOT}/results/genediag"
GWAS = f"{ROOT}/data/gwas_gsmap"
OUT = f"{ROOT}/results"
CONTROL = "EducationalAttainment"

# Node-restricted: expression defines or is confined to the sinoatrial node.
# Deliberately excludes genes that are merely node-ENRICHED but broadly cardiac
# (SLC8A1, KCNJ3, TBX5), because those cannot separate the two hypotheses.
NODE_GENES = ["HCN1", "HCN4", "SHOX2", "TBX3", "TBX18", "ISL1", "BMP4", "VSNL1",
              "CACNA1D", "CACNA1G"]
# Working atrial myocyte: sarcomere, calcium handling, oxidative metabolism.
MYOCYTE_GENES = ["MYH6", "MYH7", "TNNT2", "TTN", "ACTC1", "MYL7", "MYOM2", "LDB3",
                 "PLN", "RYR2", "CASQ2", "ATP2A2", "NPPA", "SRL", "NEBL"]


def parse(path):
    b = os.path.basename(path).replace("_Gene_Diagnostic_Info.csv", "")
    m = re.match(r"((?:SAN|AVN)__[A-Za-z0-9]+)_(.+)$", b)
    return (m.group(1), m.group(2)) if m else (None, None)


tables = {}
for p in sorted(glob.glob(f"{DIAG}/*_Gene_Diagnostic_Info.csv")):
    s, t = parse(p)
    if s is None or not s.startswith("SAN"):
        continue
    df = pd.read_csv(p).dropna(subset=["PCC"]).drop_duplicates(subset="Gene")
    # percentile rank within this (section, trait): 0 = top gene, 100 = bottom.
    df["pct"] = (1 - df.PCC.rank(pct=True)) * 100
    tables[(s, t)] = df.set_index("Gene")

san = sorted({s for s, _ in tables})
traits = sorted({t for _, t in tables})
print(f"SAN sections: {len(san)}   traits: {len(traits)}\n")
report = {}

# ============================================================ TEST A
print("=" * 92)
print("TEST A.  delta = percentile(node genes) - percentile(atrial myocyte genes)")
print("         more negative = node genes ranked better than working-myocyte genes")
print("=" * 92)

rows = []
for (s, t), df in tables.items():
    # curated
    nod = df.reindex([g for g in NODE_GENES if g in df.index]).pct
    myo = df.reindex([g for g in MYOCYTE_GENES if g in df.index]).pct
    # empirical: gsMap's own compartment assignment for every gene
    ann = df["Annotation"].astype(str)
    e_nod = df.loc[ann == "node", "pct"]
    e_myo = df.loc[ann.isin(["myocardium_atrial", "myocardium"]), "pct"]
    rows.append(dict(
        section=s, trait=t,
        cur_node=nod.median(), cur_myo=myo.median(),
        cur_delta=nod.median() - myo.median(), n_cur_node=len(nod),
        emp_node=e_nod.median(), emp_myo=e_myo.median(),
        emp_delta=e_nod.median() - e_myo.median(),
        n_emp_node=len(e_nod), n_emp_myo=len(e_myo)))

d = pd.DataFrame(rows)
d.to_csv(f"{OUT}/genediag_dissociation.tsv", sep="\t", index=False)

order = [t for t in ["RestingHeartRate", "AtrialFibrillation", "PRinterval",
                     "HRV_RMSSD", "HRV_RMSSDc", "HRV_SDNN", "HRV_SDNNc", CONTROL]
         if t in traits]

print(f"{'trait':<24}{'curated node':>14}{'curated myo':>13}{'delta':>9}"
      f"{'| emp node':>12}{'emp myo':>10}{'delta':>9}")
for t in order:
    g = d[d.trait == t]
    tag = "  <- non-cardiac control" if t == CONTROL else ""
    print(f"{t:<24}{g.cur_node.mean():>13.1f}%{g.cur_myo.mean():>12.1f}%"
          f"{g.cur_delta.mean():>+9.1f}{g.emp_node.mean():>11.1f}%"
          f"{g.emp_myo.mean():>9.1f}%{g.emp_delta.mean():>+9.1f}{tag}")

print("\nPaired across the 8 SAN sections, each cardiac trait vs the control:")
print(f"{'trait':<24}{'curated delta diff':>20}{'p':>10}{'  empirical delta diff':>24}{'p':>10}")
ta = {}
for t in [x for x in order if x != CONTROL]:
    a = d[d.trait == t].set_index("section")
    b = d[d.trait == CONTROL].set_index("section")
    common = a.index.intersection(b.index)
    res = {}
    for key, lbl in [("cur_delta", "curated"), ("emp_delta", "empirical")]:
        diff = (a.loc[common, key] - b.loc[common, key]).dropna()
        p = wilcoxon(diff).pvalue if len(diff) >= 6 else np.nan
        res[lbl] = (float(diff.mean()), float(p) if np.isfinite(p) else None)
    ta[t] = res
    print(f"{t:<24}{res['curated'][0]:>+19.1f}"
          f"{(f'{res['curated'][1]:.4f}' if res['curated'][1] else '-'):>10}"
          f"{res['empirical'][0]:>+23.1f}"
          f"{(f'{res['empirical'][1]:.4f}' if res['empirical'][1] else '-'):>10}")
print("\npositive difference = node genes rank WORSE under the cardiac trait than under"
      "\na trait with no cardiac biology at all, relative to working myocyte genes.")
report["testA"] = ta

# ============================================================ TEST B
print("\n" + "=" * 92)
print("TEST B.  GWAS power vs reproducibility of gsMap's gene-level output")
print("=" * 92)

# signal content of each GWAS
sig = {}
for p in sorted(glob.glob(f"{GWAS}/*.sumstats.gz")):
    t = os.path.basename(p).replace(".sumstats.gz", "")
    z = pd.read_csv(p, sep="\t", usecols=["Z"]).Z.to_numpy()
    z = z[np.isfinite(z)]
    sig[t] = dict(n_snp=int(z.size), mean_chi2=float(np.mean(z ** 2)),
                  n_gws=int((np.abs(z) > 5.45).sum()))

# cross-section reproducibility: does the same trait give the same gene ranking on
# two independent sections? this needs no external reference and no other trait.
rep = {}
for t in traits:
    rs = []
    secs = [s for s in san if (s, t) in tables]
    for i, s1 in enumerate(secs):
        for s2 in secs[i + 1:]:
            a, b = tables[(s1, t)], tables[(s2, t)]
            c = a.index.intersection(b.index)
            if len(c) < 5000:
                continue
            rs.append(pearsonr(a.loc[c, "PCC"], b.loc[c, "PCC"])[0])
    if rs:
        rep[t] = float(np.mean(rs))

print(f"{'trait':<24}{'N':>11}{'gw-sig SNPs':>13}{'mean chi2':>11}"
      f"{'cross-section r':>17}")
brow = []
for t in order:
    s = sig.get(t)
    if s is None:
        continue
    n = pd.read_csv(f"{GWAS}/{t}.sumstats.gz", sep="\t", usecols=["N"], nrows=1).N.iloc[0]
    print(f"{t:<24}{n:>11,}{s['n_gws']:>13,}{s['mean_chi2']:>11.3f}"
          f"{rep.get(t, np.nan):>17.3f}")
    brow.append(dict(trait=t, N=int(n), n_gws=s["n_gws"],
                     mean_chi2=s["mean_chi2"], repro=rep.get(t, np.nan)))

bt = pd.DataFrame(brow).dropna(subset=["repro"])
bt.to_csv(f"{OUT}/genediag_power_repro.tsv", sep="\t", index=False)
if len(bt) >= 4:
    r1 = pearsonr(np.log10(bt.n_gws.clip(lower=1)), bt.repro)
    r2 = pearsonr(bt.mean_chi2, bt.repro)
    print(f"\nlog10(gw-sig SNPs) vs reproducibility : r = {r1[0]:+.3f}  p = {r1[1]:.4f}")
    print(f"mean chi-square    vs reproducibility : r = {r2[0]:+.3f}  p = {r2[1]:.4f}")
    lo = bt[bt.repro < 0.5]
    hi = bt[bt.repro >= 0.8]
    if len(lo) and len(hi):
        print(f"\nreproducible (r>=0.8) traits carry {hi.n_gws.min():,}-{hi.n_gws.max():,}"
              f" genome-wide significant SNPs")
        print(f"unreliable   (r<0.5)  traits carry {lo.n_gws.min():,}-{lo.n_gws.max():,}")
        print("=> practical floor for a trustworthy gsMap gene-level result sits between"
              f" {lo.n_gws.max():,} and {hi.n_gws.min():,} genome-wide significant SNPs")
    report["testB"] = dict(r_gws=float(r1[0]), p_gws=float(r1[1]),
                           r_chi2=float(r2[0]), p_chi2=float(r2[1]),
                           table=bt.to_dict("records"))

# ============================================================ TEST C
print("\n" + "=" * 92)
print("TEST C.  Is the node ranking under the control BETTER than under cardiac traits?")
print("         (direct, no myocyte reference — the simplest form of the question)")
print("=" * 92)
rc = {}
for t in [x for x in order if x != CONTROL]:
    a = d[d.trait == t].set_index("section")
    b = d[d.trait == CONTROL].set_index("section")
    c = a.index.intersection(b.index)
    for key, lbl in [("cur_node", "curated"), ("emp_node", "empirical")]:
        diff = (a.loc[c, key] - b.loc[c, key]).dropna()
        if len(diff) >= 6:
            p = wilcoxon(diff).pvalue
            rc.setdefault(t, {})[lbl] = (float(diff.mean()), float(p))
print(f"{'trait':<24}{'curated node pctile diff':>26}{'p':>9}"
      f"{'empirical':>13}{'p':>9}")
for t, v in rc.items():
    print(f"{t:<24}{v['curated'][0]:>+25.1f}{v['curated'][1]:>9.4f}"
          f"{v['empirical'][0]:>+13.1f}{v['empirical'][1]:>9.4f}")
print("\npositive = node genes sit LOWER in the cardiac trait's ranking than in the"
      "\nnon-cardiac control's ranking on the very same spots.")
report["testC"] = rc

with open(f"{OUT}/genediag_dissociation.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwrote {OUT}/genediag_dissociation.tsv, genediag_power_repro.tsv, "
      f"genediag_dissociation.json")
