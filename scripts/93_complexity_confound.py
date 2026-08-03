"""
Is "node affinity" really transcriptional COMPLEXITY meeting POLYGENICITY?
=========================================================================

Script 92 broke the positive result from script 91. The per-trait ordering came out
backwards: resting heart rate favours the node, but so does educational attainment
almost as strongly (+0.093, 8/8, p = 0.0078), while heart-rate variability — the trait
that measures autonomic control of the pacemaker itself — favours it LEAST. No account
of conduction biology produces that ordering.

A specific alternative does produce it.

  * Node spots are unusually gene-DIVERSE for their depth: 2,246 genes at 5,865 UMI,
    against atrial myocardium's 2,324 genes at 7,964 UMI. Per UMI spent, a node spot
    detects far more distinct genes. Stratifying on total UMI — which every correction
    so far has done — does not control this at all.
  * A highly polygenic trait spreads its heritability thinly over many genes, so its
    spot-level score rises wherever many genes are detectable. Educational attainment
    is the most polygenic trait in common use; resting heart rate is also highly
    polygenic. Atrial fibrillation and PR interval are comparatively locus-driven
    (PITX2; SCN5A/SCN10A, TBX5), and HRV has few loci at all.

Together those two facts predict exactly the observed ordering, with no conduction
biology involved. Two tests:

  TEST 1  Stratify on genes detected instead of UMI, and on complexity residualised
          against depth. If node affinity survives UMI-stratification but dies under
          complexity-stratification, complexity is the confound.
  TEST 2  Across traits, does node affinity track a polygenicity proxy? If node
          affinity is essentially a readout of how spread-out a trait's heritability
          is, the correlation will be strong and the biological reading collapses.

Usage:  python scripts/93_complexity_confound.py
"""

import glob
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent.parent.as_posix()
GS = f"{ROOT}/data/spatial_gsmap"
GWAS = f"{ROOT}/data/gwas_gsmap"
SPOT = f"{ROOT}/results/spotlevel"
OUT = f"{ROOT}/results"
NBIN, MINCELL = 10, 20

TRAITS = ["RestingHeartRate", "AtrialFibrillation", "PRinterval", "HRV_RMSSD",
          "HRV_RMSSDc", "HRV_SDNN", "HRV_SDNNc", "EducationalAttainment"]
RELIABLE = ["RestingHeartRate", "AtrialFibrillation", "PRinterval",
            "HRV_RMSSD", "HRV_RMSSDc", "EducationalAttainment"]


def harmonise(ann):
    return np.where(ann == "myocardium", "myocardium_atrial", ann)


qc = {}
for p in sorted(glob.glob(f"{GS}/SAN__*.h5ad")):
    s = os.path.basename(p)[:-5]
    a = ad.read_h5ad(p)
    X = a.layers["count"] if "count" in a.layers else a.X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    umi = np.asarray(X.sum(axis=1)).ravel()
    ngene = np.asarray((X > 0).sum(axis=1)).ravel()
    # complexity residualised against depth: genes detected beyond what this spot's
    # sequencing depth predicts. This is the axis UMI-stratification cannot see.
    lx = np.log10(np.maximum(umi, 1))
    ly = np.log10(np.maximum(ngene, 1))
    b = np.polyfit(lx, ly, 1)
    qc[s] = pd.DataFrame(
        {"umi": umi, "ngene": ngene, "complexity": ly - (b[0] * lx + b[1]),
         "ann": harmonise(a.obs["annotation_final"].astype(str).to_numpy())},
        index=a.obs_names)

sections = sorted(qc)
print(f"sections: {len(sections)}\n")

print("=" * 92)
print("Per-compartment depth vs complexity (pooled over 8 SAN sections)")
print("=" * 92)
allq = pd.concat(qc.values())
g = allq.groupby("ann").agg(spots=("umi", "size"), umi=("umi", "median"),
                            genes=("ngene", "median"),
                            complexity=("complexity", "median"))
g["genes_per_kUMI"] = g.genes / g.umi * 1000
print(f"{'compartment':<26}{'spots':>8}{'UMI':>9}{'genes':>8}"
      f"{'genes/kUMI':>12}{'complexity':>12}")
for c, r in g.sort_values("complexity", ascending=False).iterrows():
    mark = "  <<<" if c == "node" else ""
    print(f"{c:<26}{int(r.spots):>8,}{r.umi:>9,.0f}{r.genes:>8,.0f}"
          f"{r.genes_per_kUMI:>12.1f}{r.complexity:>+12.3f}{mark}")

report = {"compartment_complexity": g.to_dict("index")}


def load(s, t):
    p = f"{SPOT}/{s}__{t}.csv"
    return pd.read_csv(p, index_col=0) if os.path.exists(p) else None


def node_effect(s, t, stratify_on):
    d = load(s, t)
    if d is None:
        return np.nan
    c = d.index.intersection(qc[s].index)
    m = pd.DataFrame({"logp": d.loc[c, "logp"].to_numpy(),
                      "key": qc[s].loc[c, stratify_on].to_numpy(),
                      "ann": qc[s].loc[c, "ann"].to_numpy()})
    m["bin"] = pd.qcut(m.key.rank(method="first"), NBIN, labels=False)
    eff, w = [], []
    for b in range(NBIN):
        sub = m[m.bin == b]
        inn, rest = sub[sub.ann == "node"], sub[sub.ann != "node"]
        if len(inn) < MINCELL or len(rest) < MINCELL:
            continue
        eff.append(inn.logp.mean() - rest.logp.mean())
        w.append(len(inn))
    return float(np.average(eff, weights=w)) if eff else np.nan


# ================================================================ TEST 1
print("\n" + "=" * 92)
print("TEST 1.  Node affinity under three stratification variables")
print("=" * 92)
tables = {}
for key in ["umi", "ngene", "complexity"]:
    rows = [dict(section=s, trait=t, eff=node_effect(s, t, key))
            for s in sections for t in TRAITS]
    df = pd.DataFrame(rows).pivot(index="section", columns="trait", values="eff")
    df = df[[t for t in TRAITS if t in df.columns]]
    tables[key] = df.sub(df.mean(axis=1), axis=0)      # centre within section
    tables[key].to_csv(f"{OUT}/complexity_node_{key}.tsv", sep="\t")

print(f"{'trait':<24}" + "".join(f"{k:>22}" for k in tables))
t1 = {}
for t in TRAITS:
    line = f"{t:<24}"
    for key, df in tables.items():
        if t not in df.columns:
            line += f"{'-':>22}"
            continue
        v = df[t].dropna().to_numpy()
        k = int((v > 0).sum())
        pv = wilcoxon(v).pvalue if len(v) >= 6 else np.nan
        star = "*" if np.isfinite(pv) and pv < 0.05 else ""
        line += f"{v.mean():>+12.3f} {f'{k}/{len(v)}':>4}{star:<5}"
        t1.setdefault(t, {})[key] = dict(mean=float(v.mean()), pos=k, n=len(v),
                                         p=float(pv) if np.isfinite(pv) else None)
    print(line + ("   <- non-cardiac control" if t == "EducationalAttainment" else ""))
print("\n(centred node effect, sections-positive/total, * = signed-rank p<0.05)")
report["test1"] = t1

rhr = t1.get("RestingHeartRate", {})
edu = t1.get("EducationalAttainment", {})
print(f"\n{'stratified on':<16}{'RHR':>10}{'control':>10}{'difference':>13}")
for key in tables:
    a_, b_ = rhr.get(key, {}).get("mean", np.nan), edu.get(key, {}).get("mean", np.nan)
    print(f"{key:<16}{a_:>+10.3f}{b_:>+10.3f}{a_ - b_:>+13.3f}")
print("\nIf the node effect is complexity, it should shrink most under 'complexity'")
print("stratification, and the control should track resting heart rate throughout.")

# ================================================================ TEST 2
print("\n" + "=" * 92)
print("TEST 2.  Does node affinity track POLYGENICITY across traits?")
print("=" * 92)
poly = {}
for t in TRAITS:
    p = f"{GWAS}/{t}.sumstats.gz"
    if not os.path.exists(p):
        continue
    z = pd.read_csv(p, sep="\t", usecols=["Z"]).Z.to_numpy()
    z = z[np.isfinite(z)]
    chi = z ** 2
    top = np.sort(chi)[::-1]
    n_gws = int((chi > 29.7).sum())
    # share of total excess chi-square carried by the strongest 1,000 SNPs: low share
    # = heritability spread thinly = highly polygenic
    excess = np.maximum(chi - 1, 0)
    conc = float(excess[np.argsort(chi)[::-1][:1000]].sum() / max(excess.sum(), 1e-9))
    poly[t] = dict(n_snp=int(z.size), mean_chi2=float(chi.mean()),
                   n_gws=n_gws, top1k_share=conc)

pd.DataFrame(poly).T.to_csv(f"{OUT}/complexity_polygenicity.tsv", sep="\t")
print(f"{'trait':<24}{'mean chi2':>11}{'gw-sig':>9}{'top-1k share':>14}"
      f"{'node affinity':>15}")
for t in TRAITS:
    if t not in poly or t not in t1:
        continue
    na = t1[t].get("complexity", {}).get("mean", np.nan)
    print(f"{t:<24}{poly[t]['mean_chi2']:>11.3f}{poly[t]['n_gws']:>9,}"
          f"{poly[t]['top1k_share'] * 100:>13.1f}%{na:>+15.3f}")

use = [t for t in TRAITS if t in poly and t in t1 and t in RELIABLE]
if len(use) >= 5:
    na = np.array([t1[t]["complexity"]["mean"] for t in use])
    for lbl, key in [("mean chi-square", "mean_chi2"),
                     ("top-1k concentration", "top1k_share")]:
        x = np.array([poly[t][key] for t in use])
        r, pv = pearsonr(x, na)
        rs = spearmanr(x, na)[0]
        print(f"\nnode affinity vs {lbl:<24} r = {r:+.3f} (rho {rs:+.3f}), p = {pv:.4f}")
        report.setdefault("test2", {})[key] = dict(r=float(r), p=float(pv))
    print("\nA strong NEGATIVE correlation with top-1k concentration means the traits")
    print("whose heritability is most spread out are the ones that favour the node —")
    print("i.e. node affinity is a polygenicity readout, not conduction biology.")

with open(f"{OUT}/complexity_confound.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwrote {OUT}/complexity_*.tsv and complexity_confound.json")
