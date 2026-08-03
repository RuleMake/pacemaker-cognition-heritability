"""
Rank every trait by how much it favours the node — a pre-specified ordering test.
=================================================================================

Script 91 established that the node compartment and pacemaker-cell abundance favour
resting heart rate over atrial fibrillation and over PR interval, 8/8 sections, with
the built-in negative control silent. Two contrasts is a thin basis for a claim, and
pairwise tests waste information: if a per-trait "node affinity" exists at all, every
pairwise contrast is a difference of two such numbers, and all of them should be
estimated together.

Design
------
Per section and trait, compute the depth-stratified node-versus-rest effect. Then
CENTRE it across traits within that section. Centring removes everything the section
contributes in common — its depth profile, its cell composition, its overall signal
level — and leaves a relative node affinity per trait that is comparable across
sections. Each trait is then tested against zero over the eight sections.

The pre-specified prediction, written before looking
----------------------------------------------------
Heart-rate variability is the most sinoatrial trait available: it measures beat-to-beat
autonomic modulation of the pacemaker itself, whereas resting heart rate is a steady
state that contractility and fitness also set. So:

    HRV >= resting heart rate  >  PR interval  >  atrial fibrillation  >  education

If the observed ordering matches, the node result is trait-specific in a way an
artefact cannot easily produce, since nothing about depth or spot composition knows
which trait is autonomic. If HRV lands at the bottom, the script-91 result is more
likely to be something peculiar to resting heart rate's GWAS rather than node biology.

Usage:  python scripts/92_node_trait_ranking.py
"""

import glob
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent.parent.as_posix()
GS = f"{ROOT}/data/spatial_gsmap"
SRC = f"{ROOT}/data/spatial"
SPOT = f"{ROOT}/results/spotlevel"
OUT = f"{ROOT}/results"
NBIN, MINCELL = 10, 20
ATRIAL = ["aCM1", "aCM2", "aCM3", "aCM4"]

# ordered by how directly each interrogates sinoatrial pacemaking
PREDICTED = ["HRV_RMSSDc", "HRV_RMSSD", "HRV_SDNNc", "HRV_SDNN", "RestingHeartRate",
             "PRinterval", "AtrialFibrillation", "EducationalAttainment"]
RELIABLE = ["HRV_RMSSDc", "HRV_RMSSD", "RestingHeartRate", "PRinterval",
            "AtrialFibrillation", "EducationalAttainment"]   # cross-section r >= 0.74


def harmonise(ann):
    return np.where(ann == "myocardium", "myocardium_atrial", ann)


def partial_spearman(x, y, *ctrl):
    ok = np.isfinite(x) & np.isfinite(y)
    for c in ctrl:
        ok &= np.isfinite(c)
    if ok.sum() < 200:
        return np.nan
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    Z = np.column_stack([rankdata(c[ok]) for c in ctrl] + [np.ones(ok.sum())])
    rx = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ry = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


qc = {}
for p in sorted(glob.glob(f"{GS}/SAN__*.h5ad")):
    s = os.path.basename(p)[:-5]
    a = ad.read_h5ad(p)
    X = a.layers["count"] if "count" in a.layers else a.X
    qc[s] = pd.DataFrame(
        {"umi": np.asarray(X.sum(axis=1)).ravel(),
         "ann": harmonise(a.obs["annotation_final"].astype(str).to_numpy())},
        index=a.obs_names)

abund = {}
a = ad.read_h5ad(f"{SRC}/SAN.h5ad", backed="r")
sid = a.obs["sangerID"].astype(str)
cols = [c for c in ["SAN_P_cell"] + ATRIAL if c in a.obs.columns]
for s in sid.unique():
    m = (sid == s).to_numpy()
    sub = a.obs.loc[m, cols].astype(float)
    sub.index = a.obs_names[m]
    abund[f"SAN__{s}"] = sub

sections = sorted(qc)
traits = [t for t in PREDICTED
          if any(os.path.exists(f"{SPOT}/{s}__{t}.csv") for s in sections)]
print(f"sections {len(sections)}   traits {len(traits)}\n")


def load(s, t):
    p = f"{SPOT}/{s}__{t}.csv"
    return pd.read_csv(p, index_col=0) if os.path.exists(p) else None


# ---------------------------------------------------------- compartment level
rows = []
for s in sections:
    for t in traits:
        d = load(s, t)
        if d is None:
            continue
        c = d.index.intersection(qc[s].index)
        m = pd.DataFrame({"logp": d.loc[c, "logp"].to_numpy(),
                          "umi": qc[s].loc[c, "umi"].to_numpy(),
                          "ann": qc[s].loc[c, "ann"].to_numpy()})
        m["bin"] = pd.qcut(m.umi.rank(method="first"), NBIN, labels=False)
        eff, w = [], []
        for b in range(NBIN):
            sub = m[m.bin == b]
            inn, rest = sub[sub.ann == "node"], sub[sub.ann != "node"]
            if len(inn) < MINCELL or len(rest) < MINCELL:
                continue
            eff.append(inn.logp.mean() - rest.logp.mean())
            w.append(len(inn))
        if eff:
            rows.append(dict(section=s, trait=t,
                             node_eff=float(np.average(eff, weights=w))))
comp = pd.DataFrame(rows).pivot(index="section", columns="trait", values="node_eff")
comp = comp[[t for t in traits if t in comp.columns]]
comp_c = comp.sub(comp.mean(axis=1), axis=0)          # centre within section

# ---------------------------------------------------------- cell-type level
rows = []
for s in sections:
    if s not in abund or "SAN_P_cell" not in abund[s].columns:
        continue
    for t in traits:
        d = load(s, t)
        if d is None:
            continue
        c = d.index.intersection(qc[s].index).intersection(abund[s].index)
        if len(c) < 200:
            continue
        san = abund[s].loc[c, "SAN_P_cell"].to_numpy()
        if not np.isfinite(san).any() or np.nanstd(san) == 0:
            continue
        acm = abund[s].loc[c, [x for x in ATRIAL
                               if x in abund[s].columns]].sum(axis=1).to_numpy()
        rows.append(dict(section=s, trait=t,
                         pr=partial_spearman(d.loc[c, "logp"].to_numpy(), san,
                                             qc[s].loc[c, "umi"].to_numpy(), acm)))
cell = pd.DataFrame(rows).pivot(index="section", columns="trait", values="pr")
cell = cell[[t for t in traits if t in cell.columns]]
cell_c = cell.sub(cell.mean(axis=1), axis=0)

comp_c.to_csv(f"{OUT}/node_affinity_compartment.tsv", sep="\t")
cell_c.to_csv(f"{OUT}/node_affinity_celltype.tsv", sep="\t")

print("=" * 96)
print("RELATIVE NODE AFFINITY PER TRAIT  (centred within section, so section-level")
print("depth and composition cancel; tested against zero over 8 sections)")
print("=" * 96)
print(f"{'trait':<24}{'compartment':>13}{'+/n':>8}{'p':>9}"
      f"{'| cell type':>14}{'+/n':>8}{'p':>9}   reliable?")
res = {}
for t in traits:
    line = f"{t:<24}"
    entry = {}
    for lbl, df in [("comp", comp_c), ("cell", cell_c)]:
        if t not in df.columns:
            line += f"{'-':>13}{'-':>8}{'-':>9}"
            continue
        v = df[t].dropna().to_numpy()
        k = int((v > 0).sum())
        pv = wilcoxon(v).pvalue if len(v) >= 6 else np.nan
        entry[lbl] = dict(mean=float(v.mean()), pos=k, n=len(v),
                          p=float(pv) if np.isfinite(pv) else None)
        line += (f"{v.mean():>+13.3f}{f'{k}/{len(v)}':>8}"
                 f"{(f'{pv:.4f}' if np.isfinite(pv) else '-'):>9}")
    res[t] = entry
    line += f"   {'yes' if t in RELIABLE else 'NO (underpowered GWAS)'}"
    print(line)

# ---------------------------------------------------------- ordering test
print("\n" + "=" * 96)
print("PRE-SPECIFIED ORDERING TEST")
print("=" * 96)
pred = [t for t in PREDICTED if t in res and t in RELIABLE]
print("predicted (most sinoatrial first):")
print("   " + "  >  ".join(pred))
for lbl, key in [("compartment", "comp"), ("cell type", "cell")]:
    obs = sorted([t for t in pred if key in res[t]],
                 key=lambda t: -res[t][key]["mean"])
    print(f"\nobserved, {lbl}:")
    print("   " + "  >  ".join(obs))
    if len(obs) >= 4:
        pr = [pred.index(t) for t in obs]
        rho, pv = spearmanr(pr, range(len(obs)))
        print(f"   rank agreement with prediction: rho = {rho:+.3f}, p = {pv:.4f}")
        res.setdefault("_ordering", {})[key] = dict(rho=float(rho), p=float(pv),
                                                    observed=obs)

print("\n" + "=" * 96)
print("READING")
print("=" * 96)
hrv = [t for t in ["HRV_RMSSD", "HRV_RMSSDc"] if t in res]
hrv_top = all(res[t].get("cell", {}).get("mean", -9) > 0 for t in hrv) if hrv else False
rhr = res.get("RestingHeartRate", {}).get("cell", {}).get("mean", 0)
af = res.get("AtrialFibrillation", {}).get("cell", {}).get("mean", 0)
if hrv_top and rhr > af:
    print("HRV — the trait that measures autonomic modulation of the pacemaker itself —")
    print("also favours the node, and atrial fibrillation does not. Nothing about")
    print("sequencing depth or spot composition knows which trait is autonomic, so an")
    print("ordering that tracks conduction-system physiology is hard to produce by")
    print("artefact. This strengthens the script-91 result.")
elif rhr > af:
    print("Resting heart rate favours the node over atrial fibrillation, but HRV does")
    print("not follow. Either the HRV GWAS is too weak to place (SDNN especially), or")
    print("the effect is specific to resting heart rate rather than to pacemaking.")
    print("Check the reliability column before concluding.")
else:
    print("The ordering does not follow conduction-system physiology. Treat the")
    print("script-91 result with caution.")

with open(f"{OUT}/node_affinity.json", "w") as f:
    json.dump(res, f, indent=2, default=float)
print(f"\nwrote {OUT}/node_affinity_*.tsv and node_affinity.json")
