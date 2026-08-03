"""
Depth-stratified compartment comparison — the statistical twin of the thinning run.
==================================================================================

The binomial-thinning re-run is an experimental correction: it destroys the confound
at the cost of throwing away 60-70% of every spot's counts, and it cannot be trusted
if the thinned data are too shallow for gsMap to work at all. This script asks the
same question statistically, using all the data.

Why stratification rather than regression
-----------------------------------------
The tempting fix is to regress logp on log(depth) and analyse the residual. That
over-corrects here, because depth and biology are collinear by construction: atrial
myocardium is both the deepest compartment AND where atrial-fibrillation biology
genuinely lives. Regressing depth out would delete the positive control along with
the artefact, and the result would be uninterpretable.

Stratification avoids this. Spots are binned into depth deciles; the compartment
comparison happens WITHIN a bin, where depth is nearly constant; the per-bin effects
are then pooled (Mantel-Haenszel style). Crucially the quantity compared is
trait-versus-control, so anything shared by both traits at that depth cancels without
any assumption that the artefact is a constant offset — the assumption that broke the
earlier paired contrast.

Reported for every compartment:

    strat_effect = mean over depth bins of
                   [ mean logp(trait, compartment, bin) - mean logp(trait, rest, bin) ]
                 - the same quantity computed for the non-cardiac control

Positive = that compartment carries more of this trait's heritability than the rest of
the section does, beyond what its sequencing depth and a non-cardiac trait explain.

Usage:  python scripts/56_depth_stratified.py
"""

import glob
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parent.parent.as_posix()
GS = f"{ROOT}/data/spatial_gsmap"
SPOT = f"{ROOT}/results/spotlevel"
OUT = f"{ROOT}/results"
CONTROL = "EducationalAttainment"
NBIN = 10
MINCELL = 20        # spots required in a (compartment, bin) cell to use it


def harmonise(ann, region):
    """SAN sections disagree on `myocardium` vs `myocardium_atrial`; see 31_*.py."""
    if region == "SAN":
        return np.where(ann == "myocardium", "myocardium_atrial", ann)
    return ann


qc = {}
for p in sorted(glob.glob(f"{GS}/*.h5ad")):
    s = os.path.basename(p)[:-5]
    a = ad.read_h5ad(p)
    X = a.layers["count"] if "count" in a.layers else a.X
    qc[s] = pd.DataFrame(
        {"umi": np.asarray(X.sum(axis=1)).ravel(),
         "ann": harmonise(a.obs["annotation_final"].astype(str).to_numpy(),
                          s.split("__")[0])},
        index=a.obs_names)
print(f"sections: {len(qc)}")

traits = sorted({os.path.basename(p)[:-4].rsplit("__", 1)[1]
                 for p in glob.glob(f"{SPOT}/*__*.csv")})
order = [t for t in ["RestingHeartRate", "AtrialFibrillation", "PRinterval",
                     "HRV_RMSSD", "HRV_RMSSDc", "HRV_SDNN", "HRV_SDNNc"] if t in traits]


def load(s, t):
    p = f"{SPOT}/{s}__{t}.csv"
    return pd.read_csv(p, index_col=0) if os.path.exists(p) else None


def stratified(s, t):
    """Per-compartment, depth-stratified logp advantage over the rest of the section."""
    d = load(s, t)
    if d is None:
        return None
    c = d.index.intersection(qc[s].index)
    if len(c) < 300:
        return None
    m = pd.DataFrame({"logp": d.loc[c, "logp"].to_numpy(),
                      "umi": qc[s].loc[c, "umi"].to_numpy(),
                      "ann": qc[s].loc[c, "ann"].to_numpy()})
    m["bin"] = pd.qcut(m.umi.rank(method="first"), NBIN, labels=False)
    out = {}
    for a in m.ann.unique():
        eff, w = [], []
        for b in range(NBIN):
            sub = m[m.bin == b]
            inn = sub[sub.ann == a]
            rest = sub[sub.ann != a]
            if len(inn) < MINCELL or len(rest) < MINCELL:
                continue
            eff.append(inn.logp.mean() - rest.logp.mean())
            w.append(len(inn))
        if eff:
            out[a] = float(np.average(eff, weights=w))
    return out


rows = []
for s in sorted(qc):
    region = s.split("__")[0]
    ctrl = stratified(s, CONTROL)
    if ctrl is None:
        continue
    for t in order:
        cur = stratified(s, t)
        if cur is None:
            continue
        for a in cur:
            if a in ctrl:
                rows.append(dict(section=s, region=region, trait=t, compartment=a,
                                 trait_eff=cur[a], ctrl_eff=ctrl[a],
                                 strat_effect=cur[a] - ctrl[a]))

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no stratified estimates produced")
df.to_csv(f"{OUT}/depth_stratified.tsv", sep="\t", index=False)
report = {}

for region in ["SAN", "AVN"]:
    sub = df[df.region == region]
    if sub.empty:
        continue
    comps = sorted(sub.compartment.unique())
    ts = [t for t in order if t in set(sub.trait)]
    print("\n" + "=" * 96)
    print(f"{region}  —  depth-stratified compartment effect, trait minus control")
    print("=" * 96)
    print(f"{'compartment':<26}" + "".join(f"{t[:17]:>19}" for t in ts))
    for a in comps:
        line = f"{a:<26}"
        for t in ts:
            v = sub[(sub.compartment == a) & (sub.trait == t)].strat_effect.to_numpy()
            if len(v) == 0:
                line += f"{'-':>19}"
                continue
            star = ""
            if len(v) >= 6:
                try:
                    p = wilcoxon(v).pvalue
                    star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                except ValueError:
                    pass
            line += f"{v.mean():>+14.3f}{star:<5}"
        mark = ""
        if a == "node":
            mark = "  <<< pacemaker region"
        elif a == "myocardium_atrial":
            mark = "  <<< AF positive control"
        elif a == "AV_bundle":
            mark = "  <<< AV conduction"
        print(line + mark)
    print(f"\n(mean over {sub.section.nunique()} sections; "
          "* = Wilcoxon signed-rank vs 0)")

    # sign test per compartment: how many sections point the same way
    print(f"\n{'compartment':<26}" + "".join(f"{t[:17]:>19}" for t in ts))
    for a in comps:
        line = f"{a:<26}"
        for t in ts:
            v = sub[(sub.compartment == a) & (sub.trait == t)].strat_effect.to_numpy()
            if len(v) == 0:
                line += f"{'-':>19}"
                continue
            k = int((v > 0).sum())
            p = binomtest(k, len(v), 0.5).pvalue
            line += f"{f'{k}/{len(v)}':>10}{f'p={p:.3f}':>9}"
        print(line)
    print("(sections with a positive effect / total)")

    report[region] = {
        t: {a: dict(mean=float(sub[(sub.compartment == a) & (sub.trait == t)]
                               .strat_effect.mean()),
                    pos=int((sub[(sub.compartment == a) & (sub.trait == t)]
                             .strat_effect > 0).sum()),
                    n=int(len(sub[(sub.compartment == a) & (sub.trait == t)])))
            for a in comps}
        for t in ts}

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 96)
print("VERDICT")
print("=" * 96)
for label, region, comp, trait in [
        ("pacemaker hypothesis", "SAN", "node", "RestingHeartRate"),
        ("AF positive control ", "SAN", "myocardium_atrial", "AtrialFibrillation"),
        ("AV conduction       ", "AVN", "AV_bundle", "PRinterval")]:
    v = df[(df.region == region) & (df.compartment == comp)
           & (df.trait == trait)].strat_effect.to_numpy()
    if len(v) == 0:
        print(f"{label}: not evaluable")
        continue
    k = int((v > 0).sum())
    p = binomtest(k, len(v), 0.5).pvalue
    verdict = "SUPPORTED" if (v.mean() > 0 and p < 0.05) else "not supported"
    print(f"{label}: {region}/{comp} under {trait}")
    print(f"{'':<22} effect {v.mean():+.3f}   {k}/{len(v)} sections positive   "
          f"p = {p:.4f}   -> {verdict}")

with open(f"{OUT}/depth_stratified.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwrote {OUT}/depth_stratified.tsv and depth_stratified.json")
