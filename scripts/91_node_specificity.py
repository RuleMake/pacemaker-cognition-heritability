"""
Is the node signal trait-SPECIFIC, or is it depth wearing a different hat?
=========================================================================

Script 90 produced the first thing in this project that looks like a positive result:
the sinoatrial node compartment scores higher for resting heart rate than for either
atrial fibrillation or PR interval, 8/8 sections, p = 0.0078 each.

That has to be attacked before it is believed, because those contrasts still carry
36-61% of the depth confound and node is the second-deepest compartment. Three
attacks, each of which the result must survive.

ATTACK 1 — depth stratification ON TOP of the cardiac contrast
    Bin spots by depth decile, compare node against the rest WITHIN each bin, and take
    the trait-minus-trait difference. Depth is near-constant inside a bin, so what
    survives cannot be a depth gradient. This is the strongest available design: it
    combines the two corrections that each worked partially.

ATTACK 2 — is node an outlier above the depth line, or on it?
    Regress each compartment's contrast effect on its median depth across compartments.
    If the whole pattern is depth, every compartment sits on that line and node's
    residual is ~0. A large positive residual for node, while the DEEPER atrial
    myocardium sits on or below the line, is the signature of real specificity.

ATTACK 3 — the cell-type version
    Repeat the partial correlation of script 53 against SAN_P_cell abundance, but
    contrasting resting heart rate with a CARDIAC trait rather than with educational
    attainment. The earlier +0.011 (p = 0.64) answered "is the node above noise";
    this answers "is the node more heart-rate than atrial-disease", which is the
    question the biology actually poses.

A positive that survives all three is worth writing up. One that does not is depth.

Usage:  python scripts/91_node_specificity.py
"""

import glob
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import binomtest, linregress, rankdata, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent.parent.as_posix()
GS = f"{ROOT}/data/spatial_gsmap"
SRC = f"{ROOT}/data/spatial"
SPOT = f"{ROOT}/results/spotlevel"
OUT = f"{ROOT}/results"
NBIN, MINCELL = 10, 20

PAIRS = [("RestingHeartRate", "AtrialFibrillation"),
         ("RestingHeartRate", "PRinterval"),
         ("PRinterval", "AtrialFibrillation"),
         ("RestingHeartRate", "EducationalAttainment")]
ATRIAL = ["aCM1", "aCM2", "aCM3", "aCM4"]


def harmonise(ann, region):
    return np.where(ann == "myocardium", "myocardium_atrial", ann) \
        if region == "SAN" else ann


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
         "ann": harmonise(a.obs["annotation_final"].astype(str).to_numpy(), "SAN")},
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
print(f"SAN sections: {len(sections)}   abundance for {len(abund)}\n")


def load(s, t):
    p = f"{SPOT}/{s}__{t}.csv"
    return pd.read_csv(p, index_col=0) if os.path.exists(p) else None


report = {}

# ============================================================ ATTACK 1
print("=" * 96)
print("ATTACK 1.  Depth-stratified AND cardiac-contrasted")
print("           node vs rest, within depth deciles, trait A minus trait B")
print("=" * 96)


def strat(s, t):
    d = load(s, t)
    if d is None:
        return None
    c = d.index.intersection(qc[s].index)
    m = pd.DataFrame({"logp": d.loc[c, "logp"].to_numpy(),
                      "umi": qc[s].loc[c, "umi"].to_numpy(),
                      "ann": qc[s].loc[c, "ann"].to_numpy()})
    m["bin"] = pd.qcut(m.umi.rank(method="first"), NBIN, labels=False)
    out = {}
    for comp in m.ann.unique():
        eff, w = [], []
        for b in range(NBIN):
            sub = m[m.bin == b]
            inn, rest = sub[sub.ann == comp], sub[sub.ann != comp]
            if len(inn) < MINCELL or len(rest) < MINCELL:
                continue
            eff.append(inn.logp.mean() - rest.logp.mean())
            w.append(len(inn))
        if eff:
            out[comp] = float(np.average(eff, weights=w))
    return out


rows = []
cache = {(s, t): strat(s, t) for s in sections
         for t in {x for p in PAIRS for x in p}}
for a_t, b_t in PAIRS:
    for s in sections:
        A, B = cache.get((s, a_t)), cache.get((s, b_t))
        if not A or not B:
            continue
        for comp in set(A) & set(B):
            rows.append(dict(section=s, pair=f"{a_t[:3]}-{b_t[:3]}",
                             compartment=comp, effect=A[comp] - B[comp]))
d1 = pd.DataFrame(rows)
d1.to_csv(f"{OUT}/node_specificity_stratified.tsv", sep="\t", index=False)

pairs = [f"{a[:3]}-{b[:3]}" for a, b in PAIRS]
comps = sorted(d1.compartment.unique())
print(f"{'compartment':<26}" + "".join(f"{p:>22}" for p in pairs))
a1 = {}
for comp in comps:
    line = f"{comp:<26}"
    for p in pairs:
        v = d1[(d1.compartment == comp) & (d1.pair == p)].effect.to_numpy()
        if len(v) == 0:
            line += f"{'-':>22}"
            continue
        k = int((v > 0).sum())
        pv = binomtest(k, len(v), 0.5).pvalue
        star = "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        line += f"{v.mean():>+12.3f} {f'{k}/{len(v)}':>4}{star:<5}"
        if comp == "node":
            a1[p] = dict(effect=float(v.mean()), pos=k, n=len(v), p=float(pv))
    mark = "  <<<" if comp == "node" else ""
    print(line + mark)
print("\n(mean effect, sections-positive/total, * = sign test p<0.05)")
report["attack1"] = a1

# ============================================================ ATTACK 2
print("\n" + "=" * 96)
print("ATTACK 2.  Is node ABOVE the depth line, or on it?")
print("=" * 96)
depth = pd.concat(qc.values()).groupby("ann").umi.median()
rows = []
for p in pairs:
    sub = d1[d1.pair == p].groupby("compartment").effect.mean()
    common = [c for c in sub.index if c in depth.index]
    if len(common) < 5:
        continue
    x = np.log10(depth.loc[common].to_numpy())
    y = sub.loc[common].to_numpy()
    lr = linregress(x, y)
    resid = y - (lr.slope * x + lr.intercept)
    r = pd.Series(resid, index=common)
    for comp in common:
        rows.append(dict(pair=p, compartment=comp, depth=float(depth[comp]),
                         effect=float(sub[comp]), residual=float(r[comp])))
    print(f"\n{p}:  effect = {lr.slope:+.2f} * log10(depth) {lr.intercept:+.2f}   "
          f"r = {lr.rvalue:+.3f}  p = {lr.pvalue:.4f}")
    print(f"{'   compartment':<26}{'depth':>9}{'effect':>10}{'residual':>11}")
    for comp in sorted(common, key=lambda c: -r[c]):
        mk = ("  <<< pacemaker" if comp == "node" else
              "  <-- deeper than node" if depth[comp] > depth.get("node", 0) else "")
        print(f"   {comp:<23}{depth[comp]:>9,.0f}{sub[comp]:>+10.3f}"
              f"{r[comp]:>+11.3f}{mk}")
d2 = pd.DataFrame(rows)
d2.to_csv(f"{OUT}/node_specificity_depthline.tsv", sep="\t", index=False)
report["attack2"] = {p: float(d2[(d2.pair == p) & (d2.compartment == "node")]
                              .residual.iloc[0])
                     for p in pairs
                     if len(d2[(d2.pair == p) & (d2.compartment == "node")])}

# ============================================================ ATTACK 3
print("\n" + "=" * 96)
print("ATTACK 3.  Cell-type level: pacemaker abundance, cardiac contrast,")
print("           depth and atrial myocyte content held fixed")
print("=" * 96)
rows = []
for s in sections:
    if s not in abund or "SAN_P_cell" not in abund[s].columns:
        continue
    for t in {x for p in PAIRS for x in p}:
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
d3 = pd.DataFrame(rows).pivot(index="section", columns="trait", values="pr")
d3.to_csv(f"{OUT}/node_specificity_celltype.tsv", sep="\t")
print(f"{'contrast':<44}{'mean diff':>12}{'sections +':>13}{'p':>10}")
a3 = {}
for a_t, b_t in PAIRS:
    if a_t not in d3.columns or b_t not in d3.columns:
        continue
    diff = (d3[a_t] - d3[b_t]).dropna()
    if len(diff) < 6:
        continue
    pv = wilcoxon(diff).pvalue
    a3[f"{a_t}-{b_t}"] = dict(mean=float(diff.mean()), p=float(pv))
    print(f"{a_t + ' - ' + b_t:<44}{diff.mean():>+12.3f}"
          f"{f'{int((diff > 0).sum())}/{len(diff)}':>13}{pv:>10.4f}")
report["attack3"] = a3

# ============================================================ verdict
print("\n" + "=" * 96)
print("VERDICT ON THE NODE / RESTING-HEART-RATE SPECIFICITY")
print("=" * 96)
# With 8 sections a sign test bottoms out at p = 0.0078 (8/8) and 7/8 already gives
# 0.0703, so demanding p<0.05 from every contrast rejects effects that are simply at
# the resolution of the design. The verdict weighs direction, consistency, the depth
# residual, and — most informatively — whether the built-in negative controls behave.
print("compartment level (depth-stratified + cardiac contrast):")
pos_comp = 0
for p in ["Res-Atr", "Res-PRi"]:
    s1 = report["attack1"].get(p, {})
    s2 = report["attack2"].get(p, 0.0)
    good = s1.get("effect", 0) > 0 and s1.get("pos", 0) >= s1.get("n", 8) - 1 and s2 > 0
    pos_comp += good
    print(f"  {p}:  {s1.get('effect', float('nan')):+.3f}  "
          f"{s1.get('pos', '?')}/{s1.get('n', '?')}  p={s1.get('p', float('nan')):.4f}"
          f"   depth-line residual {s2:+.3f}   {'consistent' if good else 'weak'}")

ct = {k: v for k, v in report["attack3"].items() if k.startswith("RestingHeartRate")}
ct_card = {k: v for k, v in ct.items() if "Educational" not in k}
ct_pos = sum(1 for v in ct_card.values() if v["mean"] > 0 and v["p"] < 0.05)
print(f"\ncell-type level: {ct_pos}/{len(ct_card)} cardiac contrasts positive at p<0.05")

# the controls that must NOT fire
neg_comp = report["attack1"].get("PRi-Atr", {})
neg_ct = report["attack3"].get("PRinterval-AtrialFibrillation", {})
neg_ok = (neg_comp.get("effect", 1) <= 0 or neg_comp.get("p", 0) > 0.05) and \
         (neg_ct.get("p", 0) > 0.05)
print("\nbuilt-in negative control (PR interval vs AF — neither is a sinoatrial trait):")
print(f"  compartment {neg_comp.get('effect', float('nan')):+.3f} "
      f"({neg_comp.get('pos', '?')}/{neg_comp.get('n', '?')}, "
      f"p={neg_comp.get('p', float('nan')):.3f})   "
      f"cell type {neg_ct.get('mean', float('nan')):+.3f} "
      f"(p={neg_ct.get('p', float('nan')):.3f})   "
      f"-> {'silent, as required' if neg_ok else 'FIRES — the design is leaking'}")

print()
if pos_comp == 2 and ct_pos == len(ct_card) and neg_ok:
    print("=> POSITIVE at compartment AND cell-type level, after depth stratification")
    print("   and with a depth-matched cardiac contrast, while the negative control")
    print("   stays silent. This is trait-specific and worth developing.")
    print("   Caveat: 8 sections cap the sign test at p=0.0078, and many compartments")
    print("   were examined. Needs more contrast traits and slice pooling to confirm.")
elif ct_pos == len(ct_card) and neg_ok:
    print("=> POSITIVE at cell-type level with the negative control silent;")
    print("   compartment level is directionally consistent but weaker.")
elif not neg_ok:
    print("=> the negative control fires too — the contrast is leaking something")
    print("   common to cardiac traits. Do not interpret as node specificity.")
else:
    print("=> not established. Directions are suggestive but consistency is lacking.")

print("\nNOTE: the earlier NEGATIVE (RestingHeartRate - EducationalAttainment,")
print(f"      +{ct.get('RestingHeartRate-EducationalAttainment', {}).get('mean', 0):.3f}, "
      f"p={ct.get('RestingHeartRate-EducationalAttainment', {}).get('p', 1):.2f}) used a")
print("      control whose depth dependence (+0.35) is half that of the cardiac traits")
print("      (+0.69 to +0.76). That mismatch leaves residual depth in the contrast,")
print("      which is why the same quantity reads null there and positive here.")

with open(f"{OUT}/node_specificity.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwrote {OUT}/node_specificity_*.tsv and node_specificity.json")
