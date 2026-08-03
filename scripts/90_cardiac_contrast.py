"""
Contrast cardiac traits against EACH OTHER, not against a non-cardiac control.
=============================================================================

Why the earlier contrast failed, restated precisely
---------------------------------------------------
Subtracting educational attainment removed only 6% of the depth confound. The reason
was not that contrasts do not work — it was that the two traits carried DIFFERENT
amounts of the artefact: +0.76 for resting heart rate against +0.35 for the control.
Subtracting a weak artefact from a strong one leaves a strong one.

But the cardiac traits carry it in nearly equal measure: +0.761, +0.755, +0.690. If
the artefact magnitude is what governs cancellation, then a cardiac-versus-cardiac
contrast should cancel it almost completely — and the contrast that was reached for
first was simply the wrong one.

That is a sharp prediction, and it is testable on data already computed.

What it buys, if true
---------------------
A contrast between two cardiac traits asks a genuinely biological question with the
confound built out rather than argued away:

    RHR - AF   what is specific to heart rate beyond atrial disease
    PR  - AF   what is specific to atrioventricular conduction beyond atrial disease
    RHR - PR   what separates sinoatrial from atrioventricular conduction

Each of these is a real question about conduction-system biology, and each has an
internal control that the trait-versus-noise design never had. Note the direction is
signed: a compartment can be positive for one member of the pair and negative for the
other, which is more information than "is this compartment enriched".

Usage:  python scripts/90_cardiac_contrast.py
"""

import glob
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent.parent.as_posix()
GS = f"{ROOT}/data/spatial_gsmap"
SPOT = f"{ROOT}/results/spotlevel"
OUT = f"{ROOT}/results"
CONTROL = "EducationalAttainment"

CARDIAC = ["RestingHeartRate", "AtrialFibrillation", "PRinterval"]
PAIRS = [("RestingHeartRate", "AtrialFibrillation"),
         ("PRinterval", "AtrialFibrillation"),
         ("RestingHeartRate", "PRinterval")]


def harmonise(ann, region):
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


def load(s, t):
    p = f"{SPOT}/{s}__{t}.csv"
    return pd.read_csv(p, index_col=0) if os.path.exists(p) else None


sections = sorted(qc)
report = {}

# ================================================================ 1
print("=" * 94)
print("1. DOES A CARDIAC-VS-CARDIAC CONTRAST CANCEL THE DEPTH CONFOUND?")
print("=" * 94)
rows = []
for a_t, b_t in PAIRS + [(t, CONTROL) for t in CARDIAC]:
    for s in sections:
        A, B = load(s, a_t), load(s, b_t)
        if A is None or B is None:
            continue
        c = A.index.intersection(B.index).intersection(qc[s].index)
        if len(c) < 200:
            continue
        d = A.loc[c, "logp"].to_numpy() - B.loc[c, "logp"].to_numpy()
        u = qc[s].loc[c, "umi"].to_numpy()
        rows.append(dict(section=s, region=s.split("__")[0], pair=f"{a_t} - {b_t}",
                         kind="cardiac-cardiac" if b_t != CONTROL else "cardiac-control",
                         rho=float(spearmanr(d, u)[0]),
                         raw_a=float(spearmanr(A.loc[c, "logp"], u)[0])))
d1 = pd.DataFrame(rows)
d1.to_csv(f"{OUT}/cardiac_contrast_depth.tsv", sep="\t", index=False)

print(f"{'contrast':<46}{'rho(delta, UMI)':>18}{'|rho|':>9}{'sections':>10}")
for kind in ["cardiac-cardiac", "cardiac-control"]:
    sub = d1[d1.kind == kind]
    print(f"\n  -- {kind}")
    for pair in sub.pair.unique():
        g = sub[sub.pair == pair]
        print(f"  {pair:<44}{g.rho.mean():>+18.3f}{abs(g.rho.mean()):>9.3f}{len(g):>10}")
cc = d1[d1.kind == "cardiac-cardiac"].rho.abs()
ck = d1[d1.kind == "cardiac-control"].rho.abs()
raw = d1.raw_a.abs().mean()
print(f"\nuncorrected single trait      |rho| = {raw:.3f}")
print(f"cardiac minus non-cardiac     |rho| = {ck.mean():.3f}   "
      f"({(1 - ck.mean() / raw) * 100:.0f}% removed)")
print(f"cardiac minus cardiac         |rho| = {cc.mean():.3f}   "
      f"({(1 - cc.mean() / raw) * 100:.0f}% removed)   <<<")
report["depth_removal"] = dict(raw=float(raw), vs_control=float(ck.mean()),
                               vs_cardiac=float(cc.mean()))
works = cc.mean() < 0.25
print(f"\n=> cardiac-vs-cardiac contrast {'WORKS' if works else 'does not work'}"
      f"{'  — proceed to the biology below' if works else ''}")

# ================================================================ 2
print("\n" + "=" * 94)
print("2. COMPARTMENT BIOLOGY UNDER CARDIAC-VS-CARDIAC CONTRASTS")
print("=" * 94)
rows = []
for a_t, b_t in PAIRS:
    for s in sections:
        A, B = load(s, a_t), load(s, b_t)
        if A is None or B is None:
            continue
        c = A.index.intersection(B.index).intersection(qc[s].index)
        if len(c) < 200:
            continue
        d = A.loc[c, "logp"].to_numpy() - B.loc[c, "logp"].to_numpy()
        ann = qc[s].loc[c, "ann"].to_numpy()
        for comp in np.unique(ann):
            m = ann == comp
            if m.sum() < 30:
                continue
            rows.append(dict(section=s, region=s.split("__")[0],
                             pair=f"{a_t[:3]}-{b_t[:3]}", compartment=comp,
                             n=int(m.sum()),
                             effect=float(d[m].mean() - d[~m].mean())))
d2 = pd.DataFrame(rows)
d2.to_csv(f"{OUT}/cardiac_contrast_compartment.tsv", sep="\t", index=False)

hits = []
for region in ["SAN", "AVN"]:
    sub = d2[d2.region == region]
    if sub.empty:
        continue
    pairs = list(dict.fromkeys(sub.pair))
    comps = sorted(sub.compartment.unique())
    print(f"\n--- {region}   ({sub.section.nunique()} sections)")
    print(f"{'compartment':<26}" + "".join(f"{p:>24}" for p in pairs))
    for comp in comps:
        line = f"{comp:<26}"
        for p in pairs:
            v = sub[(sub.compartment == comp) & (sub.pair == p)].effect.to_numpy()
            if len(v) == 0:
                line += f"{'-':>24}"
                continue
            k = int((v > 0).sum())
            pv = binomtest(k, len(v), 0.5).pvalue
            star = "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            line += f"{v.mean():>+13.3f} {f'{k}/{len(v)}':>5}{star:<5}"
            if pv < 0.05 and len(v) >= 6:
                hits.append(dict(region=region, pair=p, compartment=comp,
                                 effect=float(v.mean()), pos=k, n=len(v),
                                 p=float(pv)))
        mark = ("  <<< pacemaker" if comp == "node" else
                "  <<< AV conduction" if comp == "AV_bundle" else "")
        print(line + mark)
print("\n(mean effect, sections-positive/total, * = sign test p<0.05)")

# ================================================================ 3
print("\n" + "=" * 94)
print("3. SIGNIFICANT COMPARTMENT SPECIFICITIES")
print("=" * 94)
if not hits:
    print("none reached p<0.05 by the sign test")
else:
    for h in sorted(hits, key=lambda x: x["p"]):
        direction = "higher" if h["effect"] > 0 else "lower"
        a_t, b_t = h["pair"].split("-")
        print(f"{h['region']:<5}{h['compartment']:<24} {direction:>7} for {a_t} "
              f"than {b_t}:  {h['effect']:+.3f}  {h['pos']}/{h['n']}  p={h['p']:.4f}")
report["hits"] = hits

with open(f"{OUT}/cardiac_contrast.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwrote {OUT}/cardiac_contrast_*.tsv and cardiac_contrast.json")
