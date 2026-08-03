"""
The verdict: did equalising sequencing depth change what gsMap found?
=====================================================================

Reads the depth-matched re-run and compares it, section by section and compartment by
compartment, against the original. Four questions, in the order that decides things.

Q1 SANITY — did thinning break gsMap?
   Binomial thinning to 2,151 UMI cuts the median genes detected roughly in half. If
   the thinned run produces nothing significant anywhere for any trait, the experiment
   is uninformative rather than negative, and the target depth has to be raised. This
   is checked FIRST so a broken run is never read as a biological result.

Q2 CAUSALITY — did the depth association actually go away?
   The thinned spots all carry the same depth, so correlating against their own depth
   is meaningless. The right comparison is against each spot's ORIGINAL depth: if
   depth was causing the map, the new p-values should no longer track the old depth.

Q3 POSITIVE CONTROL — does atrial fibrillation still find atrial myocardium?
   This held at 8/8 sections throughout, including under depth stratification. If it
   survives thinning too, the correction preserves real signal. If it dies, thinning
   removed everything and Q4 cannot be interpreted.

Q4 THE HYPOTHESIS — does the node compartment behave differently for resting heart
   rate once depth cannot help it?

Usage:  python scripts/64_depthmatched_verdict.py
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
OLD_SPOT = f"{ROOT}/results/spotlevel"
NEW_SPOT = f"{ROOT}/results/spotlevel_depthmatched"
OUT = f"{ROOT}/results"
CONTROL = "EducationalAttainment"
TRAITS = ["RestingHeartRate", "AtrialFibrillation", "PRinterval", CONTROL]

if not os.path.isdir(NEW_SPOT) or not glob.glob(f"{NEW_SPOT}/*.csv"):
    raise SystemExit(f"depth-matched spot tables not found in {NEW_SPOT} — "
                     "the re-run has not finished yet")


def harmonise(ann, region):
    if region == "SAN":
        return np.where(ann == "myocardium", "myocardium_atrial", ann)
    return ann


# original per-spot depth and compartment
qc = {}
for p in sorted(glob.glob(f"{GS}/SAN__*.h5ad")):
    s = os.path.basename(p)[:-5]
    a = ad.read_h5ad(p)
    X = a.layers["count"] if "count" in a.layers else a.X
    qc[s] = pd.DataFrame(
        {"umi": np.asarray(X.sum(axis=1)).ravel(),
         "ann": harmonise(a.obs["annotation_final"].astype(str).to_numpy(), "SAN")},
        index=a.obs_names)


def spot(dirpath, s, t):
    p = f"{dirpath}/{s}__{t}.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, index_col=0)
    if "logp" not in d.columns and "p" in d.columns:
        d["logp"] = -np.log10(d["p"].clip(lower=1e-300))
    return d


sections = sorted(qc)
have = [s for s in sections if spot(NEW_SPOT, s, "RestingHeartRate") is not None]
print(f"sections with depth-matched results: {len(have)}/{len(sections)}\n")
report = {}

# ================================================================ Q1
print("=" * 92)
print("Q1. SANITY — is there any signal left after thinning?")
print("=" * 92)
print(f"{'trait':<24}{'median logp':>14}{'max logp':>11}{'spots p<0.05':>15}"
      f"{'(original median)':>20}")
q1 = {}
for t in TRAITS:
    new, old = [], []
    for s in have:
        n, o = spot(NEW_SPOT, s, t), spot(OLD_SPOT, s, t)
        if n is not None:
            new.append(n["logp"].to_numpy())
        if o is not None:
            old.append(o["logp"].to_numpy())
    if not new:
        continue
    nv = np.concatenate(new)
    ov = np.concatenate(old) if old else np.array([np.nan])
    frac = float((nv > -np.log10(0.05)).mean())
    q1[t] = dict(median=float(np.median(nv)), max=float(nv.max()), frac_sig=frac,
                 old_median=float(np.median(ov)))
    print(f"{t:<24}{np.median(nv):>14.3f}{nv.max():>11.2f}{frac * 100:>14.1f}%"
          f"{np.median(ov):>20.3f}")
report["Q1"] = q1
alive = any(v["frac_sig"] > 0.02 for k, v in q1.items() if k != CONTROL)
print(f"\n=> thinned run {'RETAINS' if alive else 'HAS LOST'} signal"
      f"{'' if alive else '  — raise the depth target and redo; do not read Q3/Q4'}")

# ================================================================ Q2
print("\n" + "=" * 92)
print("Q2. CAUSALITY — do the new p-values still track the ORIGINAL depth?")
print("=" * 92)
rows = []
for t in TRAITS:
    for s in have:
        n, o = spot(NEW_SPOT, s, t), spot(OLD_SPOT, s, t)
        if n is None:
            continue
        c = n.index.intersection(qc[s].index)
        if len(c) < 200:
            continue
        u = qc[s].loc[c, "umi"].to_numpy()
        r = dict(section=s, trait=t,
                 new=float(spearmanr(n.loc[c, "logp"], u)[0]))
        if o is not None:
            co = o.index.intersection(qc[s].index)
            r["old"] = float(spearmanr(o.loc[co, "logp"],
                                       qc[s].loc[co, "umi"])[0])
        rows.append(r)
d2 = pd.DataFrame(rows)
d2.to_csv(f"{OUT}/depthmatched_q2.tsv", sep="\t", index=False)
print(f"{'trait':<24}{'original run':>14}{'thinned run':>14}{'reduction':>12}{'p':>10}")
for t in TRAITS:
    g = d2[d2.trait == t].dropna(subset=["old"])
    if g.empty:
        continue
    pv = wilcoxon(g.old.abs() - g.new.abs()).pvalue if len(g) >= 6 else np.nan
    print(f"{t:<24}{g.old.mean():>+14.3f}{g.new.mean():>+14.3f}"
          f"{(1 - abs(g.new.mean()) / max(abs(g.old.mean()), 1e-9)) * 100:>11.0f}%"
          f"{(f'{pv:.4f}' if np.isfinite(pv) else '-'):>10}")
print("\nA large drop means the depth association was CAUSED by depth, not merely")
print("correlated with it — the intervention removed it. Compare with the 6% that the")
print("paired trait contrast achieved.")
report["Q2"] = {t: dict(old=float(d2[d2.trait == t].old.mean()),
                        new=float(d2[d2.trait == t].new.mean()))
                for t in TRAITS if len(d2[d2.trait == t].dropna(subset=["old"]))}

# ================================================================ Q3 / Q4
print("\n" + "=" * 92)
print("Q3/Q4. COMPARTMENT RESULTS — trait minus control, on identical spots")
print("=" * 92)
rows = []
for t in [x for x in TRAITS if x != CONTROL]:
    for s in have:
        a, b = spot(NEW_SPOT, s, t), spot(NEW_SPOT, s, CONTROL)
        if a is None or b is None:
            continue
        c = a.index.intersection(b.index).intersection(qc[s].index)
        if len(c) < 200:
            continue
        d = a.loc[c, "logp"].to_numpy() - b.loc[c, "logp"].to_numpy()
        ann = qc[s].loc[c, "ann"].to_numpy()
        for comp in np.unique(ann):
            m = ann == comp
            if m.sum() < 30:
                continue
            rows.append(dict(section=s, trait=t, compartment=comp, n=int(m.sum()),
                             effect=float(d[m].mean() - d[~m].mean())))
d3 = pd.DataFrame(rows)
if d3.empty:
    raise SystemExit("no compartment estimates produced")
d3.to_csv(f"{OUT}/depthmatched_compartment.tsv", sep="\t", index=False)

ts = [t for t in TRAITS if t != CONTROL and t in set(d3.trait)]
comps = sorted(d3.compartment.unique())
print(f"{'compartment':<26}" + "".join(f"{t[:18]:>22}" for t in ts))
for comp in comps:
    line = f"{comp:<26}"
    for t in ts:
        v = d3[(d3.compartment == comp) & (d3.trait == t)].effect.to_numpy()
        if len(v) == 0:
            line += f"{'-':>22}"
            continue
        k = int((v > 0).sum())
        p = binomtest(k, len(v), 0.5).pvalue
        star = "*" if p < 0.05 else ""
        line += f"{v.mean():>+11.3f} {f'{k}/{len(v)}':>6}{star:<4}"
    mark = ("  <<< pacemaker" if comp == "node" else
            "  <<< AF control" if comp == "myocardium_atrial" else "")
    print(line + mark)
print("\n(mean effect over sections, sections-positive/total, * = sign test p<0.05)")

print("\n" + "=" * 92)
print("VERDICT")
print("=" * 92)
verdict = {}
for label, comp, trait in [
        ("POSITIVE CONTROL  AF -> atrial myocardium", "myocardium_atrial",
         "AtrialFibrillation"),
        ("HYPOTHESIS        RHR -> node            ", "node", "RestingHeartRate")]:
    v = d3[(d3.compartment == comp) & (d3.trait == trait)].effect.to_numpy()
    if len(v) == 0:
        print(f"{label}: not evaluable")
        continue
    k = int((v > 0).sum())
    p = binomtest(k, len(v), 0.5).pvalue
    res = "SURVIVES" if (v.mean() > 0 and p < 0.05) else "does not hold"
    verdict[trait] = dict(compartment=comp, effect=float(v.mean()), pos=k,
                          n=len(v), p=float(p), result=res)
    print(f"{label}: {v.mean():+.3f}   {k}/{len(v)} sections   p={p:.4f}   -> {res}")

ok_ctrl = verdict.get("AtrialFibrillation", {}).get("result") == "SURVIVES"
print()
if not alive:
    print("READ AS: uninformative — thinning removed the signal itself. Redo at a")
    print("         higher depth target (p30 or p40) before drawing any conclusion.")
elif ok_ctrl:
    print("READ AS: the correction preserves real signal (positive control survives),")
    print("         so the node result below it can be taken at face value.")
else:
    print("READ AS: the positive control did NOT survive, so this run cannot")
    print("         adjudicate the node hypothesis either way.")
report["verdict"] = verdict
report["signal_alive"] = bool(alive)

with open(f"{OUT}/depthmatched_verdict.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwrote {OUT}/depthmatched_*.tsv and depthmatched_verdict.json")
