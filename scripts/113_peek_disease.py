"""
Read the cached per-trait group analyses, including the disease traits.
=======================================================================

The final tables are regenerated once every trait is scored, but the disease-trait
results are already on disk as per-trait caches and there is no reason to wait to look
at them. Each is shown next to its GWAS power, because atrioventricular block in
particular has 436 genome-wide significant SNPs and lambda_GC = 1.010 — a null from it
means nothing at all, and the table must not let that be forgotten.

Usage:  python scripts/113_peek_disease.py
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

FOCUS = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell"]
FLOOR = 920

files = sorted(glob.glob(f"{SC}/*.group.tsv"))
if not files:
    raise SystemExit("no cached group analyses yet")

power = {}
for p in sorted(glob.glob(f"{GWAS}/*.sumstats.gz")):
    t = os.path.basename(p).replace(".sumstats.gz", "")
    z = pd.read_csv(p, sep="\t", usecols=["Z"]).Z.to_numpy()
    z = z[np.isfinite(z)]
    power[t] = dict(n_gws=int((np.abs(z) > 5.45).sum()),
                    lam=float(np.median(z ** 2) / 0.4549))

groups = {}
for p in files:
    t = os.path.basename(p).replace(".group.tsv", "")
    g = pd.read_csv(p, sep="\t", index_col=0)
    groups[t] = g

print(f"traits with cached results: {len(groups)}\n")
print("=" * 104)
print("CONDUCTION CELL RANKS, WITH GWAS POWER ALONGSIDE")
print("=" * 104)
print(f"{'trait':<26}{'gw-sig':>8}{'lambda':>8}  " +
      "".join(f"{c[:15]:>17}" for c in FOCUS) + "   power")
order = sorted(groups, key=lambda t: -power.get(t, {}).get("n_gws", 0))
out = {}
for t in order:
    g = groups[t].sort_values("assoc_mcz", ascending=False)
    lst = list(g.index)
    line = f"{t:<26}"
    pw = power.get(t, {})
    line += f"{pw.get('n_gws', 0):>8,}{pw.get('lam', float('nan')):>8.3f}  "
    ranks = {}
    for c in FOCUS:
        if c not in lst:
            line += f"{'-':>17}"
            continue
        r = lst.index(c) + 1
        z = float(g.loc[c, "assoc_mcz"])
        ranks[c] = r
        line += f"{f'#{r} (z{z:+.1f})':>17}"
    tag = ("adequate" if pw.get("n_gws", 0) >= FLOOR
           else "BELOW FLOOR — a null here means nothing")
    print(line + f"   {tag}")
    out[t] = dict(ranks=ranks, top5=lst[:5], **pw)

print("\n" + "=" * 104)
print("TOP 5 CELL STATES PER TRAIT")
print("=" * 104)
for t in order:
    g = groups[t].sort_values("assoc_mcz", ascending=False)
    items = ", ".join(f"{i}({g.loc[i, 'assoc_mcz']:+.1f})" for i in g.index[:5])
    print(f"\n{t:<26}{items}")

with open(f"{SC}/disease_peek.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nwrote {SC}/disease_peek.json")
