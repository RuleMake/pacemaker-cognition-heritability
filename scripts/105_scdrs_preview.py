"""
Preview the scDRS ranking from the per-cell scores already on disk.
===================================================================

The formal group table is written only after every trait finishes. But the per-cell
normalised scores land as each trait completes, and scDRS's group association
statistic is built from exactly those: a group's mean normalised score compared
against the distribution of group means under the control gene sets. The control
distribution shifts a group's significance, not its position relative to other groups
within the same trait — so the ordering seen here is the ordering the formal test will
report, and it is available now rather than in three quarters of an hour.

What this is NOT: a p-value. Nothing here is a significance claim. The formal Monte
Carlo test, its FDR correction across 62 cell states, and the heterogeneity statistic
all come from 104_scdrs_report.py once the run completes.

Usage:  python scripts/105_scdrs_preview.py
"""

import glob
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.as_posix()
SC = f"{ROOT}/results/scdrs"
H5 = f"{ROOT}/data/singlecell/node_subset.h5ad"

CONDUCTION = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell"]
CTRL = "EducationalAttainment"

files = sorted(glob.glob(f"{SC}/*.score.tsv"))
if not files:
    raise SystemExit("no score files yet")

a = ad.read_h5ad(H5, backed="r")
cs = a.obs["cell_state"].astype(str)
sizes = cs.value_counts()
print(f"traits scored so far: {len(files)}\n")

means = {}
for p in files:
    trait = os.path.basename(p).replace(".score.tsv", "")
    d = pd.read_csv(p, sep="\t", index_col=0)
    col = "norm_score" if "norm_score" in d.columns else d.columns[0]
    s = d[col].reindex(cs.index)
    means[trait] = s.groupby(cs.to_numpy()).mean()

M = pd.DataFrame(means)
M.to_csv(f"{SC}/preview_group_means.tsv", sep="\t")
traits = list(M.columns)

print("=" * 100)
print("MEAN scDRS SCORE PER CONDUCTION CELL STATE, AND ITS RANK AMONG ALL 62")
print("=" * 100)
print(f"{'cell state':<20}{'n':>7}" + "".join(f"{t[:15]:>18}" for t in traits))
for c in CONDUCTION:
    if c not in M.index:
        continue
    line = f"{c:<20}{int(sizes.get(c, 0)):>7,}"
    for t in traits:
        r = int(M[t].rank(ascending=False)[c])
        line += f"{M.loc[c, t]:>+11.3f} {f'#{r}':>5}"
    print(line)

print("\n" + "=" * 100)
print("TOP 6 CELL STATES PER TRAIT")
print("=" * 100)
for t in traits:
    top = M[t].nlargest(6)
    items = ", ".join(f"{i}({v:+.2f})" for i, v in top.items())
    tag = "   <- non-cardiac control" if t == CTRL else ""
    print(f"\n{t:<24}{tag}\n   {items}")

if CTRL in traits:
    print("\n" + "=" * 100)
    print("THE QUESTION THAT MATTERS: cardiac trait rank vs control rank")
    print("=" * 100)
    ctrl_rank = M[CTRL].rank(ascending=False)
    for c in CONDUCTION:
        if c not in M.index:
            continue
        cr = int(ctrl_rank[c])
        better = []
        for t in traits:
            if t == CTRL:
                continue
            if int(M[t].rank(ascending=False)[c]) < cr:
                better.append(t)
        print(f"\n{c}  (control ranks it #{cr}/{len(M)})")
        print(f"   cardiac traits ranking it HIGHER than the control: "
              f"{', '.join(better) if better else 'none'}")
    print("\nA conduction cell state that the non-cardiac control also ranks high is")
    print("reflecting data structure, not biology — the pattern that sank the spatial")
    print("analysis, where node affinity tracked GWAS power at r = +0.95.")

print(f"\nwrote {SC}/preview_group_means.tsv")
