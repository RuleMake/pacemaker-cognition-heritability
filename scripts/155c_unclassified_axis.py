"""
What is `unclassified` on the axis subset, and must it be excluded again?
========================================================================

It ranks first for every trait scored so far, at z = +18.5 for atrial fibrillation
against +9.8 for the next state — twice the runner-up. A cell type that wins every
trait by that margin is not a biological finding; on the node subset the same class
turned out to be a technical catch-all and was excluded from the reported rankings.

The question here is only whether the same call applies to this subset. If it does, the
rank columns must exclude it, and the reason has to be on the record rather than
assumed from last time.

The interaction test in 153 is unaffected either way: it uses AUC of the focus cells
against everything else in the stratum, where one extra class among seventy-five
changes nothing material. Ranks are what this protects.

Usage:  python scripts/155c_unclassified_axis.py
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.as_posix()
a = ad.read_h5ad(f"{ROOT}/data/singlecell/axis_subset.h5ad", backed="r")
o = a.obs
cs = o.cell_state.astype(str)
m = (cs == "unclassified").to_numpy()
print(f"unclassified: {m.sum():,} cells of {len(cs):,}\n")

for col in ["region", "assay", "donor_id", "cell_type"]:
    if col not in o.columns:
        continue
    v = o.loc[m, col].astype(str).value_counts(normalize=True)
    bg = o[col].astype(str).value_counts(normalize=True)
    print(f"{col}:")
    for k in v.index[:6]:
        print(f"   {k:<26}{v[k] * 100:>6.1f}%   (background {bg.get(k, 0) * 100:.1f}%)")
    print()

# Quality metrics decide it: a catch-all class of low-quality or ambiguous cells looks
# different from a real cell type on depth and gene count, and that is checkable.
print(f"{'metric':<24}{'unclassified':>16}{'everything else':>18}")
for col in ["n_genes_qc", "total_counts_qc", "pct_counts_mt", "scrublet_score"]:
    if col not in o.columns:
        continue
    v = pd.to_numeric(o[col], errors="coerce").to_numpy(dtype=float)
    print(f"{col:<24}{np.nanmedian(v[m]):>16,.1f}{np.nanmedian(v[~m]):>18,.1f}")

print("\nA class that is spread over every region and donor, and differs from the rest")
print("on depth or gene count, is a technical residue. Exclude it from rank tables and")
print("say so; do not let it occupy the top of a figure.")
