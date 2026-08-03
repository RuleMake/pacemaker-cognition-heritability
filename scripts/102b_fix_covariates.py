"""
Verify the expression scale and rebuild the scDRS covariates from the real QC columns.
=====================================================================================

Two problems with what 102 wrote:

1. The atlas stores TRANSFORMED values in X, not raw counts, so the branch that would
   have applied CPM+log1p was skipped — correctly, if X is already log-normalised,
   which is the CELLxGENE convention. That has to be confirmed rather than assumed,
   because scdrs.preprocess takes log-normalised input and would silently misbehave on
   CP10K without the log.

2. Total counts were recomputed from that transformed matrix, where every cell sums to
   roughly the same number. As a covariate it would carry no information. The original
   per-cell QC values are already in obs (n_genes, total_counts, pct_counts_mt) and are
   what should be regressed out.

Usage:  python scripts/102b_fix_covariates.py
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent.as_posix()
H5 = f"{ROOT}/data/singlecell/node_subset.h5ad"
COV = f"{ROOT}/data/scdrs/covariates.tsv"

a = ad.read_h5ad(H5)
X = a.X
X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()

d = X.data
rowsum = np.asarray(X.sum(axis=1)).ravel()
print(f"cells {a.n_obs:,}  genes {a.n_vars:,}")
print(f"nonzero values: min {d.min():.4f}  median {np.median(d):.4f}  "
      f"p99 {np.percentile(d, 99):.4f}  max {d.max():.4f}")
print(f"per-cell sums : median {np.median(rowsum):,.1f}  "
      f"IQR {np.percentile(rowsum, 25):,.1f}-{np.percentile(rowsum, 75):,.1f}")

integral = np.allclose(d[:5000], np.round(d[:5000]))
if integral:
    scale = "RAW COUNTS"
elif d.max() < 15 and np.median(rowsum) < 60000:
    scale = "log-normalised"
else:
    scale = "linear normalised (CP10K without log)"
print(f"\n=> X appears to be: {scale}")

if scale == "linear normalised (CP10K without log)":
    print("   applying log1p so scdrs.preprocess receives what it expects")
    X.data = np.log1p(X.data)
    a.X = X
    a.write_h5ad(H5, compression="gzip")
    print("   rewrote", H5)
elif scale == "RAW COUNTS":
    print("   applying CP10K + log1p")
    tot = np.asarray(X.sum(axis=1)).ravel()
    X = X.multiply((1e4 / np.maximum(tot, 1))[:, None]).tocsr()
    X.data = np.log1p(X.data)
    a.X = X
    a.write_h5ad(H5, compression="gzip")
    print("   rewrote", H5)
else:
    print("   no transformation needed")

# --- covariates from the ORIGINAL QC columns, not from the normalised matrix
print("\nrebuilding covariates from obs")
cols = {}
for want, cands in [("n_genes", ["n_genes", "n_genes_by_counts"]),
                    ("total_counts", ["total_counts"]),
                    ("pct_mt", ["pct_counts_mt"])]:
    for c in cands:
        if c in a.obs.columns:
            v = pd.to_numeric(a.obs[c], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(v).any() and np.nanstd(v) > 0:
                cols[want] = v
                print(f"  {want:<14} <- obs['{c}']   "
                      f"median {np.nanmedian(v):,.1f}  sd {np.nanstd(v):,.1f}")
                break
    else:
        print(f"  {want:<14} not available — omitted")

cov = pd.DataFrame({"const": np.ones(a.n_obs)}, index=a.obs_names)
if "n_genes" in cols:
    cov["n_genes"] = cols["n_genes"]
if "total_counts" in cols:
    cov["log_total"] = np.log10(np.maximum(cols["total_counts"], 1))
if "pct_mt" in cols:
    cov["pct_mt"] = cols["pct_mt"]
cov = cov.fillna(cov.median(numeric_only=True))
cov.index.name = "index"
cov.to_csv(COV, sep="\t")
print(f"\nwrote {COV}   columns: {list(cov.columns)}")

print("\nconduction cell states in this object:")
cs = a.obs["cell_state"].astype(str)
for c in ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]:
    n = int((cs == c).sum())
    print(f"  {c:<20}{n:>7,}" + ("   (n<50 — group test unreliable)"
                                 if 0 < n < 50 else
                                 "   (absent from node tissue)" if n == 0 else ""))
