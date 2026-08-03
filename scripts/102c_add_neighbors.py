"""
Add the kNN graph scDRS's group analysis requires — without densifying anything.
================================================================================

`scdrs.method.downstream_group_analysis` asserts `connectivities` in `adata.obsp`. It
needs the neighbour graph for its heterogeneity statistic — whether a cell group's
disease scores are uniformly elevated or driven by a subset — and refuses to run
without one, even though the association statistic itself does not use it.

Script 102 cleared `obsp` to keep the subset small and never rebuilt it, so the first
scDRS run died on the first trait. Worse, the orchestrator read that crash as "the
positive control failed" and skipped everything downstream: a broken pipeline and a
genuinely negative control look identical from outside unless the difference is made
explicit, and they call for opposite responses.

The obvious route — highly variable genes, scale, PCA — was tried and was killed by
the kernel. `sc.pp.scale` zero-centres, which densifies: 118,172 x 2,000 in float32 is
almost a gigabyte before any copies, on top of a 12 GB WSL allocation already shared
with other jobs. Uncentred PCA (truncated SVD) gives a perfectly good embedding for a
neighbour graph and never leaves the sparse representation.

Usage:  python scripts/102c_add_neighbors.py
        python scripts/102c_add_neighbors.py --h5 data/singlecell/axis_subset.h5ad
"""

import argparse
import gc
import os
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parent.parent.as_posix()

ap = argparse.ArgumentParser()
ap.add_argument("--h5", default=f"{ROOT}/data/singlecell/node_subset.h5ad")
_args = ap.parse_args()
H5 = _args.h5 if os.path.isabs(_args.h5) else f"{ROOT}/{_args.h5}"


def mem():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        return float("nan")


a = ad.read_h5ad(H5)
print(f"cells {a.n_obs:,}  genes {a.n_vars:,}   peak RSS {mem():.1f} GB")
if "connectivities" in a.obsp:
    print("connectivities already present — nothing to do")
    raise SystemExit(0)

n_vars_before = a.n_vars

print("highly variable genes ...", flush=True)
sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat")
print(f"  {int(a.var.highly_variable.sum()):,} selected   peak RSS {mem():.1f} GB")

print("uncentred PCA on the variable genes (stays sparse) ...", flush=True)
tmp = a[:, a.var.highly_variable.to_numpy()].copy()
sc.pp.pca(tmp, n_comps=50, zero_center=False, svd_solver="arpack")
a.obsm["X_pca"] = np.asarray(tmp.obsm["X_pca"], dtype=np.float32)
vr = tmp.uns.get("pca", {}).get("variance_ratio")
if vr is not None:
    print(f"  50 components, {float(np.sum(vr)) * 100:.1f}% of variance")
del tmp
gc.collect()
print(f"  peak RSS {mem():.1f} GB")

print("neighbour graph ...", flush=True)
sc.pp.neighbors(a, n_neighbors=15, n_pcs=50)
print(f"  connectivities {a.obsp['connectivities'].shape}, "
      f"{a.obsp['connectivities'].nnz:,} edges   peak RSS {mem():.1f} GB")

# scDRS scores against X; the expression matrix must come through untouched
assert a.n_vars == n_vars_before, "gene set was altered"
a.write_h5ad(H5, compression="gzip")
print(f"\nwrote {H5}  ({os.path.getsize(H5) / 1e9:.2f} GB)")

cs = a.obs["cell_state"].astype(str)
print("\nsanity — conduction states still present:")
for c in ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]:
    print(f"  {c:<20}{int((cs == c).sum()):>7,}")
