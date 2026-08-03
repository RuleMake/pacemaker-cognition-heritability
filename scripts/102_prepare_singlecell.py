"""
Subset and preprocess the heart single-cell atlas for scDRS.
============================================================

Why this is the decisive experiment
-----------------------------------
Every confound that defeated the spatial analysis is a property of Visium spots or of
GWAS traits: spot depth, spot gene diversity, trait power, trait polygenicity. The
node result flipped sign depending on which of those was stratified on, and per-trait
"node affinity" turned out to correlate with mean chi-square at r = +0.95.

scDRS removes all four by construction:
  * cells, not 55 um spots — no mixing of pacemaker cells into atrial myocardium, and
    no deconvolution estimate standing between the data and the conclusion;
  * control gene sets matched to the disease set on mean expression AND expression
    variance — that is the complexity confound, handled by design;
  * a per-trait Monte Carlo null — a trait's power and polygenicity shift its own null
    identically, so they cannot inflate it relative to another trait;
  * technical covariates regressed out per cell before scoring.

And the labels survived CELLxGENE's standardisation: `cell_state` retains Kanemaru's
76 categories, including SAN_P_cell (n=245), AVN_P_cell (155), Purkinje (110),
AVN_bundle_cell (38).

Scope
-----
Sinoatrial and atrioventricular node tissue only: 118,172 of 704,296 cells. That is
where the pacemaker cells are, and it still carries atrial cardiomyocytes, fibroblasts,
endothelium and glia as comparators — including the atrial cardiomyocytes needed for
the atrial-fibrillation positive control. The full atlas would need roughly 11 GB
resident, above what this machine allows.

Usage:  python scripts/102_prepare_singlecell.py
"""

import gc
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent.as_posix()
SRC = f"{ROOT}/data/singlecell/heart_global.h5ad"
OUT = f"{ROOT}/data/singlecell/node_subset.h5ad"

CONDUCTION = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]

a = ad.read_h5ad(SRC, backed="r")
print(f"full atlas: {a.n_obs:,} cells x {a.n_vars:,} genes")

print("\nregion values:")
for v, n in a.obs["region"].value_counts().items():
    print(f"  {str(v):<28}{n:>9,}")

# pick the node tissues by whichever column names them unambiguously
reg = a.obs["region"].astype(str)
tis = a.obs["tissue"].astype(str) if "tissue" in a.obs.columns else reg
keep = (reg.isin(["SAN", "AVN"]).to_numpy()
        | tis.str.contains("sinoatrial|atrioventricular", case=False,
                           regex=True).to_numpy())
print(f"\nselected {keep.sum():,} cells from node tissue")
if keep.sum() < 50_000:
    raise SystemExit("unexpected selection size — check the region/tissue values above")

sub = a[keep].to_memory()
del a
gc.collect()
print(f"in memory: {sub.n_obs:,} x {sub.n_vars:,}")

# .raw on these atlases is Ensembl-indexed; scanpy silently prefers it and would drop
# ~90% of symbol-matched genes. This bug already cost one run earlier in the project.
sub.raw = None
for c in ("feature_name", "gene_name", "gene_symbols"):
    if c in sub.var.columns:
        sub.var["symbol"] = sub.var[c].astype(str)
        sub.var_names = pd.Index(sub.var["symbol"])
        print(f"var_names set from var['{c}']")
        break
sub.var_names_make_unique()
sub.var.index.name = None

X = sub.X
X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
looks_raw = bool(np.allclose(X.data[:5000], np.round(X.data[:5000])))
print(f"X holds {'raw counts' if looks_raw else 'transformed values'}")
if not looks_raw and "counts" in sub.layers:
    X = sp.csr_matrix(sub.layers["counts"])
    looks_raw = True
    print("using layers['counts'] instead")
sub.X = X
sub.layers.clear()
sub.obsm.clear()
sub.varm.clear()
sub.obsp.clear()

# scDRS wants log-normalised expression plus per-cell technical covariates
tot = np.asarray(X.sum(axis=1)).ravel()
if looks_raw:
    Xn = X.multiply((1e4 / np.maximum(tot, 1))[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data)
    sub.X = Xn
    print("applied CPM + log1p")

sub.obs["n_genes_qc"] = np.asarray((X > 0).sum(axis=1)).ravel()
sub.obs["total_counts_qc"] = tot
if "pct_counts_mt" not in sub.obs.columns:
    mt = np.asarray(sub.var_names.str.upper().str.startswith("MT-"))
    sub.obs["pct_counts_mt"] = (np.asarray(X[:, mt].sum(axis=1)).ravel()
                                / np.maximum(tot, 1) * 100 if mt.any() else 0.0)

cov = pd.DataFrame({
    "const": 1,
    "n_genes": sub.obs["n_genes_qc"].to_numpy(),
    "log_total": np.log10(np.maximum(sub.obs["total_counts_qc"].to_numpy(), 1)),
    "pct_mt": sub.obs["pct_counts_mt"].to_numpy(dtype=float),
}, index=sub.obs_names)
cov.index.name = "index"
cov.to_csv(f"{ROOT}/data/scdrs/covariates.tsv", sep="\t")

print("\ncell_state composition of the conduction system:")
cs = sub.obs["cell_state"].astype(str)
for c in CONDUCTION:
    n = int((cs == c).sum())
    flag = "" if n >= 30 else "   (below 30 — scDRS group test will be unreliable)"
    print(f"  {c:<22}{n:>7,}{flag}")
print(f"\ntotal cell states present: {cs.nunique()}")
top = cs.value_counts().head(12)
print("largest groups (the comparator background):")
for v, n in top.items():
    print(f"  {v:<26}{n:>8,}")

sub.write_h5ad(OUT, compression="gzip")
print(f"\nwrote {OUT}  ({os.path.getsize(OUT) / 1e9:.2f} GB)")
print(f"covariates -> {ROOT}/data/scdrs/covariates.tsv")
