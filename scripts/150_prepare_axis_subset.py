"""
Build an atlas-wide subset that contains the WHOLE conduction axis.
===================================================================

The subset every result so far rests on was cut to sinoatrial and atrioventricular
tissue. That was the right call at the time — it fit in memory and it held the cells
the hypothesis was about — but it has a consequence that has not been stated plainly:

    every trait tested so far is an atrial or nodal readout, so every trait has had
    the same predicted answer.

The anatomical specificity map is therefore twelve confirmations and zero opportunities
for disconfirmation. Brugada came closest, pushing the nodal cells down and the
ventricular myocytes up, and that is exactly why it is the most persuasive row in the
table. The way to make the whole map persuasive is to give it traits whose predicted
cell is one that has never won before.

The atlas has them. 110 Purkinje cells sit in apex tissue, outside the old subset:

    sinoatrial node -> atrioventricular node -> His bundle -> Purkinje network

With QRS duration and bundle branch block reading out His-Purkinje conduction, and QT
interval reading out ventricular repolarisation, three traits now predict three cells
that no previous trait has hit. If they land where anatomy says, the map has survived
a real test. If they land on the sinoatrial node again, the map was never measuring
anatomy and the whole specificity argument fails — which is the point.

Sampling
--------
704,296 cells will not fit; the old subset peaked near 6 GB at 118,172. Abundant cell
states are therefore capped, and the cap is applied WITHIN each donor-region stratum
in proportion, so capping cannot quietly delete a region or a donor from a state. Rare
states — which is every conduction state — are kept whole.

Usage:  python scripts/150_prepare_axis_subset.py [--cap 1500]
"""

import argparse
import gc
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent.as_posix()
SRC = f"{ROOT}/data/singlecell/heart_global.h5ad"
OUT = f"{ROOT}/data/singlecell/axis_subset.h5ad"
COV = f"{ROOT}/data/scdrs/covariates_axis.tsv"

CONDUCTION = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
SEED = 20260801

ap = argparse.ArgumentParser()
ap.add_argument("--cap", type=int, default=1500,
                help="max cells kept per cell state")
args = ap.parse_args()

rng = np.random.default_rng(SEED)
a = ad.read_h5ad(SRC, backed="r")
obs = a.obs
print(f"full atlas: {a.n_obs:,} cells x {a.n_vars:,} genes")
print(f"regions: {dict(obs.region.astype(str).value_counts())}")
print(f"donors: {obs.donor_id.astype(str).nunique()}   "
      f"cell states: {obs.cell_state.astype(str).nunique()}")

# ------------------------------------------------------------------ sampling plan
state = obs.cell_state.astype(str).to_numpy()
strat = (obs.donor_id.astype(str) + "/" + obs.region.astype(str)).to_numpy()
pos = np.arange(len(state))

keep_idx = []
for c in pd.unique(state):
    idx = pos[state == c]
    if len(idx) <= args.cap:
        keep_idx.append(idx)
        continue
    # proportional within donor-region, so a cap cannot silently drop a region
    sub = pd.Series(strat[idx], index=idx)
    take = []
    for s, g in sub.groupby(sub.values):
        k = max(1, int(round(args.cap * len(g) / len(idx))))
        k = min(k, len(g))
        take.append(rng.choice(g.index.to_numpy(), size=k, replace=False))
    keep_idx.append(np.concatenate(take))

keep = np.sort(np.concatenate(keep_idx))
print(f"\nsampling plan: cap {args.cap} per state -> {len(keep):,} cells "
      f"({100 * len(keep) / a.n_obs:.1f}% of the atlas)")

print("\nconduction states in the plan:")
for c in CONDUCTION:
    n = int((state[keep] == c).sum())
    tot = int((state == c).sum())
    d = obs.donor_id.astype(str).to_numpy()[keep][state[keep] == c]
    r = obs.region.astype(str).to_numpy()[keep][state[keep] == c]
    print(f"  {c:<20}{n:>6,} of {tot:,}   donors {len(set(d))}   "
          f"regions {sorted(set(r))}")

if int((state[keep] == "Purkinje").sum()) < 50:
    raise SystemExit("Purkinje cells did not survive the plan — the whole point of "
                     "this subset is gone; check the cell_state values")

# ------------------------------------------------------------------ materialise
sub = a[keep].to_memory()
del a
gc.collect()
print(f"\nin memory: {sub.n_obs:,} x {sub.n_vars:,}")

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

X = sp.csr_matrix(sub.X) if not sp.issparse(sub.X) else sub.X.tocsr()
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
cov.to_csv(COV, sep="\t")

cs = sub.obs["cell_state"].astype(str)
print(f"\ncell states present: {cs.nunique()}")
print("largest groups (the comparator background):")
for v, n in cs.value_counts().head(10).items():
    print(f"  {v:<26}{n:>8,}")

sub.write_h5ad(OUT, compression="gzip")
print(f"\nwrote {OUT}  ({os.path.getsize(OUT) / 1e9:.2f} GB)")
print(f"covariates -> {COV}")
