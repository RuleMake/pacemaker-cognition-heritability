"""
What is the "unclassified" group, and why does it top every powered cardiac trait?
=================================================================================

`unclassified` (6,441 cells) ranks first for atrial fibrillation (+15.9), PR interval
(+20.7) and resting heart rate (+16.1), each time by a wide margin over the runner-up.
A group the original authors could not assign taking the headline position for the
three best-powered cardiac traits demands an explanation before any of the conduction
results are written up — it is the first thing a reviewer will ask about.

Three candidate explanations, each with a distinct signature:

  doublets            two cells captured together express both parents' programmes, so
                      any gene set finds hits. Signature: high gene count, high UMI,
                      elevated doublet score.
  detection artefact  cells that simply detect more genes score higher. scDRS regresses
                      n_genes out per cell, so this should already be handled —
                      checking confirms whether it was.
  genuine myocytes    if these are ambiguous cardiomyocytes, scoring like myocytes is
                      correct behaviour and the label is the only thing that is wrong.
                      Signature: cardiomyocyte marker expression.

The third would make the ranking legitimate; the first two would make it an artefact
to disclose. The conduction-cell conclusions do not depend on which is true — ranks
shift by at most one place if the group is dropped — but the answer has to be known.

Usage:  python scripts/106_diagnose_unclassified.py
"""

import glob
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent.as_posix()
H5 = f"{ROOT}/data/singlecell/node_subset.h5ad"
SC = f"{ROOT}/results/scdrs"

MARKERS = {
    "cardiomyocyte": ["TNNT2", "MYH7", "MYH6", "ACTC1", "TTN", "RYR2", "NPPA"],
    "pacemaker": ["HCN4", "HCN1", "SHOX2", "TBX3", "ISL1", "VSNL1"],
    "fibroblast": ["DCN", "PDGFRA", "COL1A1", "LUM"],
    "endothelial": ["PECAM1", "VWF", "CDH5"],
    "immune": ["PTPRC", "CD3E", "CD68", "LYZ"],
}

a = ad.read_h5ad(H5)
cs = a.obs["cell_state"].astype(str)
target = "unclassified"
m = (cs == target).to_numpy()
print(f"{target}: {m.sum():,} of {a.n_obs:,} cells ({m.mean() * 100:.1f}%)\n")

# ---------------------------------------------------------------- technical profile
print("=" * 92)
print("1. TECHNICAL PROFILE vs everything else")
print("=" * 92)
cols = [c for c in ["n_genes", "total_counts", "pct_counts_mt", "pct_counts_ribo",
                    "scrublet_score"] if c in a.obs.columns]
print(f"{'metric':<22}{'unclassified':>16}{'all others':>14}{'ratio':>9}")
tech = {}
for c in cols:
    v = pd.to_numeric(a.obs[c], errors="coerce").to_numpy(dtype=float)
    x, y = np.nanmedian(v[m]), np.nanmedian(v[~m])
    tech[c] = (float(x), float(y))
    print(f"{c:<22}{x:>16,.2f}{y:>14,.2f}{x / max(y, 1e-9):>9.2f}")

if "scrublet_score" in tech:
    sc_x, sc_y = tech["scrublet_score"]
    print(f"\ndoublet score {sc_x:.3f} vs {sc_y:.3f} — "
          f"{'ELEVATED, doublets are plausible' if sc_x > sc_y * 1.3 else 'not elevated'}")

# ---------------------------------------------------------------- marker identity
print("\n" + "=" * 92)
print("2. WHAT DO THEY EXPRESS? (mean log-normalised, vs the tissue average)")
print("=" * 92)
X = a.X
X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
gidx = {g: i for i, g in enumerate(a.var_names)}
Xc = X.tocsc()
print(f"{'marker set':<18}{'unclassified':>14}{'all others':>13}{'ratio':>8}   genes used")
for lbl, genes in MARKERS.items():
    ii = [gidx[g] for g in genes if g in gidx]
    if not ii:
        continue
    v = np.asarray(Xc[:, ii].mean(axis=1)).ravel()
    x, y = float(v[m].mean()), float(v[~m].mean())
    print(f"{lbl:<18}{x:>14.3f}{y:>13.3f}{x / max(y, 1e-9):>8.2f}   "
          f"{', '.join(g for g in genes if g in gidx)}")

# ---------------------------------------------------------------- where they sit
print("\n" + "=" * 92)
print("3. WHERE DO THEY COME FROM?")
print("=" * 92)
for c in ["region", "donor_id", "assay"]:
    if c not in a.obs.columns:
        continue
    n_levels = a.obs[c].astype(str).nunique()
    tab = pd.crosstab(a.obs.loc[m, c].astype(str), columns="unclassified")
    tot = a.obs[c].astype(str).value_counts()
    tab["share_of_that_group"] = (tab["unclassified"]
                                  / tot.reindex(tab.index).to_numpy() * 100)
    note = ""
    if n_levels == 1:
        # The first reading of this output took "100% 10x multiome" as evidence that
        # the group is a technical artefact. It is not: the whole dataset is that
        # assay, so the figure describes the background and says nothing about this
        # group. The doublet score and marker profile above are the real evidence.
        note = ("   [only one level in the dataset — uninformative about this group]")
    print(f"\n{c}:{note}")
    for i, r in tab.sort_values("unclassified", ascending=False).head(6).iterrows():
        print(f"   {str(i):<28}{int(r['unclassified']):>7,}"
              f"  ({r['share_of_that_group']:.1f}% of that group)")

# ---------------------------------------------------------------- does it matter?
print("\n" + "=" * 92)
print("4. DOES DROPPING IT CHANGE THE CONDUCTION RESULTS?")
print("=" * 92)
g = f"{SC}/group_analysis.tsv"
if os.path.exists(g):
    grp = pd.read_csv(g, sep="\t")
    key = "cell_state" if "cell_state" in grp.columns else grp.columns[0]
    print(f"{'trait':<24}{'SAN_P_cell':>22}{'AVN_P_cell':>22}")
    for t in sorted(grp.trait.unique()):
        sub = grp[grp.trait == t].sort_values("assoc_mcz", ascending=False)
        line = f"{t:<24}"
        for cell in ["SAN_P_cell", "AVN_P_cell"]:
            lst = list(sub[key])
            lst2 = [x for x in lst if x != target]
            if cell not in lst:
                line += f"{'-':>22}"
                continue
            r1, r2 = lst.index(cell) + 1, lst2.index(cell) + 1
            line += f"{f'{r1} -> {r2}':>22}"
        print(line)
    print("\nRanks with the group present -> with it removed. Dropping one group can")
    print("move any other by at most one place, so the conduction conclusions do not")
    print("rest on how this label is interpreted.")
else:
    print("group_analysis.tsv not found")
