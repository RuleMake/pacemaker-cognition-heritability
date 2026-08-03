"""
Three more things the result has to survive.
============================================

  A GENE-SET SIZE   top-1000 is a convention inherited from the scDRS papers, not a
                    property of the biology. If the conduction cells only lead at that
                    particular cutoff, the finding is an artefact of the cutoff. Re-run
                    the scoring at 250 / 500 / 1000 / 2000 and watch the ordering.
                    (Done with a weighted-expression statistic rather than full scDRS:
                    scoring one trait takes 23 minutes, and what is being tested is
                    whether the ORDER moves, which the cheap statistic tracks.)

  B WHICH GENES     A polygenic score can be right for the wrong reason. If the
                    heart-rate gene set lights up pacemaker cells through HCN4, SHOX2
                    and TBX3, the mechanism is interpretable and the finding says
                    something. If it runs through generic mitochondrial genes, it is
                    a much weaker claim even at the same p-value.

  C ANATOMY         SAN_P_cell should come from sinoatrial tissue and AVN_P_cell from
                    atrioventricular tissue. If the labels are anatomically scrambled
                    the annotation itself cannot be trusted, and everything built on
                    it goes with it. This is the cheapest possible sanity check and it
                    has not been run.

Usage:  python scripts/109_audit_robustness.py
"""

import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent.as_posix()
H5 = f"{ROOT}/data/singlecell/node_subset.h5ad"
GS = f"{ROOT}/data/scdrs/traits.gs"
SC = f"{ROOT}/results/scdrs"

FOCUS = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell"]
TRAITS = ["RestingHeartRate", "PRinterval", "HRV_RMSSD", "AtrialFibrillation",
          "EducationalAttainment"]
SIZES = [250, 500, 1000, 2000]
MIN_CELLS = 20

a = ad.read_h5ad(H5)
cs = a.obs["cell_state"].astype(str).to_numpy()
X = a.X
X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
gidx = {g: i for i, g in enumerate(a.var_names)}
gs = pd.read_csv(GS, sep="\t").set_index("TRAIT")
counts = pd.Series(cs).value_counts()
usable = counts[counts >= MIN_CELLS].index.tolist()
report = {}


def group_rank(genes, weights, cell):
    """Rank of `cell` by weighted mean expression, standardised per cell."""
    ii = [gidx[g] for g in genes if g in gidx]
    w = np.array([wt for g, wt in zip(genes, weights) if g in gidx], dtype=float)
    if len(ii) < 50:
        return None, None
    E = np.asarray(X[:, ii] @ (w / w.sum())).ravel()
    m = pd.Series(E).groupby(cs).mean()
    m = m[m.index.isin(usable)].sort_values(ascending=False)
    if cell not in m.index:
        return None, None
    return list(m.index).index(cell) + 1, len(m)


# ================================================================ A
print("=" * 96)
print("A. DOES THE RANKING DEPEND ON THE GENE-SET SIZE CUTOFF?")
print("=" * 96)
resA = {}
for cell in FOCUS:
    print(f"\n{cell}")
    print(f"{'trait':<24}" + "".join(f"{f'top-{s}':>12}" for s in SIZES))
    for t in TRAITS:
        if t not in gs.index:
            continue
        pairs = [p.split(":") for p in str(gs.loc[t, "GENESET"]).split(",") if ":" in p]
        pairs = [(s, float(w)) for s, w in pairs]
        pairs.sort(key=lambda x: -x[1])
        line = f"{t:<24}"
        for s in SIZES:
            sub = pairs[:s]
            r, n = group_rank([g for g, _ in sub], [w for _, w in sub], cell)
            line += f"{(f'#{r}/{n}' if r else '-'):>12}"
            resA.setdefault(cell, {}).setdefault(t, {})[s] = r
        print(line)
print("\nA finding that only exists at one cutoff is a property of the cutoff.")
report["gene_set_size"] = resA

# ================================================================ B
print("\n" + "=" * 96)
print("B. WHICH GENES CARRY THE SIGNAL?")
print("=" * 96)
CANON = {"HCN1", "HCN4", "SHOX2", "TBX3", "TBX18", "ISL1", "BMP4", "VSNL1",
         "CACNA1D", "CACNA1G", "KCNJ3", "SLC8A1", "RGS6", "POPDC2"}
resB = {}
for cell in ["SAN_P_cell", "AVN_P_cell"]:
    m = cs == cell
    print(f"\n{cell}  (n={m.sum()})")
    for t in ["RestingHeartRate", "PRinterval", "HRV_RMSSD"]:
        if t not in gs.index:
            continue
        pairs = [p.split(":") for p in str(gs.loc[t, "GENESET"]).split(",") if ":" in p]
        pairs = [(s, float(w)) for s, w in pairs if s in gidx]
        ii = [gidx[s] for s, _ in pairs]
        w = np.array([wt for _, wt in pairs])
        sub = X[:, ii]
        # contribution = weight x (mean in this cell type - mean everywhere)
        contrib = w * (np.asarray(sub[m].mean(axis=0)).ravel()
                       - np.asarray(sub.mean(axis=0)).ravel())
        order = np.argsort(-contrib)[:12]
        top = [(pairs[i][0], contrib[i]) for i in order]
        hits = [g for g, _ in top if g in CANON]
        print(f"   {t:<20}{', '.join(f'{g}' for g, _ in top[:10])}")
        if hits:
            print(f"   {'':<20}canonical conduction genes among them: "
                  f"{', '.join(hits)}")
        resB.setdefault(cell, {})[t] = dict(top=[g for g, _ in top],
                                            canonical=hits)
report["driver_genes"] = resB

# ================================================================ C
print("\n" + "=" * 96)
print("C. ARE THE LABELS ANATOMICALLY COHERENT?")
print("=" * 96)
if "region" in a.obs.columns:
    reg = a.obs["region"].astype(str).to_numpy()
    print(f"{'cell state':<20}{'n':>7}   region composition")
    okC = {}
    for c in FOCUS:
        m = cs == c
        if m.sum() == 0:
            continue
        v = pd.Series(reg[m]).value_counts(normalize=True)
        txt = ", ".join(f"{k} {p * 100:.0f}%" for k, p in v.items())
        expect = "SAN" if c.startswith("SAN") else "AVN"
        got = float(v.get(expect, 0.0))
        okC[c] = got
        flag = "  ok" if got > 0.8 else "  <- NOT where it should be"
        print(f"{c:<20}{m.sum():>7,}   {txt}{flag}")
    report["anatomy"] = okC
    print("\nSAN_P_cell should sit in sinoatrial tissue and AVN_P_cell/AVN_bundle_cell")
    print("in atrioventricular tissue. Anything else means the annotation is not")
    print("describing what its name says.")
else:
    print("  no region column")

with open(f"{SC}/audit_robustness.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwrote {SC}/audit_robustness.json")
