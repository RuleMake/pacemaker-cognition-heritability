"""
Is the gene-level result also just depth? The self-check the previous script owes.
==================================================================================

Scripts 50/51 found a clean dissociation: under cardiac traits the top gsMap genes are
working-myocyte genes (sarcomere, calcium handling, oxidative metabolism) while
node-specifying genes rank worse than they do under a non-cardiac control. That was
read as biology.

But script 52 then showed gsMap's per-spot p-value tracks sequencing depth at rho~0.76.
If p ~ depth, then a gene's PCC — the correlation between its specificity score and
p across spots — is largely the correlation between its specificity and DEPTH. Genes
marking dense, deeply-sequenced spots would top the list automatically. In this tissue
the dense spots are myocyte-rich, which would reproduce the observed "dissociation"
with no genetics involved at all.

So the honest test is:

    depth_affinity(gene) = corr(gene expression across spots, total UMI across spots)

and then corr(depth_affinity, PCC) within each trait. If that is near 1, the gene-level
finding is depth wearing a gene name and must be reported as such. If it is moderate,
the residual — PCC after removing depth affinity — is where any real signal lives, and
the node-vs-myocyte contrast should be recomputed on that residual.

Usage:  python scripts/54_gene_depth_affinity.py
"""

import glob
import json
import os
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent.parent.as_posix()
GS = f"{ROOT}/data/spatial_gsmap"
DIAG = f"{ROOT}/results/genediag"
OUT = f"{ROOT}/results"
CONTROL = "EducationalAttainment"

NODE_GENES = ["HCN1", "HCN4", "SHOX2", "TBX3", "TBX18", "ISL1", "BMP4", "VSNL1",
              "CACNA1D", "CACNA1G"]
MYOCYTE_GENES = ["MYH6", "MYH7", "TNNT2", "TTN", "ACTC1", "MYL7", "MYOM2", "LDB3",
                 "PLN", "RYR2", "CASQ2", "ATP2A2", "NPPA", "SRL", "NEBL"]


def depth_affinity(path):
    """Spearman of each gene's log-normalised expression against spot total UMI."""
    a = ad.read_h5ad(path)
    X = a.layers["count"] if "count" in a.layers else a.X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    tot = np.asarray(X.sum(axis=1)).ravel()
    # CPM + log so the trivial "more counts everywhere" is normalised out; what remains
    # is whether the gene's RELATIVE abundance rises in deep spots
    Xn = X.multiply(1e4 / np.maximum(tot, 1)[:, None]).tocsc()
    Xn.data = np.log1p(Xn.data)
    rt = pd.Series(tot).rank().to_numpy()
    aff = np.full(Xn.shape[1], np.nan)
    for j in range(Xn.shape[1]):
        s, e = Xn.indptr[j], Xn.indptr[j + 1]
        if e - s < 50:                    # too sparse for a meaningful correlation
            continue
        v = np.zeros(Xn.shape[0])
        v[Xn.indices[s:e]] = Xn.data[s:e]
        aff[j] = spearmanr(v, rt)[0]
    return pd.Series(aff, index=a.var_names), tot


def parse(p):
    b = os.path.basename(p).replace("_Gene_Diagnostic_Info.csv", "")
    m = re.match(r"((?:SAN|AVN)__[A-Za-z0-9]+)_(.+)$", b)
    return (m.group(1), m.group(2)) if m else (None, None)


san_files = sorted(glob.glob(f"{GS}/SAN__*.h5ad"))
AFF = {}
for p in san_files:
    s = os.path.basename(p)[:-5]
    AFF[s], _ = depth_affinity(p)
    print(f"  {s:<28} depth affinity for {AFF[s].notna().sum():,} genes")
print()

tables = {}
for p in sorted(glob.glob(f"{DIAG}/SAN__*_Gene_Diagnostic_Info.csv")):
    s, t = parse(p)
    if s is None:
        continue
    df = pd.read_csv(p).dropna(subset=["PCC"]).drop_duplicates(subset="Gene")
    tables[(s, t)] = df.set_index("Gene")

traits = sorted({t for _, t in tables})
order = [t for t in ["RestingHeartRate", "AtrialFibrillation", "PRinterval",
                     "HRV_RMSSD", "HRV_RMSSDc", "HRV_SDNN", "HRV_SDNNc", CONTROL]
         if t in traits]
report = {}

# ------------------------------------------------------------------ 1
print("=" * 90)
print("1. How much of a gene's PCC is explained by its affinity for deep spots?")
print("=" * 90)
rows = []
for (s, t), df in tables.items():
    if s not in AFF:
        continue
    a = AFF[s].reindex(df.index)
    ok = a.notna() & df.PCC.notna()
    if ok.sum() < 3000:
        continue
    rows.append(dict(section=s, trait=t, n=int(ok.sum()),
                     r=float(spearmanr(a[ok], df.PCC[ok])[0])))
d1 = pd.DataFrame(rows)
d1.to_csv(f"{OUT}/gene_depth_affinity.tsv", sep="\t", index=False)
print(f"{'trait':<24}{'rho(depth affinity, PCC)':>28}{'sections':>10}")
for t in order:
    g = d1[d1.trait == t]
    if g.empty:
        continue
    tag = "  <- control" if t == CONTROL else ""
    print(f"{t:<24}{g.r.mean():>+28.3f}{len(g):>10}{tag}")
report["affinity_vs_pcc"] = {t: float(d1[d1.trait == t].r.mean())
                             for t in order if len(d1[d1.trait == t])}

# ------------------------------------------------------------------ 2
print("\n" + "=" * 90)
print("2. Depth affinity of the two gene sets — are myocyte genes simply deep-spot genes?")
print("=" * 90)
aff_all = pd.concat(AFF.values(), axis=1).mean(axis=1)
for lbl, gs in [("node-specifying", NODE_GENES), ("working myocyte", MYOCYTE_GENES)]:
    v = aff_all.reindex([g for g in gs if g in aff_all.index]).dropna()
    pct = [float((aff_all < x).mean() * 100) for x in v]
    print(f"{lbl:<20} mean affinity {v.mean():+.3f}   median percentile "
          f"{np.median(pct):.0f}%   (n={len(v)})")
print(f"{'all genes':<20} mean affinity {aff_all.mean():+.3f}")
report["set_affinity"] = {
    lbl: float(aff_all.reindex([g for g in gs if g in aff_all.index]).dropna().mean())
    for lbl, gs in [("node", NODE_GENES), ("myocyte", MYOCYTE_GENES)]}

# ------------------------------------------------------------------ 3
print("\n" + "=" * 90)
print("3. The dissociation recomputed on PCC RESIDUALISED against depth affinity")
print("   (does node-vs-myocyte survive once deep-spot genes are levelled?)")
print("=" * 90)
rows = []
for (s, t), df in tables.items():
    if s not in AFF:
        continue
    a = AFF[s].reindex(df.index)
    ok = (a.notna() & df.PCC.notna()).to_numpy()
    if ok.sum() < 3000:
        continue
    x = pd.Series(a[ok]).rank().to_numpy()
    y = pd.Series(df.PCC[ok]).rank().to_numpy()
    Z = np.column_stack([x, np.ones(x.size)])
    resid = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = pd.Series(resid, index=df.index[ok])
    pct = (1 - r.rank(pct=True)) * 100
    nod = pct.reindex([g for g in NODE_GENES if g in pct.index]).dropna()
    myo = pct.reindex([g for g in MYOCYTE_GENES if g in pct.index]).dropna()
    rows.append(dict(section=s, trait=t, node=nod.median(), myo=myo.median(),
                     delta=nod.median() - myo.median()))
d3 = pd.DataFrame(rows)
d3.to_csv(f"{OUT}/gene_depth_residual.tsv", sep="\t", index=False)
print(f"{'trait':<24}{'node pctile':>13}{'myocyte pctile':>16}{'delta':>9}")
for t in order:
    g = d3[d3.trait == t]
    if g.empty:
        continue
    tag = "  <- control" if t == CONTROL else ""
    print(f"{t:<24}{g.node.mean():>12.1f}%{g.myo.mean():>15.1f}%"
          f"{g.delta.mean():>+9.1f}{tag}")

print("\nPaired vs control, on residualised PCC:")
ctrl = d3[d3.trait == CONTROL].set_index("section")
res3 = {}
for t in [x for x in order if x != CONTROL]:
    g = d3[d3.trait == t].set_index("section")
    c = g.index.intersection(ctrl.index)
    diff = (g.loc[c, "delta"] - ctrl.loc[c, "delta"]).dropna()
    if len(diff) < 6:
        continue
    p = wilcoxon(diff).pvalue
    res3[t] = (float(diff.mean()), float(p))
    keep = "survives" if p < 0.05 else "gone"
    print(f"  {t:<24}{diff.mean():>+9.1f}   p = {p:>7.4f}   [{keep}]")
report["residualised"] = res3

with open(f"{OUT}/gene_depth_affinity.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwrote {OUT}/gene_depth_affinity.tsv, gene_depth_residual.tsv, "
      f"gene_depth_affinity.json")
