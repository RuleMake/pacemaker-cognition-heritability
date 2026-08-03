"""
What IS the tissue-activity gradient? Test the library-size hypothesis.
======================================================================

The single most consequential observation so far is that gsMap's per-spot p-values
order the compartments identically for every trait, including a trait with no cardiac
biology whatsoever (educational attainment): haemorrhage < adipose < nerve <
atrial myocardium < node. Something trait-independent is driving the map.

The obvious candidate is technical, not biological. gsMap scores a spot by how
specifically its top genes are expressed there. A spot with a large library detects
more genes, so more genes can obtain a high specificity score, so more of them can
contribute LD-score signal. If that is what is happening, then:

    per-spot -log10(p)  should track  per-spot total UMI / genes detected

and it should do so for EVERY trait at roughly the same strength — including the
non-cardiac control. That is a sharp, falsifiable prediction, and it distinguishes a
technical artefact from a real biological gradient in tissue activity.

It matters well beyond this project. gsMap is applied to tissues chosen precisely
because they are structurally heterogeneous, and library size varies systematically
with cell density and tissue type across a section. If the association is strong, any
gsMap map read without a contrast is partly a map of sequencing depth.

Three quantities per spot:
    total_umi   sum of raw counts
    n_genes     number of genes detected
    mito_frac   mitochondrial fraction, a standard quality axis that should NOT drive
                heritability enrichment and so acts as a specificity check

Usage:  python scripts/52_library_size_confound.py
"""

import glob
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# resolved from the script's own location so the same file runs under Windows and
# under WSL, where only the WSL interpreter carries anndata
ROOT = Path(__file__).resolve().parent.parent.as_posix()
GS = f"{ROOT}/data/spatial_gsmap"
SPOT = f"{ROOT}/results/spotlevel"
OUT = f"{ROOT}/results"
CONTROL = "EducationalAttainment"


def qc_table(path):
    """Per-spot depth metrics from the raw count layer gsMap was given."""
    a = ad.read_h5ad(path)
    X = a.layers["count"] if "count" in a.layers else a.X
    tot = np.asarray(X.sum(axis=1)).ravel()
    ngene = np.asarray((X > 0).sum(axis=1)).ravel()
    mt = np.asarray(a.var_names.str.upper().str.startswith("MT-"))
    mito = (np.asarray(X[:, mt].sum(axis=1)).ravel() / np.maximum(tot, 1)
            if mt.any() else np.full(tot.shape, np.nan))
    return pd.DataFrame({"total_umi": tot, "n_genes": ngene, "mito_frac": mito,
                         "annotation": a.obs["annotation_final"].astype(str).to_numpy()},
                        index=a.obs_names)


qc = {}
for p in sorted(glob.glob(f"{GS}/*.h5ad")):
    s = os.path.basename(p)[:-5]
    qc[s] = qc_table(p)
    print(f"  {s:<28} {len(qc[s]):>6,} spots   median UMI "
          f"{qc[s].total_umi.median():>8,.0f}")
print()

rows = []
for p in sorted(glob.glob(f"{SPOT}/*__*.csv")):
    base = os.path.basename(p)[:-4]
    section, trait = base.rsplit("__", 1)
    if section not in qc:
        continue
    df = pd.read_csv(p, index_col=0)
    q = qc[section]
    c = df.index.intersection(q.index)
    if len(c) < 200:
        continue
    lp = df.loc[c, "logp"].to_numpy()
    r = dict(section=section, region=section.split("__")[0], trait=trait, n=len(c))
    for col in ["total_umi", "n_genes", "mito_frac"]:
        v = q.loc[c, col].to_numpy()
        ok = np.isfinite(v) & np.isfinite(lp)
        r[col] = float(spearmanr(lp[ok], v[ok])[0]) if ok.sum() >= 200 else np.nan
    rows.append(r)

d = pd.DataFrame(rows)
d.to_csv(f"{OUT}/library_size_confound.tsv", sep="\t", index=False)

traits = [t for t in ["RestingHeartRate", "AtrialFibrillation", "PRinterval",
                      "HRV_RMSSD", "HRV_RMSSDc", "HRV_SDNN", "HRV_SDNNc", CONTROL]
          if t in set(d.trait)]

print("=" * 88)
print("Spearman rho between gsMap -log10(p) and per-spot sequencing depth")
print("=" * 88)
print(f"{'trait':<24}{'total UMI':>14}{'genes detected':>18}{'mito fraction':>16}")
for t in traits:
    g = d[d.trait == t]
    tag = "   <- non-cardiac control" if t == CONTROL else ""
    print(f"{t:<24}{g.total_umi.mean():>+14.3f}{g.n_genes.mean():>+18.3f}"
          f"{g.mito_frac.mean():>+16.3f}{tag}")

print(f"\n{'':<24}{'(mean over ' + str(d.section.nunique()) + ' sections)':>14}")

sub = d[d.trait.isin(traits)]
spread = sub.groupby("trait").total_umi.mean()
print(f"\nrange across traits: {spread.min():+.3f} to {spread.max():+.3f} "
      f"(spread {spread.max() - spread.min():.3f})")
print("A large rho that is nearly constant across traits — control included — means the"
      "\nmap is substantially a map of sequencing depth, not of trait genetics.")

# ---- is depth itself compartment-structured? that is the link to the ordering
print("\n" + "=" * 88)
print("Per-compartment sequencing depth (this is what produces the fixed ordering)")
print("=" * 88)
allq = []
for s, q in qc.items():
    t = q.copy()
    t["section"] = s
    t["region"] = s.split("__")[0]
    allq.append(t)
aq = pd.concat(allq)
for region in ["SAN", "AVN"]:
    r = aq[aq.region == region]
    if r.empty:
        continue
    g = (r.groupby("annotation")
           .agg(spots=("total_umi", "size"), median_umi=("total_umi", "median"),
                median_genes=("n_genes", "median"))
           .sort_values("median_umi"))
    print(f"\n--- {region}")
    print(f"{'compartment':<24}{'spots':>9}{'median UMI':>13}{'median genes':>14}")
    for a, row in g.iterrows():
        print(f"{a:<24}{int(row.spots):>9,}{row.median_umi:>13,.0f}"
              f"{row.median_genes:>14,.0f}")

# ---- decisive: does depth explain the compartment ordering gsMap produced?
print("\n" + "=" * 88)
print("Rank agreement: compartment ordering by gsMap logp vs by sequencing depth")
print("=" * 88)
agree = []
for t in traits:
    rr = []
    for s in sorted(qc):
        p = f"{SPOT}/{s}__{t}.csv"
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, index_col=0)
        q = qc[s]
        c = df.index.intersection(q.index)
        m = pd.DataFrame({"logp": df.loc[c, "logp"].to_numpy(),
                          "umi": q.loc[c, "total_umi"].to_numpy(),
                          "ann": q.loc[c, "annotation"].to_numpy()})
        g = m.groupby("ann").median(numeric_only=True)
        if len(g) < 4:
            continue
        rr.append(spearmanr(g.logp, g.umi)[0])
    if rr:
        agree.append(dict(trait=t, mean_rho=float(np.mean(rr)),
                          min_rho=float(np.min(rr)), n=len(rr)))
ag = pd.DataFrame(agree)
if not ag.empty:
    ag.to_csv(f"{OUT}/library_size_ordering.tsv", sep="\t", index=False)
    print(f"{'trait':<24}{'rho(compartment logp, compartment UMI)':>42}{'sections':>10}")
    for _, r in ag.iterrows():
        print(f"{r.trait:<24}{r.mean_rho:>+42.3f}{int(r.n):>10}")
    print("\nrho near +1 means gsMap's compartment ranking is the sequencing-depth"
          "\nranking wearing a p-value.")

with open(f"{OUT}/library_size_confound.json", "w") as f:
    json.dump(dict(per_trait=d.groupby("trait")[["total_umi", "n_genes", "mito_frac"]]
                   .mean().to_dict("index"),
                   ordering=ag.to_dict("records") if not ag.empty else []),
              f, indent=2, default=float)
print(f"\nwrote {OUT}/library_size_confound.tsv, library_size_ordering.tsv, "
      f"library_size_confound.json")
