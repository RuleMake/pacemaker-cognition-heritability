"""
Does the genetic signal follow PACEMAKER CELL ABUNDANCE rather than the anatomical label?
========================================================================================

Why this was the gap
--------------------
Everything so far tested `annotation_final`, a BINARY anatomical label. That is
exactly where the resolution problem bites: a 55 um spot labelled `node` still
contains mostly atrial-myocardium-like cells, because the sinoatrial node is
compartmentalised (Kanemaru 2023). So a spot labelled `node` is not a pure
pacemaker spot, and testing node-vs-rest asks the wrong question.

But the dataset already carries the answer: cell2location deconvolution gives a
CONTINUOUS per-spot abundance for each cell type, including `SAN_P_cell`
(sinoatrial pacemaker cells). If heart-rate genetics really is executed by
pacemaker cells, gsMap's per-spot -log10 p should track SAN_P_cell abundance —
regardless of how the spot was labelled anatomically.

The test
--------
Per section, correlate gsMap logp against each cell-type abundance (Spearman).
Then take the trait-minus-control difference so that the tissue-activity gradient
cancels, exactly as in the compartment analysis:

    delta_rho(cell type) = rho(trait, abundance) - rho(control, abundance)

A positive, reproducible delta_rho for SAN_P_cell under RestingHeartRate — and not
under AtrialFibrillation — is the cell-level version of the original hypothesis,
and it is immune to the anatomical-label mixing problem.

Usage:  python scripts/40_gsmap_vs_celltype.py
"""

from pathlib import Path
import csv
import glob
import os
from collections import defaultdict

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = str(Path(__file__).resolve().parent.parent)
SPOT = f"{ROOT}/results/spotlevel"
OUT = f"{ROOT}/results"
CONTROL = "EducationalAttainment"

# conduction-relevant cell types plus contrasts, as named by cell2location in the atlas
FOCUS = ["SAN_P_cell", "AVN_bundle_cell", "aCM1", "aCM2", "aCM3", "aCM4",
         "vCM1", "vCM2", "vCM3_stressed", "FB1", "FB2", "FB4_activated",
         "NC1_glial", "NC2_glial_NGF+", "EC2_cap", "PC1_vent", "Adip1"]


def abundances():
    """Per-spot cell-type abundance from the source atlas, keyed by section."""
    out = {}
    for region in ["SAN", "AVN"]:
        p = f"{ROOT}/data/spatial/{region}.h5ad"
        if not os.path.exists(p):
            continue
        a = ad.read_h5ad(p, backed="r")
        obs = a.obs
        sid = obs["sangerID"].astype(str)
        cols = [c for c in FOCUS if c in obs.columns]
        for s in sid.unique():
            m = (sid == s).to_numpy()
            sub = obs.loc[m, cols].astype(float)
            sub.index = obs.index[m]
            out[f"{region}__{s}"] = sub
    return out


ABUND = abundances()
print(f"sections with abundance data: {len(ABUND)}")
print(f"cell types available: {list(next(iter(ABUND.values())).columns)}\n")

# ---------------------------------------------------------------- correlate
rows = []
for p in sorted(glob.glob(f"{SPOT}/*__*.csv")):
    base = os.path.basename(p)[:-4]
    section, trait = base.rsplit("__", 1)
    if section not in ABUND:
        continue
    df = pd.read_csv(p, index_col=0)
    ab = ABUND[section]
    common = df.index.intersection(ab.index)
    if len(common) < 200:
        continue
    lp = df.loc[common, "logp"].to_numpy()
    for ct in ab.columns:
        v = ab.loc[common, ct].to_numpy()
        if not np.isfinite(v).any() or np.nanstd(v) == 0:
            continue
        ok = np.isfinite(v) & np.isfinite(lp)
        if ok.sum() < 200:
            continue
        rho = spearmanr(lp[ok], v[ok])[0]
        rows.append(dict(section=section, region=section.split("__")[0],
                         trait=trait, cell_type=ct, n=int(ok.sum()), rho=float(rho)))

if not rows:
    raise SystemExit("no overlap between gsMap output and abundance table")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/gsmap_vs_celltype_raw.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- trait - control
piv = df.pivot_table(index=["region", "section", "cell_type"],
                     columns="trait", values="rho")
if CONTROL not in piv.columns:
    raise SystemExit(f"control {CONTROL} missing")

res = []
for trait in [c for c in piv.columns if c != CONTROL]:
    d = (piv[trait] - piv[CONTROL]).dropna()
    for (region, section, ct), val in d.items():
        res.append(dict(region=region, section=section, trait=trait,
                        cell_type=ct, delta_rho=float(val)))
rd = pd.DataFrame(res)
rd.to_csv(f"{OUT}/gsmap_vs_celltype_delta.tsv", sep="\t", index=False)

print("=" * 96)
print("delta_rho = rho(trait) - rho(control):  does the signal track this cell type,")
print("beyond what a non-cardiac trait shows on the same spots?")
print("=" * 96)

for region in ["SAN", "AVN"]:
    sub = rd[rd.region == region]
    if sub.empty:
        continue
    traits = sorted(sub.trait.unique())
    cts = sorted(sub.cell_type.unique())
    print(f"\n--- {region}  ({sub.section.nunique()} sections)")
    print(f"{'cell type':<20}" + "".join(f"{t[:20]:>22}" for t in traits))
    for ct in cts:
        line = f"{ct:<20}"
        for t in traits:
            v = sub[(sub.cell_type == ct) & (sub.trait == t)].delta_rho.to_numpy()
            if len(v) == 0:
                line += f"{'-':>22}"
                continue
            star = ""
            if len(v) >= 5:
                try:
                    p = wilcoxon(v).pvalue
                    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                except ValueError:
                    star = ""
            line += f"{v.mean():>+15.3f}{star:<7}"
        mark = "   <<< pacemaker" if ct == "SAN_P_cell" else (
               "   <<< AV bundle" if ct == "AVN_bundle_cell" else "")
        print(line + mark)

print("\n(mean across sections; * = Wilcoxon signed-rank vs 0, needs >=5 sections)")
print(f"\nwrote {OUT}/gsmap_vs_celltype_raw.tsv and {OUT}/gsmap_vs_celltype_delta.tsv")
