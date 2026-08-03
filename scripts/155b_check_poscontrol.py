"""
Where did the atrial myocytes go for atrial fibrillation on the axis subset?
===========================================================================

On the node subset the positive control passed cleanly: the best atrial cardiomyocyte
state ranked 3 of 62 for atrial fibrillation. On the axis subset no aCM state is in the
top five. That has to be explained before anything else from this subset is used,
because the positive control is not a formality — it is the statement that the pipeline
can find a known answer in this particular matrix.

Three candidate explanations, and they call for different responses:

  CAPPING    Each state is capped at 1,500 cells here. Atrial myocytes went from ~6,000
             to 1,500, so their group means are noisier. Noise lowers a z, but it does
             not systematically favour ventricular states.
  ASSAY      Atrial tissue and ventricular tissue were not run on the same chemistry
             mix. If the ventricular states sit in strata that inflate scores, the
             ranking is measuring chemistry.
  REAL       Atrial fibrillation heritability really does sit partly in ventricular
             myocardium, and the node subset simply had almost no ventricular tissue
             to compete — 118k cells from SAN and AVN only.

The third would not be a failure at all. It would mean the old ranking was conditional
on a panel that excluded the competition.

Usage:  python scripts/155b_check_poscontrol.py
"""

from pathlib import Path

import anndata as ad
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.as_posix()

for tag, sc, h5 in (("node subset", f"{ROOT}/results/scdrs",
                     f"{ROOT}/data/singlecell/node_subset.h5ad"),
                    ("axis subset", f"{ROOT}/results/scdrs_axis",
                     f"{ROOT}/data/singlecell/axis_subset.h5ad")):
    p = f"{sc}/AtrialFibrillation.group.tsv"
    if not Path(p).exists():
        continue
    g = pd.read_csv(p, sep="\t", index_col=0).sort_values("assoc_mcz", ascending=False)
    lst = list(g.index)
    a = ad.read_h5ad(h5, backed="r")
    cs = a.obs["cell_state"].astype(str)
    reg = a.obs["region"].astype(str) if "region" in a.obs.columns else None
    asy = a.obs["assay"].astype(str) if "assay" in a.obs.columns else None

    print("=" * 100)
    print(f"{tag}   {a.n_obs:,} cells   {len(lst)} states ranked")
    print("=" * 100)
    print(f"{'state':<20}{'rank':>10}{'z':>8}{'n':>8}   dominant region / assay")
    for c in lst:
        if not (c.startswith("aCM") or c.startswith("vCM")):
            continue
        m = (cs == c).to_numpy()
        r = f"#{lst.index(c) + 1}/{len(lst)}"
        d = ""
        if reg is not None:
            d += reg[m].value_counts(normalize=True).head(1).to_string().split("\n")[0]
        if asy is not None:
            top = asy[m].value_counts(normalize=True)
            d += f"   {top.index[0]} {top.iloc[0] * 100:.0f}%"
        print(f"{c:<20}{r:>10}{g.loc[c, 'assoc_mcz']:>8.1f}{int(m.sum()):>8,}   {d}")
    print()
