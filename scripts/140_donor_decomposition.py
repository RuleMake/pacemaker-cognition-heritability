"""
Decompose the conduction-cell signal donor by donor.
====================================================

The most substantive limitation on the record is that 58.7% of AVN_P_cell comes from
one donor (A61), and that dropping A61 moved resting heart rate from #1 to #9. That
number was allowed to stand as a weakness, but it was never the right test. Dropping a
donor removes 59% of the cells, so the rank falls for two entirely different reasons —
the effect might be a donor artefact, or the effect might be intact and merely no longer
resolvable with 64 cells. A rank is not an effect size, and leave-one-out cannot tell
those apart.

The right question is whether the effect appears independently INSIDE each donor.

  A  WITHIN-STRATUM RANK   Rank the cell states using only cells from one donor and one
                           region. Donor genotype, batch, dissection and region are all
                           held constant by construction, so nothing about a donor can
                           lift a state relative to its neighbours in the same stratum.
                           AVN tissue exists for three donors; A61 and AV13 both carry
                           enough AVN_P_cell to be ranked on their own. If the state
                           ranks high in both, it has replicated across independent
                           donors rather than borrowed strength from one.

  B  AUC                   Rank-biserial AUC of focus cells against every other cell in
                           the same stratum. Unlike a rank, AUC does not degrade when a
                           stratum is small — it just gets a wider interval. This is what
                           separates "the effect vanished" from "the power vanished".

  C  DONOR-BALANCED RANK   A global ranking in which every donor contributes equally
                           regardless of how many cells it gave. Scores are standardised
                           within stratum first, then averaged over strata with equal
                           weight. A61's 58.7% share becomes 1/k, so the headline
                           ranking simply cannot be a weighting artefact.

The negative controls are carried through every step. Rheumatoid arthritis has more
genome-wide significant loci than any cardiac trait here, so if within-donor ranking
manufactures conduction hits from nothing, RA is where it will show.

Usage:  python scripts/140_donor_decomposition.py
        python scripts/140_donor_decomposition.py --h5 data/singlecell/axis_subset.h5ad \
            --scdrs results/scdrs_axis
"""

import argparse
import glob
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.as_posix()

ap = argparse.ArgumentParser()
ap.add_argument("--h5", default="data/singlecell/node_subset.h5ad")
ap.add_argument("--scdrs", default="results/scdrs")
ap.add_argument("--by", default="",
                help="stratum columns, comma separated; default donor_id,region "
                     "plus assay whenever the data holds more than one assay")
ap.add_argument("--tag", default="", help="suffix for the output files")
_a = ap.parse_args()


def _abs(p):
    return p if os.path.isabs(p) else f"{ROOT}/{p}"


H5, SC = _abs(_a.h5), _abs(_a.scdrs)

FOCUS = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
# `unclassified` is not a cell type. On the axis subset it is 40% myeloid, 16%
# ventricular myocyte and 15% fibroblast, and its cells carry 48% more detected genes
# than the rest (1,594 vs 1,079). Higher detection lifts almost any gene set — the same
# complexity artefact that destroyed the spatial arm, here concentrated in one label.
# It is dropped from the RANKING but left in the background against which AUC is
# computed, since the cells are real and excluding them would be the more arbitrary act.
EXCLUDE_FROM_RANK = {"unclassified"}
CTRL = ["EducationalAttainment", "RheumatoidArthritis", "RheumatoidArthritis_noMHC"]
# A state needs this many cells in a stratum before its mean is worth ranking. Set low
# because the cell types of interest are rare by nature; n is printed alongside so the
# reader can discount a thin stratum without having to trust the threshold.
MIN_CELLS = 15
MIN_OTHER = 200

# ------------------------------------------------------------------ inputs
a = ad.read_h5ad(H5, backed="r")

# Which columns define a stratum. In the node subset every cell is 10x multiome, so
# donor and region were enough. The whole-atlas subset is a third multiome, a third
# 3' v3 and a third 3' v2, and the nodal conduction states are 100% multiome against
# that mixed background — an assay confound that simply did not exist before. Assay
# therefore joins the stratum key whenever the data holds more than one, so a cell is
# never compared against a cell processed with different chemistry.
by = [c.strip() for c in _a.by.split(",") if c.strip()]
if not by:
    by = ["donor_id", "region"]
    if "assay" in a.obs.columns and a.obs["assay"].astype(str).nunique() > 1:
        by.append("assay")
        print(f"multiple assays present ({a.obs['assay'].astype(str).nunique()}) "
              f"— assay added to the stratum key")
cols = sorted(set(by + ["cell_state"]))
obs = a.obs[cols].astype(str)
obs["stratum"] = obs[by].agg("/".join, axis=1)

files = sorted(glob.glob(f"{SC}/*.score.tsv"))
traits = [os.path.basename(p).replace(".score.tsv", "") for p in files]
print(f"cells {len(obs):,}   states {obs.cell_state.nunique()}   traits {len(traits)}")

# Strata big enough to rank inside. A donor that gave 42 cells of a region is not a
# replicate of that region, it is a handful of cells that happened to be dissected.
sizes = obs.stratum.value_counts()
strata = [s for s in sizes.index if sizes[s] >= 1000]
print(f"strata with >=1000 cells: {len(strata)}")
for s in strata:
    sub = obs[obs.stratum == s]
    have = {c: int((sub.cell_state == c).sum()) for c in FOCUS}
    have = {k: v for k, v in have.items() if v}
    print(f"  {s:<12}{sizes[s]:>8,}   " + (", ".join(f"{k} n={v}" for k, v in have.items())
                                           or "no focus state"))


def auc(x, y):
    """P(a random focus cell scores above a random other cell). 0.5 is no effect."""
    if len(x) == 0 or len(y) == 0:
        return np.nan
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(u / (len(x) * len(y)))


rows = []
balanced = {}

for t, path in zip(traits, files):
    sc = pd.read_csv(path, sep="\t", index_col=0)["norm_score"]
    sc = sc.reindex(obs.index)
    if sc.isna().any():
        raise SystemExit(f"{t}: score file does not cover every cell")

    # ---------------------------------------------------------- A + B per stratum
    zparts = []
    for s in strata:
        m = (obs.stratum == s).to_numpy()
        sub_state = obs.cell_state[m]
        v = sc[m].to_numpy()

        # standardise inside the stratum, so donor-level offsets cannot survive into C
        sd = v.std()
        zparts.append(pd.Series((v - v.mean()) / (sd if sd > 0 else 1.0),
                                index=obs.index[m]))

        counts = sub_state.value_counts()
        rankable = [c for c in counts.index
                    if counts[c] >= MIN_CELLS and c not in EXCLUDE_FROM_RANK]
        if len(rankable) < 5:
            continue
        means = {c: float(v[(sub_state == c).to_numpy()].mean()) for c in rankable}
        order = sorted(means, key=means.get, reverse=True)

        for c in FOCUS:
            if c not in rankable:
                continue
            inm = (sub_state == c).to_numpy()
            if (~inm).sum() < MIN_OTHER:
                continue
            r = order.index(c) + 1
            rows.append(dict(
                trait=t, stratum=s, state=c, n=int(inm.sum()),
                n_states=len(order), rank=r,
                pct=100.0 * (1 - (r - 1) / (len(order) - 1)),
                auc=auc(v[inm], v[~inm]),
                mean=means[c],
            ))

    # ---------------------------------------------------------- C donor-balanced
    z = pd.concat(zparts)
    df = pd.DataFrame(dict(z=z, state=obs.cell_state.reindex(z.index),
                           stratum=obs.stratum.reindex(z.index)))
    per = df.groupby(["state", "stratum"], observed=True).agg(m=("z", "mean"),
                                                              n=("z", "size"))
    per = per[per.n >= MIN_CELLS]
    per = per[~per.index.get_level_values("state").isin(EXCLUDE_FROM_RANK)]
    # equal weight per stratum: a donor with 20,995 cells counts once, as does one
    # with 1,784. This is the whole point — the 58.7% share stops mattering.
    bal = per.groupby("state", observed=True).m.mean().sort_values(ascending=False)
    balanced[t] = bal

res = pd.DataFrame(rows)
res.to_csv(f"{SC}/donor_decomposition{_a.tag}.tsv", sep="\t", index=False)

# ------------------------------------------------------------------ report A/B
print("\n" + "=" * 100)
print("A+B. WITHIN DONOR AND REGION — rank and AUC of the focus state among its own")
print("     stratum. Donor genotype, batch and region are constant inside each row.")
print("=" * 100)
for c in FOCUS:
    sub = res[res.state == c]
    if sub.empty:
        continue
    print(f"\n--- {c}")
    strat = sorted(sub.stratum.unique())
    print(f"{'trait':<28}" + "".join(f"{s + ' rank/AUC':>22}" for s in strat))
    for t in sorted(sub.trait.unique(), key=lambda x: (x in CTRL, x)):
        line = f"{t:<28}"
        for s in strat:
            r = sub[(sub.trait == t) & (sub.stratum == s)]
            if r.empty:
                line += f"{'-':>22}"
            else:
                r = r.iloc[0]
                # 'rank' collides with the DataFrame method, so index it explicitly
                cell = "#%d/%d %.3f" % (r["rank"], r["n_states"], r["auc"])
                line += f"{cell:>22}"
        print(line + ("   <- control" if t in CTRL else ""))

# ------------------------------------------------------------------ report C
print("\n" + "=" * 100)
print("C. DONOR-BALANCED GLOBAL RANK — every stratum weighted equally")
print("=" * 100)
orig = pd.read_csv(f"{SC}/group_analysis.tsv", sep="\t")
idx = "cell_state" if "cell_state" in orig.columns else orig.columns[0]

summary = {}
print(f"{'trait':<28}" + "".join(f"{c[:16]:>26}" for c in FOCUS))
print(f"{'':28}" + "".join(f"{'balanced (was)':>26}" for _ in FOCUS))
for t in sorted(balanced, key=lambda x: (x in CTRL, x)):
    bal = balanced[t]
    line = f"{t:<28}"
    o = orig[orig.trait == t].sort_values("assoc_mcz", ascending=False)
    olist = list(o[idx])
    summary[t] = {}
    for c in FOCUS:
        if c not in bal.index:
            line += f"{'-':>26}"
            continue
        r = list(bal.index).index(c) + 1
        was = olist.index(c) + 1 if c in olist else None
        summary[t][c] = dict(balanced_rank=r, n_states=len(bal),
                             original_rank=was, mean_z=float(bal[c]))
        line += f"{f'#{r}/{len(bal)} (was #{was})':>26}"
    print(line + ("   <- control" if t in CTRL else ""))

# ------------------------------------------------------------------ verdict
print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)
verdict = {}
for c in FOCUS:
    sub = res[res.state == c]
    if sub.empty:
        continue
    card = sub[~sub.trait.isin(CTRL)]
    ctrl = sub[sub.trait.isin(CTRL)]
    n_strata = sub.stratum.nunique()
    # replication: a cardiac trait counts as replicated when the state lands in the top
    # decile of its own stratum in more than one independent donor
    top = card[card.pct >= 90].groupby("trait").stratum.nunique()
    rep = sorted(top[top >= 2].index)
    verdict[c] = dict(
        n_strata=n_strata,
        median_auc_cardiac=float(card.auc.median()),
        median_auc_control=float(ctrl.auc.median()) if len(ctrl) else None,
        median_pct_cardiac=float(card.pct.median()),
        median_pct_control=float(ctrl.pct.median()) if len(ctrl) else None,
        replicated_traits=rep,
    )
    print(f"\n{c}  ({n_strata} independent strata)")
    print(f"  median AUC   cardiac {card.auc.median():.3f}   control "
          f"{ctrl.auc.median():.3f}" if len(ctrl) else "")
    print(f"  median pctile cardiac {card.pct.median():.1f}   control "
          f"{ctrl.pct.median():.1f}" if len(ctrl) else "")
    print(f"  traits in the top decile of >=2 independent donors: "
          f"{', '.join(rep) if rep else 'NONE'}")

with open(f"{SC}/donor_decomposition{_a.tag}.json", "w") as f:
    json.dump(dict(stratum_key=by, verdict=verdict, balanced=summary),
              f, indent=2, default=float)
print(f"\nwrote {SC}/donor_decomposition{_a.tag}.tsv and .json  "
      f"(strata defined by {'/'.join(by)})")
