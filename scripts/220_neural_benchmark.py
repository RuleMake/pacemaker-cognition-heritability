"""
How large is 0.674, really? The missing neural benchmark.
=========================================================

Every AUC in this project is relative: a pacemaker cell scores 0.674 for educational
attainment *against the working myocardium beside it*. That number has never been
calibrated. It could mean pacemaker cells are nearly as neural as a neural cell, or it
could mean the myocyte comparator is simply a poor place for a cognitive gene set to
land and any non-myocyte would clear 0.6.

The atlas answers this directly and we never asked it. It contains 2,365 neural-lineage
cells in this subset -- six neural-crest / glial / Schwann states -- sitting in the same
donors, the same regions and the same chemistries as the conduction populations. Scoring
them with the identical statistic gives the scale the paper is missing.

The comparator has to be strictly working cardiomyocytes for both arms, not the
"myocyte lineage" pool used for the primary table. That pool is defined as
cardiomyocytes plus any other conduction population present, which makes sense when the
focus cell is itself a conduction cell and makes no sense for glia. Running both arms
against strict CM is the only apples-to-apples comparison available, and it is the
comparator of the existing sensitivity analysis (Supplementary Table S2), so the
conduction numbers this script prints are directly checkable against it.

What each outcome would mean, written down before running:

  glia >> pacemaker    the neural signal in pacemaker cells is a diluted version of a
                       real neural signal present in the tissue; enrichment is genuine
                       but pacemaker cells are not remarkable among non-myocytes
  glia ~= pacemaker    pacemaker cells carry as much cognitive heritability as the
                       tissue's actual neural cells, which is the strong reading
  glia << pacemaker    the effect is specific to pacemaker cells rather than to
                       neural-ness, which would be surprising and would need explaining

The immune control must stay flat in BOTH arms. If rheumatoid arthritis is elevated in
glia, the comparator rather than the focus cell is doing the work and neither number
means anything.

Usage:  python scripts/220_neural_benchmark.py
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
H5 = ROOT / "data/singlecell/axis_subset.h5ad"
SC = ROOT / "results/scdrs_axis"
OUT = ROOT / "results/neural_benchmark.json"

CONDUCTION = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
NEURAL = ["NC1_glial", "NC2_glial_NGF+", "NC3_glial", "NC4_glial", "NC5_glial",
          "NC6_schwann"]
REPORT = ["EducationalAttainment", "Intelligence", "ReactionTime", "HRV_SDNN",
          "HRV_RMSSD", "RestingHeartRate", "PRinterval", "QTinterval",
          "RheumatoidArthritis", "RheumatoidArthritis_noMHC"]
MIN_CM = 30

a = ad.read_h5ad(H5, backed="r")
obs = a.obs[["donor_id", "region", "assay", "cell_state"]].astype(str)
obs["stratum"] = obs.donor_id + "/" + obs.region + "/" + obs.assay

scores = {}
for p in sorted(SC.glob("*.score.tsv")):
    t = p.name.replace(".score.tsv", "")
    s = pd.read_csv(p, sep="\t", index_col=0)["norm_score"].reindex(obs.index)
    if s.isna().any():
        raise SystemExit(f"{t}: score file does not cover every cell")
    scores[t] = s.to_numpy()

strat_codes, strat_names = pd.factorize(obs.stratum)
state = obs.cell_state.to_numpy()
IS_CM = pd.Series(state).str.match(r"^(aCM|vCM)").to_numpy()

# one pooled glial group as well as the six states, because five of the six are small
# and the pooled value is the number the paper needs
state_pooled = np.where(np.isin(state, NEURAL), "NC_all_neural", state)

print(f"cells {len(obs):,}   strata {len(strat_names)}   traits {len(scores)}")
print(f"working cardiomyocytes (comparator): {IS_CM.sum():,}")
for s in NEURAL:
    print(f"   {s:<18}{(state == s).sum():>6}")
print(f"   {'NC_all_neural':<18}{np.isin(state, NEURAL).sum():>6}   <- pooled")


def stratified(trait, labels, focus_mask):
    """van Elteren stratified AUC of each label vs working CM of the same stratum.

    Identical arithmetic to 153_axis_interaction.stratified_all(background='cm'):
    rank once per stratum, take each group's rank-sum, combine strata with weight
    1/(N+1). Strata are never pooled.
    """
    v = scores[trait]
    keep = IS_CM | focus_mask
    num, den, stat, var, ns, nc = {}, {}, {}, {}, {}, {}
    for si in range(len(strat_names)):
        m = (strat_codes == si) & keep
        N = int(m.sum())
        if N < MIN_CM:
            continue
        r = stats.rankdata(v[m])
        rs = pd.Series(r).groupby(labels[m]).agg(["sum", "size"])
        for c, (W, n1) in rs.iterrows():
            n1 = int(n1)
            n2 = N - n1
            if n2 < MIN_CM:
                continue
            auc_i = (W - n1 * (n1 + 1) / 2) / (n1 * n2)
            w = n1 * n2 / (N + 1)
            num[c] = num.get(c, 0.0) + w * auc_i
            den[c] = den.get(c, 0.0) + w
            stat[c] = stat.get(c, 0.0) + (W - n1 * (N + 1) / 2) / (N + 1)
            var[c] = var.get(c, 0.0) + n1 * n2 / (12.0 * (N + 1))
            ns[c] = ns.get(c, 0) + 1
            nc[c] = nc.get(c, 0) + n1
    return {c: dict(auc=num[c] / den[c], z=stat[c] / np.sqrt(var[c]),
                    n_strata=ns[c], n_cells=nc[c])
            for c in den if den[c] > 0 and var[c] > 0}


neural_mask = np.isin(state, NEURAL)
cond_mask = np.isin(state, CONDUCTION)

res = {"neural": {}, "conduction": {}}
for t in scores:
    res["neural"][t] = stratified(t, state_pooled, neural_mask)
    res["conduction"][t] = stratified(t, state, cond_mask)

# ------------------------------------------------------------------ report
GROUPS = ["NC_all_neural"] + NEURAL
print("\n" + "=" * 112)
print("NEURAL-LINEAGE CELLS vs WORKING CARDIOMYOCYTES of the same donor/region/assay")
print("=" * 112)
hdr = f"{'trait':<26}" + "".join(f"{g[:15]:>17}" for g in ["NC_all_neural", "NC1_glial",
                                                           "NC2_glial_NGF+"])
print(hdr)
for t in REPORT:
    if t not in res["neural"]:
        continue
    line = f"{t:<26}"
    for g in ["NC_all_neural", "NC1_glial", "NC2_glial_NGF+"]:
        r = res["neural"][t].get(g)
        line += f"{('%.3f (z%+.1f)' % (r['auc'], r['z'])) if r else '-':>17}"
    print(line)

print("\n" + "=" * 112)
print("THE COMPARISON: pooled neural cells vs the two nodal pacemaker populations")
print("   both against the SAME strict working-cardiomyocyte comparator")
print("=" * 112)
print(f"{'trait':<26}{'neural (n=?)':>18}{'SAN_P_cell':>18}{'AVN_P_cell':>18}"
      f"{'pacemaker - neural':>22}")
delta = {}
for t in REPORT:
    n = res["neural"].get(t, {}).get("NC_all_neural")
    s = res["conduction"].get(t, {}).get("SAN_P_cell")
    v = res["conduction"].get(t, {}).get("AVN_P_cell")
    if not (n and s):
        continue
    d = s["auc"] - n["auc"]
    delta[t] = d
    line = (f"{t:<26}"
            f"{'%.3f (z%+.1f)' % (n['auc'], n['z']):>18}"
            f"{'%.3f (z%+.1f)' % (s['auc'], s['z']):>18}"
            f"{('%.3f (z%+.1f)' % (v['auc'], v['z'])) if v else '-':>18}"
            f"{d:>+22.3f}")
    print(line)

n_ctrl = res["neural"].get("RheumatoidArthritis", {}).get("NC_all_neural")
print("\nCONTROL CHECK (must be flat in the neural arm too):")
if n_ctrl:
    ok = abs(n_ctrl["auc"] - 0.5) < 0.08
    print(f"   rheumatoid arthritis in neural cells: {n_ctrl['auc']:.3f} "
          f"(z{n_ctrl['z']:+.2f})  -> {'FLAT, benchmark usable' if ok else 'NOT FLAT, benchmark unusable'}")

OUT.write_text(json.dumps(res, indent=1))
print(f"\nwritten -> {OUT.relative_to(ROOT)}")
