"""
scDRS: does heart-rate heritability concentrate in pacemaker CELLS?
===================================================================

The original hypothesis, finally asked in a setting where the confounds that defeated
the spatial analysis cannot operate. Cells instead of 55 um spots, author annotations
instead of deconvolution, control gene sets matched on expression mean and variance,
and a per-trait Monte Carlo null.

Reading order matters
---------------------
The positive control is checked FIRST and everything else is conditional on it. If
atrial fibrillation does not concentrate in atrial cardiomyocytes in this dataset, the
pipeline is not working and no statement about pacemaker cells means anything —
whether it is positive or negative. That check is not optional and its result is
printed before any pacemaker number.

Group sizes are small for the conduction cell states (SAN_P_cell 245, AVN_P_cell 155,
Purkinje 110, AVN_bundle_cell 38). scDRS's group test handles small groups, but 38
cells is at the edge and is flagged rather than quietly reported.

Usage:  python scripts/103_run_scdrs.py
        python scripts/103_run_scdrs.py --ntrl 500     (faster, rougher null)
        python scripts/103_run_scdrs.py --h5 data/singlecell/axis_subset.h5ad \
            --cov data/scdrs/covariates_axis.tsv --out results/scdrs_axis
"""

import argparse
import gc
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scdrs

ROOT = Path(__file__).resolve().parent.parent.as_posix()

CONDUCTION = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
POSITIVE_CONTROL = ("AtrialFibrillation", "aCM")   # trait, cell-state prefix

ap = argparse.ArgumentParser()
ap.add_argument("--ntrl", type=int, default=1000, help="control gene sets")
# The paths are arguments so the same scorer can run the node subset and the
# whole-axis subset without a forked copy drifting out of step with this one.
ap.add_argument("--h5", default="data/singlecell/node_subset.h5ad")
ap.add_argument("--cov", default="data/scdrs/covariates.tsv")
ap.add_argument("--out", default="results/scdrs")
args = ap.parse_args()


def _abs(p):
    return p if os.path.isabs(p) else f"{ROOT}/{p}"


H5, COV, OUT = _abs(args.h5), _abs(args.cov), _abs(args.out)
GS = f"{ROOT}/data/scdrs/traits.gs"
os.makedirs(OUT, exist_ok=True)

for p in (H5, GS, COV):
    if not os.path.exists(p):
        raise SystemExit(f"missing input: {p}\n"
                         "run 102_prepare_singlecell.py and 101_magma_gene_scores.sh first")

adata = ad.read_h5ad(H5)
cov = pd.read_csv(COV, sep="\t", index_col=0)
cov = cov.loc[adata.obs_names]
print(f"cells {adata.n_obs:,}   genes {adata.n_vars:,}   covariates {list(cov.columns)}")

print("\npreprocessing (size factors + covariate regression + mean/var binning)...")
scdrs.preprocess(adata, cov=cov, n_mean_bin=20, n_var_bin=20, copy=False)

gs = scdrs.util.load_gs(GS, src_species="human", dst_species="human",
                        to_intersect=adata.var_names)
print(f"\ntrait gene sets: {len(gs)}")
for t, (genes, weights) in gs.items():
    print(f"  {t:<24}{len(genes):>5} genes present in the data")

group_col = "cell_state"
adata.obs[group_col] = adata.obs[group_col].astype(str)
sizes = adata.obs[group_col].value_counts()

# An earlier run wrote only the merged group_analysis.tsv, before per-trait caching
# existed. Split it so those traits are not scored a second time — at 23 minutes each
# that would be four hours spent reproducing numbers already on disk.
merged = f"{OUT}/group_analysis.tsv"
if os.path.exists(merged):
    m = pd.read_csv(merged, sep="\t")
    idx = "cell_state" if "cell_state" in m.columns else m.columns[0]
    for t, sub in m.groupby("trait"):
        c = f"{OUT}/{t}.group.tsv"
        if not os.path.exists(c):
            sub.set_index(idx).to_csv(c, sep="\t")
            print(f"recovered cached group analysis for {t}")

groups = {}
for trait, (genes, weights) in gs.items():
    if len(genes) < 100:
        print(f"\n!! {trait}: only {len(genes)} genes map into the data — skipped")
        continue
    # Scoring costs ~23 minutes per trait. Adding the disease traits later would
    # otherwise recompute every physiological trait a second time — four hours of
    # arithmetic to reproduce numbers already on disk. The per-trait group table is
    # cached and reused, keyed on the trait name.
    cached = f"{OUT}/{trait}.group.tsv"
    if os.path.exists(cached):
        g = pd.read_csv(cached, sep="\t", index_col=0)
        groups[trait] = g
        print(f"\n=== {trait}: reusing cached group analysis "
              f"({int((g['assoc_mcp'] < 0.05).sum())} of {len(g)} states at p<0.05)")
        continue
    print(f"\n=== {trait}   ({len(genes)} genes, {args.ntrl} control sets)", flush=True)
    # the control-score matrix is n_cells x n_ctrl — 118k x 1000 is ~0.9 GB per trait,
    # so it is written out and released rather than accumulated across traits
    df = scdrs.score_cell(adata, genes, gene_weight=weights,
                          ctrl_match_key="mean_var", n_ctrl=args.ntrl,
                          weight_opt="vs", return_ctrl_raw_score=False,
                          return_ctrl_norm_score=True, verbose=False)
    keep = [c for c in ("raw_score", "norm_score", "mc_pval", "pval", "nlog10_pval",
                        "zscore") if c in df.columns]
    df[keep].to_csv(f"{OUT}/{trait}.score.tsv", sep="\t")
    adata.obs[f"score_{trait}"] = df["norm_score"].reindex(adata.obs_names).to_numpy()
    g = scdrs.method.downstream_group_analysis(
        adata=adata, df_full_score=df, group_cols=[group_col])[group_col]
    g["trait"] = trait
    g.to_csv(cached, sep="\t")
    groups[trait] = g
    del df
    gc.collect()
    n_sig = int((g["assoc_mcp"] < 0.05).sum())
    print(f"    {n_sig} of {len(g)} cell states associated at p<0.05", flush=True)

if not groups:
    raise SystemExit("no trait produced scores")

allg = pd.concat(groups.values())
allg.index.name = "cell_state"
allg.reset_index().to_csv(f"{OUT}/group_analysis.tsv", sep="\t", index=False)

report = {}

# ---------------------------------------------------------------- positive control
print("\n" + "=" * 94)
print("POSITIVE CONTROL — does atrial fibrillation land on atrial cardiomyocytes?")
print("=" * 94)
pc_trait, pc_prefix = POSITIVE_CONTROL
ok_pc = False
if pc_trait in groups:
    g = groups[pc_trait]
    acm = g[[str(i).startswith(pc_prefix) for i in g.index]]
    print(f"{'cell state':<24}{'n':>7}{'assoc z':>10}{'assoc p':>11}{'hetero p':>11}")
    for i, r in acm.sort_values("assoc_mcp").iterrows():
        print(f"{i:<24}{int(sizes.get(i, 0)):>7,}{r['assoc_mcz']:>10.2f}"
              f"{r['assoc_mcp']:>11.4g}{r['hetero_mcp']:>11.4g}")
    ok_pc = bool((acm["assoc_mcp"] < 0.05).any())
    rank = (g.sort_values("assoc_mcz", ascending=False).index.tolist())
    top5 = rank[:5]
    print(f"\ntop 5 cell states for {pc_trait}: {', '.join(top5)}")
    print(f"=> positive control {'PASSES' if ok_pc else 'FAILS'}")
    report["positive_control"] = dict(passes=ok_pc, top5=top5)
else:
    print(f"{pc_trait} not scored — cannot validate the pipeline")

if not ok_pc:
    print("\n" + "!" * 94)
    print("The pipeline did not recover a result that must be true in this tissue.")
    print("Do not interpret the conduction-cell numbers below either way.")
    print("!" * 94)

# ---------------------------------------------------------------- the hypothesis
print("\n" + "=" * 94)
print("CONDUCTION-SYSTEM CELL STATES")
print("=" * 94)
traits = list(groups)
print(f"{'cell state':<20}{'n':>6}" + "".join(f"{t[:16]:>18}" for t in traits))
for cs in CONDUCTION:
    line = f"{cs:<20}{int(sizes.get(cs, 0)):>6,}"
    for t in traits:
        g = groups[t]
        if cs not in g.index:
            line += f"{'-':>18}"
            continue
        z, p = g.loc[cs, "assoc_mcz"], g.loc[cs, "assoc_mcp"]
        star = "**" if p < 0.01 else "*" if p < 0.05 else ""
        line += f"{z:>+13.2f}{star:<5}"
    flag = "   (n<50, unreliable)" if sizes.get(cs, 0) < 50 else ""
    print(line + flag)
print("\n(scDRS association z and Monte Carlo p; * p<0.05, ** p<0.01)")

report["conduction"] = {
    cs: {t: dict(z=float(groups[t].loc[cs, "assoc_mcz"]),
                 p=float(groups[t].loc[cs, "assoc_mcp"]),
                 n=int(sizes.get(cs, 0)))
         for t in traits if cs in groups[t].index}
    for cs in CONDUCTION if any(cs in groups[t].index for t in traits)}

# ---------------------------------------------------------------- ranking
print("\n" + "=" * 94)
print("WHERE DOES EACH TRAIT ACTUALLY LAND?  (top 5 cell states by association z)")
print("=" * 94)
for t in traits:
    g = groups[t].sort_values("assoc_mcz", ascending=False)
    items = [f"{i}({g.loc[i, 'assoc_mcz']:+.1f})" for i in g.index[:5]]
    san_rank = (list(g.index).index("SAN_P_cell") + 1
                if "SAN_P_cell" in g.index else None)
    print(f"{t:<24}{', '.join(items)}")
    if san_rank:
        print(f"{'':<24}  SAN_P_cell ranks {san_rank}/{len(g)}")
    report.setdefault("ranking", {})[t] = dict(
        top5=list(g.index[:5]),
        san_rank=san_rank, n_groups=len(g))

with open(f"{OUT}/scdrs_summary.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwrote {OUT}/group_analysis.tsv, per-trait scores, and scdrs_summary.json")
