"""
Cross-species replication: the same question, one flag changed.
===============================================================

The human pipeline is `scdrs.preprocess` then `scdrs.score_cell` on gene sets loaded
with `src_species="human", dst_species="human"`. This script is that pipeline with
`dst_species="mouse"`. Nothing else about the statistic changes, which is the point —
if the mouse and human answers differ, it is the biology or the data that differs, not
the analysis.

The comparator is the same construction as well: nodal cells against the working
cardiomyocytes dissected out of the SAME zone, van Elteren weighted across zones. In
the human data the stratum was donor/region/assay; here it is the microdissection zone,
which is the only batch structure this dataset has.

Predictions, written to disk before any score is computed
---------------------------------------------------------
  P1  positive control. Resting heart rate and heart rate variability enrich in mouse
      nodal cells (AUC > 0.55). This is the gate: mouse is embryonic, a third of human
      genes have no mouse orthologue, and if the best-established cardiac fact does not
      survive that, nothing measured here means anything and P2 is not interpretable
      whichever way it comes out.

  P2  the test. Educational attainment, intelligence and reaction time enrich in mouse
      nodal cells (AUC > 0.55). Failure here, with P1 passing, is a genuine failure to
      replicate and will be reported as one.

  P3  negative control. Rheumatoid arthritis is flat (0.45-0.55).

  P4  signed control. QT interval does not exceed the working myocytes.

The gate in P1 is the part that makes this honest. Without it, a null in P2 could
always be waved away as "mouse is different", and a positive could always be claimed as
replication — the same test cannot be allowed to mean both things.

Why 200 control gene sets here and 1000 in the human arm
--------------------------------------------------------
The human run used 1000. This one uses 200, for a practical reason and a statistical
one. Practical: the WSL virtual machine shuts down when its invoking session ends, so
this has to finish inside short foreground calls, and the script is written to resume
from whatever it has already cached. Statistical: the control sets calibrate each
cell's normalised score against expression-matched null gene sets, and the quantity
compared here is a rank statistic between cells that all received the SAME calibration.
Imprecision in the null is shared by focus and comparator cells and largely cancels in
the AUC. scDRS's own worked examples use 20 control sets for cell-type-level questions;
200 is an order of magnitude more than that and an order of magnitude less than needed
for reliable PER-CELL p-values, which this arm does not use. Per-cell p-values from the
mouse run should not be quoted.

Usage:  python scripts/211_mouse_scdrs.py --ntrl 200   # resumes from cached traits
"""

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scdrs
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--ntrl", type=int, default=1000)
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent.parent.as_posix()
H5 = f"{ROOT}/data/mouse/mouse_ccs.h5ad"
GS = f"{ROOT}/data/scdrs/traits.gs"
OUT = f"{ROOT}/results/scdrs_mouse"
PRED = f"{ROOT}/results/mouse_predictions.json"
RES = f"{ROOT}/results/mouse_replication.json"
os.makedirs(OUT, exist_ok=True)

TRAITS = ["RestingHeartRate", "HRV_SDNN", "HRV_RMSSD",
          "EducationalAttainment", "Intelligence", "ReactionTime",
          "RheumatoidArthritis", "QTinterval"]
GATE = ["RestingHeartRate", "HRV_SDNN", "HRV_RMSSD"]
TEST = ["EducationalAttainment", "Intelligence", "ReactionTime"]
FOCUS = "nodal"
COMPARATOR = "working_CM"
BAR = 0.55

# ---------------------------------------------------------------- register first
if not os.path.exists(PRED):
    json.dump(dict(
        P1_gate=dict(traits=GATE, claim=f"AUC > {BAR} vs {COMPARATOR}",
                     falsifies="whole cross-species test uninterpretable"),
        P2_test=dict(traits=TEST, claim=f"AUC > {BAR} vs {COMPARATOR}",
                     falsifies="human cognitive result does not replicate in mouse"),
        P3_negative=dict(traits=["RheumatoidArthritis"], claim="0.45 < AUC < 0.55"),
        P4_signed=dict(traits=["QTinterval"], claim="AUC <= 0.55"),
        comparator=COMPARATOR, focus=FOCUS, stratum="zone", bar=BAR,
    ), open(PRED, "w"), indent=2)
    print(f"registered predictions -> {PRED}")
else:
    print(f"predictions already registered in {PRED} (not overwritten)")

# ---------------------------------------------------------------- inputs
adata = ad.read_h5ad(H5)
print(f"cells {adata.n_obs:,}   genes {adata.n_vars:,}")
print(adata.obs.cell_class.value_counts().to_string())

cov = pd.DataFrame(index=adata.obs_names)
cov["const"] = 1
cov["n_genes"] = np.log1p(adata.obs["n_genes_by_counts"].to_numpy())
cov["pct_mt"] = adata.obs["pct_counts_mt"].to_numpy()
for s in sorted(adata.obs["sample"].unique())[1:]:
    cov[f"sample_{s}"] = (adata.obs["sample"] == s).astype(float).to_numpy()
print(f"covariates: {list(cov.columns)}")

print("\npreprocessing...")
scdrs.preprocess(adata, cov=cov, n_mean_bin=20, n_var_bin=20, copy=False)

# the one line that differs from the human run
gs = scdrs.util.load_gs(GS, src_species="human", dst_species="mouse",
                        to_intersect=adata.var_names)
gs = {t: v for t, v in gs.items() if t in TRAITS}
print(f"\ngene sets after human -> mouse orthologue mapping:")
for t in TRAITS:
    if t in gs:
        print(f"  {t:<24}{len(gs[t][0]):>5} mouse genes present")
    else:
        print(f"  {t:<24}    - not in traits.gs")

# ---------------------------------------------------------------- score
for trait, (genes, weights) in gs.items():
    dest = f"{OUT}/{trait}.score.tsv"
    if os.path.exists(dest):
        print(f"=== {trait}: cached")
        continue
    if len(genes) < 100:
        print(f"!! {trait}: only {len(genes)} orthologues map — skipped")
        continue
    print(f"=== {trait}  ({len(genes)} genes, {args.ntrl} control sets)", flush=True)
    df = scdrs.score_cell(adata, genes, gene_weight=weights,
                          ctrl_match_key="mean_var", n_ctrl=args.ntrl,
                          weight_opt="vs", return_ctrl_raw_score=False,
                          return_ctrl_norm_score=True, verbose=False)
    keep = [c for c in ("raw_score", "norm_score", "mc_pval", "pval", "zscore")
            if c in df.columns]
    df[keep].to_csv(dest, sep="\t")
    del df

# ---------------------------------------------------------------- analyse
obs = adata.obs[["zone", "sample", "cell_class"]].astype(str)
cls = obs.cell_class.to_numpy()
zcode, znames = pd.factorize(obs.zone)
IS_F = cls == FOCUS
IS_C = cls == COMPARATOR
KEEP = IS_F | IS_C
MIN_C = 30


def van_elteren(v):
    """Same statistic as the human arm; stratum is the microdissection zone."""
    num = den = stat = var = 0.0
    ns = nc = 0
    per = {}
    for si in range(len(znames)):
        m = (zcode == si) & KEEP
        N = int(m.sum())
        if N < MIN_C:
            continue
        f = IS_F[m]
        n1 = int(f.sum())
        n2 = N - n1
        if n1 < 3 or n2 < MIN_C:
            continue
        r = stats.rankdata(v[m])
        W = r[f].sum()
        auc_i = (W - n1 * (n1 + 1) / 2) / (n1 * n2)
        w = n1 * n2 / (N + 1)
        num += w * auc_i
        den += w
        stat += (W - n1 * (N + 1) / 2) / (N + 1)
        var += n1 * n2 / (12.0 * (N + 1))
        ns += 1
        nc += n1
        per[znames[si]] = dict(auc=float(auc_i), n_focus=n1, n_comp=n2)
    if den <= 0 or var <= 0:
        return None
    return dict(auc=float(num / den), z=float(stat / np.sqrt(var)),
                n_zones=ns, n_cells=nc, per_zone=per)


print("\n" + "=" * 96)
print(f"MOUSE: {FOCUS} cells vs {COMPARATOR} in the same dissection zone")
print("=" * 96)
print(f"{'trait':<26}{'AUC':>8}{'z':>8}{'zones':>7}{'cells':>7}   per-zone AUC")
res = {}
for trait in TRAITS:
    p = f"{OUT}/{trait}.score.tsv"
    if not os.path.exists(p):
        continue
    v = pd.read_csv(p, sep="\t", index_col=0)["norm_score"] \
          .reindex(obs.index).to_numpy()
    r = van_elteren(v)
    if r is None:
        continue
    res[trait] = r
    pz = "  ".join(f"{k[:4]} {d['auc']:.2f}" for k, d in r["per_zone"].items())
    print(f"{trait:<26}{r['auc']:>8.3f}{r['z']:>8.2f}{r['n_zones']:>7}"
          f"{r['n_cells']:>7}   {pz}")

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 96)
print("VERDICT AGAINST THE REGISTERED PREDICTIONS")
print("=" * 96)
gate_ok = [t for t in GATE if t in res and res[t]["auc"] > BAR]
test_ok = [t for t in TEST if t in res and res[t]["auc"] > BAR]
neg = res.get("RheumatoidArthritis", {}).get("auc")
qt = res.get("QTinterval", {}).get("auc")

print(f"  P1 gate      {len(gate_ok)}/{len([t for t in GATE if t in res])} "
      f"cardiac traits above {BAR}   {gate_ok}")
print(f"  P2 test      {len(test_ok)}/{len([t for t in TEST if t in res])} "
      f"cognitive traits above {BAR}   {test_ok}")
if neg is not None:
    print(f"  P3 negative  rheumatoid arthritis {neg:.3f} "
          f"({'flat as predicted' if 0.45 < neg < 0.55 else 'NOT FLAT'})")
if qt is not None:
    print(f"  P4 signed    QT interval {qt:.3f} "
          f"({'does not exceed myocytes' if qt <= BAR else 'EXCEEDS myocytes'})")

print()
if not gate_ok:
    verdict = "uninterpretable"
    print("  The positive control failed. Species distance, developmental stage or")
    print("  orthologue loss has cost too much signal, and NOTHING here — positive or")
    print("  negative — can be read as evidence about the human result.")
elif test_ok:
    verdict = "replicated"
    print("  The cognitive enrichment is present in mouse pacemaker cells, in a")
    print("  different species, laboratory, developmental stage and protocol. No")
    print("  artefact of the human atlas can produce that.")
else:
    verdict = "failed to replicate"
    print("  The cardiac controls survive the species change and the cognitive traits")
    print("  do not. That is a real failure to replicate and must be reported as one:")
    print("  it means the human signal is either primate-specific or an artefact of")
    print("  the human dataset that this design cannot distinguish.")

json.dump(dict(verdict=verdict, results=res, gate_ok=gate_ok, test_ok=test_ok,
               bar=BAR, n_ctrl=args.ntrl,
               gene_counts={t: len(gs[t][0]) for t in gs}),
          open(RES, "w"), indent=2)
print(f"\nwrote {RES}")
