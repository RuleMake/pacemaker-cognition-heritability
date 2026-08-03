"""
The mouse replicated. Before believing it, check the two ways it could be hollow.
================================================================================

The registered test passed: cardiac gate 3/3, cognitive 3/3, immune control flat. But
two things in the output need explaining before that means what it looks like it means.

  1  every AUC is larger than its human counterpart. Resting heart rate is 0.810 in
     mouse against 0.583 in human; heart rate variability 0.856 against 0.666. If the
     mouse contrast is simply coarser, the cognitive numbers are inflated by the same
     factor and cross-species EFFECT SIZES cannot be compared — only the pattern can.

  2  QT interval reversed. In human it is significantly DEPLETED in nodal cells
     (0.354, z = -5.6), which is the cleanest specificity result the project has. In
     mouse it is enriched (0.582). A registered prediction failed and that has to be
     reported, not explained away.

And one hypothesis that would make the whole thing circular: if the human-to-mouse
orthologue mapping collapsed the cognitive and cardiac gene sets into near-duplicates,
then "cognitive traits enrich in mouse pacemaker cells" is just the cardiac result
wearing a different label. That is testable three ways, and all three are run here:

  overlap      how many genes do the mapped sets share, in mouse and in human
  correlation  do cells' cognitive and cardiac scores move together
  subtraction  rescore the cognitive traits after deleting every gene that appears in
               any cardiac set. If the enrichment survives on the disjoint remainder,
               it is not the cardiac signal relabelled

The subtraction is the one that decides it. The other two only say how worried to be.

Usage:  python scripts/212_mouse_audit.py
"""

import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scdrs
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.as_posix()
H5 = f"{ROOT}/data/mouse/mouse_ccs.h5ad"
GS = f"{ROOT}/data/scdrs/traits.gs"
SC_M = f"{ROOT}/results/scdrs_mouse"
SC_H = f"{ROOT}/results/scdrs_axis"
H5_H = f"{ROOT}/data/singlecell/axis_subset.h5ad"
OUT = f"{ROOT}/results/mouse_audit.json"

COG = ["EducationalAttainment", "Intelligence", "ReactionTime"]
CARD = ["RestingHeartRate", "HRV_SDNN", "HRV_RMSSD"]
N_CTRL = 200

# ---------------------------------------------------------------- 1. overlap
print("=" * 92)
print("1. DID THE ORTHOLOGUE MAPPING COLLAPSE THE GENE SETS INTO EACH OTHER?")
print("=" * 92)
raw = {}
for line in Path(GS).read_text().splitlines()[1:]:
    p = line.split("\t")
    if len(p) >= 2:
        raw[p[0]] = {g.split(":")[0] for g in p[1].split(",")}

adata = ad.read_h5ad(H5)
gs_m = scdrs.util.load_gs(GS, src_species="human", dst_species="mouse",
                          to_intersect=adata.var_names)
mouse_sets = {t: set(v[0]) for t, v in gs_m.items()}

print(f"{'pair':<46}{'human |A&B|':>13}{'mouse |A&B|':>13}{'Jaccard':>10}")
ov = {}
for c in COG:
    for k in CARD:
        if c not in mouse_sets or k not in mouse_sets:
            continue
        hi = len(raw[c] & raw[k])
        mi = len(mouse_sets[c] & mouse_sets[k])
        j = mi / len(mouse_sets[c] | mouse_sets[k])
        ov[f"{c}|{k}"] = dict(human=hi, mouse=mi, jaccard=float(j))
        print(f"{c + ' x ' + k:<46}{hi:>13}{mi:>13}{j:>10.3f}")
mj = np.mean([v["jaccard"] for v in ov.values()])
print(f"\n  mean Jaccard {mj:.3f} — "
      + ("sets are near-duplicates, the result may be circular"
         if mj > 0.25 else "sets are largely disjoint"))

# ---------------------------------------------------------------- 2. correlation
print("\n" + "=" * 92)
print("2. DO THE COGNITIVE AND CARDIAC SCORES MOVE TOGETHER, CELL BY CELL?")
print("=" * 92)


def load(folder, names, index):
    out = {}
    for t in names:
        p = Path(folder) / f"{t}.score.tsv"
        if p.exists():
            out[t] = pd.read_csv(p, sep="\t", index_col=0)["norm_score"] \
                       .reindex(index).to_numpy()
    return out


obs_m = adata.obs[["zone", "cell_class"]].astype(str)
sm = load(SC_M, COG + CARD, obs_m.index)
focus_m = (obs_m.cell_class == "nodal").to_numpy()

ah = ad.read_h5ad(H5_H, backed="r")
obs_h = ah.obs[["cell_state"]].astype(str)
sh = load(SC_H, COG + CARD, obs_h.index)
focus_h = np.isin(obs_h.cell_state.to_numpy(), ["SAN_P_cell", "AVN_P_cell"])

print(f"{'pair':<46}{'mouse rho':>12}{'human rho':>12}   (within pacemaker cells)")
cor = {}
for c in COG:
    for k in CARD:
        if c in sm and k in sm and c in sh and k in sh:
            rm = stats.spearmanr(sm[c][focus_m], sm[k][focus_m]).statistic
            rh = stats.spearmanr(sh[c][focus_h], sh[k][focus_h]).statistic
            cor[f"{c}|{k}"] = dict(mouse=float(rm), human=float(rh))
            print(f"{c + ' x ' + k:<46}{rm:>12.3f}{rh:>12.3f}")

# ---------------------------------------------------------------- 3. subtraction
print("\n" + "=" * 92)
print("3. THE DECIDING TEST: rescore the cognitive traits on the DISJOINT remainder")
print("=" * 92)
card_union = set().union(*[mouse_sets[k] for k in CARD if k in mouse_sets])
print(f"  union of the three cardiac gene sets: {len(card_union)} mouse genes")

cls = obs_m.cell_class.to_numpy()
zcode, znames = pd.factorize(obs_m.zone)
IS_F = cls == "nodal"
IS_C = cls == "working_CM"
KEEP = IS_F | IS_C
MIN_C = 30


def van_elteren(v):
    num = den = stat = var = 0.0
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
        num += (n1 * n2 / (N + 1)) * ((W - n1 * (n1 + 1) / 2) / (n1 * n2))
        den += n1 * n2 / (N + 1)
        stat += (W - n1 * (N + 1) / 2) / (N + 1)
        var += n1 * n2 / (12.0 * (N + 1))
    if den <= 0 or var <= 0:
        return None
    return float(num / den), float(stat / np.sqrt(var))


print("\npreprocessing for rescoring...")
cov = pd.DataFrame(index=adata.obs_names)
cov["const"] = 1
cov["n_genes"] = np.log1p(adata.obs["n_genes_by_counts"].to_numpy())
cov["pct_mt"] = adata.obs["pct_counts_mt"].to_numpy()
for s in sorted(adata.obs["sample"].unique())[1:]:
    cov[f"sample_{s}"] = (adata.obs["sample"] == s).astype(float).to_numpy()
scdrs.preprocess(adata, cov=cov, n_mean_bin=20, n_var_bin=20, copy=False)

print(f"\n{'trait':<26}{'full set':>10}{'z':>8}{'disjoint':>11}{'z':>8}"
      f"{'n genes kept':>14}{'verdict':>14}")
sub = {}
for t in COG:
    if t not in gs_m:
        continue
    genes, weights = gs_m[t]
    gw = {g: w for g, w in zip(genes, weights)}
    keep = [g for g in genes if g not in card_union]
    dest = f"{SC_M}/{t}_disjoint.score.tsv"
    if os.path.exists(dest):
        v = pd.read_csv(dest, sep="\t", index_col=0)["norm_score"] \
              .reindex(obs_m.index).to_numpy()
    else:
        df = scdrs.score_cell(adata, keep, gene_weight=[gw[g] for g in keep],
                              ctrl_match_key="mean_var", n_ctrl=N_CTRL,
                              weight_opt="vs", return_ctrl_raw_score=False,
                              return_ctrl_norm_score=True, verbose=False)
        df[["raw_score", "norm_score"]].to_csv(dest, sep="\t")
        v = df["norm_score"].reindex(obs_m.index).to_numpy()
    a_full, z_full = van_elteren(sm[t])
    a_dis, z_dis = van_elteren(v)
    ok = a_dis > 0.55
    sub[t] = dict(full_auc=a_full, full_z=z_full, disjoint_auc=a_dis,
                  disjoint_z=z_dis, n_genes=len(keep), n_removed=len(genes) - len(keep),
                  survives=bool(ok))
    print(f"{t:<26}{a_full:>10.3f}{z_full:>8.2f}{a_dis:>11.3f}{z_dis:>8.2f}"
          f"{len(keep):>14}{('survives' if ok else 'COLLAPSES'):>14}")

print()
if all(v["survives"] for v in sub.values()):
    print("  The cognitive enrichment survives on genes that appear in NO cardiac set.")
    print("  It is not the cardiac signal relabelled by orthologue mapping.")
else:
    print("  At least one cognitive trait loses its enrichment once the genes it shares")
    print("  with the cardiac sets are removed. That trait's mouse result is not")
    print("  independent evidence and must not be reported as replication.")

json.dump(dict(overlap=ov, mean_jaccard=float(mj), correlation=cor,
               subtraction=sub, n_ctrl=N_CTRL,
               cardiac_union_size=len(card_union)), open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
