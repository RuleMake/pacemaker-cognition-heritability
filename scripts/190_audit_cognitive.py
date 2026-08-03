"""
Self-audit of the cognitive arm. Four things that were claimed but not checked.
==============================================================================

  A  The one POSITIVE claim from the glutamatergic test has no null. Part A reported
     AUC 0.608 for the programme in SAN_P_cell against the myocytes and called it
     present. But the same script's own table shows rheumatoid arthritis — a gene set
     with no business in the heart — scoring 0.582 with the same statistic. If any
     gene set scores ~0.58 in conduction cells, then 0.608 is not evidence of
     anything. A size-matched random-gene null decides it.

  B  "Intelligence is more specific than educational attainment" was read off the
     GLOBAL scDRS ranking (#44 and #43 for bundle and Purkinje). That ranking is
     assay-confounded on this subset — a fact this project established itself and then
     used the confounded numbers anyway. Redone on the myocyte-referenced statistic.

  C  AMBIENT NEURAL RNA, never tested and the most serious untested alternative.
     Sinoatrial tissue is the most densely innervated region sampled, and the
     cognitive traits hit SAN_P_cell first and glial cells immediately after. If
     pacemaker nuclei carry ambient transcripts from the nerves around them, a
     cognitive gene set would score highly there for a purely technical reason.
     Three checks: do pacemaker cells carry neural markers above other myocytes; does
     a cell's cognitive score track its neural-marker load; are they doublet-flagged.

  D  Doublet scores for the focus populations, since a pacemaker/glia doublet would
     produce exactly the observed pattern.

Usage:  python scripts/190_audit_cognitive.py
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.as_posix()
H5 = f"{ROOT}/data/singlecell/axis_subset.h5ad"
SC = f"{ROOT}/results/scdrs_axis"
OUT = f"{ROOT}/results/audit_cognitive.json"

FOCUS = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
COG = ["EducationalAttainment", "Intelligence", "ReactionTime"]
CARD = ["HRV_RMSSD", "HRV_SDNN", "RestingHeartRate"]
MIN_CM = 30
N_NULL = 500
SEED = 20260802

# markers that should be absent from a clean cardiomyocyte nucleus
NEURAL = ["PLP1", "MPZ", "S100B", "NRXN1", "PTPRZ1", "SOX10", "MBP", "CDH19",
          "NGFR", "L1CAM", "GFRA3", "SCN7A"]
GLUT = ["GRIN1", "GRIN2A", "GRIN2B", "GRIA1", "GRIA2", "GRIA3", "GRIA4", "GRIK2",
        "GRIK5", "GRM1", "GRM3", "GRM5", "GRM7", "GRM8", "SLC17A6", "SLC17A7",
        "SLC1A1", "SLC1A2", "SLC1A3", "GLS", "GLUL", "GOT1", "GOT2", "DLG4",
        "SHANK2", "SHANK3", "HOMER1", "DLGAP1"]

rng = np.random.default_rng(SEED)
a = ad.read_h5ad(H5)
state = a.obs["cell_state"].astype(str).to_numpy()
strat = (a.obs.donor_id.astype(str) + "/" + a.obs.region.astype(str) + "/"
         + a.obs.assay.astype(str)).to_numpy()
X = sp.csr_matrix(a.X) if not sp.issparse(a.X) else a.X.tocsr()
gidx = {g: i for i, g in enumerate(a.var_names)}
is_cm = pd.Series(state).str.match(r"^(aCM|vCM)").to_numpy()
findings = {}


def strat_auc(values, focus, comparator):
    num = den = stat = var = 0.0
    for s in pd.unique(strat):
        m = strat == s
        f = m & focus
        o = m & comparator & ~focus
        n1, n2 = int(f.sum()), int(o.sum())
        if n1 == 0 or n2 < MIN_CM:
            continue
        r = stats.rankdata(np.concatenate([values[f], values[o]]))
        W, N = r[:n1].sum(), n1 + n2
        num += (n1 * n2 / (N + 1)) * ((W - n1 * (n1 + 1) / 2) / (n1 * n2))
        den += n1 * n2 / (N + 1)
        stat += (W - n1 * (N + 1) / 2) / (N + 1)
        var += n1 * n2 / (12.0 * (N + 1))
    return (num / den, stat / np.sqrt(var)) if den > 0 and var > 0 else (None, None)


def mean_of(genes):
    ii = [gidx[g] for g in genes if g in gidx]
    return np.asarray(X[:, ii].mean(axis=1)).ravel() if ii else None


# ================================================================ A
print("=" * 96)
print("A. IS THE GLUTAMATERGIC PROGRAMME ABOVE A RANDOM GENE SET OF THE SAME SIZE?")
print("=" * 96)
present = [g for g in GLUT if g in gidx]
obs_auc, _ = strat_auc(mean_of(present), state == "SAN_P_cell", is_cm)
# match the null on expression level, not just count: rare genes and abundant genes
# behave differently, and a null of randomly chosen genes would be an easier target
mu = np.asarray(X.mean(axis=0)).ravel()
tgt = np.array([mu[gidx[g]] for g in present])
order = np.argsort(mu)
rank_of = np.empty(len(mu), dtype=int)
rank_of[order] = np.arange(len(mu))
null = []
for _ in range(N_NULL):
    pick = []
    for t in tgt:
        r = rank_of[np.searchsorted(mu[order], t)] if t > 0 else 0
        lo, hi = max(0, r - 250), min(len(mu) - 1, r + 250)
        pick.append(order[rng.integers(lo, hi + 1)])
    v, _ = strat_auc(np.asarray(X[:, pick].mean(axis=1)).ravel(),
                     state == "SAN_P_cell", is_cm)
    if v is not None:
        null.append(v)
null = np.array(null)
p = float((null >= obs_auc).mean())
findings["glutamatergic_vs_null"] = dict(observed=float(obs_auc),
                                         null_mean=float(null.mean()),
                                         null_p95=float(np.percentile(null, 95)),
                                         p=p, n_null=len(null))
print(f"  observed AUC (glutamatergic, SAN_P_cell vs myocytes) : {obs_auc:.3f}")
print(f"  expression-matched random sets ({len(null)}): mean {null.mean():.3f}, "
      f"95th pct {np.percentile(null, 95):.3f}")
print(f"  p = {p:.3f}")
print("\n  " + ("The programme IS above what a comparable random set gives."
                if p < 0.05 else
                "NOT above a matched random set. The one positive claim from script "
                "171\n  does not survive its own null and must be withdrawn."))

# ================================================================ B
print("\n" + "=" * 96)
print("B. COGNITIVE TRAITS ON THE MYOCYTE-REFERENCED STATISTIC (not the global rank)")
print("=" * 96)
scores = {}
for t in COG + CARD:
    p_ = Path(f"{SC}/{t}.score.tsv")
    if p_.exists():
        scores[t] = pd.read_csv(p_, sep="\t", index_col=0)["norm_score"].reindex(
            a.obs_names).to_numpy()
print(f"{'trait':<26}" + "".join(f"{c[:16]:>19}" for c in FOCUS))
resB = {}
for t in COG + CARD:
    if t not in scores:
        continue
    line = f"{t:<26}"
    resB[t] = {}
    for c in FOCUS:
        auc, z = strat_auc(scores[t], state == c, is_cm)
        if auc is None:
            line += f"{'-':>19}"
            continue
        resB[t][c] = dict(auc=float(auc), z=float(z))
        line += f"{f'{auc:.3f} (z{z:+.1f})':>19}"
    print(line)
findings["cognitive_cm_referenced"] = resB

# ================================================================ C
print("\n" + "=" * 96)
print("C. AMBIENT NEURAL RNA — do pacemaker cells carry nerve transcripts?")
print("=" * 96)
neu = mean_of(NEURAL)
print(f"{'cell state':<20}{'neural-marker AUC vs myocytes':>32}{'z':>8}")
resC = {}
for c in FOCUS:
    auc, z = strat_auc(neu, state == c, is_cm)
    if auc is None:
        continue
    resC[c] = dict(auc=float(auc), z=float(z))
    print(f"{c:<20}{auc:>32.3f}{z:>8.1f}")

print("\n  within SAN_P_cell: does a cell's cognitive score track its neural load?")
m = state == "SAN_P_cell"
resC["within_SAN"] = {}
for t in COG + CARD:
    if t not in scores:
        continue
    r = stats.spearmanr(scores[t][m], neu[m])
    resC["within_SAN"][t] = dict(rho=float(r.statistic), p=float(r.pvalue))
    print(f"    {t:<26}rho = {r.statistic:+.3f}   p = {r.pvalue:.3f}")
findings["ambient_neural"] = resC

# ================================================================ D
print("\n" + "=" * 96)
print("D. DOUBLET SCORES")
print("=" * 96)
if "scrublet_score" in a.obs.columns:
    sc = pd.to_numeric(a.obs["scrublet_score"], errors="coerce").to_numpy()
    print(f"{'group':<22}{'median scrublet':>18}{'90th pct':>12}")
    resD = {}
    for c in FOCUS + ["working myocytes"]:
        mm = is_cm if c == "working myocytes" else (state == c)
        resD[c] = dict(median=float(np.nanmedian(sc[mm])),
                       p90=float(np.nanpercentile(sc[mm], 90)))
        print(f"{c:<22}{np.nanmedian(sc[mm]):>18.4f}"
              f"{np.nanpercentile(sc[mm], 90):>12.4f}")
    findings["doublets"] = resD
else:
    print("  no scrublet_score column")

with open(OUT, "w") as f:
    json.dump(findings, f, indent=2, default=float)
print(f"\nwrote {OUT}")
