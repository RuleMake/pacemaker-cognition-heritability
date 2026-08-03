"""
Redo the refuted test with the gene set the literature actually points at.
=========================================================================

Script 141 asked whether educational attainment lights up sinoatrial pacemaker cells
through a shared neuronal programme, defined the programme from the atlas's own
`neural cell` population, and refuted the hypothesis: deleting it cost 2.9% of the
score. That result stands, but it may have tested the wrong thing. The heart's neural
cells are overwhelmingly glia and Schwann cells — the top of that programme was PLP1,
PTPRZ1, NRXN1 — and a glial programme is not what a pacemaker cell would share.

The literature is specific about what it does share. Sinoatrial pacemaker cells
co-cluster with cortical neurons in integrated single-cell space and carry a
glutamatergic system: synthesis, ionotropic and metabotropic receptors, transporters
(Protein & Cell 2021, PMID 33548033). That is a named, fixed gene set, and it is the
one the earlier test should have used.

Three questions, in the order that makes the third interpretable:

  A  Do sinoatrial pacemaker cells express the glutamatergic programme above the
     working myocytes beside them — in this dataset, not in the mouse? If not, the
     hypothesis dies here and the rest is moot.
  B  How much of each trait's driver signal in SAN_P_cell sits in that programme?
  C  Delete it. Against a size- and weight-matched random null, how much does
     educational attainment lose, and how much does HRV lose? Separation requires the
     control to lose substantially MORE, and requires the loss to be a real share of
     the score rather than merely significant — the mistake 141 made the first time.

Usage:  python scripts/171_glutamatergic.py
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
GS = f"{ROOT}/data/scdrs/traits.gs"
OUT = f"{ROOT}/results/glutamatergic.json"

FOCUS = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
TRAITS = ["EducationalAttainment", "HRV_RMSSD", "HRV_SDNN", "RestingHeartRate",
          "PRinterval", "AtrialFibrillation", "RheumatoidArthritis"]
CONTROLS = {"EducationalAttainment", "RheumatoidArthritis"}
N_PERM = 200
MIN_CM = 30
SEED = 20260801

# Fixed from the literature before looking at anything, so it cannot be tuned.
GLUT = {
    # ionotropic receptors
    "GRIN1", "GRIN2A", "GRIN2B", "GRIN2C", "GRIN2D", "GRIN3A", "GRIN3B",
    "GRIA1", "GRIA2", "GRIA3", "GRIA4",
    "GRIK1", "GRIK2", "GRIK3", "GRIK4", "GRIK5",
    # metabotropic receptors
    "GRM1", "GRM2", "GRM3", "GRM4", "GRM5", "GRM6", "GRM7", "GRM8",
    # vesicular and plasma-membrane transporters
    "SLC17A6", "SLC17A7", "SLC17A8",
    "SLC1A1", "SLC1A2", "SLC1A3", "SLC1A6", "SLC1A7",
    # synthesis and recycling
    "GLS", "GLS2", "GLUL", "GOT1", "GOT2",
    # postsynaptic scaffold
    "DLG4", "SHANK1", "SHANK2", "SHANK3", "HOMER1", "HOMER2", "DLGAP1",
}

rng = np.random.default_rng(SEED)
a = ad.read_h5ad(H5)
state = a.obs["cell_state"].astype(str).to_numpy()
strat = (a.obs.donor_id.astype(str) + "/" + a.obs.region.astype(str) + "/"
         + a.obs.assay.astype(str)).to_numpy()
X = sp.csr_matrix(a.X) if not sp.issparse(a.X) else a.X.tocsr()
gidx = {g: i for i, g in enumerate(a.var_names)}
present = sorted(g for g in GLUT if g in gidx)
print(f"glutamatergic programme: {len(present)}/{len(GLUT)} genes present in the data")
print("  " + ", ".join(present))

is_cm = pd.Series(state).str.match(r"^(aCM|vCM)").to_numpy()

# ================================================================ A
print("\n" + "=" * 96)
print("A. IS THE PROGRAMME EXPRESSED IN PACEMAKER CELLS, ABOVE THE MYOCYTES?")
print("=" * 96)
ii = [gidx[g] for g in present]
prog = np.asarray(X[:, ii].mean(axis=1)).ravel()


def stratified_auc(values, focus_mask, comparator_mask):
    """van Elteren stratified AUC of focus vs comparator, within donor/region/assay."""
    num = den = stat = var = 0.0
    for s in pd.unique(strat):
        m = strat == s
        f = m & focus_mask
        o = m & comparator_mask & ~focus_mask
        n1, n2 = int(f.sum()), int(o.sum())
        if n1 == 0 or n2 < MIN_CM:
            continue
        r = stats.rankdata(np.concatenate([values[f], values[o]]))
        W = r[:n1].sum()
        N = n1 + n2
        auc = (W - n1 * (n1 + 1) / 2) / (n1 * n2)
        w = n1 * n2 / (N + 1)
        num += w * auc
        den += w
        stat += (W - n1 * (N + 1) / 2) / (N + 1)
        var += n1 * n2 / (12.0 * (N + 1))
    if den == 0 or var <= 0:
        return None
    return num / den, stat / np.sqrt(var)


resA = {}
print(f"{'cell state':<20}{'n':>6}{'AUC vs myocytes':>20}{'z':>10}")
for c in FOCUS:
    m = state == c
    r = stratified_auc(prog, m, is_cm)
    if r is None:
        continue
    resA[c] = dict(auc=float(r[0]), z=float(r[1]))
    print(f"{c:<20}{int(m.sum()):>6}{r[0]:>20.3f}{r[1]:>10.1f}")
san_ok = resA.get("SAN_P_cell", {}).get("auc", 0) > 0.5
print("\n" + ("Sinoatrial pacemaker cells DO carry the programme above the myocytes."
              if san_ok else
              "They do NOT. The hypothesis fails at the first step and B/C are moot."))

# ================================================================ B + C
print("\n" + "=" * 96)
print("B+C. HOW MUCH OF EACH TRAIT RIDES ON IT, AND WHAT DOES DELETING IT COST?")
print("=" * 96)
gs = pd.read_csv(GS, sep="\t").set_index("TRAIT")


def score_auc(pairs, cell="SAN_P_cell"):
    """Weighted-expression score for a gene set, then stratified AUC vs myocytes."""
    idx = [gidx[g] for g, _ in pairs]
    w = np.array([v for _, v in pairs], dtype=float)
    v = np.asarray(X[:, idx] @ (w / w.sum())).ravel()
    r = stratified_auc(v, state == cell, is_cm)
    return None if r is None else r[0]


print(f"{'trait':<26}{'in set':>8}{'full AUC':>11}{'-glut':>9}{'delta':>9}"
      f"{'% of excess':>13}{'p':>8}")
resC = {}
for t in TRAITS:
    if t not in gs.index:
        continue
    pairs = [p.split(":") for p in str(gs.loc[t, "GENESET"]).split(",") if ":" in p]
    pairs = [(s, float(v)) for s, v in pairs if s in gidx]
    inset = [g for g, _ in pairs if g in GLUT]
    a0 = score_auc(pairs)
    kept = [(g, v) for g, v in pairs if g not in GLUT]
    if a0 is None or len(kept) < 50 or not inset:
        print(f"{t:<26}{len(inset):>8}   nothing to delete")
        continue
    a1 = score_auc(kept)
    # matched null: drop the same number of genes sampled to match the weight
    # distribution of the ones actually dropped
    allw = np.array([v for _, v in pairs])
    dw = np.array([v for g, v in pairs if g in GLUT])
    bins = np.quantile(allw, np.linspace(0, 1, 6))
    want = np.histogram(dw, bins=bins)[0]
    pool = [np.where((allw >= bins[i]) & (allw <= bins[i + 1]))[0] for i in range(5)]
    null = []
    for _ in range(N_PERM):
        pick = set()
        for i, k in enumerate(want):
            if k and len(pool[i]):
                pick |= set(rng.choice(pool[i], size=min(k, len(pool[i])),
                                       replace=False))
        keep2 = [p for j, p in enumerate(pairs) if j not in pick]
        if len(keep2) >= 50:
            v = score_auc(keep2)
            if v is not None:
                null.append(v)
    null = np.array(null)
    p = float((null <= a1).mean()) if len(null) else np.nan
    d = a1 - a0
    # express the loss against how far above 0.5 the trait was, so a trait sitting at
    # 0.51 cannot look impressive by losing 0.005
    excess = max(a0 - 0.5, 1e-6)
    resC[t] = dict(n_in_set=len(inset), auc=float(a0), auc_dropped=float(a1),
                   delta=float(d), frac_of_excess=float(abs(d) / excess), p=p)
    tag = "   <- control" if t in CONTROLS else ""
    print(f"{t:<26}{len(inset):>8}{a0:>11.3f}{a1:>9.3f}{d:>+9.3f}"
          f"{100 * abs(d) / excess:>12.1f}%{p:>8.3f}{tag}")

# ================================================================ verdict
print("\n" + "=" * 96)
print("VERDICT")
print("=" * 96)
edu = resC.get("EducationalAttainment", {})
hrv = max((resC.get(t, {}) for t in ("HRV_RMSSD", "HRV_SDNN")),
          key=lambda d: d.get("frac_of_excess", 0), default={})


def carries(d, frac=0.10):
    return bool(d) and d.get("p", 1) < 0.05 and d.get("frac_of_excess", 0) >= frac


sep = san_ok and carries(edu) and not carries(hrv)
print(f"  programme present in SAN_P_cell above myocytes : {san_ok}")
print(f"  deleting it costs educational attainment       : "
      f"{edu.get('frac_of_excess', float('nan')) * 100:.1f}% of its excess "
      f"(p={edu.get('p', float('nan')):.3f})")
print(f"  deleting it costs HRV                          : "
      f"{hrv.get('frac_of_excess', float('nan')) * 100:.1f}% of its excess "
      f"(p={hrv.get('p', float('nan')):.3f})")
print()
print("SUPPORTED: the cognitive signal in pacemaker cells runs through the "
      "glutamatergic\nprogramme and the cardiac signal does not." if sep else
      "NOT SUPPORTED on this test. Report the overlap as measured; do not rescue the\n"
      "hypothesis a second time by picking a third gene set.")

with open(OUT, "w") as f:
    json.dump(dict(genes=present, expression=resA, knockout=resC,
                   supported=bool(sep)), f, indent=2, default=float)
print(f"\nwrote {OUT}")
