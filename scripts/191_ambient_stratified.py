"""
Does the cognitive signal survive in pacemaker cells that carry no nerve transcripts?
====================================================================================

The audit found the signature of ambient neural RNA, and found it precisely where it
would do the most damage:

  * SAN_P_cell carries neural markers above the myocytes around it (AUC 0.584, z+4.0)
    while AVN_P_cell, AVN_bundle_cell and Purkinje do not (0.467, 0.499, 0.504).
    Sinoatrial tissue is the most densely innervated region sampled.
  * Within those 245 cells, a cell's cognitive score tracks its neural-marker load —
    educational attainment rho +0.23, intelligence +0.15, reaction time +0.13 — while
    the cardiac traits show nothing (HRV_RMSSD +0.06, HRV_SDNN +0.01).

That is what contamination looks like, and it is trait-specific in exactly the
direction that would manufacture the finding. Doublet scores do not exonerate it:
scrublet detects two nuclei captured together, not free-floating transcripts from
surrounding nerve, and SAN_P_cell's score (0.102) is identical to the myocytes'.

The decisive test is simple. Split the pacemaker cells by how much nerve transcript
they carry and ask whether the cleanest half still shows the enrichment. If the signal
lives in the contaminated cells it collapses; if it is a property of pacemaker
identity it survives.

The cardiac traits run through the same split as the internal control — they should be
unaffected either way, and if they are not, the split itself is doing something
strange and neither result means anything.

Usage:  python scripts/191_ambient_stratified.py
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
OUT = f"{ROOT}/results/ambient_stratified.json"

TRAITS = ["EducationalAttainment", "Intelligence", "ReactionTime",
          "HRV_RMSSD", "HRV_SDNN", "RestingHeartRate", "PRinterval",
          "RheumatoidArthritis"]
COG = {"EducationalAttainment", "Intelligence", "ReactionTime"}
NEURAL = ["PLP1", "MPZ", "S100B", "NRXN1", "PTPRZ1", "SOX10", "MBP", "CDH19",
          "NGFR", "L1CAM", "GFRA3", "SCN7A"]
MIN_CM = 30

a = ad.read_h5ad(H5)
state = a.obs["cell_state"].astype(str).to_numpy()
strat = (a.obs.donor_id.astype(str) + "/" + a.obs.region.astype(str) + "/"
         + a.obs.assay.astype(str)).to_numpy()
X = sp.csr_matrix(a.X) if not sp.issparse(a.X) else a.X.tocsr()
gidx = {g: i for i, g in enumerate(a.var_names)}
is_cm = pd.Series(state).str.match(r"^(aCM|vCM)").to_numpy()

ii = [gidx[g] for g in NEURAL if g in gidx]
neu = np.asarray(X[:, ii].mean(axis=1)).ravel()
san = state == "SAN_P_cell"

# Split within stratum, not globally: donors and chemistries differ in how much
# ambient signal they carry, and a global median would sort cells by batch.
low = np.zeros(len(state), dtype=bool)
for s in pd.unique(strat):
    m = san & (strat == s)
    if m.sum() >= 6:
        low[np.where(m)[0][neu[m] <= np.median(neu[m])]] = True
high = san & ~low
print(f"SAN_P_cell {san.sum()}  ->  clean half {low.sum()},  "
      f"nerve-carrying half {high.sum()}")
print(f"neural-marker mean: clean {neu[low].mean():.3f}, "
      f"carrying {neu[high].mean():.3f}")


def strat_auc(values, focus):
    num = den = stat = var = 0.0
    for s in pd.unique(strat):
        m = strat == s
        f = m & focus
        o = m & is_cm
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


print("\n" + "=" * 96)
print("AUC vs myocytes, in the clean half and the nerve-carrying half")
print("=" * 96)
print(f"{'trait':<26}{'all 245':>16}{'clean half':>16}{'carrying half':>17}"
      f"{'clean - all':>14}")
res = {}
for t in TRAITS:
    p = Path(f"{SC}/{t}.score.tsv")
    if not p.exists():
        continue
    v = pd.read_csv(p, sep="\t", index_col=0)["norm_score"].reindex(
        a.obs_names).to_numpy()
    a0, z0 = strat_auc(v, san)
    a1, z1 = strat_auc(v, low)
    a2, z2 = strat_auc(v, high)
    if a0 is None:
        continue
    res[t] = dict(all=float(a0), z_all=float(z0),
                  clean=float(a1) if a1 else None,
                  z_clean=float(z1) if z1 else None,
                  carrying=float(a2) if a2 else None,
                  z_carrying=float(z2) if z2 else None)
    tag = "  <- cognitive" if t in COG else ""
    print(f"{t:<26}{f'{a0:.3f}(z{z0:+.1f})':>16}"
          f"{f'{a1:.3f}(z{z1:+.1f})' if a1 else '-':>16}"
          f"{f'{a2:.3f}(z{z2:+.1f})' if a2 else '-':>17}"
          f"{(a1 - a0) if a1 else float('nan'):>+14.3f}{tag}")

print("\n" + "=" * 96)
print("VERDICT")
print("=" * 96)
cog = [res[t] for t in COG if t in res and res[t]["clean"] is not None]
car = [res[t] for t in ("HRV_RMSSD", "HRV_SDNN", "RestingHeartRate")
       if t in res and res[t]["clean"] is not None]
if cog and car:
    dc = float(np.mean([r["clean"] - r["all"] for r in cog]))
    dk = float(np.mean([r["clean"] - r["all"] for r in car]))
    survives = all(r["clean"] > 0.55 and r["z_clean"] > 2 for r in cog)
    print(f"  cognitive traits, clean half minus all : {dc:+.3f}")
    print(f"  cardiac traits,   clean half minus all : {dk:+.3f}")
    print()
    if survives and dc > -0.05:
        print("  The enrichment SURVIVES in pacemaker cells carrying no nerve")
        print("  transcript. Ambient neural RNA is not the explanation.")
    elif survives:
        print("  It survives but is clearly weakened. Ambient RNA contributes a real")
        print("  share and the effect size must be reported from the clean half.")
    else:
        print("  It does NOT survive. The cognitive signal in pacemaker cells is")
        print("  carried by cells contaminated with nerve transcript, and the")
        print("  cell-level localisation must be withdrawn. The genetic correlation")
        print("  is unaffected — it never depended on the single-cell data.")
    res["summary"] = dict(delta_cognitive=dc, delta_cardiac=dk, survives=bool(survives))

with open(OUT, "w") as f:
    json.dump(res, f, indent=2, default=float)
print(f"\nwrote {OUT}")
