"""
Causal test of the depth confound: equalise sequencing depth, re-run gsMap.
==========================================================================

Everything so far is correlational. gsMap's compartment ordering matches the
sequencing-depth ordering (rho ~ 0.85) and its per-spot p tracks depth (rho ~ 0.76),
but a correlation cannot prove the depth is what produces the map. And the obvious
fix — subtracting a control trait — was just shown to remove only 6% of it, because
the artefact scales with how strong the trait's signal is in that tissue rather than
being a constant offset.

The intervention settles it. Binomially thin every spot to one common UMI total, so
depth is constant by construction, then run gsMap again on the thinned data:

  * If the "node enrichment" survives, it is biology and depth was a bystander.
  * If it collapses while atrial fibrillation still finds atrial myocardium — the
    positive control that has held throughout — then depth was generating it, the
    confound is causal, and thinning is a correction that works.

Either outcome is informative, which is what makes it worth two hours of compute.

Design choices
--------------
target   the 20th percentile of the spot UMI distribution. Spots below it cannot be
         thinned up, so they are dropped; a lower target keeps more spots but throws
         away more counts from the rest. p20 keeps ~80% of spots.
thinning binomial, count-preserving, applied to the raw integer count layer that
         gsMap actually consumes. np.random.Generator with a fixed seed per section
         so the whole thing reproduces exactly.

Usage:  python scripts/60_make_depth_matched.py
"""

import glob
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent.as_posix()
SRC = f"{ROOT}/data/spatial_gsmap"
DST = f"{ROOT}/data/spatial_depthmatched"
os.makedirs(DST, exist_ok=True)

PCTL = 20          # depth target percentile, pooled over the sections being processed
PREFIX = "SAN__"   # the region the hypothesis is about; AVN adds compute for no test


def counts(a):
    return a.layers["count"] if "count" in a.layers else a.X


files = sorted(glob.glob(f"{SRC}/{PREFIX}*.h5ad"))
print(f"sections: {len(files)}")

# ---- pass 1: pooled depth distribution, so every section is thinned to ONE target
tot = []
for p in files:
    a = ad.read_h5ad(p)
    tot.append(np.asarray(counts(a).sum(axis=1)).ravel())
pool = np.concatenate(tot)
target = int(np.percentile(pool, PCTL))
print(f"pooled spots {pool.size:,}   median UMI {np.median(pool):,.0f}")
print(f"depth target = p{PCTL} = {target:,} UMI"
      f"   -> keeps {(pool >= target).sum():,} spots ({(pool >= target).mean() * 100:.0f}%)\n")

rows = []
for i, p in enumerate(files):
    s = os.path.basename(p)[:-5]
    a = ad.read_h5ad(p)
    X = counts(a)
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr().copy()
    t = np.asarray(X.sum(axis=1)).ravel()
    keep = t >= target
    a = a[keep].copy()
    X = X[keep]
    t = t[keep]

    # binomial thinning: each count is thinned with probability target/total for its
    # spot, so the expected new total is exactly `target` for every spot
    rng = np.random.default_rng(1000 + i)
    Xc = X.tocoo()
    prob = (target / t)[Xc.row]
    newdata = rng.binomial(Xc.data.astype(np.int64), np.clip(prob, 0, 1))
    Xt = sp.coo_matrix((newdata, (Xc.row, Xc.col)), shape=X.shape).tocsr()
    Xt.eliminate_zeros()

    a.layers["count"] = Xt
    a.X = Xt.astype(np.float32)

    nt = np.asarray(Xt.sum(axis=1)).ravel()
    ng = np.asarray((Xt > 0).sum(axis=1)).ravel()
    out = f"{DST}/{s}.h5ad"
    a.write_h5ad(out, compression="gzip")
    rows.append(dict(section=s, spots_in=int(keep.size), spots_kept=int(keep.sum()),
                     umi_before=float(np.median(t)), umi_after=float(np.median(nt)),
                     umi_cv_after=float(np.std(nt) / np.mean(nt)),
                     genes_after=float(np.median(ng))))
    print(f"  [{i + 1}/{len(files)}] {s:<28} {keep.sum():>5,}/{keep.size:,} spots   "
          f"UMI {np.median(t):>7,.0f} -> {np.median(nt):>6,.0f} "
          f"(CV {rows[-1]['umi_cv_after']:.3f})   genes {np.median(ng):>5,.0f}")

d = pd.DataFrame(rows)
d.to_csv(f"{ROOT}/results/depthmatched_manifest.tsv", sep="\t", index=False)

# manifest in the shape 21_run_gsmap_batch.sh expects
with open(f"{DST}/manifest.tsv", "w") as f:
    f.write("region\tsangerID\tsample\n")
    for s in d.section:
        f.write(f"{s.split('__')[0]}\t{s.split('__')[1]}\t{s}\n")

# Do not print an eyeballed "before" value here. An earlier version printed
# "(was ~0.6-0.9 before)", which was a guess rather than a measurement, and it was
# read back out of this log and quoted as data. Only umi_cv_after is computed.
print(f"\nresidual depth variation after thinning: CV = {d.umi_cv_after.mean():.4f}"
      f"  (pre-thinning CV is not computed)")
print(f"wrote {len(d)} sections to {DST}")
