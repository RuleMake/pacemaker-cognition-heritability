"""Which cell2location columns actually survive into the gsMap inputs?"""
import glob
import os
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.as_posix()
for tag, pat in [("gsMap input", f"{ROOT}/data/spatial_gsmap/SAN__*.h5ad"),
                 ("source atlas", f"{ROOT}/data/spatial/SAN.h5ad")]:
    fs = sorted(glob.glob(pat))
    if not fs:
        print(f"{tag}: no files at {pat}\n")
        continue
    a = ad.read_h5ad(fs[0], backed="r")
    print(f"=== {tag}: {os.path.basename(fs[0])}  ({a.n_obs:,} x {a.n_vars:,})")
    print(f"    obs columns ({len(a.obs.columns)}):")
    for c in a.obs.columns:
        v = a.obs[c]
        extra = ""
        if v.dtype.kind in "fiu":
            arr = v.to_numpy(dtype=float)
            extra = (f"  finite={np.isfinite(arr).sum():,}  "
                     f"mean={np.nanmean(arr):.3f}" if np.isfinite(arr).any()
                     else "  ALL NaN")
        print(f"      {c:<32}{str(v.dtype):<12}{extra}")
    print()
