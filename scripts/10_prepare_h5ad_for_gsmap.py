"""
Convert the CELLxGENE Visium h5ad into per-slide inputs that gsMap accepts.
==========================================================================

gsMap requires:
  * raw counts in `adata.layers[<data_layer>]`   (we write `layers["count"]`)
  * spatial coordinates in `adata.obsm["spatial"]`
  * human gene SYMBOLS as var_names (the CELLxGENE file uses Ensembl IDs)
  * optionally an annotation column in `.obs`

Why per-slide: gsMap's own benchmarks are 8 GB at 1,437 spots, 11 GB at 2,902,
12 GB at 3,289, and 80 GB at 120K. Memory is dominated by a ~10 GB fixed baseline,
but it still grows with spots. The SAN object merges 8 sections (27,108 spots);
splitting to ~3,400 spots per section keeps each run inside the 12 GB WSL2 cap.
Running per section is also the right unit conceptually - gsMap models one tissue
section's spatial graph.

Usage:  python scripts/10_prepare_h5ad_for_gsmap.py
Output: data/spatial_gsmap/<REGION>__<slideID>.h5ad  + a manifest TSV
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/spatial_gsmap"
OUT.mkdir(parents=True, exist_ok=True)

SLIDE_KEY = "sangerID"
ANNOT = "annotation_final"
MIN_SPOTS_PER_SLIDE = 500


def counts_matrix(a):
    """Return a raw-count matrix, or None if we cannot find one."""
    if a.raw is not None:
        X = a.raw.X
        # raw may carry more genes than .var; align on the current var_names
        if a.raw.shape[1] != a.shape[1]:
            raw_names = pd.Index(a.raw.var_names)
            keep = raw_names.get_indexer(a.var_names)
            if (keep < 0).any():
                return None
            X = X[:, keep]
        return X
    for k in ("counts", "count", "raw_counts"):
        if k in a.layers:
            return a.layers[k]
    return None


def looks_like_counts(X, n=2000):
    sub = X[:n] if X.shape[0] > n else X
    v = sub.data if sparse.issparse(sub) else np.asarray(sub).ravel()
    if v.size == 0:
        return False
    return bool(np.allclose(v[:100000], np.round(v[:100000])))


rows = []
for region in ["SAN", "AVN"]:
    src = ROOT / f"data/spatial/{region}.h5ad"
    if not src.exists():
        print(f"[skip] {src.name} missing")
        continue
    print(f"\n=== {region} ===", flush=True)
    a = sc.read_h5ad(src)

    X = counts_matrix(a)
    if X is None:
        print("  !! no raw-count matrix found - cannot prepare this region")
        continue
    print(f"  count matrix: {X.shape}, integer-valued = {looks_like_counts(X)}")

    a.X = X                       # make counts the main matrix
    a.raw = None                  # drop raw: it is Ensembl-indexed and would confuse downstream tools
    a.layers.clear()
    a.layers["count"] = X

    a.var_names = a.var["feature_name"].astype(str)
    a.var_names_make_unique()
    # var_names_make_unique() appends suffixes, so the index no longer matches the
    # source column; keeping both (with the index inheriting its name) makes h5ad
    # writing fail. Drop the redundant columns and clear the index name.
    a.var = a.var.drop(columns=[c for c in a.var.columns if c != "feature_biotype"],
                       errors="ignore")
    a.var.index.name = None

    if "spatial" not in a.obsm:
        print("  !! obsm['spatial'] missing - cannot prepare")
        continue

    keep_obs = [c for c in (SLIDE_KEY, ANNOT, "donor_id", "sex", "region") if c in a.obs.columns]
    a.obs = a.obs[keep_obs].copy()

    if SLIDE_KEY not in a.obs.columns:
        print(f"  !! {SLIDE_KEY} missing - writing region as a single object")
        slides = {region: np.ones(a.n_obs, bool)}
    else:
        slides = {s: (a.obs[SLIDE_KEY].astype(str) == s).to_numpy()
                  for s in a.obs[SLIDE_KEY].astype(str).unique()}

    for sid, m in sorted(slides.items()):
        if m.sum() < MIN_SPOTS_PER_SLIDE:
            print(f"  [skip] {sid}: only {m.sum()} spots")
            continue
        sub = a[m].copy()
        # gsMap needs at least some expression per gene within the slide
        det = np.asarray((sub.X > 0).sum(axis=0)).ravel()
        sub = sub[:, det >= 10].copy()
        p = OUT / f"{region}__{sid}.h5ad"
        sub.write_h5ad(p, compression="gzip")
        nann = sub.obs[ANNOT].nunique() if ANNOT in sub.obs.columns else 0
        print(f"  wrote {p.name:<34} {sub.n_obs:>6,} spots x {sub.n_vars:>6,} genes "
              f"| {nann} annotation levels | {p.stat().st_size/1e6:.0f} MB")
        rows.append({"region": region, "slide": sid, "sample_name": f"{region}__{sid}",
                     "path": str(p), "n_spots": int(sub.n_obs), "n_genes": int(sub.n_vars),
                     "annotation": ANNOT, "n_annot_levels": int(nann)})
    del a

man = pd.DataFrame(rows)
man.to_csv(OUT / "manifest.tsv", sep="\t", index=False)
print(f"\n=== manifest ({len(man)} slides) -> {OUT/'manifest.tsv'} ===")
if len(man):
    print(man[["sample_name", "n_spots", "n_genes", "n_annot_levels"]].to_string(index=False))
    print(f"\nspots per slide: min {man.n_spots.min():,}  median {int(man.n_spots.median()):,}  "
          f"max {man.n_spots.max():,}")
