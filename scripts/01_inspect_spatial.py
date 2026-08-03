"""
Inspect the Kanemaru et al. (Nature 2023) Visium slides downloaded from CELLxGENE.

Purpose: verify the data actually supports the planned study design, i.e. that it contains
  (a) spatial coordinates,
  (b) micro-anatomical / niche annotations (so trait enrichment can be interpreted),
  (c) usable gene symbols,
  (d) multiple donors / slides.

Usage:  python scripts/01_inspect_spatial.py data/spatial/SAN.h5ad
"""
import sys
import anndata as ad
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

path = sys.argv[1] if len(sys.argv) > 1 else "data/spatial/SAN.h5ad"

# backed mode: read structure without pulling the whole matrix into RAM
a = ad.read_h5ad(path, backed="r")

print("=" * 78)
print(f"FILE : {path}")
print(f"SHAPE: {a.n_obs:,} spots x {a.n_vars:,} genes")
print("=" * 78)

print("\n--- obs columns ---")
for c in a.obs.columns:
    s = a.obs[c]
    nu = s.nunique(dropna=True)
    kind = "categorical" if str(s.dtype) in ("category", "object") else str(s.dtype)
    head = ", ".join(map(str, list(pd.unique(s.dropna()))[:6]))
    print(f"  {c:<38} {kind:<12} n_unique={nu:<6} {head[:70]}")

print("\n--- obsm (spatial coordinates live here) ---")
for k in a.obsm.keys():
    print(f"  {k:<28} shape={a.obsm[k].shape}")

print("\n--- var columns ---")
print("  ", list(a.var.columns))
print("  first 8 var_names:", list(a.var_names[:8]))
if "feature_name" in a.var.columns:
    print("  first 8 feature_name:", list(a.var["feature_name"][:8]))

print("\n--- uns keys ---")
print("  ", list(a.uns.keys())[:25])

print("\n--- layers / raw ---")
print("  layers:", list(a.layers.keys()))
print("  raw   :", "present" if a.raw is not None else "absent")

# Donor / sample structure matters for interpreting how far results generalise
for col in ("donor_id", "sample", "library_id", "Sample", "orig.ident"):
    if col in a.obs.columns:
        print(f"\n--- breakdown by {col} ---")
        print(a.obs[col].value_counts().to_string())
        break

# Niche / region annotation is what makes spatial trait mapping interpretable
cand = [c for c in a.obs.columns
        if any(k in c.lower() for k in
               ("niche", "region", "anno", "cluster", "cell_type", "celltype", "zone", "structure"))]
print(f"\n--- candidate niche/annotation columns: {cand} ---")
for c in cand[:6]:
    vc = a.obs[c].value_counts()
    print(f"\n  [{c}]  ({len(vc)} levels)")
    print("   " + vc.head(15).to_string().replace("\n", "\n   "))
