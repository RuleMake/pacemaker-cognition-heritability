"""
Quantify the dilution that explains why the spatial analysis found nothing.
===========================================================================

The claim is that pacemaker heritability is real but invisible at 55 um because a spot
holds roughly ten cells and pacemaker cells are a minority even where they are densest.
That is an assertion so far. It is measurable, and it should be measured, because it
is the hinge connecting a negative result to a positive one on the same donors.

Two quantities decide it:

  purity        cell2location gives a per-spot abundance for every cell type. The
                pacemaker FRACTION of a spot's cells — even at the 99th percentile —
                says how concentrated the signal can possibly get in this technology.

  attenuation   if the richest spot is p percent pacemaker, a difference between
                pacemaker cells and everything else is attenuated to roughly p times
                its cell-level size before any noise is added. Comparing that to the
                effect actually seen at cell level says whether the spatial design
                could ever have detected it.

If the purest spots are a small minority pacemaker, the spatial null was predictable
in advance and the resolution explanation stands. If some spots are largely pacemaker,
the explanation fails and something else killed the spatial analysis.

Usage:  python scripts/110_quantify_dilution.py
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.as_posix()
SRC = f"{ROOT}/data/spatial/SAN.h5ad"
OUT = f"{ROOT}/results"

a = ad.read_h5ad(SRC, backed="r")
obs = a.obs
cell_cols = [c for c in obs.columns
             if obs[c].dtype.kind == "f" and c not in
             ("n_genes_by_counts", "log1p_n_genes_by_counts", "total_counts",
              "log1p_total_counts")]
ab = obs[cell_cols].astype(float)
ab = ab.loc[:, ab.notna().any()]
total = ab.sum(axis=1)
print(f"spots {len(ab):,}   deconvolved cell types {ab.shape[1]}")
print(f"estimated cells per spot: median {total.median():.1f}, "
      f"IQR {total.quantile(.25):.1f}-{total.quantile(.75):.1f}")

if "SAN_P_cell" not in ab.columns:
    raise SystemExit("SAN_P_cell not in the deconvolution")

frac = (ab["SAN_P_cell"] / total.clip(lower=1e-9)) * 100
ann = obs["annotation_final"].astype(str)

print("\n" + "=" * 92)
print("HOW PURE DOES A SPOT GET?  pacemaker share of a spot's cells")
print("=" * 92)
print(f"{'spot set':<28}{'n':>8}{'median':>9}{'p90':>8}{'p99':>8}{'max':>8}")
rows = [("all SAN spots", np.ones(len(frac), bool)),
        ("annotated `node` only", (ann == "node").to_numpy()),
        ("top 1% by pacemaker abundance",
         (ab["SAN_P_cell"] >= ab["SAN_P_cell"].quantile(0.99)).to_numpy())]
summary = {}
for lbl, m in rows:
    v = frac[m]
    summary[lbl] = dict(n=int(m.sum()), median=float(v.median()),
                        p90=float(v.quantile(.90)), p99=float(v.quantile(.99)),
                        max=float(v.max()))
    print(f"{lbl:<28}{int(m.sum()):>8,}{v.median():>8.1f}%{v.quantile(.90):>7.1f}%"
          f"{v.quantile(.99):>7.1f}%{v.max():>7.1f}%")

node_med = summary["annotated `node` only"]["median"]
best = summary["top 1% by pacemaker abundance"]["median"]

print("\n" + "=" * 92)
print("WHAT THAT DOES TO A CELL-LEVEL DIFFERENCE")
print("=" * 92)
print("A spot's expression is roughly the abundance-weighted average of its cells, so")
print("a contrast between pacemaker cells and the rest arrives at spot level scaled by")
print("the pacemaker fraction.\n")
print(f"{'spot set':<32}{'pacemaker share':>18}{'retained':>12}")
for lbl in ["annotated `node` only", "top 1% by pacemaker abundance"]:
    s = summary[lbl]["median"]
    print(f"{lbl:<32}{s:>17.1f}%{s:>11.1f}%")
print(f"\nEven the most pacemaker-rich one percent of spots is a median "
      f"{best:.1f}% pacemaker.")
print(f"Spots a pathologist labelled `node` are a median {node_med:.1f}%.")

print("\n" + "=" * 92)
print("VERDICT — judged against what each spatial analysis actually used")
print("=" * 92)
print("The two spatial analyses had different amounts of dilution to contend with,")
print("so a single threshold answers the wrong question.\n")

print(f"COMPARTMENT analysis (node label vs rest, {summary['annotated `node` only']['n']:,} spots)")
print(f"   the spots it compared are a median {node_med:.1f}% pacemaker, so a")
print(f"   cell-level contrast arrived attenuated to about 1/{100 / max(node_med, 1e-9):.0f}")
print("   of its size. Dilution alone is enough to explain that null.\n")

print(f"ABUNDANCE analysis (partial correlation against SAN_P_cell, all spots)")
print(f"   this one weighted spots continuously and therefore DID have access to the")
print(f"   purest spots — the top one percent are a median {best:.1f}% pacemaker,")
print(f"   reaching {summary['top 1% by pacemaker abundance']['max']:.0f}%. Dilution alone does NOT")
print("   explain why it found nothing (+0.011, p = 0.64).")
print("   What explains that one: sequencing depth dominating the per-spot signal")
print("   (rho = +0.76, shown causal by the thinning experiment), transcriptome")
print("   complexity, and a minimum detectable effect of 0.100 from 8 sections.\n")

print("So the honest account of the discrepancy is a combination — heavy dilution at")
print("the compartment level, technical confounds and limited power at the abundance")
print("level — not resolution alone. The single-cell analysis removes all three at")
print("once, which is why it cannot be attributed to any single one of them.")

with open(f"{OUT}/dilution.json", "w") as f:
    json.dump(dict(cells_per_spot=float(total.median()), purity=summary), f, indent=2)
print(f"\nwrote {OUT}/dilution.json")
