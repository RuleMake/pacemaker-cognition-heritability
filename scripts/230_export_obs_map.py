"""Export the minimal cell annotation needed by the figure scripts.

The figure scripts read only `results/`. The per-cell scDRS scores live there
already, but the cell-to-state mapping lives in the single-cell object, so the
distributions behind the summary statistics could not be drawn. This script
reads nothing but `.obs` from each object, in backed mode, and writes the few
columns the figures need. It computes no statistic.
"""

from pathlib import Path

import anndata as ad

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "singlecell"
OUT = ROOT / "results"

WANT = ["cell_state", "donor_id", "region", "stratum", "assay", "cell_type"]

for name, out_name in [("node_subset", "obs_node.tsv"),
                       ("axis_subset", "obs_axis.tsv")]:
    path = SRC / f"{name}.h5ad"
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    cols = [c for c in WANT if c in obs.columns]
    out = obs[cols].copy()
    out.index.name = "cell"
    out.to_csv(OUT / out_name, sep="\t")
    print(f"{name}: {a.shape[0]:,} cells -> results/{out_name}  "
          f"columns {cols}")
    if "cell_state" in cols:
        top = obs.cell_state.value_counts().head(4)
        print("   ", ", ".join(f"{k} {v:,}" for k, v in top.items()))
    a.file.close()
