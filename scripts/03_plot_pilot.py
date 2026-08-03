"""
Visualise the pilot: where does the resting-heart-rate genetic signal sit on the slide?

Produces results/pilot_SAN_map.png with three panels per slide:
  (1) micro-anatomical annotation   (2) trait score   (3) pacemaker-cell abundance

Usage:  python scripts/03_plot_pilot.py
"""

import gzip
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

P_THRESH, CLUMP_KB, WINDOW_KB, MIN_SPOTS = 5e-8, 500_000, 100_000, 50


def trait_gene_set():
    rows = []
    pn, pt = re.compile(r'gene_name "([^"]+)"'), re.compile(r'gene_type "([^"]+)"')
    with gzip.open(ROOT / "data/resource/gencode_v44lift37.gtf.gz", "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            p = line.split("\t", 9)
            if p[2] != "gene":
                continue
            mt = pt.search(p[8])
            mn = pn.search(p[8])
            c = p[0].replace("chr", "")
            if not mt or mt.group(1) != "protein_coding" or not mn:
                continue
            if c not in {str(i) for i in range(1, 23)}:
                continue
            rows.append((mn.group(1), int(c), int(p[3]), int(p[4])))
    genes = pd.DataFrame(rows, columns=["gene", "chr", "start", "end"]).drop_duplicates("gene")

    gw = pd.read_csv(ROOT / "data/gwas/RHR_ZhuZ_UKB460K.assoc.gz", sep=r"\s+",
                     usecols=["CHR", "BP", "P"], dtype=str)
    gw = gw[gw["CHR"] != "CHR"]
    gw = gw[gw["CHR"].isin([str(i) for i in range(1, 23)])].copy()
    gw["CHR"] = gw["CHR"].astype(np.int64)
    gw["BP"] = pd.to_numeric(gw["BP"], errors="coerce")
    gw["P"] = pd.to_numeric(gw["P"], errors="coerce")
    gw = gw.dropna(subset=["BP", "P"])
    gw["BP"] = gw["BP"].astype(np.int64)

    sig = gw[gw["P"] < P_THRESH].sort_values("P")
    leads = []
    for c, sub in sig.groupby("CHR", sort=False):
        taken = []
        for bp in sub["BP"].to_numpy():
            if all(abs(bp - t) >= CLUMP_KB for t in taken):
                taken.append(bp)
        leads += [(c, b) for b in taken]

    hits = set()
    for c, sub in pd.DataFrame(leads, columns=["chr", "bp"]).groupby("chr"):
        gc = genes[genes["chr"] == c]
        gs, ge, gn = gc["start"].to_numpy(), gc["end"].to_numpy(), gc["gene"].to_numpy()
        for bp in sub["bp"].to_numpy():
            hits.update(gn[(gs - WINDOW_KB <= bp) & (ge + WINDOW_KB >= bp)].tolist())
    return sorted(hits)


def plot_slide(name, panel, pace_col="SAN_P_cell"):
    a = sc.read_h5ad(ROOT / f"data/spatial/{name}.h5ad")
    a.raw = None
    a.var_names = a.var["feature_name"].astype(str)
    a.var_names_make_unique()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    det = np.asarray((a.X > 0).sum(axis=0)).ravel()
    a = a[:, det >= MIN_SPOTS].copy()

    X = np.asarray(a.X.todense() if hasattr(a.X, "todense") else a.X, dtype=np.float32)
    Z = (X - X.mean(0, keepdims=True)) / np.where(X.std(0, keepdims=True) == 0, 1,
                                                  X.std(0, keepdims=True))
    pos = {g: i for i, g in enumerate(a.var_names)}
    idx = np.array([pos[g] for g in panel if g in pos])
    score = Z[:, idx].mean(1)

    # one representative slide keeps the anatomy readable; all slides overlap in one frame
    sid_col = "sangerID" if "sangerID" in a.obs.columns else None
    if sid_col:
        sid = a.obs[sid_col].value_counts().idxmax()
        m = (a.obs[sid_col] == sid).to_numpy()
    else:
        sid, m = name, np.ones(a.n_obs, bool)

    xy = a.obsm["spatial"][m]
    ann = a.obs["annotation_final"].astype(str).to_numpy()[m]
    s = score[m]
    pc = a.obs[pace_col].to_numpy().astype(float)[m] if pace_col in a.obs.columns else None

    fig, axes = plt.subplots(1, 3 if pc is not None else 2,
                             figsize=(5.2 * (3 if pc is not None else 2), 5.0))
    fig.suptitle(f"{name} - slide {sid}   |   resting heart-rate GWAS gene set "
                 f"({len(idx)} genes)", fontsize=12)

    cats = pd.unique(ann)
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(cats):
        k = ann == c
        axes[0].scatter(xy[k, 0], -xy[k, 1], s=3, color=cmap(i % 10), label=c, linewidths=0)
    axes[0].set_title("micro-anatomical annotation")
    axes[0].legend(fontsize=6, markerscale=3, loc="best", frameon=False)

    v = np.percentile(np.abs(s - np.median(s)), 98)
    h = axes[1].scatter(xy[:, 0], -xy[:, 1], c=s, s=3, cmap="RdBu_r",
                        vmin=np.median(s) - v, vmax=np.median(s) + v, linewidths=0)
    axes[1].set_title("resting heart-rate trait score")
    plt.colorbar(h, ax=axes[1], fraction=0.046)

    if pc is not None:
        h2 = axes[2].scatter(xy[:, 0], -xy[:, 1], c=pc, s=3, cmap="viridis",
                             vmax=np.percentile(pc, 99), linewidths=0)
        axes[2].set_title("SAN pacemaker-cell abundance\n(cell2location)")
        plt.colorbar(h2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    p = OUT / f"pilot_{name}_map.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"  saved {p}")


if __name__ == "__main__":
    print("building trait gene set ...", flush=True)
    panel = trait_gene_set()
    print(f"  {len(panel)} genes", flush=True)
    for nm in ["SAN", "AVN"]:
        try:
            plot_slide(nm, panel)
        except Exception as e:
            print(f"  {nm} failed: {type(e).__name__}: {e}")
