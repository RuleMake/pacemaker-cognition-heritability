"""
An independent cohort that does not exist in humans: replicate in mouse.
=======================================================================

The single largest weakness of the human result is that there is no second cohort.
Kanemaru is the only human heart atlas that dissected sinoatrial and atrioventricular
node tissue, so "replicate in an independent human dataset" is not a request that can
be met — the dataset does not exist.

What does exist is Goodyer et al. 2019 (Circ Res, GSE132658): the developing mouse
heart, microdissected into the SAME three zones this project cares about — sinoatrial
node, atrioventricular node/His, and the Purkinje fibre network, left and right — and
sequenced at single-cell resolution. Different species, different laboratory, different
developmental stage, different dissociation protocol, different chemistry.

That is a harder test than a second human cohort in one specific way: every technical
artefact this project has spent months chasing — this atlas's ambient RNA, this atlas's
sequencing depth, this atlas's annotation calls, this donor set — is absent by
construction. A signal that survives the species change cannot be any of them.

It is a weaker test in another way, which the paper must state plainly:

  * mouse is E16.5, not adult. Pacemaker identity is established by then, but the
    tissue is developing
  * roughly a third of human genes have no usable mouse one-to-one orthologue, so
    every gene set loses members before it is scored
  * there is no mouse phenotype corresponding to educational attainment. What is being
    tested is whether the genes that carry human cognitive heritability are
    preferentially expressed in pacemaker cells, in a conserved cell type — not
    anything about mouse cognition

Cell identity is assigned here from a marker panel fixed in advance, not from the
authors' cluster labels and not from anything downstream of a GWAS. The panel is the
mouse orthologue of the panel already used to validate the human annotation, so the two
species are being asked the same question in the same words.

Usage:  python scripts/210_mouse_prepare.py
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/mouse/raw"
OUT = ROOT / "data/mouse/mouse_ccs.h5ad"
META = ROOT / "results/mouse_prepare.json"

ZONES = {
    "GSM3885058_SAN": "sinoatrial",
    "GSM3885059_AVN": "atrioventricular",
    "GSM3885060_LPF": "purkinje_left",
    "GSM3885061_RPF": "purkinje_right",
}

# Fixed before looking at the data. The nodal panel is the mouse orthologue of the
# human panel used in 151/202; the Purkinje panel is from the source paper's own
# conduction-system markers.
PANELS = {
    "cardiomyocyte": ["Tnnt2", "Actc1", "Myh6", "Ttn", "Ryr2"],
    "nodal": ["Hcn4", "Shox2", "Tbx3", "Isl1", "Vsnl1"],
    "purkinje": ["Cntn2", "Etv1", "Slit2", "Irx3"],
    "ventricular": ["Myl2", "Myh7", "Irx4"],
    "atrial": ["Nppa", "Myl7", "Sln"],
    "fibroblast": ["Col1a1", "Postn", "Dcn"],
    "endothelium": ["Pecam1", "Cdh5", "Egfl7"],
    "immune": ["Ptprc", "C1qa", "Cd68"],
    "erythroid": ["Hba-a1", "Hbb-bs", "Alas2"],
}

MIN_GENES = 300
MAX_MITO = 0.15
LEIDEN_RES = 1.2

# ------------------------------------------------------------------ load
adatas = []
for pref, zone in ZONES.items():
    stem = pref.split("_")[1]
    m = RAW / f"{pref}matrix.mtx.gz"
    g = RAW / f"{pref}genes.tsv.gz"
    b = RAW / f"{pref}barcodes.tsv.gz"
    if not m.exists():
        raise SystemExit(f"missing {m}")
    with gzip.open(m, "rb") as f:
        X = csr_matrix(mmread(f).T.tocsr())          # file is genes x cells
    genes = pd.read_csv(g, sep="\t", header=None, names=["ensembl", "symbol"])
    bcs = pd.read_csv(b, sep="\t", header=None)[0].to_numpy()
    a = sc.AnnData(X)
    a.var_names = genes.symbol.to_numpy()
    a.var["ensembl"] = genes.ensembl.to_numpy()
    # all four samples ship the identical genes.tsv, so de-duplicating each one
    # separately gives the same names in the same order and concat stays aligned
    a.var_names_make_unique()
    a.obs_names = [f"{stem}_{x}" for x in bcs]
    a.obs["zone"] = zone
    a.obs["sample"] = stem
    adatas.append(a)
    print(f"{zone:<18} {a.n_obs:>6,} cells x {a.n_vars:,} genes")

a = sc.concat(adatas, join="inner")
a.var_names_make_unique()
print(f"\nconcatenated: {a.n_obs:,} cells x {a.n_vars:,} genes")

# ------------------------------------------------------------------ QC
a.var["mt"] = a.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, log1p=False,
                           percent_top=None)
before = a.n_obs
a = a[(a.obs.n_genes_by_counts >= MIN_GENES)
      & (a.obs.pct_counts_mt <= MAX_MITO * 100)].copy()
sc.pp.filter_genes(a, min_cells=3)
print(f"QC: {before:,} -> {a.n_obs:,} cells  ({a.n_vars:,} genes kept)")

a.layers["counts"] = a.X.copy()
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
a.raw = a

# ------------------------------------------------------------------ cluster
sc.pp.highly_variable_genes(a, n_top_genes=2000, batch_key="sample")
h = a[:, a.var.highly_variable].copy()
sc.pp.scale(h, max_value=10)
sc.tl.pca(h, n_comps=50, svd_solver="arpack")
a.obsm["X_pca"] = h.obsm["X_pca"]
sc.pp.neighbors(a, n_neighbors=15, n_pcs=30)
sc.tl.leiden(a, resolution=LEIDEN_RES, key_added="leiden", flavor="igraph",
             n_iterations=2, directed=False)
print(f"leiden: {a.obs.leiden.nunique()} clusters")

# ------------------------------------------------------------------ annotate
for name, panel in PANELS.items():
    present = [g for g in panel if g in a.var_names]
    if not present:
        print(f"!! panel {name}: no genes present")
        continue
    sc.tl.score_genes(a, present, score_name=f"s_{name}", ctrl_size=50)
    print(f"panel {name:<14} {len(present)}/{len(panel)} genes: {present}")

sc_cols = [f"s_{k}" for k in PANELS if f"s_{k}" in a.obs.columns]
prof = a.obs.groupby("leiden", observed=True)[sc_cols].mean()
# z-score each panel across clusters so panels of different magnitude compete fairly
z = (prof - prof.mean()) / prof.std().replace(0, 1)
assign = z.idxmax(1).str.replace("s_", "", regex=False)

# A cluster is conduction tissue only if it is a myocyte FIRST. Nodal and Purkinje
# programmes are read within the myocyte compartment, never against fibroblasts.
is_cm = z["s_cardiomyocyte"] > 0
label = {}
for cl in prof.index:
    top = assign[cl]
    if top in ("nodal", "purkinje") and not is_cm[cl]:
        top = "non_myocyte_" + top
    elif top in ("ventricular", "atrial", "cardiomyocyte"):
        top = "working_CM"
    label[cl] = top
a.obs["cell_class"] = a.obs.leiden.map(label).astype(str)

print("\n" + "=" * 88)
print("CLUSTER ANNOTATION (z-scored panel means across clusters)")
print("=" * 88)
print(f"{'cluster':<9}{'n':>7}  " + "".join(f"{k[:9]:>11}" for k in PANELS)
      + "   label")
for cl in prof.index:
    n = int((a.obs.leiden == cl).sum())
    row = "".join(f"{z.loc[cl, f's_{k}']:>11.2f}" if f"s_{k}" in z.columns
                  else f"{'-':>11}" for k in PANELS)
    print(f"{cl:<9}{n:>7,}  {row}   {label[cl]}")

print("\n" + "=" * 88)
print("CELL CLASSES BY ZONE")
print("=" * 88)
tab = pd.crosstab(a.obs.cell_class, a.obs.zone)
print(tab.to_string())

nodal = int((a.obs.cell_class == "nodal").sum())
purk = int((a.obs.cell_class == "purkinje").sum())
work = int((a.obs.cell_class == "working_CM").sum())
print(f"\nnodal {nodal:,}   purkinje {purk:,}   working myocytes {work:,}")
if nodal < 30:
    print("!! too few nodal cells for the myocyte-referenced comparison")

a.write(OUT)
json.dump(dict(zones=ZONES, panels=PANELS, n_cells=int(a.n_obs),
               n_genes=int(a.n_vars), leiden_res=LEIDEN_RES,
               classes=tab.to_dict(),
               cluster_labels={str(k): v for k, v in label.items()}),
          open(META, "w"), indent=2)
print(f"\nwrote {OUT}\nwrote {META}")
