"""
Can the pacemaker abundance estimate be trusted? Validate the deconvolution input.
=================================================================================

The primary evidence for the negative result is a partial correlation between gsMap's
per-spot p-value and cell2location's SAN_P_cell abundance (+0.011 vs a non-cardiac
control, p = 0.64). That conclusion inherits every weakness of the abundance estimate,
and pacemaker cells are rare — exactly the regime where deconvolution is least
reliable. This was flagged as an open question and never tested.

A null built on a broken input is worth nothing, so the input gets checked directly.
Deconvolution has no ground truth here, but it makes predictions that can be falsified
against the measured transcriptome of the same spots:

  1. Concentration.  Pacemaker abundance should be highest in spots a pathologist
     annotated as node. If it is flat across compartments, it is noise.
  2. Marker agreement.  Spots with more pacemaker cells should express more HCN4 /
     SHOX2 / TBX3 / ISL1. This is the strongest available check because the markers
     are measured independently of the deconvolution.
  3. Specificity.  The same marker correlation computed for cell types with no claim
     on pacemaker markers — adipocytes, glia, fibroblasts, smooth muscle — gives the
     null band the pacemaker correlation has to clear.
  4. Calibration.  Atrial cardiomyocytes are abundant and unmistakable; whatever the
     method achieves for them is roughly the ceiling, and the pacemaker number should
     be read relative to that rather than against 1.0.

Note on where the data lives: the gsMap inputs keep only 5 obs columns, so the
abundances have to come from the source atlas (data/spatial/SAN.h5ad, 102 obs columns,
all eight sections stacked and separated by `sangerID`).

Usage:  python scripts/57_deconvolution_quality.py
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.as_posix()
SRC = f"{ROOT}/data/spatial/SAN.h5ad"
OUT = f"{ROOT}/results"

PACE_MARKERS = ["HCN4", "HCN1", "SHOX2", "TBX3", "ISL1", "BMP4", "VSNL1"]
ACM_MARKERS = ["NPPA", "MYL7", "MYH6", "SLN", "NR2F2"]
# the estimate under test, the easy calibration case, then types that should NOT
# track pacemaker markers and so define the null band
CELLTYPES = ["SAN_P_cell", "aCM1", "aCM3", "aCM4",
             "FB1", "FB3", "FB5", "NC1_glial", "EC2_cap", "EC6_ven",
             "PC1_vent", "PC2_atria", "Adip1", "SMC2_art", "Meso", "Mast"]

a = ad.read_h5ad(SRC)
a.raw = None                      # .raw is Ensembl-indexed; never let scanpy reach it
print(f"source atlas: {a.n_obs:,} spots x {a.n_vars:,} genes")

X = a.X
X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
tot = np.asarray(X.sum(axis=1)).ravel()
looks_raw = bool(np.allclose(X.data[:5000], np.round(X.data[:5000])))
print(f"X looks like {'raw counts' if looks_raw else 'normalised values'} "
      f"— {'CPM+log applied' if looks_raw else 'used as provided'}")
if looks_raw:
    X = X.multiply((1e4 / np.maximum(tot, 1))[:, None]).tocsr()
    X.data = np.log1p(X.data)
Xc = X.tocsc()

# the atlas is Ensembl-indexed; symbols live in a var column. Resolve them or the
# marker check silently finds nothing, which is how this failed the first time.
symbols = a.var_names
for col in ("feature_name", "gene_name", "gene_symbols", "symbol"):
    if col in a.var.columns:
        symbols = a.var[col].astype(str)
        print(f"gene symbols taken from var['{col}']")
        break
else:
    print("no symbol column in var — using var_names directly")

gidx = {}
for i, g in enumerate(symbols):
    gidx.setdefault(str(g), i)
missing = [g for g in PACE_MARKERS + ACM_MARKERS if g not in gidx]
if missing:
    print(f"markers absent from the atlas: {', '.join(missing)}")
if all(g not in gidx for g in PACE_MARKERS):
    raise SystemExit("no pacemaker marker resolved — check the var symbol column")


def expr(genes):
    cols = [gidx[g] for g in genes if g in gidx]
    return np.asarray(Xc[:, cols].mean(axis=1)).ravel() if cols else None


pace = expr(PACE_MARKERS)
acm = expr(ACM_MARKERS)
sid = a.obs["sangerID"].astype(str).to_numpy()
ann = a.obs["annotation_final"].astype(str).to_numpy()
present = [c for c in CELLTYPES if c in a.obs.columns
           and np.isfinite(a.obs[c].to_numpy(dtype=float)).any()]
print(f"cell types usable in SAN: {len(present)}/{len(CELLTYPES)}\n")

rows_dist, rows_conc, rows_mark = [], [], []
for s in sorted(set(sid)):
    m = sid == s
    allab = a.obs.loc[m, present].to_numpy(dtype=float)
    for ct in present:
        v = a.obs.loc[m, ct].to_numpy(dtype=float)
        if np.nanstd(v) == 0:
            continue
        if ct == "SAN_P_cell":
            rows_dist.append(dict(
                section=s, n=int(m.sum()), frac_zero=float((v <= 1e-6).mean()),
                median=float(np.nanmedian(v)), p95=float(np.nanpercentile(v, 95)),
                max=float(np.nanmax(v)),
                share=float(np.nansum(v) / max(np.nansum(allab), 1e-9))))
        innode, rest = v[ann[m] == "node"], v[ann[m] != "node"]
        if len(innode) > 30 and len(rest) > 30:
            rows_conc.append(dict(section=s, cell_type=ct,
                                  ratio=float((np.nanmean(innode) + 1e-9)
                                              / (np.nanmean(rest) + 1e-9))))
        for lbl, mk in [("pacemaker", pace), ("atrial_myocyte", acm)]:
            if mk is None:
                continue
            ok = np.isfinite(v)
            if ok.sum() < 200:
                continue
            rows_mark.append(dict(section=s, cell_type=ct, marker=lbl,
                                  rho=float(spearmanr(v[ok], mk[m][ok])[0])))

dist = pd.DataFrame(rows_dist)
conc = pd.DataFrame(rows_conc)
mark = pd.DataFrame(rows_mark)
for d, n in [(dist, "deconv_distribution"), (conc, "deconv_concentration"),
             (mark, "deconv_markers")]:
    d.to_csv(f"{OUT}/{n}.tsv", sep="\t", index=False)

print("=" * 90)
print("1. IS SAN_P_cell A PLAUSIBLE RARE POPULATION, OR MOSTLY ZEROS?")
print("=" * 90)
print(f"{'section':<22}{'spots':>8}{'% zero':>9}{'median':>10}{'p95':>9}"
      f"{'max':>9}{'% of all cells':>16}")
for _, r in dist.iterrows():
    print(f"{r.section:<22}{int(r.n):>8,}{r.frac_zero * 100:>8.1f}%{r['median']:>10.3f}"
          f"{r.p95:>9.2f}{r['max']:>9.1f}{r.share * 100:>15.2f}%")
print(f"\npooled: {dist.frac_zero.mean() * 100:.1f}% of spots have no pacemaker signal; "
      f"pacemakers are {dist.share.mean() * 100:.2f}% of deconvolved cells")

print("\n" + "=" * 90)
print("2. DOES IT CONCENTRATE WHERE A PATHOLOGIST DREW THE NODE?")
print("=" * 90)
cm = conc.groupby("cell_type").ratio.mean().sort_values(ascending=False)
print(f"{'cell type':<16}{'abundance ratio, node : rest':>32}")
for ct, v in cm.items():
    mk = "  <<< the estimate under test" if ct == "SAN_P_cell" else ""
    print(f"{ct:<16}{v:>32.2f}{mk}")
rank = list(cm.index).index("SAN_P_cell") + 1 if "SAN_P_cell" in cm.index else None
print(f"\nSAN_P_cell ranks {rank}/{len(cm)} on node concentration")

print("\n" + "=" * 90)
print("3. DOES ABUNDANCE TRACK INDEPENDENTLY MEASURED MARKER EXPRESSION?")
print("=" * 90)
print(f"   pacemaker markers: {', '.join(g for g in PACE_MARKERS if g in gidx)}")
print(f"   atrial markers   : {', '.join(g for g in ACM_MARKERS if g in gidx)}\n")
piv = mark.groupby(["cell_type", "marker"]).rho.mean().unstack()
piv = piv.reindex([c for c in CELLTYPES if c in piv.index])
print(f"{'cell type':<16}{'vs pacemaker markers':>24}{'vs atrial markers':>22}")
for ct, r in piv.iterrows():
    mk = ("  <<< under test" if ct == "SAN_P_cell" else
          "  <<< ceiling" if ct == "aCM4" else "")
    print(f"{ct:<16}{r.get('pacemaker', np.nan):>+24.3f}"
          f"{r.get('atrial_myocyte', np.nan):>+22.3f}{mk}")

san = float(piv.loc["SAN_P_cell", "pacemaker"])
ceil_ct = "aCM4" if "aCM4" in piv.index else "aCM3"
ceil = float(piv.loc[ceil_ct, "atrial_myocyte"])
others = piv.drop(index=["SAN_P_cell"])["pacemaker"].dropna()
print(f"\npacemaker abundance vs pacemaker markers : {san:+.3f}")
print(f"every other cell type, same markers      : mean {others.mean():+.3f}, "
      f"best {others.max():+.3f} ({others.idxmax()})")
print(f"{ceil_ct} vs atrial markers                 : {ceil:+.3f}   [ceiling, easy case]")

beats = san > others.max()
verdict = "USABLE" if beats else "QUESTIONABLE"
print("\n" + "=" * 90)
print(f"VERDICT: the SAN_P_cell abundance estimate is {verdict}")
print("=" * 90)
if beats:
    print("It outperforms every other cell type on the marker check, so it tracks real")
    print("pacemaker transcriptional content rather than generic cellularity. It reaches")
    print(f"{san / ceil * 100:.0f}% of what the method achieves on an abundant, easy population")
    print("— weaker, as expected for a rare type, but not noise.")
    print("=> the null result built on this estimate stands.")
else:
    print("It does NOT beat the other cell types on the marker check. Any conclusion")
    print("resting on this abundance estimate must be qualified, and the cell-type")
    print("line of evidence downgraded relative to the gene-level one, which does not")
    print("use deconvolution at all.")

with open(f"{OUT}/deconv_quality.json", "w") as f:
    json.dump(dict(san_marker_rho=san, ceiling_rho=ceil, ceiling_type=ceil_ct,
                   other_marker_rho_mean=float(others.mean()),
                   other_marker_rho_max=float(others.max()),
                   other_best=str(others.idxmax()),
                   node_concentration_rank=rank, n_celltypes=len(cm),
                   frac_zero=float(dist.frac_zero.mean()),
                   share=float(dist.share.mean()), verdict=verdict), f, indent=2)
print(f"\nwrote {OUT}/deconv_*.tsv and deconv_quality.json")
