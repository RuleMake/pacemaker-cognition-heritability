"""
Is the control signal in sinoatrial pacemaker cells just gene length?
=====================================================================

Educational attainment scores 0.674 against the working myocytes of the sinoatrial
node — higher than every cardiac trait except one. Rheumatoid arthritis is flat, so it
is not generic polygenicity. But there is a specific, well-documented mechanism that
has not been ruled out, and the evidence pointing at it is uncomfortably direct.

MAGMA assigns a gene its Z from the SNPs inside it. Long genes contain more SNPs, so
for a highly polygenic trait they accumulate more evidence, and the MAGMA paper names
Cell Adhesion Molecules as a gene set whose significance disappears once gene size and
density are corrected. The top drivers of educational attainment in SAN_P_cell in this
project are CTNNA3, CADM2, CDH13, CDH2 and ERBB4 — cell adhesion molecules, and among
the longest genes in the genome. CTNNA3 spans 1.8 Mb; CADM2 spans 1.1 Mb.

If long genes are also preferentially detected in pacemaker cells, the control signal
is an artefact of gene length twice over and the whole objection dissolves. If it
survives correction, then it is real biology — and there is literature for that too:
sinoatrial pacemaker cells co-cluster with cortical neurons in single-cell space and
carry a glutamatergic programme (Protein & Cell 2021), which would make a cognitive
trait's heritability landing there a finding rather than a nuisance.

Either answer is worth having, which is why this is worth an hour.

  A  Does MAGMA Z track gene length for each trait?
  B  Do pacemaker cells preferentially express long genes, relative to the myocytes
     they are compared against?
  C  Rebuild every gene set from length- and SNP-count-adjusted residuals, so the
     ranking cannot reward a gene for merely being big.

Part C appends `<trait>_lenadj` sets to traits.gs; scDRS then scores only those.

Usage:  python scripts/160_gene_length_confound.py
"""

import glob
import json
import os
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.as_posix()
GENES = os.path.expanduser("~/cardio/magma/genes")
GENELOC = os.path.expanduser("~/cardio/tools/NCBI37.3.gene.loc")
GS = f"{ROOT}/data/scdrs/traits.gs"
H5 = f"{ROOT}/data/singlecell/axis_subset.h5ad"
OUT = f"{ROOT}/results/gene_length_confound.json"

FOCUS = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
TOP = 1000

loc = pd.read_csv(GENELOC, sep=r"\s+", header=None,
                  names=["gene", "chr", "start", "stop", "strand", "symbol"],
                  dtype={"gene": str})
sym = dict(zip(loc.gene, loc.symbol))

files = [p for p in sorted(glob.glob(os.path.join(GENES, "*.genes.out")))
         if not re.search(r"\.chr\d+\.genes\.out$", p)]
tables = {}
for p in files:
    t = os.path.basename(p).replace(".genes.out", "")
    d = pd.read_csv(p, sep=r"\s+")
    d["symbol"] = d.GENE.astype(str).map(sym)
    d = d.dropna(subset=["symbol", "ZSTAT"]).drop_duplicates("symbol")
    d["length"] = (d.STOP - d.START).clip(lower=1)
    d["loglen"] = np.log10(d.length)
    d["lognsnp"] = np.log10(d.NSNPS.clip(lower=1))
    tables[t] = d
print(f"{len(tables)} traits with MAGMA gene scores\n")

# ================================================================ A
print("=" * 96)
print("A. DOES MAGMA Z TRACK GENE LENGTH?  (Spearman rho over ~18,000 genes)")
print("=" * 96)
print(f"{'trait':<28}{'rho(Z, length)':>16}{'rho(Z, nSNP)':>15}"
      f"{'median len of top-1000':>26}{'vs all':>10}")
resA = {}
for t, d in sorted(tables.items()):
    r1 = stats.spearmanr(d.ZSTAT, d.loglen).statistic
    r2 = stats.spearmanr(d.ZSTAT, d.lognsnp).statistic
    top = d.nlargest(TOP, "ZSTAT")
    resA[t] = dict(rho_len=float(r1), rho_nsnp=float(r2),
                   median_len_top=float(top.length.median()),
                   median_len_all=float(d.length.median()))
    print(f"{t:<28}{r1:>16.3f}{r2:>15.3f}{top.length.median() / 1000:>22,.0f} kb"
          f"{d.length.median() / 1000:>8,.0f} kb")
print("\nA positive rho means longer genes get higher Z. Every trait shows some of")
print("this — it is how the statistic works. What matters is whether it is worse for")
print("the control than for the cardiac traits.")

# ================================================================ B
print("\n" + "=" * 96)
print("B. DO PACEMAKER CELLS PREFERENTIALLY EXPRESS LONG GENES?")
print("=" * 96)
a = ad.read_h5ad(H5)
cs = a.obs["cell_state"].astype(str).to_numpy()
X = sp.csr_matrix(a.X) if not sp.issparse(a.X) else a.X.tocsr()
glen = pd.Series({r.symbol: r.length for r in
                  loc.assign(length=(loc.stop - loc.start).clip(lower=1))
                  .itertuples()})
have = pd.Index(a.var_names).intersection(glen.index)
gi = {g: i for i, g in enumerate(a.var_names)}
idx = [gi[g] for g in have]
L = np.log10(glen[have].to_numpy(dtype=float))

is_cm = pd.Series(cs).str.match(r"^(aCM|vCM)").to_numpy()
print(f"{'group':<22}{'n':>7}{'rho(mean expr, log length)':>30}")
resB = {}
for name, m in [(c, cs == c) for c in FOCUS] + [("working myocytes", is_cm),
                                                ("all other cells", ~is_cm)]:
    if m.sum() == 0:
        continue
    mu = np.asarray(X[m][:, idx].mean(axis=0)).ravel()
    ok = mu > 0
    rho = stats.spearmanr(mu[ok], L[ok]).statistic
    resB[name] = float(rho)
    print(f"{name:<22}{int(m.sum()):>7,}{rho:>30.3f}")
print("\nIf a pacemaker cell's expression is more long-gene-weighted than the myocytes")
print("it is compared with, a long-gene-heavy set scores higher there for no")
print("biological reason at all.")

# ================================================================ C
print("\n" + "=" * 96)
print("C. REBUILDING EVERY GENE SET FROM LENGTH-ADJUSTED RESIDUALS")
print("=" * 96)
gs = pd.read_csv(GS, sep="\t")
existing = set(gs.TRAIT)
rows = []
resC = {}
for t, d in sorted(tables.items()):
    # residual of Z after removing what length and SNP count explain; a gene now has
    # to beat the expectation for a gene of its size rather than merely be large
    Xd = np.column_stack([np.ones(len(d)), d.loglen.to_numpy(), d.lognsnp.to_numpy()])
    beta, *_ = np.linalg.lstsq(Xd, d.ZSTAT.to_numpy(), rcond=None)
    d = d.assign(resid=d.ZSTAT.to_numpy() - Xd @ beta)
    top = d.nlargest(TOP, "resid")
    top = top[top.resid > 0]
    old = set(d.nlargest(TOP, "ZSTAT").symbol)
    kept = len(old & set(top.symbol)) / max(len(old), 1)
    resC[t] = dict(overlap_with_original=float(kept), n=len(top),
                   median_len=float(top.length.median()))
    print(f"{t:<28}{len(top):>6} genes   overlap with original {kept * 100:>5.1f}%"
          f"   median length {top.length.median() / 1000:>6,.0f} kb")
    name = f"{t}_lenadj"
    if name not in existing:
        # weights are the residuals rescaled to the original Z range, so scDRS sees a
        # comparable weight distribution rather than a differently-scaled one
        w = top.resid.to_numpy()
        w = w / w.max() * float(d.ZSTAT.max())
        rows.append(dict(TRAIT=name,
                         GENESET=",".join(f"{s}:{z:.4f}"
                                          for s, z in zip(top.symbol, w))))

if rows:
    pd.concat([gs, pd.DataFrame(rows)]).to_csv(GS, sep="\t", index=False)
    print(f"\nappended {len(rows)} length-adjusted sets -> {GS}")
else:
    print("\nlength-adjusted sets already present")

with open(OUT, "w") as f:
    json.dump(dict(magma=resA, expression=resB, rebuilt=resC), f, indent=2)
print(f"wrote {OUT}")
print("\nNext: score them (scDRS caches, so only the new sets are computed)")
print("  python scripts/103_run_scdrs.py --h5 data/singlecell/axis_subset.h5ad \\")
print("      --cov data/scdrs/covariates_axis.tsv --out results/scdrs_axis")
