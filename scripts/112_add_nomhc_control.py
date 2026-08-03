"""
Rebuild the rheumatoid arthritis gene set without the MHC.
==========================================================

Rheumatoid arthritis was brought in for one job: to be a well-powered, non-cardiac,
non-neural trait, so that "does a strong polygenic signal favour pacemaker cells
regardless of what the trait is about" could be answered. Educational attainment
cannot answer it, being a neural trait aimed at the most neuron-like cell in the heart.

But 34.3% of the RA gene set's weight sits in a single 1-Mb clump, and that clump is
the MHC (34.2% of total weight on chr6:25-34 Mb). The MHC is gene-dense and in extended
linkage disequilibrium, so MAGMA assigns high Z to dozens of neighbouring genes from
what is effectively one signal. An RA score built that way mostly asks which cells
express MHC-region genes — a question with an obvious answer in any tissue containing
antigen-presenting cells.

That would still put RA on immune cells and away from pacemaker cells, which is the
direction needed. But the inference would be weak: a critic could say pacemaker cells
score low for RA because RA is an MHC score, not because they are indifferent to
polygenic signal in general. Removing the region and rebuilding from the next best
genes gives a control that carries the intended meaning.

The original RA set is kept as well — the two together show whether the MHC was doing
the work.

Usage:  python scripts/112_add_nomhc_control.py
"""

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.as_posix()
GS = f"{ROOT}/data/scdrs/traits.gs"
GENES = os.path.expanduser("~/cardio/magma/genes/RheumatoidArthritis.genes.out")
GENELOC = os.path.expanduser("~/cardio/tools/NCBI37.3.gene.loc")
NAME = "RheumatoidArthritis_noMHC"
MHC = (6, 25_000_000, 34_000_000)
TOP = 1000

if not os.path.exists(GENES):
    raise SystemExit(f"{GENES} not found — MAGMA has not scored rheumatoid arthritis")

loc = pd.read_csv(GENELOC, sep=r"\s+", header=None,
                  names=["gene", "chr", "start", "stop", "strand", "symbol"],
                  dtype={"gene": str})
sym = dict(zip(loc.gene, loc.symbol))
pos = loc.set_index("gene")[["chr", "start"]]

d = pd.read_csv(GENES, sep=r"\s+")
d["symbol"] = d.GENE.astype(str).map(sym)
d = d.join(pos, on=d.GENE.astype(str))
d = d.dropna(subset=["symbol", "ZSTAT"]).drop_duplicates("symbol")
d["chr"] = pd.to_numeric(d.chr, errors="coerce")

in_mhc = (d.chr == MHC[0]) & (d.start.between(MHC[1], MHC[2]))
print(f"genes scored: {len(d):,}")
print(f"in the MHC window chr6:{MHC[1]/1e6:.0f}-{MHC[2]/1e6:.0f} Mb: {int(in_mhc.sum())}")

before = d.nlargest(TOP, "ZSTAT")
print(f"\noriginal top {TOP}: {int(before.index.isin(d[in_mhc].index).sum())} "
      f"are MHC genes, max Z {before.ZSTAT.max():.2f}")

kept = d[~in_mhc]
after = kept.nlargest(TOP, "ZSTAT")
after = after[after.ZSTAT > 0]
print(f"after removing the MHC: {len(after)} genes, max Z {after.ZSTAT.max():.2f}, "
      f"median Z {after.ZSTAT.median():.2f}")
print(f"chromosomes represented: {int(after.chr.nunique())}")

gs = pd.read_csv(GS, sep="\t")
gs = gs[gs.TRAIT != NAME]
row = pd.DataFrame([dict(
    TRAIT=NAME,
    GENESET=",".join(f"{s}:{z:.4f}" for s, z in zip(after.symbol, after.ZSTAT)))])
gs = pd.concat([gs, row], ignore_index=True)
gs.to_csv(GS, sep="\t", index=False)
print(f"\nwrote {GS} — now {len(gs)} traits, including {NAME}")
print("\nRun 103_run_scdrs.py to score it; the cache means only this one is computed.")
