"""
Is each gene set genuinely polygenic, or a single locus in disguise?
====================================================================

scDRS weights the top 1,000 genes by MAGMA Z. Genes in linkage disequilibrium inherit
each other's signal, so a single strong locus can contribute dozens of neighbouring
genes and a "polygenic score" becomes a very expensive way of scoring one region. The
classic offenders are the MHC on chromosome 6 for immune traits and PITX2 on 4q25 for
atrial fibrillation.

The fragility test in 107 already showed that dropping the strongest 50 genes leaves
most of the effect, which argues against single-locus dominance. This measures it
directly instead of inferring it:

  chromosome spread   how evenly the set is distributed. A set concentrated on one
                      chromosome is suspect.
  locus clumping      genes within 1 Mb of each other collapsed into one locus. The
                      number of independent loci, and the share of total weight held
                      by the largest one, is the number that matters.
  MHC share           chr6:25-34 Mb specifically, because it is large, gene-dense and
                      in extended LD, and it inflates immune gene sets in particular.

Usage:  python scripts/111_audit_geneset_spread.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.as_posix()
GS = f"{ROOT}/data/scdrs/traits.gs"
GENELOC = os.path.expanduser("~/cardio/tools/NCBI37.3.gene.loc")
OUT = f"{ROOT}/results/scdrs"

CLUMP_KB = 1000       # genes within 1 Mb are treated as one locus
MHC = (6, 25_000_000, 34_000_000)

loc = pd.read_csv(GENELOC, sep=r"\s+", header=None,
                  names=["gene", "chr", "start", "stop", "strand", "symbol"],
                  dtype={"gene": str})
loc = loc.drop_duplicates("symbol").set_index("symbol")
gs = pd.read_csv(GS, sep="\t")
print(f"traits: {len(gs)}\n")

print("=" * 100)
print(f"{'trait':<24}{'genes':>7}{'chrs':>6}{'loci':>7}{'largest locus':>15}"
      f"{'top chr share':>15}{'MHC share':>11}   verdict")
print("=" * 100)
report = {}
for _, row in gs.iterrows():
    trait = row.TRAIT
    pairs = [p.split(":") for p in str(row.GENESET).split(",") if ":" in p]
    d = pd.DataFrame([(s, float(w)) for s, w in pairs], columns=["symbol", "w"])
    d = d.join(loc[["chr", "start"]], on="symbol").dropna(subset=["chr"])
    if d.empty:
        continue
    d["chr"] = pd.to_numeric(d.chr, errors="coerce")
    d = d.dropna(subset=["chr"])
    tot = d.w.sum()

    # clump: sort by position, start a new locus whenever the gap exceeds CLUMP_KB
    d = d.sort_values(["chr", "start"]).reset_index(drop=True)
    newloc = (d.chr != d.chr.shift()) | (d.start - d.start.shift() > CLUMP_KB * 1000)
    d["locus"] = newloc.cumsum()
    per_locus = d.groupby("locus").w.sum().sort_values(ascending=False)
    largest = float(per_locus.iloc[0] / tot)
    top_chr = float(d.groupby("chr").w.sum().max() / tot)
    mhc = d[(d.chr == MHC[0]) & (d.start.between(MHC[1], MHC[2]))]
    mhc_share = float(mhc.w.sum() / tot)

    ok = largest < 0.05 and top_chr < 0.20 and mhc_share < 0.10
    print(f"{trait:<24}{len(d):>7}{int(d.chr.nunique()):>6}{len(per_locus):>7}"
          f"{largest * 100:>14.1f}%{top_chr * 100:>14.1f}%{mhc_share * 100:>10.1f}%"
          f"   {'polygenic' if ok else 'CONCENTRATED'}")
    report[trait] = dict(n_genes=int(len(d)), n_chr=int(d.chr.nunique()),
                         n_loci=int(len(per_locus)), largest_locus=largest,
                         top_chr_share=top_chr, mhc_share=mhc_share, ok=bool(ok))

print("\nlargest locus = share of total gene weight held by one 1-Mb clump.")
print("A genuinely polygenic set spreads across hundreds of loci with no single one")
print("above a few percent. Anything concentrated is scoring a region, not a trait.")

bad = [t for t, v in report.items() if not v["ok"]]
print(f"\n{len(report) - len(bad)} of {len(report)} sets look polygenic")
if bad:
    print("concentrated sets — interpret their cell-type results with care:")
    for t in bad:
        v = report[t]
        print(f"  {t:<24}largest locus {v['largest_locus'] * 100:.1f}%, "
              f"top chromosome {v['top_chr_share'] * 100:.1f}%, "
              f"MHC {v['mhc_share'] * 100:.1f}%")

with open(f"{OUT}/geneset_spread.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwrote {OUT}/geneset_spread.json")
