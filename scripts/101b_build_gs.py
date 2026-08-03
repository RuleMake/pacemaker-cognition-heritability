"""
Build the scDRS gene-set file from whatever MAGMA has finished so far.
=====================================================================

Step 3 of 101_magma_gene_scores.sh only runs after all eight traits complete, roughly
two hours. But the single most important question — does the pipeline recover atrial
fibrillation in atrial cardiomyocytes — can be answered from the first trait alone,
and there is no reason to learn the answer two hours late. If the positive control
fails, the remaining seven traits are not worth computing yet.

So this does the same conversion on demand, over whatever .genes.out files exist.

MAGMA reports Entrez gene IDs; the single-cell object is indexed by HGNC symbol, so
the mapping goes through the same gene location file MAGMA was given. Genes with a
negative Z are dropped: scDRS weights genes by evidence for the trait, and a negative
statistic is evidence of nothing.

Usage:  python scripts/101b_build_gs.py
        python scripts/101b_build_gs.py --min-genes 5000   (accept a partial run)
"""

import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.as_posix()
GENES = os.path.expanduser("~/cardio/magma/genes")
GENELOC = os.path.expanduser("~/cardio/tools/NCBI37.3.gene.loc")
OUT = f"{ROOT}/data/scdrs/traits.gs"

ap = argparse.ArgumentParser()
ap.add_argument("--min-genes", type=int, default=10000,
                help="reject a trait whose MAGMA run produced fewer genes than this")
ap.add_argument("--top", type=int, default=1000, help="gene set size")
args = ap.parse_args()

if not os.path.isdir(GENES):
    raise SystemExit(f"{GENES} does not exist — MAGMA has not run")

loc = pd.read_csv(GENELOC, sep=r"\s+", header=None,
                  names=["gene", "chr", "start", "stop", "strand", "symbol"],
                  dtype={"gene": str})
sym = dict(zip(loc.gene, loc.symbol))

rows = []
# per-chromosome intermediates are named <trait>.chr<N>.genes.out and would otherwise
# be picked up as if each were its own trait
files = [p for p in sorted(glob.glob(os.path.join(GENES, "*.genes.out")))
         if not re.search(r"\.chr\d+\.genes\.out$", p)]
if not files:
    raise SystemExit(f"no completed traits yet in {GENES} "
                     "(only per-chromosome intermediates)")

for p in files:
    trait = os.path.basename(p).replace(".genes.out", "")
    d = pd.read_csv(p, sep=r"\s+")
    if len(d) < args.min_genes:
        print(f"  {trait:<24}SKIPPED — {len(d):,} genes, below --min-genes "
              f"{args.min_genes:,} (run still in progress?)")
        continue
    d["symbol"] = d.GENE.astype(str).map(sym)
    d = d.dropna(subset=["symbol", "ZSTAT"]).drop_duplicates("symbol")
    top = d.nlargest(args.top, "ZSTAT")
    top = top[top.ZSTAT > 0]
    rows.append(dict(TRAIT=trait,
                     GENESET=",".join(f"{s}:{z:.4f}"
                                      for s, z in zip(top.symbol, top.ZSTAT))))
    print(f"  {trait:<24}{len(d):>7,} genes scored, set {len(top):>5}, "
          f"max Z {d.ZSTAT.max():>6.2f}, median Z {d.ZSTAT.median():>5.2f}")

if not rows:
    raise SystemExit("nothing usable yet")

pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
print(f"\nwrote {OUT}  ({len(rows)} trait(s))")
