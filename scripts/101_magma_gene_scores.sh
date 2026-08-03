#!/usr/bin/env bash
# Gene-level association scores for every trait, then scDRS gene sets.
#
# scDRS needs, per trait, a set of putative disease genes with weights. The standard
# input is MAGMA's per-gene Z statistic, which aggregates the SNPs in a gene window
# while accounting for the LD among them.
#
# Run per chromosome rather than genome-wide
# ------------------------------------------
# The reference panel ships as 22 per-chromosome plink filesets and MAGMA's --bfile
# takes one. Merging would need plink, which is not installed. Running per chromosome
# is exactly equivalent for what is needed here: no gene spans a chromosome boundary,
# so every gene's Z comes from the same SNPs either way. What per-chromosome splitting
# gives up is the genome-wide gene-gene correlation used by MAGMA's gene-SET analysis,
# and scDRS consumes per-gene scores, never gene sets.
#
# Two things learned from the first attempt
# -----------------------------------------
# * MAGMA re-reads the ENTIRE p-value file on every invocation. With 5 million SNPs and
#   22 invocations that was most of the wall clock. The file is now split by chromosome
#   once per trait, so each run reads ~1/22 as much.
# * Traits ran alphabetically, which put the negative control second and the actual
#   hypothesis last. Order is now by importance, so a partial run is still useful:
#   the positive control and the hypothesis land first.
#
# Resume is per chromosome, so an interrupted trait does not restart from scratch.
#
#   wsl -d Ubuntu -e bash .../101_magma_gene_scores.sh
#   TRAITS=RestingHeartRate,HRV_RMSSDc bash .../101_magma_gene_scores.sh

set -uo pipefail
LIN="$HOME/cardio"
TOOLS="$LIN/tools"
REF="$LIN/resource/gsMap_resource/LD_Reference_Panel/1000G_EUR_Phase3_plink"
WIN="${CARDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GWAS="$WIN/data/gwas_gsmap"
WORK="$LIN/magma"
OUTGS="$WIN/data/scdrs"
mkdir -p "$WORK/annot" "$WORK/genes" "$WORK/pval" "$OUTGS"
MAGMA="$TOOLS/magma"
GENELOC="$TOOLS/NCBI37.3.gene.loc"

# positive control first, then the hypothesis, then the negative control, then the
# mechanistically-specific traits, then the underpowered ones
PRIORITY=(AtrialFibrillation RestingHeartRate EducationalAttainment
          HRV_RMSSDc HRV_RMSSD PRinterval HRV_SDNN HRV_SDNNc)

if [ -n "${TRAITS:-}" ]; then
  IFS=',' read -ra T <<< "$TRAITS"
else
  T=()
  for t in "${PRIORITY[@]}"; do
    [ -f "$GWAS/${t}.sumstats.gz" ] && T+=("$t")
  done
  for f in "$GWAS"/*.sumstats.gz; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); b="${b%.sumstats.gz}"
    [[ " ${T[*]} " == *" $b "* ]] || T+=("$b")
  done
fi
echo "traits, in priority order: ${T[*]}"

# ---------------------------------------------------------------- 1. annotate
echo
echo "=== step 1: SNP -> gene annotation (reference-only, computed once)"
for c in $(seq 1 22); do
  OUT="$WORK/annot/chr${c}"
  [ -f "${OUT}.genes.annot" ] && continue
  awk '{print $2"\t"$1"\t"$4}' "$REF/1000G.EUR.QC.${c}.bim" > "$WORK/annot/chr${c}.snploc"
  "$MAGMA" --annotate window=10,10 \
           --snp-loc "$WORK/annot/chr${c}.snploc" \
           --gene-loc "$GENELOC" --out "$OUT" > /dev/null 2>&1
done
echo "  ready: $(ls "$WORK"/annot/*.genes.annot 2>/dev/null | wc -l)/22 chromosomes"

# ---------------------------------------------------------------- 2. gene analysis
source "$LIN/venv/bin/activate"
for t in "${T[@]}"; do
  GENESOUT="$WORK/genes/${t}.genes.out"
  if [ -f "$GENESOUT" ] && [ "$(wc -l < "$GENESOUT")" -gt 10000 ]; then
    echo; echo "=== $t: already complete ($(( $(wc -l < "$GENESOUT") - 1 )) genes)"
    continue
  fi
  echo; echo "=== $t   $(date '+%T')"
  START=$(date +%s)

  # split the p-values by chromosome so MAGMA stops re-reading five million rows
  if [ ! -f "$WORK/pval/${t}.chr22.pval" ]; then
    python - "$GWAS/${t}.sumstats.gz" "$REF" "$WORK/pval" "$t" <<'PY'
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

src, ref, out, trait = sys.argv[1:5]
d = pd.read_csv(src, sep="\t", usecols=["SNP", "Z", "N"])
d = d[np.isfinite(d.Z)]
# logsf keeps the tail exact where 2*sf would underflow to 0 and MAGMA would then
# discard the strongest SNPs in the study
d["P"] = np.exp(np.log(2) + norm.logsf(np.abs(d.Z.to_numpy()))).clip(1e-300, 1.0)
d = d.set_index("SNP")
tot = 0
for c in range(1, 23):
    bim = pd.read_csv(f"{ref}/1000G.EUR.QC.{c}.bim", sep="\t", header=None,
                      usecols=[1], names=["SNP"])
    sub = d.reindex(bim.SNP.to_numpy()).dropna(subset=["P"])
    sub.reset_index()[["SNP", "P", "N"]].to_csv(
        f"{out}/{trait}.chr{c}.pval", sep="\t", index=False)
    tot += len(sub)
print(f"    {len(d):,} SNPs -> {tot:,} matched to the reference, "
      f"min P = {d.P.min():.3g}")
PY
  fi

  # Chromosomes are independent — no gene spans a boundary and each run reads its own
  # p-value slice — so they parallelise cleanly. Serially this was ~3 chromosomes per
  # ten minutes, which put eight traits at five to nine hours. JOBS defaults to leaving
  # two cores for whatever else is running.
  JOBS="${JOBS:-$(( $(nproc) > 3 ? $(nproc) - 2 : 1 ))}"
  echo "    running 22 chromosomes ${JOBS} at a time"
  run_chr() {
    local c="$1"
    [ -f "$WORK/genes/${t}.chr${c}.genes.out" ] && return 0
    "$MAGMA" --bfile "$REF/1000G.EUR.QC.${c}" \
             --pval "$WORK/pval/${t}.chr${c}.pval" ncol=N \
             --gene-annot "$WORK/annot/chr${c}.genes.annot" \
             --out "$WORK/genes/${t}.chr${c}" > /dev/null 2>&1
  }
  # largest chromosomes first so the long pole starts immediately rather than last
  for c in 1 2 3 6 5 4 7 11 12 17 19 8 10 9 16 14 15 20 22 13 18 21; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    run_chr "$c" &
  done
  wait

  head -1 "$WORK/genes/${t}.chr1.genes.out" 2>/dev/null > "$GENESOUT"
  for c in $(seq 1 22); do
    tail -n +2 "$WORK/genes/${t}.chr${c}.genes.out" 2>/dev/null >> "$GENESOUT"
  done
  n=$(( $(wc -l < "$GENESOUT") - 1 ))
  if [ "$n" -gt 10000 ]; then
    rm -f "$WORK/genes/${t}.chr"*.genes.out "$WORK/genes/${t}.chr"*.genes.raw \
          "$WORK/genes/${t}.chr"*.log "$WORK/pval/${t}.chr"*.pval
    echo "    $(( ($(date +%s) - START) / 60 )) min, ${n} genes"
  else
    echo "    !! only ${n} genes — leaving intermediates for inspection"
  fi
done

# ---------------------------------------------------------------- 3. scDRS gene sets
echo
echo "=== step 3: build scDRS gene sets"
python - "$WORK/genes" "$GENELOC" "$OUTGS/traits.gs" <<'PY'
import glob
import os
import sys

import pandas as pd

genes_dir, geneloc, out = sys.argv[1], sys.argv[2], sys.argv[3]

# MAGMA reports Entrez IDs; the single-cell object is indexed by HGNC symbol
loc = pd.read_csv(geneloc, sep=r"\s+", header=None,
                  names=["gene", "chr", "start", "stop", "strand", "symbol"],
                  dtype={"gene": str})
sym = dict(zip(loc.gene, loc.symbol))

import re

rows = []
for p in sorted(glob.glob(os.path.join(genes_dir, "*.genes.out"))):
    if re.search(r"\.chr\d+\.genes\.out$", p):     # per-chromosome intermediate
        continue
    trait = os.path.basename(p).replace(".genes.out", "")
    d = pd.read_csv(p, sep=r"\s+")
    if len(d) < 10000:
        print(f"  {trait:<24}SKIPPED — only {len(d)} genes, run looks incomplete")
        continue
    d["symbol"] = d.GENE.astype(str).map(sym)
    d = d.dropna(subset=["symbol", "ZSTAT"]).drop_duplicates("symbol")
    top = d.nlargest(1000, "ZSTAT")
    top = top[top.ZSTAT > 0]          # a negative Z carries no disease signal
    rows.append(dict(TRAIT=trait,
                     GENESET=",".join(f"{s}:{z:.4f}"
                                      for s, z in zip(top.symbol, top.ZSTAT))))
    print(f"  {trait:<24}{len(d):>7,} genes scored, top set {len(top):>5}, "
          f"max Z {d.ZSTAT.max():.2f}")

if rows:
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"\nwrote {out}  ({len(rows)} traits)")
else:
    print("\nno trait produced a usable gene set")
PY

echo
echo "finished $(date '+%F %T')"
