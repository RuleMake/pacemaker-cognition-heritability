#!/usr/bin/env bash
# Install scDRS and MAGMA for the single-cell test of the pacemaker hypothesis.
#
# Why scDRS is the right tool here
# --------------------------------
# Every confound fought so far is a property of Visium spots or of GWAS traits:
#   spot depth, spot gene diversity, trait power, trait polygenicity.
# scDRS is built to neutralise exactly these:
#   * control gene sets are matched to the disease gene set on mean expression AND
#     expression variance, which is the complexity confound by construction;
#   * the null is generated per trait by Monte Carlo, so a trait's power and
#     polygenicity cannot inflate its score relative to another trait;
#   * cells are cells — no 55 um mixing, no deconvolution estimate to trust.
#
# And the annotation survived: CELLxGENE kept Kanemaru's `cell_state` (76 categories),
# including SAN_P_cell n=245, AVN_P_cell n=155, Purkinje n=110, AVN_bundle_cell n=38.
#
# MAGMA supplies the gene-level scores scDRS consumes. Its reference panel is already
# on disk — gsMap shipped 1000G EUR Phase 3 in plink format, 9,997,231 variants across
# 22 chromosomes — so only the binary and a gene location file are missing.

set -uo pipefail
LIN="$HOME/cardio"
RES="$LIN/resource/gsMap_resource"
TOOLS="$LIN/tools"
mkdir -p "$TOOLS"
source "$LIN/venv/bin/activate"

echo "=============================================================="
echo " 1. scdrs"
echo "=============================================================="
if python -c "import scdrs" 2>/dev/null; then
  echo "already installed: $(python -c 'import scdrs;print(scdrs.__version__)')"
else
  pip install scdrs
  python -c "import scdrs; print('scdrs', scdrs.__version__)"
fi

echo
echo "=============================================================="
echo " 2. magma"
echo "=============================================================="
if [ -x "$TOOLS/magma" ]; then
  echo "already present"
else
  cd "$TOOLS"
  for U in \
    "https://vu.nl/en/download/cbc7d1a4-2c22-4b0d-97e6-8b3e9cbbc0cd" \
    "https://ctg.cncr.nl/software/MAGMA/prog/magma_v1.10.zip" \
    "https://vu.nl/en/download/57e4a1f9-79dd-4b5f-8b8a-4b1e2e8e5b3f"
  do
    echo "trying $U"
    curl -fsSL --retry 3 -o magma.zip "$U" && break
  done
  if [ -f magma.zip ]; then
    unzip -o magma.zip >/dev/null 2>&1 && chmod +x magma 2>/dev/null
  fi
  if [ -x "$TOOLS/magma" ]; then
    echo "installed: $("$TOOLS/magma" --version 2>&1 | head -1)"
  else
    echo "!! magma could not be fetched automatically."
    echo "   Not fatal — 101_make_gene_scores.py falls back to a windowed"
    echo "   gene score computed directly from the summary statistics."
  fi
fi

echo
echo "=============================================================="
echo " 3. gene location file for MAGMA"
echo "=============================================================="
BED="$RES/genome_annotation/gtf/genecode_v46lift37_protein_coding.bed"
LOC="$TOOLS/gene_loc.txt"
if [ -f "$BED" ]; then
  echo "source bed:"
  head -2 "$BED"
  awk -F'\t' 'NR>0 && $1 ~ /^chr[0-9]+$/ {
        chr=$1; sub(/^chr/,"",chr);
        name = ($4 != "") ? $4 : $5;
        if (name != "") print name"\t"chr"\t"$2"\t"$3
     }' "$BED" | sort -u -k1,1 > "$LOC"
  echo "wrote $LOC : $(wc -l < "$LOC") genes"
  head -3 "$LOC"
else
  echo "!! protein-coding bed not found at $BED"
fi

echo
echo "=============================================================="
echo " 4. plink reference check"
echo "=============================================================="
P="$RES/LD_Reference_Panel/1000G_EUR_Phase3_plink"
echo "per-chromosome bfiles: $(ls "$P"/*.bed 2>/dev/null | wc -l)"
echo "variants total       : $(cat "$P"/*.bim 2>/dev/null | wc -l)"
echo
echo "setup finished $(date '+%F %T')"
