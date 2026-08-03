#!/usr/bin/env bash
# What is in gsMap's enhancer and gtf resource, and is MAGMA already satisfiable?
#
# Two things turned up while planning the next round:
#   * genome_annotation/enhancer exists — gsMap accepts --enhancer_annotation_file,
#     and cardiac common variants sit overwhelmingly in enhancers rather than gene
#     bodies, so the default gene-window SNP-to-gene mapping may be the reason
#     conduction genes never surfaced.
#   * LD_Reference_Panel is 1000G EUR Phase 3 in plink format, which is exactly what
#     MAGMA needs as its reference, so scDRS gene scoring needs only the MAGMA binary
#     and a gene location file rather than another 700 MB download.
set -uo pipefail
R="$HOME/cardio/resource/gsMap_resource"

echo "=== enhancer annotation"
ls -la "$R/genome_annotation/enhancer/" 2>/dev/null
for f in "$R"/genome_annotation/enhancer/*; do
  [ -e "$f" ] || continue
  echo "--- $(basename "$f")   ($(du -h "$f" | cut -f1), $(wc -l < "$f") lines)"
  head -3 "$f"
done

echo
echo "=== gtf"
ls -la "$R/genome_annotation/gtf/" 2>/dev/null
for f in "$R"/genome_annotation/gtf/*; do
  [ -e "$f" ] || continue
  echo "--- $(basename "$f")   ($(du -h "$f" | cut -f1))"
done

echo
echo "=== plink reference (for MAGMA)"
ls "$R/LD_Reference_Panel/1000G_EUR_Phase3_plink/" | head -4
echo "chromosomes present: $(ls "$R"/LD_Reference_Panel/1000G_EUR_Phase3_plink/*.bim | wc -l)"
echo "total variants: $(cat "$R"/LD_Reference_Panel/1000G_EUR_Phase3_plink/*.bim | wc -l)"

echo
echo "=== is scdrs / magma already available?"
source "$HOME/cardio/venv/bin/activate"
python -c "import scdrs; print('scdrs', scdrs.__version__)" 2>/dev/null || echo "scdrs: NOT installed"
command -v magma >/dev/null && magma --version 2>&1 | head -1 || echo "magma: NOT installed"
