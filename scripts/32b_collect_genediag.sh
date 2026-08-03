#!/usr/bin/env bash
# Pull gsMap's per-trait gene diagnostic tables out of WSL.
#
# gsMap writes one of these per (section, trait) during the report step. Each row is a
# gene with:
#   Annotation  the compartment where that gene's spot-specificity score peaks
#   Median_GSS  its median gene-specificity score
#   PCC         correlation between the gene's GSS across spots and -log10(p) across spots
#
# PCC is the interesting column: it says which genes actually DRIVE the spatial pattern
# of heritability enrichment. Nothing so far has looked at it, so the biology behind the
# node signal has never been inspected — only its geometry.
#
#   bash 32b_collect_genediag.sh

set -euo pipefail
WORK="$HOME/cardio/work"
DEST="$CARDIO_ROOT/results/genediag"
mkdir -p "$DEST"

n=0
for f in "$WORK"/*/*/report/*/*_Gene_Diagnostic_Info.csv; do
  [ -e "$f" ] || continue
  cp "$f" "$DEST/$(basename "$f")"
  n=$((n+1))
done
echo "copied $n gene-diagnostic tables"
du -sh "$DEST"
