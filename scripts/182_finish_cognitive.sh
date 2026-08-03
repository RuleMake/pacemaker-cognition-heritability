#!/usr/bin/env bash
# Finish the cognitive arm once the length-adjusted scoring releases the machine.
#
# Four things are queued behind it, in dependency order. Running them alongside scDRS
# is what got the previous attempt killed by the kernel with no traceback, so this
# waits rather than competing.
#
#   171  the glutamatergic test — redoes the refuted neuronal-programme test with the
#        gene set the literature actually names, instead of the heart's glia
#   MAGMA + gene sets for the two cognitive traits, so they can be scored per cell
#   170  genetic correlation, re-run with intelligence and reaction time included
#   181  bidirectional MR
#
# 172 (spatial power) has already run; it needs nothing from any of these.
set -uo pipefail
WIN="${CARDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="$HOME/cardio/venv/bin/python"
LOG="$WIN/logs/finish_cognitive.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the length-adjusted scoring run and any fetch to finish"
while pgrep -f "103_run_scdrs" > /dev/null || pgrep -f "85_fetch" > /dev/null; do
  sleep 30
done
say "machine is free"

need_gb=3
free_gb=$(df -BG --output=avail "/mnt/c" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$free_gb" ] && [ "$free_gb" -lt "$need_gb" ]; then
  say "ABORT: only ${free_gb} GB free on C:, need ${need_gb}. Try: wsl --shutdown"
  exit 1
fi
say "disk check: ${free_gb} GB free"

say "-- 171 glutamatergic programme"
"$PY" "$WIN/scripts/171_glutamatergic.py" 2>&1 | tee -a "$LOG"

# The cognitive traits need gene-level scores before they can be scored per cell.
# MAGMA discovers new sumstats on its own and resumes per chromosome.
if [ -f "$WIN/data/gwas_gsmap/Intelligence.sumstats.gz" ] \
   || [ -f "$WIN/data/gwas_gsmap/ReactionTime.sumstats.gz" ]; then
  say "-- MAGMA for the cognitive traits"
  TRAITS=Intelligence,ReactionTime bash "$WIN/scripts/101_magma_gene_scores.sh" 2>&1 \
    | tee -a "$LOG"
  say "-- restoring the MHC-excluded control (101b drops it on rebuild)"
  "$PY" "$WIN/scripts/112_add_nomhc_control.py" 2>&1 | tee -a "$LOG"
  say "-- scoring the cognitive traits on the axis subset"
  "$PY" "$WIN/scripts/103_run_scdrs.py" \
    --h5 data/singlecell/axis_subset.h5ad \
    --cov data/scdrs/covariates_axis.tsv \
    --out results/scdrs_axis 2>&1 | tee -a "$LOG"
else
  say "!! neither cognitive trait was fetched — skipping MAGMA and scoring"
fi

say "-- 170 genetic correlation, now including the cognitive traits"
"$PY" "$WIN/scripts/170_ldsc_rg.py" 2>&1 | tee -a "$LOG"

say "-- 181 bidirectional MR"
"$PY" "$WIN/scripts/181_mr_bidirectional.py" 2>&1 | tee -a "$LOG"

say "-- 153 stratified interaction and scorecard, with everything present"
"$PY" "$WIN/scripts/153_axis_interaction.py" 2>&1 | tee -a "$LOG"

say "======== cognitive arm complete"
