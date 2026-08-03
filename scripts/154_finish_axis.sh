#!/usr/bin/env bash
# Finish the whole-axis run without supervision.
#
# The scorer currently running loaded its gene-set file at startup, so it is working
# through the fourteen traits that existed then. The three ventricular traits — QRS
# duration, QT interval, bundle branch block — plus the restored MHC-excluded control
# were written to traits.gs afterwards and are invisible to it.
#
# scDRS caches per trait, so re-running the scorer once it exits computes only what is
# missing and reuses the rest. Then the analysis that actually tests the hypothesis
# runs: the stratified decomposition, and the interaction test scored against the
# predictions registered before any of this was computed.
#
# The node subset is topped up last. It has no Purkinje cells, so it cannot answer the
# ventricular question — but running the same three traits there shows where they land
# when the correct cell is absent from the panel, which is worth knowing and is exactly
# the situation every earlier trait was scored in.
set -uo pipefail
WIN="${CARDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="$HOME/cardio/venv/bin/python"
LOG="$WIN/logs/finish_axis.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the axis scoring run to finish"
while pgrep -f "103_run_scdrs" > /dev/null; do sleep 30; done
say "it has exited"

# C: hit zero bytes once already and cost twenty minutes of scoring to a bare
# OSError. Check before starting rather than discovering it at the write.
need_gb=3
free_gb=$(df -BG --output=avail "/mnt/c" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$free_gb" ] && [ "$free_gb" -lt "$need_gb" ]; then
  say "ABORT: only ${free_gb} GB free on C:, need ${need_gb}."
  say "  Reclaim with: wsl --shutdown   (releases the VHD; safe once jobs are done)"
  exit 1
fi
say "disk check: ${free_gb} GB free on C:"

say "-- scoring the traits added after the run started (axis subset)"
"$PY" "$WIN/scripts/103_run_scdrs.py" \
  --h5 data/singlecell/axis_subset.h5ad \
  --cov data/scdrs/covariates_axis.tsv \
  --out results/scdrs_axis 2>&1 | tee -a "$LOG"

say "-- stratified decomposition (donor / region / assay)"
"$PY" "$WIN/scripts/140_donor_decomposition.py" \
  --h5 data/singlecell/axis_subset.h5ad \
  --scdrs results/scdrs_axis --tag _axis 2>&1 | tee -a "$LOG"

say "-- interaction test and pre-registered scorecard"
"$PY" "$WIN/scripts/153_axis_interaction.py" 2>&1 | tee -a "$LOG"

say "-- topping up the node subset with the same new traits"
"$PY" "$WIN/scripts/103_run_scdrs.py" 2>&1 | tee -a "$LOG"

for s in 104_scdrs_report.py 113_peek_disease.py; do
  say "-- $s"
  "$PY" "$WIN/scripts/$s" 2>&1 | tee -a "$LOG"
done

say "======== axis run complete"
