#!/usr/bin/env bash
# Finish the run: score the late-added control, then produce every final table.
#
# Two traits were added after the orchestrator had already loaded its gene-set file:
# rheumatoid arthritis without the MHC (the deciding negative control, rebuilt because
# a third of the original set's weight sat in one locus). scDRS caches per trait, so
# re-running the scorer computes only what is missing and reuses the rest — about
# twenty minutes rather than five hours.
#
# Then every report and audit is regenerated against the complete set of traits, so
# the final tables are internally consistent rather than assembled from runs made at
# different times with different trait lists.
set -uo pipefail
WIN="${CARDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="$HOME/cardio/venv/bin/python"
LOG="$WIN/logs/finish.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the main scDRS run to finish"
while pgrep -f "110_run_all.sh" > /dev/null || pgrep -f "103_run_scdrs" > /dev/null; do
  sleep 30
done
say "main run finished"

# The previous attempt lost the deciding control trait to a bare
# `OSError: [Errno 5] Input/output error` when C: hit zero bytes mid-write — twenty
# minutes of scoring discarded, and a disk-full failure that reads like a code bug.
# Check before starting rather than discovering it at the write.
need_gb=2
free_gb=$(df -BG --output=avail "/mnt/c" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$free_gb" ] && [ "$free_gb" -lt "$need_gb" ]; then
  say "ABORT: only ${free_gb} GB free on C:, need at least ${need_gb}."
  say "  Reclaim with: wsl --shutdown   (releases the VHD; safe once jobs are done)"
  say "  or scripts/121_reclaim_disk.py --delete"
  exit 1
fi
say "disk check: ${free_gb} GB free on C:"

say "-- scoring any trait not yet cached (expect RheumatoidArthritis_noMHC)"
"$PY" "$WIN/scripts/103_run_scdrs.py" 2>&1 | tee -a "$LOG"

for s in 104_scdrs_report.py 107_audit_scdrs.py 108_assay_confound.py \
         109_audit_robustness.py 111_audit_geneset_spread.py; do
  say "-- $s"
  "$PY" "$WIN/scripts/$s" 2>&1 | tee -a "$LOG"
done

say "======== all analyses complete"
