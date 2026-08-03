#!/usr/bin/env bash
# Run everything still outstanding, unattended, in dependency order.
#
# Four pieces are in flight or queued and they have real dependencies between them.
# This drives them to completion so the whole thing can be left alone:
#
#   1. depth-matched gsMap re-run  -> its verdict script (causal test of the depth
#                                     confound; the last thing owed on the spatial line)
#   2. MAGMA gene scores           -> scDRS gene sets
#   3. scDRS on 118,172 node cells -> the decisive test of the pacemaker hypothesis
#   4. disease-trait GWAS          -> MAGMA -> scDRS again, now including AV block,
#                                     atrial flutter, Brugada, heart failure, DCM
#
# Ordering rule that matters: scDRS runs on the physiological traits FIRST, because
# its atrial-fibrillation positive control decides whether anything downstream is
# worth computing. If atrial fibrillation does not land on atrial cardiomyocytes, the
# disease traits are not run — the pipeline would be broken and more traits would just
# produce more uninterpretable numbers. That decision is made here, automatically,
# and recorded in the log either way.
#
#   wsl -d Ubuntu -e bash .../110_run_all.sh

set -uo pipefail
LIN="$HOME/cardio"
WIN="${CARDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="$LIN/venv/bin/python"
LOG="$WIN/logs/run_all.log"
mkdir -p "$(dirname "$LOG")"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_for() {  # file description max_minutes
  local f="$1" d="$2" max="${3:-240}" waited=0
  while [ ! -e "$f" ]; do
    if [ "$waited" -ge $((max * 6)) ]; then
      say "TIMEOUT after ${max} min waiting for $d"
      return 1
    fi
    sleep 10; waited=$((waited + 1))
  done
  say "$d ready"
}

say "======== run_all started"

# ---------------------------------------------------------------- 1. spatial verdict
say "-- step 1: depth-matched gsMap verdict"
if wait_for "$WIN/results/spotlevel_depthmatched" "depth-matched spot tables" 180; then
  # the collection step writes the directory before the last section lands; give the
  # batch a chance to finish all eight before reading
  while pgrep -f "63_run_depthmatched_fast" > /dev/null; do sleep 20; done
  "$PY" "$WIN/scripts/64_depthmatched_verdict.py" 2>&1 | tee -a "$LOG"
else
  say "SKIPPED — depth-matched run never produced output"
fi

# ---------------------------------------------------------------- 2. wait for MAGMA
say "-- step 2: waiting for MAGMA gene scores"
while pgrep -f "101_magma_gene_scores" > /dev/null; do sleep 30; done
"$PY" "$WIN/scripts/101b_build_gs.py" 2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------- 3. scDRS
say "-- step 3: scDRS on the physiological traits"
if [ ! -f "$WIN/data/scdrs/traits.gs" ]; then
  say "ABORT — no gene sets were produced; MAGMA must have failed"
  exit 1
fi
"$PY" "$WIN/scripts/102c_add_neighbors.py" 2>&1 | tee -a "$LOG"
"$PY" "$WIN/scripts/103_run_scdrs.py" 2>&1 | tee -a "$LOG"
RC=$?

# A crash and a genuinely negative positive control look identical from outside unless
# the difference is made explicit — and they call for opposite responses. A crash means
# fix the code and rerun; a negative control means the biology cannot be read. The
# first attempt died on a missing kNN graph and was logged as "positive control FAIL",
# which would have buried a trivial bug under a scientific-sounding conclusion.
PC=$("$PY" - <<'PY'
import json
import os

p = "$CARDIO_ROOT/results/scdrs/scdrs_summary.json"
if not os.path.exists(p):
    print("CRASHED")
else:
    try:
        d = json.load(open(p))
        pc = d.get("positive_control")
        print("PASS" if pc and pc.get("passes") else
              "FAIL" if pc else "CRASHED")
    except Exception:
        print("CRASHED")
PY
)
say "scDRS exit $RC, positive control: $PC"

# ---------------------------------------------------------------- 4. disease traits
if [ "$PC" = "CRASHED" ]; then
  say "-- step 4 SKIPPED: scDRS did not finish. This is a CODE failure, not a"
  say "   scientific result — check the traceback above, fix, and rerun step 3."
elif [ "$PC" != "PASS" ]; then
  say "-- step 4 SKIPPED: the atrial-fibrillation control did not land on atrial"
  say "   cardiomyocytes, so the pipeline cannot be trusted and adding disease"
  say "   traits would only produce more uninterpretable numbers."
else
  say "-- step 4: disease traits"
  # The fetcher runs as a WINDOWS python process, which pgrep inside WSL cannot see —
  # waiting on the process would return instantly and read a half-written directory.
  # Wait on the artefact it writes when it is genuinely done.
  wait_for "$WIN/results/disease_traits.json" "disease-trait vetting" 240 || \
    say "   proceeding with whatever converted files exist"
  NEW=$(ls "$WIN"/data/gwas_gsmap/*.sumstats.gz 2>/dev/null \
        | xargs -n1 basename 2>/dev/null | sed 's/\.sumstats\.gz//' \
        | grep -E 'AVblock|AtrialFlutter|Brugada|HeartFailure|DCM' | paste -sd, -)
  if [ -z "$NEW" ]; then
    say "   no disease trait passed vetting — nothing to add"
  else
    say "   scoring: $NEW"
    TRAITS="$NEW" bash "$WIN/scripts/101_magma_gene_scores.sh" 2>&1 | tee -a "$LOG"
    "$PY" "$WIN/scripts/101b_build_gs.py" 2>&1 | tee -a "$LOG"
    say "-- step 5: scDRS re-run including the disease traits"
    "$PY" "$WIN/scripts/103_run_scdrs.py" 2>&1 | tee -a "$LOG"
  fi
fi

say "======== run_all finished"
