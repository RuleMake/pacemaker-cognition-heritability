#!/usr/bin/env bash
# Depth-matched gsMap re-run, stopping each section once its results exist.
#
# Why not just use 21_run_gsmap_batch.sh
# --------------------------------------
# quick_mode always ends with a `report` stage that renders a few hundred PNGs per
# trait. Section 1 reached its Cauchy results in about 7 minutes and then spent 45
# more inside report — 8 sections would have taken ~7 hours to produce numbers that
# were ready after 1. There is no flag to skip it, and the step-by-step subcommands
# are not an alternative because run_generate_ldscore recomputes LD scores from
# scratch instead of using quick_mode's precomputed weights.
#
# So: launch quick_mode, watch the log, and stop the section the moment it moves on
# to report. That transition is unambiguous — gsMap logs `gsMap.diagnosis` as the
# first thing report does, and by then spatial_ldsc and cauchy_combination have both
# finished writing. The Cauchy files are verified present before anything is killed,
# so a section is never recorded as done on the strength of a log line alone.
#
#   wsl -d Ubuntu -e bash /mnt/c/.../scripts/63_run_depthmatched_fast.sh

set -uo pipefail

LIN="$HOME/cardio"
VENV="$LIN/venv"
RES="$LIN/resource/gsMap_resource"
WIN="${CARDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GWASDIR="$WIN/data/gwas_gsmap"
H5DIR="$WIN/data/spatial_depthmatched"
WORKROOT="$LIN/work_dm"
LOGDIR="$WIN/logs/gsmap_depthmatched_fast"
TIMEOUT=3600          # per-section ceiling, so one bad section cannot stall the batch

TRAITS=(RestingHeartRate AtrialFibrillation PRinterval EducationalAttainment)
mkdir -p "$LOGDIR" "$WORKROOT"
source "$VENV/bin/activate"

mapfile -t S < <(awk -F'\t' 'NR>1{print $3}' "$H5DIR/manifest.tsv")
echo "=============================================================="
echo " sections : ${#S[@]}    traits: ${TRAITS[*]}"
echo " started  : $(date '+%F %T')"
echo "=============================================================="

cauchy_of() { echo "$WORKROOT/$1/$1/cauchy_combination/$1_$2.Cauchy.csv.gz"; }

ok=0; skip=0; fail=0; i=0
for SAMPLE in "${S[@]}"; do
  i=$((i+1))

  done_all=1
  for t in "${TRAITS[@]}"; do
    [ -f "$(cauchy_of "$SAMPLE" "$t")" ] || done_all=0
  done
  if [ "$done_all" -eq 1 ]; then
    echo "[$i/${#S[@]}] $SAMPLE: already complete — skipped"
    skip=$((skip+1)); continue
  fi

  CFG="$WORKROOT/${SAMPLE}_gwas.yaml"
  : > "$CFG"
  for t in "${TRAITS[@]}"; do
    echo "${t}: ${GWASDIR}/${t}.sumstats.gz" >> "$CFG"
  done

  LOG="$LOGDIR/${SAMPLE}.log"
  START=$(date +%s)
  echo "[$i/${#S[@]}] $SAMPLE: starting $(date '+%T')"

  gsmap quick_mode \
      --workdir "$WORKROOT/$SAMPLE" \
      --sample_name "$SAMPLE" \
      --gsMap_resource_dir "$RES" \
      --hdf5_path "$H5DIR/${SAMPLE}.h5ad" \
      --annotation 'annotation_final' \
      --data_layer 'count' \
      --sumstats_config_file "$CFG" \
      > "$LOG" 2>&1 &
  PID=$!

  # wait for either: report started (results are in), the process ended, or timeout
  while kill -0 "$PID" 2>/dev/null; do
    if grep -qa 'gsMap.diagnosis' "$LOG" 2>/dev/null; then
      have=1
      for t in "${TRAITS[@]}"; do
        [ -f "$(cauchy_of "$SAMPLE" "$t")" ] || have=0
      done
      if [ "$have" -eq 1 ]; then
        echo "      results complete — stopping before the report stage"
        kill "$PID" 2>/dev/null
        sleep 3
        kill -9 "$PID" 2>/dev/null
        break
      fi
    fi
    if [ $(( $(date +%s) - START )) -gt "$TIMEOUT" ]; then
      echo "      TIMEOUT after ${TIMEOUT}s — killing"
      kill -9 "$PID" 2>/dev/null
      break
    fi
    sleep 10
  done
  wait "$PID" 2>/dev/null

  MIN=$(( ($(date +%s) - START) / 60 ))
  have=1
  for t in "${TRAITS[@]}"; do
    [ -f "$(cauchy_of "$SAMPLE" "$t")" ] || have=0
  done
  if [ "$have" -eq 1 ]; then
    echo "[$i/${#S[@]}] $SAMPLE: OK in ${MIN} min"
    ok=$((ok+1))
  else
    echo "[$i/${#S[@]}] $SAMPLE: FAILED after ${MIN} min"
    tail -3 "$LOG" | sed 's/^/      /'
    fail=$((fail+1))
  fi
  df -h / | tail -1 | awk '{print "      disk: "$4" free"}'
done

echo
echo "=============================================================="
echo " done $(date '+%F %T'):  ok=$ok  failed=$fail  skipped=$skip"
echo "=============================================================="

python - "$WORKROOT" "$WIN/results/gsmap_cauchy_all_depthmatched.tsv" <<'PY'
import csv
import glob
import gzip
import os
import sys

work, out = sys.argv[1], sys.argv[2]
rows, header = [], None
for p in glob.glob(os.path.join(work, "*", "*", "cauchy_combination", "*.Cauchy.csv.gz")):
    base = os.path.basename(p).replace(".Cauchy.csv.gz", "")
    sample = os.path.basename(os.path.dirname(os.path.dirname(p)))
    trait = base[len(sample) + 1:] if base.startswith(sample + "_") else base
    with gzip.open(p, "rt") as f:
        r = csv.reader(f)
        h = next(r)
        header = header or (["sample", "trait"] + h)
        rows.extend([sample, trait] + line for line in r)
with open(out, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    if header:
        w.writerow(header)
        w.writerows(rows)
print(f"collected {len(rows)} rows from {len({r[0] for r in rows})} sections "
      f"x {len({r[1] for r in rows})} traits -> {out}")
PY

# spot-level tables, needed for the depth-correlation comparison
python - "$WORKROOT" "$WIN/results/spotlevel_depthmatched" <<'PY'
import glob
import os
import sys

import pandas as pd

work, dest = sys.argv[1], sys.argv[2]
os.makedirs(dest, exist_ok=True)
n = 0
for p in glob.glob(os.path.join(work, "*", "*", "spatial_ldsc", "*.csv.gz")):
    base = os.path.basename(p).replace(".csv.gz", "")
    sample = os.path.basename(os.path.dirname(os.path.dirname(p)))
    trait = base[len(sample) + 1:] if base.startswith(sample + "_") else base
    df = pd.read_csv(p)
    df.to_csv(os.path.join(dest, f"{sample}__{trait}.csv"), index=False)
    n += 1
print(f"copied {n} spot-level tables -> {dest}")
PY

echo "batch finished $(date '+%F %T')"
