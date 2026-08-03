#!/usr/bin/env bash
# Run gsMap across every prepared section, then collect the results.
#
#   bash 21_run_gsmap_batch.sh                 # all 16 sections, all traits
#   SLIDES="SAN__A SAN__B" bash 21_run_gsmap_batch.sh
#   TRAITS=RestingHeartRate bash 21_run_gsmap_batch.sh
#
# Notes
# -----
# * One section per gsMap invocation (memory), but ALL traits inside that one
#   invocation: steps 1-3 (latent representation, gene specificity, LD scores) are
#   per-section and trait-independent, so passing traits together computes them once.
#   Measured: a 2,852-spot section with 3 traits takes ~12 min end to end.
# * A section that fails does not stop the batch — it is recorded and skipped, so an
#   overnight run cannot be wiped out by one bad input.
# * Sections already carrying a Cauchy result for every requested trait are skipped,
#   which makes the script resumable after an interrupt.

set -uo pipefail

LIN="$HOME/cardio"
VENV="$LIN/venv"
RES="$LIN/resource/gsMap_resource"
WIN="${CARDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GWASDIR="$WIN/data/gwas_gsmap"
# H5DIR/WORKROOT/TAG let the same script run an alternative input set — used for the
# depth-matched re-run — without touching the original results.
H5DIR="${H5DIR:-$WIN/data/spatial_gsmap}"
WORKROOT="${WORKROOT:-$LIN/work}"
TAG="${TAG:-}"
MANIFEST="$H5DIR/manifest.tsv"
LOGDIR="$WIN/logs/gsmap_batch${TAG:+_$TAG}"
mkdir -p "$LOGDIR" "$LIN/results" "$WORKROOT"

# default traits = every sumstats file present
if [ -n "${TRAITS:-}" ]; then
  IFS=',' read -ra T <<< "$TRAITS"
else
  T=()
  for f in "$GWASDIR"/*.sumstats.gz; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); T+=("${b%.sumstats.gz}")
  done
fi

if [ -n "${SLIDES:-}" ]; then
  read -ra S <<< "$SLIDES"
else
  mapfile -t S < <(awk -F'\t' 'NR>1{print $3}' "$MANIFEST")
fi

# Fail fast on a missing sumstats file. Without this the loop happily writes a
# non-existent path into the config and every section dies in seconds — which is
# exactly what happened when one HRV download silently failed.
missing=0
for t in "${T[@]}"; do
  [ -f "$GWASDIR/${t}.sumstats.gz" ] || { echo "!! missing sumstats: ${t}.sumstats.gz"; missing=1; }
done
if [ "$missing" -eq 1 ]; then
  echo "   available:"
  ls "$GWASDIR"/*.sumstats.gz 2>/dev/null | sed 's#.*/#     #; s#\.sumstats\.gz##'
  echo "aborting before touching any section."
  exit 1
fi

echo "=============================================================="
echo " sections : ${#S[@]}"
echo " traits   : ${T[*]}"
echo " started  : $(date '+%F %T')"
echo "=============================================================="

source "$VENV/bin/activate"

SUMMARY="$LOGDIR/batch_status.tsv"
[ -f "$SUMMARY" ] || printf "section\tstatus\tminutes\tfinished\n" > "$SUMMARY"

ok=0; fail=0; skip=0; i=0
for SAMPLE in "${S[@]}"; do
  i=$((i+1))
  H5AD="$H5DIR/${SAMPLE}.h5ad"
  WORK="$WORKROOT/$SAMPLE"
  CFG="$WORKROOT/${SAMPLE}_gwas.yaml"

  if [ ! -f "$H5AD" ]; then
    echo "[$i/${#S[@]}] $SAMPLE: MISSING h5ad — skipped"
    printf "%s\tmissing_input\t-\t%s\n" "$SAMPLE" "$(date '+%F %T')" >> "$SUMMARY"
    fail=$((fail+1)); continue
  fi

  # resumable: skip when every requested trait already has a Cauchy result
  done_all=1
  for t in "${T[@]}"; do
    [ -f "$WORK/$SAMPLE/cauchy_combination/${SAMPLE}_${t}.Cauchy.csv.gz" ] || done_all=0
  done
  if [ "$done_all" -eq 1 ]; then
    echo "[$i/${#S[@]}] $SAMPLE: already complete — skipped"
    skip=$((skip+1)); continue
  fi

  : > "$CFG"
  for t in "${T[@]}"; do
    echo "${t}: ${GWASDIR}/${t}.sumstats.gz" >> "$CFG"
  done

  echo "[$i/${#S[@]}] $SAMPLE: starting $(date '+%T')"
  START=$(date +%s)
  if gsmap quick_mode \
        --workdir "$WORK" \
        --sample_name "$SAMPLE" \
        --gsMap_resource_dir "$RES" \
        --hdf5_path "$H5AD" \
        --annotation 'annotation_final' \
        --data_layer 'count' \
        --sumstats_config_file "$CFG" \
        > "$LOGDIR/${SAMPLE}.log" 2>&1; then
    MIN=$(( ($(date +%s) - START) / 60 ))
    echo "[$i/${#S[@]}] $SAMPLE: OK in ${MIN} min"
    printf "%s\tok\t%s\t%s\n" "$SAMPLE" "$MIN" "$(date '+%F %T')" >> "$SUMMARY"
    ok=$((ok+1))
  else
    MIN=$(( ($(date +%s) - START) / 60 ))
    echo "[$i/${#S[@]}] $SAMPLE: FAILED after ${MIN} min — see $LOGDIR/${SAMPLE}.log"
    tail -5 "$LOGDIR/${SAMPLE}.log" | sed 's/^/      /'
    printf "%s\tfailed\t%s\t%s\n" "$SAMPLE" "$MIN" "$(date '+%F %T')" >> "$SUMMARY"
    fail=$((fail+1))
  fi
  df -h / | tail -1 | awk '{print "      disk: "$4" free"}'
done

echo
echo "=============================================================="
echo " done $(date '+%F %T'):  ok=$ok  failed=$fail  skipped=$skip"
echo "=============================================================="

# ---- collect every Cauchy result into one table
python - "$WORKROOT" "$WIN/results/gsmap_cauchy_all${TAG:+_$TAG}.tsv" <<'PY'
import sys, gzip, csv, os, glob
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
        for line in r:
            rows.append([sample, trait] + line)
with open(out, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    if header:
        w.writerow(header)
        w.writerows(rows)
print(f"collected {len(rows)} rows from "
      f"{len({r[0] for r in rows})} sections x {len({r[1] for r in rows})} traits")
print(f"-> {out}")
PY

echo "batch finished $(date '+%F %T')"
