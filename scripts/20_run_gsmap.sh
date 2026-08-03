#!/usr/bin/env bash
# Run gsMap on one Visium section.
#
#   bash 20_run_gsmap.sh                       # default: smallest SAN slide, all 3 traits
#   bash 20_run_gsmap.sh SAN__HCAHeartST13233997
#   TRAITS=RestingHeartRate bash 20_run_gsmap.sh SAN__HCAHeartST13228105
#
# Design notes
# ------------
# * One section per run. gsMap's published memory figures are 11 GB at 2,902 spots
#   and 12 GB at 3,289; WSL2 here is capped at 12 GB, so the merged 27K-spot object
#   would not fit. Per-section is also the right unit — gsMap models one section's
#   spatial graph.
# * Outputs go to the Linux filesystem, not /mnt/c. gsMap writes many small files
#   and the 9p mount makes that painfully slow. Results are copied back at the end.
# * No --homolog_file: the data is already human.
# * All three traits run in one invocation via --sumstats_config_file, so the
#   expensive per-section steps (latent representation, gene specificity, LD scores)
#   are computed once and reused across traits.

set -euo pipefail

SAMPLE="${1:-SAN__HCAHeartST13228105}"
TRAITS="${TRAITS:-RestingHeartRate,AtrialFibrillation,EducationalAttainment}"

LIN="$HOME/cardio"
VENV="$LIN/venv"
RES="$LIN/resource/gsMap_resource"
WIN="${CARDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
H5AD="$WIN/data/spatial_gsmap/${SAMPLE}.h5ad"
GWASDIR="$WIN/data/gwas_gsmap"
WORK="$LIN/work/$SAMPLE"
CFG="$LIN/work/${SAMPLE}_gwas.yaml"

echo "=============================================================="
echo " sample : $SAMPLE"
echo " traits : $TRAITS"
echo " workdir: $WORK"
echo "=============================================================="

[ -f "$H5AD" ] || { echo "!! missing $H5AD"; exit 1; }
[ -d "$RES" ]  || { echo "!! missing resource dir $RES — run setup_wsl_gsmap.sh"; exit 1; }

# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$WORK" "$LIN/results"

# ---- GWAS config: trait_name -> sumstats path
: > "$CFG"
IFS=',' read -ra T <<< "$TRAITS"
for t in "${T[@]}"; do
  f="$GWASDIR/${t}.sumstats.gz"
  [ -f "$f" ] || { echo "!! missing sumstats $f"; exit 1; }
  echo "${t}: ${f}" >> "$CFG"
done
echo "--- GWAS config ---"; cat "$CFG"; echo

# ---- sanity-check the input the way gsMap will read it
python - "$H5AD" <<'PY'
import sys, anndata as ad, numpy as np
a = ad.read_h5ad(sys.argv[1], backed="r")
print(f"    spots x genes : {a.n_obs:,} x {a.n_vars:,}")
print(f"    layers        : {list(a.layers.keys())}")
print(f"    obsm          : {list(a.obsm.keys())}")
assert "count" in a.layers, "layers['count'] missing"
assert "spatial" in a.obsm, "obsm['spatial'] missing"
print(f"    annotation    : {a.obs['annotation_final'].nunique()} levels")
print(f"    first genes   : {list(a.var_names[:5])}")
PY

echo
echo "==> gsmap quick_mode  (expect ~11-12 GB peak; close other apps)"
START=$(date +%s)
gsmap quick_mode \
  --workdir "$WORK" \
  --sample_name "$SAMPLE" \
  --gsMap_resource_dir "$RES" \
  --hdf5_path "$H5AD" \
  --annotation 'annotation_final' \
  --data_layer 'count' \
  --sumstats_config_file "$CFG"
echo "==> finished in $(( ($(date +%s) - START) / 60 )) min"

# ---- copy the small result files back to the Windows side
DEST="$WIN/results/gsmap/$SAMPLE"
mkdir -p "$DEST"
find "$WORK" -type f \( -name "*.csv*" -o -name "*.html" -o -name "*.png" -o -name "*.pdf" \) \
     -exec cp --parents -t "$DEST" {} + 2>/dev/null || true
echo
echo "outputs under : $WORK"
echo "copied back to: $DEST"
find "$WORK" -name "*gsMap*" -maxdepth 3 | head -20
