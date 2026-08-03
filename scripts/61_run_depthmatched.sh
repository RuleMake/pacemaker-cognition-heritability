#!/usr/bin/env bash
# Launch the depth-matched gsMap re-run.
#
# A thin wrapper purely so the /mnt/... paths never pass through Git Bash, which
# rewrites them into Windows paths and silently produces an empty run.
#
#   wsl -d Ubuntu -e bash /mnt/c/.../scripts/61_run_depthmatched.sh

set -euo pipefail
WIN="${CARDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export H5DIR="$WIN/data/spatial_depthmatched"
export WORKROOT="$HOME/cardio/work_dm"
export TAG="depthmatched"
export TRAITS="RestingHeartRate,AtrialFibrillation,PRinterval,EducationalAttainment"

exec bash "$WIN/scripts/21_run_gsmap_batch.sh"
