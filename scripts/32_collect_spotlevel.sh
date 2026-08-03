#!/usr/bin/env bash
# Pull the spot-level gsMap tables out of the WSL filesystem into a flat directory
# on the Windows side, so the contrast analysis can read them.
#
#   bash 32_collect_spotlevel.sh
#
# The batch script only aggregates the Cauchy summaries; the per-spot tables stay
# in ~/cardio/work. Those are what the trait-contrast analysis needs, and they are
# small (~200 KB each), so copying them out costs nothing. Safe to run repeatedly
# while the batch is still going — it simply picks up whatever has finished.

set -uo pipefail

LIN="$HOME/cardio"
DEST="$CARDIO_ROOT/results/spotlevel"
mkdir -p "$DEST"

n=0
for p in "$LIN"/work/*/*/report/*/gsMap_plot/*_gsMap_plot.csv; do
  [ -e "$p" ] || continue
  trait=$(basename "$(dirname "$(dirname "$p")")")
  # .../work/<sample>/<sample>/report/<trait>/gsMap_plot/<file>
  sample=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$p")")")")")
  cp -f "$p" "$DEST/${sample}__${trait}.csv"
  n=$((n+1))
done

echo "collected $n spot-level tables -> $DEST"
ls "$DEST" | sed 's/^/  /' | head -30
echo
echo "sections: $(ls "$DEST" | sed 's/__[A-Za-z]*\.csv$//' | sort -u | wc -l)"
echo "traits  : $(ls "$DEST" | sed 's/^.*__//; s/\.csv$//' | sort -u | tr '\n' ' ')"
