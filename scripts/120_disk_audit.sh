#!/usr/bin/env bash
# What is using disk, and what could be reclaimed without losing anything needed.
#
# C: is down to 6 GB free and the pipeline still has to convert a 447 MB file and
# score four more traits. Rather than guessing, this measures each category and
# separates what is still needed from what is regenerable.
set -uo pipefail
W="$HOME/cardio"

echo "=== gsMap working directories"
printf "  %-34s %s\n" "original run (work)" "$(du -sh "$W/work" 2>/dev/null | cut -f1)"
printf "  %-34s %s\n" "depth-matched (work_dm)" "$(du -sh "$W/work_dm" 2>/dev/null | cut -f1)"
printf "  %-34s %s\n" "magma" "$(du -sh "$W/magma" 2>/dev/null | cut -f1)"

echo
echo "=== inside the original run — regenerable vs needed"
for sub in report generate_ldscore find_latent_representations latent_to_gene \
           spatial_ldsc cauchy_combination; do
  t=$(du -cs "$W"/work/*/*/"$sub" 2>/dev/null | tail -1 | cut -f1)
  printf "  %-34s %8s MB\n" "$sub" "$((${t:-0} / 1024))"
done

echo
echo "  report/ holds the PNG galleries gsMap renders per trait per section."
echo "  The only thing needed from it — Gene_Diagnostic_Info.csv — was already"
echo "  copied to results/genediag (96 files). The plots are pure output."
echo
echo "  cauchy_combination and spatial_ldsc hold the actual results and were"
echo "  already collected into results/. Keeping them costs little and makes any"
echo "  re-analysis possible without recomputing."

echo
echo "=== raw GWAS downloads still on disk (Windows side)"
ls -la "$CARDIO_ROOT/data/gwas/" 2>/dev/null | tail -n +2 | \
  awk '{printf "  %-32s %6.0f MB\n", $9, $5/1048576}'
echo "  These are pre-conversion originals. Every one has a converted counterpart"
echo "  in data/gwas_gsmap, and the fetch scripts re-download on demand."

echo
echo "=== free space"
df -h / | tail -1 | awk '{print "  WSL volume: "$4" free of "$2}'
