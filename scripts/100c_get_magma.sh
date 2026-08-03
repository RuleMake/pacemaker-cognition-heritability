#!/usr/bin/env bash
# Fetch the MAGMA binary, trying every published location.
#
# The first attempt failed silently, which could be a dead URL or a network problem
# from inside WSL. This distinguishes the two: reachability is tested first, then each
# candidate URL is tried with errors shown rather than swallowed.
set -uo pipefail
TOOLS="$HOME/cardio/tools"
mkdir -p "$TOOLS"
cd "$TOOLS"

echo "=== reachability"
for H in ctg.cncr.nl vu.nl zenodo.org github.com; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "https://$H/" || echo "fail")
  echo "  $H -> $code"
done

echo
echo "=== download attempts"
URLS=(
  "https://ctg.cncr.nl/software/MAGMA/prog/magma_v1.10_static.zip"
  "https://ctg.cncr.nl/software/MAGMA/prog/magma_v1.10.zip"
  "https://ctg.cncr.nl/software/MAGMA/prog/archive/magma_v1.10_static.zip"
  "https://vu.nl/en/download/magma_v1.10_static.zip"
)
for U in "${URLS[@]}"; do
  echo "-- $U"
  if curl -fL --max-time 180 --retry 2 -o magma.zip "$U" 2>&1 | tail -2; then
    if [ -s magma.zip ] && file magma.zip | grep -qi zip; then
      echo "   got $(du -h magma.zip | cut -f1)"
      break
    fi
  fi
  rm -f magma.zip
done

if [ -s magma.zip ]; then
  unzip -o magma.zip
  chmod +x magma 2>/dev/null || true
  ./magma --version 2>&1 | head -2
else
  echo
  echo "MAGMA unavailable. 101_make_gene_scores.py will compute the gene statistics"
  echo "itself using the 1000G genotypes already on disk — see that script for the"
  echo "model and its equivalence to MAGMA's snp-wise mean."
fi
