#!/usr/bin/env bash
# Fetch MAGMA from the SURFsara share links the CNCR page actually points at.
#
# ctg.cncr.nl/software/MAGMA/prog/... is stale — it returns a 132 KB HTML page rather
# than an archive. The live links are ownCloud shares, which serve the file only when
# /download is appended.
#
# Build 37 throughout: the 1000G reference shipped with gsMap is Phase 3 (b37) and the
# GTF is a lift37, so the b37 gene location file is the matching one.
set -uo pipefail
TOOLS="$HOME/cardio/tools"
mkdir -p "$TOOLS"
cd "$TOOLS"

fetch() {  # url outfile description
  echo "-- $3"
  curl -fL --max-time 600 --retry 3 --retry-delay 3 -s -o "$2" "$1/download" || {
    echo "   FAILED"; return 1; }
  echo "   $(du -h "$2" | cut -f1)  $(file -b "$2" | cut -c1-60)"
}

fetch "https://vu.data.surfsara.nl/index.php/s/lxDgt2dNdNr6DYt" magma.zip \
      "MAGMA v1.10 linux static"
fetch "https://vu.data.surfsara.nl/index.php/s/Pj2orwuF2JYyKxq" gene_loc_b37.zip \
      "gene locations, NCBI build 37"

if file magma.zip 2>/dev/null | grep -qi zip; then
  unzip -o magma.zip >/dev/null
  chmod +x magma
  echo
  ./magma --version 2>&1 | head -2
else
  echo "magma archive still not usable"
fi

if file gene_loc_b37.zip 2>/dev/null | grep -qi zip; then
  unzip -o gene_loc_b37.zip >/dev/null
  ls -la *.gene.loc 2>/dev/null | head
  head -3 *.gene.loc 2>/dev/null
fi

echo
echo "contents of $TOOLS:"
ls -la "$TOOLS"
