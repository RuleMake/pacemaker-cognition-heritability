#!/usr/bin/env bash
# Resume the CELLxGENE single-cell download, and report what annotation survived.
#
# The first attempt used `curl -sL --retry 3 -o`, which cannot recover from a STALLED
# connection: curl only counts an error, and a socket that stops delivering bytes
# without closing is not an error. It sat at 1.41 GB for forty minutes.
#
# Fixes here:
#   -C -                     resume from whatever is already on disk
#   --speed-limit/-time      treat <50 KB/s for 30 s as a failure, so retry can fire
#   --retry 20 --retry-delay turn a stall into a retry loop instead of a hang
#
# The point of the download is one question: did CELLxGENE's standardised 12-type
# ontology keep any trace of Kanemaru's 75 author cell states? Without a pacemaker
# label the file is useless for scDRS, so that check runs first and the script says
# so plainly either way.

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/singlecell
OUT=data/singlecell/heart_global.h5ad
URL="https://datasets.cellxgene.cziscience.com/1e1e07c3-bfcb-4a0d-91de-c2614a891409.h5ad"

have=$( [ -f "$OUT" ] && stat -c %s "$OUT" || echo 0 )
echo "start $(date '+%F %T')  — resuming from $((have / 1024 / 1024)) MB"

curl -L -C - --retry 20 --retry-delay 5 --retry-all-errors \
     --speed-limit 50000 --speed-time 30 \
     --progress-bar -o "$OUT" "$URL"
rc=$?
echo "curl exit $rc, size now $(stat -c %s "$OUT" | awk '{printf "%.2f GB", $1/1073741824}')"
[ $rc -eq 0 ] || { echo "download incomplete — rerun this script to resume"; exit $rc; }

python - "$OUT" <<'PY'
import re
import sys

import anndata as ad

a = ad.read_h5ad(sys.argv[1], backed="r")
print(f"\nshape: {a.n_obs:,} cells x {a.n_vars:,} genes\n")
print("obs columns:")
for c in a.obs.columns:
    print(f"  {c:<38} n_unique={a.obs[c].nunique(dropna=True)}")

PAT = r"pacemaker|SAN_P|sinoatrial|sino-atrial|nodal|node|bundle|Purkinje|AVN|CCS|conduct"
found = False
for c in a.obs.columns:
    vals = {str(x) for x in a.obs[c].dropna().unique()[:400]}
    hits = sorted(v for v in vals if re.search(PAT, v, re.I))
    if not hits:
        continue
    found = True
    print(f"\n*** '{c}' carries conduction-system labels:")
    s = a.obs[c].astype(str)
    for v in hits:
        print(f"      {v:<44} n={(s == v).sum():,}")

print("\n" + "=" * 70)
if found:
    print("VERDICT: a conduction-system label survived — scDRS is possible.")
else:
    print("VERDICT: no pacemaker/conduction label anywhere in obs.")
    print("CELLxGENE's standardised ontology erased the author cell states, so this")
    print("file cannot test the pacemaker hypothesis. scDRS would need the author")
    print("originals from the Heart Cell Atlas rather than the CELLxGENE mirror.")
PY
echo "finish $(date '+%F %T')"
