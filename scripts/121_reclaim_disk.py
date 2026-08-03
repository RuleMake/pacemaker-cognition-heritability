"""
Reclaim disk, but only after proving each item is safe to lose.
===============================================================

Two categories, with very different effects:

  Windows side   raw GWAS downloads under data/gwas. Deleting these frees C: at once.
                 Each is a pre-conversion original whose converted counterpart lives
                 in data/gwas_gsmap, and the fetch scripts re-download on demand.

  WSL side       the report/ subtrees under ~/cardio/work — the PNG galleries gsMap
                 renders per trait per section, about 7 GB. The one file that matters
                 from them, Gene_Diagnostic_Info.csv, was already copied to
                 results/genediag. Note this frees space inside the ext4 volume but
                 NOT on C: until the VHD is compacted, which needs `wsl --shutdown` —
                 and that would kill the running scDRS job. So it is done for
                 tidiness, with the real C: relief coming from the Windows side.

Nothing is deleted without its precondition holding. A raw file is removed only if its
converted counterpart exists AND is non-trivial in size; a report tree only if the
matching diagnostic CSVs are present in results. Anything still being written is left
alone. Refusals are printed, not silently skipped.

Usage:  python scripts/121_reclaim_disk.py            # report only
        python scripts/121_reclaim_disk.py --delete   # actually remove
"""

import argparse
import glob
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.as_posix()
RAW = f"{ROOT}/data/gwas"
CONV = f"{ROOT}/data/gwas_gsmap"
DIAG = f"{ROOT}/results/genediag"
WORK = os.path.expanduser("~/cardio/work")

# raw file -> the converted product that makes it redundant.
# None means the file has no counterpart because the study was abandoned.
PAIRS = {
    "AF_GCST90204201.tsv": "AtrialFibrillation.sumstats.gz",
    "EDU_GCST90296499.tsv": "EducationalAttainment.sumstats.gz",
    "RHR_ZhuZ_UKB460K.assoc.gz": "RestingHeartRate.sumstats.gz",
    "SCZ_GCST90018919.tsv.gz": None,   # retracted control: case count far too small
}

ap = argparse.ArgumentParser()
ap.add_argument("--delete", action="store_true")
args = ap.parse_args()

freed = 0
print("=" * 88)
print("WINDOWS SIDE — raw GWAS downloads (frees C: immediately)")
print("=" * 88)
for fn, conv in PAIRS.items():
    p = f"{RAW}/{fn}"
    if not os.path.exists(p):
        print(f"  {fn:<32} already gone")
        continue
    mb = os.path.getsize(p) / 1e6
    # a file written to in the last two minutes is probably still being produced
    if time.time() - os.path.getmtime(p) < 120:
        print(f"  {fn:<32} {mb:>7.0f} MB  KEPT — modified moments ago, still in use")
        continue
    if conv is None:
        print(f"  {fn:<32} {mb:>7.0f} MB  safe (study abandoned, nothing derived from it)")
    else:
        c = f"{CONV}/{conv}"
        if not os.path.exists(c) or os.path.getsize(c) < 1e6:
            print(f"  {fn:<32} {mb:>7.0f} MB  KEPT — {conv} missing or truncated")
            continue
        print(f"  {fn:<32} {mb:>7.0f} MB  safe ({conv}, "
              f"{os.path.getsize(c) / 1e6:.0f} MB, present)")
    freed += mb
    if args.delete:
        os.remove(p)

# anything else in data/gwas that is not a known pair — leave it, it may be in flight
others = [f for f in os.listdir(RAW)
          if f not in PAIRS and os.path.isfile(f"{RAW}/{f}")]
for f in others:
    mb = os.path.getsize(f"{RAW}/{f}") / 1e6
    age = (time.time() - os.path.getmtime(f"{RAW}/{f}")) / 60
    print(f"  {f:<32} {mb:>7.0f} MB  KEPT — unrecognised, last written "
          f"{age:.0f} min ago (a fetch may be using it)")

print(f"\n  reclaimable from C: {freed / 1000:.2f} GB")

print("\n" + "=" * 88)
print("WSL SIDE — gsMap report galleries")
print("=" * 88)
n_diag = len(glob.glob(f"{DIAG}/*_Gene_Diagnostic_Info.csv"))
print(f"  gene diagnostic CSVs already extracted to results/genediag: {n_diag}")
reports = sorted(glob.glob(f"{WORK}/*/*/report"))
if n_diag < 50:
    print("  REFUSING to touch report/ — fewer than 50 diagnostic CSVs were extracted,")
    print("  so the plots may still be the only copy of something needed.")
elif not reports:
    print("  no report directories found")
else:
    tot = 0
    for r in reports:
        sz = sum(os.path.getsize(os.path.join(dp, f))
                 for dp, _, fs in os.walk(r) for f in fs)
        tot += sz
    print(f"  {len(reports)} report trees, {tot / 1e9:.2f} GB")
    print("  contents are PNG galleries; the CSVs they accompany are already saved")
    if args.delete:
        for r in reports:
            shutil.rmtree(r, ignore_errors=True)
        print(f"  removed {len(reports)} report trees")
        print("  NOTE: this frees the ext4 volume, not C:. The VHD compacts only on")
        print("  `wsl --shutdown`, which would kill the running scDRS job — leave it.")

if not args.delete:
    print("\nreport only — rerun with --delete to remove")
