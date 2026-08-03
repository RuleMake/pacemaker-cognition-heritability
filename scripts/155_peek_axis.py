"""
Look at whatever the axis run has finished so far, without waiting for all of it.
================================================================================

The positive control gates everything: if atrial fibrillation does not concentrate in
atrial cardiomyocytes on this subset, nothing downstream means anything, and it is
better to find that out at trait two than at trait seventeen. The four conduction cell
types are printed with GWAS power alongside, because bundle branch block has 219
genome-wide significant SNPs and a null from it carries no information at all.

Usage:  python scripts/155_peek_axis.py [--dir results/scdrs_axis]
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.as_posix()
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/scdrs_axis")
a = ap.parse_args()
SC = a.dir if os.path.isabs(a.dir) else f"{ROOT}/{a.dir}"
GWAS = f"{ROOT}/data/gwas_gsmap"

FOCUS = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
FLOOR = 920
# a mixture class with 48% more detected genes than the rest — see 155c
EXCLUDE = {"unclassified"}

files = sorted(glob.glob(f"{SC}/*.group.tsv"))
if not files:
    raise SystemExit(f"nothing scored yet in {SC}")
print(f"{len(files)} traits scored so far\n")

power = {}
for p in sorted(glob.glob(f"{GWAS}/*.sumstats.gz")):
    t = os.path.basename(p).replace(".sumstats.gz", "")
    try:
        z = pd.read_csv(p, sep="\t", usecols=["Z"]).Z.to_numpy()
    except Exception:
        continue
    z = z[np.isfinite(z)]
    power[t] = int((np.abs(z) > 5.45).sum())

print("=" * 104)
print(f"{'trait':<24}{'gw-sig':>8}  top 5 cell states")
print("=" * 104)
rows = {}
for p in files:
    t = os.path.basename(p).replace(".group.tsv", "")
    g = pd.read_csv(p, sep="\t", index_col=0).sort_values("assoc_mcz", ascending=False)
    g = g[~g.index.isin(EXCLUDE)]
    rows[t] = g
    top = ", ".join(f"{i}({g.loc[i, 'assoc_mcz']:+.1f})" for i in g.index[:5])
    n = power.get(t, 0)
    flag = "" if n >= FLOOR else " !"
    print(f"{t:<24}{n:>8,}{flag} {top}")

print("\n" + "=" * 104)
print("CONDUCTION CELL RANKS")
print("=" * 104)
print(f"{'trait':<24}" + "".join(f"{c[:16]:>19}" for c in FOCUS))
for t, g in rows.items():
    lst = list(g.index)
    line = f"{t:<24}"
    for c in FOCUS:
        if c not in lst:
            line += f"{'-':>19}"
            continue
        line += f"{f'#{lst.index(c) + 1}/{len(lst)} z{g.loc[c, chr(122)] if False else g.loc[c, 'assoc_mcz']:+.1f}':>19}"
    print(line)
print("\n! = below the 920 gw-sig power floor; a null from that trait means nothing")
if "axis" in SC:
    # true only of the whole-atlas subset: the node subset is single-assay throughout,
    # so 100% multiome is its background rather than a property of any cell type
    print("NOTE: these global ranks are assay-confounded on this subset (the nodal")
    print("states are 100% multiome against a 34% multiome background). The stratified")
    print("analysis in 153 is the one that decides anything.")
