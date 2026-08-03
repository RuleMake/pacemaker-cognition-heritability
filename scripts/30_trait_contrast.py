"""
Trait-vs-trait contrast: is there anatomy-specific signal beyond the tissue gradient?
====================================================================================

The problem
-----------
Running gsMap on a SAN section gives the SAME compartment ordering for every trait:

    haemorrhage < adipose_tissue < nerve < myocardium_atrial < node

including a non-cardiac control (educational attainment). That ordering tracks
tissue transcriptional activity — necrotic/haemorrhagic tissue lowest, fat next,
working myocardium highest — not trait-specific anatomy. Taken at face value the
per-trait p-values would "confirm" a sinoatrial hit for literally any trait.

The test
--------
If resting heart rate really is executed at the sinoatrial node, its signal there
must exceed what a non-cardiac trait shows at the SAME spots. So work with the
PAIRED, per-spot difference

    d(spot) = -log10 p_trait(spot) - (-log10 p_control(spot))

Because both traits are scored on the identical spots, tissue activity, spot RNA
content and cell density cancel out. What survives is trait-specific.

Then ask whether d is larger in the node than elsewhere, using a permutation test
that shuffles compartment labels across spots (10,000 draws), which preserves the
spatial distribution of d and only breaks its link to anatomy.

Reading the result
------------------
  node clearly highest for RestingHeartRate vs EducationalAttainment
        -> anatomy-specific signal exists; the project stands.
  node no different from myocardium_atrial once the control is subtracted
        -> Visium's 55 um resolution only separates cell types, not micro-anatomy;
           the topic needs reframing (see PROJECT-STATE.md acceptance criteria).

Usage:  python scripts/30_trait_contrast.py [<results_dir_of_one_section>]
"""

from pathlib import Path
import csv
import glob
import os
import sys

import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)
DEFAULT = (f"{ROOT}/results/gsmap/SAN__HCAHeartST13228105/home/<user>/cardio/"
           f"work/SAN__HCAHeartST13228105/SAN__HCAHeartST13228105/report")
BASE = sys.argv[1] if len(sys.argv) > 1 else DEFAULT

CONTROL = "EducationalAttainment"
N_PERM = 10000
RNG = np.random.default_rng(0)


def load(base):
    out = {}
    for p in sorted(glob.glob(os.path.join(base, "*", "gsMap_plot", "*_gsMap_plot.csv"))):
        trait = os.path.basename(os.path.dirname(os.path.dirname(p)))
        rows = list(csv.DictReader(open(p)))
        key = [r[""] for r in rows]
        logp = np.array([float(r["logp"]) for r in rows])
        ann = np.array([r["annotation"] for r in rows])
        out[trait] = dict(key=key, logp=logp, ann=ann)
    return out


d = load(BASE)
if CONTROL not in d:
    sys.exit(f"control trait {CONTROL} not found; have {list(d)}")
print(f"section: {os.path.basename(os.path.dirname(BASE))}")
print(f"traits : {list(d)}\n")

ctrl = d[CONTROL]
order = {k: i for i, k in enumerate(ctrl["key"])}

for trait, v in d.items():
    if trait == CONTROL:
        continue
    # align spot-wise; both files come from the same section so keys must match
    idx = np.array([order[k] for k in v["key"]])
    diff = v["logp"] - ctrl["logp"][idx]
    ann = v["ann"]

    print(f"═══ {trait}  minus  {CONTROL}   (n={len(diff):,} spots)")
    print(f"{'compartment':<24}{'n':>6}{'mean d':>10}{'median d':>11}"
          f"{'z_vs_perm':>11}{'emp_p':>9}")

    comps = [c for c in set(ann) if (ann == c).sum() >= 50]
    stats = []
    for c in sorted(comps):
        m = ann == c
        obs = diff[m].mean()
        null = np.empty(N_PERM)
        n_c = m.sum()
        for i in range(N_PERM):
            null[i] = diff[RNG.choice(len(diff), n_c, replace=False)].mean()
        z = (obs - null.mean()) / null.std()
        p = (1 + (null >= obs).sum()) / (1 + N_PERM)
        stats.append((c, int(n_c), obs, float(np.median(diff[m])), z, p))

    for c, n_c, obs, med, z, p in sorted(stats, key=lambda x: -x[4]):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{c:<24}{n_c:>6}{obs:>10.3f}{med:>11.3f}{z:>+11.2f}{p:>9.4f} {star}")

    top = max(stats, key=lambda x: x[4])
    print(f"\n  -> strongest trait-specific compartment: {top[0]} (z={top[4]:+.2f})")
    if top[0] == "node":
        print("     node wins after removing the tissue gradient — anatomy-specific.")
    else:
        print("     node does NOT win once the control is subtracted.")
    print()
