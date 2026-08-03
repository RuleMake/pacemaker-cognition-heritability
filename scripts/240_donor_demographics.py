"""Donor demographics for the conduction-axis dataset.

AJHG requires the sex of human subjects to be reported. This reads the
categorical codes straight out of the .h5ad obs group and tabulates, for the
whole axis dataset and for the donors contributing each conduction population,
the sex, age and self-reported ethnicity recorded by the atlas authors.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT / "data" / "singlecell" / "axis_subset.h5ad"
OUT = ROOT / "results" / "donor_demographics.json"

STATES = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]


def col(obs, name):
    """Return one obs column as a list of python strings."""
    g = obs[name]
    if isinstance(g, h5py.Group):                      # categorical
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in g["categories"][:]]
        codes = g["codes"][:]
        return [cats[i] if i >= 0 else "NA" for i in codes]
    v = g[:]
    if v.dtype.kind == "S":
        return [x.decode() for x in v]
    if v.dtype.kind == "O":
        return [x.decode() if isinstance(x, bytes) else str(x) for x in v]
    return [str(x) for x in v]


def main():
    with h5py.File(H5, "r") as f:
        obs = f["obs"]
        donor = col(obs, "donor_id")
        sex = col(obs, "sex")
        age = col(obs, "age")
        eth = col(obs, "self_reported_ethnicity")
        state = col(obs, "cell_state")

    per_donor = {}
    for d, s, a, e in zip(donor, sex, age, eth):
        per_donor.setdefault(d, (s, a, e))

    out = {"n_donors_axis_dataset": len(per_donor),
           "sex_all_donors": dict(Counter(v[0] for v in per_donor.values())),
           "age_all_donors": dict(Counter(v[1] for v in per_donor.values())),
           "ethnicity_all_donors": dict(Counter(v[2] for v in per_donor.values())),
           "per_population": {}}

    by_state = defaultdict(set)
    for d, st in zip(donor, state):
        if st in STATES:
            by_state[st].add(d)

    for st in STATES:
        ds = sorted(by_state[st])
        out["per_population"][st] = {
            "n_donors": len(ds),
            "donors": ds,
            "sex": dict(Counter(per_donor[d][0] for d in ds)),
            "age": sorted({per_donor[d][1] for d in ds}),
            "ethnicity": dict(Counter(per_donor[d][2] for d in ds)),
        }

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
