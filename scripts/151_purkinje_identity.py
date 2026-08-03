"""
Are the 110 cells labelled Purkinje actually Purkinje?
======================================================

The entire ventricular arm of the specificity test rests on this one label. Purkinje
fibres are among the hardest cardiac cells to recover — they are rare, fragile, and
sit inside dense ventricular myocardium — so "Purkinje" appearing in an obs column is
not by itself evidence that Purkinje cells were captured. If the label is really
ventricular myocytes with an odd profile, then QRS duration landing on it proves
nothing about the conduction system, and QRS duration NOT landing on it proves nothing
either. The label has to be checked before it is used, not after it gives a nice
answer.

The check is marker expression against panels fixed from the literature, and it has a
built-in discriminator: Purkinje cells must look ventricular (they are part of the
ventricle) AND carry the conduction programme (which ordinary ventricular myocytes do
not). A label that is high on ventricular identity but flat on GJA5 / IRX3 / CNTN2 is
a working myocyte, whatever it is called.

The same panels are run on the nodal cells, where the answer is already known from the
node-subset analysis. If SAN_P_cell fails its own panel, the panels are wrong, not the
data — that is what makes this a check rather than a confirmation.

Usage:  python scripts/151_purkinje_identity.py
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.as_posix()
H5 = f"{ROOT}/data/singlecell/axis_subset.h5ad"
OUT = f"{ROOT}/results/purkinje_identity.json"

PANELS = {
    "ventricular conduction": ["GJA5", "IRX3", "CNTN2", "ETV1", "PCP4", "SCN5A"],
    "pacemaker / nodal":      ["HCN4", "SHOX2", "TBX3", "ISL1", "VSNL1", "CACNA1D"],
    "ventricular identity":   ["MYL2", "MYH7", "IRX4", "HEY2"],
    "atrial identity":        ["NPPA", "MYL7", "NR2F2", "KCNJ3"],
    "working myocyte":        ["TTN", "RYR2", "ACTC1", "TNNT2"],
}
FOCUS = ["Purkinje", "SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell"]
REF_PREFIX = {"vCM": "ventricular myocyte", "aCM": "atrial myocyte"}

a = ad.read_h5ad(H5)
cs = a.obs["cell_state"].astype(str).to_numpy()
genes = {g: i for i, g in enumerate(a.var_names)}
X = a.X

groups = {c: (cs == c) for c in FOCUS if (cs == c).any()}
for pre in REF_PREFIX:
    m = pd.Series(cs).str.startswith(pre).to_numpy()
    if m.any():
        groups[pre + "*"] = m
groups["all other cells"] = ~np.isin(cs, list(FOCUS)) & ~pd.Series(cs).str.startswith(
    tuple(REF_PREFIX)).to_numpy()

print(f"{'group':<20}{'n':>7}   " + "".join(f"{k[:20]:>22}" for k in PANELS))
scores = {}
for name, m in groups.items():
    line = f"{name:<20}{int(m.sum()):>7}   "
    scores[name] = {}
    for pname, panel in PANELS.items():
        ii = [genes[g] for g in panel if g in genes]
        if not ii:
            line += f"{'-':>22}"
            continue
        # z of this group's mean against the spread of group means, so panels with
        # different absolute expression are comparable
        v = float(np.asarray(X[m][:, ii].mean(axis=0)).ravel().mean())
        scores[name][pname] = v
        line += f"{v:>22.3f}"
    print(line)

# ------------------------------------------------------------------ standardise
print("\n" + "=" * 110)
print("SAME NUMBERS AS z ACROSS GROUPS — which group each panel actually marks")
print("=" * 110)
tab = pd.DataFrame(scores).T
z = (tab - tab.mean()) / tab.std()
print(z.round(2).to_string())

# ------------------------------------------------------------------ verdict
print("\n" + "=" * 110)
print("VERDICT")
print("=" * 110)
findings = {}


def top(panel):
    return z[panel].idxmax()


checks = [
    ("ventricular conduction panel should mark Purkinje",
     top("ventricular conduction") == "Purkinje"),
    ("pacemaker panel should mark a nodal cell, not Purkinje",
     top("pacemaker / nodal") in ("SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell")),
    ("Purkinje should look ventricular, not atrial",
     z.loc["Purkinje", "ventricular identity"]
     > z.loc["Purkinje", "atrial identity"]),
    ("Purkinje must be separable from ordinary ventricular myocytes",
     z.loc["Purkinje", "ventricular conduction"]
     > z.loc["vCM*", "ventricular conduction"] if "vCM*" in z.index else False),
]
for label, ok in checks:
    findings[label] = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")

allok = all(findings.values())
print("\n" + ("The Purkinje label carries the ventricular conduction programme and is "
              "distinct\nfrom working ventricular myocytes. It can be used."
              if allok else
              "The Purkinje label does NOT behave like ventricular conduction tissue.\n"
              "Any result resting on it — positive or negative — must be withdrawn."))

with open(OUT, "w") as f:
    json.dump(dict(panel_means=scores, checks=findings, usable=allok),
              f, indent=2, default=float)
print(f"\nwrote {OUT}")
