"""
If QRS duration lands on Purkinje cells, which genes put it there?
==================================================================

A rank says a gene set is elevated in a cell type. It does not say the result is
interpretable. The sinoatrial arm of this project only became convincing when the top
drivers of heart-rate variability in pacemaker cells turned out to be RGS6 and CHRM2 —
the muscarinic receptor and its RGS partner, which is the vagal control machinery of
the sinoatrial node itself. That is a mechanism, not a p-value.

The ventricular arm needs the same treatment, and it has an equally sharp expectation.
The genetics of QRS duration and of ventricular conduction disease is dominated by the
cardiac sodium channel locus — SCN5A and its neighbour SCN10A — together with the TBX
and IRX transcription factors that specify the conduction system. If QRS duration
scores highly on Purkinje cells through SCN5A / SCN10A / IRX3, the result means what it
appears to mean. If it scores highly through sarcomere genes shared with every myocyte,
the ranking is a cell-size effect wearing a conduction label.

The comparison is what makes it a test: the same decomposition is run for the
heart-rate traits on the same cells. If Purkinje cells simply attract whatever is
cardiac, the driver lists will be interchangeable.

Usage:  python scripts/156_ventricular_drivers.py
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent.as_posix()
H5 = f"{ROOT}/data/singlecell/axis_subset.h5ad"
GS = f"{ROOT}/data/scdrs/traits.gs"
OUT = f"{ROOT}/results/ventricular_drivers.json"

CELLS = ["Purkinje", "AVN_bundle_cell", "SAN_P_cell", "AVN_P_cell"]
TRAITS = ["QRSduration", "BundleBranchBlock", "QTinterval", "Brugada",
          "HRV_RMSSD", "RestingHeartRate", "PRinterval"]

# fixed in advance from the ventricular-conduction literature
VENT_CONDUCTION = {"SCN5A", "SCN10A", "IRX3", "IRX5", "GJA5", "CNTN2", "TBX3", "TBX5",
                   "HCN4", "KCNJ2", "NKX2-5", "ETV1", "PCP4", "SLMAP", "CASQ2"}
# the sinoatrial machinery, for contrast
VAGAL = {"RGS6", "CHRM2", "KCNJ3", "KCNJ5", "GNB4", "CACNA1D", "SHOX2", "ISL1",
         "VSNL1", "GNAO1"}
# repolarisation, which QT should run through if the QT prediction is right
REPOL = {"KCNQ1", "KCNH2", "KCNE1", "KCNJ2", "NOS1AP", "SCN5A", "CACNA1C", "ATP1B1",
         "LITAF", "PLN"}

a = ad.read_h5ad(H5)
cs = a.obs["cell_state"].astype(str).to_numpy()
X = sp.csr_matrix(a.X) if not sp.issparse(a.X) else a.X.tocsr()
gidx = {g: i for i, g in enumerate(a.var_names)}
gs = pd.read_csv(GS, sep="\t").set_index("TRAIT")

report = {}
for cell in CELLS:
    m = cs == cell
    if m.sum() == 0:
        continue
    print("=" * 100)
    print(f"{cell}   (n={m.sum()})")
    print("=" * 100)
    report[cell] = {}
    for t in TRAITS:
        if t not in gs.index:
            print(f"   {t:<20}not scored")
            continue
        pairs = [p.split(":") for p in str(gs.loc[t, "GENESET"]).split(",") if ":" in p]
        pairs = [(s, float(w)) for s, w in pairs if s in gidx]
        if len(pairs) < 50:
            continue
        ii = [gidx[s] for s, _ in pairs]
        w = np.array([wt for _, wt in pairs])
        sub = X[:, ii]
        # contribution = weight x (mean in this cell type - mean everywhere)
        contrib = w * (np.asarray(sub[m].mean(axis=0)).ravel()
                       - np.asarray(sub.mean(axis=0)).ravel())
        order = np.argsort(-contrib)[:40]
        top = [pairs[i][0] for i in order]
        hits = {
            "ventricular conduction": [g for g in top if g in VENT_CONDUCTION],
            "vagal / nodal": [g for g in top if g in VAGAL],
            "repolarisation": [g for g in top if g in REPOL],
        }
        print(f"\n   {t}")
        print(f"      top: {', '.join(top[:10])}")
        for k, v in hits.items():
            if v:
                print(f"      {k:<24}{', '.join(v)}")
        report[cell][t] = dict(top=top[:20], **{k: v for k, v in hits.items()})
    print()

# ------------------------------------------------------------------ contrast
print("=" * 100)
print("ARE THE DRIVER LISTS INTERCHANGEABLE? (Jaccard of top 40, within Purkinje)")
print("=" * 100)
if "Purkinje" in report:
    ts = [t for t in TRAITS if t in report["Purkinje"]]
    sets = {t: set(report["Purkinje"][t]["top"]) for t in ts}
    print(f"{'':<22}" + "".join(f"{t[:12]:>14}" for t in ts))
    for t1 in ts:
        line = f"{t1:<22}"
        for t2 in ts:
            s1, s2 = sets[t1], sets[t2]
            line += f"{len(s1 & s2) / max(len(s1 | s2), 1):>14.2f}"
        print(line)
    print("\nIf the ventricular traits and the heart-rate traits share their drivers in")
    print("Purkinje cells, the cell is attracting anything cardiac and the ranking is")
    print("not evidence about conduction.")

with open(OUT, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nwrote {OUT}")
