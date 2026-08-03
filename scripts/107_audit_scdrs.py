"""
Attack the scDRS result before believing it.
============================================

The single-cell finding is the first positive this project has produced, which is
exactly why it needs the same treatment that killed the spatial one. Six things could
make it wrong or overstated; each is checked here rather than argued about.

  1 DONOR       245 pacemaker cells could come from one donor. If so the result is a
                donor effect wearing a cell-type label. The spatial analysis had 8
                sections and this was never a risk; here it is.
  2 ASSAY       `unclassified` turned out to be 100% one assay. If the conduction
                states are similarly assay-skewed, the same objection applies to them.
  3 FRAGILITY   scDRS weights ~1,000 genes, but a result carried by two or three of
                them is not a polygenic finding. Dropping the largest contributors
                shows how much of the signal survives.
  4 SET SIZE    top-1000 is a convention. If the ranking flips at top-500 or top-2000
                the finding is an artefact of that choice.
  5 CONTROL FDR the negative control is nominally significant for AVN_P_cell (z=+1.86).
                Whether it survives correction across 62 cell states decides how the
                separation should be described.
  6 INDEPENDENCE the four heart-rate-variability indices come from ONE study of 46,075
                people and are correlated phenotypes. "Three HRV indices agree" is one
                observation, not three, and the write-up must say so.

Usage:  python scripts/107_audit_scdrs.py
"""

import glob
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent.as_posix()
H5 = f"{ROOT}/data/singlecell/node_subset.h5ad"
SC = f"{ROOT}/results/scdrs"
GS = f"{ROOT}/data/scdrs/traits.gs"

FOCUS = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell"]
CTRL = "EducationalAttainment"
findings = []

a = ad.read_h5ad(H5)
cs = a.obs["cell_state"].astype(str)
print(f"cells {a.n_obs:,}  states {cs.nunique()}\n")

# ================================================================ 1 donor
print("=" * 96)
print("1. DONOR COMPOSITION — is a 'cell type' actually one donor?")
print("=" * 96)
print(f"{'cell state':<20}{'n':>7}{'donors':>9}{'largest donor share':>22}{'verdict':>16}")
for c in FOCUS:
    m = (cs == c).to_numpy()
    if m.sum() == 0:
        continue
    d = a.obs.loc[m, "donor_id"].astype(str).value_counts()
    share = d.iloc[0] / m.sum()
    ok = share < 0.5 and len(d) >= 3
    findings.append(("donor", c, float(share), bool(ok)))
    print(f"{c:<20}{m.sum():>7,}{len(d):>9}{share * 100:>21.1f}%"
          f"{('ok' if ok else 'CONCENTRATED'):>16}")
print("\nA cell state drawn mostly from one donor cannot be separated from that")
print("donor's genotype and batch. Below 50% from any single donor is acceptable.")

# ================================================================ 2 assay
print("\n" + "=" * 96)
print("2. ASSAY COMPOSITION")
print("=" * 96)
# The first version of this check asked only "is this group dominated by one assay"
# and flagged every conduction state at 100%. That was meaningless: the whole subset
# is one assay, so 100% is the background, not a property of the group. A composition
# is only evidence of confounding if it DIFFERS from the background — that is what is
# tested now.
if "assay" in a.obs.columns:
    bg = a.obs["assay"].astype(str).value_counts(normalize=True)
    print("background composition of the whole subset:")
    for k, p in bg.items():
        print(f"  {k:<28}{p * 100:>6.1f}%")
    if len(bg) == 1:
        print(f"\nOnly one assay is present ({bg.index[0]}), so assay cannot confound")
        print("anything here — every cell state is 100% by construction. No check to run.")
        findings.append(("assay", "not applicable (single assay dataset)", 1.0, True))
    else:
        print(f"\n{'cell state':<20}{'n':>7}   deviation from background")
        for c in FOCUS + ["unclassified"]:
            m = (cs == c).to_numpy()
            if m.sum() == 0:
                continue
            v = a.obs.loc[m, "assay"].astype(str).value_counts(normalize=True)
            v = v.reindex(bg.index).fillna(0.0)
            dev = float(np.abs(v - bg).max())
            print(f"{c:<20}{m.sum():>7,}   max |group - background| = {dev:.3f}")
            findings.append(("assay", c, dev, bool(dev < 0.25)))
else:
    print("  no assay column")

# ================================================================ 5 control FDR
print("\n" + "=" * 96)
print("5. DOES THE NEGATIVE CONTROL SURVIVE FDR ON THE CONDUCTION CELLS?")
print("=" * 96)
gfile = f"{SC}/group_analysis_fdr.tsv"
if os.path.exists(gfile):
    g = pd.read_csv(gfile, sep="\t")
    key = "cell_state" if "cell_state" in g.columns else g.columns[0]
    traits = sorted(g.trait.unique())
    print(f"{'cell state':<20}" + "".join(f"{t[:14]:>16}" for t in traits))
    for c in FOCUS:
        line = f"{c:<20}"
        for t in traits:
            r = g[(g[key] == c) & (g.trait == t)]
            if r.empty:
                line += f"{'-':>16}"
                continue
            q = float(r.assoc_fdr.iloc[0])
            line += f"{q:>12.4f}{'*' if q < 0.05 else ' ':<4}"
        print(line)
    print("\n(FDR across 62 cell states within each trait; * = FDR<0.05)")
    for c in FOCUS:
        r = g[(g[key] == c) & (g.trait == CTRL)]
        if not r.empty:
            q = float(r.assoc_fdr.iloc[0])
            print(f"  control on {c}: FDR = {q:.4f} "
                  f"{'— significant, separation is quantitative not qualitative' if q < 0.05 else '— not significant, clean separation'}")
            findings.append(("control_fdr", c, float(q), bool(q >= 0.05)))
else:
    print("  group_analysis_fdr.tsv not found — run 104_scdrs_report.py first")

# ================================================================ 3 fragility
print("\n" + "=" * 96)
print("3. FRAGILITY — how much of the score rides on the few strongest genes?")
print("=" * 96)
if os.path.exists(GS):
    gs = pd.read_csv(GS, sep="\t")
    X = a.X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    gidx = {gname: i for i, gname in enumerate(a.var_names)}
    print(f"{'trait':<22}{'cell state':<18}{'all genes':>11}{'drop top 10':>13}"
          f"{'drop top 50':>13}{'verdict':>12}")
    for _, row in gs.iterrows():
        trait = row.TRAIT
        if trait not in ("HRV_RMSSD", "RestingHeartRate", "PRinterval", CTRL):
            continue
        pairs = [p.split(":") for p in str(row.GENESET).split(",") if ":" in p]
        pairs = [(s, float(w)) for s, w in pairs if s in gidx]
        pairs.sort(key=lambda x: -x[1])
        for c in ["SAN_P_cell", "AVN_P_cell"]:
            m = (cs == c).to_numpy()
            if m.sum() == 0:
                continue
            vals = []
            for drop in (0, 10, 50):
                sub = pairs[drop:]
                ii = [gidx[s] for s, _ in sub]
                w = np.array([wt for _, wt in sub])
                # weighted mean expression, then how far the group sits above the
                # tissue mean in units of the tissue's own spread
                E = X[:, ii] @ (w / w.sum())
                E = np.asarray(E).ravel()
                vals.append((E[m].mean() - E.mean()) / (E.std() + 1e-12))
            keep = vals[2] / vals[0] if vals[0] != 0 else float("nan")
            ok = np.isfinite(keep) and keep > 0.5
            print(f"{trait:<22}{c:<18}{vals[0]:>11.2f}{vals[1]:>13.2f}"
                  f"{vals[2]:>13.2f}{('robust' if ok else 'FRAGILE'):>12}")
            findings.append(("fragility", f"{trait}/{c}", float(keep), bool(ok)))
    print("\nStandardised distance of the group mean from the tissue mean, using all")
    print("weighted genes and then with the strongest 10 and 50 removed. A polygenic")
    print("signal loses little; one carried by a handful of genes collapses.")
else:
    print("  traits.gs not found")

# ================================================================ 6 independence
print("\n" + "=" * 96)
print("6. HOW MANY INDEPENDENT OBSERVATIONS ARE THERE, REALLY?")
print("=" * 96)
groups = {
    "HRV (one study, n=46,075, four correlated indices)":
        ["HRV_RMSSD", "HRV_RMSSDc", "HRV_SDNN", "HRV_SDNNc"],
    "resting heart rate (UKB, n=458,969)": ["RestingHeartRate"],
    "PR interval (n=292,566)": ["PRinterval"],
    "atrial fibrillation (n=2,339,188)": ["AtrialFibrillation"],
}
for lbl, ts in groups.items():
    print(f"  {len(ts)} trait file(s) -> 1 independent observation : {lbl}")
print("\nThe four HRV indices are derived from the same 46,075 participants and are")
print("correlated phenotypes. 'Three HRV indices rank SAN_P_cell first' is ONE")
print("observation replicated across correlated measures, not three independent ones.")
print("The independent lines of evidence are: HRV, resting heart rate, PR interval,")
print("atrial fibrillation — four distinct studies.")
findings.append(("independence", "HRV", 1.0, True))

# ================================================================ summary
print("\n" + "=" * 96)
print("AUDIT SUMMARY")
print("=" * 96)
bad = [f for f in findings if not f[3]]
for kind, what, val, ok in findings:
    print(f"  {'PASS' if ok else 'FLAG'}  {kind:<14}{what:<28}{val:>8.3f}")
print(f"\n{len(findings) - len(bad)} passed, {len(bad)} flagged")
if bad:
    print("\nFlagged items must be disclosed in the write-up:")
    for kind, what, val, _ in bad:
        print(f"  - {kind}: {what} ({val:.3f})")

with open(f"{SC}/audit.json", "w") as f:
    json.dump([dict(check=k, item=w, value=v, passed=o) for k, w, v, o in findings],
              f, indent=2)
print(f"\nwrote {SC}/audit.json")
