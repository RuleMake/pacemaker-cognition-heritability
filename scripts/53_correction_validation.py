"""
Does the paired contrast actually remove the depth confound? And what survives it?
=================================================================================

Script 52 established that gsMap's per-spot -log10(p) tracks sequencing depth at
rho ~ 0.75 in well-powered cardiac traits, and that its compartment ordering IS the
depth ordering (rho ~ 0.85). A confound of that size makes any single-trait gsMap map
in this tissue uninterpretable.

The paired trait-minus-control contrast was introduced earlier to cancel it. That was
an assumption. This script tests it, and then uses the corrected signal to answer the
one substantive question still open.

Part 1 — does the correction work?
    Correlate delta_logp = logp(trait) - logp(control) against depth, spot by spot.
    If the contrast works, the ~0.75 correlation should collapse toward zero. If it
    does not, every "corrected" result reported so far is still confounded and has to
    be withdrawn.

Part 2 — conditional analysis, the last unexcluded explanation
    Pacemaker abundance and depth are themselves correlated (node spots are dense).
    So the earlier finding "logp tracks SAN_P_cell abundance" could be depth in
    disguise. Partial Spearman of logp against SAN_P_cell CONTROLLING for total UMI —
    and separately controlling for atrial myocyte abundance — asks whether anything
    pacemaker-specific is left once both are held fixed.

Part 3 — does the correction rescue or bury the hypothesis?
    Re-rank the cell types using delta_logp instead of raw logp. If SAN_P_cell climbs
    once depth is removed, the hypothesis was buried by an artefact and deserves
    another look. If it does not move, the negative result stands on corrected data.

Usage:  python scripts/53_correction_validation.py
"""

import glob
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent.parent.as_posix()
GS = f"{ROOT}/data/spatial_gsmap"
SPOT = f"{ROOT}/results/spotlevel"
OUT = f"{ROOT}/results"
CONTROL = "EducationalAttainment"

FOCUS = ["SAN_P_cell", "AVN_bundle_cell", "aCM1", "aCM2", "aCM3", "aCM4",
         "vCM1", "vCM2", "vCM3_stressed", "FB1", "FB2", "FB4_activated",
         "NC1_glial", "NC2_glial_NGF+", "EC2_cap", "PC1_vent", "Adip1"]
ATRIAL = ["aCM1", "aCM2", "aCM3", "aCM4"]


def partial_spearman(x, y, *controls):
    """Spearman of x vs y with the linear effect of each control removed on ranks."""
    ok = np.isfinite(x) & np.isfinite(y)
    for c in controls:
        ok &= np.isfinite(c)
    if ok.sum() < 200:
        return np.nan
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    Z = np.column_stack([rankdata(c[ok]) for c in controls] + [np.ones(ok.sum())])
    rx = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ry = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ inputs
qc, abund = {}, {}
for p in sorted(glob.glob(f"{GS}/*.h5ad")):
    s = os.path.basename(p)[:-5]
    a = ad.read_h5ad(p)
    X = a.layers["count"] if "count" in a.layers else a.X
    qc[s] = pd.DataFrame(
        {"total_umi": np.asarray(X.sum(axis=1)).ravel(),
         "annotation": a.obs["annotation_final"].astype(str).to_numpy()},
        index=a.obs_names)
    cols = [c for c in FOCUS if c in a.obs.columns]
    if cols:
        abund[s] = a.obs[cols].astype(float)

# cell2location columns live in the source atlas rather than the gsMap input for some
# builds; fall back to it so the conditional analysis is not silently skipped
if not abund:
    for region in ["SAN", "AVN"]:
        p = f"{ROOT}/data/spatial/{region}.h5ad"
        if not os.path.exists(p):
            continue
        a = ad.read_h5ad(p, backed="r")
        sid = a.obs["sangerID"].astype(str)
        cols = [c for c in FOCUS if c in a.obs.columns]
        for s in sid.unique():
            m = (sid == s).to_numpy()
            sub = a.obs.loc[m, cols].astype(float)
            sub.index = a.obs_names[m]
            abund[f"{region}__{s}"] = sub

print(f"sections with depth: {len(qc)}   with cell2location: {len(abund)}\n")


def load(section, trait):
    p = f"{SPOT}/{section}__{trait}.csv"
    return pd.read_csv(p, index_col=0) if os.path.exists(p) else None


sections = sorted(qc)
traits = sorted({os.path.basename(p)[:-4].rsplit("__", 1)[1]
                 for p in glob.glob(f"{SPOT}/*__*.csv")})
cardiac = [t for t in traits if t != CONTROL]
order = [t for t in ["RestingHeartRate", "AtrialFibrillation", "PRinterval",
                     "HRV_RMSSD", "HRV_RMSSDc", "HRV_SDNN", "HRV_SDNNc"] if t in traits]
report = {}

# ================================================================ PART 1
print("=" * 92)
print("PART 1.  Does the trait-minus-control contrast remove the depth confound?")
print("=" * 92)
rows = []
for t in order:
    for s in sections:
        a, b = load(s, t), load(s, CONTROL)
        if a is None or b is None:
            continue
        c = a.index.intersection(b.index).intersection(qc[s].index)
        if len(c) < 200:
            continue
        raw = a.loc[c, "logp"].to_numpy()
        delta = raw - b.loc[c, "logp"].to_numpy()
        umi = qc[s].loc[c, "total_umi"].to_numpy()
        rows.append(dict(section=s, trait=t,
                         raw=spearmanr(raw, umi)[0], delta=spearmanr(delta, umi)[0]))
p1 = pd.DataFrame(rows)
p1.to_csv(f"{OUT}/correction_depth.tsv", sep="\t", index=False)

print(f"{'trait':<24}{'raw logp vs UMI':>18}{'delta logp vs UMI':>20}"
      f"{'reduction':>12}{'p':>10}")
for t in order:
    g = p1[p1.trait == t]
    if g.empty:
        continue
    r, dd = g.raw.mean(), g.delta.mean()
    pv = wilcoxon(g.raw.abs() - g.delta.abs()).pvalue if len(g) >= 6 else np.nan
    print(f"{t:<24}{r:>+18.3f}{dd:>+20.3f}"
          f"{(1 - abs(dd) / max(abs(r), 1e-9)) * 100:>11.0f}%"
          f"{(f'{pv:.4f}' if np.isfinite(pv) else '-'):>10}")
report["part1"] = {t: dict(raw=float(p1[p1.trait == t].raw.mean()),
                           delta=float(p1[p1.trait == t].delta.mean()))
                   for t in order if len(p1[p1.trait == t])}

# ================================================================ PART 2
print("\n" + "=" * 92)
print("PART 2.  Partial correlation: is anything pacemaker-specific left after")
print("         holding sequencing depth and atrial myocyte content fixed?")
print("=" * 92)
rows = []
for t in traits:
    for s in sections:
        if s not in abund or "SAN_P_cell" not in abund[s].columns:
            continue
        d = load(s, t)
        if d is None:
            continue
        c = d.index.intersection(qc[s].index).intersection(abund[s].index)
        if len(c) < 200:
            continue
        lp = d.loc[c, "logp"].to_numpy()
        san = abund[s].loc[c, "SAN_P_cell"].to_numpy()
        if not np.isfinite(san).any() or np.nanstd(san) == 0:
            continue
        umi = qc[s].loc[c, "total_umi"].to_numpy()
        acm = abund[s].loc[c, [x for x in ATRIAL if x in abund[s].columns]] \
            .sum(axis=1).to_numpy()
        rows.append(dict(
            section=s, trait=t,
            plain=spearmanr(lp, san, nan_policy="omit")[0],
            given_umi=partial_spearman(lp, san, umi),
            given_umi_acm=partial_spearman(lp, san, umi, acm)))
p2 = pd.DataFrame(rows)
if p2.empty:
    print("  no section carries SAN_P_cell abundance — conditional analysis skipped")
else:
    p2.to_csv(f"{OUT}/correction_partial.tsv", sep="\t", index=False)
    print(f"{'trait':<24}{'rho(logp,SAN_P)':>18}{'| depth':>12}"
          f"{'| depth+aCM':>14}{'sections':>10}")
    for t in [x for x in order + [CONTROL] if x in set(p2.trait)]:
        g = p2[p2.trait == t]
        tag = "  <- control" if t == CONTROL else ""
        print(f"{t:<24}{g.plain.mean():>+18.3f}{g.given_umi.mean():>+12.3f}"
              f"{g.given_umi_acm.mean():>+14.3f}{len(g):>10}{tag}")
    ctrl = p2[p2.trait == CONTROL].set_index("section")
    print(f"\n{'trait minus control, after conditioning on depth + aCM:':<60}")
    for t in order:
        g = p2[p2.trait == t].set_index("section")
        c = g.index.intersection(ctrl.index)
        if len(c) < 6:
            continue
        diff = (g.loc[c, "given_umi_acm"] - ctrl.loc[c, "given_umi_acm"]).dropna()
        pv = wilcoxon(diff).pvalue if len(diff) >= 6 else np.nan
        flag = "  <-- pacemaker-specific" if (diff.mean() > 0 and pv < 0.05) else ""
        print(f"  {t:<24}{diff.mean():>+9.3f}   p = "
              f"{(f'{pv:.4f}' if np.isfinite(pv) else '-'):>8}{flag}")
    report["part2"] = {t: dict(
        plain=float(p2[p2.trait == t].plain.mean()),
        given_umi=float(p2[p2.trait == t].given_umi.mean()),
        given_umi_acm=float(p2[p2.trait == t].given_umi_acm.mean()))
        for t in set(p2.trait)}

# ================================================================ PART 3
print("\n" + "=" * 92)
print("PART 3.  Cell-type ranking on depth-corrected signal (delta logp)")
print("         does the pacemaker climb once the artefact is gone?")
print("=" * 92)
rows = []
for t in order:
    for s in sections:
        if s not in abund:
            continue
        a, b = load(s, t), load(s, CONTROL)
        if a is None or b is None:
            continue
        c = a.index.intersection(b.index).intersection(abund[s].index)
        if len(c) < 200:
            continue
        delta = a.loc[c, "logp"].to_numpy() - b.loc[c, "logp"].to_numpy()
        for ct in abund[s].columns:
            v = abund[s].loc[c, ct].to_numpy()
            ok = np.isfinite(v) & np.isfinite(delta)
            if ok.sum() < 200 or np.nanstd(v[ok]) == 0:
                continue
            rows.append(dict(section=s, trait=t, cell_type=ct,
                             rho=float(spearmanr(delta[ok], v[ok])[0])))
p3 = pd.DataFrame(rows)
if not p3.empty:
    p3.to_csv(f"{OUT}/correction_celltype_rank.tsv", sep="\t", index=False)
    for t in order:
        g = p3[p3.trait == t]
        if g.empty:
            continue
        m = g.groupby("cell_type").rho.mean().sort_values(ascending=False)
        if "SAN_P_cell" not in m.index:
            continue
        rank = int(list(m.index).index("SAN_P_cell")) + 1
        top = ", ".join(m.index[:3])
        print(f"{t:<24} SAN_P_cell rank {rank:>2}/{len(m)}  "
              f"(rho {m['SAN_P_cell']:+.3f})   top: {top}")
    report["part3"] = {
        t: int(list(p3[p3.trait == t].groupby("cell_type").rho.mean()
                    .sort_values(ascending=False).index).index("SAN_P_cell")) + 1
        for t in order
        if len(p3[p3.trait == t]) and "SAN_P_cell" in set(p3[p3.trait == t].cell_type)}

with open(f"{OUT}/correction_validation.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwrote {OUT}/correction_*.tsv and correction_validation.json")
