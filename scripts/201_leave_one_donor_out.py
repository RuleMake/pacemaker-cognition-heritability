"""
Does the result survive losing any one donor?
=============================================

The primary statistic is already stratified by donor, region and assay, so no cell is
ever ranked against a cell from a different donor. That protects against batch effects
but NOT against concentration: if one donor happens to contribute 60% of the pacemaker
cells and that donor's cells behave differently, the pooled statistic is that donor's
result wearing a weighted average as a disguise.

Two questions, and they are not the same one:

  leave-one-out   drop each donor's strata and recompute. If the effect collapses when
                  a particular donor leaves, the effect was that donor's.
  per-donor       compute the effect inside each donor separately. This counts how many
                  independent people show it, which is what "replication" means when
                  there is no second cohort to be had.

The second is the harder test and the one a reviewer will ask for, because a pooled
effect can be positive while every individual donor is null — the weights do the work.

The bar is not p < 0.05 per donor. A single donor contributes tens of pacemaker cells,
so per-donor tests are underpowered by construction; demanding significance from each
would be the "unpowered negative" error this project just spent a session cataloguing.
The bar is the SIGN and the SIZE: does every donor point the same way, and is the
spread across donors small compared to the effect?

Usage:  python scripts/201_leave_one_donor_out.py
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.as_posix()
H5 = f"{ROOT}/data/singlecell/axis_subset.h5ad"
SC = f"{ROOT}/results/scdrs_axis"
OUT = f"{ROOT}/results/leave_one_donor_out.json"

FOCUS = ["SAN_P_cell", "AVN_P_cell"]
TRAITS = ["EducationalAttainment", "Intelligence", "ReactionTime",
          "HRV_SDNN", "HRV_RMSSD", "RestingHeartRate",
          "RheumatoidArthritis", "QTinterval"]
MIN_CM = 30

# ------------------------------------------------------------------ inputs
a = ad.read_h5ad(H5, backed="r")
obs = a.obs[["donor_id", "region", "assay", "cell_state"]].astype(str)
obs["stratum"] = obs.donor_id + "/" + obs.region + "/" + obs.assay

scores = {}
for t in TRAITS:
    p = Path(SC) / f"{t}.score.tsv"
    if p.exists():
        s = pd.read_csv(p, sep="\t", index_col=0)["norm_score"].reindex(obs.index)
        if s.isna().any():
            raise SystemExit(f"{t}: score file does not cover every cell")
        scores[t] = s.to_numpy()
TRAITS = [t for t in TRAITS if t in scores]

state = obs.cell_state.to_numpy()
donor = obs.donor_id.to_numpy()
strat_codes, strat_names = pd.factorize(obs.stratum)
IS_CM = pd.Series(state).str.match(r"^(aCM|vCM)").to_numpy()
IS_FOCUS = np.isin(state, FOCUS)
KEEP = IS_CM | IS_FOCUS


def van_elteren(trait, cells_mask):
    """Myocyte-referenced stratified AUC + Z, restricted to `cells_mask`."""
    v = scores[trait]
    num, den, stat, var, ns, nc = {}, {}, {}, {}, {}, {}
    for si in range(len(strat_names)):
        m = (strat_codes == si) & KEEP & cells_mask
        N = int(m.sum())
        if N < MIN_CM:
            continue
        r = stats.rankdata(v[m])
        rs = pd.Series(r).groupby(state[m]).agg(["sum", "size"])
        for c, (W, n1) in rs.iterrows():
            n1 = int(n1)
            n2 = N - n1
            if n2 < MIN_CM or c not in FOCUS:
                continue
            auc_i = (W - n1 * (n1 + 1) / 2) / (n1 * n2)
            w = n1 * n2 / (N + 1)
            num[c] = num.get(c, 0.0) + w * auc_i
            den[c] = den.get(c, 0.0) + w
            stat[c] = stat.get(c, 0.0) + (W - n1 * (N + 1) / 2) / (N + 1)
            var[c] = var.get(c, 0.0) + n1 * n2 / (12.0 * (N + 1))
            ns[c] = ns.get(c, 0) + 1
            nc[c] = nc.get(c, 0) + n1
    return {c: dict(auc=num[c] / den[c], z=stat[c] / np.sqrt(var[c]),
                    n_strata=ns[c], n_cells=nc[c])
            for c in den if den[c] > 0 and var[c] > 0}


ALL = np.ones(len(obs), dtype=bool)
full = {t: van_elteren(t, ALL) for t in TRAITS}

# which donors actually carry each focus state
donors_of = {c: sorted(set(donor[state == c])) for c in FOCUS}
print("=" * 100)
print("DONOR STRUCTURE")
print("=" * 100)
for c in FOCUS:
    n = [(d, int(((state == c) & (donor == d)).sum())) for d in donors_of[c]]
    n.sort(key=lambda x: -x[1])
    tot = sum(k for _, k in n)
    top = n[0][1] / tot * 100
    print(f"{c:<18} {tot:>4} cells over {len(n)} donors   "
          f"largest donor holds {top:.0f}%   {[k for _, k in n]}")

# ------------------------------------------------------------------ 1. LODO
print("\n" + "=" * 100)
print("1. LEAVE ONE DONOR OUT — does any single donor carry the effect?")
print("=" * 100)
lodo = {}
for c in FOCUS:
    print(f"\n--- {c} ---")
    print(f"{'trait':<24}{'full':>8}{'LODO min':>10}{'LODO max':>10}"
          f"{'worst donor':>16}{'verdict':>14}")
    for t in TRAITS:
        if c not in full[t]:
            continue
        vals = {}
        for d in donors_of[c]:
            r = van_elteren(t, donor != d)
            if c in r:
                vals[d] = r[c]["auc"]
        if not vals:
            continue
        lo, hi = min(vals.values()), max(vals.values())
        worst = min(vals, key=vals.get)
        f = full[t][c]["auc"]
        # a real effect should not depend on any one person: the whole LODO range
        # must stay on the same side of 0.5 as the pooled estimate
        if f > 0.5:
            ok = lo > 0.5
        else:
            ok = hi < 0.5
        lodo[f"{t}|{c}"] = dict(full=f, lodo_min=lo, lodo_max=hi,
                                worst_donor=worst, robust=bool(ok),
                                per_donor_dropped={k: float(v) for k, v in vals.items()})
        print(f"{t:<24}{f:>8.3f}{lo:>10.3f}{hi:>10.3f}{worst:>16}"
              f"{('robust' if ok else 'FRAGILE'):>14}")

# ------------------------------------------------------------------ 2. per donor
print("\n" + "=" * 100)
print("2. PER DONOR — how many independent people show it?")
print("=" * 100)
print("Each donor alone. Underpowered by construction, so read the sign and the")
print("spread, not the p-values.\n")
per = {}
for c in FOCUS:
    print(f"--- {c} ---")
    ds = donors_of[c]
    print(f"{'trait':<24}" + "".join(f"{d[:8]:>10}" for d in ds)
          + f"{'n>0.5':>8}{'median':>9}")
    for t in TRAITS:
        row, ns = {}, 0
        for d in ds:
            r = van_elteren(t, donor == d)
            if c in r:
                row[d] = r[c]["auc"]
        if not row:
            continue
        v = np.array(list(row.values()))
        ns = int((v > 0.5).sum())
        per[f"{t}|{c}"] = dict(per_donor={k: float(x) for k, x in row.items()},
                               n_above_half=ns, n_donors=len(row),
                               median=float(np.median(v)))
        cells = "".join(f"{row[d]:>10.3f}" if d in row else f"{'-':>10}" for d in ds)
        print(f"{t:<24}{cells}{ns:>4}/{len(row):<3}{np.median(v):>9.3f}")
    print()

# ------------------------------------------------------------------ verdict
print("=" * 100)
print("VERDICT")
print("=" * 100)
frag = [k for k, v in lodo.items() if not v["robust"]]
cog = ["EducationalAttainment", "Intelligence", "ReactionTime"]
for c in FOCUS:
    hits = [per[f"{t}|{c}"] for t in cog if f"{t}|{c}" in per]
    if hits:
        tot = hits[0]["n_donors"]
        every = all(h["n_above_half"] == h["n_donors"] for h in hits)
        print(f"  {c}: cognitive traits point the same way in "
              f"{'ALL' if every else 'not all'} {tot} donors")
if frag:
    print(f"\n  FRAGILE (LODO range crosses 0.5): {', '.join(frag)}")
else:
    print("\n  No trait x cell result flips sign when any single donor is removed.")

json.dump(dict(donors_of={k: list(v) for k, v in donors_of.items()},
               full={t: full[t] for t in TRAITS},
               lodo=lodo, per_donor=per), open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
