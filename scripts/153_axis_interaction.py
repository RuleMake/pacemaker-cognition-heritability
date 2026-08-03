"""
Does each trait hit its OWN conduction structure? The interaction test.
======================================================================

A ranking says "HRV likes sinoatrial pacemaker cells". It does not say the sinoatrial
node is special to HRV, because the ranking is computed against a background that
differs between structures. On this subset those backgrounds differ in a way that
matters: the nodal conduction states are 100% 10x multiome while the atlas is only a
third multiome, and Purkinje cells sit in apex tissue that is half 3' v2. Comparing a
multiome cell type against a mixed background and calling the difference anatomy is the
same class of error as the sequencing-depth artefact that killed the spatial arm.

So the statistic is never a raw score and never a global rank. It is

    AUC(focus cells vs every other cell in the SAME donor, region and assay)

Inside one stratum the chemistry, batch, genotype and dissection are all constant, so
none of them can move a cell type relative to its neighbours.

Why the strata are combined rather than filtered
------------------------------------------------
The 110 Purkinje cells are spread over 12 donor/region/assay strata: 36, 27, 11, 8, 7,
7, 4, 4, 2, 2, 1, 1. A minimum-size rule of 15 cells would keep two strata and throw
away 43% of the cells — and would rest the ventricular arm on two batches, which is the
donor objection this project has already had to answer once.

Instead every stratum contributes, weighted by how much it can say. This is van
Elteren's stratified Wilcoxon: each stratum yields its own rank-sum, they are combined
with weights 1/(N+1), and a stratum of two cells simply carries less weight than a
stratum of thirty-six rather than being discarded. Strata are still never pooled — no
cell is ever ranked against a cell from a different donor, region or chemistry.

The claim, and how it can fail
------------------------------
    D = [AUC_SAN(heart-rate traits) - AUC_SAN(QRS traits)]
      + [AUC_AX (QRS traits)        - AUC_AX (heart-rate traits)]

Each term still carries how strong a trait is in general — a trait scoring well
everywhere inflates the first and deflates the second by the same amount. Their SUM
cancels that, and what is left is anatomy. D <= 0 means the traits do not separate by
structure and the specificity map is measuring cell class, not anatomical position.

Significance is by exact enumeration: reassign which traits belong to which arm in
every possible way and see where the observed D falls.

Usage:  python scripts/153_axis_interaction.py
"""

import json
import os
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.as_posix()
H5 = f"{ROOT}/data/singlecell/axis_subset.h5ad"
SC = f"{ROOT}/results/scdrs_axis"
PRED = f"{ROOT}/results/axis_predictions.json"

SAN_TRAITS = ["HRV_RMSSD", "HRV_RMSSDc", "HRV_SDNN", "HRV_SDNNc", "RestingHeartRate"]
HIS_TRAITS = ["QRSduration", "BundleBranchBlock"]
CONTROLS = ["EducationalAttainment", "RheumatoidArthritis", "RheumatoidArthritis_noMHC"]
FOCUS = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
EXCLUDE_FROM_RANK = {"unclassified"}
MIN_OTHER = 100
MIN_CM = 30      # the myocyte comparator is a smaller pool; do not demand 100 of it

# ------------------------------------------------------------------ inputs
a = ad.read_h5ad(H5, backed="r")
obs = a.obs[["donor_id", "region", "assay", "cell_state"]].astype(str)
obs["stratum"] = obs.donor_id + "/" + obs.region + "/" + obs.assay

scores = {}
for p in sorted(Path(SC).glob("*.score.tsv")):
    t = p.name.replace(".score.tsv", "")
    s = pd.read_csv(p, sep="\t", index_col=0)["norm_score"].reindex(obs.index)
    if s.isna().any():
        raise SystemExit(f"{t}: score file does not cover every cell")
    scores[t] = s.to_numpy()
if not scores:
    raise SystemExit(f"nothing scored in {SC}")

have = set(scores)
SAN_TRAITS = [t for t in SAN_TRAITS if t in have]
HIS_TRAITS = [t for t in HIS_TRAITS if t in have]
print(f"cells {len(obs):,}   strata {obs.stratum.nunique()}   traits {len(have)}")
print(f"sinoatrial arm : {', '.join(SAN_TRAITS) or 'NONE'}")
print(f"His-Purkinje arm: {', '.join(HIS_TRAITS) or 'NONE'}")
if not HIS_TRAITS:
    # not fatal: the stratified table below is still worth printing, and printing it
    # on a partial run is how this script gets debugged before the traits land
    print("!! no His-Purkinje trait scored yet — the interaction will be skipped")

strat_codes, strat_names = pd.factorize(obs.stratum)
state = obs.cell_state.to_numpy()


def stratified_all(trait, background="all"):
    """van Elteren stratified AUC + Z for EVERY cell state, in one pass per stratum.

    `background` decides what each state is ranked against, and it matters more than
    it looks. With "all", a conduction cell is compared to everything in its stratum —
    fibroblasts, immune cells, endothelium — and a cardiac gene set will put any
    myocyte-like cell above those. That contrast is real but it is not the question:
    it makes SAN_P_cell win nearly every trait, because a sinoatrial stratum contains
    few cardiomyocytes to compete with, while an apex stratum is full of ventricular
    myocytes for Purkinje to be diluted against. The two structures are not facing
    comparable opposition.

    With "cm" each conduction cell is ranked only against the working cardiomyocytes
    beside it, in the same donor, region and chemistry. That is the anatomically
    precise question — is the conduction cell enriched ABOVE the working muscle of its
    own chamber — and it puts the sinoatrial node and the Purkinje network on equal
    footing.

    Ranking the whole stratum once and then summing each state's ranks is exactly the
    Mann-Whitney statistic of that state against the rest of its stratum — the same
    number as ranking each state separately, at a fraction of the work. It also yields
    the assay- and donor-clean version of the global ranking, which is the table the
    global scDRS ranking cannot honestly provide on this subset.

    Every stratum contributes, weighted 1/(N+1). A stratum holding one Purkinje cell
    is kept and simply carries little weight; a minimum-size rule would have discarded
    43% of the Purkinje cells and rested the ventricular arm on two batches.
    """
    v = scores[trait]
    keep = np.ones(len(state), dtype=bool) if background == "all" else (IS_CM | IS_FOCUS)
    floor = MIN_OTHER if background == "all" else MIN_CM
    num, den = {}, {}
    stat, var = {}, {}
    ns, nc = {}, {}
    for si in range(len(strat_names)):
        m = (strat_codes == si) & keep
        N = int(m.sum())
        if N < floor:
            continue
        r = stats.rankdata(v[m])
        st = state[m]
        rs = pd.Series(r).groupby(st).agg(["sum", "size"])
        for c, (W, n1) in rs.iterrows():
            n1 = int(n1)
            n2 = N - n1
            if n2 < floor:
                continue
            auc_i = (W - n1 * (n1 + 1) / 2) / (n1 * n2)
            w = n1 * n2 / (N + 1)
            num[c] = num.get(c, 0.0) + w * auc_i
            den[c] = den.get(c, 0.0) + w
            stat[c] = stat.get(c, 0.0) + (W - n1 * (N + 1) / 2) / (N + 1)
            var[c] = var.get(c, 0.0) + n1 * n2 / (12.0 * (N + 1))
            ns[c] = ns.get(c, 0) + 1
            nc[c] = nc.get(c, 0) + n1
    out = {}
    for c in den:
        if den[c] > 0 and var[c] > 0:
            out[c] = dict(auc=num[c] / den[c], z=stat[c] / np.sqrt(var[c]),
                          n_strata=ns[c], n_cells=nc[c])
    return out


IS_CM = pd.Series(state).str.match(r"^(aCM|vCM)").to_numpy()
IS_FOCUS = np.isin(state, FOCUS)
print(f"working cardiomyocytes available as the comparator: {IS_CM.sum():,}")

ALL = {t: stratified_all(t, "all") for t in scores}
CM = {t: stratified_all(t, "cm") for t in scores}


def stratified_auc(cell, trait):
    """Primary statistic: against the working myocytes of the same stratum."""
    return CM.get(trait, {}).get(cell)


# ------------------------------------------------------------------ 1 table
print("\n" + "=" * 104)
print("1. STRATIFIED AUC vs WORKING MYOCYTES of the same donor / region / assay")
print("   (0.5 = no different from the muscle next to it)")
print("=" * 104)
tab = {}
present = [c for c in FOCUS if (state == c).any()]
print(f"{'trait':<28}" + "".join(f"{c[:17]:>21}" for c in present))
for t in sorted(have, key=lambda x: (x in CONTROLS, x)):
    line = f"{t:<28}"
    tab[t] = {}
    for c in present:
        r = stratified_auc(c, t)
        if r is None:
            line += f"{'-':>21}"
            continue
        tab[t][c] = r
        cell = "%.3f (z%+.1f)" % (r["auc"], r["z"])
        line += f"{cell:>21}"
    print(line + ("   <- control" if t in CONTROLS else ""))

print("\nstrata used / cells used per focus type:")
for c in present:
    r = next((tab[t][c] for t in tab if c in tab[t]), None)
    if r:
        print(f"   {c:<20}{r['n_strata']:>3} strata, {r['n_cells']:>4} cells")

# ------------------------------------------------------------------ 2 interaction
print("\n" + "=" * 104)
print("2. INTERACTION — difference of differences across structures")
print("=" * 104)


def armauc(cell, traits):
    vals = [tab[t][cell]["auc"] for t in traits if cell in tab.get(t, {})]
    return float(np.mean(vals)) if vals else None


def interaction(san_t, his_t):
    a1, b1 = armauc("SAN_P_cell", san_t), armauc("SAN_P_cell", his_t)
    a2, b2 = armauc("Purkinje", his_t), armauc("Purkinje", san_t)
    if None in (a1, b1, a2, b2):
        return None
    return (a1 - b1) + (a2 - b2), (a1 - b1), (a2 - b2)


obs_D = interaction(SAN_TRAITS, HIS_TRAITS)
out = {}
if obs_D is None:
    print("   cannot be formed — a required cell type or trait is missing")
else:
    D, t1, t2 = obs_D
    print(f"   inside SAN tissue : heart-rate arm - His arm = {t1:+.3f}")
    print(f"   inside apex tissue: His arm - heart-rate arm = {t2:+.3f}")
    print(f"   interaction D     = {D:+.3f}")
    out.update(D=float(D), term_san=float(t1), term_ax=float(t2))

    pool = SAN_TRAITS + HIS_TRAITS
    null = []
    for his in combinations(pool, len(HIS_TRAITS)):
        r = interaction([t for t in pool if t not in his], list(his))
        if r:
            null.append(r[0])
    if null:
        p = float(np.mean(np.array(null) >= D))
        out.update(perm_p=p, perm_n=len(null), null_mean=float(np.mean(null)))
        print(f"\n   exact permutation over all {len(null)} arm assignments: p = {p:.4f}"
              f"   (null mean {np.mean(null):+.3f})")

    if D > 0 and t1 > 0 and t2 > 0:
        print("\n   ANATOMY HOLDS in both directions")
    elif D > 0:
        print("\n   PARTIAL: the interaction is positive but one direction carries it")
    else:
        print("\n   FAILED: the traits do not separate by structure. The specificity")
        print("   map is measuring cell class, not anatomical position.")

# ------------------------------------------------------------------ 3 scorecard
print("\n" + "=" * 104)
print("3. PRE-REGISTERED PREDICTIONS — scored against results/axis_predictions.json")
print("=" * 104)
reg = json.load(open(PRED)) if os.path.exists(PRED) else {}
preds = reg.get("predictions", {})
novel = set(reg.get("novel", []))
print("""3a. AS REGISTERED — rank among all cell states, 'none of them' = outside top 5.
    This is the criterion written down before scoring. The myocyte-based statistic
    below is a better measurement, but it was defined AFTER seeing that the strata
    are not compositionally comparable, so it cannot be scored as a prediction.
""")
rows = []
print(f"{'trait':<26}{'predicted':<30}{'best conduction cell':<24}"
      f"{'overall #1':<16}result")
for t in sorted(have):
    p = preds.get(t)
    if not p or t not in ALL:
        continue
    allst = {c: d["auc"] for c, d in ALL[t].items() if c not in EXCLUDE_FROM_RANK}
    cond = {c: allst[c] for c in FOCUS if c in allst}
    if not cond:
        continue
    order = sorted(allst, key=allst.get, reverse=True)
    win = max(cond, key=cond.get)
    rank = order.index(win) + 1
    want = p["conduction"]
    if want is None:
        ok, wtxt = rank > 5, "none in the top 5"
    else:
        want = [want] if isinstance(want, str) else want
        ok, wtxt = win in want, " or ".join(want)
    star = " *" if t in novel else ""
    print(f"{t:<26}{wtxt:<30}{win + f' #{rank}/{len(order)}':<24}"
          f"{order[0][:14]:<16}{'PASS' if ok else 'fail'}{star}")
    rows.append(dict(trait=t, predicted=wtxt, observed=win, rank=rank,
                     n_states=len(order), passed=bool(ok), novel=t in novel,
                     criterion="registered"))

print("""
3b. POST HOC — AUC against the working myocytes of the same donor/region/assay.
    Introduced because a sinoatrial stratum holds few cardiomyocytes to compete with
    while an apex stratum is full of them, so the ranking above is not comparable
    across structures. The z > 2 bar was chosen after seeing the numbers. Read this
    as a measurement, not as a passed test.
""")
print(f"{'trait':<26}{'predicted':<30}{'strongest conduction cell':<30}result")
for t in sorted(have):
    p = preds.get(t)
    if not p or t not in CM:
        continue
    cond = {c: CM[t][c] for c in FOCUS if c in CM[t]}
    if not cond:
        continue
    win = max(cond, key=lambda c: cond[c]["auc"])
    elevated = [c for c in cond if cond[c]["z"] > 2]
    want = p["conduction"]
    if want is None:
        ok, wtxt = not elevated, "none above the myocytes"
    else:
        want = [want] if isinstance(want, str) else want
        ok = win in want and cond[win]["z"] > 2
        wtxt = " or ".join(want)
    star = " *" if t in novel else ""
    obs = "%s %.3f (z%+.1f)" % (win, cond[win]["auc"], cond[win]["z"])
    print(f"{t:<26}{wtxt:<30}{obs:<30}{'PASS' if ok else 'fail'}{star}")
    rows.append(dict(trait=t, predicted=wtxt, observed=win, auc=cond[win]["auc"],
                     z=cond[win]["z"], n_elevated=len(elevated),
                     passed=bool(ok), novel=t in novel, criterion="post_hoc"))

sc = pd.DataFrame(rows)
if len(sc):
    for crit, g in sc.groupby("criterion"):
        print(f"\n  {crit:<12}{int(g.passed.sum())}/{len(g)} held", end="")
        if int(g.novel.sum()):
            print(f"   (of the {int(g.novel.sum())} never tested before: "
                  f"{int(g[g.novel].passed.sum())}/{int(g.novel.sum())})", end="")
        print()
    sc.to_csv(f"{SC}/axis_scorecard.tsv", sep="\t", index=False)

out["stratified_auc"] = {t: {c: tab[t][c] for c in tab[t]} for t in tab}
# the full stratified ranking, which is the honest replacement for the global one
full = []
for t, d in ALL.items():
    o = sorted((c for c in d if c not in EXCLUDE_FROM_RANK),
               key=lambda c: -d[c]["auc"])
    for i, c in enumerate(o, 1):
        full.append(dict(trait=t, cell_state=c, rank=i, n_states=len(o),
                         auc=d[c]["auc"], z=d[c]["z"], n_cells=d[c]["n_cells"],
                         n_strata=d[c]["n_strata"]))
pd.DataFrame(full).to_csv(f"{SC}/stratified_ranking.tsv", sep="\t", index=False)
with open(f"{SC}/axis_interaction.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nwrote {SC}/axis_interaction.json and axis_scorecard.tsv")
