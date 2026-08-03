"""
Empirical calibration of the primary within-stratum statistic
=============================================================

The primary statistic is a van Elteren-weighted Mann-Whitney AUC of a conduction
population against the myocyte lineage of the same donor, region and chemistry,
with the null variance taken analytically as sum_strata n1*n2/(12*(N+1)).

That formula is not an approximation. For untied ranks it is the *exact*
randomisation variance of the statistic, so the objection "your standard error
assumes independent cells" is, in its simple form, answerable by arithmetic
rather than by simulation. What the formula does assume is that cells within a
stratum are EXCHANGEABLE under the null, and that is the assumption worth
attacking, because two things can break it:

  ties            scDRS normalised scores are continuous, but ties from Monte
                  Carlo flooring would shrink the true variance below the formula,
                  making z conservative rather than inflated. Test A measures it.

  nuisance
  structure       if pacemaker cells differ from myocytes in sequencing depth, and
                  depth moves the score, then a label permutation that ignores
                  depth is testing a null nobody believes. Test B permutes only
                  within depth blocks, so the null is allowed to use the same
                  depth structure the observed data has.

  donor
  clustering      245 cells from 6 donors are not 245 independent observations of
                  anything. Test C resamples donors, not cells, which is the
                  interval a reviewer means when asking how many people show it.

Test A can only exonerate the analytic SE. Tests B and C can kill the result.

Usage:  python scripts/250_permutation_null.py [n_perm]        (default 10000)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "results" / "obs_axis.tsv"
COV = ROOT / "data" / "scdrs" / "covariates_axis.tsv"
SC = ROOT / "results" / "scdrs_axis"
OUT_JSON = ROOT / "results" / "permutation_null.json"
OUT_TSV = ROOT / "results" / "permutation_null.tsv"

# identical to 201_leave_one_donor_out.py
MIN_CM = 30
FOCUS = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
TRAITS = ["EducationalAttainment", "Intelligence", "ReactionTime",
          "HRV_SDNN", "HRV_RMSSD", "RestingHeartRate",
          "QTinterval", "Brugada",
          "RheumatoidArthritis", "RheumatoidArthritis_noMHC"]

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
N_BOOT = 10000
DEPTH_BLOCKS = 5          # quintiles of log total UMI, within stratum
SEED = 20260803

rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ inputs
obs = pd.read_csv(OBS, sep="\t", index_col=0)
cov = pd.read_csv(COV, sep="\t", index_col=0).reindex(obs.index)
if cov["log_total"].isna().any():
    raise SystemExit("covariates do not cover every cell")

obs["stratum"] = obs.donor_id.astype(str) + "/" + obs.region.astype(str) + "/" + obs.assay.astype(str)
state = obs.cell_state.to_numpy().astype(str)
donor = obs.donor_id.to_numpy().astype(str)
depth = cov["log_total"].to_numpy()
strat_codes, strat_names = pd.factorize(obs.stratum)

IS_CM = pd.Series(state).str.match(r"^(aCM|vCM)").to_numpy()
KEEP_ANY = IS_CM | np.isin(state, FOCUS)

scores = {}
for t in TRAITS:
    p = SC / f"{t}.score.tsv"
    if not p.exists():
        print(f"  [skip] {t}: no score file")
        continue
    s = pd.read_csv(p, sep="\t", index_col=0)["norm_score"].reindex(obs.index)
    if s.isna().any():
        raise SystemExit(f"{t}: score file does not cover every cell")
    scores[t] = s.to_numpy()
TRAITS = [t for t in TRAITS if t in scores]


# ------------------------------------------------------------------ geometry
def contributing_strata(focus):
    """Strata that pass the MIN_CM gates for this focus population.

    Reproduces 201_leave_one_donor_out.py exactly: N counts every KEEP cell in
    the stratum, n2 = N - n1 therefore includes any *other* focus population
    sharing the stratum. That is the published comparator, quirks included.
    """
    out = []
    for si in range(len(strat_names)):
        m = (strat_codes == si) & KEEP_ANY
        N = int(m.sum())
        if N < MIN_CM:
            continue
        idx = np.flatnonzero(m)
        is_focus = state[idx] == focus
        n1 = int(is_focus.sum())
        n2 = N - n1
        if n1 == 0 or n2 < MIN_CM:
            continue
        out.append(dict(si=si, idx=idx, is_focus=is_focus, n1=n1, n2=n2, N=N,
                        donor=donor[idx][0]))
    return out


def statistic(blocks, values):
    """van Elteren weighted AUC and analytic z, from a fixed labelling."""
    num = den = stat = var = 0.0
    for b in blocks:
        r = stats.rankdata(values[b["idx"]])
        W = r[b["is_focus"]].sum()
        n1, n2, N = b["n1"], b["n2"], b["N"]
        w = n1 * n2 / (N + 1)
        num += w * (W - n1 * (n1 + 1) / 2) / (n1 * n2)
        den += w
        stat += (W - n1 * (N + 1) / 2) / (N + 1)
        var += n1 * n2 / (12.0 * (N + 1))
    return num / den, stat / np.sqrt(var), np.sqrt(var)


def sample_sums(r, n1, n_rep, gen):
    """Sums of n1 ranks drawn without replacement from r, n_rep times."""
    N = r.size
    if n1 >= N:
        return np.full(n_rep, r.sum())
    out = np.empty(n_rep)
    batch = max(1, int(2_000_000 // N))
    for lo in range(0, n_rep, batch):
        hi = min(lo + batch, n_rep)
        u = gen.random((hi - lo, N))
        pick = np.argpartition(u, n1 - 1, axis=1)[:, :n1]
        out[lo:hi] = r[pick].sum(axis=1)
    return out


def permute(blocks, values, n_rep, gen, depth_blocked):
    """Null draws of (auc, z) under within-stratum label permutation.

    depth_blocked=True permutes only among cells in the same depth quintile of
    the same stratum, so the focus group keeps its observed depth profile and
    the null is no longer required to believe depth is irrelevant.
    """
    auc_num = np.zeros(n_rep)
    auc_den = 0.0
    z_num = np.zeros(n_rep)
    z_var = 0.0
    for b in blocks:
        r = stats.rankdata(values[b["idx"]])
        n1, n2, N = b["n1"], b["n2"], b["N"]
        if depth_blocked:
            d = depth[b["idx"]]
            # quintiles of this stratum; duplicate edges collapse harmlessly
            q = pd.qcut(d, DEPTH_BLOCKS, labels=False, duplicates="drop")
            W = np.zeros(n_rep)
            for g in np.unique(q):
                sel = q == g
                k = int(b["is_focus"][sel].sum())
                if k == 0:
                    continue
                W += sample_sums(r[sel], k, n_rep, gen)
        else:
            W = sample_sums(r, n1, n_rep, gen)
        w = n1 * n2 / (N + 1)
        auc_num += w * (W - n1 * (n1 + 1) / 2) / (n1 * n2)
        auc_den += w
        z_num += (W - n1 * (N + 1) / 2) / (N + 1)
        z_var += n1 * n2 / (12.0 * (N + 1))
    return auc_num / auc_den, z_num / np.sqrt(z_var)


def donor_bootstrap(blocks, values, n_rep, gen):
    """Resample donors with replacement; recompute the weighted AUC."""
    donors = sorted({b["donor"] for b in blocks})
    by_donor = {d: [b for b in blocks if b["donor"] == d] for d in donors}
    pre = {}
    for b in blocks:
        r = stats.rankdata(values[b["idx"]])
        W = r[b["is_focus"]].sum()
        n1, n2, N = b["n1"], b["n2"], b["N"]
        pre[id(b)] = (n1 * n2 / (N + 1), (W - n1 * (n1 + 1) / 2) / (n1 * n2))
    out = np.empty(n_rep)
    for i in range(n_rep):
        pick = gen.choice(donors, size=len(donors), replace=True)
        num = den = 0.0
        for d in pick:
            for b in by_donor[d]:
                w, a = pre[id(b)]
                num += w * a
                den += w
        out[i] = num / den if den > 0 else np.nan
    return out, len(donors)


# ------------------------------------------------------------------ run
rows = []
print(f"n_perm = {N_PERM}   n_boot = {N_BOOT}   depth blocks = {DEPTH_BLOCKS}\n")

for focus in FOCUS:
    blocks = contributing_strata(focus)
    if not blocks:
        print(f"{focus}: no stratum passes the gates")
        continue
    n_cells = sum(b["n1"] for b in blocks)
    n_donors = len({b["donor"] for b in blocks})
    print("=" * 104)
    print(f"{focus}   {n_cells} cells   {len(blocks)} strata   {n_donors} donors")
    print("=" * 104)
    print(f"{'trait':<28} {'AUC':>7} {'z':>8} | {'permSD':>7} {'p_free':>9} "
          f"{'p_depth':>9} | {'donor 95% CI':>20} {'crosses .5':>11}")

    for t in TRAITS:
        v = scores[t]
        auc, z, se = statistic(blocks, v)

        g = np.random.default_rng(abs(hash((SEED, focus, t))) % (2**32))
        _, z_free = permute(blocks, v, N_PERM, g, depth_blocked=False)
        _, z_depth = permute(blocks, v, N_PERM, g, depth_blocked=True)

        p_free = (1 + np.sum(np.abs(z_free) >= abs(z))) / (N_PERM + 1)
        p_depth = (1 + np.sum(np.abs(z_depth) >= abs(z))) / (N_PERM + 1)

        boot, nd = donor_bootstrap(blocks, v, N_BOOT, g)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        crosses = "yes" if lo <= 0.5 <= hi else "no"

        print(f"{t:<28} {auc:>7.3f} {z:>+8.2f} | {z_free.std():>7.3f} "
              f"{p_free:>9.2e} {p_depth:>9.2e} | "
              f"{lo:>8.3f} to {hi:<8.3f} {crosses:>11}")

        rows.append(dict(
            focus=focus, trait=t, n_cells=n_cells, n_strata=len(blocks),
            n_donors=nd, auc=auc, z_analytic=z, se_analytic_stat=se,
            perm_z_mean_free=float(z_free.mean()), perm_z_sd_free=float(z_free.std()),
            perm_z_mean_depth=float(z_depth.mean()), perm_z_sd_depth=float(z_depth.std()),
            p_perm_free=float(p_free), p_perm_depth=float(p_depth),
            boot_lo=float(lo), boot_hi=float(hi),
            boot_median=float(np.nanmedian(boot)),
            boot_crosses_half=bool(lo <= 0.5 <= hi),
        ))
    print()

df = pd.DataFrame(rows)
df.to_csv(OUT_TSV, sep="\t", index=False, float_format="%.6g")

meta = dict(n_perm=N_PERM, n_boot=N_BOOT, depth_blocks=DEPTH_BLOCKS, seed=SEED,
            min_cm=MIN_CM, focus=FOCUS, traits=TRAITS,
            note=("perm_z_sd_free near 1.0 confirms the analytic van Elteren "
                  "variance is the exact randomisation variance; deviation below "
                  "1.0 would indicate ties making the analytic z conservative."))
OUT_JSON.write_text(json.dumps(dict(meta=meta, rows=rows), indent=2), encoding="utf-8")
print(f"wrote {OUT_TSV}")
print(f"wrote {OUT_JSON}")
