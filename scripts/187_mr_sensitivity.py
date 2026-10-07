"""
Sensitivity analyses for the cardiac -> cognition MR null (added 2026-10-01)
===========================================================================

The 2026-10-01 audit (MR-AUDIT-2026-10-01.md, O3-O5) listed four threats to the reading
"genetically predicted HRV has no detectable effect on cognitive traits". None was
examined before. Each is tested here; every one of them could have moved the estimate.

  hr_corrected_hrv   HRV instruments are mostly resting-heart-rate loci, so the HRV and
                     heart-rate nulls are not independent evidence. The heart-rate-
                     corrected HRV GWAS (same study, heart rate regressed out) gives
                     instruments for HRV not shared with heart rate.
  mvmr               Multivariable MR of the cognitive trait on RMSSD and resting heart
                     rate jointly, with Sanderson's conditional F for each exposure, to
                     ask whether RMSSD has an effect conditional on heart rate.
  intersection       Instruments whose lead SNP is absent from the outcome GWAS were
                     dropped, not proxied. Re-clumping the exposure's genome-wide-
                     significant SNPs among those the outcome carries recovers them.
  winners_curse      Instruments were selected and estimated in the same exposure GWAS,
                     which inflates |beta_x| and attenuates the ratio toward zero.
                     Each beta_x is replaced by its conditional MLE given |Z| > 5.45
                     (Ghosh, Zou and Wright 2008).

Writes results/mr_v2_sensitivity.tsv. Run after scripts/184_mr_v2.py stage 5.

Usage:  python scripts/187_mr_sensitivity.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("diag", ROOT / "scripts" / "183_mr_diagnose.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
OUT = ROOT / "results" / "mr_v2_sensitivity.tsv"
COG = D.COGNITIVE
CARD = D.AUTONOMIC
THR = D.PTHRESH
R2, WIN = 0.001, 10_000_000

bim = D.bim_with_rows()
data = D.outcomes()


def ivw_row(m, analysis, ex, ou, **extra):
    bx, by, sy = m.beta_x.to_numpy(), m.beta_y.to_numpy(), m.se_y.to_numpy()
    b, _, se, q, df = D.ivw(bx, by, sy)
    return dict(analysis=analysis, exposure=ex, outcome=ou, n_iv=len(m), ivw=b, ivw_se=se,
                ivw_p=float(2 * stats.norm.sf(abs(b / se))), Q_over_df=q / df, **extra)


def read_at(trait, snps):
    out = []
    for ch in pd.read_csv(D.GWAS / f"{trait}.sumstats.gz", sep="\t",
                          usecols=["SNP", "A1", "A2", "Z", "N"], chunksize=2_000_000):
        s = ch[ch.SNP.isin(snps)]
        if len(s):
            out.append(s)
    return D.clean(pd.concat(out, ignore_index=True).drop_duplicates("SNP"))


rows = []

# ---------------------------------------------------------------- 1. HR-corrected HRV
for ex in ["HRV_RMSSDc", "HRV_SDNNc"]:
    sig = []
    for ch in pd.read_csv(D.GWAS / f"{ex}.sumstats.gz", sep="\t",
                          usecols=["SNP", "A1", "A2", "Z", "N"], chunksize=2_000_000):
        s = ch[np.abs(ch.Z) > THR]
        if len(s):
            sig.append(s)
    sig = D.clean(pd.concat(sig, ignore_index=True).drop_duplicates("SNP"))
    keep = D.clump_r2(sig, bim, R2, WIN)
    inst = sig[sig.SNP.isin(keep)]
    print(f"{ex}: {len(sig)} GWS -> {len(inst)} instruments", flush=True)
    for ou in COG:
        o = read_at(ou, set(inst.SNP))
        m, _, _ = D.harmonise(inst, o)
        if len(m) >= 5:          # registered floor for descriptive reporting
            rows.append(ivw_row(m, "hr_corrected_hrv", ex, ou, F_mean=float((m.Z_x ** 2).mean())))
            print(f"   -> {ou:<24} k={len(m)}  IVW {rows[-1]['ivw']:+.4f} ({rows[-1]['ivw_se']:.4f})", flush=True)

# ---------------------------------------------------------------- 2. MVMR RMSSD + RHR
ivs = D.build(clump_map={t: set(D.clump_r2(D.clean(pd.read_csv(D.CACHE / f"sig_{t}.tsv", sep="\t")),
                                           bim, R2, WIN)) for t in ["HRV_RMSSD", "RestingHeartRate"]})
u = pd.concat([ivs["HRV_RMSSD"], ivs["RestingHeartRate"]]).drop_duplicates("SNP")
u = u.merge(bim[["SNP", "row"]], on="SNP", how="left") if "row" not in u else u
joint = D.clump_r2(u[["SNP", "A1", "A2", "Z", "N", "beta", "se"]], bim, R2, WIN)
base = data["HRV_RMSSD"][data["HRV_RMSSD"].SNP.isin(joint)]
x2, _, _ = D.harmonise(base, data["RestingHeartRate"])
x2 = x2[["SNP", "A1_x", "A2_x", "beta_x", "se_x", "beta_y", "se_y"]].rename(
    columns={"A1_x": "A1", "A2_x": "A2", "beta_x": "b1", "se_x": "s1", "beta_y": "b2", "se_y": "s2"})
for ou in COG:
    m, _, _ = D.harmonise(x2.assign(Z=0.0, N=1.0, beta=x2.b1, se=x2.s1), data[ou])
    if len(m) < 10:
        continue
    b1, b2 = m.b1.to_numpy(), m.b2.to_numpy()
    s1, s2 = m.s1.to_numpy(), m.s2.to_numpy()
    by, sy = m.beta_y.to_numpy(), m.se_y.to_numpy()
    w = 1 / sy ** 2
    X = np.column_stack([b1, b2])
    A = X.T @ (X * w[:, None])
    coef = np.linalg.solve(A, X.T @ (by * w))
    res = by - X @ coef
    sig2 = max(1.0, np.sum(w * res ** 2) / (len(by) - 2))
    se = np.sqrt(np.diag(np.linalg.inv(A)) * sig2)

    def cond_f(bt, bo, st, so):
        """Sanderson, Spiller and Bowden (2021) conditional F for exposure t given o,
        as in the MVMR package's strength_mvmr: delta from the unweighted regression of
        the t effects on the o effects through the origin, then
        Q = sum((bt - delta * bo)^2 / (st^2 + delta^2 so^2)) and F = Q / (L - 1)."""
        delta = float(np.sum(bt * bo) / np.sum(bo ** 2))
        q = np.sum((bt - delta * bo) ** 2 / (st ** 2 + delta ** 2 * so ** 2))
        return float(q / (len(bt) - 1))

    rows.append(dict(analysis="mvmr", exposure="HRV_RMSSD|RestingHeartRate", outcome=ou,
                     n_iv=len(m), ivw=float(coef[0]), ivw_se=float(se[0]),
                     ivw_p=float(2 * stats.norm.sf(abs(coef[0] / se[0]))),
                     beta_rhr=float(coef[1]), beta_rhr_se=float(se[1]),
                     cond_F_rmssd=cond_f(b1, b2, s1, s2), cond_F_rhr=cond_f(b2, b1, s2, s1)))
    print(f"MVMR -> {ou:<24} k={len(m)}  RMSSD {coef[0]:+.4f} ({se[0]:.4f})  "
          f"condF {rows[-1]['cond_F_rmssd']:.1f}", flush=True)

# ---------------------------------------------------------------- 3. intersection clumping
for ex in CARD:
    sig = D.clean(pd.read_csv(D.CACHE / f"sig_{ex}.tsv", sep="\t"))
    for ou in COG:
        present = sig[sig.SNP.isin(set(data[ou].SNP))]
        mm, _, _ = D.harmonise(present, data[ou])
        cand = present[present.SNP.isin(set(mm.SNP))]
        keep = D.clump_r2(cand, bim, R2, WIN)
        m, _, _ = D.harmonise(cand[cand.SNP.isin(keep)], data[ou])
        rows.append(ivw_row(m, "intersection_clumping", ex, ou))
        print(f"intersection {ex:<18}-> {ou:<24} k={len(m)}  IVW {rows[-1]['ivw']:+.4f} "
              f"({rows[-1]['ivw_se']:.4f})  P {rows[-1]['ivw_p']:.3f}", flush=True)

# ---------------------------------------------------------------- 4. winner's curse
def cond_mle(z, c=THR):
    """Conditional MLE of the mean of z given |z| > c (Ghosh, Zou and Wright 2008)."""
    def nll(mu):
        return -(stats.norm.logpdf(z - mu)
                 - np.log(stats.norm.sf(c - mu) + stats.norm.cdf(-c - mu)))
    lo, hi = (0.0, z) if z > 0 else (z, 0.0)
    return optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded").x


ivp = D.build(clump_map={t: set(D.clump_r2(D.clean(pd.read_csv(D.CACHE / f"sig_{t}.tsv", sep="\t")),
                                           bim, R2, WIN)) for t in CARD})
for ex in CARD:
    for ou in COG:
        m, _, _ = D.harmonise(ivp[ex], data[ou])
        zc = np.array([cond_mle(z) for z in m.Z_x.to_numpy()])
        m2 = m.assign(beta_x=zc / np.sqrt(m.N_x.to_numpy()))
        naive = D.ivw(m.beta_x.to_numpy(), m.beta_y.to_numpy(), m.se_y.to_numpy())[0]
        r = ivw_row(m2, "winners_curse_corrected", ex, ou,
                    mean_shrink_bx=float(np.mean(zc / m.Z_x.to_numpy())),
                    F_mean_corrected=float(np.mean(zc ** 2)), ivw_naive=float(naive))
        rows.append(r)
        print(f"winner's curse {ex:<18}-> {ou:<24} |bx| x{r['mean_shrink_bx']:.2f}  "
              f"IVW {r['ivw']:+.4f} ({r['ivw_se']:.4f}) vs naive {naive:+.4f}", flush=True)

pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
print(f"\nwrote {OUT}")
