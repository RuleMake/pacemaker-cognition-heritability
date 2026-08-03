"""
Bidirectional Mendelian randomisation: which way does the arrow point?
======================================================================

Genetic correlation says educational attainment and heart rate variability share
architecture (rg = +0.26, p = 1.2e-5) and that the sharing is specific — it is absent
for atrial fibrillation and QT interval. It cannot say whether autonomic tone
influences cognition, cognition influences autonomic tone, or a third thing drives
both. Only an instrumented analysis can separate those, and it can only do it in one
direction at a time, which is why it is run both ways.

Honest scope, stated first
--------------------------
This is not a definitive causal analysis and should not be written up as one.

  * Instruments are pruned by DISTANCE (one lead SNP per 1 Mb), not by r^2 against a
    reference panel. Distance pruning keeps some correlated instruments, which
    narrows the confidence interval more than it should.
  * Effect sizes are reconstructed as beta = Z / sqrt(N), the standard approximation
    for standardised continuous traits. Every estimate is therefore in SD per SD.
  * Educational attainment, intelligence and reaction time are heavily UK Biobank;
    the HRV study is a separate 46,075-person cohort, but overlap with resting heart
    rate (UK Biobank) is likely and sample overlap biases MR towards the observational
    association.
  * Horizontal pleiotropy is the standing threat for any cognitive exposure, since
    education instruments affect smoking, BMI, blood pressure and much else — all of
    which change heart rate. MR-Egger and the weighted median are reported precisely
    because they fail differently from IVW.

Three estimators, agreement between them is the evidence:
  IVW              efficient, but assumes no directional pleiotropy
  MR-Egger         allows directional pleiotropy; its intercept tests for it
  weighted median  valid if under half the instrument weight is invalid

Steiger filtering removes instruments that explain more variance in the outcome than
in the exposure, which is the most common way a bidirectional MR gets its own
direction backwards.

Usage:  python scripts/181_mr_bidirectional.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.as_posix()
GWAS = f"{ROOT}/data/gwas_gsmap"
OUT = f"{ROOT}/results/mr_bidirectional.json"

COGNITIVE = ["EducationalAttainment", "Intelligence", "ReactionTime"]
AUTONOMIC = ["HRV_RMSSD", "HRV_SDNN", "RestingHeartRate"]
NEGATIVE = ["AtrialFibrillation", "QTinterval"]   # should show nothing either way

PTHRESH = 5.45          # |Z| for p < 5e-8
CLUMP_BP = 1_000_000
MIN_IV = 5
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _clean(d):
    d = d[np.isfinite(d.Z) & np.isfinite(d.N)]
    d = d.assign(A1=d.A1.astype(str).str.upper(), A2=d.A2.astype(str).str.upper())
    d = d[d.A1.isin(list("ACGT")) & d.A2.isin(list("ACGT"))]
    amb = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}
    d = d[~pd.Series(list(zip(d.A1, d.A2)), index=d.index).isin(amb)]
    # standardised-trait approximation; everything downstream is SD per SD
    return d.assign(beta=d.Z / np.sqrt(d.N), se=1.0 / np.sqrt(d.N))[
        ["SNP", "A1", "A2", "Z", "N", "beta", "se"]]


def load(t, keep_snps=None, sig_only=False):
    """Read one trait, keeping only what this analysis can possibly use.

    Reading eight files of 7-9 million rows whole costs several gigabytes and was
    killed by the kernel with no traceback — the same mistake made twice already in
    this project, once with the raw GWAS files and once in the LDSC script. MR needs
    only two slices: the genome-wide significant SNPs of a trait when it acts as the
    exposure, and that trait's values at other traits' instruments when it acts as the
    outcome. Both are tiny, so both are filtered on the way in.
    """
    p = f"{GWAS}/{t}.sumstats.gz"
    if not os.path.exists(p):
        return None
    out = []
    for ch in pd.read_csv(p, sep="\t", usecols=["SNP", "A1", "A2", "Z", "N"],
                          chunksize=1_000_000):
        if sig_only:
            ch = ch[np.abs(ch.Z) > PTHRESH]
        if keep_snps is not None:
            ch = ch[ch.SNP.isin(keep_snps)]
        if len(ch):
            out.append(ch)
    if not out:
        return None
    return _clean(pd.concat(out, ignore_index=True))


def instruments(d, bim):
    """Genome-wide significant lead SNPs, one per 1 Mb window."""
    sig = d[np.abs(d.Z) > PTHRESH].merge(bim, on="SNP")
    if sig.empty:
        return sig
    sig = sig.sort_values("Z", key=np.abs, ascending=False)
    keep, taken = [], {}
    for r in sig.itertuples():
        w = taken.setdefault(r.chrom, [])
        if all(abs(r.bp - b) > CLUMP_BP for b in w):
            w.append(r.bp)
            keep.append(r.SNP)
    return sig[sig.SNP.isin(keep)]


def harmonise(expo, outc):
    m = expo.merge(outc, on="SNP", suffixes=("_x", "_y"))
    a1, a2 = m.A1_x.to_numpy(), m.A2_x.to_numpy()
    b1, b2 = m.A1_y.to_numpy(), m.A2_y.to_numpy()
    c1 = np.array([COMP.get(v, "N") for v in b1])
    c2 = np.array([COMP.get(v, "N") for v in b2])
    same = ((a1 == b1) & (a2 == b2)) | ((a1 == c1) & (a2 == c2))
    flip = ((a1 == b2) & (a2 == b1)) | ((a1 == c2) & (a2 == c1))
    m = m[same | flip].copy()
    m.loc[flip[same | flip], ["beta_y", "Z_y"]] *= -1
    return m


def steiger(m):
    """Keep instruments that explain more variance in the exposure than the outcome."""
    r2x = m.Z_x ** 2 / (m.Z_x ** 2 + m.N_x)
    r2y = m.Z_y ** 2 / (m.Z_y ** 2 + m.N_y)
    return m[r2x > r2y]


def ivw(bx, by, sy):
    w = 1.0 / sy ** 2
    b = np.sum(w * bx * by) / np.sum(w * bx ** 2)
    se = np.sqrt(1.0 / np.sum(w * bx ** 2))
    # inflate by the residual scatter, as the standard random-effects IVW does
    q = np.sum(w * (by - b * bx) ** 2)
    df = max(len(bx) - 1, 1)
    se *= max(1.0, np.sqrt(q / df))
    return b, se, float(q), df


def egger(bx, by, sy):
    # orient all exposure effects positive, which is what makes the intercept
    # interpretable as directional pleiotropy
    s = np.sign(bx)
    bx, by = bx * s, by * s
    w = 1.0 / sy ** 2
    X = np.column_stack([np.ones(len(bx)), bx])
    A = X.T @ (X * w[:, None])
    coef = np.linalg.solve(A, X.T @ (by * w))
    resid = by - X @ coef
    dof = max(len(bx) - 2, 1)
    cov = np.linalg.inv(A) * (np.sum(w * resid ** 2) / dof)
    return coef[1], np.sqrt(cov[1, 1]), coef[0], np.sqrt(cov[0, 0])


def wmedian(bx, by, sy, n_boot=1000, seed=7):
    r = by / bx
    w = (bx ** 2) / (sy ** 2)
    o = np.argsort(r)
    r, w = r[o], w[o]
    cw = np.cumsum(w) - 0.5 * w
    cw /= np.sum(w)
    est = np.interp(0.5, cw, r)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        byb = rng.normal(by, sy)
        rb = byb / bx
        ob = np.argsort(rb)
        rb2, wb = rb[ob], ((bx ** 2) / (sy ** 2))[ob]
        cwb = (np.cumsum(wb) - 0.5 * wb) / np.sum(wb)
        boots.append(np.interp(0.5, cwb, rb2))
    return est, float(np.std(boots))


# ------------------------------------------------------------------ run
REFDIR = os.path.expanduser(
    "~/cardio/resource/gsMap_resource/LD_Reference_Panel/1000G_EUR_Phase3_plink")
bims = []
for ch in range(1, 23):
    p = f"{REFDIR}/1000G.EUR.QC.{ch}.bim"
    if os.path.exists(p):
        bims.append(pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 3],
                                names=["chrom", "SNP", "bp"]))
bim = pd.concat(bims, ignore_index=True)
print(f"reference positions: {len(bim):,} SNPs")

# pass 1 — instruments only, from every trait that can act as an exposure
ALL = COGNITIVE + AUTONOMIC + NEGATIVE
IVS = {}
for t in ALL:
    d = load(t, sig_only=True)
    if d is None:
        print(f"  !! {t}: not fetched, or no genome-wide significant SNP")
        continue
    IVS[t] = instruments(d, bim)
    print(f"  {t:<26}{len(d):>7,} genome-wide significant   "
          f"{len(IVS[t]):>4} independent after 1 Mb pruning")

# pass 2 — every trait's values, but only at the SNPs some trait uses as instruments
need = set()
for s in IVS.values():
    need |= set(s.SNP)
print(f"\nlooking up {len(need):,} instrument SNPs in each outcome")
data = {}
for t in ALL:
    d = load(t, keep_snps=need)
    if d is not None:
        data[t] = d

print("\n" + "=" * 108)
print("BIDIRECTIONAL MR — all estimates in SD of outcome per SD of exposure")
print("=" * 108)
print(f"{'exposure':<24}{'outcome':<22}{'nIV':>5}{'IVW':>19}{'Egger':>19}"
      f"{'median':>17}{'Egger icept p':>15}")
res = {}
pairs = ([(c, a) for c in COGNITIVE for a in AUTONOMIC]
         + [(a, c) for c in COGNITIVE for a in AUTONOMIC]
         + [(c, n) for c in COGNITIVE for n in NEGATIVE])
for ex, ou in pairs:
    if ex not in IVS or ou not in data or len(IVS[ex]) < MIN_IV:
        continue
    m = harmonise(IVS[ex], data[ou])
    m = steiger(m)
    if len(m) < MIN_IV:
        print(f"{ex:<24}{ou:<22}{len(m):>5}   too few instruments after Steiger")
        continue
    bx, by, sy = m.beta_x.to_numpy(), m.beta_y.to_numpy(), m.se_y.to_numpy()
    b1, s1, q, df = ivw(bx, by, sy)
    b2, s2, i0, si = egger(bx, by, sy)
    b3, s3 = wmedian(bx, by, sy)
    pi = float(2 * stats.norm.sf(abs(i0 / si))) if si > 0 else np.nan
    p1 = float(2 * stats.norm.sf(abs(b1 / s1)))
    res[f"{ex}->{ou}"] = dict(n_iv=len(m), ivw=b1, ivw_se=s1, ivw_p=p1,
                              egger=b2, egger_se=s2, egger_intercept=i0,
                              egger_intercept_p=pi, wmedian=b3, wmedian_se=s3,
                              Q=q, Q_df=df)
    star = " *" if p1 < 0.05 else ""
    print(f"{ex:<24}{ou:<22}{len(m):>5}"
          f"{f'{b1:+.3f}({s1:.3f})':>19}{f'{b2:+.3f}({s2:.3f})':>19}"
          f"{f'{b3:+.3f}':>17}{pi:>15.3f}{star}")

print("\n" + "=" * 108)
print("READING")
print("=" * 108)
print("""  An effect is worth mentioning only when IVW, Egger and the weighted median agree
  in sign and rough size AND the Egger intercept is not significant. A significant
  intercept means directional pleiotropy is present and the IVW estimate is not a
  causal effect. Negative-control outcomes (atrial fibrillation, QT interval) should
  show nothing; if a cognitive exposure "causes" QT interval, the instruments are
  pleiotropic and none of the cardiac results can be trusted either.""")

with open(OUT, "w") as f:
    json.dump(res, f, indent=2, default=float)
print(f"\nwrote {OUT}")
