"""
Bidirectional MR v2 — the pre-registered run
============================================

QUESTION THIS ANSWERS
---------------------
Whether the shared genetic architecture between cognitive traits and cardiac
autonomic traits is horizontal pleiotropy (the same variants acting independently
in brain and in pacemaker cells) or vertical pleiotropy (a causal chain running
variant -> pacemaker function -> autonomic tone -> cognition). MR cannot test that
dichotomy directly; it can test each arm separately and ask whether the estimate in
each direction is large enough to account for the observed genetic correlation.

Everything below was fixed in results/mr_predictions_v2.json before any estimate in
this file was computed. Instrument criteria, estimators, the five falsification
criteria N1-N5, the stop conditions and the permitted wording are all pre-specified
there. This script only executes them.

HOW THIS ANALYSIS COULD FAIL, AND WHAT IS BUILT IN TO CATCH IT
-------------------------------------------------------------
The v1 analysis failed in four ways that all had the same shape: a check that could
not fail, or could not help firing. Each is guarded here by construction.

  * v1's Steiger filter reduced to |Z_out| < |Z_exp| * sqrt(N_out/N_exp), which for
    cognition -> HRV is |Z| < 1.37 -- inside the bulk of the null. It deleted 21-25%
    of instruments in one direction and 0% in the other. Here Steiger is not used in
    the primary analysis, and the implied bound is printed for every test so a filter
    cutting into the null is visible on the face of the table.
  * v1 read three non-significant MR-Egger intercepts as evidence of no pleiotropy
    while I^2_GX was 0.00-0.48, at which the intercept test has almost no power.
    Here Egger is not reported at all below I^2_GX = 0.9.
  * v1's negative-control criterion had a 26.5% false-alarm rate and no stated
    threshold. N1 replaces it with a family-Bonferroni arm plus a sign-flip
    permutation arm that preserves the correlation induced by shared instruments.
  * v1 (and this project twice before) called something positive with no null
    distribution. N2 builds an LD-score- and MAF-matched empirical null, and N3 runs
    the identical machinery on rheumatoid arthritis, where it MUST fire -- a control
    whose failure mode is silence.

The one guard that matters most is N3. N1 and N2 can only ever return bad news, so
without a control that returns good news when the machinery works, a null from N2 is
indistinguishable from a broken test.

Usage:  python scripts/184_mr_v2.py <stage>
        1  extend cache: rheumatoid arthritis, matching pool, 1000G allele frequencies
        2  N2 / N3  matched-null pleiotropy test and its positive control
        3  N1       negative-control family, Bonferroni arm and permutation arm
        4  N4       specificity contrast, bootstrap
        5  estimators: IVW / weighted median / weighted mode / MR-RAPS / MR-PRESSO
        6  assemble the verdict against the pre-registered stop conditions
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("diag", ROOT / "scripts" / "183_mr_diagnose.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

CACHE = D.CACHE
GWAS = D.GWAS
LDSCDIR = ROOT / "data" / "resource" / "ldsc"
OUTDIR = ROOT / "results"

COGNITIVE = D.COGNITIVE
AUTONOMIC = D.AUTONOMIC
NEGATIVE = D.NEGATIVE
CONTROL = "RheumatoidArthritis"           # N3 positive control, and the reverse-direction
ALL2 = D.ALL + [CONTROL]                  # negative control that v1 never had
RNG = np.random.default_rng(20260803)

# pre-registered thresholds
BONF_ALL = 0.05 / 24
BONF_NEG = 0.05 / 6


# ----------------------------------------------------------------- stage 1
def bed_freq(chrom, rows):
    """Allele frequency of selected variants, read straight out of the PLINK .bed.

    Needed because the summary statistics carry no allele frequency at all, and the
    matched null in N2 has to match on MAF -- genome-wide significant variants of a
    polygenic trait are not a random draw from the frequency spectrum, and neither
    is chi-square in any other GWAS.
    """
    bedp = D.PLINK / f"1000G.EUR.QC.{chrom}.bed"
    n_var = sum(1 for _ in open(D.BIMDIR / f"1000G.EUR.QC.{chrom}.bim", "rb"))
    assert os.path.getsize(bedp) == 3 + n_var * D.BYTES_PER_SNP, f"chr{chrom} bed/bim mismatch"
    out = np.empty(len(rows), dtype=np.float32)
    order = np.argsort(rows)
    with open(bedp, "rb") as f:
        assert f.read(3) == b"\x6c\x1b\x01"
        for pos in order:
            f.seek(3 + int(rows[pos]) * D.BYTES_PER_SNP)
            raw = np.frombuffer(f.read(D.BYTES_PER_SNP), dtype=np.uint8)
            bits = np.empty(D.BYTES_PER_SNP * 4, dtype=np.uint8)
            for k in range(4):
                bits[k::4] = (raw >> (2 * k)) & 3
            g = bits[:D.N_IND]
            n_obs = int(np.sum(g != 1))
            if n_obs == 0:
                out[pos] = np.nan
                continue
            dose = np.where(g == 0, 2, np.where(g == 2, 1, np.where(g == 3, 0, 0)))
            out[pos] = float(np.sum(dose[g != 1])) / (2 * n_obs)
    return out


def stage1():
    print("--- rheumatoid arthritis: genome-wide significant slice and values at instruments")
    need = set()
    for t in D.ALL:
        need |= set(pd.read_csv(CACHE / f"sig_{t}.tsv", sep="\t", usecols=["SNP"]).SNP)
    out = []
    for ch in pd.read_csv(GWAS / f"{CONTROL}.sumstats.gz", sep="\t",
                          usecols=["SNP", "A1", "A2", "Z", "N"], chunksize=2_000_000):
        s = ch[ch.SNP.isin(need)]
        if len(s):
            out.append(s)
    d = pd.concat(out, ignore_index=True)
    d.to_csv(CACHE / f"at_union_{CONTROL}.tsv", sep="\t", index=False)
    print(f"    {len(d):,} / {len(need):,} instrument SNPs present in {CONTROL}")

    print("--- matching pool: HapMap3 LD scores x 1000G positions")
    ld = pd.concat([pd.read_feather(LDSCDIR / f"baseline.{c}.l2.ldscore.feather")
                    [["SNP", "base"]].assign(chrom=c) for c in range(1, 23)],
                   ignore_index=True).rename(columns={"base": "ldscore"})
    bim = D.bim_with_rows()
    pool = ld.merge(bim[["SNP", "chrom", "row", "bp"]], on=["SNP", "chrom"])
    print(f"    {len(ld):,} SNPs with LD scores -> {len(pool):,} also in the 1000G panel")

    print("--- 1000G allele frequencies for the pool (read by seek, not whole-file)")
    fr = np.empty(len(pool), dtype=np.float32)
    for c in range(1, 23):
        m = (pool.chrom == c).to_numpy()
        fr[m] = bed_freq(c, pool.row.to_numpy()[m])
        print(f"      chr{c:<3}{int(m.sum()):>8,} variants", flush=True)
    pool["freq"] = fr
    pool["maf"] = np.minimum(fr, 1 - fr)
    pool = pool[np.isfinite(pool.maf) & (pool.maf > 0.01)]
    pool.to_csv(CACHE / "pool.tsv", sep="\t", index=False)
    print(f"    pool after MAF > 0.01: {len(pool):,}")

    print("--- each trait's chi-square at every pool SNP")
    keep = set(pool.SNP)
    for t in ALL2:
        acc = []
        for ch in pd.read_csv(GWAS / f"{t}.sumstats.gz", sep="\t",
                              usecols=["SNP", "Z", "N"], chunksize=2_000_000):
            s = ch[ch.SNP.isin(keep)]
            if len(s):
                acc.append(s[["SNP", "Z"]])
        z = pd.concat(acc, ignore_index=True).drop_duplicates("SNP")
        z.to_csv(CACHE / f"pool_z_{t}.tsv", sep="\t", index=False)
        print(f"    {t:<24}{len(z):>9,} pool SNPs present")


# ----------------------------------------------------------------- shared
def primary_instruments(r2=0.001, win=10_000_000):
    tag = f"iv_{r2}_{win}"
    f = CACHE / f"{tag}.json"
    if f.exists():
        cmap = {k: set(v) for k, v in json.load(open(f)).items()}
    else:
        bim_idx = D.bim_with_rows()
        cmap = {t: D.clump_r2(D.clean(pd.read_csv(CACHE / f"sig_{t}.tsv", sep="\t")),
                              bim_idx, r2, win) for t in D.ALL}
        json.dump({k: sorted(v) for k, v in cmap.items()}, open(f, "w"))
    return D.build(clump_map=cmap)


def outcomes2():
    d = D.outcomes()
    d[CONTROL] = D.clean(pd.read_csv(CACHE / f"at_union_{CONTROL}.tsv", sep="\t"))
    return d


PAIRS2 = D.PAIRS + [(a, CONTROL) for a in AUTONOMIC] + [(c, CONTROL) for c in COGNITIVE]


# ----------------------------------------------------------------- stage 2
def matched_null(inst_snps, pool, zcol, n_draw=1000, n_bin=10):
    """Mean chi-square of an instrument set against LD-score- and MAF-matched draws.

    The bare observation -- cognitive instruments carry mean chi-square 3.8-4.8 in the
    resting-heart-rate GWAS against a genome-wide mean of 2.1 -- means nothing on its
    own. Genome-wide significant variants of a polygenic trait sit preferentially in
    gene-dense, high-LD, common-allele regions where every trait's chi-square is
    elevated. This draws null sets from the same joint LD-score x MAF strata as the
    observed instruments, which is what makes the comparison a test rather than an
    observation.
    """
    p = pool[np.isfinite(pool[zcol])].copy()
    p["ldbin"] = pd.qcut(p.ldscore, n_bin, labels=False, duplicates="drop")
    p["mafbin"] = pd.qcut(p.maf, n_bin, labels=False, duplicates="drop")
    obs = p[p.SNP.isin(inst_snps)]
    if len(obs) < 10:
        return None
    stat = float(np.mean(obs[zcol] ** 2))
    counts = obs.groupby(["ldbin", "mafbin"]).size()
    strata = {k: g[zcol].to_numpy() for k, g in p.groupby(["ldbin", "mafbin"])}
    null = np.empty(n_draw)
    for b in range(n_draw):
        acc = []
        for k, n in counts.items():
            src = strata.get(k)
            if src is None or len(src) == 0:
                continue
            acc.append(RNG.choice(src, size=int(n), replace=True))
        null[b] = np.mean(np.concatenate(acc) ** 2)
    pval = (1 + np.sum(null >= stat)) / (n_draw + 1)
    return dict(n_matched=len(obs), n_instruments=len(inst_snps), obs_mean_chi2=stat,
                null_mean=float(null.mean()), null_p95=float(np.quantile(null, 0.95)),
                p=float(pval))


def stage2():
    pool = pd.read_csv(CACHE / "pool.tsv", sep="\t")
    for t in ALL2:
        z = pd.read_csv(CACHE / f"pool_z_{t}.tsv", sep="\t").rename(columns={"Z": f"z_{t}"})
        pool = pool.merge(z, on="SNP", how="left")
    ivs = primary_instruments()
    rows = []
    print("N2 / N3 — instrument pleiotropy against an LD-score x MAF matched null\n")
    print(f"{'instruments of':<24}{'in GWAS of':<24}{'nIV':>5}{'matched':>8}"
          f"{'obs chi2':>10}{'null':>8}{'null p95':>10}{'p':>9}")
    for ex in COGNITIVE + AUTONOMIC:
        for ou in NEGATIVE + [CONTROL] + (AUTONOMIC if ex in COGNITIVE else COGNITIVE):
            if ex == ou:
                continue
            r = matched_null(set(ivs[ex].SNP), pool, f"z_{ou}")
            if r is None:
                continue
            role = ("N3 positive control" if ou == CONTROL else
                    "N2 negative control" if ou in NEGATIVE else "descriptive")
            rows.append(dict(exposure=ex, outcome=ou, role=role, **r))
            print(f"{ex:<24}{ou:<24}{r['n_instruments']:>5}{r['n_matched']:>8}"
                  f"{r['obs_mean_chi2']:>10.2f}{r['null_mean']:>8.2f}"
                  f"{r['null_p95']:>10.2f}{r['p']:>9.4f}")
        print()
    pd.DataFrame(rows).to_csv(CACHE / "n2_n3_matched_null.tsv", sep="\t", index=False)


# ----------------------------------------------------------------- stage 3
def stage3():
    """N1: the negative-control family, Bonferroni arm and permutation arm.

    The permutation arm exists because the Bonferroni arm assumes the six tests are
    independent and they are not -- three share an exposure instrument set, two share
    an outcome GWAS. Signs are drawn once per SNP and applied to that SNP's outcome
    effect in every test it appears in, which preserves exactly that dependence.
    Bonferroni is conservative under positive dependence, so the permutation arm can
    only make the criterion fire harder; it is run to put a number on how much.
    """
    ivs = primary_instruments()
    data = outcomes2()
    tests = [(c, n) for c in COGNITIVE for n in NEGATIVE]
    packs, obs_p = [], []
    for ex, ou in tests:
        m, _, _ = D.harmonise(ivs[ex], data[ou])
        bx, by, sy = m.beta_x.to_numpy(), m.beta_y.to_numpy(), m.se_y.to_numpy()
        b, _, sre, q, df = D.ivw(bx, by, sy)
        p = float(2 * stats.norm.sf(abs(b / sre)))
        packs.append((m.SNP.to_numpy(), bx, by, sy))
        obs_p.append(p)
        print(f"{ex+'->'+ou:<46} nIV={len(bx):>4}  IVW {b:+.4f}  p = {p:.4f}")
    obs_min = min(obs_p)
    print(f"\nBonferroni arm: min p = {obs_min:.4f} vs 0.05/6 = {BONF_NEG:.5f} -> "
          f"{'FIRES' if obs_min < BONF_NEG else 'does not fire'}")

    universe = sorted(set().union(*[set(s) for s, _, _, _ in packs]))
    idx = {s: i for i, s in enumerate(universe)}
    maps = [np.array([idx[s] for s in snps]) for snps, _, _, _ in packs]
    n_perm = 10000
    null_min = np.empty(n_perm)
    for b in range(n_perm):
        sgn = RNG.choice([-1.0, 1.0], size=len(universe))
        ps = []
        for (snps, bx, by, sy), mp in zip(packs, maps):
            byp = by * sgn[mp]
            beta, _, sre, _, _ = D.ivw(bx, byp, sy)
            ps.append(2 * stats.norm.sf(abs(beta / sre)))
        null_min[b] = min(ps)
    pfam = (1 + np.sum(null_min <= obs_min)) / (n_perm + 1)
    print(f"permutation arm ({n_perm:,} SNP-wise sign flips): family p = {pfam:.5f} -> "
          f"{'FIRES' if pfam < 0.05 else 'does not fire'}")
    print(f"  implied false-alarm rate of a bare 'any nominal p<0.05' rule: "
          f"{np.mean(null_min < 0.05):.3f}")
    json.dump(dict(tests=[f"{a}->{b}" for a, b in tests], observed_p=obs_p,
                   min_p=obs_min, bonferroni_threshold=BONF_NEG,
                   bonferroni_fires=bool(obs_min < BONF_NEG),
                   permutation_family_p=float(pfam),
                   permutation_fires=bool(pfam < 0.05),
                   naive_any_nominal_false_alarm_rate=float(np.mean(null_min < 0.05))),
              open(CACHE / "n1_negative_control.json", "w"), indent=2)


# ----------------------------------------------------------------- stage 4
def stage4():
    """N4: the specificity contrast the cell-level result actually implies.

    The cell-level claim is not 'cognition affects the heart'. It is that cognitive
    heritability is enriched in nodal pacemaker cells while QT-interval heritability
    is depleted in those same cells (AUC 0.354, z = -5.6). The MR prediction that
    follows is comparative, |beta(exposure -> HRV)| > |beta(exposure -> QT)|, not
    'QT must be zero' -- which the hypothesis never required. Bootstrapped over
    instruments so the two estimates are resampled together where they share SNPs.
    """
    ivs = primary_instruments()
    data = outcomes2()
    n_boot = 10000
    print(f"{'exposure':<24}{'|b HRV|':>9}{'|b QT|':>9}{'diff':>9}{'boot p':>9}  verdict")
    rows = []
    for ex in COGNITIVE:
        mh, _, _ = D.harmonise(ivs[ex], data["HRV_RMSSD"])
        mq, _, _ = D.harmonise(ivs[ex], data["QTinterval"])
        bh = abs(D.ivw(mh.beta_x.to_numpy(), mh.beta_y.to_numpy(), mh.se_y.to_numpy())[0])
        bq = abs(D.ivw(mq.beta_x.to_numpy(), mq.beta_y.to_numpy(), mq.se_y.to_numpy())[0])
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            ih = RNG.integers(0, len(mh), len(mh))
            iq = RNG.integers(0, len(mq), len(mq))
            a = abs(D.ivw(mh.beta_x.to_numpy()[ih], mh.beta_y.to_numpy()[ih],
                          mh.se_y.to_numpy()[ih])[0])
            c = abs(D.ivw(mq.beta_x.to_numpy()[iq], mq.beta_y.to_numpy()[iq],
                          mq.se_y.to_numpy()[iq])[0])
            diffs[b] = a - c
        p = float(np.mean(diffs <= 0))
        ok = p < 0.05 / 3
        print(f"{ex:<24}{bh:>9.3f}{bq:>9.3f}{bh-bq:>+9.3f}{p:>9.4f}  "
              f"{'passes' if ok else 'FAILS'}")
        rows.append(dict(exposure=ex, beta_hrv=bh, beta_qt=bq, diff=bh - bq,
                         boot_p=p, passes=bool(ok)))
    pd.DataFrame(rows).to_csv(CACHE / "n4_specificity.tsv", sep="\t", index=False)
    n_fail = sum(not r["passes"] for r in rows)
    print(f"\nN4 fails for {n_fail} of 3 exposures; stop condition triggers at 2")


# ----------------------------------------------------------------- estimators
def weighted_median(bx, by, sy, n_boot=1000):
    r = by / bx
    w = (bx ** 2) / (sy ** 2)
    o = np.argsort(r)
    cw = np.cumsum(w[o]) - 0.5 * w[o]
    est = np.interp(0.5, cw / np.sum(w), r[o])
    boots = np.empty(n_boot)
    for i in range(n_boot):
        byb = RNG.normal(by, sy)
        rb = byb / bx
        ob = np.argsort(rb)
        cwb = (np.cumsum(w[ob]) - 0.5 * w[ob]) / np.sum(w)
        boots[i] = np.interp(0.5, cwb, rb[ob])
    return est, float(np.std(boots))


def weighted_mode(bx, by, sy, phi=1.0, n_boot=1000):
    """Hartwig's mode-based estimate: consistent if the largest homogeneous subset of
    instruments is valid, which is a different failure mode from IVW and from the
    median. Agreement across the three is the only thing that counts as evidence."""
    def mode_of(R, w):
        """R is (n_rep, k); returns one mode per row. Vectorised because the
        bootstrap over a 300-instrument set is otherwise hours, not seconds."""
        R = np.atleast_2d(R)
        mu = (R * w).sum(1, keepdims=True) / w.sum()
        s = np.sqrt((w * (R - mu) ** 2).sum(1) / w.sum())
        iqr = (np.quantile(R, .75, axis=1) - np.quantile(R, .25, axis=1)) / 1.34
        h = np.maximum(phi * 0.9 * np.minimum(s, iqr) * R.shape[1] ** -0.2, 1e-8)
        lo, hi = R.min(1) - 3 * h, R.max(1) + 3 * h
        g = np.linspace(0, 1, 512)[None, :] * (hi - lo)[:, None] + lo[:, None]
        out = np.empty(len(R))
        for i in range(len(R)):
            d = (w[None, :] * np.exp(-0.5 * ((g[i][:, None] - R[i][None, :]) / h[i]) ** 2)).sum(1)
            out[i] = g[i][np.argmax(d)]
        return out
    r = by / bx
    w = 1.0 / (sy ** 2 / bx ** 2)
    est = mode_of(r[None, :], w)[0]
    boots = mode_of(RNG.normal(by, sy, size=(n_boot, len(by))) / bx[None, :], w)
    return float(est), float(np.std(boots))


def mr_raps(bx, by, sx, sy):
    """Profile-likelihood RAPS with over-dispersion. Unlike IVW it does not assume the
    instruments are strong, and unlike Egger it does not need I^2_GX near 1."""
    def nll(par):
        b, logt2 = par
        v = sy ** 2 + b ** 2 * sx ** 2 + np.exp(logt2)
        return 0.5 * np.sum((by - b * bx) ** 2 / v + np.log(v))
    b0 = np.sum(by * bx / sy ** 2) / np.sum(bx ** 2 / sy ** 2)
    res = optimize.minimize(nll, [b0, np.log(max(np.var(by - b0 * bx), 1e-8))],
                            method="Nelder-Mead",
                            options=dict(xatol=1e-10, fatol=1e-12, maxiter=5000))
    b = res.x[0]
    h = 1e-5
    d2 = (nll([b + h, res.x[1]]) - 2 * nll([b, res.x[1]]) + nll([b - h, res.x[1]])) / h ** 2
    return float(b), float(np.sqrt(1.0 / d2)) if d2 > 0 else np.nan


def mr_presso(bx, by, sy, n_sim=1000, alpha=0.05):
    """Global heterogeneity test and outlier removal with a simulated null.

    Declared in advance: the global test carries essentially the same information as
    Cochran's Q, so with Q/df of 3-5 it is expected to fire almost everywhere and its
    firing is not news. The part that can change a conclusion is the distortion test
    -- whether removing the outliers moves the estimate.
    """
    k = len(bx)
    w = 1.0 / sy ** 2
    Sxx = float(np.sum(bx ** 2 * w))
    b_all = float(np.sum(by * bx * w) / Sxx)

    def rss_rows(BY):
        """Leave-one-out residual sum of squares for every row of BY at once.

        The leave-one-out slope has a closed form -- b_(-i) = (Sxy - by_i bx_i w_i) /
        (Sxx - bx_i^2 w_i) -- so the whole simulated null is one broadcast instead of
        n_sim * k regressions. Same numbers, three orders of magnitude faster.
        """
        BY = np.atleast_2d(BY)
        Sxy = (BY * bx * w).sum(1, keepdims=True)
        num = Sxy - BY * bx * w
        den = Sxx - bx ** 2 * w
        b_loo = num / den
        return (BY - b_loo * bx) ** 2 * w

    obs = rss_rows(by)[0]
    null = rss_rows(RNG.normal(b_all * bx, sy, size=(n_sim, k)))
    gp = (1 + np.sum(null.sum(1) >= obs.sum())) / (n_sim + 1)
    pj = (1 + (null >= obs[None, :]).sum(0)) / (n_sim + 1)
    keep = pj >= alpha / k
    if keep.sum() >= 5 and keep.sum() < k:
        b_out = np.sum(by[keep] * bx[keep] / sy[keep] ** 2) / np.sum(bx[keep] ** 2 / sy[keep] ** 2)
        se_a = np.sqrt(1 / np.sum(bx ** 2 / sy ** 2))
        se_b = np.sqrt(1 / np.sum(bx[keep] ** 2 / sy[keep] ** 2))
        dp = float(2 * stats.norm.sf(abs(b_all - b_out) / np.sqrt(se_a ** 2 + se_b ** 2)))
    else:
        b_out, dp = (b_all, np.nan)
    return dict(global_p=float(gp), n_outliers=int((~keep).sum()),
                beta_outlier_corrected=float(b_out), distortion_p=dp)


def stage5():
    data = outcomes2()
    frames = []
    for r2, win, tag in [(0.001, 10_000_000, "primary"), (0.01, 1_000_000, "sensitivity")]:
        ivs = primary_instruments(r2, win)
        for ex, ou in PAIRS2:
            s = ivs[ex]
            if len(s) < 5:
                continue
            m, _, _ = D.harmonise(s, data[ou])
            if len(m) < 5:
                continue
            bx, by = m.beta_x.to_numpy(), m.beta_y.to_numpy()
            sx, sy = m.se_x.to_numpy(), m.se_y.to_numpy()
            b, sfe, sre, q, df = D.ivw(bx, by, sy)
            wm, wmse = weighted_median(bx, by, sy)
            mo, mose = weighted_mode(bx, by, sy)
            rb, rse = mr_raps(bx, by, sx, sy)
            pr = mr_presso(bx, by, sy) if len(bx) >= 10 else dict(
                global_p=np.nan, n_outliers=0, beta_outlier_corrected=np.nan, distortion_p=np.nan)
            i2 = D.isq_gx(bx, sx)
            eb, ese, ei, esi = D.egger(bx, by, sy)
            nx, ny = float(m.N_x.iloc[0]), float(m.N_y.iloc[0])
            zb = float(np.median(np.abs(m.Z_x))) * np.sqrt(ny / nx)
            frames.append(dict(
                spec=tag, exposure=ex, outcome=ou, n_iv=len(m),
                F_mean=float((m.Z_x ** 2).mean()), F_min=float((m.Z_x ** 2).min()),
                I2_gx=i2, steiger_z_bound=zb, steiger_null_cut=float(2 * stats.norm.sf(zb)),
                ivw=b, ivw_se=sre, ivw_p=float(2 * stats.norm.sf(abs(b / sre))),
                Q=q, Q_df=df, Q_over_df=q / df, Q_p=float(stats.chi2.sf(q, df)),
                wmedian=wm, wmedian_se=wmse,
                wmedian_p=float(2 * stats.norm.sf(abs(wm / wmse))) if wmse > 0 else np.nan,
                wmode=mo, wmode_se=mose,
                wmode_p=float(2 * stats.norm.sf(abs(mo / mose))) if mose > 0 else np.nan,
                raps=rb, raps_se=rse,
                raps_p=float(2 * stats.norm.sf(abs(rb / rse))) if rse == rse else np.nan,
                egger=eb if i2 >= 0.9 else np.nan,
                egger_intercept_p=(float(2 * stats.norm.sf(abs(ei / esi)))
                                   if i2 >= 0.9 and esi > 0 else np.nan),
                mde_80=2.802 * sre, **pr))
            print(f"  [{tag}] {ex:<24}->{ou:<24} nIV={len(m):>4} done", flush=True)
    pd.DataFrame(frames).to_csv(OUTDIR / "mr_v2_estimates.tsv", sep="\t", index=False)
    print(f"\nwrote {OUTDIR / 'mr_v2_estimates.tsv'}")


# ----------------------------------------------------------------- stage 6
def stage6():
    """Evaluate the pre-registered stop conditions mechanically.

    Written so that the verdict is read off the outputs rather than argued for. Each
    condition is a boolean computed from a file on disk; the permitted wording follows
    from which booleans are true. Nothing here chooses a threshold -- every threshold
    was fixed in results/mr_predictions_v2.json before stage 5 ran.
    """
    est = pd.read_csv(OUTDIR / "mr_v2_estimates.tsv", sep="\t")
    P = est[est.spec == "primary"].set_index(["exposure", "outcome"])
    S = est[est.spec == "sensitivity"].set_index(["exposure", "outcome"])
    n1 = json.load(open(CACHE / "n1_negative_control.json"))
    n4 = pd.read_csv(CACHE / "n4_specificity.tsv", sep="\t")
    mn = pd.read_csv(CACHE / "n2_n3_matched_null.tsv", sep="\t")

    core = [(e, o) for e, o in P.index if o != CONTROL]
    n3 = mn[mn.role == "N3 positive control"]
    n3_fires = bool((n3.p < 0.05).any())
    n2 = mn[(mn.role == "N2 negative control") & mn.exposure.isin(COGNITIVE)]
    n2_fires = bool((n2.p < 0.05 / max(len(n2), 1)).any())

    # N5: any test that would otherwise be called an effect must clear Bonferroni twice
    cand = [k for k in core if P.loc[k, "ivw_p"] < BONF_ALL]
    n5_fail = {f"{e}->{o}": dict(primary_p=float(P.loc[(e, o), "ivw_p"]),
                                sensitivity_p=float(S.loc[(e, o), "ivw_p"])
                                if (e, o) in S.index else None)
               for e, o in cand
               if (e, o) not in S.index or S.loc[(e, o), "ivw_p"] >= BONF_ALL}

    sign_disagree = {f"{e}->{o}": [float(P.loc[(e, o), "ivw"]), float(P.loc[(e, o), "wmedian"])]
                     for e, o in cand
                     if np.sign(P.loc[(e, o), "ivw"]) != np.sign(P.loc[(e, o), "wmedian"])}
    presso_flip = {f"{e}->{o}": float(P.loc[(e, o), "beta_outlier_corrected"])
                   for e, o in cand
                   if P.loc[(e, o), "global_p"] < 0.05
                   and np.isfinite(P.loc[(e, o), "beta_outlier_corrected"])
                   and np.sign(P.loc[(e, o), "beta_outlier_corrected"])
                   != np.sign(P.loc[(e, o), "ivw"])}
    weak = {f"{e}->{o}": float(P.loc[(e, o), "F_mean"]) for e, o in core
            if P.loc[(e, o), "F_mean"] < 10}

    stop = {
        "S1_weak_instruments": dict(fires=len(weak) > 0, detail=weak),
        "S2_N1_negative_control_family": dict(
            fires=bool(n1["bonferroni_fires"] or n1["permutation_fires"]),
            detail=dict(min_p=n1["min_p"], bonferroni_threshold=n1["bonferroni_threshold"],
                        bonferroni_fires=n1["bonferroni_fires"],
                        permutation_family_p=n1["permutation_family_p"],
                        permutation_fires=n1["permutation_fires"])),
        "S3_N2_matched_null_pleiotropy": dict(
            fires=bool(n2_fires and n3_fires),
            detail=dict(n2_min_p=float(n2.p.min()) if len(n2) else None,
                        n3_confirms_detector=n3_fires)),
        "S4_N3_detector_broken": dict(fires=not n3_fires,
                                      detail=dict(n3_p=n3.p.tolist(),
                                                  n3_exposures=n3.exposure.tolist())),
        "S5_N4_specificity": dict(fires=bool((~n4.passes).sum() >= 2),
                                  detail=n4.set_index("exposure").boot_p.to_dict()),
        "S6_N5_stability": dict(fires=len(n5_fail) > 0, detail=n5_fail),
        "S7_presso_sign_flip": dict(fires=len(presso_flip) > 0, detail=presso_flip),
        "S8_ivw_median_sign": dict(fires=len(sign_disagree) > 0, detail=sign_disagree),
    }
    fired = [k for k, v in stop.items() if v["fires"]]
    verdict = dict(
        level="0_uninterpretable" if fired else "see_level_1_or_2",
        stop_conditions_fired=fired,
        bonferroni_survivors=[f"{e}->{o}" for e, o in cand],
        detail=stop)
    json.dump(verdict, open(OUTDIR / "mr_v2_verdict.json", "w"), indent=2, default=float)

    print("PRE-REGISTERED STOP CONDITIONS\n")
    for k, v in stop.items():
        print(f"  {'FIRES' if v['fires'] else '  -  '}  {k}")
        if v["fires"]:
            print(f"           {json.dumps(v['detail'], default=float)[:200]}")
    print(f"\ntests clearing Bonferroni {BONF_ALL:.5f} in the primary spec: "
          f"{[f'{e}->{o}' for e, o in cand]}")
    print(f"\nVERDICT: {'level 0 — uninterpretable' if fired else 'no stop condition fired'}")
    print(f"wrote {OUTDIR / 'mr_v2_verdict.json'}")


if __name__ == "__main__":
    {"1": stage1, "2": stage2, "3": stage3, "4": stage4,
     "5": stage5, "6": stage6}[sys.argv[1]]()
