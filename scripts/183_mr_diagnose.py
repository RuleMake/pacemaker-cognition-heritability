"""
Diagnosis of the failed bidirectional MR — read-only, no new causal estimates
============================================================================

QUESTION THIS ANSWERS
---------------------
Not "what is the causal effect". Only: *why is `results/mr_bidirectional.json`
uninterpretable, and which of the three suspected causes actually carry the blame?*
The three suspects, in the order the project raised them:

  (a) the pre-declared negative-control criterion fired, but 24 tests at nominal
      alpha = 0.05 make firing nearly unavoidable -> the criterion, not the data,
      may be at fault;
  (b) Q >> df in many tests -> the instrument sets may be invalid;
  (c) distance pruning (one lead SNP per 1 Mb) is not r^2 clumping -> correlated
      instruments may be inflating Q and shrinking standard errors.

HOW THIS DIAGNOSIS COULD FAIL
-----------------------------
Every number below is recomputed from the same summary statistics the original run
used, so a systematic error in those files (wrong N, wrong allele coding, deflated
Z from genomic control) is invisible here and would corrupt diagnosis and original
alike. Two guards are therefore built in: stage 1 prints the raw per-file facts
(row count, N, allele alphabet, Z variance) so a corrupt input shows up before any
statistic is computed, and stage 4 re-runs the whole pipeline under an alternative
pruning rule so that any conclusion that survives only under 1 Mb pruning is
flagged rather than reported.

Two specific ways this script could *look* right and be wrong, guarded explicitly:
  * If Q/df were driven by outcome-GWAS sample size rather than by pleiotropy, then
    the tests with Q/df ~ 1 would be under-powered, not clean. Stage 3 tests this
    directly instead of assuming either way.
  * If the Steiger filter never removed anything, it is not a filter. Stage 2
    counts what it removed and what it removed *for*.

WHY IT IS BUILT THIS WAY
------------------------
Staged, with every stage cached to disk, because reading eight 7-9 M-row files
whole has been killed by the kernel twice in this project. Each stage is
independently re-runnable; nothing is held in memory across stages.

Usage:  python scripts/183_mr_diagnose.py <stage>
        stage 1  raw file facts + cache genome-wide-significant slices
        stage 2  reproduce the original instrument sets; account for every dropped SNP
        stage 3  Q, F, I^2_GX, and whether Q/df tracks pleiotropy or outcome power
        stage 4  pruning sensitivity: 1 Mb / 5 Mb / 10 Mb distance vs true r^2 clumping
"""

import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
GWAS = ROOT / "data" / "gwas_gsmap"
BIMDIR = ROOT / "data" / "resource" / "bim"
CACHE = ROOT / "work" / "mr_diag"
CACHE.mkdir(parents=True, exist_ok=True)

COGNITIVE = ["EducationalAttainment", "Intelligence", "ReactionTime"]
AUTONOMIC = ["HRV_RMSSD", "HRV_SDNN", "RestingHeartRate"]
NEGATIVE = ["AtrialFibrillation", "QTinterval"]
ALL = COGNITIVE + AUTONOMIC + NEGATIVE

PTHRESH = 5.45
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
AMBIG = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}


# ----------------------------------------------------------------- stage 1
def stage1():
    """Raw facts per file, and cache the genome-wide-significant slice of each.

    Prints Z variance because a GWAS that has been genomic-control corrected has
    var(Z) systematically below its LDSC intercept, and the outcome Z scores are
    exactly what the heterogeneity statistic Q is built from. If one trait's Z is
    deflated, its Q is deflated too, and that is not heterogeneity information.
    """
    rows = []
    for t in ALL:
        p = GWAS / f"{t}.sumstats.gz"
        n_rows = 0
        n_sig = 0
        z2_sum = 0.0
        nvals = set()
        alleles = set()
        sig_chunks = []
        for ch in pd.read_csv(p, sep="\t", usecols=["SNP", "A1", "A2", "Z", "N"],
                              chunksize=2_000_000):
            n_rows += len(ch)
            z = ch.Z.to_numpy(dtype=float)
            good = np.isfinite(z)
            z2_sum += float(np.sum(z[good] ** 2))
            nvals.update(pd.unique(ch.N)[:5].tolist())
            alleles.update(pd.unique(ch.A1.astype(str))[:8].tolist())
            s = ch[np.abs(ch.Z) > PTHRESH]
            n_sig += len(s)
            if len(s):
                sig_chunks.append(s)
        sig = pd.concat(sig_chunks, ignore_index=True)
        sig.to_csv(CACHE / f"sig_{t}.tsv", sep="\t", index=False)
        rows.append(dict(trait=t, n_snp=n_rows, n_sig=n_sig,
                         mean_chi2=z2_sum / n_rows,
                         n_distinct_N=len(nvals), N=sorted(nvals)[0],
                         allele_alphabet="".join(sorted(alleles))[:20]))
        print(f"{t:<24} {n_rows:>10,} SNPs  {n_sig:>7,} GWS  "
              f"mean chi2 {z2_sum/n_rows:6.3f}  N={sorted(nvals)[0]:,}  "
              f"distinctN={len(nvals)}  alleles={''.join(sorted(alleles))[:14]}")
    pd.DataFrame(rows).to_csv(CACHE / "file_facts.tsv", sep="\t", index=False)

    # cache every trait's values at the union of all traits' GWS SNPs, so later
    # stages can re-prune without touching the big files again
    need = set()
    for t in ALL:
        need |= set(pd.read_csv(CACHE / f"sig_{t}.tsv", sep="\t", usecols=["SNP"]).SNP)
    print(f"\nunion of genome-wide-significant SNPs across all traits: {len(need):,}")
    for t in ALL:
        p = GWAS / f"{t}.sumstats.gz"
        out = []
        for ch in pd.read_csv(p, sep="\t", usecols=["SNP", "A1", "A2", "Z", "N"],
                              chunksize=2_000_000):
            ch = ch[ch.SNP.isin(need)]
            if len(ch):
                out.append(ch)
        d = pd.concat(out, ignore_index=True)
        d.to_csv(CACHE / f"at_union_{t}.tsv", sep="\t", index=False)
        print(f"  {t:<24} {len(d):>7,} / {len(need):,} union SNPs present "
              f"({100*len(d)/len(need):.1f}%)")


# ----------------------------------------------------------------- helpers
def load_bim():
    f = CACHE / "bim.tsv"
    if f.exists():
        return pd.read_csv(f, sep="\t")
    bims = [pd.read_csv(BIMDIR / f"1000G.EUR.QC.{c}.bim", sep="\t", header=None,
                        usecols=[0, 1, 3], names=["chrom", "SNP", "bp"])
            for c in range(1, 23)]
    bim = pd.concat(bims, ignore_index=True)
    bim.to_csv(f, sep="\t", index=False)
    return bim


def clean(d):
    """Identical to the original script's _clean, so attrition is comparable."""
    d = d[np.isfinite(d.Z) & np.isfinite(d.N)]
    d = d.assign(A1=d.A1.astype(str).str.upper(), A2=d.A2.astype(str).str.upper())
    d = d[d.A1.isin(list("ACGT")) & d.A2.isin(list("ACGT"))]
    d = d[~pd.Series(list(zip(d.A1, d.A2)), index=d.index).isin(AMBIG)]
    return d.assign(beta=d.Z / np.sqrt(d.N), se=1.0 / np.sqrt(d.N))[
        ["SNP", "A1", "A2", "Z", "N", "beta", "se"]]


def prune_distance(sig, bim, window_bp):
    sig = sig.merge(bim, on="SNP").drop_duplicates("SNP")
    if sig.empty:
        return sig
    sig = sig.sort_values("Z", key=np.abs, ascending=False)
    keep, taken = [], {}
    for r in sig.itertuples():
        w = taken.setdefault(r.chrom, [])
        if all(abs(r.bp - b) > window_bp for b in w):
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
    keep = same | flip
    m = m[keep].copy()
    m.loc[flip[keep], ["beta_y", "Z_y"]] *= -1
    return m, int(keep.sum()), int((~keep).sum())


def steiger_mask(m):
    r2x = m.Z_x ** 2 / (m.Z_x ** 2 + m.N_x)
    r2y = m.Z_y ** 2 / (m.Z_y ** 2 + m.N_y)
    return (r2x > r2y).to_numpy()


def ivw(bx, by, sy):
    w = 1.0 / sy ** 2
    b = np.sum(w * bx * by) / np.sum(w * bx ** 2)
    se = np.sqrt(1.0 / np.sum(w * bx ** 2))
    q = np.sum(w * (by - b * bx) ** 2)
    df = max(len(bx) - 1, 1)
    return b, se, se * max(1.0, np.sqrt(q / df)), float(q), df


def egger(bx, by, sy):
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


def isq_gx(bx, sx):
    """I^2_GX: how badly MR-Egger's no-measurement-error assumption is violated.

    Below ~0.9 the Egger slope is attenuated towards the null and its intercept
    towards zero, which means a non-significant Egger intercept is not evidence of
    no pleiotropy. Reported because the original run read three non-significant
    intercepts as reassurance.
    """
    bx = np.abs(bx)
    w = 1.0 / sx ** 2
    mu = np.sum(w * bx) / np.sum(w)
    q = np.sum(w * (bx - mu) ** 2)
    k = len(bx)
    return max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0


PAIRS = ([(c, a) for c in COGNITIVE for a in AUTONOMIC]
         + [(a, c) for c in COGNITIVE for a in AUTONOMIC]
         + [(c, n) for c in COGNITIVE for n in NEGATIVE])


def build(window_bp=1_000_000, clump_map=None):
    """Return instrument sets under a given pruning rule."""
    bim = load_bim()
    ivs = {}
    for t in ALL:
        sig = clean(pd.read_csv(CACHE / f"sig_{t}.tsv", sep="\t"))
        if clump_map is not None:
            keep = clump_map.get(t, set())
            sig = sig.merge(bim, on="SNP").drop_duplicates("SNP")
            ivs[t] = sig[sig.SNP.isin(keep)]
        else:
            ivs[t] = prune_distance(sig, bim, window_bp)
    return ivs


def outcomes():
    return {t: clean(pd.read_csv(CACHE / f"at_union_{t}.tsv", sep="\t")) for t in ALL}


def run_all(ivs, data, tag):
    rows = []
    for ex, ou in PAIRS:
        s = ivs[ex]
        if len(s) < 5:
            continue
        m, n_match, n_mismatch = harmonise(s, data[ou])
        n_in_outcome = len(s.merge(data[ou][["SNP"]], on="SNP"))
        sm = steiger_mask(m)
        m2 = m[sm]
        if len(m2) < 5:
            rows.append(dict(tag=tag, exposure=ex, outcome=ou, n_pruned=len(s),
                             n_in_outcome=n_in_outcome, n_harmonised=n_match,
                             n_allele_mismatch=n_mismatch,
                             n_steiger_dropped=int((~sm).sum()), n_iv=len(m2)))
            continue
        bx = m2.beta_x.to_numpy(); by = m2.beta_y.to_numpy()
        sy = m2.se_y.to_numpy(); sx = m2.se_x.to_numpy()
        b, se_fe, se_re, q, df = ivw(bx, by, sy)
        eb, ese, ei, esi = egger(bx, by, sy)
        f_stats = m2.Z_x.to_numpy() ** 2
        rows.append(dict(
            tag=tag, exposure=ex, outcome=ou,
            n_pruned=len(s), n_in_outcome=n_in_outcome, n_harmonised=n_match,
            n_allele_mismatch=n_mismatch, n_steiger_dropped=int((~sm).sum()),
            n_iv=len(m2),
            N_exposure=int(m2.N_x.iloc[0]), N_outcome=int(m2.N_y.iloc[0]),
            F_mean=float(f_stats.mean()), F_min=float(f_stats.min()),
            I2_gx=isq_gx(bx, sx),
            ivw=b, ivw_se_fixed=se_fe, ivw_se_random=se_re,
            ivw_p=float(2 * stats.norm.sf(abs(b / se_re))),
            Q=q, Q_df=df, Q_over_df=q / df,
            Q_p=float(stats.chi2.sf(q, df)),
            egger_intercept=ei, egger_intercept_p=float(2 * stats.norm.sf(abs(ei / esi))),
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- stage 2
def stage2():
    """Reproduce the original run and account for every instrument that vanished.

    The original log shows educational attainment pruning to 128 instruments but
    only 70 surviving into the resting-heart-rate test. A 45% loss has to be
    attributed to something: absent from the outcome file, allele mismatch, or
    Steiger. Which one it is changes the reading completely -- coverage loss is
    benign, Steiger loss on a large-N outcome is selection against the causal
    signal itself.
    """
    ivs = build(1_000_000)
    print("instruments after 1 Mb distance pruning (original rule):")
    for t in ALL:
        print(f"  {t:<24}{len(ivs[t]):>5}")
    data = outcomes()
    df = run_all(ivs, data, "distance_1Mb")
    df.to_csv(CACHE / "reproduce_1Mb.tsv", sep="\t", index=False)
    cols = ["exposure", "outcome", "n_pruned", "n_in_outcome", "n_harmonised",
            "n_allele_mismatch", "n_steiger_dropped", "n_iv", "ivw",
            "ivw_se_random", "ivw_p", "Q", "Q_df"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\n" + df[cols].to_string(index=False))


# ----------------------------------------------------------------- stage 3
def stage3():
    """Is Q/df measuring pleiotropy, or measuring how well-powered the outcome is?

    If a fixed amount of horizontal pleiotropy exists per instrument, the expected
    excess of Q over df grows in proportion to the outcome GWAS sample size,
    because Q is built from outcome Z scores. Under that model, Q/df ~ 1 in the
    cognition -> HRV tests would mean the HRV GWAS (N = 46,075) is too small to
    see the pleiotropy, not that those instruments are clean. The distinction
    decides whether the paper's cleanest-looking estimate is clean or blind.

    Test: regress (Q/df - 1) on outcome N across all tests sharing an exposure.
    A slope indistinguishable from zero refutes the power explanation.
    """
    df = pd.read_csv(CACHE / "reproduce_1Mb.tsv", sep="\t")
    df = df[df.n_iv >= 5].copy()
    df["excess"] = df.Q_over_df - 1.0
    print("Q/df against outcome sample size, within each exposure:")
    print(f"{'exposure':<24}{'outcome':<22}{'N_out':>12}{'nIV':>5}"
          f"{'Q/df':>8}{'Q_p':>10}{'F_mean':>9}{'I2_gx':>8}")
    for ex in df.exposure.unique():
        s = df[df.exposure == ex].sort_values("N_outcome")
        for r in s.itertuples():
            print(f"{r.exposure:<24}{r.outcome:<22}{r.N_outcome:>12,}{r.n_iv:>5}"
                  f"{r.Q_over_df:>8.2f}{r.Q_p:>10.2e}{r.F_mean:>9.1f}{r.I2_gx:>8.3f}")
        print()

    # pooled test: does excess heterogeneity scale with outcome N?
    x = np.log10(df.N_outcome.to_numpy(dtype=float))
    y = df.excess.to_numpy()
    sl, ic, r, p, se = stats.linregress(x, y)
    print(f"pooled regression of (Q/df - 1) on log10(N_outcome): "
          f"slope {sl:+.3f} (se {se:.3f}), r = {r:+.3f}, p = {p:.2e}, n = {len(x)}")
    print("  a positive slope means Q/df is reporting outcome power, not instrument quality")

    # what effect size would each null test have detected?
    print("\nminimum detectable effect at 80% power, alpha = 0.05 (two-sided):")
    print(f"{'exposure':<24}{'outcome':<22}{'IVW':>9}{'se_RE':>9}{'MDE':>9}")
    for r in df.itertuples():
        mde = 2.802 * r.ivw_se_random
        print(f"{r.exposure:<24}{r.outcome:<22}{r.ivw:>+9.3f}"
              f"{r.ivw_se_random:>9.3f}{mde:>9.3f}")


# ----------------------------------------------------------------- stage 4
def stage4():
    """Distance pruning versus real r^2 clumping: how much of (b) does (c) explain?

    The suspicion is that 1 Mb distance pruning keeps correlated instruments, and
    that this both inflates Q and shrinks standard errors. Both halves are testable
    and they are not obviously both true: perfectly correlated instruments make Q
    *smaller* relative to df, not larger, because their ratio estimates agree by
    construction. This stage measures the sign rather than assuming it.

    Widening the distance window is the cheap probe (no genotypes needed); true
    r^2 clumping against the 1000 Genomes EUR panel is run separately in stage 5
    once the .bed files are unpacked.
    """
    data = outcomes()
    frames = []
    for w, tag in [(1_000_000, "distance_1Mb"),
                   (5_000_000, "distance_5Mb"),
                   (10_000_000, "distance_10Mb")]:
        ivs = build(w)
        print(f"\n--- {tag} ---")
        for t in ALL:
            print(f"  {t:<24}{len(ivs[t]):>5}")
        frames.append(run_all(ivs, data, tag))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(CACHE / "pruning_sensitivity.tsv", sep="\t", index=False)

    piv = out.pivot_table(index=["exposure", "outcome"], columns="tag",
                          values=["n_iv", "Q_over_df", "ivw", "ivw_se_random"])
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\n" + piv.to_string())


# ----------------------------------------------------------------- stage 5
PLINK = ROOT / "data" / "resource" / "plink" / "1000G_EUR_Phase3_plink"
N_IND = 489
BYTES_PER_SNP = (N_IND + 3) // 4


def read_dosages(chrom, idx):
    """Read selected SNP rows out of a PLINK .bed without loading the whole file.

    PLINK 1 .bed is SNP-major, so row i lives at a fixed offset and can be seeked
    to directly. Only a few thousand rows per chromosome are ever needed, which is
    why no genotype file has to be converted or held in memory.

    Guard: the file length must equal 3 + n_variants * ceil(n_samples/4). If the
    .bim used for indexing were a different build or a different variant set, this
    assertion fails instead of silently returning the wrong genotypes -- which
    would produce plausible but meaningless r^2 values.
    """
    bedp = PLINK / f"1000G.EUR.QC.{chrom}.bed"
    bimp = BIMDIR / f"1000G.EUR.QC.{chrom}.bim"
    n_var = sum(1 for _ in open(bimp, "rb"))
    assert os.path.getsize(bedp) == 3 + n_var * BYTES_PER_SNP, (
        f"chr{chrom}: .bed size does not match .bim variant count -- "
        f"the two files are not the same panel")
    out = np.empty((len(idx), N_IND), dtype=np.float32)
    with open(bedp, "rb") as f:
        magic = f.read(3)
        assert magic == b"\x6c\x1b\x01", f"chr{chrom}: not a SNP-major PLINK bed"
        for r, i in enumerate(idx):
            f.seek(3 + i * BYTES_PER_SNP)
            raw = np.frombuffer(f.read(BYTES_PER_SNP), dtype=np.uint8)
            bits = np.empty(BYTES_PER_SNP * 4, dtype=np.uint8)
            for k in range(4):
                bits[k::4] = (raw >> (2 * k)) & 3
            g = bits[:N_IND].astype(np.float32)
            # plink codes: 0 = hom A1, 1 = missing, 2 = het, 3 = hom A2
            d = np.where(g == 0, 2.0, np.where(g == 2, 1.0, np.where(g == 3, 0.0, np.nan)))
            m = np.isnan(d)
            if m.any():
                d[m] = np.nanmean(d) if (~m).any() else 0.0
            out[r] = d
    # standardise so r^2 is a plain dot product
    out -= out.mean(axis=1, keepdims=True)
    sd = out.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return out / sd


def clump_r2(sig, bim_idx, r2_thresh, window_bp):
    """Greedy r^2 clumping, the operation the original run replaced with distance."""
    d = sig.merge(bim_idx, on="SNP").drop_duplicates("SNP")
    keep = []
    for chrom, sub in d.groupby("chrom"):
        sub = sub.sort_values("Z", key=np.abs, ascending=False).reset_index(drop=True)
        G = read_dosages(chrom, sub.row.to_numpy())
        bp = sub.bp.to_numpy()
        alive = np.ones(len(sub), dtype=bool)
        for i in range(len(sub)):
            if not alive[i]:
                continue
            keep.append(sub.SNP.iloc[i])
            near = alive & (np.abs(bp - bp[i]) <= window_bp)
            near[i] = False
            if near.any():
                r = (G[near] @ G[i]) / N_IND
                alive[np.flatnonzero(near)[r ** 2 >= r2_thresh]] = False
            alive[i] = False
    return set(keep)


def bim_with_rows():
    f = CACHE / "bim_rows.tsv"
    if f.exists():
        return pd.read_csv(f, sep="\t")
    parts = []
    for c in range(1, 23):
        b = pd.read_csv(BIMDIR / f"1000G.EUR.QC.{c}.bim", sep="\t", header=None,
                        usecols=[0, 1, 3], names=["chrom", "SNP", "bp"])
        b["row"] = np.arange(len(b))
        parts.append(b)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(f, sep="\t", index=False)
    return out


def stage5():
    """Real r^2 clumping against 1000 Genomes EUR, versus the distance rule used.

    This is the direct test of suspicion (c). It can fail in an informative way:
    if distance pruning had been keeping correlated instruments, clumping should
    *reduce* the instrument count and *widen* standard errors. If instead the
    counts barely move, then correlated instruments were never the problem and
    the heterogeneity has to be explained by something else.

    Caveat recorded rather than hidden: with 489 reference samples the sampling
    floor on an r^2 estimate is about 1/489 = 0.002, so an r^2 < 0.001 threshold
    sits below what this panel can resolve. r^2 < 0.01 within 1 Mb is therefore
    reported alongside it as the threshold the panel can actually support.
    """
    bim_idx = bim_with_rows()
    data = outcomes()
    frames = []
    for r2t, win, tag in [(0.001, 10_000_000, "r2_0.001_10Mb"),
                          (0.01, 1_000_000, "r2_0.01_1Mb")]:
        cmap = {}
        for t in ALL:
            sig = clean(pd.read_csv(CACHE / f"sig_{t}.tsv", sep="\t"))
            cmap[t] = clump_r2(sig, bim_idx, r2t, win)
            print(f"  [{tag}] {t:<24}{len(cmap[t]):>5}")
        ivs = build(clump_map=cmap)
        frames.append(run_all(ivs, data, tag))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(CACHE / "clump_sensitivity.tsv", sep="\t", index=False)

    base = pd.read_csv(CACHE / "reproduce_1Mb.tsv", sep="\t")
    base["tag"] = "distance_1Mb"
    allr = pd.concat([base, out], ignore_index=True)
    piv = allr.pivot_table(index=["exposure", "outcome"], columns="tag",
                           values=["n_iv", "Q_over_df", "ivw", "ivw_se_random", "ivw_p"])
    with pd.option_context("display.width", 260, "display.max_columns", 60):
        print("\n" + piv.to_string())


# ----------------------------------------------------------------- stage 6
def stage6():
    """What does the Steiger filter actually remove, and what does removing it do?

    Steiger keeps an instrument when r2x > r2y. With beta = Z/sqrt(N) that reduces
    to |Z_y| < |Z_x| * sqrt(N_y / N_x). When the exposure GWAS is much larger than
    the outcome GWAS, that bound falls *inside* the null distribution of Z_y, so
    the filter deletes ordinary noise -- and with it any real outcome signal --
    and truncates exactly the tail that the heterogeneity statistic is built from.

    This stage prints the implied |Z_y| bound for every test, the share of a
    standard normal it excludes, and Q with and without the filter. If the filter
    is doing nothing (bound far in the tail) it is not a filter; if it is cutting
    into the bulk, it is not a direction test either. Either way it is not what it
    was reported to be, and this stage says which.
    """
    ivs = build(1_000_000)
    data = outcomes()
    print(f"{'exposure':<24}{'outcome':<22}{'|Zy| bound':>11}{'null cut %':>11}"
          f"{'dropped':>9}{'obs %':>8}{'Q/df keep':>11}{'Q/df all':>10}"
          f"{'IVW keep':>10}{'IVW all':>9}")
    rows = []
    for ex, ou in PAIRS:
        s = ivs[ex]
        if len(s) < 5:
            continue
        m, _, _ = harmonise(s, data[ou])
        if len(m) < 5:
            continue
        sm = steiger_mask(m)
        nx, ny = float(m.N_x.iloc[0]), float(m.N_y.iloc[0])
        zx_med = float(np.median(np.abs(m.Z_x)))
        bound = zx_med * np.sqrt(ny / nx)
        cut = float(2 * stats.norm.sf(bound))
        _, _, se_k, qk, dfk = ivw(m[sm].beta_x.to_numpy(), m[sm].beta_y.to_numpy(),
                                  m[sm].se_y.to_numpy()) if sm.sum() >= 5 else (0, 0, 0, np.nan, 1)
        bk = ivw(m[sm].beta_x.to_numpy(), m[sm].beta_y.to_numpy(),
                 m[sm].se_y.to_numpy())[0] if sm.sum() >= 5 else np.nan
        ba, _, _, qa, dfa = ivw(m.beta_x.to_numpy(), m.beta_y.to_numpy(),
                                m.se_y.to_numpy())
        print(f"{ex:<24}{ou:<22}{bound:>11.2f}{100*cut:>10.1f}%"
              f"{int((~sm).sum()):>9}{100*(~sm).mean():>7.1f}%"
              f"{qk/dfk:>11.2f}{qa/dfa:>10.2f}{bk:>+10.3f}{ba:>+9.3f}")
        rows.append(dict(exposure=ex, outcome=ou, z_bound=bound, null_cut=cut,
                         n_dropped=int((~sm).sum()), frac_dropped=float((~sm).mean()),
                         Q_df_kept=qk / dfk, Q_df_all=qa / dfa, ivw_kept=bk, ivw_all=ba))
    pd.DataFrame(rows).to_csv(CACHE / "steiger_bite.tsv", sep="\t", index=False)


if __name__ == "__main__":
    {"1": stage1, "2": stage2, "3": stage3, "4": stage4,
     "5": stage5, "6": stage6}[sys.argv[1]]()
