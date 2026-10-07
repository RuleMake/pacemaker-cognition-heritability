"""
Cross-trait LD score regression: is the pacemaker-cognition overlap population-level too?
=========================================================================================

Educational attainment enriches in sinoatrial pacemaker cells and nowhere else in the
conduction system. Three explanations have been killed — generic polygenicity
(rheumatoid arthritis is flat), the heart's own neural programme (refuted), gene length
(refuted; resting heart rate is more length-biased than education). What remains is
that the overlap is real.

A cell-level observation deserves a population-level counterpart. If sinoatrial
pacemaker cells genuinely carry heritability for both a cardiac and a cognitive trait,
the two traits should share genetic architecture genome-wide, and that is measurable
directly from the summary statistics already on disk.

LDSC is not installed and its released version needs Python 2, so the estimator is
implemented here. It is short and it is standard:

    E[z1j z2j] = (sqrt(N1 N2) rho_g / M) * l_j  +  intercept
    E[z^2_j]   = (N h2 / M) * l_j               +  intercept

with regression weights that down-weight high-LD SNPs and a block jackknife for the
standard errors. Sample overlap inflates the intercept but leaves the slope unbiased,
which is the whole reason to use LD score regression rather than correlating effects.

A re-implementation has to earn trust before its answer counts, so two internal checks
run first and are printed before anything else:

    POSITIVE  rg(HRV_RMSSD, HRV_SDNN) — same study, same people, correlated phenotypes.
              Must come back near 1. If it does not, the code is wrong.
    NEGATIVE  rg(EducationalAttainment, RheumatoidArthritis) — no expected relation.
              Must come back near 0.

Heritabilities are printed alongside so they can be checked against published values.

Usage:  python scripts/170_ldsc_rg.py
"""

import gzip
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.as_posix()
GWAS = f"{ROOT}/data/gwas_gsmap"
# the HapMap3 weights ship with gsMap's LDSC resource; a copy lives in the project so the
# script also runs from Windows, where the WSL home directory is not visible
LD = f"{ROOT}/data/resource/ldsc_weights_hm3_no_hla"
if not os.path.exists(f"{LD}/weights.1.l2.ldscore.gz"):
    LD = os.path.expanduser(
        "~/cardio/resource/gsMap_resource/LDSC_resource/weights_hm3_no_hla")
OUT = f"{ROOT}/results/ldsc_rg.json"
# per-block delete values for the cognition x autonomic pairs, read by
# scripts/186_mr_rg_consistency.py to propagate rg and h2 uncertainty jointly
OUT_JK = f"{ROOT}/results/ldsc_rg_jackknife.json"

COG = ["EducationalAttainment", "Intelligence", "ReactionTime"]
PAIRS = [
    ("HRV_RMSSD", "HRV_SDNN", "same-study control — cannot fail, see below"),
    ("RestingHeartRate", "HRV_RMSSD", "QUANTITATIVE control, published -0.74 to -0.55"),
]
# One trait was never enough. If the shape "strong with HRV, absent with myocyte
# traits" is a property of cognition rather than of educational attainment in
# particular, it has to repeat in intelligence and reaction time.
for c in COG:
    PAIRS += [
        (c, "HRV_RMSSD", "THE QUESTION"),
        (c, "HRV_SDNN", "THE QUESTION"),
        (c, "RestingHeartRate", "pacemaker output"),
        (c, "PRinterval", "conduction, not pacemaker"),
        (c, "AtrialFibrillation", "myocyte disease — expect ~0"),
        (c, "QTinterval", "myocyte repolarisation — expect ~0"),
    ]
PAIRS.append(("EducationalAttainment", "RheumatoidArthritis",
              "reference only — education is pleiotropic, not a valid null"))
# heart-rate-corrected HRV, for the MR sensitivity analysis on HRV not shared with heart rate
PAIRS.append(("EducationalAttainment", "HRV_RMSSDc", "MR sensitivity: HR-corrected HRV"))
# The three cognitive traits are NOT independent replicates, and the "shape test"
# reported earlier — all three rank both HRV indices above the four other cardiac
# traits, p = (1/15)^3 — silently assumed they were. Measuring rg among them is what
# turns that number from wrong into defensible.
for i, c1 in enumerate(COG):
    for c2 in COG[i + 1:]:
        PAIRS.append((c1, c2, "how independent are the cognitive traits?"))
N_BLOCK = 200

# ------------------------------------------------------------------ LD scores
parts = []
for ch in range(1, 23):
    p = f"{LD}/weights.{ch}.l2.ldscore.gz"
    if os.path.exists(p):
        parts.append(pd.read_csv(p, sep="\t", compression="gzip"))
ld = pd.concat(parts, ignore_index=True)
ld = ld[["CHR", "SNP", "BP", "MAF", "L2"]].dropna()
# M_5_50: LDSC's denominator is the count of common SNPs in the reference, not the
# count that happen to survive merging with a particular GWAS
M = int((ld.MAF > 0.05).sum())
print(f"LD scores: {len(ld):,} SNPs on {ld.CHR.nunique()} chromosomes, M_5_50 = {M:,}")

# Read each file in chunks and keep only HapMap3 SNPs on the way in. Reading eight
# 10-million-row files whole costs several gigabytes and, run alongside scDRS, was
# killed by the kernel with no traceback — the same mistake this project already made
# once with the raw GWAS files. Restricting to the ~1.2M reference SNPs first cuts
# each trait tenfold and it is the set LDSC uses anyway.
HM3 = set(ld.SNP)
sumstats = {}
need = {t for a, b, _ in PAIRS for t in (a, b)}
for t in sorted(need):
    p = f"{GWAS}/{t}.sumstats.gz"
    if not os.path.exists(p):
        print(f"  !! {t}: no sumstats")
        continue
    keep = []
    for ch in pd.read_csv(p, sep="\t", usecols=["SNP", "A1", "A2", "Z", "N"],
                          chunksize=2_000_000):
        keep.append(ch[ch.SNP.isin(HM3)])
    d = pd.concat(keep, ignore_index=True)
    del keep
    d = d[np.isfinite(d.Z) & np.isfinite(d.N)]
    # standard LDSC filters
    d = d[d.N >= 0.67 * d.N.max()]
    d = d[d.Z ** 2 < max(80.0, 0.001 * d.N.max())]
    d["A1"] = d.A1.astype(str).str.upper()
    d["A2"] = d.A2.astype(str).str.upper()
    d = d[d.A1.isin(list("ACGT")) & d.A2.isin(list("ACGT"))]
    # Strand-ambiguous SNPs cannot be aligned between studies without allele
    # frequencies, and guessing wrong flips the sign of Z. LDSC drops them; so do we.
    amb = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}
    d = d[~pd.Series(list(zip(d.A1, d.A2)), index=d.index).isin(amb)]
    d["Z"] = d.Z.astype(np.float32)
    d["N"] = d.N.astype(np.float32)
    sumstats[t] = d[["SNP", "A1", "A2", "Z", "N"]]
    print(f"  {t:<26}{len(d):>10,} SNPs after filtering, N = {int(d.N.median()):,}")


def _reg(x, y, w, blocks):
    """Weighted least squares slope + block-jackknife SE."""
    def slope(m):
        X = np.column_stack([np.ones(m.sum()), x[m]])
        W = w[m]
        A = X.T @ (X * W[:, None])
        b = X.T @ (y[m] * W)
        return np.linalg.solve(A, b)

    full = slope(np.ones(len(x), dtype=bool))
    ests = []
    for b in range(blocks.max() + 1):
        m = blocks != b
        if m.sum() > 10:
            ests.append(slope(m))
    ests = np.array(ests)
    n = len(ests)
    se = np.sqrt((n - 1) / n * ((ests - ests.mean(0)) ** 2).sum(0))
    return full, se, ests


COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def merged(t1, t2=None):
    d = sumstats[t1].rename(columns={"Z": "Z1", "N": "N1", "A1": "A1_1", "A2": "A2_1"})
    if t2:
        d = d.merge(sumstats[t2].rename(columns={"Z": "Z2", "N": "N2",
                                                 "A1": "A1_2", "A2": "A2_2"}),
                    on="SNP")
        # THE BUG THIS FIXES: merging on rsID alone assumes both studies coded the
        # effect allele the same way. They routinely do not. Where A1 is swapped, the
        # sign of Z is reversed, and z1*z2 for those SNPs enters the regression with
        # the wrong sign — which drags every rg toward zero. It is why the second
        # control, rg(resting heart rate, HRV), came back -0.12 against a published
        # -0.55 to -0.74, while the first control passed: HRV_RMSSD and HRV_SDNN come
        # from one study and share allele coding, so that control could never have
        # caught it. A positive control that cannot fail is not a control.
        a1, a2 = d.A1_1.to_numpy(), d.A2_1.to_numpy()
        b1, b2 = d.A1_2.to_numpy(), d.A2_2.to_numpy()
        same = (a1 == b1) & (a2 == b2)
        flip = (a1 == b2) & (a2 == b1)
        cs = np.array([COMP.get(x, "N") for x in b1])
        cs2 = np.array([COMP.get(x, "N") for x in b2])
        same |= (a1 == cs) & (a2 == cs2)          # other strand, same orientation
        flip |= (a1 == cs2) & (a2 == cs)          # other strand, swapped
        d = d[same | flip].copy()
        d.loc[flip[same | flip], "Z2"] *= -1
        d = d.drop(columns=["A1_1", "A2_1", "A1_2", "A2_2"])
    d = d.merge(ld[["SNP", "L2", "CHR", "BP"]], on="SNP")
    d = d[d.L2 > 0].sort_values(["CHR", "BP"]).reset_index(drop=True)
    # contiguous genomic blocks so the jackknife respects LD
    d["block"] = (np.arange(len(d)) * N_BLOCK // len(d)).astype(int)
    return d


def _h2_fit(z, n, l2, blocks):
    """Univariate LDSC slope (= h2) with LDSC's one reweighting step; returns the full
    fit, its jackknife SE and the per-block delete estimates."""
    x = l2 * n / M
    y = z ** 2
    w = 1.0 / np.maximum(l2, 1.0)
    (_, s0), _, _ = _reg(x, y, w, blocks)
    # one reweighting step, as LDSC does: down-weight SNPs the model expects to be noisy
    w = w / np.maximum(1.0 + max(s0, 0.0) * x, 1e-8) ** 2
    return _reg(x, y, w, blocks)


def h2(t):
    d = merged(t)
    (icept, s), se, _ = _h2_fit(d.Z1.to_numpy(), d.N1.to_numpy(), d.L2.to_numpy(),
                                d.block.to_numpy())
    return dict(h2=float(s), se=float(se[1]), intercept=float(icept), n_snp=len(d))


JK = {}


def rg(t1, t2, h1=None, h2_=None):
    """Genetic correlation with ldsc --rg's estimand and jackknife.

    Both heritabilities and the genetic covariance are fitted on the pair-merged SNP
    set with shared blocks, and the ratio is jackknifed, as in ldsc --rg. The regression
    weights are simplified (1 / LD score, with one reweighting step for each h2) rather
    than ldsc's full heteroskedasticity weights.

    Corrected 2026-10-01. The first version divided the genetic-covariance slope by
    heritabilities estimated on each trait's own SNP set and took the SE of the
    numerator alone, so the ratio was neither the standard estimator nor jackknifed as
    a ratio (MR-AUDIT-2026-10-01.md, E3). Here h1, h2 and the genetic covariance are all
    fitted on the pair-merged SNP set with the same 200 genomic blocks, and the ratio
    rg = rho / sqrt(h1 h2) is jackknifed as a whole from its delete-one-block values.
    """
    d = merged(t1, t2)
    blocks = d.block.to_numpy()
    l2 = d.L2.to_numpy()
    Nn = np.sqrt(d.N1.to_numpy() * d.N2.to_numpy())
    x = l2 * Nn / M
    y = d.Z1.to_numpy() * d.Z2.to_numpy()
    w = 1.0 / np.maximum(l2, 1.0)
    (icept, s), _, ests = _reg(x, y, w, blocks)
    (_, s1), _, e1 = _h2_fit(d.Z1.to_numpy(), d.N1.to_numpy(), l2, blocks)
    (_, s2), _, e2 = _h2_fit(d.Z2.to_numpy(), d.N2.to_numpy(), l2, blocks)
    r = float(s) / np.sqrt(max(s1, 1e-6) * max(s2, 1e-6))
    r_del = ests[:, 1] / np.sqrt(np.maximum(e1[:, 1], 1e-6) * np.maximum(e2[:, 1], 1e-6))
    nb = len(r_del)
    rse = float(np.sqrt((nb - 1) / nb * np.sum((r_del - r_del.mean()) ** 2)))
    z = r / rse if rse > 0 else np.nan
    from scipy.stats import norm
    JK[f"{t1}|{t2}"] = dict(gencov=float(s), h2_1=float(s1), h2_2=float(s2),
                            gencov_del=ests[:, 1].tolist(), h2_1_del=e1[:, 1].tolist(),
                            h2_2_del=e2[:, 1].tolist(), n_snp=len(d))
    return dict(rg=r, se=rse, z=float(z), p=float(2 * norm.sf(abs(z))),
                intercept=float(icept), n_snp=len(d), h2_1_merged=float(s1),
                h2_2_merged=float(s2), gencov=float(s))


print("\n" + "=" * 96)
print("HERITABILITY (observed scale) — check these against published values")
print("=" * 96)
H = {}
for t in sorted(sumstats):
    H[t] = h2(t)
    print(f"{t:<26}h2 = {H[t]['h2']:.4f} ({H[t]['se']:.4f})   "
          f"intercept {H[t]['intercept']:.3f}   {H[t]['n_snp']:,} SNPs")

print("\n" + "=" * 96)
print("GENETIC CORRELATION")
print("=" * 96)
print(f"{'trait 1':<26}{'trait 2':<24}{'rg':>9}{'se':>8}{'p':>11}  note")
res = {}
for a, b, note in PAIRS:
    if a not in H or b not in H:
        continue
    r = rg(a, b, H[a], H[b])
    res[f"{a}|{b}"] = dict(**r, note=note)
    flag = " <<<" if note == "THE QUESTION" else ""
    print(f"{a:<26}{b:<24}{r['rg']:>9.3f}{r['se']:>8.3f}{r['p']:>11.2e}  "
          f"{note}{flag}")

print("\n" + "=" * 96)
print("READING")
print("=" * 96)
pc = res.get("HRV_RMSSD|HRV_SDNN", {}).get("rg")
qc = res.get("RestingHeartRate|HRV_RMSSD", {}).get("rg")
nc = res.get("EducationalAttainment|RheumatoidArthritis", {}).get("rg")
print(f"  same-study control  rg(HRV_RMSSD, HRV_SDNN)   = {pc:+.3f}   expect ~1")
print(f"  QUANTITATIVE control rg(RestingHeartRate, HRV) = {qc:+.3f}   "
      f"published -0.74 to -0.55")
print(f"  (rg(Education, RheumatoidArthritis)            = {nc:+.3f})")
print("""
  The same-study control cannot fail: HRV_RMSSD and HRV_SDNN come from one study and
  share allele coding, so it passed even when effect alleles were being mismatched
  across studies and every cross-study rg was collapsing toward zero. The control that
  decides whether this implementation works is the quantitative one against a
  published value.

  Educational attainment is NOT a valid null partner — it is the most pleiotropic
  trait in human genetics and correlates with most disease outcomes, so a non-zero
  rg with rheumatoid arthritis is expected and says nothing about the code. It is
  printed for reference only.""")
# the decisive check is agreement with a published estimate, not a "should be zero"
ok = (pc is not None and pc > 0.6) and (qc is not None and -1.1 < qc < -0.45)
print("\n  " + ("CONTROLS BEHAVE — the estimates above can be read"
                if ok else
                "QUANTITATIVE CONTROL FAILS — do not use these numbers"))

with open(OUT, "w") as f:
    json.dump(dict(h2=H, rg=res, controls_ok=bool(ok), M=M,
                   method="ldsc --rg: h1, h2 and gencov on the pair-merged SNP set, "
                          "delete-one-block ratio jackknife over 200 blocks"), f, indent=2)
with open(OUT_JK, "w") as f:
    json.dump({k: v for k, v in JK.items()
               if k.split("|")[0] in COG and k.split("|")[1] in
               ("HRV_RMSSD", "HRV_SDNN", "RestingHeartRate", "HRV_RMSSDc")}, f)
print(f"\nwrote {OUT}\nwrote {OUT_JK}")
