"""
How large would the MR effect have to be for HRV alone to explain the genetic correlation?
=========================================================================================

Post hoc (Data S3: the comparison was specified after the first MR run and is reported as
descriptive). It turns a null MR estimate into a statement with content: if genetic
variation in HRV influenced a cognitive trait only through HRV itself (full mediation,
no feedback), the genetic covariance would equal beta * h2_exposure on the standardized
scale, so

    beta_needed = rho_g / h2_exposure = rg * sqrt(h2_outcome / h2_exposure).

This script replaces a computation that existed only as a results file
(results/mr_v2_rg_consistency.json had no generating code; MR-AUDIT-2026-10-01.md, E4).
Changes from that computation:

  * rho_g and h2_exposure come from the same pair-merged SNP set and the same 200 genomic
    blocks (scripts/170_ldsc_rg.py writes the delete-one-block values), and the SE of
    beta_needed is the jackknife SE of the ratio. The first version propagated se(rg) and
    both se(h2) as if independent, counted only a quarter of Var(h2_exposure) and ignored
    the covariance of rho_g with h2_exposure.
  * The 95% interval of beta_needed is a Fieller interval, because the ratio of a noisy
    numerator to a noisy denominator is skewed.
  * It reports the share of the genetic correlation that mediation through the exposure
    could account for, pi = beta_MR / beta_needed, with a Fieller upper bound. That is the
    quantity a reader needs: not only "full mediation is excluded" but "at most this much".

On genomic control: the educational-attainment GWAS was GC-corrected (LD score intercept
0.516), which deflates its Z, its h2 and the MR estimate with EA as outcome by the same
factor. The deflation therefore cancels in z, P and pi; beta_needed and beta_MR for EA
are both on the deflated scale, so their absolute values understate the undeflated ones
by the same factor and the comparison between them is unchanged.

Usage:  python scripts/186_mr_rg_consistency.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "mr_v2_rg_consistency.json"
COG = ["EducationalAttainment", "Intelligence", "ReactionTime"]
EXPO = ["HRV_RMSSD", "HRV_SDNN", "RestingHeartRate", "HRV_RMSSDc"]

jk = json.loads((RES / "ldsc_rg_jackknife.json").read_text())
est = pd.read_csv(RES / "mr_v2_estimates.tsv", sep="\t")
est = est[est.spec == "primary"].set_index(["exposure", "outcome"])
sens_f = RES / "mr_v2_sensitivity.tsv"
sens = (pd.read_csv(sens_f, sep="\t").query("analysis == 'hr_corrected_hrv'")
        .set_index(["exposure", "outcome"]) if sens_f.exists() else None)


def fieller(a, b, va, vb, cab, level=0.95):
    """Fieller interval for a / b given variances and covariance."""
    z2 = stats.norm.isf((1 - level) / 2) ** 2
    A = b ** 2 - z2 * vb
    B = -2 * (a * b - z2 * cab)
    C = a ** 2 - z2 * va
    disc = B ** 2 - 4 * A * C
    if A <= 0 or disc < 0:
        return (np.nan, np.nan)            # unbounded: denominator not distinguishable from 0
    r = np.sqrt(disc)
    return tuple(sorted(((-B - r) / (2 * A), (-B + r) / (2 * A))))


rows = []
for ex in EXPO:
    for ou in COG:
        key = f"{ou}|{ex}"
        if key not in jk:
            continue
        j = jk[key]
        gc, hx = j["gencov"], j["h2_2"]
        gd, hd = np.array(j["gencov_del"]), np.array(j["h2_2_del"])
        nb = len(gd)
        f = (nb - 1) / nb
        bn = gc / hx
        bn_del = gd / hd
        se_bn = float(np.sqrt(f * np.sum((bn_del - bn_del.mean()) ** 2)))
        v_gc = f * np.sum((gd - gd.mean()) ** 2)
        v_h = f * np.sum((hd - hd.mean()) ** 2)
        c_gh = f * np.sum((gd - gd.mean()) * (hd - hd.mean()))
        lo, hi = fieller(gc, hx, v_gc, v_h, c_gh)
        if ex == "HRV_RMSSDc":
            if sens is None or (ex, ou) not in sens.index:
                continue
            r = sens.loc[(ex, ou)]
        else:
            r = est.loc[(ex, ou)]
        b, se = float(r.ivw), float(r.ivw_se)
        z = (b - bn) / np.sqrt(se ** 2 + se_bn ** 2)
        # share of the genetic correlation that mediation through the exposure accounts for
        pi = b / bn
        pi_lo, pi_hi = fieller(b, bn, se ** 2, se_bn ** 2, 0.0)
        rows.append(dict(test=f"{ex}->{ou}", exposure=ex, outcome=ou,
                         rg=gc / np.sqrt(j["h2_1"] * hx),
                         beta_needed=bn, se_needed=se_bn, needed_ci95=[lo, hi],
                         ivw=b, ivw_se=se, z=float(z), p=float(2 * stats.norm.sf(abs(z))),
                         share_mediated=pi, share_mediated_ci95=[pi_lo, pi_hi],
                         analysis="sensitivity" if ex == "HRV_RMSSDc" else "primary"))
        print(f"{ex:>16} -> {ou:<22} needed {bn:+.3f} ({se_bn:.3f}) [{lo:+.3f}, {hi:+.3f}]"
              f"   IVW {b:+.4f} ({se:.4f})   z {z:+.2f}  P {2 * stats.norm.sf(abs(z)):.1e}"
              f"   share {pi:+.3f} [{pi_lo:+.3f}, {pi_hi:+.3f}]")

OUT.write_text(json.dumps(rows, indent=2))
print(f"\nwrote {OUT}")
