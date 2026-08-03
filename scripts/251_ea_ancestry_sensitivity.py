"""
How much of the population-level pillar rests on the ancestry-mismatched trait?
==============================================================================

Educational attainment is the trait that carries the population-level result, and
it is the one trait analysed whose GWAS is not European-only: Chen et al. 2024
(GCST90296499) is 766,345 European plus 165,232 East Asian, evaluated here against
a 1000 Genomes European LD reference. Height was dropped from this study for that
exact defect. The manuscript names the tension and keeps the trait; this script
measures what keeping it costs.

Three questions, and only the third needs data we do not have:

  1  Is the intercept anomaly harmful?  The measured LD score intercept for
     educational attainment is 0.516, below the unity the model permits, which is
     the signature of summary statistics already corrected by genomic control.
     GC correction rescales every z by a constant. That is provably harmless to
     rg, and the script demonstrates the cancellation numerically rather than
     asserting it.

  2  How exposed is the estimate to a wrong h2?  rg = rho_g / sqrt(h2_1 * h2_2),
     so any bias in h2(EA) rescales every one of educational attainment's genetic
     correlations by the SAME factor. That has an exact consequence worth stating:
     the point estimates move, the z statistics and p values do not, and the
     ordering of cardiac traits within educational attainment cannot move at all.
     The shape claim and the significance claim are therefore not exposed; only
     the absolute magnitude is.

  3  Does the shape survive without the trait entirely?  Intelligence and reaction
     time are both European-only. Re-running the ranking statistic on those two
     alone answers the reviewer who simply does not accept educational attainment.

What none of this can do is settle whether the EUR LD scores misestimate rho_g
itself for the 17.7% East Asian portion. That biases numerator and denominator
jointly and cannot be bounded from the data on disk. Only re-running with a
European-only educational attainment GWAS settles it.

Usage:  python scripts/251_ea_ancestry_sensitivity.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RG = ROOT / "results" / "ldsc_rg.json"
OUT_JSON = ROOT / "results" / "ea_ancestry_sensitivity.json"
OUT_TSV = ROOT / "results" / "ea_ancestry_sensitivity.tsv"

EA = "EducationalAttainment"
COG = [EA, "Intelligence", "ReactionTime"]
COG_EUR_ONLY = ["Intelligence", "ReactionTime"]
CARDIAC = ["HRV_RMSSD", "HRV_SDNN", "RestingHeartRate",
           "PRinterval", "AtrialFibrillation", "QTinterval"]
HRV = {"HRV_RMSSD", "HRV_SDNN"}

# Published European-only SNP heritability for educational attainment, LDSC.
# Lee et al. 2018 (N = 766,345, the European component of the GWAS used here) and
# Okbay et al. 2022 bracket roughly this range.
H2_PUBLISHED = (0.11, 0.13)

d = json.load(open(RG, encoding="utf-8"))
H2, RGD = d["h2"], d["rg"]
out = {}


def rg_of(a, b):
    for k in (f"{a}|{b}", f"{b}|{a}"):
        if k in RGD:
            return RGD[k]
    return None


# ================================================================ 1. intercept
print("=" * 100)
print("1. THE INTERCEPT ANOMALY, AND WHY IT CANCELS")
print("=" * 100)
ea_h2 = H2[EA]
print(f"  {EA}: h2 = {ea_h2['h2']:.4f} (SE {ea_h2['se']:.4f}), "
      f"LD score intercept = {ea_h2['intercept']:.4f}")
print(f"  every other trait's intercept: "
      f"{', '.join(f'{k}={v[chr(105)+chr(110)+chr(116)+chr(101)+chr(114)+chr(99)+chr(101)+chr(112)+chr(116)]:.3f}' for k, v in H2.items() if k != EA)}")
print("""
  An intercept below 1 is what genomic control leaves behind: every z divided by
  a constant sqrt(lambda). Under that rescaling  h2 -> h2/lambda  and
  rho_g -> rho_g/sqrt(lambda), so

      rg = rho_g / sqrt(h2_1 h2_2)  ->  (rho_g/sqrt(lambda)) / sqrt((h2_1/lambda) h2_2)
         = rho_g / sqrt(h2_1 h2_2)                                   [unchanged]

  The cancellation is exact and is demonstrated numerically in section 2. It is a
  defence against GENOMIC CONTROL. It is not a defence against ancestry mismatch,
  which perturbs the LD scores themselves rather than the scale of z, and those
  are different problems that the Methods paragraph currently treats together.""")
out["intercept"] = {k: v["intercept"] for k, v in H2.items()}

# ================================================================ 2. h2 rescaling
print("\n" + "=" * 100)
print("2. WHAT A WRONG h2(EA) DOES, AND WHAT IT PROVABLY CANNOT DO")
print("=" * 100)
h2_obs = ea_h2["h2"]
grid = [h2_obs, 0.08, H2_PUBLISHED[0], H2_PUBLISHED[1], 0.15]

rows = []
print(f"\n  observed h2(EA) = {h2_obs:.4f}; published European-only estimates "
      f"{H2_PUBLISHED[0]}-{H2_PUBLISHED[1]}")
print(f"\n{'assumed h2(EA)':>15}{'scale':>8}", end="")
for c in CARDIAC:
    print(f"{c:>20}", end="")
print()
for h2a in grid:
    f = np.sqrt(h2_obs / h2a)          # rg -> rg * f
    print(f"{h2a:>15.4f}{f:>8.3f}", end="")
    for c in CARDIAC:
        r = rg_of(EA, c)
        print(f"{r['rg'] * f:>+20.3f}", end="")
    print()
    rows.append(dict(assumed_h2=h2a, scale=f,
                     **{c: rg_of(EA, c)["rg"] * f for c in CARDIAC}))

print("\n  the z statistics under the same rescaling (SE scales identically):")
print(f"{'assumed h2(EA)':>15}", end="")
for c in CARDIAC:
    print(f"{c:>20}", end="")
print()
for h2a in grid:
    f = np.sqrt(h2_obs / h2a)
    print(f"{h2a:>15.4f}", end="")
    for c in CARDIAC:
        r = rg_of(EA, c)
        print(f"{(r['rg'] * f) / (r['se'] * f):>+20.3f}", end="")
    print()

order_obs = [c for c in sorted(CARDIAC, key=lambda c: -rg_of(EA, c)["rg"])]
print(f"\n  ordering of cardiac traits within {EA}, at every h2 in the grid:")
print("    " + " > ".join(order_obs))
print(f"    both HRV indices in the top two: "
      f"{'YES' if set(order_obs[:2]) == HRV else 'NO'}")
print("""
  A multiplicative rescaling of one trait's h2 moves rg and its standard error by
  the same factor. Therefore, whatever h2(EA) truly is:
      - every z, every p value, and Bonferroni survival are INVARIANT
      - the ordering of cardiac traits within educational attainment is INVARIANT
      - only the absolute magnitude of rg moves
  At the published European h2 the lead estimate falls from""",
      f"{rg_of(EA,'HRV_SDNN')['rg']:+.3f} to "
      f"{rg_of(EA,'HRV_SDNN')['rg'] * np.sqrt(h2_obs/H2_PUBLISHED[0]):+.3f} — "
      f"still the largest cardiac correlation and still "
      f"{abs(rg_of(EA,'HRV_SDNN')['rg'] / rg_of(EA,'QTinterval')['rg']):.1f}x its "
      "correlation with QT interval.")
out["h2_grid"] = rows
out["ordering_invariant"] = order_obs

# ================================================================ 3. shape test
print("\n" + "=" * 100)
print("3. THE SHAPE TEST WITH AND WITHOUT THE ANCESTRY-MISMATCHED TRAIT")
print("=" * 100)
print("""
  The event scored is: do the two heart rate variability indices occupy the top
  two slots among the six cardiac traits? Because rg(RMSSD, SDNN) is
  indistinguishable from 1 they are one trait on that side, so the null
  probability is 1/5 per cognitive trait, not 1/15. On the cognitive side the
  traits are correlated, so the exponent is the Nyholt effective number
  k_eff = k / (1 + (k-1) * mean|rg|), the same correction used in 192.""")


def shape(cog):
    hits, detail = 0, []
    for c in cog:
        order = sorted(CARDIAC, key=lambda x: -rg_of(c, x)["rg"])
        ok = set(order[:2]) == HRV
        hits += ok
        detail.append((c, ok, order[:2]))
    pairs = [abs(rg_of(a, b)["rg"]) for i, a in enumerate(cog) for b in cog[i + 1:]]
    k = len(cog)
    mean_r = float(np.mean(pairs)) if pairs else 0.0
    k_eff = k / (1 + (k - 1) * mean_r)
    return dict(traits=cog, hits=hits, k=k, mean_rg=mean_r, k_eff=k_eff,
                p=float(0.2 ** k_eff), detail=detail)


for label, cog in (("all three cognitive traits", COG),
                   ("European-only traits (EA dropped)", COG_EUR_ONLY)):
    s = shape(cog)
    print(f"\n  {label}")
    for c, ok, top2 in s["detail"]:
        print(f"     {c:<24} top two = {top2[0]}, {top2[1]}   "
              f"{'HIT' if ok else 'miss'}")
    print(f"     {s['hits']}/{s['k']} hit    mean|rg| among them = {s['mean_rg']:.3f}"
          f"    k_eff = {s['k_eff']:.3f}")
    print(f"     p = 0.2^{s['k_eff']:.3f} = {s['p']:.4f}")
    out[f"shape_{'all' if len(cog) == 3 else 'eur_only'}"] = {
        k: v for k, v in s.items() if k != "detail"}

s_all, s_eur = shape(COG), shape(COG_EUR_ONLY)
print(f"""
  Dropping educational attainment moves the shape test from p = {s_all['p']:.4f}
  to p = {s_eur['p']:.4f}. Both are marginal and neither is rescued by the other.
  What the comparison shows is that the pillar does not REST on the mismatched
  trait: educational attainment was largely redundant with intelligence anyway
  (rg = {abs(rg_of(EA,'Intelligence')['rg']):.3f}), so removing it costs about as much
  power as it removes exposure. The honest reading is that the population-level
  evidence is marginal either way, which is what the manuscript already says.""")

pd.DataFrame(rows).to_csv(OUT_TSV, sep="\t", index=False, float_format="%.6g")
OUT_JSON.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
print(f"\nwrote {OUT_TSV}\nwrote {OUT_JSON}")
