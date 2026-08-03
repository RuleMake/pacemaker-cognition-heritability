"""
Was the spatial null the expected outcome? Close the reversal quantitatively.
============================================================================

The spatial arm ranked the sinoatrial node LAST for heart rate variability, across
8 of 8 sections. The single-cell arm ranks sinoatrial pacemaker cells first. The
project has explained that reversal three ways — dilution, technical confounding,
insufficient power — but has never done the one calculation that would settle whether
the spatial result was a failure of the tissue or simply the arithmetic working out
as it had to.

That calculation is available now, because the pieces are measured rather than assumed:

  effect     the single-cell effect, as a stratified AUC of pacemaker cells against
             the working myocytes beside them, converted to a standardised difference
  dilution   the measured pacemaker fraction of a Visium spot — median 6.2% in spots
             the authors annotated as node, 28.0% in the top 1% by abundance
  design     8 sections, and the actual number of node spots in them

A spot's score is a mixture: a pacemaker cell contributes its excess only in
proportion to how much of the spot it is. So the expected spot-level effect is the
cell-level effect multiplied by purity, and the question becomes whether an effect
that small was ever detectable with the number of spots and sections available.

If the answer is that the spatial design had, say, a 6% chance of seeing the effect
it was looking for, then "spatial found nothing" is not evidence against the
hypothesis. It is what the design was guaranteed to produce, and the reversal needs
no further explanation.

SUPERSEDED IN PART — read scripts/192_corrections.py for the numbers to quote.
-----------------------------------------------------------------------------
This script feeds in the cell-level effect measured against WORKING MYOCYTES, but the
spatial test compared node-region spots against spots from other compartments, whose
background is every cell type rather than myocytes. Against an all-cell background the
pacemaker effect is AUC 0.837 (d = 1.39), not 0.666 (d = 0.61), and 192 redoes the
calculation both ways. The per-section sample-size bug — every scenario was given the
node-spot count, inflating the 272-spot scenario's power from 62% to 96% — is fixed
here as well as there.

Usage:  python scripts/172_spatial_power.py
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.as_posix()
DIL = f"{ROOT}/results/dilution.json"
AX = f"{ROOT}/results/scdrs_axis/axis_interaction.json"
OUT = f"{ROOT}/results/spatial_power.json"

N_SECTIONS = 8
ALPHA = 0.05

# ------------------------------------------------------------------ inputs
d = json.load(open(DIL)) if os.path.exists(DIL) else {}
pur = d.get("purity", {})
node = pur.get("annotated `node` only", {})
top1 = pur.get("top 1% by pacemaker abundance", {})
n_node_spots = int(node.get("n", 4505))
purity_median = float(node.get("median", 6.216)) / 100.0
purity_top = float(top1.get("median", 27.97)) / 100.0
n_top_spots = int(top1.get("n", 272))

auc = None
if os.path.exists(AX):
    j = json.load(open(AX))
    auc = (j.get("stratified_auc", {}).get("HRV_SDNN", {})
           .get("SAN_P_cell", {}).get("auc"))
if auc is None:
    auc = 0.666      # measured value, kept as a documented fallback
# AUC -> Cohen's d for the underlying difference (normal approximation)
d_cell = float(np.sqrt(2) * stats.norm.ppf(auc))

print("MEASURED INPUTS")
print(f"  single-cell effect        AUC {auc:.3f}  ->  d = {d_cell:.3f}")
print(f"  node spots                {n_node_spots:,} across {N_SECTIONS} sections "
      f"({n_node_spots // N_SECTIONS:,} per section)")
print(f"  pacemaker fraction        median {purity_median * 100:.1f}% in node spots, "
      f"{purity_top * 100:.1f}% in the top 1% ({n_top_spots} spots)")


def power_two_sample(dd, n1, n2, alpha=ALPHA, two_sided=True):
    """Power of a two-sample test to detect standardised difference dd."""
    if n1 < 2 or n2 < 2 or dd <= 0:
        return 0.0
    se = np.sqrt(1.0 / n1 + 1.0 / n2)
    crit = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    return float(stats.norm.sf(crit - dd / se))


print("\n" + "=" * 92)
print("EXPECTED SPOT-LEVEL EFFECT AND POWER")
print("=" * 92)
print(f"{'scenario':<40}{'purity':>9}{'d at spot':>12}{'n spots':>10}"
      f"{'power':>10}")
rows = {}
scenarios = [
    ("all annotated node spots", purity_median, n_node_spots),
    ("top 1% by pacemaker abundance", purity_top, n_top_spots),
    ("hypothetical: 50% pure spots", 0.50, n_top_spots),
]
# comparator pool: everything that is not a node spot. Its exact size barely matters
# once it is large, which it is.
n_bg = 27108
for label, p, n in scenarios:
    d_spot = d_cell * p
    pw = power_two_sample(d_spot, n, n_bg)
    rows[label] = dict(purity=p, d_spot=float(d_spot), n=n, power=float(pw))
    print(f"{label:<40}{p * 100:>8.1f}%{d_spot:>12.3f}{n:>10,}{pw * 100:>9.1f}%")

print("\n" + "=" * 92)
print("THE TEST THAT WAS ACTUALLY RUN: sign test across 8 sections")
print("=" * 92)
print("The spatial verdict was not a pooled comparison — it was whether the node")
print("ranked high in each section, read as 8 independent attempts. A per-section")
print("test that is itself underpowered makes the sign test across sections")
print("hopeless, and that compounding is the number that matters.\n")
print(f"{'scenario':<40}{'power/section':>16}{'P(>=6 of 8)':>14}"
      f"{'P(8 of 8)':>12}")
for label, p, n in scenarios:
    d_spot = d_cell * p
    # Each scenario has its OWN spot count per section. The first version used the
    # node-spot count for every row, which handed the 272-spot top-1% scenario the
    # 4,505-spot design's sample size and inflated its power from 62% to 96%.
    pw = power_two_sample(d_spot, max(n // N_SECTIONS, 2), n_bg // N_SECTIONS)
    p6 = float(sum(stats.binom.pmf(k, N_SECTIONS, pw) for k in range(6, 9)))
    p8 = float(stats.binom.pmf(8, N_SECTIONS, pw))
    rows[label].update(power_per_section=float(pw), p_6of8=p6, p_8of8=p8)
    print(f"{label:<40}{pw * 100:>15.1f}%{p6 * 100:>13.1f}%{p8 * 100:>11.1f}%")

# ------------------------------------------------------------------ verdict
print("\n" + "=" * 92)
print("VERDICT")
print("=" * 92)
base = rows["all annotated node spots"]
print(f"  The effect measured at single-cell resolution (d = {d_cell:.2f}) becomes")
print(f"  d = {base['d_spot']:.3f} at the spot level once diluted to "
      f"{base['purity'] * 100:.1f}% purity.")
print(f"  Power to detect that per section: {base['power_per_section'] * 100:.1f}%.")
print(f"  Probability of seeing it in 6 or more of 8 sections: "
      f"{base['p_6of8'] * 100:.1f}%.")
print()
rich = rows["top 1% by pacemaker abundance"]
if base["p_6of8"] < 0.20:
    print("  The COMPARTMENT-LEVEL spatial test could not have found this effect. Its")
    print("  null is the expected outcome and carries no information.")
else:
    print("  The compartment-level test had power, so its null is not dilution.")

print()
if rich["p_6of8"] > 0.50:
    print(f"  But the ABUNDANCE-BASED test is a different matter. Restricted to the")
    print(f"  {rich['n']} most pacemaker-rich spots ({rich['purity'] * 100:.0f}% pure) "
          f"it had {rich['power_per_section'] * 100:.0f}% power per section and")
    print(f"  {rich['p_6of8'] * 100:.0f}% probability of hitting 6 of 8 — and it still "
          f"found nothing.")
    print()
    print("  So dilution does NOT explain the spatial null on its own, and saying so")
    print("  would repeat an error this project has already had to retract once. The")
    print("  correct split is:")
    print("    * compartment-level null  -> uninformative, killed by dilution")
    print("    * abundance-based null    -> informative, and NOT explained by dilution")
    print("                                 The sequencing-depth artefact is what is")
    print("                                 left to explain it.")
else:
    print("  Neither the compartment-level nor the abundance-based test had power;")
    print("  dilution alone accounts for both nulls.")

print("\n  Caveat, stated rather than buried: this treats a spot's score as a linear")
print("  mixture of its cells, which is the same assumption deconvolution makes. It")
print("  ignores the sequencing-depth artefact, which acts on top of dilution and can")
print("  only make detection worse — so this is an upper bound on spatial power.")

with open(OUT, "w") as f:
    json.dump(dict(auc=auc, d_cell=d_cell, n_sections=N_SECTIONS,
                   scenarios=rows), f, indent=2)
print(f"\nwrote {OUT}")
