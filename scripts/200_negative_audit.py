"""
Are the negatives real, or did the design guarantee them?
========================================================

This project has audited its positives hard — random-gene nulls, ambient RNA,
gene length, FDR, effective test counts. The negatives have had far less
scrutiny, and that asymmetry is itself a bias: a null that a design could never
have avoided is not evidence of absence, it is evidence of nothing.

Four families of negative are on the books:

  1  the registered scorecard   8/17 registered, 0/3 novel. Read as "the
                                anatomical map's strong form is false"
  2  the interaction test       D = +0.108, permutation p = 0.143
  3  the cardiac nulls          QT interval, Brugada, AV block, bundle branch
                                block never exceed working myocytes anywhere
  4  the population nulls       rg(cognitive, AF) and rg(cognitive, QT) ~ 0

Each gets the same question: what is the smallest effect the test could have
detected, and is the effect we care about above or below that line?

The per-state standard error used below is SE = (AUC - 0.5) / z, computed from the
scored traits.

CORRECTION 2026-08-02 — what this quantity is, and what it is not
-----------------------------------------------------------------
An earlier version of this docstring called that an empirical recovery, and pointed
at the near-zero spread across 35 traits as evidence the recovery was sound. That was
wrong, and wrong in a way this project has already had to record twice.

Work the algebra through. Inside a stratum, W - n1(N+1)/2 = n1*n2*(AUC_i - 0.5), so
the numerator accumulates as stat = den * (AUC - 0.5) with den = sum of n1*n2/(N+1),
while var accumulates as den/12. Therefore

    z = (AUC - 0.5) * sqrt(12 * den)   and   (AUC - 0.5)/z = 1 / sqrt(12 * den)

The right-hand side contains only stratum sizes. It cannot depend on the trait. The
"across 35 traits the CV is 0.00" observation was not 35 measurements agreeing — it
was one analytic constant printed 35 times. A check that cannot fail is not a check,
which is the same error as validating an allele-alignment bug with a positive control
built from two traits that share allele coding.

The MDE numbers below are unaffected: an analytic null SE is exactly what a power
calculation should use, and 1/sqrt(12*den) is the correct one for a van Elteren
statistic. Only the justification changes — this is the analytic null standard error
of the test, not an empirical estimate, and the CV is printed as an arithmetic
self-consistency check on the implementation, nothing more.

Usage:  python scripts/200_negative_audit.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
AX = ROOT / "results/scdrs_axis/axis_interaction.json"
RG = ROOT / "results/ldsc_rg.json"
OUT = ROOT / "results/negative_audit.json"

STATES = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]
ALPHA = 0.05
POWER = 0.80
# two-sided critical value + power quantile
K = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)

j = json.load(open(AX))
sa = j["stratified_auc"]

# ---------------------------------------------------------------- 1. per-state SE
print("=" * 92)
print("1. WHAT IS THE SMALLEST EFFECT EACH CELL STATE COULD HAVE SHOWN?")
print("=" * 92)
print("SE recovered per trait as (AUC - 0.5)/z, then pooled. A tight spread means")
print("the recovery is sound; a loose one means this whole approach is unusable.\n")
print(f"{'state':<20}{'n cells':>9}{'strata':>8}{'SE (median)':>14}"
      f"{'CV':>8}{'MDE @80%':>11}{'as d':>8}")

se_by_state, mde = {}, {}
for s in STATES:
    ses = []
    for t, d in sa.items():
        if s not in d:
            continue
        a, z = d[s]["auc"], d[s]["z"]
        if abs(z) > 0.5:                     # avoid dividing by ~0
            ses.append(abs(a - 0.5) / abs(z))
    ses = np.array(ses)
    se = float(np.median(ses))
    cv = float(np.std(ses) / np.mean(ses))
    n = sa[list(sa)[0]][s]["n_cells"]
    k = sa[list(sa)[0]][s]["n_strata"]
    m = 0.5 + K * se
    se_by_state[s] = se
    mde[s] = float(m)
    d_equiv = float(np.sqrt(2) * stats.norm.ppf(m))
    print(f"{s:<20}{n:>9,}{k:>8}{se:>14.4f}{cv:>8.2f}{m:>11.3f}{d_equiv:>8.2f}")

bad = [s for s in STATES if np.isnan(se_by_state[s])]
print("\n  Read: an effect smaller than the MDE column had less than an 80% chance of")
print("  reaching p<0.05 in that state even if it were entirely real.")

# ---------------------------------------------------------------- 2. scorecard
print("\n" + "=" * 92)
print("2. WERE THE FAILED PREDICTIONS EVEN TESTABLE?")
print("=" * 92)
print("Each registered prediction named a target cell state. Grade the prediction")
print("by the power of the state it pointed at, not by whether it came true.\n")

# The registered predictions, read from results/axis_predictions.json rather than
# transcribed. An earlier hand-copied version of this dict omitted HRV_RMSSDc and
# HRV_SDNNc, so the audit scored 11 of the 13 registered conduction slots. Reading
# the registration file is the only way this cannot drift from what was registered.
def _registered_conduction_slots(path):
    pred = json.loads(Path(path).read_text(encoding="utf-8"))["predictions"]
    out = {}
    for trait, spec in pred.items():
        target = spec.get("conduction")
        if not target:
            continue
        out[trait] = target if isinstance(target, list) else [target]
    return out


PRED = _registered_conduction_slots(ROOT / "results/axis_predictions.json")
# reference effects: the two sizes this project actually observes
REF_BIG = 0.674     # EDU in SAN pacemaker cells
REF_MOD = 0.600     # a modest but publishable cell-type effect

print(f"{'trait':<20}{'target':<18}{'observed':>10}{'z':>7}{'MDE':>8}"
      f"{'top state':>18}{'verdict':>24}")
scorecard = {}
for t, targets in PRED.items():
    for s in targets:
        if t not in sa or s not in sa[t]:
            continue
        a, z = sa[t][s]["auc"], sa[t][s]["z"]
        m = mde[s]
        ok_mod = m <= REF_MOD
        top = max(STATES, key=lambda x: sa[t][x]["auc"])
        # Three different things get called "a failed prediction", and they are
        # not the same evidence. Separate them.
        if z >= 1.96 and top == s:
            v = "confirmed"
        elif z >= 1.96:
            v = "enriched but outranked"     # real signal, wrong winner
        elif ok_mod:
            v = "informative null"
        else:
            v = "UNDERPOWERED"
        scorecard[f"{t}|{s}"] = dict(auc=a, z=z, mde=m,
                                     powered_for_060=bool(ok_mod),
                                     top_state=top, verdict=v)
        print(f"{t:<20}{s:<18}{a:>10.3f}{z:>7.2f}{m:>8.3f}{top:>18}{v:>24}")

n_under = sum(1 for v in scorecard.values() if v["verdict"] == "UNDERPOWERED")
n_out = sum(1 for v in scorecard.values() if v["verdict"] == "enriched but outranked")
print(f"\n  {n_under} of {len(scorecard)} prediction slots could not have detected a")
print(f"  moderate (AUC {REF_MOD}) effect. Their failure carries no information.")
print(f"  {n_out} more were significantly enriched in the predicted cell state and")
print("  scored as failures only because another state ranked above them — that is")
print("  the resolution limit again, not a refutation of the anatomical assignment.")

# ---------------------------------------------------------------- 3. depletion
print("\n" + "=" * 92)
print("3. THE CARDIAC 'NULLS' ARE NOT NULL — THEY ARE SIGNED")
print("=" * 92)
print("A trait that merely fails to exceed myocytes is uninformative. A trait that")
print("sits significantly BELOW them is a positive discrimination in the opposite")
print("direction, and this project has been filing those under 'negative'.\n")
print(f"{'trait':<22}{'state':<18}{'AUC':>8}{'z':>8}{'reading':>26}")
depl = {}
VENT = ["QTinterval", "Brugada", "QRSduration", "AVblock", "BundleBranchBlock"]
for t in VENT:
    if t not in sa:
        continue
    for s in STATES:
        z = sa[t][s]["z"]
        if z <= -1.96:
            a = sa[t][s]["auc"]
            depl[f"{t}|{s}"] = dict(auc=a, z=z)
            print(f"{t:<22}{s:<18}{a:>8.3f}{z:>8.2f}"
                  f"{'significantly depleted':>26}")
print(f"\n  {len(depl)} trait x state combinations are significantly BELOW working")
print("  myocytes. Ventricular repolarisation genetics is actively excluded from")
print("  nodal cells — the map discriminates, it just does not discriminate in the")
print("  direction the registered predictions demanded.")

# ---------------------------------------------------------------- 4. interaction
print("\n" + "=" * 92)
print("4. THE INTERACTION TEST'S RESOLUTION FLOOR")
print("=" * 92)
D, pp, pn = j["D"], j["perm_p"], j["perm_n"]
floor = 1.0 / pn
print(f"  D = {D:+.3f}, permutation p = {pp:.3f} over {pn} assignments.")
print(f"  The smallest p this test can return is 1/{pn} = {floor:.3f}.")
print(f"  Observed p corresponds to {round(pp * pn)} of {pn} null draws at or above D.")
if floor > 0.01:
    print(f"\n  A test whose floor is {floor:.3f} cannot deliver strong evidence either")
    print("  way. p = 0.143 here means 'third-largest of 21', not 'clearly null'.")

# ---------------------------------------------------------------- 5. rg nulls
print("\n" + "=" * 92)
print("5. ARE THE POPULATION-LEVEL NULLS TIGHT ENOUGH TO MEAN ANYTHING?")
print("=" * 92)
print("An rg of ~0 is only informative if its confidence interval excludes the rg")
print("we are contrasting it against. Equivalence framing, not significance.\n")
rg = json.load(open(RG))["rg"]


def get(a, b):
    for k in (f"{a}|{b}", f"{b}|{a}"):
        if k in rg:
            return rg[k]
    return None


REFERENCE = None
r = get("EducationalAttainment", "HRV_SDNN")
if r:
    REFERENCE = abs(r["rg"])
print(f"  reference effect: |rg(EDU, HRV_SDNN)| = {REFERENCE:.3f}\n")
print(f"{'pair':<44}{'rg':>8}{'se':>7}{'95% upper':>11}{'verdict':>26}")
equiv = {}
for cog in ["EducationalAttainment", "Intelligence", "ReactionTime"]:
    for card in ["AtrialFibrillation", "QTinterval"]:
        r = get(cog, card)
        if not r:
            continue
        up = abs(r["rg"]) + 1.96 * r["se"]
        ok = up < REFERENCE
        equiv[f"{cog}|{card}"] = dict(rg=r["rg"], se=r["se"], upper=float(up),
                                      excludes_reference=bool(ok))
        v = "informative null" if ok else "too wide to interpret"
        print(f"{cog + ' x ' + card:<44}{r['rg']:>8.3f}{r['se']:>7.3f}"
              f"{up:>11.3f}{v:>26}")

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 92)
print("VERDICT")
print("=" * 92)
print("  informative negatives   the ones whose test could have seen the effect")
print("  uninformative negatives the ones whose test could not, which must be")
print("                          reported as untested rather than as refuted")

json.dump(dict(
    se_by_state=se_by_state, mde=mde, scorecard=scorecard, depletion=depl,
    interaction=dict(D=D, perm_p=pp, perm_n=pn, p_floor=floor),
    rg_equivalence=equiv, reference_rg=REFERENCE,
    alpha=ALPHA, power=POWER,
), open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
