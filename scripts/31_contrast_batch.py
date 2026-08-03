"""
Trait contrast across every section, plus a cross-section consistency test.
==========================================================================

`30_trait_contrast.py` showed, on one section, that subtracting a non-cardiac
control removes the tissue-activity gradient and leaves anatomy-specific signal.
One section proves nothing on its own: with ~5 compartments, one lands on top by
chance one time in five.

This runs the same contrast on every finished section and asks whether the winner
is REPRODUCIBLE — the claim a reviewer will actually test.

Per section, per trait (vs control), per compartment:
    d(spot) = -log10 p_trait - (-log10 p_control)      paired, identical spots
    rank compartments by median d

Across sections:
  * how often each compartment ranks first, per region and trait
  * a sign test on the head-to-head pair that matters for each trait
  * binomial p for that many wins under no preference

Statistics note: a label-shuffling z is NOT comparable across compartments of
different size — a bigger compartment has a tighter null, so equal effects give a
bigger z. Ranking therefore uses median d, which is size-independent and robust.

Input: results/spotlevel/<section>__<trait>.csv  (see 32_collect_spotlevel.sh)

Usage:  python scripts/31_contrast_batch.py
"""

from pathlib import Path
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest

ROOT = str(Path(__file__).resolve().parent.parent)
SRC = f"{ROOT}/results/spotlevel"
OUT = f"{ROOT}/results"
CONTROL = "EducationalAttainment"
MIN_SPOTS = 50

# what each trait should hit if the premise holds, and the rival it must beat
EXPECT = {
    "RestingHeartRate":    ("node", "myocardium_atrial"),
    "AtrialFibrillation":  ("myocardium_atrial", "node"),
    "PRinterval":          ("AV_bundle", "myocardium_ventricular"),
}


def harmonise(ann, region):
    """
    The source atlas labels the same structure two different ways across sections.
    Verified by cross-tabulating annotation_final against sangerID in SAN.h5ad:

        myocardium        appears ONLY in HCAHeartST10659160 (2175) and
                          HCAHeartST12992072 (2291)
        myocardium_atrial appears ONLY in the other six sections (606-2368)

    They are mutually exclusive — never both on one section. The sinoatrial node
    sits in the right atrium and the myocardium surrounding it is atrial, and the
    AVN sections do distinguish myocardium_ventricular from myocardium_atrial,
    so a bare `myocardium` is the loose spelling of atrial myocardium.

    Merged for SAN sections only. AVN sections keep their explicit labels, where
    `myocardium` never appears.
    """
    if region == "SAN":
        return np.where(ann == "myocardium", "myocardium_atrial", ann)
    return ann


def load():
    d = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(SRC, "*__*.csv"))):
        base = os.path.basename(p)[:-4]
        section, trait = base.rsplit("__", 1)
        rows = list(csv.DictReader(open(p)))
        if not rows:
            continue
        region = section.split("__")[0]
        d[section][trait] = dict(
            key=[r[""] for r in rows],
            logp=np.array([float(r["logp"]) for r in rows]),
            ann=harmonise(np.array([r["annotation"] for r in rows]), region),
        )
    return d


sections = load()
print(f"sections found: {len(sections)}")
usable = {s: t for s, t in sections.items() if CONTROL in t and len(t) >= 2}
print(f"usable (have {CONTROL} + >=1 trait): {len(usable)}\n")

records = []
for sec, d in sorted(usable.items()):
    ctrl = d[CONTROL]
    order = {k: i for i, k in enumerate(ctrl["key"])}
    region = sec.split("__")[0]
    for trait, v in d.items():
        if trait == CONTROL:
            continue
        try:
            idx = np.array([order[k] for k in v["key"]])
        except KeyError:
            print(f"  {sec}/{trait}: spot keys do not match control — skipped")
            continue
        diff = v["logp"] - ctrl["logp"][idx]
        ann = v["ann"]
        for c in sorted(set(ann)):
            m = ann == c
            if m.sum() < MIN_SPOTS:
                continue
            records.append(dict(section=sec, region=region, trait=trait, compartment=c,
                                n=int(m.sum()), median_d=float(np.median(diff[m])),
                                mean_d=float(diff[m].mean())))

if not records:
    raise SystemExit("no usable sections yet")

with open(f"{OUT}/contrast_per_section.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0]), delimiter="\t")
    w.writeheader()
    w.writerows(records)

# ---------------------------------------------------------------- winners
by = defaultdict(list)
for r in records:
    by[(r["region"], r["trait"], r["section"])].append(r)

winners = defaultdict(Counter)
head2head = defaultdict(lambda: [0, 0])
for (region, trait, sec), rs in by.items():
    top = max(rs, key=lambda x: x["median_d"])
    winners[(region, trait)][top["compartment"]] += 1
    md = {r["compartment"]: r["median_d"] for r in rs}
    exp = EXPECT.get(trait)
    if exp and exp[0] in md and exp[1] in md:
        if md[exp[0]] > md[exp[1]]:
            head2head[(region, trait)][0] += 1
        else:
            head2head[(region, trait)][1] += 1

print("=" * 84)
print("WHICH COMPARTMENT RANKS FIRST (median d) — counted across sections")
print("=" * 84)
for (region, trait), cnt in sorted(winners.items()):
    tot = sum(cnt.values())
    exp = EXPECT.get(trait, ("?", "?"))[0]
    mark = "  <-- as predicted" if cnt.most_common(1)[0][0] == exp else ""
    print(f"  {region:<4} {trait:<22} " +
          ", ".join(f"{c}={n}/{tot}" for c, n in cnt.most_common()) + mark)

print()
print("=" * 84)
print("HEAD-TO-HEAD SIGN TEST (predicted compartment vs its rival)")
print("=" * 84)
summary = {}
for (region, trait), (a, b) in sorted(head2head.items()):
    n = a + b
    if n == 0:
        continue
    win, lose = EXPECT[trait]
    p = binomtest(a, n, 0.5, alternative="two-sided").pvalue
    ok = "PASS" if a > b else ("tie" if a == b else "FAIL")
    print(f"  {region:<4} {trait:<22} {win} {a}/{n} vs {lose} {b}/{n}   "
          f"binom p={p:.4f}   {ok}")
    summary[f"{region}|{trait}"] = dict(predicted=win, rival=lose,
                                        wins=a, losses=b, n=n, p=p, verdict=ok)

json.dump(dict(n_sections=len(usable), control=CONTROL,
               winners={f"{k[0]}|{k[1]}": dict(v) for k, v in winners.items()},
               head_to_head=summary),
          open(f"{OUT}/contrast_summary.json", "w"), indent=2)

print(f"\nwrote {OUT}/contrast_per_section.tsv and {OUT}/contrast_summary.json")
print(f"\n(Interim result — {len(usable)} of 16 sections done so far.)")
