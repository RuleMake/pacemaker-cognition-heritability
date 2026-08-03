"""How much of each signal do the knockouts actually remove?

Script 141 printed "SEPARABLE" off two p-values. A p-value says a loss is larger than
chance; it says nothing about whether the loss matters. If deleting the neuronal
programme costs educational attainment 3% of its z, then "the brain signal rides on the
neuronal programme" is not a supportable sentence no matter how small p is.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.as_posix()
d = json.load(open(f"{ROOT}/results/scdrs/control_dissection.json"))
k = d["knockout"]

print(f"{'trait':<26}{'full z':>8}   {'knockout':<10}{'n drop':>7}{'dz':>8}"
      f"{'% of z':>9}{'p':>8}")
for t, v in k.items():
    z0 = v["full"]["z"]
    for lab in ("neuronal", "vagal"):
        x = v.get(lab, {})
        if "delta" not in x:
            print(f"{t:<26}{z0:>8.2f}   {lab:<10}{x.get('n_dropped', 0):>7}"
                  f"{'n/a':>8}{'':>9}{'':>8}")
            continue
        frac = 100 * abs(x["delta"]) / abs(z0)
        print(f"{t:<26}{z0:>8.2f}   {lab:<10}{x['n_dropped']:>7}"
              f"{x['delta']:>8.3f}{frac:>8.1f}%{x['p']:>8.3f}")

print("\ngene-set overlap with the neuronal programme (how much could be removed):")
prog = set(d["programme"])
for t, drv in d["drivers"].items():
    print(f"  {t:<26}{len(set(drv[:50]) & prog):>3}/50 top drivers are in the programme")

print("""
READING
-------
The neuronal programme is 16-26 genes deep in every trait's set and removing it costs
0.5-2.9% of the score. It does not carry educational attainment's signal in pacemaker
cells any more than it carries the cardiac ones. The hypothesis that the control hits
these cells because pacemaker cells run a neuronal programme is REFUTED by its own
test, and the neural cells in this atlas are mostly glia and Schwann cells anyway,
which is probably why: a glial programme is not the neuronal-excitability programme a
pacemaker cell would plausibly share.

The vagal genes are the opposite case. Nine genes carry 11.2% of HRV's score — by far
the largest per-gene contribution measured anywhere in this project, and the direct
quantitative version of the RGS6/CHRM2 mechanism claim. The same knockout costs resting
heart rate 5.4%, PR interval 4.1% and atrial fibrillation 1.0%: the closer a trait is
to vagal control of the sinoatrial node, the more it loses.

Educational attainment loses 5.0% to a four-gene vagal knockout, so part of its signal
does run through pacemaker genes. Those four are RGS6, CACNA1D and HCN1 — genes with
established, separate roles in brain and in pacemaker function. That is pleiotropy in
the genes themselves, not a flaw in the cell annotation.

What survives as the defensible statement:
  * the control is not generic polygenicity          (rheumatoid arthritis, #35/60)
  * the control is not the neural-cell programme     (refuted here)
  * the control runs on a nearly disjoint gene set   (Jaccard 0.02 vs HRV)
  * a small shared component is real and is pleiotropy at the gene level
""")
