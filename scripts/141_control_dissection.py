"""
Why does a brain trait light up sinoatrial pacemaker cells?
===========================================================

Educational attainment was chosen as a negative control on the assumption that a
cognitive trait has no business in the sinoatrial node. It ranks #2 there, and after
donor balancing it ranks #1 of 58. That is now the weakest part of the result, and
"unexplained control signal" is exactly the sentence a reviewer stops at.

Two explanations are possible and they have opposite consequences.

  CONFOUND   Pacemaker cells are simply attractive to any polygenic gene set — high
             expression, long genes, whatever it may be — and the cardiac hits are the
             same artefact. If so, the whole result is worth much less.

  SHARED     Pacemaker cells genuinely run a neuronal-like transcriptional programme.
     PROGRAMME  They are electrically excitable, spontaneously depolarising, densely
             innervated cells, and they express ion channels and synaptic machinery
             that elsewhere in the body belong to neurons. A brain trait would then hit
             them for a real reason, through genes that are NOT the ones carrying the
             cardiac signal.

Rheumatoid arthritis already rules out the crude form of CONFOUND: it has more
genome-wide significant loci than any cardiac trait here and ranks the same cells
#56/58. So the question is not "does any gene set hit pacemaker cells" but "is the
brain signal the same signal as the heart signal, or a different one sharing a cell".

That is a question about genes, and it is answerable without rescoring anything.

  1 PROGRAMME   Define the neuronal programme from this dataset's own neural cells,
                not from an external list, so the definition cannot be tuned.
  2 DRIVERS     Decompose each trait's score in SAN_P_cell into per-gene contributions.
  3 OVERLAP     Ask how much the brain drivers and the heart drivers share.
  4 KNOCKOUT    Delete the neuronal programme from every gene set and re-rank. Removing
                genes always costs something, so the loss is measured against random
                gene removals matched on set size and weight — otherwise "it went down"
                means nothing.
  5 RECIPROCAL  Delete the vagal/pacemaker genes instead. If the two signals are
                separable, each knockout should hurt its own trait and spare the other.

Usage:  python scripts/141_control_dissection.py
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent.as_posix()
H5 = f"{ROOT}/data/singlecell/node_subset.h5ad"
GS = f"{ROOT}/data/scdrs/traits.gs"
SC = f"{ROOT}/results/scdrs"

FOCUS = "SAN_P_cell"
TRAITS = ["HRV_RMSSD", "RestingHeartRate", "PRinterval", "AtrialFibrillation",
          "EducationalAttainment", "RheumatoidArthritis"]
CONTROL_TRAITS = {"EducationalAttainment", "RheumatoidArthritis"}
N_PROGRAMME = 300
N_PERM = 200
MIN_CELLS = 20
SEED = 20260801

# The vagal / pacemaker machinery, fixed in advance so the reciprocal knockout cannot
# be assembled after seeing which genes happen to help.
VAGAL = {"RGS6", "CHRM2", "KCNJ3", "KCNJ5", "GNB4", "GIRK1", "GIRK4", "HCN1", "HCN4",
         "CACNA1D", "CACNA1G", "SLC8A1", "SHOX2", "TBX3", "TBX18", "ISL1", "VSNL1",
         "POPDC2", "ADCY5", "ADRB1", "GNAI1", "GNAI2", "GNAO1"}

rng = np.random.default_rng(SEED)
a = ad.read_h5ad(H5)
cs = a.obs["cell_state"].astype(str).to_numpy()
ct = a.obs["cell_type"].astype(str).to_numpy()
X = sp.csr_matrix(a.X) if not sp.issparse(a.X) else a.X.tocsr()
gidx = {g: i for i, g in enumerate(a.var_names)}
counts = pd.Series(cs).value_counts()
usable = set(counts[counts >= MIN_CELLS].index)

print(f"cells {X.shape[0]:,}  genes {X.shape[1]:,}")
print("neural cell states present:",
      ", ".join(f"{k}({v})" for k, v in
                pd.Series(cs[ct == "neural cell"]).value_counts().head(8).items()))

# ============================================================ 1 neuronal programme
print("\n" + "=" * 96)
print("1. NEURONAL PROGRAMME, DEFINED FROM THIS DATASET'S OWN NEURAL CELLS")
print("=" * 96)
neu = ct == "neural cell"
# Exclude the focus cells from both sides: the programme must be defined without
# reference to pacemaker cells, or the later test becomes circular.
other = ~neu & (cs != FOCUS)
mu_n = np.asarray(X[neu].mean(axis=0)).ravel()
mu_o = np.asarray(X[other].mean(axis=0)).ravel()
lfc = np.log2((mu_n + 0.1) / (mu_o + 0.1))
expressed = mu_n > 0.1
lfc[~expressed] = -np.inf
prog_idx = np.argsort(-lfc)[:N_PROGRAMME]
programme = set(np.asarray(a.var_names)[prog_idx])
print(f"n={neu.sum():,} neural cells define a {len(programme)}-gene programme")
print("top: " + ", ".join(list(np.asarray(a.var_names)[prog_idx][:15])))

# ============================================================ helpers
gs = pd.read_csv(GS, sep="\t").set_index("TRAIT")


def geneset(t):
    pairs = [p.split(":") for p in str(gs.loc[t, "GENESET"]).split(",") if ":" in p]
    return [(s, float(w)) for s, w in pairs if s in gidx]


def rank_of(pairs, cell=FOCUS):
    """Rank of `cell` among cell states by weighted mean expression."""
    if len(pairs) < 50:
        return None, None, None
    ii = [gidx[g] for g, _ in pairs]
    w = np.array([wt for _, wt in pairs], dtype=float)
    E = np.asarray(X[:, ii] @ (w / w.sum())).ravel()
    m = pd.Series(E).groupby(cs).mean()
    m = m[m.index.isin(usable)].sort_values(ascending=False)
    if cell not in m.index:
        return None, None, None
    # z of the focus mean against the spread across cell states: a rank alone cannot
    # say whether #1 is a nose ahead or a mile ahead
    z = float((m[cell] - m.mean()) / m.std())
    return list(m.index).index(cell) + 1, len(m), z


# ============================================================ 2+3 drivers, overlap
print("\n" + "=" * 96)
print("2+3. WHAT CARRIES EACH TRAIT'S SIGNAL IN SAN_P_cell")
print("=" * 96)
m_focus = cs == FOCUS
drivers = {}
for t in TRAITS:
    pairs = geneset(t)
    ii = [gidx[g] for g, _ in pairs]
    w = np.array([wt for _, wt in pairs])
    sub = X[:, ii]
    contrib = w * (np.asarray(sub[m_focus].mean(axis=0)).ravel()
                   - np.asarray(sub.mean(axis=0)).ravel())
    order = np.argsort(-contrib)[:50]
    drivers[t] = [pairs[i][0] for i in order]
    top = drivers[t][:10]
    inprog = [g for g in drivers[t] if g in programme]
    invag = [g for g in drivers[t] if g in VAGAL]
    tag = "  <- control" if t in CONTROL_TRAITS else ""
    print(f"\n{t}{tag}")
    print(f"   top drivers : {', '.join(top)}")
    print(f"   neuronal    : {len(inprog):>2}/50  {', '.join(inprog[:8])}")
    print(f"   vagal/pacer : {len(invag):>2}/50  {', '.join(invag[:8])}")

print("\n--- driver overlap (top 50), Jaccard")
print(f"{'':<26}" + "".join(f"{t[:12]:>14}" for t in TRAITS))
for t1 in TRAITS:
    line = f"{t1:<26}"
    for t2 in TRAITS:
        s1, s2 = set(drivers[t1]), set(drivers[t2])
        line += f"{len(s1 & s2) / len(s1 | s2):>14.2f}"
    print(line)

# ============================================================ 4+5 knockouts
print("\n" + "=" * 96)
print("4+5. KNOCKOUTS, AGAINST A SIZE- AND WEIGHT-MATCHED RANDOM NULL")
print("=" * 96)
print("Removing genes always costs signal. The question is whether removing THESE")
print("genes costs more than removing the same number of comparable genes at random.\n")
print(f"{'trait':<26}{'full':>14}{'-neuronal':>26}{'-vagal':>26}")
res = {}
for t in TRAITS:
    pairs = geneset(t)
    r0, n0, z0 = rank_of(pairs)
    line = f"{t:<26}{f'#{r0}/{n0} z{z0:+.1f}':>14}"
    res[t] = dict(full=dict(rank=r0, n=n0, z=z0))

    for label, drop in (("neuronal", programme), ("vagal", VAGAL)):
        kept = [(g, w) for g, w in pairs if g not in drop]
        n_drop = len(pairs) - len(kept)
        if n_drop == 0 or len(kept) < 50:
            line += f"{'n/a':>26}"
            res[t][label] = dict(n_dropped=n_drop)
            continue
        r1, _, z1 = rank_of(kept)

        # matched null: drop the same number of genes, sampled to match the weight
        # distribution of the ones actually dropped, so the comparison is not just
        # "we removed the highest-weighted genes"
        dropped_w = np.array([w for g, w in pairs if g in drop])
        allw = np.array([w for _, w in pairs])
        bins = np.quantile(allw, np.linspace(0, 1, 6))
        want = np.histogram(dropped_w, bins=bins)[0]
        pool = [np.where((allw >= bins[i]) & (allw <= bins[i + 1]))[0]
                for i in range(5)]
        null_z = []
        for _ in range(N_PERM):
            pick = []
            for i, k in enumerate(want):
                if k and len(pool[i]):
                    pick += list(rng.choice(pool[i], size=min(k, len(pool[i])),
                                            replace=False))
            keep = [p for j, p in enumerate(pairs) if j not in set(pick)]
            if len(keep) >= 50:
                null_z.append(rank_of(keep)[2])
        null_z = np.array([v for v in null_z if v is not None])
        # one-sided: how unusual is a loss this large
        p = float((null_z <= z1).mean()) if len(null_z) else np.nan
        delta = z1 - z0
        line += f"{f'#{r1} z{z1:+.1f} d{delta:+.2f} p={p:.3f}':>26}"
        res[t][label] = dict(rank=r1, z=z1, delta=float(delta), p=p,
                             n_dropped=int(n_drop),
                             null_mean=float(null_z.mean()) if len(null_z) else None)
    print(line + ("   <- control" if t in CONTROL_TRAITS else ""))

# ============================================================ verdict
print("\n" + "=" * 96)
print("VERDICT")
print("=" * 96)
edu = res.get("EducationalAttainment", {})
hrv = res.get("HRV_RMSSD", {})


def fmt(d, k):
    x = d.get(k, {})
    if "delta" not in x:
        return "n/a"
    return f"dz {x['delta']:+.2f} (p={x['p']:.3f})"


print(f"  neuronal knockout   EducationalAttainment {fmt(edu, 'neuronal'):<24}"
      f"HRV_RMSSD {fmt(hrv, 'neuronal')}")
print(f"  vagal knockout      EducationalAttainment {fmt(edu, 'vagal'):<24}"
      f"HRV_RMSSD {fmt(hrv, 'vagal')}")
print()


def carries(d, k, frac=0.10):
    """A programme 'carries' a signal only if deleting it removes a real share of it.

    The first version of this test asked only for p < 0.05. That was wrong: with 200
    permutations a 3% loss is easily significant and just as easily irrelevant. A
    programme that accounts for 3% of a score does not explain that score.
    """
    x = d.get(k, {})
    if "delta" not in x or not d.get("full", {}).get("z"):
        return False
    return x["p"] < 0.05 and abs(x["delta"]) / abs(d["full"]["z"]) >= frac


sep = carries(edu, "neuronal") and carries(hrv, "vagal") and not carries(hrv, "neuronal")
print("SEPARABLE: the brain signal rides on the neuronal programme and the heart "
      "signal does not" if sep else
      "The neuronal programme does NOT carry the control signal — the hypothesis "
      "this script was written to test is refuted. Report the driver-gene overlap "
      "instead, which is where the separation actually shows.")

with open(f"{SC}/control_dissection.json", "w") as f:
    json.dump(dict(programme=sorted(programme), drivers=drivers, knockout=res,
                   separable=bool(sep)), f, indent=2, default=float)
print(f"\nwrote {SC}/control_dissection.json")
