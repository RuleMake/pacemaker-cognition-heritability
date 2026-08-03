"""
Discriminant validity: does the pipeline give enrichment for ANY trait, or only the right one?
==============================================================================================

The permutation null shows the resting-heart-rate signal beats random gene sets.
It does NOT show the pipeline is trait-specific - a pipeline that returns
"enriched on the node" for every trait would pass that test too.

So: three traits, ONE slide (SAN), all compartments. Same slide throughout, so
nothing here can be explained by batch differences between sections.

    trait                       expectation on the SAN slide
    --------------------------  -----------------------------------------------
    resting heart rate          enriched on `node` (SAN sets the rate)
    atrial fibrillation         enriched on `myocardium_atrial`, NOT on `node`
                                (AF is a disease of atrial myocardium)
    schizophrenia               null everywhere (non-cardiac; brain trait, and the
                                trait gsMap's own paper mapped to brain regions)

A clean result is a diagonal: each cardiac trait lights up its own compartment and
the brain trait lights up nothing.

Efficiency note: for each permutation the per-spot score is computed once and then
evaluated against every compartment, so adding compartments is nearly free.

Usage:  python scripts/05_discriminant_validity.py
"""

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

P_THRESH, CLUMP_KB, WINDOW_KB, MIN_SPOTS = 5e-8, 500_000, 100_000, 50
N_DECILES, N_PERM, MIN_COMPARTMENT = 10, 1000, 200
RNG = np.random.default_rng(0)

TRAITS = {
    "resting_heart_rate": dict(path="data/gwas/RHR_ZhuZ_UKB460K.assoc.gz",
                               expect="node"),
    "atrial_fibrillation": dict(path="data/gwas/AF_GCST90204201.tsv",
                                expect="myocardium_atrial"),
    "schizophrenia": dict(path="data/gwas/SCZ_GCST90018919.tsv.gz",
                          expect="(none - negative control)"),
}

CANON = {
    "resting_heart_rate": ["HCN4", "MYH6", "SCN5A", "CACNA1D", "RGS6", "GNB4"],
    "atrial_fibrillation": ["PITX2", "ZFHX3", "KCNN3", "NEURL1", "CAV1", "TBX5"],
    "schizophrenia": ["DRD2", "GRIN2A", "CACNA1C", "TCF4", "ZNF804A"],
}


# ---------------------------------------------------------------- gene models
def gene_models():
    rows = []
    pn, pt = re.compile(r'gene_name "([^"]+)"'), re.compile(r'gene_type "([^"]+)"')
    with gzip.open(ROOT / "data/resource/gencode_v44lift37.gtf.gz", "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            p = line.split("\t", 9)
            if p[2] != "gene":
                continue
            mt, mn = pt.search(p[8]), pn.search(p[8])
            c = p[0].replace("chr", "")
            if not mt or mt.group(1) != "protein_coding" or not mn:
                continue
            if c not in {str(i) for i in range(1, 23)}:
                continue
            rows.append((mn.group(1), int(c), int(p[3]), int(p[4])))
    return pd.DataFrame(rows, columns=["gene", "chr", "start", "end"]).drop_duplicates("gene")


# ---------------------------------------------------------------- GWAS parsing
CHR_KEYS = ["chr", "chromosome", "chrom", "#chrom"]
POS_KEYS = ["bp", "pos", "position", "base_pair_location", "bp_hg19"]
P_KEYS = ["p", "p_value", "pval", "pvalue", "p-value"]


def pick(cols, keys):
    low = {c.lower().strip(): c for c in cols}
    for k in keys:
        if k in low:
            return low[k]
    for c in cols:                                  # fall back to substring match
        cl = c.lower()
        if any(cl == k or cl.endswith("_" + k) for k in keys):
            return c
    return None


def read_gwas(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        header = f.readline().rstrip("\n")
    sep = "\t" if "\t" in header else r"\s+"
    cols = re.split("\t" if sep == "\t" else r"\s+", header.strip())
    cc, cp, pv = pick(cols, CHR_KEYS), pick(cols, POS_KEYS), pick(cols, P_KEYS)
    if not all([cc, cp, pv]):
        raise ValueError(f"cannot identify columns in {path}: {cols[:12]}")
    df = pd.read_csv(path, sep=sep, usecols=[cc, cp, pv], dtype=str,
                     engine="c" if sep == "\t" else "python")
    df.columns = ["CHR", "BP", "P"]
    df = df[df["CHR"] != "CHR"]
    df["CHR"] = df["CHR"].astype(str).str.replace("chr", "", regex=False)
    df = df[df["CHR"].isin([str(i) for i in range(1, 23)])].copy()
    df["CHR"] = df["CHR"].astype(np.int64)
    for c in ("BP", "P"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["BP", "P"])
    df["BP"] = df["BP"].astype(np.int64)
    return df, (cc, cp, pv)


def trait_genes(path, genes):
    gw, used = read_gwas(ROOT / path)
    sig = gw[gw["P"] < P_THRESH].sort_values("P")
    leads = []
    for c, sub in sig.groupby("CHR", sort=False):
        taken = []
        for bp in sub["BP"].to_numpy():
            if all(abs(bp - t) >= CLUMP_KB for t in taken):
                taken.append(bp)
        leads += [(c, b) for b in taken]
    hits = set()
    for c, sub in pd.DataFrame(leads, columns=["chr", "bp"]).groupby("chr"):
        gc = genes[genes["chr"] == c]
        gs, ge, gn = gc["start"].to_numpy(), gc["end"].to_numpy(), gc["gene"].to_numpy()
        for bp in sub["bp"].to_numpy():
            hits.update(gn[(gs - WINDOW_KB <= bp) & (ge + WINDOW_KB >= bp)].tolist())
    return sorted(hits), len(gw), len(sig), len(leads), used


# ---------------------------------------------------------------- main
print("[1] gene models ...", flush=True)
GENES = gene_models()
print(f"    {len(GENES):,} protein-coding genes", flush=True)

print("[2] loading SAN slide ...", flush=True)
a = sc.read_h5ad(ROOT / "data/spatial/SAN.h5ad")
a.raw = None                                        # else score helpers silently use Ensembl-indexed raw
a.var_names = a.var["feature_name"].astype(str)
a.var_names_make_unique()
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
det = np.asarray((a.X > 0).sum(axis=0)).ravel()
a = a[:, det >= MIN_SPOTS].copy()

X = np.asarray(a.X.todense() if hasattr(a.X, "todense") else a.X, dtype=np.float32)
mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
sd[sd == 0] = 1.0
Z = (X - mu) / sd
gmean = mu.ravel()
del X
POS = {g: i for i, g in enumerate(a.var_names)}
DEC = pd.qcut(pd.Series(gmean), N_DECILES, labels=False, duplicates="drop").to_numpy()

ann = a.obs["annotation_final"].astype(str).to_numpy()
vc = pd.Series(ann).value_counts()
COMPS = [c for c in vc.index if vc[c] >= MIN_COMPARTMENT]
MASKS = {c: (ann == c) for c in COMPS}
print(f"    {a.n_obs:,} spots x {a.n_vars:,} genes", flush=True)
print(f"    compartments tested: {[(c, int(vc[c])) for c in COMPS]}", flush=True)


def aucs(idx):
    s = Z[:, idx].mean(1)
    out = {}
    for c, m in MASKS.items():
        u, _ = mannwhitneyu(s[m], s[~m], alternative="greater")
        out[c] = u / (m.sum() * (~m).sum())
    return out


results = {"slide": "SAN", "n_spots": int(a.n_obs), "n_perm": N_PERM,
           "compartments": {c: int(vc[c]) for c in COMPS}, "traits": {}}

for tname, cfg in TRAITS.items():
    p = ROOT / cfg["path"]
    if not p.exists():
        print(f"\n[!] {tname}: {p.name} not found - skipped", flush=True)
        continue
    print(f"\n[3] {tname} ...", flush=True)
    tg, n_var, n_sig, n_lead, used = trait_genes(cfg["path"], GENES)
    idx = np.array([POS[g] for g in tg if g in POS])
    rec = [g for g in CANON.get(tname, []) if g in set(tg)]
    print(f"    columns used {used}; {n_var:,} variants, {n_sig:,} sig, "
          f"{n_lead} leads -> {len(tg)} genes ({len(idx)} expressed)", flush=True)
    print(f"    canonical recovered: {rec}", flush=True)
    if len(idx) < 30:
        print("    too few genes - skipped", flush=True)
        continue

    obs = aucs(idx)
    want = pd.Series(DEC[idx]).value_counts().to_dict()
    pools = {d: np.setdiff1d(np.where(DEC == d)[0], idx) for d in want}

    null = {c: np.empty(N_PERM) for c in COMPS}
    for i in range(N_PERM):
        pick_idx = np.concatenate([RNG.choice(pools[d], size=min(k, len(pools[d])), replace=False)
                                   for d, k in want.items()])
        for c, v in aucs(pick_idx).items():
            null[c][i] = v
        if (i + 1) % 500 == 0:
            print(f"    perm {i+1}/{N_PERM}", flush=True)

    tr = {"expect": cfg["expect"], "n_genes": int(len(idx)),
          "n_lead_snps": int(n_lead), "canonical_recovered": rec, "compartments": {}}
    for c in COMPS:
        z = (obs[c] - null[c].mean()) / null[c].std()
        tr["compartments"][c] = {
            "auc": float(obs[c]), "null_mean": float(null[c].mean()),
            "null_sd": float(null[c].std()), "z": float(z),
            "emp_p": float((1 + (null[c] >= obs[c]).sum()) / (1 + N_PERM))}
    results["traits"][tname] = tr

(OUT / "discriminant_validity.json").write_text(json.dumps(results, indent=2))

# ---------------------------------------------------------------- report
print("\n" + "=" * 92)
print("DISCRIMINANT VALIDITY - z(enrichment) per compartment, SAN slide")
print("=" * 92)
tn = list(results["traits"].keys())
print(f"{'compartment':<28}{'n':>7}  " + "".join(f"{t[:22]:>24}" for t in tn))
for c in COMPS:
    line = f"{c:<28}{vc[c]:>7}  "
    for t in tn:
        d = results["traits"][t]["compartments"][c]
        star = "***" if d["emp_p"] < 0.001 else "**" if d["emp_p"] < 0.01 else \
               "*" if d["emp_p"] < 0.05 else ""
        line += f"{d['z']:>+20.2f}{star:<4}"
    print(line)
print("-" * 92)
for t in tn:
    print(f"  {t:<24} expected: {results['traits'][t]['expect']}  "
          f"({results['traits'][t]['n_genes']} genes)")
print("=" * 92)
print(f"saved -> {OUT/'discriminant_validity.json'}")
