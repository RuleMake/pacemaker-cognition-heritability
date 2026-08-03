"""
PILOT / go-no-go test for the proposed study
============================================

Question
--------
Does the common-variant genetic signal for RESTING HEART RATE concentrate, in space,
on the sinoatrial node (SAN) - the structure that physiologically sets heart rate?

If yes, the full gsMap analysis (which does this properly, via S-LDSC on genome-wide
signal rather than a lead-SNP gene set) is very likely to work and the project stands.
If no, the premise needs rethinking before investing in the WSL2/gsMap setup.

This is deliberately a CRUDE proxy for gsMap: lead SNP -> nearby genes -> per-spot
expression score. A feasibility probe, not the final analysis.

Design
------
trait     : resting heart rate (Zhu et al., UKB n=458,969, GCST007609, GRCh37)
tissue    : Kanemaru et al. Nature 2023 Visium - SAN (27,108 spots) and AVN (24,026)
score     : per-spot mean of per-gene z-scores over the trait gene set
readout 1 : node spots vs all other niches (AUC / Mann-Whitney)
readout 2 : Spearman correlation with cell2location SAN pacemaker-cell abundance
null      : random gene sets matched to the trait set's EXPRESSION-DECILE profile,
            which controls for the trait genes simply being higher-expressed
            (higher-expressed genes are less noisy and show cleaner spatial pattern).
specificity: the same test on the AVN slide - atrioventricular, not the pacemaker.

Usage:  python scripts/02_pilot_rhr_san.py
"""

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parent.parent
GWAS = ROOT / "data/gwas/RHR_ZhuZ_UKB460K.assoc.gz"
GTF = ROOT / "data/resource/gencode_v44lift37.gtf.gz"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

P_THRESH = 5e-8
CLUMP_KB = 500_000
WINDOW_KB = 100_000
MIN_SPOTS = 50          # gene must be detected in >= this many spots
N_DECILES = 10
N_PERM = 2000
RNG = np.random.default_rng(0)

CANON = ["HCN4", "HCN1", "MYH6", "TBX3", "SHOX2", "ISL1", "GJA5", "SCN5A",
         "KCNJ3", "CACNA1D", "FOXP2", "RGS6", "GNB4", "NKX2-5", "TBX5", "MYH7"]

# ---------------------------------------------------------------- 1. gene models
print("[1/5] parsing GENCODE GRCh37 gene models ...", flush=True)
rows = []
pat_name = re.compile(r'gene_name "([^"]+)"')
pat_type = re.compile(r'gene_type "([^"]+)"')
with gzip.open(GTF, "rt") as f:
    for line in f:
        if line[0] == "#":
            continue
        p = line.split("\t", 9)
        if p[2] != "gene":
            continue
        mt = pat_type.search(p[8])
        if not mt or mt.group(1) != "protein_coding":
            continue
        mn = pat_name.search(p[8])
        if not mn:
            continue
        c = p[0].replace("chr", "")
        if c not in {str(i) for i in range(1, 23)}:
            continue
        rows.append((mn.group(1), int(c), int(p[3]), int(p[4])))
genes = pd.DataFrame(rows, columns=["gene", "chr", "start", "end"]).drop_duplicates("gene")
print(f"      {len(genes):,} autosomal protein-coding genes", flush=True)

# ---------------------------------------------------------------- 2. GWAS -> genes
print("[2/5] GWAS -> independent leads -> trait gene set ...", flush=True)
gw = pd.read_csv(GWAS, sep=r"\s+", usecols=["CHR", "BP", "P"], dtype=str)
gw = gw[gw["CHR"] != "CHR"]                                  # file concatenates per-chr outputs
gw = gw[gw["CHR"].isin([str(i) for i in range(1, 23)])].copy()
gw["CHR"] = gw["CHR"].astype(np.int64)
gw["BP"] = pd.to_numeric(gw["BP"], errors="coerce")
gw["P"] = pd.to_numeric(gw["P"], errors="coerce")
gw = gw.dropna(subset=["BP", "P"])
gw["BP"] = gw["BP"].astype(np.int64)

sig = gw[gw["P"] < P_THRESH].sort_values("P")
leads = []
for c, sub in sig.groupby("CHR", sort=False):
    taken = []
    for bp in sub["BP"].to_numpy():
        if all(abs(bp - t) >= CLUMP_KB for t in taken):
            taken.append(bp)
    leads += [(c, b) for b in taken]
print(f"      {len(gw):,} variants, {len(sig):,} at P<5e-8, "
      f"{len(leads):,} independent leads", flush=True)

hits = set()
ldf = pd.DataFrame(leads, columns=["chr", "bp"])
for c, sub in ldf.groupby("chr"):
    gc = genes[genes["chr"] == c]
    gs, ge, gn = gc["start"].to_numpy(), gc["end"].to_numpy(), gc["gene"].to_numpy()
    for bp in sub["bp"].to_numpy():
        hits.update(gn[(gs - WINDOW_KB <= bp) & (ge + WINDOW_KB >= bp)].tolist())
trait_genes = sorted(hits)
recovered = [g for g in CANON if g in hits]
print(f"      {len(trait_genes)} trait genes; canonical recovered: {recovered}", flush=True)


# ---------------------------------------------------------------- helpers
def load_slide(name):
    a = sc.read_h5ad(ROOT / f"data/spatial/{name}.h5ad")
    a.raw = None                       # critical: else score-style helpers silently use raw (Ensembl IDs)
    a.var_names = a.var["feature_name"].astype(str)
    a.var_names_make_unique()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    det = np.asarray((a.X > 0).sum(axis=0)).ravel()
    a = a[:, det >= MIN_SPOTS].copy()
    return a


def zmatrix(a):
    """dense per-gene z-scored matrix (spots x genes), float32."""
    X = np.asarray(a.X.todense() if hasattr(a.X, "todense") else a.X, dtype=np.float32)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu.ravel()


def evaluate(Z, idx, is_node, pace):
    s = Z[:, idx].mean(axis=1)
    u, _ = mannwhitneyu(s[is_node], s[~is_node], alternative="greater")
    auc = u / (is_node.sum() * (~is_node).sum())
    rho = spearmanr(s, pace)[0] if pace is not None else np.nan
    return auc, rho, s


def run_slide(name, node_label="node", pace_col="SAN_P_cell"):
    print(f"\n[3/5] {name}: loading and scoring ...", flush=True)
    a = load_slide(name)
    Z, gmean = zmatrix(a)
    gene_pos = {g: i for i, g in enumerate(a.var_names)}
    panel = [g for g in trait_genes if g in gene_pos]
    idx = np.array([gene_pos[g] for g in panel])
    print(f"      {a.n_obs:,} spots x {a.n_vars:,} expressed genes; "
          f"{len(panel)}/{len(trait_genes)} trait genes usable", flush=True)

    ann = a.obs["annotation_final"].astype(str).to_numpy()
    is_node = ann == node_label
    pace = a.obs[pace_col].to_numpy().astype(float) if pace_col in a.obs.columns else None
    print(f"      '{node_label}' spots: {is_node.sum():,}", flush=True)
    if is_node.sum() < 50:
        print("      too few node spots - skipping", flush=True)
        return None

    auc, rho, _ = evaluate(Z, idx, is_node, pace)

    # expression-decile-matched null
    dec = pd.qcut(pd.Series(gmean), N_DECILES, labels=False, duplicates="drop").to_numpy()
    want = pd.Series(dec[idx]).value_counts().to_dict()
    pools = {d: np.setdiff1d(np.where(dec == d)[0], idx) for d in want}
    for d, k in want.items():
        if len(pools[d]) < k:
            print(f"      warn: decile {d} pool {len(pools[d])} < needed {k}", flush=True)

    print(f"[4/5] {name}: {N_PERM} decile-matched permutations ...", flush=True)
    n_auc = np.empty(N_PERM)
    n_rho = np.empty(N_PERM)
    for i in range(N_PERM):
        pick = np.concatenate([
            RNG.choice(pools[d], size=min(k, len(pools[d])), replace=False)
            for d, k in want.items()
        ])
        n_auc[i], n_rho[i], _ = evaluate(Z, pick, is_node, pace)
        if (i + 1) % 500 == 0:
            print(f"      {i+1}/{N_PERM}", flush=True)

    r = {
        "slide": name,
        "n_spots": int(a.n_obs),
        "n_node_spots": int(is_node.sum()),
        "n_genes_scored": int(len(panel)),
        "auc_node_vs_rest": float(auc),
        "null_auc_mean": float(n_auc.mean()),
        "null_auc_sd": float(n_auc.std()),
        "z_auc": float((auc - n_auc.mean()) / n_auc.std()),
        "emp_p_auc": float((1 + (n_auc >= auc).sum()) / (1 + N_PERM)),
    }
    if pace is not None:
        r.update({
            "spearman_vs_pacemaker": float(rho),
            "null_rho_mean": float(n_rho.mean()),
            "null_rho_sd": float(n_rho.std()),
            "z_rho": float((rho - n_rho.mean()) / n_rho.std()),
            "emp_p_rho": float((1 + (n_rho >= rho).sum()) / (1 + N_PERM)),
        })
    del Z, a
    return r


# ---------------------------------------------------------------- run
results = {"trait": "resting heart rate (GCST007609, n=458,969)",
           "n_lead_snps": int(len(leads)),
           "n_trait_genes": int(len(trait_genes)),
           "canonical_recovered": recovered,
           "slides": []}

for nm, pc in [("SAN", "SAN_P_cell"), ("AVN", "SAN_P_cell")]:
    try:
        r = run_slide(nm, pace_col=pc)
        if r:
            results["slides"].append(r)
    except Exception as e:                                   # one slide failing must not kill the run
        print(f"      {nm} FAILED: {type(e).__name__}: {e}", flush=True)

print("\n[5/5] writing results ...", flush=True)
(OUT / "pilot_rhr_san.json").write_text(json.dumps(results, indent=2))

print("\n" + "=" * 70)
print("PILOT RESULT - resting heart rate genetic signal vs cardiac anatomy")
print("=" * 70)
print(f"lead SNPs {results['n_lead_snps']} -> {results['n_trait_genes']} genes; "
      f"canonical recovered: {', '.join(recovered)}")
for r in results["slides"]:
    print(f"\n--- {r['slide']}  ({r['n_spots']:,} spots, "
          f"{r['n_node_spots']:,} node spots, {r['n_genes_scored']} genes) ---")
    print(f"  node vs rest : AUC {r['auc_node_vs_rest']:.4f}  "
          f"null {r['null_auc_mean']:.4f}+/-{r['null_auc_sd']:.4f}  "
          f"z={r['z_auc']:+.1f}  emp p={r['emp_p_auc']:.4f}")
    if "spearman_vs_pacemaker" in r:
        print(f"  vs pacemaker : rho {r['spearman_vs_pacemaker']:+.4f}  "
              f"null {r['null_rho_mean']:+.4f}+/-{r['null_rho_sd']:.4f}  "
              f"z={r['z_rho']:+.1f}  emp p={r['emp_p_rho']:.4f}")
print("=" * 70)
print(f"saved -> {OUT/'pilot_rhr_san.json'}")
