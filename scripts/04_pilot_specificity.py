"""
Corrected specificity arm of the pilot.
=======================================

The first pass compared the AVN slide using `SAN_P_cell`, which is ALL-NaN there:
cell2location was run per region with a region-appropriate reference, so each slide
carries only its own conduction-cell column. It also used only the 246 `node` spots
while the AV conduction system on that slide is `node` + `AV_bundle` (1,565 spots).

This script redoes both slides with slide-appropriate definitions:

  SAN slide : compartment = {node}                 conduction cell = SAN_P_cell
  AVN slide : compartment = {node, AV_bundle}      conduction cell = AVN_bundle_cell

Biological expectation
----------------------
Resting heart rate is set by the SINOATRIAL node. The AV node / His bundle governs
atrioventricular conduction delay (PR interval), not intrinsic rate. So the trait
score should enrich on the SAN and should NOT particularly enrich on the AV
conduction system. A clean null in AVN is the specificity control, not a failure.

Usage:  python scripts/04_pilot_specificity.py
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
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

P_THRESH, CLUMP_KB, WINDOW_KB, MIN_SPOTS = 5e-8, 500_000, 100_000, 50
N_DECILES, N_PERM = 10, 2000
RNG = np.random.default_rng(0)

SLIDES = {
    "SAN": dict(compartment={"node"}, cellcol="SAN_P_cell"),
    "AVN": dict(compartment={"node", "AV_bundle"}, cellcol="AVN_bundle_cell"),
}


def trait_gene_set():
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
    genes = pd.DataFrame(rows, columns=["gene", "chr", "start", "end"]).drop_duplicates("gene")

    gw = pd.read_csv(ROOT / "data/gwas/RHR_ZhuZ_UKB460K.assoc.gz", sep=r"\s+",
                     usecols=["CHR", "BP", "P"], dtype=str)
    gw = gw[gw["CHR"] != "CHR"]
    gw = gw[gw["CHR"].isin([str(i) for i in range(1, 23)])].copy()
    gw["CHR"] = gw["CHR"].astype(np.int64)
    for c in ("BP", "P"):
        gw[c] = pd.to_numeric(gw[c], errors="coerce")
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

    hits = set()
    for c, sub in pd.DataFrame(leads, columns=["chr", "bp"]).groupby("chr"):
        gc = genes[genes["chr"] == c]
        gs, ge, gn = gc["start"].to_numpy(), gc["end"].to_numpy(), gc["gene"].to_numpy()
        for bp in sub["bp"].to_numpy():
            hits.update(gn[(gs - WINDOW_KB <= bp) & (ge + WINDOW_KB >= bp)].tolist())
    return sorted(hits), len(leads)


def run(name, cfg, panel_genes):
    print(f"\n=== {name} ===", flush=True)
    a = sc.read_h5ad(ROOT / f"data/spatial/{name}.h5ad")
    a.raw = None
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

    pos = {g: i for i, g in enumerate(a.var_names)}
    idx = np.array([pos[g] for g in panel_genes if g in pos])

    ann = a.obs["annotation_final"].astype(str).to_numpy()
    is_c = np.isin(ann, list(cfg["compartment"]))
    col = cfg["cellcol"]
    cellab = a.obs[col].to_numpy().astype(float) if col in a.obs.columns else None
    ok = np.isfinite(cellab) if cellab is not None else None

    print(f"  {a.n_obs:,} spots x {a.n_vars:,} genes | {len(idx)} trait genes usable")
    print(f"  compartment {sorted(cfg['compartment'])}: {is_c.sum():,} spots")
    print(f"  {col}: {0 if ok is None else int(ok.sum()):,} finite values", flush=True)

    def ev(ix):
        s = Z[:, ix].mean(1)
        u, _ = mannwhitneyu(s[is_c], s[~is_c], alternative="greater")
        auc = u / (is_c.sum() * (~is_c).sum())
        rho = spearmanr(s[ok], cellab[ok])[0] if (ok is not None and ok.sum() > 100) else np.nan
        return auc, rho

    auc, rho = ev(idx)

    dec = pd.qcut(pd.Series(gmean), N_DECILES, labels=False, duplicates="drop").to_numpy()
    want = pd.Series(dec[idx]).value_counts().to_dict()
    pools = {d: np.setdiff1d(np.where(dec == d)[0], idx) for d in want}

    print(f"  {N_PERM} decile-matched permutations ...", flush=True)
    na, nr = np.empty(N_PERM), np.empty(N_PERM)
    for i in range(N_PERM):
        pick = np.concatenate([RNG.choice(pools[d], size=min(k, len(pools[d])), replace=False)
                               for d, k in want.items()])
        na[i], nr[i] = ev(pick)
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{N_PERM}", flush=True)

    r = {"slide": name, "compartment": sorted(cfg["compartment"]),
         "n_spots": int(a.n_obs), "n_compartment_spots": int(is_c.sum()),
         "n_genes": int(len(idx)),
         "auc": float(auc), "null_auc_mean": float(na.mean()), "null_auc_sd": float(na.std()),
         "z_auc": float((auc - na.mean()) / na.std()),
         "emp_p_auc": float((1 + (na >= auc).sum()) / (1 + N_PERM))}
    if np.isfinite(rho):
        r.update({"cellcol": col, "rho": float(rho),
                  "null_rho_mean": float(nr.mean()), "null_rho_sd": float(nr.std()),
                  "z_rho": float((rho - nr.mean()) / nr.std()),
                  "emp_p_rho": float((1 + (nr >= rho).sum()) / (1 + N_PERM))})
    else:
        r["cellcol"] = f"{col} (unavailable on this slide - not tested)"
    del Z, a
    return r


if __name__ == "__main__":
    print("building trait gene set ...", flush=True)
    panel, n_leads = trait_gene_set()
    print(f"  {n_leads} leads -> {len(panel)} genes", flush=True)

    res = {"trait": "resting heart rate (GCST007609, n=458,969)",
           "n_lead_snps": n_leads, "n_trait_genes": len(panel), "slides": []}
    for nm, cfg in SLIDES.items():
        res["slides"].append(run(nm, cfg, panel))
    (OUT / "pilot_specificity.json").write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 74)
    print("SPECIFICITY RESULT - resting heart rate signal across conduction structures")
    print("=" * 74)
    for r in res["slides"]:
        print(f"\n{r['slide']}  compartment={r['compartment']}  "
              f"({r['n_compartment_spots']:,}/{r['n_spots']:,} spots)")
        print(f"  compartment vs rest : AUC {r['auc']:.4f}  "
              f"null {r['null_auc_mean']:.4f}+/-{r['null_auc_sd']:.4f}  "
              f"z={r['z_auc']:+.2f}  emp p={r['emp_p_auc']:.4f}")
        if "rho" in r:
            print(f"  vs {r['cellcol']:<18}: rho {r['rho']:+.4f}  "
                  f"null {r['null_rho_mean']:+.4f}+/-{r['null_rho_sd']:.4f}  "
                  f"z={r['z_rho']:+.2f}  emp p={r['emp_p_rho']:.4f}")
        else:
            print(f"  vs cell abundance   : {r['cellcol']}")
    print("=" * 74)
