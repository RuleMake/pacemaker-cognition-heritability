"""
What genes actually drive the gsMap signal? (the never-inspected diagnostic)
===========================================================================

Everything up to now examined the GEOMETRY of the gsMap output: which compartments
light up, whether the ordering survives a paired contrast, how cell-type abundance
tracks the per-spot p-value. None of it looked at the BIOLOGY — which genes gsMap
itself says are responsible.

gsMap writes that answer for every (section, trait) as a gene diagnostic table. Its
`PCC` column is the correlation, across spots, between a gene's spot-specificity
score and the -log10 p-value. High PCC = this gene's spatial pattern is what makes
the enrichment map look the way it does.

Three tests, in increasing order of severity:

  1. Where do canonical pacemaker genes rank?
     If resting heart rate's spatial signal were executed by pacemaker cells, HCN4 /
     SHOX2 / TBX3 / ISL1 should sit near the top. If they sit in the middle of 15,000
     genes, the "node enrichment" is not pacemaker biology whatever its p-value says.

  2. How much do the top genes differ between a cardiac trait and a non-cardiac
     control run on the SAME spots?
     Overlap should be modest if the signal is trait-specific.

  3. What is the correlation of the entire PCC vector between traits?
     This is the strongest form of the question. If corr(PCC_trait, PCC_control) is
     near 1, gsMap's gene-level output in this tissue is determined by tissue
     structure and is carrying almost no trait-specific information — which would
     explain every negative result obtained so far, and is itself the finding.

Usage:  python scripts/50_gene_diagnostics.py
"""

from pathlib import Path
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = str(Path(__file__).resolve().parent.parent)
DIAG = f"{ROOT}/results/genediag"
OUT = f"{ROOT}/results"
CONTROL = "EducationalAttainment"

# Canonical human sinoatrial-node / pacemaker genes. Deliberately broad: pacemaking
# channels, the transcription factors that specify the node, and markers reported for
# human SAN_P_cells. Genes absent from the dataset are reported as absent rather than
# silently dropped, because an absent marker is itself diagnostic.
PACEMAKER = [
    # funny current and pacemaker channels
    "HCN1", "HCN4", "CACNA1D", "CACNA1G", "CACNA2D2", "KCNJ3", "KCNJ5", "SLC8A1",
    # node-specifying transcription factors
    "SHOX2", "TBX3", "TBX18", "TBX5", "ISL1", "PITX2",
    # reported human SAN_P_cell markers
    "BMP4", "VSNL1", "RGS6", "GNG11", "SMOC2", "NTM", "CPNE5",
]
# NKX2-5 is deliberately included as an inverted control: the node is NKX2-5-negative.
NEGATIVE_MARKER = ["NKX2-5"]

# Working-myocardium contrast set — if these outrank the pacemaker genes, the signal
# is being carried by ordinary cardiomyocyte biology.
MYOCARDIUM = ["MYH6", "MYH7", "TNNT2", "TTN", "NPPA", "MYL7", "ACTC1", "RYR2", "PLN"]


def parse(path):
    """SAN__HCAHeartST10659160_RestingHeartRate_Gene_Diagnostic_Info.csv"""
    b = os.path.basename(path).replace("_Gene_Diagnostic_Info.csv", "")
    m = re.match(r"((?:SAN|AVN)__[A-Za-z0-9]+)_(.+)$", b)
    return (m.group(1), m.group(2)) if m else (None, None)


files = sorted(glob.glob(f"{DIAG}/*_Gene_Diagnostic_Info.csv"))
print(f"gene diagnostic tables: {len(files)}")

tables = {}
for p in files:
    section, trait = parse(p)
    if section is None:
        continue
    df = pd.read_csv(p)
    df = df.dropna(subset=["PCC"]).drop_duplicates(subset="Gene").set_index("Gene")
    tables[(section, trait)] = df

sections = sorted({s for s, _ in tables})
traits = sorted({t for _, t in tables})
san = [s for s in sections if s.startswith("SAN")]
print(f"sections: {len(sections)} ({len(san)} SAN)   traits: {len(traits)}\n")

report = {}

# ---------------------------------------------------------------- 1. marker ranks
print("=" * 88)
print("1. RANK OF CANONICAL PACEMAKER GENES  (SAN sections, by PCC, out of ~15,400)")
print("=" * 88)


def rank_of(df, gene):
    """1-based rank by descending PCC; None when the gene is not in the table."""
    if gene not in df.index:
        return None
    return int((df.PCC > df.loc[gene, "PCC"]).sum()) + 1


rank_rows = []
for gene in PACEMAKER + NEGATIVE_MARKER + MYOCARDIUM:
    grp = ("pacemaker" if gene in PACEMAKER else
           "node-negative" if gene in NEGATIVE_MARKER else "myocardium")
    for trait in traits:
        rs, ns = [], []
        for s in san:
            df = tables.get((s, trait))
            if df is None:
                continue
            r = rank_of(df, gene)
            if r is None:
                continue
            rs.append(r)
            ns.append(len(df))
        if rs:
            rank_rows.append(dict(gene=gene, group=grp, trait=trait,
                                  mean_rank=float(np.mean(rs)),
                                  pct=float(np.mean(rs) / np.mean(ns) * 100),
                                  n_sections=len(rs)))

rk = pd.DataFrame(rank_rows)
if rk.empty:
    raise SystemExit("no marker genes found in any table")
rk.to_csv(f"{OUT}/genediag_marker_ranks.tsv", sep="\t", index=False)

present = sorted(rk.gene.unique())
missing = [g for g in PACEMAKER + NEGATIVE_MARKER + MYOCARDIUM if g not in present]

show_traits = ["RestingHeartRate", "AtrialFibrillation", "PRinterval",
               "HRV_RMSSDc", CONTROL]
show_traits = [t for t in show_traits if t in traits]

print(f"{'gene':<12}{'group':<14}" + "".join(f"{t[:16]:>18}" for t in show_traits))
for grp in ["pacemaker", "node-negative", "myocardium"]:
    for gene in [g for g in (PACEMAKER + NEGATIVE_MARKER + MYOCARDIUM)
                 if g in present and rk[rk.gene == g].group.iloc[0] == grp]:
        line = f"{gene:<12}{grp:<14}"
        for t in show_traits:
            v = rk[(rk.gene == gene) & (rk.trait == t)]
            line += f"{int(v.mean_rank.iloc[0]):>18,}" if len(v) else f"{'-':>18}"
        print(line)
    print()

if missing:
    print(f"absent from the dataset entirely: {', '.join(missing)}\n")
report["missing_markers"] = missing

# group summary: median percentile of each gene class
print(f"{'class':<16}" + "".join(f"{t[:16]:>18}" for t in show_traits))
for grp in ["pacemaker", "myocardium"]:
    line = f"{grp:<16}"
    for t in show_traits:
        v = rk[(rk.group == grp) & (rk.trait == t)].pct
        line += f"{v.median():>17.1f}%" if len(v) else f"{'-':>18}"
    print(line)
print("(median percentile of the gene class; 50% = no better than a random gene)")

report["marker_percentiles"] = {
    grp: {t: float(rk[(rk.group == grp) & (rk.trait == t)].pct.median())
          for t in traits if len(rk[(rk.group == grp) & (rk.trait == t)])}
    for grp in ["pacemaker", "myocardium"]
}

# ---------------------------------------------------------------- 2. top-gene overlap
print("\n" + "=" * 88)
print(f"2. TOP-100 GENE OVERLAP WITH THE NON-CARDIAC CONTROL  ({CONTROL})")
print("=" * 88)

ov_rows = []
for trait in traits:
    if trait == CONTROL:
        continue
    for s in san:
        a, b = tables.get((s, trait)), tables.get((s, CONTROL))
        if a is None or b is None:
            continue
        ta = set(a.PCC.nlargest(100).index)
        tb = set(b.PCC.nlargest(100).index)
        ov_rows.append(dict(section=s, trait=trait, overlap=len(ta & tb)))

ov = pd.DataFrame(ov_rows)
if not ov.empty:
    ov.to_csv(f"{OUT}/genediag_top100_overlap.tsv", sep="\t", index=False)
    g = ov.groupby("trait").overlap.agg(["mean", "min", "max", "count"])
    print(f"{'trait':<24}{'mean shared':>14}{'min':>8}{'max':>8}{'sections':>10}")
    for t, r in g.iterrows():
        print(f"{t:<24}{r['mean']:>13.0f}%{int(r['min']):>8}{int(r['max']):>8}"
              f"{int(r['count']):>10}")
    print("(out of 100; chance overlap for two random draws from ~15,400 genes is <1)")
    report["top100_overlap"] = {t: float(r["mean"]) for t, r in g.iterrows()}

# ---------------------------------------------------------------- 3. whole-vector corr
print("\n" + "=" * 88)
print("3. CORRELATION OF THE ENTIRE PCC VECTOR BETWEEN TRAITS (same section, same spots)")
print("=" * 88)

cor_rows = []
for s in san:
    have = [t for t in traits if (s, t) in tables]
    for i, t1 in enumerate(have):
        for t2 in have[i + 1:]:
            a, b = tables[(s, t1)], tables[(s, t2)]
            common = a.index.intersection(b.index)
            if len(common) < 5000:
                continue
            r = pearsonr(a.loc[common, "PCC"], b.loc[common, "PCC"])[0]
            rs = spearmanr(a.loc[common, "PCC"], b.loc[common, "PCC"])[0]
            cor_rows.append(dict(section=s, t1=t1, t2=t2, n=len(common),
                                 pearson=float(r), spearman=float(rs)))

cr = pd.DataFrame(cor_rows)
if not cr.empty:
    cr.to_csv(f"{OUT}/genediag_trait_pcc_corr.tsv", sep="\t", index=False)
    m = cr.groupby(["t1", "t2"]).pearson.mean().reset_index()
    piv = m.pivot(index="t1", columns="t2", values="pearson")
    order = [t for t in ["RestingHeartRate", "AtrialFibrillation", "PRinterval",
                         "HRV_RMSSD", "HRV_RMSSDc", "HRV_SDNN", "HRV_SDNNc", CONTROL]
             if t in traits]
    print(f"{'':<20}" + "".join(f"{t[:11]:>13}" for t in order))
    for a in order:
        line = f"{a[:19]:<20}"
        for b in order:
            v = (piv.loc[a, b] if a in piv.index and b in piv.columns else np.nan)
            if not np.isfinite(v):
                v = (piv.loc[b, a] if b in piv.index and a in piv.columns else np.nan)
            line += f"{v:>13.3f}" if np.isfinite(v) else (
                f"{'—':>13}" if a == b else f"{'':>13}")
        print(line)

    vs_ctrl = cr[(cr.t1 == CONTROL) | (cr.t2 == CONTROL)].pearson
    within = cr[(cr.t1 != CONTROL) & (cr.t2 != CONTROL)].pearson
    print(f"\ncardiac trait vs non-cardiac control : r = {vs_ctrl.mean():.3f} "
          f"(range {vs_ctrl.min():.3f}–{vs_ctrl.max():.3f}, n={len(vs_ctrl)})")
    print(f"cardiac trait vs cardiac trait       : r = {within.mean():.3f} "
          f"(range {within.min():.3f}–{within.max():.3f}, n={len(within)})")
    report["pcc_corr_vs_control"] = float(vs_ctrl.mean())
    report["pcc_corr_within_cardiac"] = float(within.mean())

# ---------------------------------------------------------------- 4. what ARE the top genes
print("\n" + "=" * 88)
print("4. COMPARTMENT ANNOTATION OF THE TOP-100 GENES (SAN sections, pooled)")
print("=" * 88)
ann_rows = []
for trait in show_traits:
    c = Counter()
    for s in san:
        df = tables.get((s, trait))
        if df is None:
            continue
        top = df.nlargest(100, "PCC")
        c.update(top["Annotation"].astype(str))
    tot = sum(c.values()) or 1
    ann_rows.append((trait, c, tot))

labels = sorted({k for _, c, _ in ann_rows for k in c})
print(f"{'trait':<24}" + "".join(f"{l[:14]:>16}" for l in labels))
for trait, c, tot in ann_rows:
    print(f"{trait[:23]:<24}" + "".join(f"{c[l] / tot * 100:>15.0f}%" for l in labels))

for trait in show_traits:
    genes = Counter()
    for s in san:
        df = tables.get((s, trait))
        if df is not None:
            genes.update(df.nlargest(20, "PCC").index.tolist())
    top = [g for g, _ in genes.most_common(12)]
    print(f"\n{trait:<22} top genes: {', '.join(top)}")
report["top_genes"] = {
    t: [g for g, _ in Counter(
        [x for s in san if (s, t) in tables
         for x in tables[(s, t)].nlargest(20, "PCC").index]).most_common(12)]
    for t in show_traits}

with open(f"{OUT}/genediag_summary.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nwrote {OUT}/genediag_*.tsv and genediag_summary.json")
