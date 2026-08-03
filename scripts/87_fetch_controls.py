"""
Two controls the design is missing, and why the current ones are not enough.
============================================================================

The scDRS preview exposed a flaw in the control design that the spatial analysis never
had to face.

Educational attainment was chosen as the non-cardiac control. At compartment level that
was fine. At CELL level it is not: its top-ranked cell states in heart tissue are
NC1_glial and NC2_glial_NGF+ — it behaves as the neural trait it is, correctly. And
sinoatrial pacemaker cells are the most neuron-like cells in the heart: highly
excitable, with an ion-channel repertoire overlapping neurons. So educational
attainment ranking SAN_P_cell #2 is not obviously an artefact — it may be real neural
biology landing on a neuron-like cell. Either way it cannot adjudicate whether a
cardiac trait's interest in pacemaker cells is specific.

  HEIGHT       a negative control that is non-cardiac AND non-neural, and about as
               polygenic as a trait gets. If pacemaker cells rank high for height too,
               the ranking is structural. If height ignores them while heart-rate
               variability puts them first, the signal is cardiac.

  RHEUMATOID   a SECOND positive control, aimed at a completely different cell class.
  ARTHRITIS    The existing one — atrial fibrillation — lands on cardiomyocytes, but
               its top hits are ventricular rather than atrial, so it demonstrates
               only that the pipeline finds "cardiomyocytes", not that it resolves
               cell types finely. This tissue is full of T, B, NK and myeloid cells.
               An autoimmune trait must land on those and nowhere near pacemaker
               cells. That is a far sharper test of resolution than one myocyte
               subtype versus another.

Both are large, European-ancestry, well-powered, and have full summary statistics.

Usage:  python scripts/87_fetch_controls.py
"""

import gzip
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent.as_posix()
RAW = f"{ROOT}/data/gwas"
OUT = f"{ROOT}/data/gwas_gsmap"
os.makedirs(RAW, exist_ok=True)
FTP = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"

TARGETS = [
    ("Height", "GCST90245848", "GCST90245001-GCST90246000",
     "非心血管且非神经的阴性对照 — 极度多基因"),
    ("RheumatoidArthritis", "GCST90132223", "GCST90132001-GCST90133000",
     "第二个阳性对照 — 必须命中免疫细胞，不得靠近起搏细胞"),
]

RS = ["rs_id", "rsid", "variant_id", "SNP", "snp"]
EA = ["effect_allele", "EA", "A1"]
OA = ["other_allele", "OA", "A2"]
BE = ["beta", "BETA", "effect_size", "b"]
ORC = ["odds_ratio", "OR"]
SE = ["standard_error", "SE", "se"]
PV = ["p_value", "P", "pval", "pvalue"]
NC = ["n", "N", "sample_size", "n_total"]


def pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


report = {}
for name, acc, bucket, why in TARGETS:
    target = f"{OUT}/{name}.sumstats.gz"
    if os.path.exists(target):
        print(f"[skip] {name} already present")
        continue
    print(f"\n{'=' * 88}\n=== {name}  ({acc})\n    {why}", flush=True)
    url = f"{FTP}/{bucket}/{acc}/"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            html = r.read().decode("utf-8", "replace")
        files = [f for f in re.findall(r'href="([^"?/][^"]*)"', html)
                 if f.endswith((".tsv.gz", ".txt.gz", ".tsv", ".txt"))]
    except Exception as e:
        print(f"    !! listing failed: {type(e).__name__} — {url}")
        report[name] = dict(status="rejected", reason="listing failed")
        continue
    if not files:
        print(f"    !! no data file at {url}")
        report[name] = dict(status="rejected", reason="no data file")
        continue

    raw = f"{RAW}/{acc}_ctrl.tsv.gz"
    print(f"    downloading {files[0]} ...", flush=True)
    rc = subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "5",
                         "--speed-limit", "50000", "--speed-time", "60",
                         "-s", "-o", raw, url + files[0]]).returncode
    if rc != 0:
        print(f"    !! download failed ({rc})")
        report[name] = dict(status="rejected", reason=f"curl {rc}")
        continue
    print(f"    {os.path.getsize(raw) / 1e6:.0f} MB", flush=True)

    try:
        with gzip.open(raw, "rt") as f:
            head = next(f)
    except Exception:
        with open(raw) as f:
            head = next(f)
    sep = "\t" if "\t" in head else ("," if "," in head else r"\s+")
    cols = pd.read_csv(raw, sep=sep, nrows=5000, low_memory=False).columns.tolist()
    print(f"    columns: {', '.join(cols[:10])}")

    c_rs, c_be, c_or = pick(cols, RS), pick(cols, BE), pick(cols, ORC)
    c_se, c_p, c_n = pick(cols, SE), pick(cols, PV), pick(cols, NC)
    c_ea, c_oa = pick(cols, EA), pick(cols, OA)
    if c_rs is None or (c_be is None and c_or is None) or (c_se is None and c_p is None):
        print("    !! REJECTED: missing an essential column")
        os.remove(raw)
        report[name] = dict(status="rejected", reason="missing columns")
        continue

    use = list(dict.fromkeys([c for c in [c_rs, c_ea, c_oa, c_be, c_or, c_se,
                                          c_p, c_n] if c]))
    parts = []
    for ch in pd.read_csv(raw, sep=sep, usecols=use, chunksize=2_000_000,
                          low_memory=False):
        ch = ch[ch[c_rs].astype(str).str.startswith("rs")]
        if ch.empty:
            continue
        if c_be:
            beta = pd.to_numeric(ch[c_be], errors="coerce").to_numpy(dtype=float)
        else:
            o = pd.to_numeric(ch[c_or], errors="coerce").to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                beta = np.log(np.where(o > 0, o, np.nan))
        if c_p:
            pv = np.clip(pd.to_numeric(ch[c_p], errors="coerce")
                         .to_numpy(dtype=float), 1e-300, 1.0)
            z = np.sign(beta) * norm.isf(pv / 2.0)
        else:
            se = pd.to_numeric(ch[c_se], errors="coerce").to_numpy(dtype=float)
            z = np.where(se > 0, beta / se, np.nan)
        parts.append(pd.DataFrame({
            "SNP": ch[c_rs].astype(str).to_numpy(),
            "A1": (ch[c_ea].astype(str).str.upper().to_numpy() if c_ea else "A"),
            "A2": (ch[c_oa].astype(str).str.upper().to_numpy() if c_oa else "G"),
            "Z": z,
            "N": (pd.to_numeric(ch[c_n], errors="coerce").to_numpy()
                  if c_n else np.nan)}))
    if not parts:
        print("    !! REJECTED: nothing usable")
        os.remove(raw)
        report[name] = dict(status="rejected", reason="no usable rows")
        continue
    d = pd.concat(parts, ignore_index=True)
    d = d[np.isfinite(d.Z)]
    n_gws = int((d.Z.abs() > 5.45).sum())
    lam = float(np.median(d.Z ** 2) / 0.4549)
    print(f"    {len(d):,} SNPs, |Z|max {d.Z.abs().max():.1f}, "
          f"gw-sig {n_gws:,}, lambda_GC {lam:.3f}")
    if n_gws < 500:
        print(f"    !! REJECTED: {n_gws} genome-wide significant SNPs — a control "
              "must be well powered or it proves nothing either way")
        os.remove(raw)
        report[name] = dict(status="rejected", reason=f"only {n_gws} gw-sig")
        continue
    d["N"] = int(d.N.median()) if d.N.notna().any() else 250000
    d.drop_duplicates(subset="SNP").to_csv(target, sep="\t", index=False,
                                           compression="gzip", float_format="%.6g")
    print(f"    -> {name}.sumstats.gz  ACCEPTED")
    report[name] = dict(status="accepted", n_snp=int(len(d)), n_gws=n_gws,
                        lambda_gc=lam, accession=acc)
    os.remove(raw)

print("\nSUMMARY")
for k, v in report.items():
    print(f"  {v['status']:<9}{k:<24}{v.get('reason', '')}"
          f"{v.get('n_gws', '')}" + (" gw-sig" if v["status"] == "accepted" else ""))
p = f"{ROOT}/results/control_traits.json"
with open(p, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\nwrote {p}")
