"""
Fetch and convert the remaining heart-rate-variability GWAS.
===========================================================

Why HRV at all: published GWAS of RESTING HEART RATE enrich for cardiac-structure
genes, whereas HRV GWAS enrich for sinoatrial-node genes. The original hypothesis
was therefore tested with the wrong trait — resting heart rate is the node's
physiological output, but its common-variant signal sits mostly elsewhere.

Four indices, deliberately chosen to separate mechanisms:
  RMSSD            short-term variability, mainly parasympathetic (vagal) tone
  SDNN             overall variability, sympathetic + parasympathetic
  RMSSD corrected  the above with the effect of heart rate itself regressed out
  SDNN corrected   likewise

The corrected versions matter most: HRV correlates strongly with heart rate, so
without correction one cannot tell autonomic modulation of the node apart from
plain rate. If the node hypothesis is right, the corrected indices should localise
most cleanly.

Each file is ~1.5 GB raw. They are downloaded, converted, and the raw file deleted
immediately, so peak disk use stays at one file rather than four.

Usage:  python scripts/12_fetch_hrv_traits.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/gwas"
OUT = ROOT / "data/gwas_gsmap"
OUT.mkdir(parents=True, exist_ok=True)

BASE = ("https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
        "GCST90281001-GCST90282000")

# accession -> (output name, N)
TRAITS = {
    "GCST90281264": ("HRV_RMSSDc", 46075),   # corrected RMSSD
    "GCST90281265": ("HRV_SDNN", 46075),
    "GCST90281266": ("HRV_SDNNc", 46075),    # corrected SDNN
}


def convert(src: Path, name: str, n: int) -> int:
    df = pd.read_csv(src, sep="\t",
                     usecols=["rs_id", "effect_allele", "other_allele",
                              "beta", "standard_error"],
                     dtype={"rs_id": str, "effect_allele": str, "other_allele": str})
    df = df[df.rs_id.astype(str).str.startswith("rs")]
    for c in ("beta", "standard_error"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["beta", "standard_error"])
    df = df[df.standard_error > 0]
    out = pd.DataFrame({
        "SNP": df.rs_id,
        "A1": df.effect_allele.str.upper(),
        "A2": df.other_allele.str.upper(),
        "Z": df.beta / df.standard_error,
        "N": n,
    }).drop_duplicates(subset="SNP")
    if len(out) < 100_000:
        raise SystemExit(f"ABORT {name}: only {len(out):,} SNPs — check columns")
    p = OUT / f"{name}.sumstats.gz"
    out.to_csv(p, sep="\t", index=False, compression="gzip", float_format="%.6g")
    print(f"    -> {p.name}: {len(out):,} SNPs, |Z|max={out.Z.abs().max():.1f}, N={n:,}")
    return len(out)


for acc, (name, n) in TRAITS.items():
    target = OUT / f"{name}.sumstats.gz"
    if target.exists():
        print(f"[skip] {name} already converted")
        continue
    raw = RAW / f"{acc}.tsv"
    print(f"\n=== {name} ({acc})", flush=True)
    print("    downloading ~1.5 GB ...", flush=True)
    r = subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(raw),
                        f"{BASE}/{acc}/{acc}.tsv"])
    if r.returncode != 0 or not raw.exists():
        print(f"    !! download failed for {acc}")
        continue
    print(f"    downloaded {raw.stat().st_size/1e6:.0f} MB, converting ...", flush=True)
    try:
        convert(raw, name, n)
    finally:
        # free the raw file straight away — disk is the binding constraint here
        raw.unlink(missing_ok=True)
        print("    raw file removed")

print("\n=== gwas_gsmap contents ===")
for p in sorted(OUT.glob("*.sumstats.gz")):
    print(f"  {p.stat().st_size/1e6:>6.0f} MB  {p.name}")
