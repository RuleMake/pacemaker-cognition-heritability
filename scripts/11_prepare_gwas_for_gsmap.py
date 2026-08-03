"""
Convert GWAS summary statistics into the SNP/A1/A2/Z/N format gsMap expects.
===========================================================================

gsMap wants:   SNP  A1  A2  Z  N        (SNP = rs number)

Two situations here:

  * AF and EDU come from the GWAS Catalog and already carry rsIDs in `variant_id`.
  * The resting-heart-rate file (Zhu et al., UKB) identifies variants as
    `10:48698435_A_G` - position, not rsID. gsMap can convert positions to rsIDs
    but only with a dbSNP reference (~15 GB), which does not fit on this disk.
    Instead we map against the 1000G EUR Phase3 `.bim` files that ship inside
    gsMap_resource. That loses nothing: gsMap restricts the analysis to variants
    in that panel anyway, so any SNP the mapping drops would have been dropped
    downstream regardless.

Allele handling: a bim record and a GWAS record can list the two alleles in
either order. We accept a match in either orientation and flip the sign of Z
when the effect allele corresponds to the bim's A2, so Z is always expressed
relative to A1 as written out.

Usage:  python scripts/11_prepare_gwas_for_gsmap.py
Output: data/gwas_gsmap/<TRAIT>.sumstats.gz
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BIM = ROOT / "data/resource/bim"
OUT = ROOT / "data/gwas_gsmap"
OUT.mkdir(parents=True, exist_ok=True)

TRAITS = {
    "RestingHeartRate": dict(
        path="data/gwas/RHR_ZhuZ_UKB460K.assoc.gz", n=458969, style="pos",
        cols=dict(chr="CHR", pos="BP", a1="A1", a2="A0", beta="BETA", se="SE", p="P")),
    "AtrialFibrillation": dict(
        path="data/gwas/AF_GCST90204201.tsv", n=2339188, style="rsid",
        cols=dict(snp="variant_id", a1="effect_allele", a2="other_allele",
                  beta="beta", se="standard_error", p="p_value")),
    # NB: this file's `variant_id` is chr_pos_ref_alt ("6_33025792_C_CA"), not an
    # rsID, so it takes the position-mapping path like the heart-rate file.
    "EducationalAttainment": dict(
        path="data/gwas/EDU_GCST90296499.tsv", n=931577, style="pos",
        cols=dict(chr="chromosome", pos="base_pair_location", a1="effect_allele",
                  a2="other_allele", beta="beta", se="standard_error", p="p_value")),
}


def load_bim():
    frames = []
    for c in range(1, 23):
        p = BIM / f"1000G.EUR.QC.{c}.bim"
        if not p.exists():
            continue
        d = pd.read_csv(p, sep=r"\s+", header=None,
                        names=["chr", "snp", "cm", "pos", "a1", "a2"],
                        dtype={"chr": np.int8, "snp": str, "pos": np.int64,
                               "a1": str, "a2": str})
        frames.append(d[["chr", "snp", "pos", "a1", "a2"]])
    b = pd.concat(frames, ignore_index=True)
    b["key"] = b["chr"].astype(str) + ":" + b["pos"].astype(str)
    return b


def read_any(path, usecols):
    p = ROOT / path
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        head = f.readline()
    sep = "\t" if "\t" in head else r"\s+"
    df = pd.read_csv(p, sep=sep, usecols=usecols, dtype=str,
                     engine="c" if sep == "\t" else "python")
    return df


def finish(df, n, name):
    """df has SNP, A1, A2, BETA, SE (+ optional flip already applied)."""
    for c in ("BETA", "SE"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["BETA", "SE"])
    df = df[df["SE"] > 0]
    df["Z"] = df["BETA"] / df["SE"]
    df = df[np.isfinite(df["Z"])]
    df["N"] = n
    out = df[["SNP", "A1", "A2", "Z", "N"]].copy()
    out["A1"] = out["A1"].str.upper()
    out["A2"] = out["A2"].str.upper()
    out = out.drop_duplicates(subset="SNP")
    # A silently-empty or tiny output is the dangerous failure mode here: the file
    # still parses downstream and the analysis just returns nothing meaningful.
    if len(out) < 100_000:
        raise SystemExit(
            f"ABORT {name}: only {len(out):,} SNPs survived - expected >1e5. "
            f"Check the identifier format and column mapping before continuing.")
    p = OUT / f"{name}.sumstats.gz"
    out.to_csv(p, sep="\t", index=False, compression="gzip",
               float_format="%.6g")
    print(f"    -> {p.name}: {len(out):,} SNPs, "
          f"|Z|max={out.Z.abs().max():.1f}, N={n:,}, {p.stat().st_size/1e6:.0f} MB")
    return out


bim = None
for name, cfg in TRAITS.items():
    src = ROOT / cfg["path"]
    if not src.exists():
        print(f"[skip] {name}: {src.name} not found")
        continue
    print(f"\n=== {name} ===", flush=True)
    c = cfg["cols"]

    if cfg["style"] == "rsid":
        df = read_any(cfg["path"], [c["snp"], c["a1"], c["a2"], c["beta"], c["se"]])
        df.columns = ["SNP", "A1", "A2", "BETA", "SE"]
        df = df[df["SNP"].astype(str).str.startswith("rs")]
        print(f"    {len(df):,} variants with rsIDs")
        finish(df, cfg["n"], name)

    else:  # position-keyed -> map through the 1000G bim
        if bim is None:
            print("    loading 1000G EUR bim ...", flush=True)
            bim = load_bim()
            print(f"    {len(bim):,} reference SNPs", flush=True)
        df = read_any(cfg["path"], [c["chr"], c["pos"], c["a1"], c["a2"],
                                    c["beta"], c["se"]])
        df.columns = ["CHR", "POS", "A1", "A2", "BETA", "SE"]
        df = df[df["CHR"] != "CHR"]
        df["CHR"] = pd.to_numeric(df["CHR"], errors="coerce")
        df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
        df = df.dropna(subset=["CHR", "POS"])
        df = df[df["CHR"].between(1, 22)]
        df["key"] = df["CHR"].astype(np.int64).astype(str) + ":" + \
                    df["POS"].astype(np.int64).astype(str)
        n_in = len(df)
        m = df.merge(bim[["key", "snp", "a1", "a2"]], on="key", how="inner")
        print(f"    {n_in:,} variants -> {len(m):,} matched to the reference panel "
              f"({100*len(m)/n_in:.1f}%)")

        gu1, gu2 = m["A1"].str.upper(), m["A2"].str.upper()
        bu1, bu2 = m["a1"].str.upper(), m["a2"].str.upper()
        same = (gu1 == bu1) & (gu2 == bu2)
        flip = (gu1 == bu2) & (gu2 == bu1)
        keep = same | flip
        print(f"    allele-consistent: {keep.sum():,} "
              f"(same {same.sum():,}, flipped {flip.sum():,}); "
              f"dropped {int((~keep).sum()):,}")
        m = m[keep].copy()
        m["BETA"] = pd.to_numeric(m["BETA"], errors="coerce")
        m.loc[flip[keep].to_numpy(), "BETA"] *= -1     # express effect w.r.t. bim A1
        m["SNP"], m["A1"], m["A2"] = m["snp"], bu1[keep].to_numpy(), bu2[keep].to_numpy()
        m.loc[flip[keep].to_numpy(), "A1"] = bu1[keep][flip[keep]].to_numpy()
        finish(m[["SNP", "A1", "A2", "BETA", "SE"]], cfg["n"], name)

print(f"\nall outputs in {OUT}")
for p in sorted(OUT.glob("*.sumstats.gz")):
    with gzip.open(p, "rt") as f:
        print(f"\n--- {p.name}")
        for i, line in enumerate(f):
            print("   ", line.rstrip())
            if i >= 3:
                break
