"""
Correct two intake errors found by the 2026-10-01 MR audit (MR-AUDIT-2026-10-01.md)
===================================================================================

1. QT interval (GCST90165291, Hoffmann 2022): the deposited file's effect and other
   alleles are swapped relative to beta. At every QT locus with an established direction
   the file's Z has the opposite sign (NOS1AP rs12143842-T, KCNE1 D85N rs1805128-T and
   KCNH2 K897T rs1805123-T all prolong QT in the literature). Every QT-outcome MR estimate
   and every genetic correlation with QT therefore had its sign reversed. Fix: negate Z.

2. Rheumatoid arthritis (GCST90132223, Ishigaki 2022, European stratum): the deposit had
   no sample-size column and scripts/87_fetch_controls.py:176 wrote a hard-coded 250,000
   into every row. The study is 22,350 cases + 74,823 controls = 97,173. Under
   beta = Z / sqrt(N) every RA-outcome effect was shrunk by sqrt(97173/250000). Fix: N.

The check that should have caught (1) at intake is built in here and is re-run after the
fix: the signed Z of reference alleles at canonical loci must agree with the literature.
Originals are kept in data/gwas_gsmap/_pre_fix/. The script refuses to run twice.

Usage:  python scripts/261_fix_qt_sign_ra_n.py
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GWAS = ROOT / "data" / "gwas_gsmap"
BACK = GWAS / "_pre_fix"
BACK.mkdir(exist_ok=True)

# (rsID, allele whose increasing effect on the trait is established, source)
KNOWN = {
    "QTinterval": [("rs12143842", "T", "NOS1AP, Arking 2006"),
                   ("rs1805128", "T", "KCNE1 D85N"),
                   ("rs1805123", "T", "KCNH2 K897T: T (Lys) longer than G (Thr)")],
    "RheumatoidArthritis": [("rs2476601", "A", "PTPN22 R620W risk allele")],
    "AtrialFibrillation": [("rs2200733", "T", "PITX2 4q25 risk allele")],
}
RA_N = 97_173


def signed_z(trait, path):
    want = {s for s, _, _ in KNOWN[trait]}
    hit = []
    for ch in pd.read_csv(path, sep="\t", usecols=["SNP", "A1", "A2", "Z", "N"],
                          chunksize=2_000_000):
        s = ch[ch.SNP.isin(want)]
        if len(s):
            hit.append(s)
    d = pd.concat(hit).set_index("SNP")
    out = []
    for snp, allele, src in KNOWN[trait]:
        r = d.loc[snp]
        z = r.Z if r.A1 == allele else -r.Z if r.A2 == allele else np.nan
        out.append((snp, allele, float(z), src))
    return out


def report(trait, path):
    res = signed_z(trait, path)
    for snp, allele, z, src in res:
        print(f"    {trait:<20}{snp:<12}{allele}: Z = {z:+7.2f}   {src}")
    return all(z > 0 for _, _, z, _ in res)


print("--- direction check before the fix")
ok_before = {t: report(t, GWAS / f"{t}.sumstats.gz") for t in KNOWN}
if ok_before["QTinterval"]:
    raise SystemExit("QT is already correctly oriented; refusing to flip it again.")

for t in ["QTinterval", "RheumatoidArthritis"]:
    src = GWAS / f"{t}.sumstats.gz"
    dst = BACK / f"{t}.sumstats.gz"
    if not dst.exists():
        shutil.copy2(src, dst)
    tmp = GWAS / f"{t}.sumstats.tmp.gz"
    first = True
    n_rows = 0
    for ch in pd.read_csv(dst, sep="\t", chunksize=2_000_000):
        if t == "QTinterval":
            ch["Z"] = -ch["Z"]
        else:
            ch["N"] = RA_N
        ch.to_csv(tmp, sep="\t", index=False, mode="w" if first else "a",
                  header=first, compression="gzip")
        first = False
        n_rows += len(ch)
    tmp.replace(src)
    print(f"--- rewrote {t}: {n_rows:,} rows ({'Z negated' if t == 'QTinterval' else f'N = {RA_N:,}'})")

print("--- direction check after the fix")
ok_after = {t: report(t, GWAS / f"{t}.sumstats.gz") for t in KNOWN}
assert all(ok_after.values()), "a canonical locus still points the wrong way"
n = pd.read_csv(GWAS / "RheumatoidArthritis.sumstats.gz", sep="\t", usecols=["N"], nrows=5).N.unique()
assert list(n) == [RA_N], n
print("all canonical loci oriented as published; RA N =", RA_N)
