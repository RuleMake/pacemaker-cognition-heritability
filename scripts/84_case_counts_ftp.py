"""
Case and control counts from the FTP metadata, before committing to any download.
=================================================================================

The REST study endpoint does not carry the case/control split, but every summary-
statistics directory on the FTP ships a `*-meta.yaml` beside the data file, and that
file does.

This matters because all the conduction-disease candidates are PheWAS studies with
headline sample sizes above 600,000 while their case counts are unknown. For a binary
trait the usable power is the effective sample size

    N_eff = 4 / (1/cases + 1/controls)

which for 3,000 cases against 614,000 controls is under 12,000 — far below what the
headline suggests. Picking a study on its headline N is exactly the mistake that
produced a useless schizophrenia control earlier in this project, where the file held
one genome-wide significant SNP in 25 million variants.

Traits are ranked by N_eff so the download list is decided on power rather than on
how appealing the phenotype label is.

Usage:  python scripts/84_case_counts_ftp.py
"""

import json
import re
import urllib.error
import urllib.request

FTP = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"

STUDIES = {
    "GCST90480156": "AV block (PheCode 426.2)",
    "GCST90480154": "Complete AV block (PheCode 426.21)",
    "GCST90481992": "Mobitz II AV block",
    "GCST90480169": "Atrial flutter (PheCode 427.22)",
    "GCST90086158": "Brugada syndrome (Barc 2022)",
    "GCST90162626": "Heart failure (Levin 2022)",
    "GCST90018834": "Dilated cardiomyopathy",
    "GCST90084003": "Sick sinus syndrome ICD10 I49.5",
}


def bucket(acc):
    n = int(re.sub(r"\D", "", acc))
    lo = ((n - 1) // 1000) * 1000 + 1
    return f"GCST{lo}-GCST{lo + 999}"


def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=45) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def num(text, *keys):
    for k in keys:
        m = re.search(rf"^{k}\s*:\s*'?\"?([0-9,]+)", text, re.M | re.I)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


rows = []
for acc, label in STUDIES.items():
    base = f"{FTP}/{bucket(acc)}/{acc}/"
    idx = fetch(base) or ""
    meta_name = None
    for f in re.findall(r'href="([^"?/][^"]*)"', idx):
        if f.endswith(".yaml") or f.endswith(".yml"):
            meta_name = f
            break
    txt = fetch(base + meta_name) if meta_name else None
    if not txt:
        rows.append(dict(acc=acc, label=label, cases=None, controls=None,
                         n_eff=None, note="no metadata file"))
        continue
    cases = num(txt, "case_count", "ncase", "n_cases")
    controls = num(txt, "control_count", "ncontrol", "n_controls")
    total = num(txt, "sample_size", "samplesize")
    n_eff = (4.0 / (1.0 / cases + 1.0 / controls)) if cases and controls else None
    rows.append(dict(acc=acc, label=label, cases=cases, controls=controls,
                     total=total, n_eff=n_eff,
                     build=(re.search(r"genome_assembly\s*:\s*(\S+)", txt) or
                            [None, "?"])[1],
                     note=""))

rows.sort(key=lambda r: -(r.get("n_eff") or 0))

print("=" * 104)
print("Case/control split and effective sample size")
print("=" * 104)
print(f"{'accession':<15}{'cases':>9}{'controls':>11}{'N_eff':>10}{'build':>9}"
      f"  trait")
for r in rows:
    e = r.get("n_eff")
    print(f"{r['acc']:<15}{(r['cases'] or '?'):>9}{(r['controls'] or '?'):>11}"
          f"{(f'{e:,.0f}' if e else '?'):>10}{str(r.get('build', '?')):>9}  "
          f"{r['label']}  {r.get('note', '')}")

print("\n" + "=" * 104)
print("RECOMMENDATION")
print("=" * 104)
print("For scDRS the gene set is built from MAGMA Z statistics, which need genuine")
print("association signal; a trait with N_eff below ~20,000 will yield a top-1000 gene")
print("set that is mostly noise and scDRS will correctly find nothing anywhere.\n")
take, skip = [], []
for r in rows:
    e = r.get("n_eff")
    (take if (e and e >= 20000) else skip).append(r)
print("download:")
for r in take:
    print(f"  {r['acc']}  N_eff {r['n_eff']:,.0f}   {r['label']}")
if not take:
    print("  none clear the bar on metadata alone")
print("\nskip or verify by hand:")
for r in skip:
    e = r.get("n_eff")
    print(f"  {r['acc']}  N_eff {(f'{e:,.0f}' if e else 'unknown')}   {r['label']}")

with open("results/case_counts_ftp.json", "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)
print("\nwrote results/case_counts_ftp.json")
