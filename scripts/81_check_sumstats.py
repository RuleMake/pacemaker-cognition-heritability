"""
Do the untried conduction traits actually have downloadable summary statistics?
==============================================================================

The catalogue's study-search endpoint returns `fullSummaryStatistics: false` and
`associationCount: 0` for everything, so it cannot be used to judge availability.
The authoritative answer is the FTP tree: a study with summary statistics has a
directory at

    .../summary_statistics/GCST<lo>-GCST<hi>/<accession>/

This checks each candidate directly and reports the file that would be downloaded,
so a trait can be ruled in or out before any compute is committed.

The list is ordered by how directly the trait interrogates the conduction system:
sick sinus syndrome IS sinoatrial node failure, AV block IS atrioventricular node
failure, QRS duration measures His-Purkinje conduction. Resting heart rate — the
trait the whole project has been built on — is a downstream physiological output and
sits at the bottom of that ordering.

Usage:  python scripts/81_check_sumstats.py
"""

import json
import re
import urllib.error
import urllib.request

FTP = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"

CANDIDATES = [
    # accession,        label,                          why it matters
    ("GCST90225551", "Sick sinus syndrome (N=937,235)",
     "窦房结衰竭本身 — 若有信号必在窦房结"),
    ("GCST90084003", "Sick sinus syndrome ICD10 I49.5",
     "备选，UKB PheWAS"),
    ("GCST90480156", "AV block (PheCode 426.2, N=617,488)",
     "房室结衰竭 — 对应 AVN 切片"),
    ("GCST90480154", "Complete AV block (N=630,658)",
     "完全性阻滞，表型更纯"),
    ("GCST90179161", "QRS duration (N=252,730)",
     "His-Purkinje 传导时间"),
    ("GCST90179153", "QT interval (N=252,730)",
     "心室复极 — 应定位到心室肌，可作阳性对照"),
    ("GCST90480169", "Atrial flutter (PheCode 427.22)",
     "峡部依赖折返，解剖定位明确"),
    ("GCST90086158", "Brugada syndrome (N=12,821)",
     "右室流出道传导"),
    ("GCST005787",   "Heart rate response to exercise",
     "自主神经，比静息心率专一"),
    ("GCST90162626", "Heart failure (N=1,665,481)",
     "大性状，作阳性对照"),
    ("GCST90018834", "Dilated cardiomyopathy (N=533,543)",
     "心肌病 — 应定位到心室肌"),
]


def bucket(acc):
    n = int(re.sub(r"\D", "", acc))
    lo = ((n - 1) // 1000) * 1000 + 1
    return f"GCST{lo}-GCST{lo + 999}"


def listing(url):
    try:
        with urllib.request.urlopen(url, timeout=45) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return f"__HTTP{e.code}__"
    except Exception as e:
        return f"__ERR:{type(e).__name__}__"


print("=" * 104)
print("Summary-statistics availability on the GWAS Catalog FTP")
print("=" * 104)
print(f"{'accession':<15}{'status':<12}{'file':<52}{'MB':>7}")

ok = {}
for acc, label, why in CANDIDATES:
    url = f"{FTP}/{bucket(acc)}/{acc}/"
    html = listing(url)
    if html.startswith("__"):
        print(f"{acc:<15}{'ABSENT':<12}{html:<52}{'-':>7}")
        continue
    files = re.findall(r'href="([^"?/][^"]*)"', html)
    files = [f for f in files if re.search(r"\.(tsv|txt|gz|h)", f, re.I)]
    if not files:
        print(f"{acc:<15}{'EMPTY':<12}{'directory exists but no data file':<52}{'-':>7}")
        continue
    main = max(files, key=len)
    for f in files:
        if re.search(r"\.(tsv|txt)(\.gz)?$", f, re.I) and "meta" not in f.lower():
            main = f
            break
    size = "?"
    try:
        req = urllib.request.Request(url + main, method="HEAD")
        with urllib.request.urlopen(req, timeout=45) as r:
            cl = r.headers.get("Content-Length")
            if cl:
                size = f"{int(cl) / 1e6:.0f}"
    except Exception:
        pass
    print(f"{acc:<15}{'AVAILABLE':<12}{main[:50]:<52}{size:>7}")
    ok[acc] = dict(label=label, why=why, url=url + main, size_mb=size)

print("\n" + "=" * 104)
print("USABLE TRAITS")
print("=" * 104)
if not ok:
    print("none")
for acc, d in ok.items():
    print(f"\n{acc}  {d['label']}")
    print(f"   {d['why']}")
    print(f"   {d['size_mb']} MB   {d['url']}")

with open("results/sumstats_availability.json", "w") as f:
    json.dump(ok, f, indent=2, ensure_ascii=False)
print(f"\n{len(ok)}/{len(CANDIDATES)} usable -> results/sumstats_availability.json")
