"""
Which cardiac GWAS exist that we never tried?
=============================================

The whole project tested resting heart rate as a proxy for pacemaker function. That
choice deserves scrutiny: resting heart rate is a physiological OUTPUT shaped by
contractility, fitness, autonomic tone and thyroid status, and its GWAS is known to
enrich for cardiac structure genes. Finding that its heritability sits in working
myocardium is arguably the correct answer to the question we asked — just not the
question we wanted answered.

The disease of the sinoatrial node is sick sinus syndrome. The disease of the
atrioventricular node is heart block. The ventricular conduction system has its own
readout in QRS duration. None of these were tried.

This script asks the GWAS Catalog what is actually available for each, and reports
sample size and association count so a trait can be judged on power before any
compute is spent — the mistake made once already with schizophrenia, where a study
was picked on total N while its case count was tiny.

Usage:  python scripts/80_search_traits.py
"""

import json
import time
import urllib.parse
import urllib.request

API = "https://www.ebi.ac.uk/gwas/rest/api"

QUERIES = [
    ("sick sinus syndrome", "窦房结病变本身的疾病 — 最该定位到窦房结"),
    ("sinoatrial node disease", "同上，别名"),
    ("bradyarrhythmia", "缓慢性心律失常"),
    ("atrioventricular block", "房室传导阻滞 — 房室结/His束疾病"),
    ("pacemaker implantation", "起搏器植入 — 传导系统衰竭的硬终点"),
    ("QRS duration", "心室内传导时间 — His-Purkinje 系统"),
    ("QT interval", "心室复极"),
    ("Brugada syndrome", "右室流出道传导异常"),
    ("heart rate response to exercise", "自主神经调控，比静息心率更专一"),
    ("heart rate recovery", "迷走再激活"),
    ("atrial flutter", "峡部依赖折返 — 解剖定位明确"),
    ("supraventricular tachycardia", "室上速"),
    ("heart failure", "对照用的大性状"),
    ("dilated cardiomyopathy", "心肌病"),
]


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def search(term):
    u = (f"{API}/studies/search/findByEfoTrait?"
         f"efoTrait={urllib.parse.quote(term)}&size=100")
    try:
        d = get(u)
    except Exception:
        return []
    return d.get("_embedded", {}).get("studies", [])


def by_text(term):
    """Fall back to the free-text study search when the EFO label does not match."""
    u = (f"{API}/studies/search/findByDiseaseTrait?"
         f"diseaseTrait={urllib.parse.quote(term)}&size=100")
    try:
        d = get(u)
    except Exception:
        return []
    return d.get("_embedded", {}).get("studies", [])


def size_of(st):
    n = 0
    cases = None
    for a in st.get("ancestries", []) or []:
        if a.get("type") == "initial":
            n += a.get("numberOfIndividuals") or 0
    for c in st.get("diseaseTrait", {}) or {}:
        pass
    return n, cases


print("=" * 100)
print("GWAS Catalog — cardiac conduction traits never tested in this project")
print("=" * 100)

found = {}
for term, why in QUERIES:
    studies = search(term) or by_text(term)
    time.sleep(0.4)
    if not studies:
        print(f"\n### {term:<38} — 无匹配研究   [{why}]")
        continue
    rows = []
    for st in studies:
        n, _ = size_of(st)
        rows.append(dict(
            acc=st.get("accessionId"),
            trait=(st.get("diseaseTrait") or {}).get("trait", ""),
            n=n,
            assoc=st.get("associationCount") or 0,
            pmid=(st.get("publicationInfo") or {}).get("pubmedId"),
            year=((st.get("publicationInfo") or {}).get("publicationDate") or "")[:4],
            ss=bool(st.get("fullSummaryStatistics")),
        ))
    rows.sort(key=lambda r: (-r["assoc"], -r["n"]))
    print(f"\n### {term:<38} [{why}]")
    print(f"{'accession':<16}{'N':>11}{'assoc':>7}{'year':>6}{'sumstats':>10}  trait")
    for r in rows[:5]:
        mark = "YES" if r["ss"] else "no"
        print(f"{r['acc']:<16}{r['n']:>11,}{r['assoc']:>7}{r['year']:>6}{mark:>10}  "
              f"{r['trait'][:44]}")
    keep = [r for r in rows if r["ss"] and r["assoc"] >= 3]
    if keep:
        found[term] = keep[:3]

print("\n" + "=" * 100)
print("USABLE (full summary statistics available AND >=3 reported associations)")
print("=" * 100)
if not found:
    print("none — every candidate lacks downloadable summary statistics")
for term, rows in found.items():
    print(f"\n{term}")
    for r in rows:
        print(f"   {r['acc']}  N={r['n']:,}  assoc={r['assoc']}  "
              f"PMID {r['pmid']}  {r['trait'][:50]}")

with open("results/trait_search.json", "w") as f:
    json.dump(found, f, indent=2)
print("\nwrote results/trait_search.json")
