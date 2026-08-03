"""
Find GWAS for the ventricular half of the conduction system.
============================================================

The atlas contains 110 Purkinje cells, in apex tissue that the SAN/AVN subset never
included. That closes the anatomical gap in the conduction axis:

    sinoatrial node  ->  atrioventricular node  ->  His bundle  ->  Purkinje network

and it makes the specificity argument falsifiable in a way it has not been so far.
Every trait tested to date is an atrial or nodal readout, so every trait has had the
same predicted answer. QRS duration is the conduction time through the His-Purkinje
system and should land on Purkinje cells, NOT on the sinoatrial node. QT interval is
ventricular repolarisation and should land on ventricular myocytes, not on conduction
tissue at all. Two traits, two different predicted cells, both different from
everything tested so far — that is a real test, not another confirmation.

This asks the GWAS Catalog what exists with full summary statistics, and reports
sample size so power can be judged before compute is spent.

Usage:  python scripts/145_search_ventricular_traits.py
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.ebi.ac.uk/gwas/rest/api"
ROOT = Path(__file__).resolve().parent.parent.as_posix()

QUERIES = [
    ("cognitive function measurement", "认知功能 —— 扩展 rg 谱"),
    ("intelligence", "智力"),
    ("reaction time", "反应时"),
    ("QRS complex", "同上，别名"),
    ("QT interval", "心室复极 —— 预测落在心室肌，不在传导组织"),
    ("PR interval", "已有，作为检索校准"),
    ("bundle branch block", "束支阻滞 —— 希浦系统疾病本身"),
    ("heart rate variability measurement", "已有，作为检索校准"),
]


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def studies_for(term):
    """EFO label first, then the free-text disease-trait index as a fallback."""
    for path, key in (("findByEfoTrait", "efoTrait"),
                      ("findByDiseaseTrait", "diseaseTrait")):
        u = f"{API}/studies/search/{path}?{key}={urllib.parse.quote(term)}&size=200"
        try:
            d = get(u)
        except Exception:
            continue
        st = d.get("_embedded", {}).get("studies", [])
        if st:
            return st
    return []


def n_of(st):
    return sum((a.get("numberOfIndividuals") or 0)
               for a in (st.get("ancestries") or [])
               if a.get("type") == "initial")


out = {}
for q, why in QUERIES:
    print("=" * 92)
    print(f"{q}   —— {why}")
    studies = studies_for(q)
    time.sleep(0.4)
    if not studies:
        print("  no matching study")
        continue

    rows = [dict(accession=s.get("accessionId"),
                 trait=(s.get("diseaseTrait") or {}).get("trait", "")[:30],
                 n=n_of(s),
                 sample=(s.get("initialSampleSize") or "")[:60],
                 year=((s.get("publicationInfo") or {}).get("publicationDate")
                       or "")[:4],
                 pmid=(s.get("publicationInfo") or {}).get("pubmedId"),
                 # the field is `fullPvalueSet`; `fullSummaryStatistics` does not
                 # exist on this endpoint and silently filtered everything away
                 ss=bool(s.get("fullPvalueSet")))
            for s in studies]
    # full summary statistics are non-negotiable: MAGMA needs every SNP, not the
    # catalogue's curated top hits
    rows = [r for r in rows if r["ss"]]
    rows.sort(key=lambda r: -r["n"])
    if not rows:
        print("  none with full summary statistics")
    print(f"    {'accession':<16}{'year':<6}{'N':>10}  {'trait':<30}sample")
    for r in rows[:12]:
        print(f"    {r['accession']:<16}{r['year']:<6}{r['n']:>10,}  "
              f"{r['trait']:<30}{r['sample']}")
    out[q] = rows

with open(f"{ROOT}/results/ventricular_trait_search.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {ROOT}/results/ventricular_trait_search.json")
