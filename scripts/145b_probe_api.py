"""Probe what the GWAS Catalog study record actually contains, before trusting a field.

The first pass filtered on `fullSummaryStatistics` and returned nothing even for PR
interval and HRV, which this project has already downloaded and analysed. When a
calibration query that must succeed returns empty, the filter is wrong, not the world.
"""

import json
import urllib.parse
import urllib.request

API = "https://www.ebi.ac.uk/gwas/rest/api"


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


for term in ["QRS duration", "PR interval"]:
    for path, key in (("findByEfoTrait", "efoTrait"),
                      ("findByDiseaseTrait", "diseaseTrait")):
        u = f"{API}/studies/search/{path}?{key}={urllib.parse.quote(term)}&size=5"
        try:
            d = get(u)
        except Exception as e:
            print(f"{term} via {path}: {e}")
            continue
        st = d.get("_embedded", {}).get("studies", [])
        print(f"\n=== {term} via {path}: {len(st)} studies")
        if st:
            s = st[0]
            print("keys:", sorted(s.keys()))
            for k in ["accessionId", "fullSummaryStatistics", "summaryStatisticsLocation",
                      "associationCount", "initialSampleSize", "diseaseTrait"]:
                print(f"   {k} = {json.dumps(s.get(k))[:120]}")
            break
