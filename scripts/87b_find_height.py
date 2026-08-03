"""
Find a height GWAS that actually has summary statistics on the FTP.
===================================================================

The accession guessed for height did not exist. Rather than guess again, this asks the
catalogue for height studies, then checks each one's FTP directory directly — the REST
API's `fullSummaryStatistics` field is unreliable, as established earlier when it
reported false for every trait including ones with 861 MB files sitting on the server.

Why height specifically: rheumatoid arthritis is now in hand and is non-cardiac,
non-neural and extremely well powered, so it already serves as the sharp control. But
RA has an expected target IN this tissue — its resident immune cells — so it tests
resolution rather than absence. Height has no expected target in heart tissue at all,
which makes it the cleaner test of whether pacemaker cells simply attract polygenic
signal regardless of what the trait is about.

Usage:  python scripts/87b_find_height.py
"""

import json
import re
import urllib.request

API = "https://www.ebi.ac.uk/gwas/rest/api"
FTP = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def bucket(acc):
    n = int(re.sub(r"\D", "", acc))
    lo = ((n - 1) // 1000) * 1000 + 1
    return f"GCST{lo}-GCST{lo + 999}"


def has_sumstats(acc):
    url = f"{FTP}/{bucket(acc)}/{acc}/"
    try:
        with urllib.request.urlopen(url, timeout=45) as r:
            html = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    files = [f for f in re.findall(r'href="([^"?/][^"]*)"', html)
             if f.endswith((".tsv.gz", ".txt.gz"))]
    if not files:
        return None
    try:
        req = urllib.request.Request(url + files[0], method="HEAD")
        with urllib.request.urlopen(req, timeout=45) as r:
            mb = int(r.headers.get("Content-Length", 0)) / 1e6
    except Exception:
        mb = 0
    return url + files[0], mb


cands = []
for term in ["body height", "height"]:
    try:
        d = get(f"{API}/studies/search/findByEfoTrait?efoTrait={term}&size=200")
    except Exception:
        continue
    for st in d.get("_embedded", {}).get("studies", []):
        acc = st.get("accessionId")
        n = sum((a.get("numberOfIndividuals") or 0)
                for a in (st.get("ancestries") or [])
                if a.get("type") == "initial")
        trait = (st.get("diseaseTrait") or {}).get("trait", "")
        if acc and n > 100_000 and "height" in trait.lower():
            cands.append((acc, n, trait))

cands = sorted({c[0]: c for c in cands}.values(), key=lambda x: -x[1])
print(f"height studies with N > 100,000: {len(cands)}\n")
print(f"{'accession':<16}{'N':>12}  trait")
found = None
for acc, n, trait in cands[:15]:
    res = has_sumstats(acc)
    mark = f"{res[1]:.0f} MB" if res else "no sumstats"
    print(f"{acc:<16}{n:>12,}  {trait[:44]:<46}{mark}")
    if res and found is None:
        found = (acc, n, trait, res[0], res[1])

print()
if found:
    acc, n, trait, url, mb = found
    print(f"USE: {acc}  N={n:,}  {trait}")
    print(f"     {mb:.0f} MB  {url}")
    print(f"     bucket {bucket(acc)}")
else:
    print("no height study on the FTP — rheumatoid arthritis alone will have to")
    print("carry the non-cardiac non-neural control role, which it can: it is")
    print("well powered and its expected target in this tissue is immune cells.")
