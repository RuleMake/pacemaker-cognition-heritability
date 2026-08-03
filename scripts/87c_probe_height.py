"""Probe known large height GWAS accessions directly on the FTP."""
import re
import urllib.request

FTP = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"

# well-known height studies; the EFO search endpoint returned nothing usable, so these
# are checked against the file server rather than against the catalogue's metadata
CANDS = [
    ("GCST006901", "GIANT 2018 Yengo, N=693,529"),
    ("GCST90029007", "UKB height"),
    ("GCST90104314", "height, multi-ancestry"),
    ("GCST90002409", "height, Sakaue/BBJ"),
    ("GCST90435351", "recent height meta-analysis"),
    ("GCST90308142", "height"),
]


def bucket(acc):
    n = int(re.sub(r"\D", "", acc))
    lo = ((n - 1) // 1000) * 1000 + 1
    return f"GCST{lo}-GCST{lo + 999}"


print(f"{'accession':<16}{'status':<14}{'file':<46}{'MB':>7}")
best = None
for acc, note in CANDS:
    url = f"{FTP}/{bucket(acc)}/{acc}/"
    try:
        with urllib.request.urlopen(url, timeout=45) as r:
            html = r.read().decode("utf-8", "replace")
    except Exception as e:
        code = getattr(e, "code", type(e).__name__)
        print(f"{acc:<16}{'absent':<14}{str(code):<46}{'-':>7}")
        continue
    files = [f for f in re.findall(r'href="([^"?/][^"]*)"', html)
             if f.endswith((".tsv.gz", ".txt.gz", ".gz"))]
    if not files:
        print(f"{acc:<16}{'empty':<14}{'directory exists, no data file':<46}{'-':>7}")
        continue
    f0 = files[0]
    mb = 0
    try:
        req = urllib.request.Request(url + f0, method="HEAD")
        with urllib.request.urlopen(req, timeout=45) as r:
            mb = int(r.headers.get("Content-Length", 0)) / 1e6
    except Exception:
        pass
    print(f"{acc:<16}{'AVAILABLE':<14}{f0[:44]:<46}{mb:>7.0f}")
    if best is None:
        best = (acc, bucket(acc), f0, mb, note)

print()
if best:
    acc, bk, f0, mb, note = best
    print(f"USE  {acc}   {note}")
    print(f"     bucket {bk}   file {f0}   {mb:.0f} MB")
else:
    print("none available — rheumatoid arthritis carries the non-cardiac,")
    print("non-neural control role on its own. It is well powered (34,164 gw-sig)")
    print("and its expected target here is immune cells, so if it ranks pacemaker")
    print("cells low the required inference still holds.")
