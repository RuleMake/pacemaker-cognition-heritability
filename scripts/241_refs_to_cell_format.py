"""Convert the reference list to AJHG (Cell Press) style.

Three AJHG requirements are handled together, because they interact:
  (a) references are numbered by order of first appearance in the text;
  (b) "et al." is permitted only when a work has 10 or more authors, so shorter
      author lists must be given in full;
  (c) the entry format is Cell, not Vancouver.

Author lists are fetched from CrossRef, falling back to PubMed, and cached, so
the script is idempotent and offline-rerunnable once the cache exists.
"""

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "MANUSCRIPT-v3.md"
CACHE = ROOT / "results" / "reference_metadata.json"
# CrossRef asks callers to identify themselves for the polite pool. Set
# CROSSREF_MAILTO to your own address before running; the default keeps a
# personal address out of the public record.
UA = {"User-Agent": "ajhg-refs/1.0 (mailto:%s)"
      % os.environ.get("CROSSREF_MAILTO", "anonymous@example.org")}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", "replace")


def from_crossref(doi):
    j = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    au = []
    for a in j.get("author", []) or []:
        fam, giv = a.get("family"), a.get("given")
        if fam:
            au.append((fam, giv or ""))
        elif a.get("name"):
            au.append((a["name"], ""))
    d = j.get("issued", {}).get("date-parts", [[None]])[0]
    return {
        "authors": au,
        "year": d[0] if d else None,
        "title": re.sub(r"<[^>]+>", "", (j.get("title") or [""])[0]).strip(),
        "journal": (j.get("short-container-title") or j.get("container-title") or [""])[0],
        "volume": j.get("volume"), "pages": j.get("page"), "doi": j.get("DOI"),
    }


def from_pubmed(pmid):
    j = json.loads(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                       "?db=pubmed&retmode=json&id=" + pmid))["result"][pmid]
    au = []
    for a in j.get("authors", []):
        nm = a.get("name", "")
        parts = nm.rsplit(" ", 1)
        au.append((parts[0], parts[1] if len(parts) == 2 else ""))
    ids = {x.get("idtype"): x.get("value") for x in j.get("articleids", [])}
    return {
        "authors": au, "year": (j.get("pubdate") or "")[:4],
        "title": j.get("title", "").rstrip("."),
        "journal": j.get("source", ""), "volume": j.get("volume"),
        "pages": j.get("pages"), "doi": ids.get("doi", ""),
    }


def initials(given):
    """'Kazumasa' -> 'K.'   'Siew Yen' -> 'S.Y.'   'J.-P.' -> 'J.-P.'"""
    if not given:
        return ""
    out = []
    for part in re.split(r"[\s]+", given.strip()):
        if not part:
            continue
        sub = [p for p in part.split("-") if p]
        out.append("-".join(p[0].upper() + "." for p in sub))
    return "".join(out)


def fmt_authors(au):
    """Cell style. AJHG allows 'et al.' only at 10 authors or more."""
    names = [f"{fam}, {initials(giv)}".rstrip(", ") for fam, giv in au]
    if not names:
        return ""
    if len(names) >= 10:
        return ", ".join(names[:10]) + ", et al."
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + ", and " + names[-1]


def fmt_entry(m, note):
    au = fmt_authors(m["authors"])
    title = m["title"].rstrip(".")
    jr = (m["journal"] or "").replace("&amp;", "and")
    vol = m["volume"] or ""
    pg = (m["pages"] or "").replace("--", "–")
    bits = [f"{au} ({m['year']}).", f"{title}.", f"{jr} {vol}".strip() + ("," if pg else ".")]
    if pg:
        bits.append(pg + ".")
    if m.get("doi"):
        bits.append(f"https://doi.org/{m['doi']}.")
    s = " ".join(b for b in bits if b.strip())
    return s + (f" {note}" if note else "")


def main():
    text = MS.read_text(encoding="utf-8")
    refs = dict(re.findall(r"^\[\^(\d+)\]: (.+)$", text, re.M))
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    for num, raw in refs.items():
        if num in cache:
            continue
        doi = re.search(r"doi:(10\.\S+?)(?:[.\s]|$)", raw)
        pmid = re.search(r"PMID\s*(\d+)", raw)
        try:
            if doi:
                cache[num] = from_crossref(doi.group(1).rstrip("."))
            elif pmid:
                cache[num] = from_pubmed(pmid.group(1))
            else:
                cache[num] = None
        except Exception as e:                     # fall back before giving up
            try:
                cache[num] = from_pubmed(pmid.group(1)) if pmid else None
            except Exception:
                print(f"  [^{num}] FETCH FAILED: {e}", file=sys.stderr)
                cache[num] = None
        time.sleep(0.12)
    CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")

    # accession / stratum notes carried over from the Vancouver entries
    notes = {}
    for num, raw in refs.items():
        n = re.findall(r"(GWAS Catalog GCST\w+[^.)]*)", raw)
        extra = re.findall(r"\(([^()]*(?:stratum|accession|phenotype|QRS duration)[^()]*)\)", raw)
        parts = [x.strip() for x in n + extra]
        if parts:
            notes[num] = "(" + "; ".join(dict.fromkeys(parts)) + ")"

    # order of first appearance, over the text that precedes the reference list
    body = text[: text.index("## References")]
    order, seen = [], set()
    for m in re.finditer(r"\[\^(\d+)\]", body):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            order.append(m.group(1))
    missing = [k for k in refs if k not in seen]
    assert not missing, f"uncited references: {missing}"

    remap = {old: str(i + 1) for i, old in enumerate(order)}

    lines, failed = [], []
    for old in order:
        m = cache.get(old)
        if not m or not m.get("authors"):
            failed.append(old)
            lines.append(f"{remap[old]}. {refs[old]}")
            continue
        lines.append(f"{remap[old]}. " + fmt_entry(m, notes.get(old, "")))

    # renumber the in-text markers in one pass through placeholders
    new_body = re.sub(r"\[\^(\d+)\]", lambda mm: "\x00%s\x01" % remap[mm.group(1)], body)
    new_body = re.sub(r"\x00(\d+)\x01", r"[^\1]", new_body)

    out = new_body + "## References\n\n" + "\n\n".join(lines) + "\n"
    MS.write_text(out, encoding="utf-8")

    print(f"renumbered {len(order)} references by first appearance")
    et_al = sum(1 for l in lines if "et al." in l)
    print(f"entries using 'et al.' (>= 10 authors): {et_al}")
    if failed:
        print("NO METADATA, left in original form:", failed)


if __name__ == "__main__":
    main()
