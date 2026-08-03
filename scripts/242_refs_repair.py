"""Repair pass for 241: refetch the references whose identifier was mis-parsed,
then rebuild the whole Cell-format list from the cache.

The first pass used a non-greedy DOI pattern that stopped at the first internal
period, so any DOI containing one (10.3389/fnins.2019.00710) was truncated and
the lookup failed. Identifiers are re-read here from the pre-format backup,
which still holds the Vancouver entries.
"""

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

conv = import_module("241_refs_to_cell_format")

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "MANUSCRIPT-v3.md"
OLD = ROOT / "MANUSCRIPT-v3-prefmt-backup.md"
CACHE = ROOT / "results" / "reference_metadata.json"

DOI_RE = re.compile(r"doi:\s*(10\.\d{4,9}/[^\s,;]+)", re.I)
PMID_RE = re.compile(r"PMID\s*(\d+)")


def main():
    old_text = OLD.read_text(encoding="utf-8")
    raw = dict(re.findall(r"^\[\^(\d+)\]: (.+)$", old_text, re.M))
    cache = json.loads(CACHE.read_text(encoding="utf-8"))

    fixed = 0
    for num, entry in raw.items():
        if cache.get(num) and cache[num].get("authors"):
            continue
        doi = DOI_RE.search(entry)
        pmid = PMID_RE.search(entry)
        meta = None
        if doi:
            try:
                meta = conv.from_crossref(doi.group(1).rstrip("."))
            except Exception as e:
                print(f"  [^{num}] crossref failed: {e}")
        if (not meta or not meta.get("authors")) and pmid:
            try:
                meta = conv.from_pubmed(pmid.group(1))
            except Exception as e:
                print(f"  [^{num}] pubmed failed: {e}")
        if meta and meta.get("authors"):
            cache[num] = meta
            fixed += 1
        else:
            print(f"  [^{num}] STILL UNRESOLVED: {entry[:90]}")
        time.sleep(0.15)

    CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"refetched {fixed} entries")

    # notes to carry over (accessions, ancestry strata)
    notes = {}
    for num, entry in raw.items():
        n = re.findall(r"(GWAS Catalog GCST\w+[^.)]*)", entry)
        extra = re.findall(r"\(([^()]*(?:stratum|accession|phenotype|QRS duration)[^()]*)\)", entry)
        parts = [x.strip().rstrip(")") for x in n + extra]
        if parts:
            notes[num] = "(" + "; ".join(dict.fromkeys(parts)) + ")"

    # the manuscript is already renumbered; recover new -> old from first-appearance order
    new_text = MS.read_text(encoding="utf-8")
    body = new_text[: new_text.index("## References")]
    order, seen = [], set()
    for m in re.finditer(r"\[\^(\d+)\]", old_text[: old_text.index("## References")]):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            order.append(m.group(1))

    lines, unresolved = [], []
    for i, old in enumerate(order):
        m = cache.get(old)
        if not m or not m.get("authors"):
            unresolved.append(old)
            lines.append(f"{i + 1}. {raw[old]}")
            continue
        lines.append(f"{i + 1}. " + conv.fmt_entry(m, notes.get(old, "")))

    MS.write_text(body + "## References\n\n" + "\n\n".join(lines) + "\n", encoding="utf-8")

    n_etal = sum(1 for l in lines if "et al." in l)
    print(f"rebuilt {len(lines)} entries | 'et al.' used in {n_etal} (>= 10 authors)")
    if unresolved:
        print("unresolved:", unresolved)


if __name__ == "__main__":
    main()
