"""Fix consortium handling in the reference list.

CrossRef returns group entries in the same array as people, and for some
consortium papers it also returns structural labels ("Corresponding authors",
"Steering committee") as if they were authors. The first pass therefore put
consortium names in the first-author position and, for 1000 Genomes, printed
two section headings as names.

Here personal authors are separated from group names, structural labels are
dropped, duplicated names are collapsed, and any consortium is placed after the
personal authors. The 10-author threshold for "et al." counts people only.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

conv = import_module("241_refs_to_cell_format")

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "MANUSCRIPT-v3.md"
OLD = ROOT / "MANUSCRIPT-v3-prefmt-backup.md"
CACHE = ROOT / "results" / "reference_metadata.json"

# section headings CrossRef sometimes emits inside the author array
LABELS = {"corresponding authors", "corresponding author", "steering committee",
          "production group", "writing group", "analysis group", "management committee",
          "data coordination center", "sequencing group", "consortium coordination"}

GROUP_WORDS = ("consortium", "group", "team", "committee", "network", "study",
               "project", "investigators", "collaboration", "cogent")

# CrossRef also emits contributing institutions and sampling sites in the author
# array. Only names that denote an actual study group are kept as authors.
REAL_GROUP = re.compile(
    r"\b(Consortium|Research Team|Working Group|Study Group|Cohort Study|"
    r"Genetics Cent(?:er|re)|Project)\b", re.I)
MAX_GROUPS = 3


def split_authors(authors):
    """Return (people, groups) with labels and institutions dropped, repeats collapsed."""
    people, groups, seen = [], [], set()
    for fam, giv in authors:
        fam = (fam or "").strip()
        giv = (giv or "").strip()
        low = fam.lower()
        if low in LABELS:
            continue
        is_group = not giv and (any(w in low for w in GROUP_WORDS) or " " in fam)
        if is_group:
            if REAL_GROUP.search(fam) and fam not in groups:
                groups.append(fam)
            continue
        key = (fam.lower(), giv.lower())
        if key in seen:
            continue
        seen.add(key)
        people.append((fam, giv))
    if not people:                     # a purely corporate author, e.g. the Task Force document
        groups = [f for f, g in authors if not (g or "").strip()][:1]
    return people, groups[:MAX_GROUPS]


def fmt_authors(authors):
    people, groups = split_authors(authors)
    names = [f"{f}, {conv.initials(g)}".rstrip(", ") for f, g in people]
    if names:
        if len(names) >= 10:
            base = ", ".join(names[:10]) + ", et al."
        elif len(names) == 1:
            base = names[0]
        else:
            base = ", ".join(names[:-1]) + ", and " + names[-1]
    else:
        base = ""
    if groups:
        joined = "; ".join(groups)
        base = f"{base}; {joined}" if base else joined
    return base


def clean(s):
    s = (s or "")
    for a, b in (("&gt;", ">"), ("&lt;", "<"), ("&amp;", "and"), ("&#x2019;", "'")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def fmt_entry(m, note):
    au = fmt_authors(m["authors"])
    title = clean(m["title"]).rstrip(".")
    jr = clean(m["journal"])
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
    old_text = OLD.read_text(encoding="utf-8")
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    raw = dict(re.findall(r"^\[\^(\d+)\]: (.+)$", old_text, re.M))

    anchor = "A global ranking here would read chemistry as anatomy, so none was used."
    added = " ethical approval and donor consent are described in the original publications of the human atlas[^6] and the mouse dataset[^12]."
    reconstructed = old_text.replace(anchor, anchor + added, 1)
    order, seen = [], set()
    for m in re.finditer(r"\[\^(\d+)\]", reconstructed[: reconstructed.index("## References")]):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            order.append(m.group(1))
    assert len(order) == 60, len(order)

    notes = {}
    for num, entry in raw.items():
        acc = re.findall(r"GWAS Catalog (GCST\w+)", entry)
        strat = re.findall(r"\(([^()]*(?:ancestry stratum|multi-ancestry stratum)[^()]*)\)", entry)
        bits = []
        if acc:
            bits.append("GWAS Catalog " + ", ".join(dict.fromkeys(acc)))
        for s in strat:
            s = re.sub(r",?\s*GWAS Catalog GCST\w+", "", s).strip(" ,.")
            if s:
                bits.append(s)
        if bits:
            notes[num] = "(" + "; ".join(bits) + ")"

    lines = [f"{i + 1}. " + fmt_entry(cache[o], notes.get(o, "")) for i, o in enumerate(order)]

    t = MS.read_text(encoding="utf-8")
    head = t[: t.index("## References")]
    tail = t[t.index("\n\n---\n\n## Figure titles and legends"):]
    MS.write_text(head + "## References\n\n" + "\n\n".join(lines) + tail, encoding="utf-8")

    print("entries rebuilt:", len(lines))
    for i, o in enumerate(order):
        people, groups = split_authors(cache[o]["authors"])
        if groups:
            print(f"  ref {i+1}: {len(people)} people + group(s) {groups}")
    bad = [i + 1 for i, l in enumerate(lines)
           if "et al." in l and len(split_authors(cache[order[i]]['authors'])[0]) < 10]
    print("et al. used below 10 personal authors:", bad)
    print("HTML entities left:", sum(l.count("&gt;") + l.count("&amp;") for l in lines))


if __name__ == "__main__":
    main()
