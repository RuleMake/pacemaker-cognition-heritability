"""Resynchronise the reference list with the in-text numbering.

Script 242 rebuilt the list using the first-appearance order of the pre-format
backup, but the in-text markers had been renumbered against the text as it
stood after the AJHG prose edits. Those edits inserted one sentence carrying two
citations (the atlas and the mouse dataset) into Material and Methods, which
shifts the order. Reconstructing that text exactly recovers the true mapping.

Verification is by content, not by trust: the entry a marker resolves to is
checked against the sentence the marker sits in.
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

ETHICS_ANCHOR = "A global ranking here would read chemistry as anatomy, so none was used."
ETHICS_ADDED = (" The 22 donors are 11 female and 11 male, aged 20 to 75 years; the six donors "
                "contributing sinoatrial pacemaker cells are three female and three male, and the "
                "three contributing atrioventricular pacemaker cells are two female and one male. "
                "This study analysed only previously published, de-identified human data and "
                "previously published mouse data, and no new human or animal material was "
                "collected; ethical approval and donor consent are described in the original "
                "publications of the human atlas[^6] and the mouse dataset[^12].")


def main():
    old = OLD.read_text(encoding="utf-8")
    assert old.count(ETHICS_ANCHOR) == 1
    reconstructed = old.replace(ETHICS_ANCHOR, ETHICS_ANCHOR + ETHICS_ADDED, 1)
    pre_refs = reconstructed[: reconstructed.index("## References")]

    order, seen = [], set()
    for m in re.finditer(r"\[\^(\d+)\]", pre_refs):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            order.append(m.group(1))
    assert len(order) == 60, f"expected 60 references, got {len(order)}"

    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    raw = dict(re.findall(r"^\[\^(\d+)\]: (.+)$", old, re.M))

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

    lines = [f"{i + 1}. " + conv.fmt_entry(cache[o], notes.get(o, ""))
             for i, o in enumerate(order)]

    t = MS.read_text(encoding="utf-8")
    head = t[: t.index("## References")]
    tail = t[t.index("\n\n---\n\n## Figure titles and legends"):]
    MS.write_text(head + "## References\n\n" + "\n\n".join(lines) + tail, encoding="utf-8")

    # --- verify by content: spot-check markers against the sentence they sit in ---
    new = MS.read_text(encoding="utf-8")
    entries = {int(m.group(1)): m.group(2)
               for m in re.finditer(r"^(\d+)\. (.+)$",
                                    re.search(r"## References\n\n(.*?)\n\n---\n\n## Figure titles",
                                              new, re.S).group(1), re.M)}
    checks = [("spatially resolved multiomic atlas", "Kanemaru"),
              ("GSE132658", "Goodyer"),
              ("MAGMA v1.10", "de Leeuw"),
              ("Scrublet", "Wolock"),
              ("Benjamini-Hochberg procedure", "Benjamini"),
              ("scDRS", "Zhang")]
    body = new[: new.index("## References")]
    print("content verification")
    for phrase, expect in checks:
        i = body.find(phrase)
        if i < 0:
            print(f"  ?    phrase not found: {phrase}")
            continue
        m = re.search(r"\[\^(\d+)\]", body[i:i + 400])
        got = entries[int(m.group(1))] if m else ""
        ok = expect.lower() in got.lower()
        print(f"  {'OK  ' if ok else 'FAIL'} {phrase:34s} -> [{m.group(1)}] {got[:60]}")

    used = [int(x) for x in re.findall(r"\[\^(\d+)\]", body)]
    first = list(dict.fromkeys(used))
    print("\nfirst-appearance order is 1..N:", first == list(range(1, len(first) + 1)))
    print("entries:", len(entries), "| distinct cited in main text:", len(set(used)))


if __name__ == "__main__":
    main()
