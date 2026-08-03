"""Final tidy of the Cell-format reference list.

Fixes what the automated conversion could not: publisher title case, JATS
subscript artifacts, inconsistent journal abbreviations, hyphenated page
ranges, one corporate-author entry with no machine-readable record, and a
duplicated accession note.
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

ABBREV = {
    "Biological Psychiatry": "Biol. Psychiatry",
    "Biol Psychiatry": "Biol. Psychiatry",
    "Biomedical Journal": "Biomed. J.",
    "Biometrics": "Biometrics",
    "Cell Systems": "Cell Syst.",
    "Circulation": "Circulation",
    "Circulation Research": "Circ. Res.",
    "Commun Biol": "Commun. Biol.",
    "Development": "Development",
    "Eur J Epidemiol": "Eur. J. Epidemiol.",
    "European Heart Journal": "Eur. Heart J.",
    "Front. Neurosci.": "Front. Neurosci.",
    "Genetic Epidemiology": "Genet. Epidemiol.",
    "Genetics": "Genetics",
    "Genome Biol": "Genome Biol.",
    "Genome Med": "Genome Med.",
    "International Journal of Cardiology": "Int. J. Cardiol.",
    "International Journal of Epidemiology": "Int. J. Epidemiol.",
    "Journal of Biological Chemistry": "J. Biol. Chem.",
    "Journal of the American College of Cardiology": "J. Am. Coll. Cardiol.",
    "Journal of the Royal Statistical Society Series B: Statistical Methodology":
        "J. R. Stat. Soc. Series B Stat. Methodol.",
    "Nat Biotechnol": "Nat. Biotechnol.",
    "Nat Commun": "Nat. Commun.",
    "Nat Genet": "Nat. Genet.",
    "Nat Hum Behav": "Nat. Hum. Behav.",
    "Nat Rev Genet": "Nat. Rev. Genet.",
    "Nature": "Nature",
    "Neuroscience and Biobehavioral Reviews": "Neurosci. Biobehav. Rev.",
    "PLoS Comput Biol": "PLoS Comput. Biol.",
    "Proc. Natl. Acad. Sci. U.S.A.": "Proc. Natl. Acad. Sci. USA",
    "Progress in Biophysics and Molecular Biology": "Prog. Biophys. Mol. Biol.",
    "Protein Cell": "Protein Cell",
    "Respir Res": "Respir. Res.",
    "Sci Rep": "Sci. Rep.",
    "Science": "Science",
    "Social Psychological and Personality Science": "Soc. Psychol. Personal. Sci.",
    "The FASEB Journal": "FASEB J.",
    "Transl Psychiatry": "Transl. Psychiatry",
    "eLife": "eLife",
}

# tokens that must keep their capitals when a title is lowered to sentence case
KEEP = {
    "African", "American", "Asian", "Bernard", "Biobank", "Brugada", "Cav1.3",
    "Claude", "European", "Genomes", "HapMap", "Hispanic", "Japanese", "Latin",
    "Leiden", "Louvain", "Mendelian", "Purkinje", "Scrublet", "UK", "Veteran",
    "Program", "Egger", "Hochberg", "Benjamini", "GABAergic", "G",
}


def sentence_case(title):
    """Lower publisher title case, protecting acronyms, gene symbols and names."""
    title = re.sub(r"\s+", " ", title).strip()
    parts = re.split(r"(?<=[:.?]) ", title)
    out_parts = []
    for part in parts:
        words = part.split(" ")
        new = []
        for i, w in enumerate(words):
            core = w.strip("(),;:.[]")
            if i == 0 or not core:
                new.append(w)
                continue
            if core in KEEP or any(ch.isdigit() for ch in core):
                new.append(w)
                continue
            # all-caps or internal capitals: an acronym or a gene symbol
            if core.upper() == core or core[1:] != core[1:].lower():
                new.append(w)
                continue
            if core[0].isupper():
                w = w.replace(core, core[0].lower() + core[1:], 1)
            new.append(w)
        out_parts.append(" ".join(new))
    return " ".join(out_parts)


def main():
    raw = dict(re.findall(r"^\[\^(\d+)\]: (.+)$", OLD.read_text(encoding="utf-8"), re.M))
    cache = json.loads(CACHE.read_text(encoding="utf-8"))

    # the one corporate-author record with no usable machine metadata
    cache["50"] = {
        "authors": [("Task Force of the European Society of Cardiology and the North "
                     "American Society of Pacing and Electrophysiology", "")],
        "year": 1996,
        "title": ("Heart rate variability: standards of measurement, physiological "
                  "interpretation and clinical use"),
        "journal": "Circulation", "volume": "93", "pages": "1043-1065", "doi": "",
    }

    for m in cache.values():
        if not m:
            continue
        m["title"] = sentence_case(re.sub(r"\s+", " ", m["title"]))
        j = (m.get("journal") or "").replace("&amp;", "and").strip()
        m["journal"] = ABBREV.get(j, j)
        if m.get("pages"):
            m["pages"] = re.sub(r"(\d)-(\d)", r"\1–\2", m["pages"])
    CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")

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

    text = MS.read_text(encoding="utf-8")
    body = text[: text.index("## References")]
    order, seen = [], []
    for m in re.finditer(r"\[\^(\d+)\]",
                         OLD.read_text(encoding="utf-8").split("## References")[0]):
        if m.group(1) not in seen:
            seen.append(m.group(1))
            order.append(m.group(1))

    lines = [f"{i + 1}. " + conv.fmt_entry(cache[o], notes.get(o, ""))
             for i, o in enumerate(order)]
    MS.write_text(body + "## References\n\n" + "\n\n".join(lines) + "\n", encoding="utf-8")

    print(f"{len(lines)} entries rebuilt")
    for i in (0, 1, 2, 5, 14, 22, 59):
        print("\n" + lines[i][:300])


if __name__ == "__main__":
    main()
