"""Pre-delivery checks for the six main figures.

1. Every number in the per-figure source data still matches results/.
2. Text in the vector output is text, not outlines.
3. Colour-vision and greyscale renderings of each figure.
4. A 50%-scale raster to check that 5 pt annotation survives reduction.

Writes its renderings to figures/qa/ and prints a pass/fail line per check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image

from _style import (FIGDIR, MDE, MOUSE_CLASS, MOUSE_CLASS_HATCH, RESULTS,
                    SE, SRCDIR, load)

QA = FIGDIR / "qa"
QA.mkdir(parents=True, exist_ok=True)
FIGS = ["Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "Fig6"]
EXTENDED = [f"FigureS{i}" for i in range(1, 8)]
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# --------------------------------------------------------------------------- #
# 1  source data against results/
# --------------------------------------------------------------------------- #
STRAT = load("scdrs_axis/axis_interaction.json")["stratified_auc"]
RG = load("ldsc_rg.json")["rg"]
NEG = load("negative_audit.json")
LODO = load("leave_one_donor_out.json")
MREP = load("mouse_replication.json")["results"]

STATE_KEY = {"SAN pacemaker": "SAN_P_cell", "AVN pacemaker": "AVN_P_cell",
             "AV bundle": "AVN_bundle_cell", "Purkinje": "Purkinje"}
TRAIT_KEY = {
    "Educational attainment": "EducationalAttainment", "Intelligence": "Intelligence",
    "Reaction time": "ReactionTime", "HRV, SDNN corrected": "HRV_SDNNc",
    "HRV, SDNN": "HRV_SDNN", "HRV, RMSSD corrected": "HRV_RMSSDc",
    "HRV, RMSSD": "HRV_RMSSD", "Resting heart rate": "RestingHeartRate",
    "PR interval": "PRinterval", "QRS duration": "QRSduration",
    "Bundle branch block": "BundleBranchBlock",
    "Atrioventricular block": "AVblock", "Atrial flutter": "AtrialFlutter",
    "Atrial fibrillation": "AtrialFibrillation", "Heart failure": "HeartFailure",
    "Brugada syndrome": "Brugada", "QT interval": "QTinterval",
    "Rheumatoid arthritis (control)": "RheumatoidArthritis",
    "Rheumatoid arthritis, MHC-free": "RheumatoidArthritis_noMHC"}

a = pd.read_excel(SRCDIR / "Fig2.xlsx", sheet_name="a_heatmap")
bad = [r for _, r in a.iterrows()
       if abs(STRAT[TRAIT_KEY[r.trait]][STATE_KEY[r.cell_state]]["auc"] - r.AUC) > 5e-5
       or abs(STRAT[TRAIT_KEY[r.trait]][STATE_KEY[r.cell_state]]["z"] - r.z) > 5e-3]
check("Fig2 a: 76 AUC and z values match axis_interaction.json",
      not bad and len(a) == 76, f"{len(a)} cells, {len(bad)} mismatches")

c = pd.read_excel(SRCDIR / "Fig2.xlsx", sheet_name="d_ambient_split")
AMB = load("ambient_stratified.json")
AMB_KEY = {"Educational attainment": "EducationalAttainment",
           "Intelligence": "Intelligence", "Reaction time": "ReactionTime",
           "HRV, SDNN": "HRV_SDNN", "Rheumatoid arthritis": "RheumatoidArthritis"}
bad = [r.trait for _, r in c.iterrows()
       if abs(AMB[AMB_KEY[r.trait]]["clean"] - r.clean_125) > 5e-5
       or abs(AMB[AMB_KEY[r.trait]]["all"] - r.all_245) > 5e-5]
check("Fig2 c: ambient split matches ambient_stratified.json", not bad, str(bad))

b = pd.read_excel(SRCDIR / "Fig2.xlsx", sheet_name="b_forest")
bad = [r.trait for _, r in b.iterrows()
       if abs(SE[STATE_KEY[r.cell_state]] - r.SE) > 5e-5
       or abs(MDE[STATE_KEY[r.cell_state]] - r.MDE) > 5e-4]
check("Fig2 b: SE and MDE match negative_audit.json", not bad, str(bad))

d3 = pd.read_excel(SRCDIR / "Fig3.xlsx", sheet_name="a_immune_control")
bad = [(r.gene_set, r.cell_state) for _, r in d3.iterrows()
       if abs(STRAT["RheumatoidArthritis" if r.gene_set == "34,164 variants"
                    else "RheumatoidArthritis_noMHC"][
           STATE_KEY[r.cell_state]]["auc"] - r.AUC) > 5e-5]
check("Fig3 a: immune control matches axis_interaction.json", not bad, str(bad))

a4 = pd.read_excel(SRCDIR / "Fig6.xlsx", sheet_name="a_rg_matrix")
COG = {"Educational attainment": "EducationalAttainment",
       "Intelligence": "Intelligence", "Reaction time": "ReactionTime"}
CARD = {"HRV, RMSSD": "HRV_RMSSD", "HRV, SDNN": "HRV_SDNN",
        "Resting heart rate": "RestingHeartRate", "PR interval": "PRinterval",
        "Atrial fibrillation": "AtrialFibrillation", "QT interval": "QTinterval"}
bad = []
for _, r in a4.iterrows():
    k1, k2 = COG[r.cognitive], CARD[r.cardiovascular]
    v = RG.get(f"{k1}|{k2}") or RG[f"{k2}|{k1}"]
    if abs(v["rg"] - r.rg) > 5e-5 or abs(v["se"] - r.SE) > 5e-5:
        bad.append((r.cognitive, r.cardiovascular))
check("Fig6 a: 18 genetic correlations match ldsc_rg.json",
      not bad and len(a4) == 18, f"{len(a4)} pairs, {len(bad)} mismatches")

c4 = pd.read_excel(SRCDIR / "Fig6.xlsx", sheet_name="c_equivalence")
check("Fig6 c: equivalence bounds match negative_audit.json",
      len(c4) == len(NEG["rg_equivalence"])
      and all(abs(u - v["upper"]) < 5e-5 for u, v
              in zip(c4.upper_bound, NEG["rg_equivalence"].values())))

a6 = pd.read_excel(SRCDIR / "Fig5.xlsx", sheet_name="a_leave_one_donor_out")
LODO_KEY = dict(TRAIT_KEY, **{"Rheumatoid arthritis": "RheumatoidArthritis"})
bad = []
for _, r in a6.iterrows():
    d = LODO["lodo"][f"{LODO_KEY[r.trait]}|{r.cell_state}"]
    if (abs(d["lodo_min"] - r.lodo_min) > 5e-5
            or abs(d["lodo_max"] - r.lodo_max) > 5e-5
            or abs(d["full"] - r.full) > 5e-5):
        bad.append((r.trait, r.cell_state))
check("Fig5 a: leave-one-donor-out ranges match leave_one_donor_out.json",
      not bad, str(bad))

d6 = pd.read_excel(SRCDIR / "Fig5.xlsx", sheet_name="d_mouse_scorecard")
MK = {"Resting heart rate": "RestingHeartRate", "HRV, SDNN": "HRV_SDNN",
      "HRV, RMSSD": "HRV_RMSSD", "Educational attainment": "EducationalAttainment",
      "Intelligence": "Intelligence", "Reaction time": "ReactionTime",
      "Rheumatoid arthritis": "RheumatoidArthritis", "QT interval": "QTinterval"}
bad = [r.trait for _, r in d6.iterrows()
       if abs(MREP[MK[r.trait]]["auc"] - r.mouse_AUC) > 5e-5]
check("Fig5 d: mouse scorecard matches mouse_replication.json", not bad, str(bad))
check("Fig5 d: exactly one pre-registered prediction failed",
      list(d6.outcome).count("fail") == 1
      and d6.loc[d6.outcome == "fail", "prediction"].iloc[0] == "P4",
      f"failed: {d6.loc[d6.outcome == 'fail', 'trait'].tolist()}")

b5 = pd.read_excel(SRCDIR / "Fig4.xlsx", sheet_name="b_depletion")
check("Fig4 b: the five readable depletions, none from the bundle column",
      len(b5) == 5 and "AV bundle" not in set(b5.cell_state),
      ", ".join(f"{r.trait}/{r.cell_state}" for r in b5.itertuples()))

# --------------------------------------------------------------------------- #
# 2  vector text is still text
# --------------------------------------------------------------------------- #
for f in FIGS + EXTENDED:
    svg = (FIGDIR / f"{f}.svg").read_text(encoding="utf-8")
    pdf = (FIGDIR / f"{f}.pdf").read_bytes()
    check(f"{f}: SVG keeps live text, PDF embeds a named font",
          "<text" in svg and b"Arial" in pdf,
          f"{svg.count('<text')} text elements")

# ED1 must agree with the correction reported everywhere else
ed1 = pd.read_excel(SRCDIR / "FigureS6.xlsx",
                    sheet_name="a_all_140_tests")
check("FigureS6: the volcano plots all 140 tests inside its axis limits",
      len(ed1) == 140 and (ed1.auc - 0.5).abs().max() < 0.30,
      f"widest effect {(ed1.auc - 0.5).abs().max():.4f}")
ed1c = pd.read_excel(SRCDIR / "FigureS6.xlsx", sheet_name="c_direction")
check("FigureS6: the distal populations have no significant enrichments",
      int(ed1c.loc[ed1c.cell_state == "AV bundle", "enriched"].iloc[0]) == 0
      and int(ed1c.loc[ed1c.cell_state == "Purkinje", "enriched"].iloc[0]) == 0,
      ", ".join(f"{r.cell_state} {r.enriched}up/{r.depleted}dn"
                for r in ed1c.itertuples()))
ed4d = pd.read_excel(SRCDIR / "FigureS3.xlsx", sheet_name="d_power")
check("FigureS3: both power comparators are drawn and the text's one is marked",
      len(ed4d) == 4 and int(ed4d.used_in_the_text.sum()) == 2,
      f"{len(ed4d)} scenarios")
stored = load("corrections.json")["fdr"]
check("FigureS6: 140 tests, 61 at FDR < 0.05, matching corrections.json",
      len(ed1) == stored["n_tests"] and int((ed1.fdr < 0.05).sum()) == stored["n_survive"],
      f"{len(ed1)} tests, {int((ed1.fdr < 0.05).sum())} significant")

ed2 = pd.read_excel(SRCDIR / "FigureS4.xlsx", sheet_name="comparator_shift")
shift = ed2.groupby("cell_state")["shift"].apply(lambda s: s.abs().max())
check("FigureS4: sinoatrial and Purkinje are unmoved by the comparator change",
      shift["SAN pacemaker"] < 1e-9 and shift["Purkinje"] < 1e-9,
      ", ".join(f"{k} {v:.4f}" for k, v in shift.items()))

ed3 = pd.read_excel(SRCDIR / "FigureS7.xlsx", sheet_name="mr_v2_tests")
mrv2 = pd.read_csv(RESULTS / "mr_v2_estimates.tsv", sep="	")
# the drawn table labels its rows in prose, so match on the numbers themselves:
# every drawn estimate must be one of the stored ones, to four decimals
_stored = set(zip(mrv2.ivw.round(4), mrv2.ivw_se.round(4)))
_bad = [r.test for r in ed3.itertuples()
        if (round(r.IVW, 4), round(r.IVW_SE, 4)) not in _stored]
check("FigureS7: every drawn estimate is one stored in mr_v2_estimates.tsv",
      len(ed3) > 0 and not _bad, f"{len(ed3)} drawn, {len(_bad)} not found")

# --------------------------------------------------------------------------- #
# 3  colour vision and greyscale
# --------------------------------------------------------------------------- #
def _lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


# Viénot, Brettel & Mollon (1999) dichromat simulation
RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                    [3.45565, 27.1554, 3.86714],
                    [0.0299566, 0.184309, 1.46709]])
LMS2RGB = np.linalg.inv(RGB2LMS)
DEUTAN = np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]])
PROTAN = np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]])


def simulate(img, M):
    rgb = _lin(img[..., :3].astype(np.float64) / 255.0)
    lms = rgb @ RGB2LMS.T
    out = (lms @ M.T) @ LMS2RGB.T
    return (_srgb(out) * 255).astype(np.uint8)


for f in FIGS + EXTENDED:
    src = Image.open(FIGDIR / "preview" / f"{f}.png").convert("RGB")
    arr = np.asarray(src)
    Image.fromarray(simulate(arr, DEUTAN)).save(QA / f"{f}_deuteranopia.png")
    Image.fromarray(simulate(arr, PROTAN)).save(QA / f"{f}_protanopia.png")
    src.convert("L").save(QA / f"{f}_greyscale.png")
    w, h = src.size
    src.resize((w // 2, h // 2), Image.LANCZOS).save(QA / f"{f}_50percent.png")
check("colour-vision, greyscale and 50% renderings written",
      len(list(QA.glob("*.png"))) == 4 * (len(FIGS) + len(EXTENDED)), str(QA))


def _grey(h):
    """ITU-R 601 luma, the transform PIL's convert("L") applies."""
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b


# the Figure S5 legend states that the panel survives greyscale. Hatching carries
# the two excluded classes; every other class has only its grey level, and the
# scored class and the comparator once sat 2.5 levels apart out of 255.
_solid = [k for k in MOUSE_CLASS if k not in MOUSE_CLASS_HATCH]
_worst = min((abs(_grey(MOUSE_CLASS[a]) - _grey(MOUSE_CLASS[b])), a, b)
             for i, a in enumerate(_solid) for b in _solid[i + 1:])
check("FigureS5: unhatched classes stay apart in greyscale", _worst[0] >= 12,
      f"closest {_worst[1]} / {_worst[2]}, {_worst[0]:.1f} of 255")

# --------------------------------------------------------------------------- #
# 4  page geometry and legends
# --------------------------------------------------------------------------- #
import re

PT = 72.0 / 25.4
sizes, over = {}, []
for f in FIGS + EXTENDED:
    box = re.search(rb"/MediaBox\s*\[([^\]]*)\]",
                    (FIGDIR / f"{f}.pdf").read_bytes())
    x0, y0, x1, y1 = [float(v) for v in box.group(1).split()]
    w, h = (x1 - x0) / PT, (y1 - y0) / PT
    sizes[f] = (w, h)
    if w > 183.5 or h > 225.5:
        over.append(f"{f} {w:.0f}x{h:.0f}")
check("every canvas fits the 183 x 225 mm journal page", not over, str(over))
check("every canvas is a permitted column width",
      all(any(abs(w - c) < 0.6 for c in (89, 120, 144, 150, 172, 183))
          for w, _ in sizes.values()),
      ", ".join(f"{k} {v[0]:.0f}x{v[1]:.0f}" for k, v in sizes.items()))

cap = (FIGDIR.parent / "FIGURE-CAPTIONS.md").read_text(encoding="utf-8")
blocks = {m.group(1): b for b in re.split(r"\n---\n", cap)
          for m in [re.search(r"\*\*(Figure S?\d+)\.", b)] if m}
long = {k: len(v.split()) for k, v in blocks.items() if len(v.split()) > 300}
# A panel referred to in lower case reads as a word. "in e" and "follow a" were
# caught by eye; "the comparisons a is built from" survived two sweeps, because
# excluding the letter a to avoid the English article also excluded panel A.
lower = [" ".join(m.group(0).split()) for b in blocks.values()
         for m in re.finditer(r"\b[a-f]\s+(?:is|are|was|were|gives|shows|puts)\b", b)]
check("no legend refers to a panel in lower case", not lower, str(lower))

check("one legend per figure, each under 300 words and free of em dashes",
      len(blocks) == len(FIGS) + len(EXTENDED) and not long
      and "—" not in cap,
      f"{len(blocks)} legends, longest {max(len(v.split()) for v in blocks.values())} words")

panels = {"Fig1": "ABCDEF", "Fig2": "ABCDE", "Fig3": "ABCDEF", "Fig4": "ABCDE",
          "Fig5": "ABCDE", "Fig6": "ABCD", "FigureS1": "AB",
          "FigureS2": "AB", "FigureS3": "ABCD",
          "FigureS4": "AB", "FigureS5": "AB",
          "FigureS6": "ABC", "FigureS7": "AB"}
missing = []
for f, letters in panels.items():
    name = f.replace("FigureS", "Figure S") if f.startswith("FigureS") \
        else f.replace("Fig", "Figure ")
    blk = blocks[name]
    svg = (FIGDIR / f"{f}.svg").read_text(encoding="utf-8")
    for ch in letters:
        if f"({ch})" not in blk:
            missing.append(f"{f}:{ch} not in legend")
        if f">{ch}<" not in svg:
            missing.append(f"{f}:{ch} not on canvas")
check("every drawn panel letter is described in its legend, and vice versa",
      not missing, str(missing))

# submission requires both series to be cited in numerical order.
# The manuscript file is chosen here rather than hard-coded, because during the
# 2026-08-03 renumbering two manuscript files carried figure callouts under two
# different numberings, and a check that silently read the stale one passed the
# whole way through. Whichever file this resolves to is printed by every
# manuscript check below, so it is never in doubt which prose was validated.
CANDIDATES = ["MANUSCRIPT-v3.md", "MANUSCRIPT.md"]
MS = next((FIGDIR.parent / c for c in CANDIDATES
           if (FIGDIR.parent / c).exists()), None)
assert MS is not None, CANDIDATES
body_full = MS.read_text(encoding="utf-8")
body = body_full.split("## Tables")[0]


def first_citation(pat):
    seen = []
    for m in re.finditer(pat, body):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


mains = first_citation(r"Figure (\d)")
eds = first_citation(r"Figure S(\d)")
check(f"{MS.name} cites both figure series in numerical order",
      mains == list(range(1, len(FIGS) + 1))
      and eds == list(range(1, len(EXTENDED) + 1)),
      f"main {mains}, extended {eds}")

# every figure script records, at render time, any axes whose data ran past its
# own limits. A truncated tail is invisible on the finished canvas, so this is
# the one class of defect visual review cannot catch.
import json as _json
clip = []
for f in FIGS + EXTENDED:
    rec = QA / f"_clip_{f}.json"
    if not rec.exists():
        clip.append(f"{f} was not re-rendered")
        continue
    clip += [f"{f}: {line}" for line in _json.loads(rec.read_text())]
check("no panel draws data outside its own axis limits", not clip, str(clip))

raster = []
for f in FIGS + EXTENDED:
    im = Image.open(FIGDIR / f"{f}.tif")
    dpi = im.info.get("dpi", (0, 0))
    if im.mode != "RGB" or round(dpi[0]) != 600:
        raster.append(f"{f} {im.mode} {dpi[0]:.0f} dpi")
    im.close()
check("every TIFF is flat RGB at 600 dpi, with no alpha channel",
      not raster, str(raster))

# the contract table in FIGURE-QA.md records each canvas; keep it honest
qa_doc = (FIGDIR.parent / "FIGURE-QA.md").read_text(encoding="utf-8")
drift = []
for m in re.finditer(r"(?m)^\| (\d) \|.*\| (\d+) × (\d+) mm \|$", qa_doc):
    f = f"Fig{m.group(1)}"
    want = (float(m.group(2)), float(m.group(3)))
    got = sizes[f]
    if abs(got[0] - want[0]) > 0.6 or abs(got[1] - want[1]) > 0.6:
        drift.append(f"{f} drawn {got[0]:.0f}x{got[1]:.0f}, recorded {want[0]:.0f}x{want[1]:.0f}")
check("the figure contract records the size each figure is actually drawn at",
      not drift, str(drift))

# the archetype in section 3 also names how many panels each figure has
cited_panels = {n: set() for n in range(1, len(FIGS) + 1)}
for m in re.finditer(r"(?:Figure |and )(\d)([A-F])\b", body):
    cited_panels[int(m.group(1))].add(m.group(2))
uncited = [f"Figure {n[-1]}{c}" for n, letters in panels.items() if n in FIGS
           for c in letters if c not in cited_panels[int(n[-1])]]
check(f"every main-figure panel is cited in {MS.name}",
      not uncited, str(uncited))

# and the other direction. The numeral-order check above reads numbers only,
# so it cannot see prose that has kept its callouts while the figures moved
# underneath them; a citation to a panel that is not drawn can.
ghosts = [f"Figure {n}{c}" for n, letters in cited_panels.items()
          for c in sorted(letters) if c not in panels[f"Fig{n}"]]
check(f"{MS.name} cites no panel that is not drawn", not ghosts, str(ghosts))

# The supplemental list in the manuscript names each figure in its own words, in
# a form no figure-number pattern matches, so a renumbering can move the figures
# and leave the list describing the old ones. Require a long word in common
# between each list entry and the matching legend title.
_list = re.search(r"Supplemental figures\.\*\* (.+)", body_full)
_entries = dict(re.findall(r"S(\d), ([^.]+)\.", _list.group(1))) if _list else {}
_titles = {m.group(1): m.group(2) for b in blocks.values()
           for m in [re.match(r"\*\*Figure S(\d)\. ([^*]+)\*\*", b.strip())] if m}


def _words(x):
    # five-character prefixes, so cluster and clustering count as one word
    return {w[:5] for w in re.findall(r"[a-z]{5,}", x.lower())}


_adrift = [f"S{n}" for n in _titles
           if n not in _entries
           or not (_words(_titles[n]) & _words(_entries[n]))]
check(f"{MS.name} describes each supplemental figure as its legend does",
      bool(_entries) and not _adrift,
      f"{len(_entries)} listed, adrift: {_adrift}" if _entries
      else "no supplemental figure list found")

# The manuscript carries its own copy of the main-figure legends. Two copies of
# the same text drift, and the figures were renumbered under both of them, so
# require them to be the same text.
_sec = re.search(r"## Figure titles and legends\n(.*?)(?=\n## )", body_full, re.S)


def _flat(x):
    return " ".join(x.replace("---", " ").split())


if _sec:
    _starts = [m.start() for m in re.finditer(r"\*\*Figure \d\.", _sec.group(1))]
    _embedded = {re.match(r"\*\*(Figure \d)\.", _sec.group(1)[a:]).group(1):
                 _flat(_sec.group(1)[a:b])
                 for a, b in zip(_starts, _starts[1:] + [len(_sec.group(1))])}
else:
    _embedded = {}
_drifted = [k for k, v in _embedded.items()
            if k not in blocks or _flat(blocks[k]) != v]
check(f"{MS.name} carries the same figure legends as FIGURE-CAPTIONS.md",
      len(_embedded) == len(FIGS) and not _drifted,
      f"{len(_embedded)} embedded, drifted: {_drifted}" if _embedded
      else "no embedded legends found")

# Panels must also be cited in order, or a reordered figure silently leaves its
# callouts pointing at the wrong panel. Figure 2 is a declared exception: its
# Results paragraph opens on the cell-state ranking in e (see FIGURE-QA.md 7).
PANEL_ORDER_EXEMPT = {2}
order_first = {n: [] for n in range(1, len(FIGS) + 1)}
for m in re.finditer(r"(?:Figure |and )(\d)([A-F])\b", body):
    if m.group(2) not in order_first[int(m.group(1))]:
        order_first[int(m.group(1))].append(m.group(2))
misordered = [f"Fig{n} cited {''.join(v)}" for n, v in order_first.items()
              if n not in PANEL_ORDER_EXEMPT and v != sorted(v)]
check(f"{MS.name} cites panels in panel order, exceptions declared",
      not misordered, str(misordered) if misordered
      else f"in order, Figure {sorted(PANEL_ORDER_EXEMPT)[0]} exempt")

print()
print("FAILED:" if fails else "all checks passed", *fails, sep="\n  " if fails else "")
sys.exit(1 if fails else 0)
