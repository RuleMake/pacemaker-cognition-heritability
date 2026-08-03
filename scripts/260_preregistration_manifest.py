"""
Generate the pre-registration manifest that the repository can actually support.
===============================================================================

The claim this study wants to make is that each set of predictions was written
before the scores it predicts were computed. The evidence for that claim has to
be stated at its true strength, and no stronger.

What exists:  file modification times on the analysis machine, plus the fact
              that every prediction file precedes, by minutes, the first output
              of the run it predicts.
What does not
exist:        a version-control history contemporaneous with the analysis. This
              repository was created at submission. Its commits date the
              deposition, not the work.

Those are different evidentiary strengths and the manifest says so. It also
records a SHA-256 for every prediction file, which is the part that IS strong:
once the archive carries a DOI, anyone can check that the prediction files in it
are byte-identical to the ones described here, so nothing can be quietly edited
after the fact.

Usage:  python scripts/260_preregistration_manifest.py
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "PREREGISTRATION.md"

# (prediction file, directory holding the outputs it predicts, glob, label)
ARMS = [
    ("results/axis_predictions.json", "results/scdrs_axis", "*.score.tsv",
     "Conduction axis: per-cell scoring of 19 gene sets in the human atlas"),
    ("results/mouse_predictions.json", "results/scdrs_mouse", "*.score.tsv",
     "Cross-species replication: mouse embryonic conduction zones"),
    ("results/mr_predictions_v2.json", "results", "mr_v2_estimates.tsv",
     "Bidirectional Mendelian randomization, re-specified stopping rules"),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ts(dt):
    return datetime.fromtimestamp(dt).strftime("%Y-%m-%d %H:%M:%S")


rows, ok = [], True
for pred, outdir, glob, label in ARMS:
    pp = ROOT / pred
    outs = sorted((ROOT / outdir).glob(glob), key=lambda p: p.stat().st_mtime)
    if not pp.exists() or not outs:
        raise SystemExit(f"missing: {pred} or outputs in {outdir}/{glob}")
    p_mt, s_mt = pp.stat().st_mtime, outs[0].stat().st_mtime
    gap = (s_mt - p_mt) / 60.0
    ok &= gap > 0
    rows.append(dict(label=label, pred=pred, pred_time=ts(p_mt),
                     first_out=str(outs[0].relative_to(ROOT)).replace("\\", "/"),
                     first_out_time=ts(s_mt), gap_min=round(gap, 1),
                     sha256=sha256(pp), n_bytes=pp.stat().st_size))

lines = [
    "# Pre-registration record",
    "",
    "Predictions, falsification conditions and stopping rules for this study were",
    "written to disk before the corresponding results were computed. This file",
    "documents that ordering and states exactly how strong the evidence for it is.",
    "",
    "## The three registered arms",
    "",
    "| Arm | Prediction file | Written | First output it predicts | Output written | Gap |",
    "|---|---|---|---|---|---|",
]
for r in rows:
    lines.append(f"| {r['label']} | `{r['pred']}` | {r['pred_time']} | "
                 f"`{r['first_out']}` | {r['first_out_time']} | "
                 f"{r['gap_min']} min |")

lines += [
    "",
    f"Ordering holds in all {len(rows)} arms: "
    f"{'yes' if ok else 'NO — DO NOT PUBLISH THIS FILE'}.",
    "",
    "## Integrity hashes",
    "",
    "SHA-256 of each prediction file as deposited. Once this repository is archived",
    "under a DOI, these let any reader confirm the prediction files have not been",
    "edited after the fact.",
    "",
    "| Prediction file | Bytes | SHA-256 |",
    "|---|---|---|",
]
for r in rows:
    lines.append(f"| `{r['pred']}` | {r['n_bytes']:,} | `{r['sha256']}` |")

lines += [
    "",
    "## What this evidence is, and what it is not",
    "",
    "**It is** a consistent ordering recorded by the filesystem of the analysis",
    "machine, corroborated by the analysis logs, together with hashes that fix the",
    "content of the prediction files from the moment of deposition onward.",
    "",
    "**It is not** a contemporaneous version-control history. This repository was",
    "created when the manuscript was submitted, so its commit dates record the",
    "deposition and not the analysis. File modification times on a single machine",
    "can be set by the person who owns the machine, and readers should weigh them",
    "accordingly. No commit in this repository has been backdated.",
    "",
    "**What would have been stronger**, and what a future study should do instead:",
    "deposit the predictions with a third party that timestamps them independently",
    "— OSF, AsPredicted, or a public commit pushed to a hosted repository on the",
    "day the predictions were written. The value of pre-registration comes from the",
    "record being outside the author's control, and that is the part missing here.",
    "",
    "## Why it still matters here",
    "",
    "Two of the registered predictions failed and are reported as failures: the",
    "ventricular-arm prediction on the human conduction axis, and the QT-interval",
    "prediction in mouse. Educational attainment entered the study labelled as a",
    "negative control and became its strongest positive. None of that history",
    "survives without a record written in advance.",
    "",
    f"Generated by `scripts/260_preregistration_manifest.py`.",
]

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
(ROOT / "results" / "preregistration_manifest.json").write_text(
    json.dumps(dict(ordering_holds=bool(ok), arms=rows), indent=2), encoding="utf-8")
print("\n".join(lines))
print(f"\nwrote {OUT}")
