# Cardiac pacemaker cells carry common-variant heritability for human cognitive traits but not for ventricular repolarization

Analysis code, pre-registered predictions and derived result tables for the study
of the same name.

Single-cell heritability for eighteen traits was scored in a human heart atlas
that dissected sinoatrial and atrioventricular nodal tissue, with each conduction
population compared only against the myocyte lineage of the same donor, region
and sequencing chemistry. Genome-wide genetic correlations were estimated by LD
score regression and causal structure tested by bidirectional Mendelian
randomization.

## What is in this repository

| Path | Contents |
|---|---|
| `scripts/` | The full analysis pipeline, numbered in execution order |
| `scripts/fig/` | Figure generation |
| `results/*.json`, `results/*.tsv` | Derived result tables and audit outputs |
| `results/*_predictions.json` | The pre-registered predictions and stopping rules |
| `PREREGISTRATION.md` | Prediction/scoring ordering, integrity hashes, and an honest statement of how strong that evidence is |
| `DATA-SOURCES.md` | Every input dataset with its accession |

## What is deliberately not here

**Raw data.** Every input is public and is identified by accession in
`DATA-SOURCES.md` and in Table S14 of the manuscript. The GWAS summary
statistics, the human cell atlas and the mouse expression matrices are
redistributed by their originators under their own terms, and re-hosting them
here is not ours to do.

**Per-cell score matrices.** These are derived, but they run to hundreds of
megabytes. They are in the archived release rather than in git history.

**Working notes and manuscript drafts.** The project directory also holds
internal deliberation, self-critique logs and superseded drafts, several of which
contain claims that were later withdrawn on the evidence. Publishing those
alongside the analysis would misrepresent what the study concluded.

## Pre-registration

Predictions, falsification conditions and Mendelian randomization stopping rules
were written to disk before the corresponding results were computed, in three
arms, with gaps of 12 to 24 minutes between each prediction file and the first
output it predicts.

**This repository was created at submission.** Its commit dates therefore record
the deposition, not the analysis, and no commit has been backdated. The ordering
evidence is filesystem modification times on the analysis machine plus the
analysis logs. `PREREGISTRATION.md` sets out what that does and does not
establish, and records SHA-256 hashes so the prediction files cannot be edited
after deposition without detection.

Two registered predictions failed and are reported as failures. Educational
attainment entered the study labelled as a negative control and became its
strongest positive. Neither fact survives without a record written in advance.

## Reproducing

Python 3.12. Scripts run in numeric order; each writes to `results/` and is
independently re-runnable given its inputs.

```
pip install numpy pandas scipy scikit-learn anndata scanpy matplotlib
```

`scdrs` and MAGMA are external and are listed with their versions in
Supplementary Methods (Data S3). LD score regression is re-implemented in
`scripts/170_ldsc_rg.py` because the released version requires Python 2; it is
validated against published estimates before use, and that check runs first and
prints before anything else.

Two audits added after the first review round:

- `scripts/250_permutation_null.py` — empirical null for the primary within-stratum
  statistic (10,000 within-stratum label permutations, a depth-blocked variant, and
  a donor-cluster bootstrap)
- `scripts/251_ea_ancestry_sensitivity.py` — exposure of the population-level
  estimates to the ancestry composition of the educational attainment GWAS

## Citation

*Citation to be added on publication.*

## Contact

*Corresponding author contact to be added.*

## License

Code in this repository is released under the MIT License (see `LICENSE`).
Result tables derived from third-party data remain subject to the terms of their
original sources, which are listed in `DATA-SOURCES.md`.
