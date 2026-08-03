# Data sources

Every input to this study is public. None of it is redistributed from this
repository; each item below is obtained from its originator under that
originator's terms. Full per-trait detail, including sample sizes, ancestry
composition and genome-wide-significant variant counts, is in Table S14 of the
manuscript, and variant counts before and after harmonization are in Table S1.

## Single-cell and spatial

| Dataset | Source | Notes |
|---|---|---|
| Human spatially resolved multiomic heart atlas, eight regions with sinoatrial and atrioventricular nodal tissue dissected | CELLxGENE Discover; accession in the original publication | Used with the authors' 76 cell-state labels, without re-annotation. Two subsets analysed: the nodal dataset (118,172 cells, all 10x multiome) and the conduction-axis dataset (78,134 nuclei, 76 states, 22 donors, 8 regions) |
| Mouse embryonic day 16.5 conduction-zone single-cell RNA-seq | GEO `GSE132658` | Four microdissected zones (SAN, AVN, LPF, RPF). 15,193 cells, 29 Leiden clusters |

## GWAS summary statistics

The trait that carries the population-level result:

| Trait | Accession | N | Ancestry |
|---|---|---|---|
| Educational attainment | GWAS Catalog `GCST90296499` | 931,577 | European 766,345 + East Asian 165,232 |

The ancestry composition of this dataset is a named limitation of the study: it
is evaluated against a European LD reference, and a different trait was excluded
from the study for that same defect. `scripts/251_ea_ancestry_sensitivity.py`
quantifies the exposure. A European-only re-estimate requires the European
component on its own, which is distributed by the SSGAC rather than through the
GWAS Catalog.

All remaining GWAS — intelligence, reaction time, the two heart rate variability
indices and their corrected forms, resting heart rate, PR interval, QRS duration,
atrial fibrillation, atrial flutter, QT interval, bundle branch block,
atrioventricular block, Brugada syndrome, heart failure, and the rheumatoid
arthritis control with its MHC-free rebuild — are listed with their source
publication and accession in Table S14.

## Reference panels and tools

| Resource | Use |
|---|---|
| 1000 Genomes Phase 3 European | LD reference for LD score regression and for instrument pruning |
| HapMap3 LD scores, no HLA | LD score regression weights |
| NCBI b37 gene locations | MAGMA gene annotation |

Software versions and parameters are in Supplementary Methods (Data S3). URLs for
tools without a DOI are in the Web resources section of the manuscript.

## Terms

Redistribution terms differ between sources. Anyone reproducing this analysis
should obtain each dataset directly from the accession given above and observe
the terms attached to it there.
