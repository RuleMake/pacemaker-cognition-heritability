"""Table S5: residual technical association of the cell-level scDRS scores.

This table did not exist. The three values the Methods quoted for it came from
`results/library_size_confound.json`, which `scripts/52_library_size_confound.py`
computes per Visium *spot* for the gsMap arm, not per cell for scDRS. Those are
different quantities on different data, and the spot-level figures belong to the
spatial analysis (Data S2) rather than to the per-cell scoring section.

What is computed here is the quantity the Methods actually needs: for every trait,
the Spearman correlation between the normalized per-cell scDRS score and each of
the three technical covariates that were regressed out, over all 78,134 cells of
the conduction-axis dataset and within each conduction population separately.

The normalized score is used because it is the score every downstream statistic in
the paper is built on. The raw score is reported alongside it so the effect of the
covariate regression is visible rather than asserted.
"""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results" / "scdrs_axis"
COV = ROOT / "data" / "scdrs" / "covariates_axis.tsv"
OBS = ROOT / "results" / "obs_axis.tsv"
OUT_TSV = ROOT / "results" / "table_s11_depth_celllevel.tsv"
OUT_JSON = ROOT / "results" / "table_s11_depth_celllevel.json"

POPS = {
    "SAN_P_cell": "SAN pacemaker",
    "AVN_P_cell": "AVN pacemaker",
    "AVN_bundle_cell": "AV bundle",
    "Purkinje": "Purkinje",
}
COVARIATES = {"log_total": "total UMI", "n_genes": "genes detected", "pct_mt": "mitochondrial fraction"}
# display names, matching Table 1
TRAITS = {
    "EducationalAttainment": "Educational attainment", "Intelligence": "Intelligence",
    "ReactionTime": "Reaction time", "HRV_SDNNc": "HRV, SDNN corrected",
    "HRV_SDNN": "HRV, SDNN", "HRV_RMSSDc": "HRV, RMSSD corrected",
    "HRV_RMSSD": "HRV, RMSSD", "RestingHeartRate": "Resting heart rate",
    "PRinterval": "PR interval", "QRSduration": "QRS duration",
    "HeartFailure": "Heart failure", "AtrialFlutter": "Atrial flutter",
    "AtrialFibrillation": "Atrial fibrillation", "BundleBranchBlock": "Bundle branch block",
    "AVblock": "Atrioventricular block", "Brugada": "Brugada syndrome",
    "QTinterval": "QT interval", "RheumatoidArthritis": "Rheumatoid arthritis (control)",
    "RheumatoidArthritis_noMHC": "Rheumatoid arthritis, MHC-free",
}


def main() -> None:
    cov = pd.read_csv(COV, sep="\t", index_col=0)
    obs = pd.read_csv(OBS, sep="\t", index_col=0)[["cell_state"]]

    rows = []
    for key, label in TRAITS.items():
        f = SCORES / f"{key}.score.tsv"
        if not f.exists():
            raise SystemExit(f"missing score file: {f}")
        s = pd.read_csv(f, sep="\t", index_col=0)[["raw_score", "norm_score"]]
        j = s.join(cov, how="inner").join(obs, how="inner")
        for scope, sub in [("All cells", j)] + [
            (name, j[j.cell_state == code]) for code, name in POPS.items()
        ]:
            if len(sub) < 6:
                continue
            row = {"Trait": label, "Population": scope, "n": len(sub)}
            for c, cl in COVARIATES.items():
                row[f"rho_norm_{c}"] = round(spearmanr(sub.norm_score, sub[c]).statistic, 3)
            row["rho_raw_log_total"] = round(spearmanr(sub.raw_score, sub.log_total).statistic, 3)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TSV, sep="\t", index=False)

    allc = df[df.Population == "All cells"]
    worst = allc.rho_norm_log_total.abs().max()
    summary = {
        "n_cells": int(allc.n.iloc[0]),
        "n_traits": int(allc.Trait.nunique()),
        "max_abs_rho_norm_log_total_all_cells": float(worst),
        "max_abs_rho_norm_any_covariate_all_cells": float(
            allc[[f"rho_norm_{c}" for c in COVARIATES]].abs().to_numpy().max()),
        "max_abs_rho_within_conduction": float(
            df[df.Population != "All cells"].rho_norm_log_total.abs().max()),
        "note": ("Spearman correlations of the normalized per-cell scDRS score with the "
                 "covariates regressed out during scoring. Not to be confused with the "
                 "per-spot gsMap correlations in results/library_size_confound.json, "
                 "which belong to the spatial arm."),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote {OUT_TSV.relative_to(ROOT)}  ({len(df)} rows)")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"\n  {summary['n_traits']} traits x {summary['n_cells']:,} cells")
    print(f"  largest |rho| with total UMI, all cells:   {worst:.3f}")
    print(f"  largest |rho| with any covariate, all cells: "
          f"{summary['max_abs_rho_norm_any_covariate_all_cells']:.3f}")
    print(f"  largest |rho| within a conduction population: "
          f"{summary['max_abs_rho_within_conduction']:.3f}")


if __name__ == "__main__":
    main()
