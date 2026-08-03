"""
Pre-registered predictions for the whole-axis run.
==================================================

WRITTEN BEFORE THE SCORING RUN. This file exists so the anatomical specificity claim
can fail. Every trait scored so far is an atrial or nodal readout, so every trait has
had the same predicted answer, and twelve confirmations of a prediction that was never
at risk are worth less than they look.

Three traits now enter whose predicted cell has never won:

    QRS duration        His-Purkinje conduction time   -> Purkinje / AVN_bundle_cell
    bundle branch block disease of the same structure  -> Purkinje / AVN_bundle_cell
    QT interval         ventricular repolarisation     -> ventricular myocytes,
                                                          and NO conduction cell

and the atlas now contains the cell they point at: 110 Purkinje cells from 12 donors,
in apex tissue the old subset never included.

The claim being tested is not "conduction traits hit conduction cells". It is the
strong form:

    within the conduction system, each trait hits the SPECIFIC structure that
    generates its phenotype, and not the others.

The sinoatrial node and the Purkinje network are both pacemaker-adjacent conduction
tissue with overlapping ion-channel programmes. A method that merely notices
"electrically active cell" will rank them together. Anatomy says they must separate,
and in a direction fixed in advance: heart-rate traits to the sinoatrial node, QRS
traits to Purkinje, and neither to the other.

FALSIFICATION, stated in advance
--------------------------------
  * If QRS duration and bundle branch block rank SAN_P_cell above Purkinje, the map
    is not measuring anatomy and the specificity argument fails.
  * If QT interval ranks any conduction cell in its top 5, the ventricular-myocyte
    prediction fails and the map is measuring "cardiac excitability", not structure.
  * If every trait ranks every conduction cell highly, the map is measuring cell
    class, not anatomical position, and the whole figure should be dropped.

Power caveat, also stated in advance
------------------------------------
QRS duration has only N = 30,975 — the larger study is an exome chip that cannot map
onto the reference panel. Bundle branch block is a PheWAS trait from the same family
as atrioventricular block, which fell below the power floor. A POSITIVE result from an
underpowered trait is still evidence; a NULL from one means nothing. Both are reported
with their genome-wide significant counts alongside, and no null from either will be
counted as a failed prediction.

Usage:  python scripts/152_axis_predictions.py          (writes the registry)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.as_posix()
OUT = f"{ROOT}/results/axis_predictions.json"

CONDUCTION = ["SAN_P_cell", "AVN_P_cell", "AVN_bundle_cell", "Purkinje"]

# predicted winner among the conduction cell types, and the predicted cell class
# overall. `None` for conduction means "no conduction cell should win".
PREDICTIONS = {
    "HRV_RMSSD":         dict(conduction="SAN_P_cell", overall="SAN_P_cell",
                              why="迷走调控窦房结"),
    "HRV_RMSSDc":        dict(conduction="SAN_P_cell", overall="SAN_P_cell",
                              why="同上"),
    "HRV_SDNN":          dict(conduction="SAN_P_cell", overall="SAN_P_cell",
                              why="同上"),
    "HRV_SDNNc":         dict(conduction="SAN_P_cell", overall="SAN_P_cell",
                              why="同上"),
    "RestingHeartRate":  dict(conduction="SAN_P_cell", overall="SAN_P_cell",
                              why="窦房结固有频率"),
    "PRinterval":        dict(conduction="AVN_P_cell", overall="AVN_P_cell",
                              why="房室结传导延迟"),
    "AVblock":           dict(conduction="AVN_P_cell", overall="AVN_P_cell",
                              why="房室结疾病本身"),
    "QRSduration":       dict(conduction=["Purkinje", "AVN_bundle_cell"],
                              overall=["Purkinje", "AVN_bundle_cell"],
                              why="希浦系统传导时间 —— 新预测，从未有性状命中"),
    "BundleBranchBlock": dict(conduction=["Purkinje", "AVN_bundle_cell"],
                              overall=["Purkinje", "AVN_bundle_cell"],
                              why="束支/希浦系统疾病本身 —— 新预测"),
    "QTinterval":        dict(conduction=None, overall="vCM",
                              why="心室复极 —— 预测任何传导细胞都不进前 5"),
    "Brugada":           dict(conduction=["Purkinje", "AVN_bundle_cell"],
                              overall="vCM",
                              why="右室流出道传导；已知压低结区起搏细胞"),
    "AtrialFibrillation": dict(conduction=None, overall="aCM",
                               why="心房肌 —— 阳性对照"),
    "AtrialFlutter":     dict(conduction=None, overall="aCM",
                              why="心房肌，峡部折返"),
    "HeartFailure":      dict(conduction=None, overall=["vCM", "SMC"],
                              why="心室肌/血管"),
    "EducationalAttainment": dict(conduction=None, overall="neural",
                                  why="阴性对照 —— 神经细胞"),
    "RheumatoidArthritis": dict(conduction=None, overall="immune",
                                why="阴性对照 —— 免疫细胞"),
    "RheumatoidArthritis_noMHC": dict(conduction=None, overall="immune",
                                      why="同上，剔除 MHC"),
}

# the three that make this a test rather than a confirmation
NOVEL = ["QRSduration", "BundleBranchBlock", "QTinterval"]

if __name__ == "__main__":
    reg = dict(conduction_cells=CONDUCTION, predictions=PREDICTIONS, novel=NOVEL,
               registered_before_scoring=True)
    with open(OUT, "w") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)
    print(f"registered {len(PREDICTIONS)} predictions -> {OUT}")
    print(f"\n{'trait':<28}{'conduction winner':<34}overall")
    for t, p in PREDICTIONS.items():
        c = p["conduction"]
        c = "none of them" if c is None else (c if isinstance(c, str) else " or ".join(c))
        o = p["overall"]
        o = o if isinstance(o, str) else " or ".join(o)
        star = " *" if t in NOVEL else ""
        print(f"{t:<28}{c:<34}{o}{star}")
    print("\n* = predicted cell has never won before; these are the falsification tests")
