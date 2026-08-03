"""
One donor dissents. Is that a failure to replicate, or a failure of annotation?
===============================================================================

Leave-one-donor-out (`201`) is clean: no cognitive result depends on any single donor.
But the per-donor table shows one donor, A61, sitting below 0.5 for almost everything
in its sinoatrial cells — educational attainment 0.479, intelligence 0.449, heart rate
variability 0.488, resting heart rate 0.266.

The tempting move is to drop it and say the rest replicate. That move is only legitimate
if the exclusion criterion is INDEPENDENT of the outcome being measured. Dropping a
donor because it disagrees with the hypothesis is how a 5/6 becomes a 6/6 and how a
paper becomes wrong.

So the criterion here is not the trait scores. It is marker identity: do this donor's
"sinoatrial pacemaker cells" actually express the pacemaker programme? The panel is the
same fixed one used to validate the Purkinje annotation before any trait was scored —
HCN4, SHOX2, TBX3, ISL1, VSNL1, CACNA1D — and it knows nothing about GWAS.

Two outcomes, both publishable, and they say opposite things:

  A61 also fails the marker check   the cells are mislabelled or of poor quality. The
                                    exclusion is justified on QC grounds, stated as
                                    such, and the trait result is reported both ways.
  A61 passes the marker check       these are genuine pacemaker cells that do not carry
                                    the signal. That is real between-donor heterogeneity
                                    and it must be reported as a limitation, not hidden.

Note which control decides it: resting heart rate at 0.266 is a POSITIVE control
failing. If a donor cannot reproduce the single best-established fact in this project —
that heart rate genetics is enriched in pacemaker cells — the question of whether it
reproduces the cognitive finding was never well posed.

Usage:  python scripts/202_donor_identity_qc.py
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.as_posix()
H5 = f"{ROOT}/data/singlecell/axis_subset.h5ad"
LODO = f"{ROOT}/results/leave_one_donor_out.json"
OUT = f"{ROOT}/results/donor_identity_qc.json"

PANEL = ["HCN4", "SHOX2", "TBX3", "ISL1", "VSNL1", "CACNA1D"]
FOCUS = "SAN_P_cell"
MIN_CM = 30
CHUNK = 10000

a = ad.read_h5ad(H5, backed="r")
obs = a.obs[["donor_id", "region", "assay", "cell_state"]].astype(str)
obs["stratum"] = obs.donor_id + "/" + obs.region + "/" + obs.assay

sym = a.var["feature_name"].astype(str) if "feature_name" in a.var.columns \
    else pd.Series(a.var_names, index=a.var_names)
idx = {g: int(np.where(sym.to_numpy() == g)[0][0])
       for g in PANEL if (sym.to_numpy() == g).any()}
missing = [g for g in PANEL if g not in idx]
print(f"panel genes found: {list(idx)}" + (f"   MISSING {missing}" if missing else ""))
if len(idx) < 4:
    raise SystemExit("too few panel genes present to judge identity")

# pull only the panel columns, in row chunks — never the whole matrix
cols = [idx[g] for g in PANEL if g in idx]
mat = np.zeros((a.n_obs, len(cols)), dtype=np.float32)
for s in range(0, a.n_obs, CHUNK):
    e = min(s + CHUNK, a.n_obs)
    mat[s:e] = np.asarray(a.X[s:e][:, cols].todense())
print(f"pulled {mat.shape[0]:,} x {mat.shape[1]} expression values")

state = obs.cell_state.to_numpy()
donor = obs.donor_id.to_numpy()
strat_codes, strat_names = pd.factorize(obs.stratum)
IS_CM = pd.Series(state).str.match(r"^(aCM|vCM)").to_numpy()
IS_FOCUS = state == FOCUS
KEEP = IS_CM | IS_FOCUS


def panel_auc(cells_mask):
    """Myocyte-referenced stratified AUC of the pacemaker panel, van Elteren weights.

    Genes are standardised WITHIN each stratum before averaging so that one highly
    expressed gene cannot dominate the panel, and so donor-level depth differences
    cannot move the score.
    """
    num = den = stat = var = 0.0
    ns = nc = 0
    for si in range(len(strat_names)):
        m = (strat_codes == si) & KEEP & cells_mask
        N = int(m.sum())
        if N < MIN_CM:
            continue
        sub = mat[m]
        sd = sub.std(0)
        sd[sd == 0] = 1.0
        v = ((sub - sub.mean(0)) / sd).mean(1)
        f = IS_FOCUS[m]
        n1 = int(f.sum())
        n2 = N - n1
        if n1 < 1 or n2 < MIN_CM:
            continue
        r = stats.rankdata(v)
        W = r[f].sum()
        num += (n1 * n2 / (N + 1)) * ((W - n1 * (n1 + 1) / 2) / (n1 * n2))
        den += n1 * n2 / (N + 1)
        stat += (W - n1 * (N + 1) / 2) / (N + 1)
        var += n1 * n2 / (12.0 * (N + 1))
        ns += 1
        nc += n1
    if den <= 0 or var <= 0:
        return None
    return dict(auc=float(num / den), z=float(stat / np.sqrt(var)),
                n_strata=ns, n_cells=nc)


ALL = np.ones(len(obs), dtype=bool)
overall = panel_auc(ALL)
print(f"\npacemaker panel, all donors: AUC {overall['auc']:.3f} "
      f"(z {overall['z']:+.1f}, {overall['n_cells']} cells)")

lodo = json.load(open(LODO))
donors = lodo["donors_of"][FOCUS]

print("\n" + "=" * 92)
print("PACEMAKER IDENTITY PER DONOR — the criterion, computed without any GWAS")
print("=" * 92)
print(f"{'donor':<10}{'n cells':>9}{'panel AUC':>12}{'z':>8}"
      f"{'RestHR (pos ctrl)':>20}{'EDU':>8}{'verdict':>14}")
rows = {}
for d in donors:
    r = panel_auc(donor == d)
    if r is None:
        continue
    hr = lodo["per_donor"].get(f"RestingHeartRate|{FOCUS}", {}) \
                          .get("per_donor", {}).get(d)
    ed = lodo["per_donor"].get(f"EducationalAttainment|{FOCUS}", {}) \
                          .get("per_donor", {}).get(d)
    # identity is judged on the panel alone; 0.5 is the no-difference line
    ok = r["auc"] > 0.5 and r["z"] > 1.96
    rows[d] = dict(panel=r, resting_hr=hr, edu=ed, identity_ok=bool(ok))
    print(f"{d:<10}{r['n_cells']:>9}{r['auc']:>12.3f}{r['z']:>8.2f}"
          f"{(hr if hr is not None else float('nan')):>20.3f}"
          f"{(ed if ed is not None else float('nan')):>8.3f}"
          f"{('pacemaker' if ok else 'FAILS PANEL'):>14}")

fails = [d for d, v in rows.items() if not v["identity_ok"]]

print("\n" + "=" * 92)
print("VERDICT")
print("=" * 92)
if fails:
    print(f"  {', '.join(fails)} do not express the pacemaker programme above the")
    print("  working myocytes of the same stratum. The exclusion criterion is the")
    print("  marker panel, decided before any trait was involved.")
    keep = ~np.isin(donor, fails)
    print("\n  Trait results with those donors removed (report BOTH numbers in the paper):")
    print(f"\n{'trait':<26}{'all donors':>12}{'QC-passing':>13}{'delta':>9}")
    import sys
    sys.path.insert(0, str(Path(ROOT) / "scripts"))
    # recompute trait AUCs on the QC-passing subset using 201's machinery
    SC = f"{ROOT}/results/scdrs_axis"
    qc = {}
    for t in ["EducationalAttainment", "Intelligence", "ReactionTime",
              "HRV_SDNN", "HRV_RMSSD", "RestingHeartRate", "RheumatoidArthritis"]:
        p = Path(SC) / f"{t}.score.tsv"
        if not p.exists():
            continue
        v = pd.read_csv(p, sep="\t", index_col=0)["norm_score"] \
              .reindex(obs.index).to_numpy()
        res = {}
        for lab, msk in (("all", ALL), ("qc", keep)):
            num = den = stat = var = 0.0
            for si in range(len(strat_names)):
                m = (strat_codes == si) & KEEP & msk
                N = int(m.sum())
                if N < MIN_CM:
                    continue
                f = IS_FOCUS[m]
                n1 = int(f.sum())
                n2 = N - n1
                if n1 < 1 or n2 < MIN_CM:
                    continue
                r = stats.rankdata(v[m])
                W = r[f].sum()
                num += (n1 * n2 / (N + 1)) * ((W - n1 * (n1 + 1) / 2) / (n1 * n2))
                den += n1 * n2 / (N + 1)
                stat += (W - n1 * (N + 1) / 2) / (N + 1)
                var += n1 * n2 / (12.0 * (N + 1))
            res[lab] = dict(auc=float(num / den), z=float(stat / np.sqrt(var)))
        qc[t] = res
        print(f"{t:<26}{res['all']['auc']:>12.3f}{res['qc']['auc']:>13.3f}"
              f"{res['qc']['auc'] - res['all']['auc']:>+9.3f}")
    rows["_qc_recompute"] = qc
    print("\n  The primary analysis stays the ALL-DONOR number. The QC-passing number")
    print("  is a sensitivity analysis, because excluding donors after seeing data is")
    print("  exactly the move that needs to be visible rather than quietly applied.")
else:
    print("  Every donor's annotated pacemaker cells express the pacemaker programme.")
    print("  The dissenting donor is genuine between-donor heterogeneity and must be")
    print("  reported as a limitation — there is no QC ground to exclude it.")

# ------------------------------------------------------------------ dose-response
print("\n" + "=" * 92)
print("DOSE-RESPONSE: does the signal track how pacemaker-like the cells are?")
print("=" * 92)
print("A binary 'does it replicate' wastes what the donors are telling us. If the")
print("enrichment is a property of pacemaker identity rather than of a batch, then")
print("donors whose cells are more pacemaker-like should show more of it.\n")

SC = f"{ROOT}/results/scdrs_axis"
TR = ["EducationalAttainment", "Intelligence", "ReactionTime", "HRV_SDNN",
      "RestingHeartRate", "RheumatoidArthritis"]
panel_by_donor = {d: rows[d]["panel"]["auc"] for d in rows if d in donors}
ds = sorted(panel_by_donor)
pv = np.array([panel_by_donor[d] for d in ds])

print(f"{'trait':<26}{'Spearman vs panel':>20}{'p':>9}   donor AUCs (panel order)")
dose = {}
order = [d for d in sorted(ds, key=lambda x: panel_by_donor[x])]
for t in TR:
    pd_ = lodo["per_donor"].get(f"{t}|{FOCUS}", {}).get("per_donor", {})
    if len(pd_) < len(ds):
        continue
    tv = np.array([pd_[d] for d in ds])
    rho, p = stats.spearmanr(pv, tv)
    dose[t] = dict(rho=float(rho), p=float(p),
                   by_panel_order=[float(pd_[d]) for d in order])
    seq = " ".join(f"{pd_[d]:.2f}" for d in order)
    print(f"{t:<26}{rho:>20.3f}{p:>9.3f}   {seq}")
print(f"{'':<26}{'':>20}{'':>9}   panel: "
      + " ".join(f"{panel_by_donor[d]:.2f}" for d in order))

# The obvious objection: if a trait's gene set CONTAINS the panel genes, this is
# circular. Check it rather than argue about it.
print("\n  gene-set overlap with the marker panel (circularity check):")
GS = Path(ROOT) / "data/scdrs/traits.gs"
sets = {}
if GS.exists():
    for line in GS.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            sets[parts[0]] = {g.split(":")[0] for g in parts[1].split(",")}
overlap = {}
for t in TR:
    if t not in sets:
        print(f"    {t:<26} gene set not found in traits.gs")
        continue
    hits = sorted(set(PANEL) & sets[t])
    overlap[t] = hits
    if hits:
        print(f"    {t:<26} contains {hits}  <-- circular, discount this row")
    else:
        print(f"    {t:<26} contains none of the panel")

# Rather than discount rows, remove the circularity: keep only panel genes that appear
# in NO trait's gene set. What survives are the canonical pacemaker transcription
# factors, which is the cleaner definition of identity anyway.
used = set().union(*sets.values()) if sets else set()
CLEAN = [g for g in PANEL if g not in used]
print(f"\n  circularity-free panel: {CLEAN}")
clean_dose = {}
if len(CLEAN) >= 2:
    ccols = [i for i, g in enumerate([g for g in PANEL if g in idx]) if g in CLEAN]
    mat_full = mat
    mat = mat[:, ccols]
    cpanel = {d: panel_auc(donor == d)["auc"] for d in ds}
    cv = np.array([cpanel[d] for d in ds])
    corder = sorted(ds, key=lambda x: cpanel[x])
    print(f"\n{'trait':<26}{'Spearman':>10}{'p':>9}   donor AUCs (clean-panel order)")
    for t in TR:
        pd_ = lodo["per_donor"].get(f"{t}|{FOCUS}", {}).get("per_donor", {})
        if len(pd_) < len(ds):
            continue
        rho, p = stats.spearmanr(cv, np.array([pd_[d] for d in ds]))
        clean_dose[t] = dict(rho=float(rho), p=float(p))
        print(f"{t:<26}{rho:>10.3f}{p:>9.3f}   "
              + " ".join(f"{pd_[d]:.2f}" for d in corder))
    print(f"{'':<26}{'':>10}{'':>9}   panel: "
          + " ".join(f"{cpanel[d]:.2f}" for d in corder))
    mat = mat_full

# ------------------------------------------------------------------ cell level
print("\n" + "=" * 92)
print("SAME QUESTION AT CELL LEVEL — 245 cells instead of 6 donors")
print("=" * 92)
print("Within each stratum, rank the pacemaker cells by how strongly they express the")
print("identity panel and ask whether that ordering predicts their trait score. This")
print("is the same design as the ambient-RNA test, pointed at a different covariate,")
print("and it has forty times the sample size of the donor-level version.\n")
ccols = [i for i, g in enumerate([g for g in PANEL if g in idx]) if g in CLEAN]
pscore = np.full(len(obs), np.nan)
for si in range(len(strat_names)):
    m = (strat_codes == si) & IS_FOCUS
    if m.sum() < 5:
        continue
    sub = mat[m][:, ccols]
    sd = sub.std(0)
    sd[sd == 0] = 1.0
    pscore[m] = ((sub - sub.mean(0)) / sd).mean(1)

print(f"{'trait':<26}{'pooled rho':>12}{'p':>10}{'n cells':>9}{'n strata':>10}")
cell_level = {}
for t in TR:
    p = Path(SC) / f"{t}.score.tsv"
    if not p.exists():
        continue
    v = pd.read_csv(p, sep="\t", index_col=0)["norm_score"].reindex(obs.index).to_numpy()
    zs, ws, n = [], [], 0
    for si in range(len(strat_names)):
        m = (strat_codes == si) & IS_FOCUS & ~np.isnan(pscore)
        k = int(m.sum())
        if k < 10:
            continue
        r, _ = stats.spearmanr(pscore[m], v[m])
        if np.isnan(r):
            continue
        zs.append(np.arctanh(np.clip(r, -0.999, 0.999)))
        ws.append(k - 3)
        n += k
    if not zs:
        continue
    z = float(np.average(zs, weights=ws))
    se = float(1 / np.sqrt(sum(ws)))
    pval = float(2 * stats.norm.sf(abs(z / se)))
    cell_level[t] = dict(rho=float(np.tanh(z)), p=pval, n_cells=n, n_strata=len(zs))
    print(f"{t:<26}{np.tanh(z):>12.3f}{pval:>10.4f}{n:>9}{len(zs):>10}")
print("\n  Fisher-z combination across strata, weighted by n-3. A positive rho means")
print("  the more pacemaker-like a cell is, the higher its trait score — dose-response")
print("  inside a single cell type, which no batch or donor effect can produce.")

json.dump(dict(panel=PANEL, overall=overall, per_donor=rows, fails_panel=fails,
               dose_response=dose, panel_overlap=overlap, clean_panel=CLEAN,
               dose_response_clean=clean_dose, cell_level=cell_level),
          open(OUT, "w"), indent=2, default=float)
print(f"\nwrote {OUT}")
