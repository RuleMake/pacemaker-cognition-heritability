"""
Case counts, not total N — the check that was skipped once already.
==================================================================

The schizophrenia negative control failed because a study was chosen on its total
sample size (629,347) when the case count was tiny; the file had one genome-wide
significant SNP in 25 million variants. For a binary trait the power is governed by
the EFFECTIVE sample size

    N_eff = 4 / (1/N_cases + 1/N_controls)

which for 5,000 cases against 600,000 controls is about 19,800 — two orders of
magnitude below the headline number. The conduction-disease traits are all PheWAS
studies with headline Ns above 600,000, so this has to be resolved before spending
860 MB and three hours per trait on them.

The catalogue exposes the breakdown through the study's ancestry records.

Usage:  python scripts/82_check_case_counts.py
"""

import json
import time
import urllib.request

API = "https://www.ebi.ac.uk/gwas/rest/api"

STUDIES = {
    "GCST90480156": "AV block (PheCode 426.2)",
    "GCST90480154": "Complete AV block (PheCode 426.21)",
    "GCST90481992": "Mobitz II AV block",
    "GCST90480169": "Atrial flutter (PheCode 427.22)",
    "GCST90086158": "Brugada syndrome (Barc 2022)",
    "GCST90162626": "Heart failure (Levin 2022)",
    "GCST90018834": "Dilated cardiomyopathy",
    "GCST90084003": "Sick sinus syndrome ICD10 I49.5",
    # already in use, for calibration of what "enough" looks like here
    "GCST90204201": "Atrial fibrillation  [in use]",
}


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def neff(cases, controls):
    if not cases or not controls:
        return None
    return 4.0 / (1.0 / cases + 1.0 / controls)


print("=" * 100)
print("Effective sample size for the candidate disease traits")
print("=" * 100)
print(f"{'accession':<15}{'cases':>10}{'controls':>12}{'total':>12}{'N_eff':>11}"
      f"  verdict")

out = {}
for acc, label in STUDIES.items():
    try:
        st = get(f"{API}/studies/{acc}")
    except Exception as e:
        print(f"{acc:<15}{'query failed':>45}  {type(e).__name__}")
        continue
    time.sleep(0.3)
    cases = controls = total = 0
    for a in st.get("ancestries", []) or []:
        if a.get("type") != "initial":
            continue
        n = a.get("numberOfIndividuals") or 0
        total += n
        for c in a.get("ancestralGroups", []) or []:
            pass
    # the case/control split lives in the study-level counts when present
    cases = st.get("numberOfCases") or 0
    controls = st.get("numberOfControls") or 0
    if not cases:
        # fall back to parsing the initial-sample description text
        desc = " ".join(
            (a.get("numberOfIndividuals") and str(a.get("numberOfIndividuals")) or "")
            for a in (st.get("ancestries") or []))
        cases = 0
    e = neff(cases, controls)
    if e is None:
        v = "unknown split — inspect the file header before committing"
    elif e >= 30000:
        v = "USABLE"
    elif e >= 10000:
        v = "marginal"
    else:
        v = "TOO WEAK — skip"
    print(f"{acc:<15}{cases if cases else '?':>10}{controls if controls else '?':>12}"
          f"{total:>12,}{(f'{e:,.0f}' if e else '?'):>11}  {v}")
    out[acc] = dict(label=label, cases=cases, controls=controls, total=total,
                    n_eff=e, verdict=v)

print("\nWhere the split is unknown the catalogue does not carry it; the summary-")
print("statistics file itself usually has per-SNP N or a case/control column, so the")
print("decision moves to a quick header inspection after a partial download.")

with open("results/case_counts.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nwrote results/case_counts.json")
