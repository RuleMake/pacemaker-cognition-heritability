"""
Fetch the conduction-disease GWAS, inspecting each before committing to it.
===========================================================================

Path B: the project has only ever tested physiological outputs. Resting heart rate is
what the sinoatrial node produces, not what goes wrong with it; the disease of that
tissue is sick sinus syndrome, and of the atrioventricular node, heart block. Eight
AVN sections have been sitting unused against any disease of the AVN.

Every file is inspected before it is trusted. Four things have already gone wrong here
and each is now handled explicitly rather than by assumption:

  odds ratios      Case-control studies report OR, not beta, and the first version
                   demanded a column literally named beta — discarding all three
                   PheWAS traits this arm exists to test. log(OR) IS the beta.
  ambiguous SE     Whether `standard_error` refers to the OR or to log(OR) is not
                   always stated. Z is therefore derived from the p-value and the sign
                   of the effect, which is exact, and cross-checked against beta/se.
  missing N        MAGMA weights every gene by sample size, so a default would corrupt
                   every result silently. Where no N column exists it is recovered
                   from the standard error and allele frequency, and where that fails
                   the trait is rejected rather than guessed at.
  no rsIDs         A file may fill `variant_id` with chr:pos. On build 37 those map
                   back through the reference panel; the match rate is the guard, so a
                   build mismatch is rejected instead of silently mis-assigning rsIDs.

Files are read in chunks. Two of these are 18 million rows and reading one whole while
scDRS holds 6-8 GB was killed by the kernel.

Runs from WSL — the reference bim files live there with the gsMap resources.

Usage:  python scripts/85_fetch_disease_traits.py
        python scripts/85_fetch_disease_traits.py --only AVblock,DCM
"""

import argparse
import gzip
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent.as_posix()
RAW = f"{ROOT}/data/gwas"
OUT = f"{ROOT}/data/gwas_gsmap"
REFDIR = os.path.expanduser(
    "~/cardio/resource/gsMap_resource/LD_Reference_Panel/1000G_EUR_Phase3_plink")
os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

FTP = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"
TRAITS = [
    ("AVblock", "GCST90480156", "GCST90480001-GCST90481000",
     "房室结疾病本身 — 对应 8 张 AVN 切片，从未测过"),
    ("AVblockComplete", "GCST90480154", "GCST90480001-GCST90481000",
     "完全性阻滞，表型更纯"),
    ("AtrialFlutter", "GCST90480169", "GCST90480001-GCST90481000",
     "峡部依赖折返 — 与房颤机制不同，同组织对照"),
    ("Brugada", "GCST90086158", "GCST90086001-GCST90087000",
     "右室流出道传导 — 单祖先、GRCh37"),
    ("HeartFailure", "GCST90162626", "GCST90162001-GCST90163000",
     "大性状阳性对照 — 应定位到心室肌"),
    ("DCM", "GCST90018834", "GCST90018001-GCST90019000",
     "扩张型心肌病 — 应定位到心室肌"),
    # --- 心室传导轴（2026-08-01 追加）。前面每一个性状都是心房或结区读数，
    # 所以每一个的预期答案都相同，特异性图谱因此只被证实、从未被证伪。
    # 图谱里现在有 110 个 Purkinje 细胞（心尖组织），下面三个性状各自指向
    # 不同的、此前从未命中过的细胞，是真正可证伪的检验。
    # 最大的 QRS 研究 GCST007103（N=77,898）是 exome chip，位点几乎全是罕见编码变异，
    # 与 1000G EUR 参考面板重合极低，MAGMA 无法用。退而取 2025 年这项常规全基因组
    # 关联（N=30,975），功效偏低必须并列报告：阳性有信息量，阴性没有。
    ("QRSduration", "GCST90624624", "GCST90624001-GCST90625000",
     "希浦系统传导时间 — 预测 Purkinje，不应是窦房结"),
    ("QTinterval", "GCST90165291", "GCST90165001-GCST90166000",
     "心室复极 — 预测心室肌，不应是任何传导细胞"),
    ("BundleBranchBlock", "GCST90475956", "GCST90475001-GCST90476000",
     "束支阻滞 — 希浦系统疾病本身，与房室阻滞同族 PheWAS"),
    # --- 认知性状（2026-08-01 追加）。教育程度与心率变异性的遗传相关 rg=+0.26，
    # 而与房颤/QT 为零。若这个形状在其他认知性状上重现，它就是一条规律而非一个观察。
    # 两项都是欧裔、连续性状，可同时用于 rg 与 MR。
    ("Intelligence", "GCST006250", "GCST006001-GCST007000",
     "智力 — Savage 2018，N=269,867 欧裔"),
    ("ReactionTime", "GCST90179115", "GCST90179001-GCST90180000",
     "反应时 — N=432,297 欧裔，认知加工速度"),
]

# Sample sizes taken from the GWAS Catalog study record, for depositions whose own
# files cannot supply a trustworthy one. Only add an entry when the catalogue value has
# actually been checked against the publication.
N_OVERRIDE = {
    "Intelligence": 269_867,     # Savage 2018, European ancestry
    "ReactionTime": 432_297,     # GCST90179115, European ancestry
}

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--min-gws", type=int, default=150)
ap.add_argument("--chunk", type=int, default=2_000_000)
args = ap.parse_args()
want = {s.strip() for s in args.only.split(",") if s.strip()}

RS = ["rs_id", "rsid", "variant_id", "SNP", "snp", "rsids"]
EA = ["effect_allele", "EA", "A1", "alt"]
OA = ["other_allele", "OA", "A2", "ref"]
BE = ["beta", "BETA", "effect_size", "b"]
ORC = ["odds_ratio", "OR", "or"]
SE = ["standard_error", "SE", "se", "standard_error_of_beta"]
PV = ["p_value", "P", "pval", "p", "pvalue"]
NC = ["n", "N", "sample_size", "n_total"]
NCASE = ["n_cases", "N_cases", "case_count", "ncase"]
NCTRL = ["n_controls", "N_controls", "control_count", "ncontrol"]
AF = ["effect_allele_frequency", "eaf", "freq", "AF"]
CHR = ["chromosome", "chr", "CHR", "#CHROM"]
BP = ["base_pair_location", "bp", "pos", "position", "BP"]


def pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def peek(path):
    # not every deposition is gzipped; pandas infers compression from the suffix, so
    # only the header sniff needs to branch
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        head = next(f)
    sep = "\t" if "\t" in head else ("," if "," in head else r"\s+")
    return pd.read_csv(path, sep=sep, nrows=50_000, low_memory=False), sep


def load_bim():
    parts = []
    for ch in range(1, 23):
        p = f"{REFDIR}/1000G.EUR.QC.{ch}.bim"
        if os.path.exists(p):
            parts.append(pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 3],
                                     names=["chrom", "SNP", "bp"]))
    return pd.concat(parts, ignore_index=True) if parts else None


def reject(name, raw, reason, **extra):
    print(f"    !! REJECTED: {reason}")
    if raw and os.path.exists(raw):
        os.remove(raw)
    return dict(status="rejected", reason=reason, **extra)


report = {}
BIM = None

for name, acc, bucket, why in TRAITS:
    if want and name not in want:
        continue
    target = f"{OUT}/{name}.sumstats.gz"
    if os.path.exists(target):
        print(f"[skip] {name} already converted")
        continue

    print(f"\n{'=' * 88}\n=== {name}  ({acc})\n    {why}", flush=True)
    url = f"{FTP}/{bucket}/{acc}/"

    def listing(u):
        """Usable data files at a URL, best format first.

        Older depositions do not follow the modern naming at all: one study here
        publishes a bare `.tsv`, another only a `.zip`, and both were rejected as
        'no data file' when the matcher demanded `.tsv.gz`. Uncompressed files are
        accepted, and `harmonised/` is tried when the top level offers nothing —
        that subdirectory carries the GWAS Catalog's standardised columns and is
        usually the better input anyway.
        """
        try:
            with urllib.request.urlopen(u, timeout=60) as r:
                html = r.read().decode("utf-8", "replace")
        except Exception:
            return []
        f = re.findall(r'href="([^"?/][^"]*)"', html)
        good = [x for x in f if x.endswith((".tsv.gz", ".txt.gz", ".tsv", ".txt"))
                and not x.endswith(("-meta.yaml", ".md5", "readme.txt"))]
        # harmonised files: prefer `.h.` (fully harmonised) over `.f.` (formatted)
        good.sort(key=lambda x: (0 if ".h.tsv" in x else 1, 0 if x.endswith(".gz")
                                 else 1))
        return good

    files = listing(url)
    sub = ""
    if not files:
        files = listing(url + "harmonised/")
        sub = "harmonised/"
    if not files:
        report[name] = reject(name, None, "no data file in the FTP listing")
        continue
    fname = files[0]
    url = url + sub

    raw = f"{RAW}/{acc}" + (".tsv.gz" if fname.endswith(".gz") else ".tsv")
    print(f"    downloading {fname} ...", flush=True)
    rc = subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "5",
                         "--speed-limit", "50000", "--speed-time", "60",
                         "-s", "-o", raw, url + fname]).returncode
    if rc != 0 or not os.path.exists(raw):
        report[name] = reject(name, raw, f"download failed (curl {rc})")
        continue
    print(f"    {os.path.getsize(raw) / 1e6:.0f} MB", flush=True)

    try:
        sample, sep = peek(raw)
    except Exception as e:
        report[name] = reject(name, raw, f"unreadable: {type(e).__name__}")
        continue
    cols = list(sample.columns)
    print(f"    columns: {', '.join(cols[:12])}{' ...' if len(cols) > 12 else ''}")

    c_rs, c_be, c_or = pick(cols, RS), pick(cols, BE), pick(cols, ORC)
    c_se, c_p, c_n = pick(cols, SE), pick(cols, PV), pick(cols, NC)
    c_ea, c_oa, c_af = pick(cols, EA), pick(cols, OA), pick(cols, AF)
    c_case, c_ctrl = pick(cols, NCASE), pick(cols, NCTRL)
    c_chr, c_bp = pick(cols, CHR), pick(cols, BP)

    frac_rs = (sample[c_rs].astype(str).str.startswith("rs").mean()
               if c_rs else 0.0)
    map_by_pos = frac_rs < 0.5
    if c_rs:
        print(f"    '{c_rs}': {frac_rs * 100:.0f}% look like rsIDs")
    if map_by_pos:
        if c_chr is None or c_bp is None:
            report[name] = reject(name, raw, "no rsIDs and no coordinates")
            continue
        print(f"    mapping {c_chr}:{c_bp} against the build 37 reference")
        if BIM is None:
            BIM = load_bim()
        if BIM is None:
            report[name] = reject(name, raw, f"reference bim not found at {REFDIR}")
            continue

    if c_be is None and c_or is None:
        report[name] = reject(name, raw, "neither beta nor odds ratio present")
        continue
    if c_se is None and c_p is None:
        report[name] = reject(name, raw, "no standard error and no p-value")
        continue

    print(f"    effect {c_be or c_or}{'' if c_be else ' (odds ratio -> log)'}"
          f"   se {c_se}   p {c_p}   af {c_af}", flush=True)

    use = [c for c in [c_rs, c_ea, c_oa, c_be, c_or, c_se, c_p, c_n, c_case,
                       c_ctrl, c_af] if c]
    if map_by_pos:
        use = [c for c in use if c != c_rs] + [c_chr, c_bp]
    use = list(dict.fromkeys(use))

    parts, n_in, n_kept = [], 0, 0
    zse_num, zp_num = [], []
    for chunk in pd.read_csv(raw, sep=sep, usecols=use, chunksize=args.chunk,
                             low_memory=False):
        n_in += len(chunk)
        if map_by_pos:
            chunk["chrom"] = pd.to_numeric(
                chunk[c_chr].astype(str).str.replace("chr", "", case=False),
                errors="coerce")
            chunk["bp"] = pd.to_numeric(chunk[c_bp], errors="coerce")
            chunk = chunk.merge(BIM, on=["chrom", "bp"], how="inner")
            rs = chunk["SNP"].astype(str)
        else:
            chunk = chunk[chunk[c_rs].astype(str).str.startswith("rs")]
            rs = chunk[c_rs].astype(str)
        if chunk.empty:
            continue
        n_kept += len(chunk)

        if c_be:
            beta = pd.to_numeric(chunk[c_be], errors="coerce").to_numpy(dtype=float)
        else:
            o = pd.to_numeric(chunk[c_or], errors="coerce").to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                beta = np.log(np.where(o > 0, o, np.nan))

        se = (pd.to_numeric(chunk[c_se], errors="coerce").to_numpy(dtype=float)
              if c_se else None)
        z_se = (np.where((se is not None) & np.isfinite(se) & (se > 0), beta / se,
                         np.nan) if se is not None else None)
        z_p = None
        if c_p:
            pv = np.clip(pd.to_numeric(chunk[c_p], errors="coerce")
                         .to_numpy(dtype=float), 1e-300, 1.0)
            z_p = np.sign(beta) * norm.isf(pv / 2.0)
        if z_se is not None and z_p is not None:
            ok = np.isfinite(z_se) & np.isfinite(z_p) & (np.abs(z_p) > 2)
            if ok.sum():
                zse_num.append(np.abs(z_se[ok]))
                zp_num.append(np.abs(z_p[ok]))

        af = (pd.to_numeric(chunk[c_af], errors="coerce").to_numpy(dtype=float)
              if c_af else np.full(len(chunk), np.nan))
        parts.append(pd.DataFrame({
            "SNP": rs.to_numpy(),
            "A1": (chunk[c_ea].astype(str).str.upper().to_numpy() if c_ea else "A"),
            "A2": (chunk[c_oa].astype(str).str.upper().to_numpy() if c_oa else "G"),
            "z_se": z_se if z_se is not None else np.nan,
            "z_p": z_p if z_p is not None else np.nan,
            "se": se if se is not None else np.nan,
            "af": af,
            "n": (pd.to_numeric(chunk[c_n], errors="coerce").to_numpy()
                  if c_n else np.nan),
            "ncase": (pd.to_numeric(chunk[c_case], errors="coerce").to_numpy()
                      if c_case else np.nan),
            "nctrl": (pd.to_numeric(chunk[c_ctrl], errors="coerce").to_numpy()
                      if c_ctrl else np.nan),
        }))

    if not parts:
        report[name] = reject(name, raw, "no usable rows after filtering")
        continue
    d = pd.concat(parts, ignore_index=True)
    del parts
    if map_by_pos:
        rate = n_kept / max(n_in, 1)
        print(f"    coordinate match {n_kept:,}/{n_in:,} ({rate * 100:.1f}%)")
        if rate < 0.20:
            report[name] = reject(name, raw, f"build mismatch, only "
                                             f"{rate * 100:.0f}% of positions matched")
            continue

    # which Z route to trust
    use_se = False
    if zse_num and zp_num:
        ratio = float(np.median(np.concatenate(zse_num))
                      / max(np.median(np.concatenate(zp_num)), 1e-12))
        agree = 0.9 < ratio < 1.1
        print(f"    |Z| beta/se vs p-value: ratio {ratio:.3f} "
              f"({'consistent' if agree else 'DISAGREE — using the p-value route'})")
        use_se = agree
    elif d.z_se.notna().any() and not d.z_p.notna().any():
        use_se = True
    z = d.z_se.to_numpy() if use_se else d.z_p.to_numpy()
    keep = np.isfinite(z)
    d, z = d[keep].reset_index(drop=True), z[keep]

    # effective sample size
    #
    # A published N from the GWAS Catalog outranks anything derived here, and using it
    # is not the "guessing" this script refuses to do — it is the study's own metadata.
    # It is needed in two situations that both occurred:
    #   * the harmonised file carries no N column at all and the SE route fails, so
    #     the trait is rejected outright (intelligence);
    #   * the SE route succeeds but is wrong, because it assumes the phenotype was
    #     analysed on a unit-variance scale. Reaction time came back at 925,884
    #     against a catalogue value of 432,297 — a 2.1x overestimate, which would have
    #     halved its LDSC heritability and distorted every genetic correlation it
    #     appears in.
    if N_OVERRIDE.get(name):
        n_eff = float(N_OVERRIDE[name])
        print(f"    N from the GWAS Catalog: {n_eff:,.0f} (overriding any estimate)")
    elif d.ncase.notna().any() and d.nctrl.notna().any():
        nc, nk = float(d.ncase.median()), float(d.nctrl.median())
        n_eff = 4.0 / (1.0 / nc + 1.0 / nk)
        print(f"    cases {nc:,.0f}  controls {nk:,.0f}  N_eff {n_eff:,.0f}")
    elif d.n.notna().any():
        n_eff = float(d.n.median())
        print(f"    per-SNP N median {n_eff:,.0f}")
    elif d.af.notna().any() and d.se.notna().any():
        # The standard error carries the sample size through the allele frequency:
        #   quantitative trait   se^2 = 1 / (2 N f (1-f))          -> N   = 1/(2 f(1-f) se^2)
        #   case-control, logOR  se^2 = 2 / (N_eff f (1-f))        -> N_eff = 2/(f(1-f) se^2)
        # The two differ by a factor of four, so the case-control form is used whenever
        # the effect came in as an odds ratio.
        #
        # Scope of this number: MAGMA's snp-wise mean model derives each gene p-value
        # from the SNP p-values and the LD among them — N is recorded, not used, so a
        # misestimate does not move the gene ranking that scDRS consumes. It must not
        # be read as a power metric across traits; genome-wide significant count and
        # lambda_GC, both reported above, are the honest ones.
        f = d.af.to_numpy(dtype=float)
        s = d.se.to_numpy(dtype=float)
        m = np.isfinite(f) & (f > 0.05) & (f < 0.95) & np.isfinite(s) & (s > 0)
        const = 2.0 if c_or else 0.5          # case-control vs quantitative
        n_eff = (float(np.median(const / (f[m] * (1 - f[m]) * s[m] ** 2)))
                 if m.sum() > 1000 else float("nan"))
        print(f"    N_eff from allele frequency and SE: {n_eff:,.0f} "
              f"({m.sum():,} common variants, "
              f"{'case-control' if c_or else 'quantitative'} form)")
    else:
        n_eff = float("nan")
        print("    no sample-size information available")

    n_gws = int((np.abs(z) > 5.45).sum())
    lam = float(np.median(z ** 2) / 0.4549)
    print(f"    {len(d):,} SNPs, |Z|max {np.abs(z).max():.1f}, "
          f"gw-sig {n_gws:,}, lambda_GC {lam:.3f}")

    if not np.isfinite(n_eff) or n_eff < 1000:
        report[name] = reject(name, raw, "effective sample size not establishable",
                              n_snp=len(d), n_gws=n_gws, lambda_gc=lam)
        continue
    if n_gws < args.min_gws:
        report[name] = reject(name, raw, f"only {n_gws} genome-wide significant SNPs; "
                                         f"a MAGMA gene set from this would be noise",
                              n_snp=len(d), n_gws=n_gws, lambda_gc=lam)
        continue

    out = pd.DataFrame({"SNP": d.SNP, "A1": d.A1, "A2": d.A2, "Z": z,
                        "N": int(n_eff)}).drop_duplicates(subset="SNP")
    out.to_csv(target, sep="\t", index=False, compression="gzip",
               float_format="%.6g")
    print(f"    -> {os.path.basename(target)}  {len(out):,} SNPs  ACCEPTED")
    report[name] = dict(status="accepted", n_snp=len(out), n_gws=n_gws,
                        n_eff=n_eff, lambda_gc=lam, accession=acc)
    os.remove(raw)
    del d, out

print(f"\n{'=' * 88}\nSUMMARY")
for k, v in report.items():
    if v["status"] == "accepted":
        print(f"  ACCEPTED  {k:<20}{v['n_snp']:>12,} SNPs  {v['n_gws']:>6,} gw-sig  "
              f"N_eff {v['n_eff']:,.0f}  lambda {v['lambda_gc']:.3f}")
    else:
        print(f"  rejected  {k:<20}{v['reason']}")

p = f"{ROOT}/results/disease_traits.json"
old = {}
if os.path.exists(p):
    try:
        old = json.load(open(p))
    except Exception:
        old = {}
old.update(report)
with open(p, "w") as f:
    json.dump(old, f, indent=2, ensure_ascii=False)
print(f"\nwrote {p}")
