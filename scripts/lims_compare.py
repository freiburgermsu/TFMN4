"""
lims_compare - Cross-environment comparison of the LIMS re-analysis + the auto-generated
report. Reads every environment's outputs (outputs_lims/barcode/<env>/,
outputs_lims/copy_number/<env>/) and refits nothing, except one targeted check: the
TFMN4 amplicon counts re-scored on the WGS transfers {3,4,6,13} with the s03 estimator,
so WGS and amplicon are compared on the SAME estimand.

Tables (outputs_lims/comparison/):
  environment_summary_barcode.csv        one row per barcode environment (design, depth, OD,
                                         selection, alleles, global fit, segmented model)
  environment_summary_copy_number.csv    one row per copy-number environment
  allele_effects_across_environments.csv verA/verB marginal effects, long (env x allele)
  allele_concordance.csv                 pairwise Spearman/Pearson of allele effects between envs
  variant_effects_across_environments.csv bridged per-generation variant effects, long
  wgs_vs_amplicon_TFMN4_frequency.csv    same culture/transfer/variant frequency, both assays
  wgs_vs_amplicon_TFMN4_slopes.csv       per (culture, variant) s: WGS, amplicon@4T, amplicon@8T
  validation_wgs_db_vs_repo.csv          DB-ingested WGS run vs the repo's committed outputs/
Figures (outputs_lims/comparison/figures/): environment_overview, allele_effects_heatmap,
  allele_concordance, top_variants_by_environment, wgs_vs_amplicon_TFMN4,
  copy_number_by_environment.
Report: outputs_lims/REPORT.md
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import config as C                      # default dataset: only its pure helpers are used
from s14c_segmented_figures import (styled, SURFACE, INK, INK2, MUTED, BASELINE, GRID,
                                    BLUE, BLUE_LIGHT, BLUE_DARK, ORANGE)

ROOT = Path(__file__).resolve().parent.parent
LIMS = ROOT / "lims_data"
OUT = ROOT / "outputs_lims"
CMP = OUT / "comparison"
FIG = CMP / "figures"
for _d in (CMP, FIG):
    _d.mkdir(parents=True, exist_ok=True)

MAN = json.loads((LIMS / "manifest.json").read_text())
BC = [d for d in MAN["barcode_datasets"]]
CNS = [d for d in MAN["copy_number_datasets"]]
SENS = [d for d in BC if str(d.get("role", "")).startswith("sensitivity")]
NEW_BC = [d for d in BC if d["assay"] != "wgs" and d not in SENS]   # the environments analysed anew
WGS = next((d for d in BC if d["assay"] == "wgs"), None)

# data-viz categorical slots (fixed order) and the diverging pair (blue <-> red, gray midpoint)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
NONTEST = "#d7d6cf"
CARBON_ABBR = {"methoxybenzoate": "4MXB", "guaiacol": "guaiacol", "isovanillate": "isovanillate",
               "veratrate": "veratrate", "vanillate": "vanillate", "pyruvate": "pyruvate"}


def label(ds):
    """Short environment label, e.g. 'TFMN7 · guaiacol · ACN3882'."""
    exp = "TFMN7" if str(ds["experiment"]).startswith("TFMN7") else ds["experiment"]
    lab = f"{exp} · {CARBON_ABBR.get(ds['carbon'], ds['carbon'])} · {ds['strain']}"
    if ds.get("excluded"):
        return lab + " · excl " + ", ".join(ds["excluded"])
    return lab + (" · WGS" if ds["assay"] == "wgs" else "")


def env_order(dsets):
    pri = {"methoxybenzoate": 0, "guaiacol": 1, "isovanillate": 2, "veratrate": 3}
    return sorted(dsets, key=lambda d: (d["experiment"] != "TFMN4", pri.get(d["carbon"], 9), d["strain"]))


NEW_BC = env_order(NEW_BC)
ENV_COLOR = {d["name"]: SLOTS[i] for i, d in enumerate(NEW_BC)}


def rd(p, **kw):
    return pd.read_csv(p, **kw) if Path(p).exists() else None


def bc_dir(ds):
    return ROOT / ds["out"]


def log_value(ds, pattern):
    p = bc_dir(ds) / "run.log"
    if not p.exists():
        return np.nan
    m = re.search(pattern, p.read_text())
    return float(m.group(1)) if m else np.nan


def metric(df, key, col="value"):
    if df is None:
        return np.nan
    r = df.loc[df.iloc[:, 0] == key, col]
    return r.iat[0] if len(r) else np.nan


# =========================================================================== tables
def barcode_summary():
    rows = []
    for ds in BC:
        o = bc_dir(ds)
        d4 = rd(o / "intermediate/barcode_4transfer_long.csv")
        res = rd(o / "selection_coefficients_long.csv")
        if d4 is None or res is None:
            rows.append(dict(environment=label(ds), dataset=ds["name"], status="MISSING OUTPUTS")); continue
        ss = rd(o / "sample_summary.csv"); al = rd(o / "allele_growth_advantage.csv")
        gauge = rd(o / "sample_gauge.csv"); glob = rd(o / "global_variant_growth_rates.csv")
        msel = rd(o / "segmented_model_selection.csv"); mcs = rd(o / "model_comparison_summary.csv")
        bulk = rd(o / "intermediate/od_bulk_rate.csv")
        axis = rd(o / "intermediate/od_time_axis.csv").set_index("transfer")
        T = ds["transfers"]; t0, tN = T[0], T[-1]
        rich = d4[d4.Count > 0].groupby(["Sample", "Transfer"])["Candidate"].nunique().unstack()
        est = res[res.estimable]
        rel = bulk[bulk.reliable & bulk.transfer.isin(ds["reliable_od"])]
        win = ss["winner_verA"].value_counts()
        rob = al[al.robust_selection]
        robA = rob[rob.allele_kind == "verA"].sort_values("marginal_s_per_generation", ascending=False)
        robB = rob[rob.allele_kind == "verB"].sort_values("marginal_s_per_generation", ascending=False)
        gpc = (axis.loc[tN, "cumgen"] - axis.loc[t0, "cumgen"]) / (tN - t0)
        n_comp = int(gauge["bridge_component"].nunique()) if gauge is not None and "bridge_component" in gauge else 1
        rows.append(dict(
            environment=label(ds), dataset=ds["name"], role=ds["role"], experiment=ds["experiment"],
            condition=ds["condition"], carbon=ds["carbon"], strain=ds["strain"], assay=ds["assay"],
            n_cultures=ds["n_cultures"], transfers=",".join(map(str, T)), n_transfers=len(T),
            median_depth=ds["median_depth"], n_variants=int(d4["Candidate"].nunique()),
            richness_first_median=float(rich[t0].median()), richness_last_median=float(rich[tN].median()),
            hours_per_transfer=ds["hours_per_transfer_measured"], generations_per_cycle=round(float(gpc), 3),
            mu_bulk_median_per_h=round(float(rel.mu_bulk_per_h.median()), 4) if len(rel) else np.nan,
            mu_bulk_n_reliable_cells=int(len(rel)), tau_exp_median_h=round(float(rel.tau_exp_h.median()), 2) if len(rel) else np.nan,
            n_pairs=len(res), n_estimable=len(est),
            s_per_gen_median=round(float(est.s_per_generation.median()), 4),
            s_per_gen_p05=round(float(est.s_per_generation.quantile(0.05)), 4),
            s_per_gen_p95=round(float(est.s_per_generation.quantile(0.95)), 4),
            frac_quality_high_good=round(float(est.quality.isin(["high", "good"]).mean()), 3),
            frac_monotonic=round(float(est.monotonic.mean()), 3),
            top_winner_verA=win.index[0] if len(win) else None, top_winner_verA_wins=int(win.iloc[0]) if len(win) else 0,
            n_robust_verA=len(robA), n_robust_verB=len(robB),
            top_robust_verA=(f"{robA.allele.iat[0]} ({robA.marginal_s_per_generation.iat[0]:+.3f}/gen)" if len(robA) else ""),
            top_robust_verB=(f"{robB.allele.iat[0]} ({robB.marginal_s_per_generation.iat[0]:+.3f}/gen)" if len(robB) else ""),
            bridge_components=n_comp,
            global_fit_freq_R2=log_value(ds, r"freq pseudo-R2=([\d.]+)"),
            global_fit_OD_rmse_per_h=log_value(ds, r"RMSE=([\d.]+)/h"),
            global_rate_min=round(float(glob.r_global_per_h.min()), 3), global_rate_max=round(float(glob.r_global_per_h.max()), 3),
            n_breakpoint_testable=int(metric(msel, "n_breakpoint_testable")),
            n_two_phase_adopted=int(metric(msel, "n_two_phase_adopted_increasing")),
            n_held_decrease=int(metric(msel, "n_held_constant_decrease_not_permitted")),
            n_weak_constant=int(metric(msel, "n_weak_constant")),
            loocv_ratio_elbow_learned=metric(mcs, "LOOCV_RMSE_ratio_adopted_elbow_learned", "four_variable"),
            loocv_4var_better_elbow_learned=metric(mcs, "adopted_4var_predicts_better_elbow_learned", "four_variable"),
            loocv_ratio_elbow_supplied=metric(mcs, "LOOCV_RMSE_ratio_adopted_elbow_supplied", "four_variable"),
            status="ok",
        ))
    df = pd.DataFrame(rows)
    df.to_csv(CMP / "environment_summary_barcode.csv", index=False)
    return df


def cn_summary():
    rows = []
    for ds in CNS:
        o = ROOT / ds["out"]; tag = ds["tag"]
        r = rd(o / f"{tag}_growth_rates.csv"); seg = rd(o / f"{tag}_segmented.csv")
        mcs = rd(o / "model_comparison_summary.csv")
        if r is None:
            rows.append(dict(environment=ds["name"], status="MISSING OUTPUTS")); continue
        relb = r[r.tier == "reliable"]
        amp = r[r.call == "amplifying"].sort_values("rate_per_transfer", ascending=False)
        los = r[r.call == "losing"]
        tst = seg[seg.breakpoint_testable] if seg is not None else pd.DataFrame()
        rows.append(dict(
            environment=f"{tag} · {ds['carbon']} · {ds['platform']}", dataset=ds["name"], region=ds["region"],
            carbon=ds["carbon"], conditions="; ".join(ds["conditions"]), experiments=",".join(ds["experiments"]),
            n_lineages=ds["n_lineages"], n_fitted=len(r), n_ci_reliable=len(relb),
            n_amplifying_fdr=len(amp), n_losing_fdr=len(los),
            median_rate_reliable=round(float(relb.rate_per_transfer.median()), 4) if len(relb) else np.nan,
            top_amplifier=(f"{amp.lineage.iat[0]} r={amp.rate_per_transfer.iat[0]:+.3f}±{amp.se.iat[0]:.3f} "
                           f"(q={amp.q_value.iat[0]:.3f})" if len(amp) else ""),
            n_breakpoint_testable=int(len(tst)),
            n_two_phase_adopted=int((tst.model_selected == "two_phase").sum()) if len(tst) else 0,
            n_decelerating=int((tst.unconstrained_direction == "decelerating").sum()) if len(tst) else 0,
            loto_form_vs_line=metric(mcs, "median_LOTO_RMSE_FORM_vs_line", "interpretation"),
            transfers=",".join(map(str, ds["transfers"])), status="ok"))
    df = pd.DataFrame(rows)
    df.to_csv(CMP / "environment_summary_copy_number.csv", index=False)
    return df


def allele_tables():
    frames = []
    for ds in NEW_BC + ([WGS] if WGS else []):
        a = rd(bc_dir(ds) / "allele_growth_advantage.csv")
        if a is None:
            continue
        a = a[["allele_kind", "allele", "marginal_s_per_cycle", "marginal_s_per_generation", "se",
               "n_pairs", "n_samples", "drift_null_p", "exceeds_drift_null", "robust_selection"]].copy()
        a.insert(0, "environment", label(ds)); a.insert(1, "dataset", ds["name"])
        frames.append(a)
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(CMP / "allele_effects_across_environments.csv", index=False)
    # pairwise concordance between the NEW environments (alleles seen in >=3 pairs in both)
    envs = [label(d) for d in NEW_BC]
    rows = []
    for i, ea in enumerate(envs):
        for eb in envs[i + 1:]:
            A = long[(long.environment == ea) & (long.n_pairs >= 3)]
            B = long[(long.environment == eb) & (long.n_pairs >= 3)]
            j = A.merge(B, on=["allele_kind", "allele"], suffixes=("_a", "_b"))
            if len(j) >= 5:
                rho = stats.spearmanr(j.marginal_s_per_generation_a, j.marginal_s_per_generation_b)[0]
                r = stats.pearsonr(j.marginal_s_per_generation_a, j.marginal_s_per_generation_b)[0]
                rb = j[j.robust_selection_a & j.robust_selection_b]
                agree = (np.sign(rb.marginal_s_per_generation_a) == np.sign(rb.marginal_s_per_generation_b)).sum()
            else:
                rho = r = np.nan; rb = j.iloc[:0]; agree = 0
            rows.append(dict(env_a=ea, env_b=eb, n_shared_alleles=len(j), spearman=rho, pearson=r,
                             n_robust_in_both=len(rb), robust_sign_agreement=int(agree)))
    conc = pd.DataFrame(rows)
    conc.to_csv(CMP / "allele_concordance.csv", index=False)
    return long, conc


def variant_tables():
    frames = []
    for ds in NEW_BC:
        v = rd(bc_dir(ds) / "variant_bridged_relative.csv"); g = rd(bc_dir(ds) / "global_variant_growth_rates.csv")
        if v is None or g is None:
            continue
        lap = "ci_laplace_lo" in g and not bool(g["boot_loso_refits_converged"].all())
        g = g.assign(ci_lo=g["ci_laplace_lo"] if lap else g["ci_boot_lo"],
                     ci_hi=g["ci_laplace_hi"] if lap else g["ci_boot_hi"],
                     ci_kind="Laplace" if lap else "bootstrap")
        m = v.merge(g[["Candidate", "r_global_per_h", "ci_lo", "ci_hi", "ci_kind", "rel_advantage_per_h",
                       "identifiability", "has_slope_info"]], on="Candidate", how="left")
        m.insert(0, "environment", label(ds)); m.insert(1, "dataset", ds["name"])
        frames.append(m)
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(CMP / "variant_effects_across_environments.csv", index=False)
    return long


def _s03_slopes(d4, transfers, se=False):
    """The s03 estimator (CLR weighted-LS slope per transfer) on a count table."""
    out = []
    x = np.array(transfers, float)
    for s, sub in d4.groupby("Sample"):
        counts = (sub.pivot_table(index="Candidate", columns="Transfer", values="Count",
                                  aggfunc="sum", fill_value=0).reindex(columns=transfers, fill_value=0))
        counts = counts[counts.sum(axis=1) > 0]
        clr = C.clr_matrix(counts)
        for cand in counts.index:
            cv = counts.loc[cand].to_numpy(float)
            if (cv > 0).sum() < C.MIN_OBS_TIMEPOINTS:
                continue
            f = C.weighted_linfit(x, clr.loc[cand].to_numpy(float), cv + C.PSEUDOCOUNT)
            out.append(dict(Sample=s, Candidate=cand, s=f["slope"], n_obs=int((cv > 0).sum()),
                            **({"w": 1 / max(f["slope_se"], 1e-3) ** 2} if se else {})))
    return pd.DataFrame(out)


def wgs_vs_amplicon():
    """Same TFMN4 cultures, two assays: frequencies at the shared transfers, and slopes on
    the same 4 transfers (identical estimand) + the 8-transfer amplicon slope."""
    amp = next((d for d in NEW_BC if d["experiment"] == "TFMN4"), None)
    if WGS is None or amp is None:
        return None, None, {}
    dw = rd(bc_dir(WGS) / "intermediate/barcode_4transfer_long.csv")
    da = rd(bc_dir(amp) / "intermediate/barcode_4transfer_long.csv")
    shared_T = [t for t in WGS["transfers"] if t in amp["transfers"]]
    shared_S = sorted(set(dw.Sample) & set(da.Sample))
    k = ["Sample", "Transfer", "Candidate"]
    w = dw[dw.Sample.isin(shared_S) & dw.Transfer.isin(shared_T)][k + ["Count", "freq"]]
    a = da[da.Sample.isin(shared_S) & da.Transfer.isin(shared_T)][k + ["Count", "freq"]]
    fq = w.merge(a, on=k, how="outer", suffixes=("_wgs", "_amp")).fillna(0.0)
    fq.to_csv(CMP / "wgs_vs_amplicon_TFMN4_frequency.csv", index=False)
    # cross-assay discordance: variant clearly present by amplicon but absent from WGS (or vice versa)
    # Conservative: the observing assay's frequency is taken at its 95% LOWER Poisson bound
    # (so a single WGS read, ~0.5% +/- huge, cannot by itself predict amplicon reads).
    dw_ = fq.groupby(["Sample", "Transfer"])["Count_wgs"].transform("sum")
    da_ = fq.groupby(["Sample", "Transfer"])["Count_amp"].transform("sum")
    lo = lambda k: np.where(k > 0, stats.chi2.ppf(0.025, 2 * np.maximum(k, 1)) / 2, 0.0)
    f_amp_lo = lo(fq.Count_amp.to_numpy()) / np.maximum(da_.to_numpy(), 1)
    f_wgs_lo = lo(fq.Count_wgs.to_numpy()) / np.maximum(dw_.to_numpy(), 1)
    fq_ = fq.assign(expected_wgs=f_amp_lo * dw_, expected_amp=f_wgs_lo * da_)
    fq_["p_absent_in_wgs"] = np.exp(-fq_.expected_wgs)          # Poisson P(0 WGS reads)
    fq_["p_absent_in_amp"] = np.exp(-fq_.expected_amp)
    alpha = 0.05 / max(2 * len(fq_), 1)                         # Bonferroni over every tested cell
    disc = fq_[((fq_.Count_wgs == 0) & (fq_.p_absent_in_wgs < alpha)) |
               ((fq_.Count_amp == 0) & (fq_.p_absent_in_amp < alpha))]
    disc.sort_values(["Candidate", "Sample", "Transfer"]).to_csv(CMP / "cross_assay_discordance_TFMN4.csv", index=False)
    both = fq[(fq.Count_wgs > 0) & (fq.Count_amp > 0)]
    st = dict(n_cells=len(set(zip(fq.Sample, fq.Transfer))), n_rows=len(fq),
              freq_pearson_log=float(stats.pearsonr(np.log10(both.freq_wgs), np.log10(both.freq_amp))[0]),
              freq_spearman=float(stats.spearmanr(both.freq_wgs, both.freq_amp)[0]),
              frac_wgs_variants_seen_in_amplicon=float(((fq.Count_wgs > 0) & (fq.Count_amp > 0)).sum() / max((fq.Count_wgs > 0).sum(), 1)),
              frac_amp_reads_in_wgs_variants=float(fq.loc[fq.Count_wgs > 0, "Count_amp"].sum() / max(fq.Count_amp.sum(), 1)))
    topw = w.sort_values("Count").groupby(["Sample", "Transfer"]).tail(1).set_index(["Sample", "Transfer"])["Candidate"]
    topa = a.sort_values("Count").groupby(["Sample", "Transfer"]).tail(1).set_index(["Sample", "Transfer"])["Candidate"]
    jt = pd.concat([topw.rename("w"), topa.rename("a")], axis=1).dropna()
    st["dominant_variant_agreement"] = float((jt.w == jt.a).mean()); st["n_cells_compared"] = int(len(jt))
    # slopes: WGS s03 (4T) | amplicon re-scored on the same 4T | amplicon s03 (8T)
    sw = rd(bc_dir(WGS) / "selection_coefficients_long.csv")
    sw = sw[sw.estimable][["Sample", "Candidate", "s_per_cycle", "n_obs_timepoints"]].rename(
        columns={"s_per_cycle": "s_wgs_4T", "n_obs_timepoints": "n_obs_wgs"})
    sa4 = _s03_slopes(a.assign(Count=a.Count.astype(float)), shared_T).rename(columns={"s": "s_amp_4T", "n_obs": "n_obs_amp4"})
    sa8 = rd(bc_dir(amp) / "selection_coefficients_long.csv")
    sa8 = sa8[sa8.estimable][["Sample", "Candidate", "s_per_cycle"]].rename(columns={"s_per_cycle": "s_amp_8T"})
    sl = sw.merge(sa4, on=["Sample", "Candidate"], how="outer").merge(sa8, on=["Sample", "Candidate"], how="left")
    sl.to_csv(CMP / "wgs_vs_amplicon_TFMN4_slopes.csv", index=False)
    j = sl.dropna(subset=["s_wgs_4T", "s_amp_4T"])
    st.update(n_slope_pairs=len(j),
              slope_pearson_same_4T=float(stats.pearsonr(j.s_wgs_4T, j.s_amp_4T)[0]),
              slope_spearman_same_4T=float(stats.spearmanr(j.s_wgs_4T, j.s_amp_4T)[0]))
    # Each assay's slopes are relative to the geometric mean of the variants IT detects (the
    # amplicon sees ~5x more), so absolute values are not comparable across assays; alleles are
    # compared gauge-free, as a difference from the median allele and as a percentile.
    rows = []
    for nm, d, T in (("wgs_4T", dw, shared_T), ("amp_4T", da, shared_T), ("amp_allT", da, amp["transfers"])):
        sl_ = _s03_slopes(d[d.Sample.isin(shared_S) & d.Transfer.isin(T)], T, se=True)
        sl_ = sl_.merge(d.drop_duplicates("Candidate")[["Candidate", "verA"]], on="Candidate")
        mm = sl_.groupby("verA").apply(lambda z: np.sum(z.w * z.s) / np.sum(z.w), include_groups=False)
        rows.append(pd.DataFrame({f"{nm}_minus_median": (mm - mm.median()).round(4), f"{nm}_percentile": mm.rank(pct=True).round(3)}))
    alle = pd.concat(rows, axis=1).reset_index().rename(columns={"index": "verA"})
    alle.to_csv(CMP / "wgs_vs_amplicon_TFMN4_alleles.csv", index=False)
    st["allele_table"] = alle
    j2 = sl.dropna(subset=["s_amp_4T", "s_amp_8T"])
    st.update(n_slope_pairs_4T_vs_8T=len(j2),
              slope_spearman_amp4T_vs_amp8T=float(stats.spearmanr(j2.s_amp_4T, j2.s_amp_8T)[0]),
              n_estimable_wgs=int(sw.shape[0]), n_estimable_amp8=int(sa8.shape[0]))
    return fq, sl, st


def validation():
    """DB-ingested WGS run vs the repo's committed WGS outputs (same data, same code)."""
    if WGS is None:
        return None
    A, B = ROOT / "outputs", bc_dir(WGS)
    rows = []
    for pa in sorted(A.rglob("*.csv")):
        if "~lock" in pa.name or pa.name.startswith("._"):
            continue
        f = pa.relative_to(A); pb = B / f
        if not pb.exists():
            rows.append(dict(file=str(f), status="missing in DB run")); continue
        a = pd.read_csv(pa); b = pd.read_csv(pb)
        common = [c for c in a.columns if c in b.columns]       # LIMS runs may add columns
        if len(a) != len(b):
            rows.append(dict(file=str(f), status=f"rows {len(a)} vs {len(b)}")); continue
        if "Candidate" in common and a.Candidate.is_unique and set(a.Candidate) == set(b.Candidate):
            b = b.set_index("Candidate").loc[a.Candidate].reset_index()   # tables sorted by value
        a, b = a[common], b[common]
        num = a.select_dtypes("number").columns; oth = [c for c in common if c not in num]
        da, db = a[num].to_numpy(float), b[num].to_numpy(float)
        dd = np.where(np.isnan(da) & np.isnan(db), 0.0, np.abs(da - db)); dd = np.where(np.isnan(dd), np.inf, dd)
        mx = float(dd.max()) if dd.size else 0.0
        ntxt = int((a[oth].fillna("").astype(str).values != b[oth].fillna("").astype(str).values).sum()) if oth else 0
        col = num[int(np.unravel_index(np.argmax(dd), dd.shape)[1])] if dd.size and mx > 0 else ""
        rows.append(dict(file=str(f), status="identical" if mx == 0 and ntxt == 0 else "differs",
                         max_abs_numeric_diff=mx, column_of_max=col, text_cells_differing=ntxt))
    df = pd.DataFrame(rows)
    df.to_csv(CMP / "validation_wgs_db_vs_repo.csv", index=False)
    return df


def s12_vs_repo():
    """global_variant_growth_rates.csv of the DB WGS run vs the committed one, per column group:
    the s12 point estimates vs the s12b refit columns (converged here, iteration-capped there)."""
    if WGS is None or not (ROOT / "outputs" / "global_variant_growth_rates.csv").exists():
        return None
    a = pd.read_csv(ROOT / "outputs" / "global_variant_growth_rates.csv")
    b = rd(bc_dir(WGS) / "global_variant_growth_rates.csv")
    j = a.merge(b, on="Candidate", suffixes=("_repo", "_db"))
    mx = lambda cols: float(max((j[f"{c}_repo"] - j[f"{c}_db"]).abs().max() for c in cols))
    return dict(s12=mx(["r_global_per_h", "se_laplace"]),
                s12b=mx(["ci_boot_lo", "ci_boot_hi", "se_boot", "loso_heterogeneity_sd", "sens_to_OD_weight"]),
                ident=int((j.identifiability_repo != j.identifiability_db).sum()), n=len(j),
                converged=bool(b["boot_loso_refits_converged"].all()) if "boot_loso_refits_converged" in b else False)


def sensitivity():
    """Main run vs its sensitivity copy with suspect candidates removed."""
    out = []
    for sd in SENS:
        base = next(d for d in BC if d["name"] == sd["name"].rsplit("_excl_", 1)[0])
        ob, os_ = bc_dir(base), bc_dir(sd)
        rows = []
        for kind in ("verA", "verB"):
            A = rd(ob / f"allele_effects_{kind}.csv"); B = rd(os_ / f"allele_effects_{kind}.csv")
            j = A.merge(B, on="allele", suffixes=("_main", "_excl"))
            rows += [(f"{kind}_marginal_spearman_main_vs_excl",
                      round(float(stats.spearmanr(j.marginal_s_per_cycle_main, j.marginal_s_per_cycle_excl)[0]), 4)),
                     (f"{kind}_robust_main", int(A.robust_selection.sum())),
                     (f"{kind}_robust_excl", int(B.robust_selection.sum())),
                     (f"{kind}_robust_only_in_main", ", ".join(sorted(set(A.allele[A.robust_selection]) - set(B.allele[B.robust_selection])))),
                     (f"{kind}_robust_only_in_excl", ", ".join(sorted(set(B.allele[B.robust_selection]) - set(A.allele[A.robust_selection]))))]
        aA = rd(ob / "allele_effects_verA.csv").set_index("allele"); aB = rd(os_ / "allele_effects_verA.csv").set_index("allele")
        for al in sd.get("excluded_verA", []):
            if al in aA.index:
                rows.append((f"{al}_marginal_s_per_cycle_main", f"{aA.loc[al, 'marginal_s_per_cycle']:+.3f} "
                             f"(robust {bool(aA.loc[al, 'robust_selection'])}); excluded from the sensitivity run"))
        for al in ("A81", "A90", "A78"):
            if al in aA.index and al in aB.index:
                rows.append((f"{al}_marginal_s_per_cycle_main_vs_excl",
                             f"{aA.loc[al, 'marginal_s_per_cycle']:+.3f} vs {aB.loc[al, 'marginal_s_per_cycle']:+.3f} "
                             f"(robust {bool(aA.loc[al, 'robust_selection'])} vs {bool(aB.loc[al, 'robust_selection'])})"))
        gA = rd(ob / "global_variant_growth_rates.csv"); gB = rd(os_ / "global_variant_growth_rates.csv")
        jg = gA.merge(gB, on="Candidate", suffixes=("_main", "_excl"))
        rows.append(("global_rel_advantage_spearman_main_vs_excl",
                     round(float(stats.spearmanr(jg.rel_advantage_per_h_main, jg.rel_advantage_per_h_excl)[0]), 4)))
        rows.append(("global_rel_advantage_max_abs_change_per_h",
                     round(float((jg.rel_advantage_per_h_main - jg.rel_advantage_per_h_excl).abs().max()), 4)))
        tA = list(gA[gA.has_slope_info].nlargest(12, "r_global_per_h").Candidate)
        tB = list(gB[gB.has_slope_info].nlargest(12, "r_global_per_h").Candidate)
        rows.append(("top12_fastest_overlap", f"{len(set(tA) & set(tB))}/12"))
        for key in ("n_breakpoint_testable", "n_two_phase_adopted_increasing"):
            rows.append((key + "_main_vs_excl", f"{int(metric(rd(ob / 'segmented_model_selection.csv'), key))} vs "
                                                 f"{int(metric(rd(os_ / 'segmented_model_selection.csv'), key))}"))
        rows.append(("global_fit_freq_R2_main_vs_excl", f"{log_value(base, r'freq pseudo-R2=([\d.]+)'):.3f} vs "
                                                        f"{log_value(sd, r'freq pseudo-R2=([\d.]+)'):.3f}"))
        df = pd.DataFrame(rows, columns=["metric", "value"])
        df.to_csv(CMP / f"sensitivity_excl_{sd['name'].split('_excl_')[1]}.csv", index=False)
        out.append((sd, base, df))
    return out


@styled
def fig_sensitivity(sens):
    for sd, base, df in sens:
        cand = sd["excluded_verA"][0] if sd.get("excluded_verA") else sd["excluded"][0]
        da = rd(bc_dir(base) / "intermediate/barcode_4transfer_long.csv")
        dw = rd(bc_dir(WGS) / "intermediate/barcode_4transfer_long.csv") if WGS else None
        hit = (lambda d: d.verA == cand) if sd.get("excluded_verA") else (lambda d: d.Candidate == cand)
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
        a = axs[0]
        T = base["transfers"]
        for smp, g in da.groupby("Sample"):
            f = g[hit(g)].groupby("Transfer")["freq"].sum().reindex(T).fillna(0.0)
            a.plot(T, np.clip(f.values, 1e-5, None), color=BLUE, lw=1.2, alpha=0.6, marker="o", ms=3.5)
        if dw is not None:
            for t in WGS["transfers"]:
                sub = dw[dw.Transfer == t]
                fw = sub[hit(sub)].groupby("Sample")["freq"].sum().reindex(sorted(sub.Sample.unique())).fillna(0)
                a.scatter([t] * len(fw), np.clip(fw.values, 1.5e-5, None), marker="v", s=40, color=ORANGE,
                          zorder=4, label="WGS, same cultures (0 reads -> plotted at floor)" if t == WGS["transfers"][0] else None)
        a.set_yscale("log"); a.set_ylim(1e-5, 1)
        a.set_xlabel("transfer"); a.set_ylabel(f"{cand} total frequency (0 -> floor)")
        a.set_title(f"(a) {cand}: absent through T8 in all {da.Sample.nunique()} cultures, 6–30% at T10\n"
                    f"(blue = amplicon per culture; orange = WGS of the same cultures)", fontsize=10)
        a.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        a = axs[1]
        for kind, col in (("verA", SLOTS[0]), ("verB", SLOTS[1])):
            A = rd(bc_dir(base) / f"allele_effects_{kind}.csv"); B = rd(bc_dir(sd) / f"allele_effects_{kind}.csv")
            j = A.merge(B, on="allele", suffixes=("_m", "_x"))
            a.scatter(j.marginal_s_per_cycle_m, j.marginal_s_per_cycle_x, s=24, color=col, edgecolors=SURFACE,
                      linewidths=1, label=kind, zorder=3)
            for al in ("A81", "A90", "B63"):
                hh = j[j.allele == al]
                for r in hh.itertuples():
                    a.annotate(al, (r.marginal_s_per_cycle_m, r.marginal_s_per_cycle_x), xytext=(6, -10),
                               textcoords="offset points", fontsize=8, color=INK)
        lim = a.get_xlim(); a.plot(lim, lim, color=BASELINE, lw=1.1, ls="--", zorder=1)
        a.set_xlabel("allele marginal s per cycle — main run (all data)")
        a.set_ylabel(f"— sensitivity run (without {cand})")
        a.set_title(f"(b) Allele effects with vs without {cand}", fontsize=10)
        a.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
        fig.suptitle(f"Suspected contamination check — {label(base)} amplicon", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(FIG / f"sensitivity_excl_{sd['name'].split('_excl_')[1]}.png", dpi=140, bbox_inches="tight"); plt.close(fig)


# =========================================================================== figures
@styled
def fig_overview(summ):
    envs = NEW_BC
    fig, ax = plt.subplots(2, 2, figsize=(14, 10.5))
    # (a) richness: median distinct variants per transfer
    a = ax[0, 0]
    for ds in envs:
        d4 = rd(bc_dir(ds) / "intermediate/barcode_4transfer_long.csv")
        rich = d4[d4.Count > 0].groupby(["Sample", "Transfer"])["Candidate"].nunique().unstack()
        med = rich.median(axis=0)
        a.plot(med.index, med.values, color=ENV_COLOR[ds["name"]], lw=2, marker="o", ms=5,
               mec=SURFACE, mew=1.5, label=label(ds), solid_capstyle="round")
    a.set_yscale("log"); a.set_xlabel("transfer"); a.set_ylabel("median # distinct variants per culture")
    lo, hi = a.get_ylim(); a.set_ylim(lo, hi * 12)          # headroom for the legend
    a.set_title("(a) Richness collapse (selective sweeps)")
    a.legend(fontsize=7.5, framealpha=0.9, loc="upper center", ncol=2)
    # (b) community bulk rate per culture
    a = ax[0, 1]
    ylab = []
    for i, ds in enumerate(envs):
        b = rd(bc_dir(ds) / "intermediate/od_bulk_rate.csv")
        rel = b[b.reliable & b.transfer.isin(ds["reliable_od"])]
        per = rel.groupby("Sample")["mu_bulk_per_h"].mean()
        jit = (np.random.default_rng(i).random(len(per)) - 0.5) * 0.3
        a.scatter(per.values, i + jit, s=30, color=BLUE, alpha=0.75, edgecolors=SURFACE, linewidths=1, zorder=3)
        if len(per):
            a.plot([per.median()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2, zorder=4)
        ylab.append(f"{label(ds)}  (n={len(per)})")
    a.set_yticks(range(len(envs))); a.set_yticklabels(ylab, fontsize=8); a.invert_yaxis()
    a.set_xlabel("community exponential-phase rate  μ_bulk  (per h; culture mean of reliable cells)")
    a.set_title("(b) Community μ_bulk — the one absolute rate")
    a.grid(axis="y", visible=False)
    # (c) per-trajectory relative fitness (per generation)
    a = ax[1, 0]
    data = []
    for ds in envs:
        r = rd(bc_dir(ds) / "selection_coefficients_long.csv")
        data.append(r.loc[r.estimable, "s_per_generation"].dropna().to_numpy())
    bp = a.boxplot(data, vert=False, widths=0.5, showfliers=False, patch_artist=True,
                   medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED),
                   capprops=dict(color=MUTED), boxprops=dict(edgecolor=BLUE_DARK, lw=1))
    for p in bp["boxes"]:
        p.set_facecolor(BLUE_LIGHT)
    a.axvline(0, color=BASELINE, lw=1.2)
    a.set_yticks(range(1, len(envs) + 1)); a.set_yticklabels([f"{label(d)}  (n={len(x)})" for d, x in zip(envs, data)], fontsize=8)
    a.invert_yaxis(); a.set_xlabel("selection coefficient  s  (per generation; estimable trajectories)")
    a.set_title("(c) Relative fitness spread (box = IQR, whiskers 1.5×IQR)")
    a.grid(axis="y", visible=False)
    # (d) 4-variable model classification
    a = ax[1, 1]
    for i, ds in enumerate(envs):
        r = summ[summ.dataset == ds["name"]].iloc[0]
        nest = r.n_estimable
        segs = [(r.n_two_phase_adopted, BLUE, "increasing two-phase adopted"),
                (r.n_held_decrease, ORANGE, "held constant — decrease not permitted"),
                (r.n_weak_constant, MUTED, "weak — kept constant"),
                (nest - r.n_breakpoint_testable, NONTEST, "not breakpoint-testable")]
        left = 0
        for val, col, _ in segs:
            frac = val / nest if nest else 0
            a.barh(i, frac, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.55)
            left += frac
        a.text(1.01, i, f"{int(r.n_two_phase_adopted)} adopted / {int(r.n_breakpoint_testable)} testable",
               va="center", fontsize=7.5, color=INK2, transform=a.get_yaxis_transform())
    a.set_yticks(range(len(envs))); a.set_yticklabels([label(d) for d in envs], fontsize=8); a.invert_yaxis()
    a.set_xlim(0, 1); a.set_xlabel("fraction of estimable trajectories")
    a.set_title("(d) 4-parameter (breakpoint) model decisions")
    a.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in segs], [n for _, _, n in segs],
             fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, framealpha=0.9)
    a.grid(axis="y", visible=False)
    fig.suptitle("LIMS re-analysis — every environment simulated separately (barcode competition)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.94, 0.97])
    fig.savefig(FIG / "environment_overview.png", dpi=140, bbox_inches="tight"); plt.close(fig)


@styled
def fig_allele_heatmap(long):
    envs = [label(d) for d in NEW_BC]
    lg = long[long.environment.isin(envs)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 10))
    for j, kind in enumerate(("verA", "verB")):
        k = lg[lg.allele_kind == kind]
        rob = set(k.loc[k.robust_selection, "allele"])
        multi = k.groupby("allele")["environment"].nunique()
        pick = [x for x in rob if multi.get(x, 0) >= 2]
        if len(pick) > 30:
            mx = k[k.allele.isin(pick)].groupby("allele")["marginal_s_per_generation"].apply(lambda s: s.abs().max())
            pick = list(mx.sort_values(ascending=False).index[:30])
        M = k[k.allele.isin(pick)].pivot_table(index="allele", columns="environment",
                                                 values="marginal_s_per_generation").reindex(columns=envs)
        R = k[k.allele.isin(pick)].pivot_table(index="allele", columns="environment",
                                                 values="robust_selection", aggfunc="first").reindex(columns=envs)
        M = M.loc[M.mean(axis=1).sort_values(ascending=False).index]
        R = R.reindex(M.index)
        a = axs[j]
        if M.empty:
            a.set_title(f"({'ab'[j]}) {kind}: no robust allele seen in >=2 environments"); a.axis("off"); continue
        v = np.nanmax(np.abs(M.to_numpy(float)))
        cm = DIV.copy(); cm.set_bad(SURFACE)
        im = a.imshow(np.ma.masked_invalid(M.to_numpy(float)), cmap=cm, vmin=-v, vmax=v, aspect="auto")
        for (yi, xi), val in np.ndenumerate(R.to_numpy()):
            if val is True or val == 1.0:
                a.scatter(xi, yi, s=26, facecolors="none", edgecolors=INK, linewidths=1.1)
        a.set_xticks(range(len(envs))); a.set_xticklabels(envs, rotation=35, ha="right", fontsize=8)
        a.set_yticks(range(len(M))); a.set_yticklabels(M.index, fontsize=7.5)
        a.set_title(f"({'ab'[j]}) {kind} marginal selection (per generation)\n"
                    f"robust in ≥1 environment, seen in ≥2  ·  ○ = drift-confirmed robust there", fontsize=10)
        a.grid(False)
        cb = fig.colorbar(im, ax=a, fraction=0.04, pad=0.02)
        cb.set_label("s per generation  (blue = favoured, red = disfavoured; blank = absent)", fontsize=8)
    fig.suptitle("Enzyme-allele effects across carbon sources (each environment fitted separately)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "allele_effects_heatmap.png", dpi=140, bbox_inches="tight"); plt.close(fig)


@styled
def fig_allele_concordance(long, conc):
    envs = [label(d) for d in NEW_BC]
    mxb = [label(d) for d in NEW_BC if d["carbon"] == "methoxybenzoate"]
    gua = [label(d) for d in NEW_BC if d["carbon"] == "guaiacol"]
    pairs = []
    if len(mxb) >= 2:
        pairs.append((mxb[0], mxb[1], "(a) Same environment (4MXB), two experiments"))
    if len(gua) >= 2:
        pairs.append((gua[0], gua[1], "(b) Same environment (guaiacol), two host strains"))
    fig, axs = plt.subplots(1, len(pairs) + 1, figsize=(5.4 * (len(pairs) + 1), 5.2))
    axs = np.atleast_1d(axs)
    for a, (ea, eb, title) in zip(axs, pairs):
        A = long[(long.environment == ea) & (long.n_pairs >= 3)]
        B = long[(long.environment == eb) & (long.n_pairs >= 3)]
        j = A.merge(B, on=["allele_kind", "allele"], suffixes=("_a", "_b"))
        for kind, col in (("verA", SLOTS[0]), ("verB", SLOTS[1])):
            jj = j[j.allele_kind == kind]
            a.scatter(jj.marginal_s_per_generation_a, jj.marginal_s_per_generation_b, s=34, color=col,
                      edgecolors=SURFACE, linewidths=1.2, label=f"{kind} ({len(jj)})", zorder=3)
        if len(j):
            lim = float(np.nanmax(np.abs(j[["marginal_s_per_generation_a", "marginal_s_per_generation_b"]].to_numpy()))) * 1.1
            a.plot([-lim, lim], [-lim, lim], color=BASELINE, lw=1.1, ls="--", zorder=1)
            a.axhline(0, color=GRID, lw=0.8); a.axvline(0, color=GRID, lw=0.8)
            a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
            rr = conc[((conc.env_a == ea) & (conc.env_b == eb)) | ((conc.env_a == eb) & (conc.env_b == ea))]
            if len(rr) and np.isfinite(rr.spearman.iat[0]):
                a.text(0.03, 0.97, f"Spearman ρ = {rr.spearman.iat[0]:+.2f}\nn = {len(j)} shared alleles",
                       transform=a.transAxes, va="top", fontsize=8.5, color=INK2)
        a.set_xlabel(f"{ea}\nmarginal s per generation"); a.set_ylabel(f"{eb}\nmarginal s per generation")
        a.set_title(title, fontsize=10); a.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    # pairwise correlation matrix
    a = axs[-1]
    Mx = pd.DataFrame(np.eye(len(envs)), index=envs, columns=envs)
    for r in conc.itertuples():
        Mx.loc[r.env_a, r.env_b] = Mx.loc[r.env_b, r.env_a] = r.spearman
    im = a.imshow(Mx.to_numpy(float), cmap=DIV, vmin=-1, vmax=1)
    for (yi, xi), val in np.ndenumerate(Mx.to_numpy(float)):
        if yi != xi and np.isfinite(val):
            a.text(xi, yi, f"{val:+.2f}", ha="center", va="center", fontsize=7.5,
                   color="white" if abs(val) > 0.6 else INK)
    short = [e.replace(" · ACN3788", "").replace("TFMN7 · ", "T7 ").replace("TFMN4 · ", "T4 ") for e in envs]
    a.set_xticks(range(len(envs))); a.set_xticklabels(short, rotation=40, ha="right", fontsize=7.5)
    a.set_yticks(range(len(envs))); a.set_yticklabels(short, fontsize=7.5); a.grid(False)
    a.set_title(f"({'abc'[len(pairs)]}) Allele-effect Spearman ρ between environments", fontsize=10)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    fig.suptitle("Do enzyme-allele effects replicate across experiments, strains and carbon sources?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "allele_concordance.png", dpi=140, bbox_inches="tight"); plt.close(fig)


@styled
def fig_top_variants(vlong):
    envs = NEW_BC
    n = len(envs); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow), squeeze=False)
    for a in axs.flat:
        a.axis("off")
    for a, ds in zip(axs.flat, envs):
        a.axis("on")
        v = vlong[(vlong.dataset == ds["name"]) & (vlong.has_slope_info == True)]
        top = v.sort_values("r_global_per_h", ascending=False).head(12).iloc[::-1]
        y = np.arange(len(top))
        mu = float((top.r_global_per_h - top.rel_advantage_per_h).median())   # s12's community mu_bar
        rel = top.rel_advantage_per_h
        lo = (top.r_global_per_h - top.ci_lo).clip(lower=0); hi = (top.ci_hi - top.r_global_per_h).clip(lower=0)
        a.errorbar(rel, y, xerr=[lo, hi], fmt="o", ms=5, color=ENV_COLOR[ds["name"]],
                   mec=SURFACE, mew=1.2, ecolor=ENV_COLOR[ds["name"]], elinewidth=1.2, capsize=2, zorder=3)
        a.axvline(0, color=BASELINE, lw=1.0, zorder=1)        # = the community mean; CIs may cross it
        a.set_yticks(y); a.set_yticklabels([f"{c}  (n={int(k)})" for c, k in zip(top.Candidate, top.n_samples)], fontsize=7.5)
        a.set_xlabel(f"advantage over community mean  r − μ̄  (per h) ± {top.ci_kind.iloc[0] if len(top) else 'bootstrap'} 95% CI",
                     fontsize=8)
        a.set_title(f"{label(ds)}   (μ̄ = {mu:.2f}/h)", fontsize=10); a.grid(axis="y", visible=False)
    fig.suptitle("Fastest variants per environment — the s12 global static-rate model, fit separately per environment\n"
                 "plotted as the advantage over that environment's community mean (the data-pinned part; the absolute "
                 "level is anchor-dependent)  ·  n = cultures the variant appears in  ·  x-axes differ between panels\n"
                 "wide, skewed intervals = variants with only 1–3 reads at a timepoint (resampling often removes them "
                 "from it, so their fitted rate swings); variants with hundreds of reads have tight intervals", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(FIG / "top_variants_by_environment.png", dpi=140, bbox_inches="tight"); plt.close(fig)


@styled
def fig_wgs_vs_amp(fq, sl, st):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.0))
    a = axs[0]
    both = fq[(fq.Count_wgs > 0) & (fq.Count_amp > 0)]
    only_w = fq[(fq.Count_wgs > 0) & (fq.Count_amp == 0)]
    a.scatter(both.freq_wgs, both.freq_amp, s=14, color=BLUE, alpha=0.55, edgecolors="none", label=f"seen by both ({len(both)})")
    a.scatter(only_w.freq_wgs, np.full(len(only_w), 1.2e-4), s=12, marker="v", color=ORANGE, alpha=0.7,
              label=f"WGS only ({len(only_w)}; plotted at floor)")
    a.plot([1e-4, 1], [1e-4, 1], color=BASELINE, lw=1.1, ls="--")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(2e-3, 1.05); a.set_ylim(1e-4, 1.05)
    a.set_xlabel("WGS frequency (~200 reads)"); a.set_ylabel("amplicon frequency (~4,000 reads)")
    a.set_title(f"(a) Same culture & transfer: frequency\nPearson r(log) = {st['freq_pearson_log']:.2f}; "
                f"dominant variant agrees in {st['dominant_variant_agreement']:.0%} of cells", fontsize=9.5)
    a.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    j = sl.dropna(subset=["s_wgs_4T", "s_amp_4T"])
    a = axs[1]
    lim = float(np.nanmax(np.abs(j[["s_wgs_4T", "s_amp_4T"]].to_numpy()))) * 1.05
    a.plot([-lim, lim], [-lim, lim], color=BASELINE, lw=1.1, ls="--")
    a.scatter(j.s_wgs_4T, j.s_amp_4T, s=16, color=BLUE, alpha=0.6, edgecolors="none")
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel("WGS  s per cycle  (T3, T4, T6, T13)"); a.set_ylabel("amplicon  s per cycle  (same 4 transfers)")
    a.set_title(f"(b) Same transfers, two assays ({len(j)} trajectories)\nSpearman ρ = {st['slope_spearman_same_4T']:.2f} "
                f"(levels differ by gauge: each assay's own variant set)", fontsize=9.5)
    j2 = sl.dropna(subset=["s_amp_4T", "s_amp_8T"])
    a = axs[2]
    lim = float(np.nanmax(np.abs(j2[["s_amp_4T", "s_amp_8T"]].to_numpy()))) * 1.05
    a.plot([-lim, lim], [-lim, lim], color=BASELINE, lw=1.1, ls="--")
    a.scatter(j2.s_amp_4T, j2.s_amp_8T, s=10, color=BLUE, alpha=0.4, edgecolors="none")
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel("amplicon  s  (4 transfers)"); a.set_ylabel("amplicon  s  (all 8 transfers)")
    a.set_title(f"(c) Denser sampling, same assay ({len(j2)} trajectories)\n"
                f"Spearman ρ = {st['slope_spearman_amp4T_vs_amp8T']:.2f}", fontsize=9.5)
    fig.suptitle("TFMN4 4MXB: whole-genome (WGS) vs targeted amplicon sequencing of the SAME 11 cultures", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "wgs_vs_amplicon_TFMN4.png", dpi=140, bbox_inches="tight"); plt.close(fig)


@styled
def fig_cn(cn_sum):
    sets = [d for d in CNS]
    fig, ax = plt.subplots(figsize=(12, 0.75 * len(sets) + 2.2))
    labels = []
    for i, ds in enumerate(sets):
        r = rd(ROOT / ds["out"] / f"{ds['tag']}_growth_rates.csv")
        if r is None:
            labels.append(ds["name"]); continue
        rel = r[r.tier == "reliable"]
        sig = rel[rel.call.isin(["amplifying", "losing"])]
        ns = rel[~rel.call.isin(["amplifying", "losing"])]
        jit = (np.random.default_rng(i).random(len(ns)) - 0.5) * 0.36
        ax.scatter(ns.rate_per_transfer, i + jit, s=22, facecolors="white", edgecolors=MUTED, linewidths=1, zorder=2)
        jit = (np.random.default_rng(i + 99).random(len(sig)) - 0.5) * 0.36
        ax.scatter(sig.rate_per_transfer, i + jit, s=40, color=BLUE, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        labels.append(f"{ds['tag']} · {ds['carbon']} · {ds['platform']}  "
                      f"({len(rel)} reliable, {len(sig)} FDR-sig.)")
    ax.axvline(0, color=BASELINE, lw=1.2, ls="--")
    ax.set_yticks(range(len(sets))); ax.set_yticklabels(labels, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel("amplification rate  r = d ln(copy number) / d transfer   (reliable tier: ≥4 transfers)")
    ax.set_title("Copy-number amplification per environment (each fitted separately)\n"
                 "filled = BH-FDR significant (q<0.05); hollow = not significant  ·  "
                 "EASy rates are per EASy transfer (not comparable to robotic transfers)", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "copy_number_by_environment.png", dpi=140, bbox_inches="tight"); plt.close(fig)


# =========================================================================== report
def md_table(df, cols, fmt=None):
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt and pd.notna(v):
                v = fmt[c].format(v)
            cells.append("" if (isinstance(v, float) and np.isnan(v)) else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def report(summ, cn_sum, along, conc, st, val, sens=None):
    inv = LIMS / "inventory"
    tables = pd.read_csv(inv / "tables.csv")
    orders = pd.read_csv(inv / "barcode_orders.csv")
    cells = pd.read_csv(inv / "barcode_cells.csv")
    cult = pd.read_csv(inv / "barcode_cultures.csv")
    ctl = pd.read_csv(inv / "barcode_nodna_controls.csv")
    nm = pd.read_csv(inv / "barcode_not_modelable.csv")
    L = []
    L.append("# LIMS mirror re-analysis — every environment simulated separately\n")
    L.append(f"_Auto-generated by `scripts/lims_compare.py` from `{MAN['source_db']}` "
             f"({MAN['source_db_bytes']/1e6:.0f} MB). Interpretation and caveats: "
             f"[`docs/LIMS_REANALYSIS.md`](../docs/LIMS_REANALYSIS.md)._\n")
    L.append("## 1. What is in the database\n")
    used = {"verAB_barcodes": "barcode counts (s01)", "Copy_numbers": "copy-number trajectories (cn01)",
            "Robotic_OD": "OD curves (s02, s08)", "Robotic_ALE_samples": "culture -> condition / strain / well",
            "Samples": "metadata fallback (EASy lineages)", "Conditions": "carbon source + concentration"}
    tt = tables.copy()
    tt["role_here"] = tt["table"].map(used).fillna("inventoried only (no method in this repo consumes it)")
    L.append(f"{len(tables)} tables (every LIMS table keeps soft-deleted row versions; only live rows are used):\n")
    L.append(md_table(tt, ["table", "rows_live", "rows_soft_deleted", "role_here"]))
    L.append("\nBarcode sequencing orders (live rows):\n")
    L.append(md_table(orders, ["Seqorder", "assay", "n_seqsamples", "n_time_series_seqsamples", "reads", "frac_reads_unk_allele"],
                      {"frac_reads_unk_allele": "{:.1%}"}))
    sel = cells[cells.selected_run]
    L.append(f"\n**Barcode QC.** {len(cells)} culture×transfer×order cells. "
             f"{int((~cells.selected_run).sum())} superseded by a deeper re-sequencing of the same cell; "
             f"{int(((sel.depth < sel.min_depth)).sum())} below the amplicon depth floor "
             f"({MAN['qc']['min_depth_amplicon']} identified reads). "
             f"{MAN['n_reads_unk_dropped']:,} reads with an un-called allele (A_UNK/B_UNK) dropped "
             f"(median per-cell share {sel.frac_reads_unk_allele.median():.1%}); cultures where a half-called "
             f"lineage exceeds 10% of reads are flagged:\n")
    fl = cult[cult.flag_unk_lineage_over_10pct == True]
    L.append(md_table(fl, ["culture", "included", "max_frac_reads_unk", "top_unk_lineage", "max_unk_transfer"],
                      {"max_frac_reads_unk": "{:.0%}"}))
    if len(ctl):
        L.append("\n**Contamination check (noDNA controls should carry no barcodes):**\n")
        L.append(md_table(ctl, ["culture", "transfer", "reads", "n_variants", "top_variants"]))
    L.append(f"\n**Not modelable with the repo's time-series methods:** {len(nm)} seqsamples "
             f"({', '.join(f'{k}: {v}' for k, v in nm.reason.str.split(';').str[0].value_counts().items())}).\n")
    L.append("## 2. Environments simulated (one pipeline run each)\n")
    L.append(md_table(summ, ["environment", "role", "n_cultures", "transfers", "median_depth", "n_variants",
                             "hours_per_transfer", "generations_per_cycle"]))
    L.append("\n## 3. Validation — the DB reproduces the repo\n")
    if val is not None:
        nid = int((val.status == "identical").sum())
        L.append(f"The WGS counts ingested from the DB were run through the same code and compared with the "
                 f"committed `outputs/`: **{nid}/{len(val)} output tables identical**. Differences:\n")
        L.append(md_table(val[val.status != "identical"], ["file", "status", "max_abs_numeric_diff", "column_of_max",
                                                           "text_cells_differing"], {"max_abs_numeric_diff": "{:.3g}"}))
        txt_only = val[(val.status != "identical") & (val.max_abs_numeric_diff == 0)]["file"].tolist()
        num_diff = val.loc[val.max_abs_numeric_diff.fillna(0) > 0, "file"].tolist()
        g = s12_vs_repo()
        if g is not None and num_diff == ["global_variant_growth_rates.csv"]:
            L.append(f"\nEvery number is identical except in `global_variant_growth_rates.csv`. There the s12 point "
                     f"estimates `r_global_per_h` and `se_laplace` agree to {g['s12']:.4f}/h (4th-decimal rounding), "
                     f"while the s12b bootstrap / LOSO / OD-weight-sensitivity columns differ by up to {g['s12b']:.3f}/h "
                     f"({g['ident']}/{g['n']} `identifiability` labels change): "
                     + ("the LIMS runs solve those refits to convergence, whereas the committed `outputs/` holds the "
                        "repo's iteration-capped refits, which are not reproducible across machines anyway "
                        "(docs/LIMS_REANALYSIS.md §3). " if g["converged"] else
                        "those are iteration-capped refits, which are not reproducible across machines "
                        "(docs/LIMS_REANALYSIS.md §3). "))
        else:
            L.append(f"\nFiles with numeric differences: {', '.join(f'`{f}`' for f in num_diff) or 'none'}. ")
        L.append(f"{', '.join(f'`{f}`' for f in txt_only)} differ only in free text: the `dataset` label, and the "
                 "`interpretation` sentences, which LIMS runs generate from their own numbers — on this data they "
                 "reach the same conclusions the repo states (LOOCV parity once the elbow is learned, a clear gain "
                 "when it is supplied, a larger optimism gap for the 4-var).\n")
    L.append("## 4. Barcode results per environment\n")
    L.append(md_table(summ, ["environment", "richness_first_median", "richness_last_median", "mu_bulk_median_per_h",
                             "n_estimable", "s_per_gen_median", "s_per_gen_p95", "top_winner_verA",
                             "top_winner_verA_wins", "n_robust_verA", "n_robust_verB"],
                      {"richness_first_median": "{:.0f}", "richness_last_median": "{:.0f}", "s_per_gen_median": "{:+.3f}",
                       "s_per_gen_p95": "{:+.3f}"}))
    L.append("\n")
    L.append(md_table(summ, ["environment", "top_robust_verA", "top_robust_verB", "bridge_components",
                             "global_fit_freq_R2", "global_fit_OD_rmse_per_h", "global_rate_min", "global_rate_max"],
                      {"global_fit_freq_R2": "{:.3f}", "global_fit_OD_rmse_per_h": "{:.3f}"}))
    L.append("\n_`global_fit_OD_rmse_per_h` = mismatch between the model's community rate and the measured "
             "μ_bulk; large values mean the per-hour level of `r_global_per_h` is not OD-anchored — read the "
             "rates relative to μ̄ (`rel_advantage_per_h`)._\n")
    L.append("\n**4-parameter (breakpoint) model:**\n")
    L.append(md_table(summ, ["environment", "n_transfers", "n_estimable", "n_breakpoint_testable", "n_two_phase_adopted",
                             "n_held_decrease", "loocv_ratio_elbow_learned", "loocv_4var_better_elbow_learned",
                             "loocv_ratio_elbow_supplied"]))
    L.append("\nPer-environment folders (tables + the full figure suite of the original pipeline):\n")
    for ds in BC:
        L.append(f"- `{ds['out']}/` — {label(ds)}")
    L.append("\n![overview](comparison/figures/environment_overview.png)\n")
    L.append("![top variants](comparison/figures/top_variants_by_environment.png)\n")
    for sd, base, df in (sens or []):
        cand = ", ".join(sd["excluded"])
        disc = rd(CMP / "cross_assay_discordance_TFMN4.csv")
        L.append(f"### 4b. Suspected contamination in {label(base)}: {cand}\n")
        L.append(f"{sd['exclusion_reason']}.\n")
        if disc is not None and len(disc):
            dg = (disc.groupby("Candidate").agg(n_cells=("Sample", "size"),
                                                median_amplicon_freq=("freq_amp", "median"),
                                                wgs_reads=("Count_wgs", "sum")).reset_index()
                  .sort_values("n_cells", ascending=False))
            L.append("Systematic cross-assay check (same culture & transfer, both directions; a variant counts as "
                     "discordant when the lower 95% bound of its frequency in one assay predicts reads in the other "
                     "but none were seen; Poisson P below a Bonferroni threshold over all tested cells):\n")
            L.append(md_table(dg, ["Candidate", "n_cells", "median_amplicon_freq", "wgs_reads"],
                              {"median_amplicon_freq": "{:.1%}"}))
        L.append(f"\nThe main run keeps every read; the sensitivity run `{sd['out']}/` drops {cand}. "
                 "What changes:\n")
        L.append(md_table(df, ["metric", "value"]))
        L.append(f"\n![sensitivity](comparison/figures/sensitivity_excl_{sd['name'].split('_excl_')[1]}.png)\n")
    L.append("## 5. Across environments\n")
    L.append(md_table(conc, ["env_a", "env_b", "n_shared_alleles", "spearman", "n_robust_in_both", "robust_sign_agreement"],
                      {"spearman": "{:+.2f}"}))
    L.append("\n![alleles](comparison/figures/allele_effects_heatmap.png)\n")
    L.append("![concordance](comparison/figures/allele_concordance.png)\n")
    if st:
        L.append("### WGS vs amplicon on the same TFMN4 cultures\n")
        L.append(f"- frequency concordance (variants seen by both): Pearson r(log) = {st['freq_pearson_log']:.2f}, "
                 f"Spearman = {st['freq_spearman']:.2f}; {st['frac_wgs_variants_seen_in_amplicon']:.0%} of WGS-detected "
                 f"variant×cells are also seen by the amplicon; dominant variant agrees in "
                 f"{st['dominant_variant_agreement']:.0%} of {st['n_cells_compared']} cells.")
        L.append(f"- same transfers (s per cycle on T3/T4/T6/T13): Spearman = {st['slope_spearman_same_4T']:.2f} over "
                 f"{st['n_slope_pairs']} trajectories. Absolute levels are NOT comparable across assays (each slope is "
                 "relative to the geometric mean of the variants that assay detects), so alleles are compared as a "
                 "difference from the median allele / percentile:")
        at = st["allele_table"]
        L.append("")
        L.append(md_table(at[at.verA.isin(["A90", "A78", "A81", "A83", "A183"])],
                          ["verA", "wgs_4T_minus_median", "wgs_4T_percentile", "amp_4T_minus_median",
                           "amp_4T_percentile", "amp_allT_minus_median", "amp_allT_percentile"]))
        L.append(f"- estimable trajectories: WGS {st['n_estimable_wgs']} → amplicon (8 transfers) {st['n_estimable_amp8']}; "
                 f"amplicon 4- vs 8-transfer slopes Spearman = {st['slope_spearman_amp4T_vs_amp8T']:.2f}.\n")
        L.append("![wgs vs amplicon](comparison/figures/wgs_vs_amplicon_TFMN4.png)\n")
    L.append("## 6. Copy-number environments\n")
    L.append(md_table(cn_sum, ["environment", "n_lineages", "n_ci_reliable", "n_amplifying_fdr", "n_losing_fdr",
                               "median_rate_reliable", "top_amplifier", "n_breakpoint_testable", "n_two_phase_adopted",
                               "n_decelerating"], {"median_rate_reliable": "{:+.4f}"}))
    L.append("\n")
    for ds in CNS:
        L.append(f"- `{ds['out']}/` — {ds['tag']} copy number, {', '.join(ds['conditions'])} ({ds['platform']})")
    L.append("\n![copy number](comparison/figures/copy_number_by_environment.png)\n")
    (OUT / "REPORT.md").write_text("\n".join(L))


def main():
    summ = barcode_summary()
    cn_sum = cn_summary()
    along, conc = allele_tables()
    vlong = variant_tables()
    fq, sl, st = wgs_vs_amplicon()
    val = validation()
    sens = sensitivity() if SENS else []
    figs = [("environment_overview", lambda: fig_overview(summ)),
            ("allele_effects_heatmap", lambda: fig_allele_heatmap(along)),
            ("allele_concordance", lambda: fig_allele_concordance(along, conc)),
            ("top_variants_by_environment", lambda: fig_top_variants(vlong)),
            ("wgs_vs_amplicon_TFMN4", lambda: fig_wgs_vs_amp(fq, sl, st) if fq is not None else None),
            ("copy_number_by_environment", lambda: fig_cn(cn_sum)),
            ("sensitivity", lambda: fig_sensitivity(sens))]
    for name, fn in figs:
        try:
            fn(); print(f"[lims_compare] wrote figures/{name}.png")
        except Exception as e:
            print(f"[lims_compare] figure {name} skipped ({type(e).__name__}: {e})")
    if st:
        pd.Series({k: v for k, v in st.items() if k != "allele_table"}).to_csv(
            CMP / "wgs_vs_amplicon_TFMN4_stats.csv", header=["value"])
    report(summ, cn_sum, along, conc, st, val, sens)
    print(f"[lims_compare] {len(summ)} barcode + {len(cn_sum)} copy-number environments summarised; "
          f"wrote {CMP.relative_to(ROOT)}/ and {(OUT / 'REPORT.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
