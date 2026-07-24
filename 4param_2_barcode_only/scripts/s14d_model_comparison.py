"""
s14d - Head-to-head comparison of the 4-variable segmented model (s14) against the
original 1-variable constant-rate model (s03) on the SAME dataset.

The two models are NESTED: the 4-variable broken-stick (logA, r_init, r_final, tau)
reduces exactly to the 1-variable line (logA, s) when r_init == r_final. So a raw
goodness-of-fit contest is rigged — the model with more parameters cannot fit worse.
This module therefore reports the *fair* comparisons:

  1. COVERAGE   - which trajectories each model can fit, and where their output is
                  identical (the 4-var only differs from the 1-var where it is both
                  breakpoint-testable AND its gates fire).
  2. IN-SAMPLE  - weighted R²/RSS (shown, but flagged as favouring the 4-var by
                  construction — not a fair basis on its own).
  3. PARSIMONY  - BIC, which charges for the extra parameters. How many trajectories
                  BIC actually prefers the broken-stick for, and how the increasing-
                  only + strong-evidence policy narrows that to the deployed set.
  4. PREDICTION - leave-one-timepoint-out CV (the fair, out-of-sample test) under two
                  regimes:
                    (a) elbow LEARNED per fold  -> the honest test of the whole model;
                    (b) elbow SUPPLIED (fixed)  -> isolates the two-phase *shape*.
                  The gap between (a) and (b) measures how identifiable the breakpoint
                  location is at this sampling density.
  5. BLIND SPOT - the single slope averages the two phases into one number; how much
                  of the real dynamics that hides.

Outputs (config.OUT): model_comparison_summary.csv, model_comparison_by_trajectory.csv
Figures (config.FIG): model_comparison.png, model_comparison_blindspot.png
Only meaningful where trajectories are breakpoint-testable (>=4 transfers); 3-transfer
datasets get a one-line "models are identical everywhere" summary and no figures.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C
from s14_segmented_growth import (fit_constant, fit_two_phase, wls, bic,
                                   MIN_SEG_OBS, MIN_OBS_FOR_BREAKPOINT)

TRANSFERS = list(C.FOUR_TRANSFER_SET)
XT = np.array(TRANSFERS, float)
N = len(TRANSFERS)
INTERIOR = TRANSFERS[1:-1]


# --------------------------------------------------------------------------- fits
def _traj(d4, sample, cand):
    sub = d4[d4.Sample == sample]
    counts = (sub.pivot_table(index="Candidate", columns="Transfer", values="Count",
                              aggfunc="sum", fill_value=0).reindex(columns=TRANSFERS, fill_value=0))
    clr = C.clr_matrix(counts)
    return counts.loc[cand].to_numpy(float), clr.loc[cand].to_numpy(float)


def _feasible_knots(x, obs):
    ks = []
    for kt in INTERIOR:
        if (obs & (x <= kt)).sum() >= MIN_SEG_OBS and (obs & (x >= kt)).sum() >= MIN_SEG_OBS:
            ks.append(kt)
    return ks


def _best_two_phase(x, y, w, obs, increasing_only):
    best = None
    for kt in _feasible_knots(x, obs):
        tp = fit_two_phase(x, y, w, kt)
        if increasing_only and not (tp["r_final"] > tp["r_init"]):
            continue
        if best is None or tp["wrss"] < best["wrss"]:
            best = tp; best["knot"] = kt
    return best


def _loocv(cvec, yvec, knot=None):
    """Leave-one-timepoint-out CV RMSE for the constant vs broken-stick form, over the
    SAME folds. knot=None -> re-select the best feasible knot per fold (fair test);
    knot=value -> hold the elbow fixed (isolates the shape). Returns (rmse_const,
    rmse_two_phase, n_folds); folds where the two-phase design is rank-deficient are
    skipped for BOTH models so the comparison stays paired."""
    w = cvec + C.PSEUDOCOUNT
    seA, seB = [], []
    for i in range(N):
        tr = [j for j in range(N) if j != i]
        xt, yt, wt = XT[tr], yvec[tr], w[tr]
        knots = [knot] if knot is not None else INTERIOR
        best = None
        for kt in knots:
            h = np.maximum(0.0, xt - kt); XB = np.column_stack([np.ones(3), xt, h])
            if np.linalg.matrix_rank(XB) < 3:
                continue
            cB, wrss, _ = wls(XB, yt, wt)
            if best is None or wrss < best[1]:
                best = (cB, wrss, kt)
        if best is None:
            continue
        cB, _, kt = best
        XA = np.column_stack([np.ones(3), xt]); cA, _, _ = wls(XA, yt, wt)
        predA = np.array([1, XT[i]]) @ cA
        predB = np.array([1, XT[i], max(0.0, XT[i] - kt)]) @ cB
        seA.append((yvec[i] - predA) ** 2); seB.append((yvec[i] - predB) ** 2)
    if len(seB) < 2:
        return np.nan, np.nan, len(seB)
    return float(np.sqrt(np.mean(seA))), float(np.sqrt(np.mean(seB))), len(seB)


# --------------------------------------------------------------------------- compute
def compute(seg, d4):
    dep = seg.set_index(["Sample", "Candidate"])
    rows = []
    for _, r in seg[seg.breakpoint_testable].iterrows():
        cvec, yvec = _traj(d4, r.Sample, r.Candidate)
        w = cvec + C.PSEUDOCOUNT; obs = cvec > 0
        c = fit_constant(XT, yvec, w); bic_c = bic(c["wrss"], N, 2)
        b_unc = _best_two_phase(XT, yvec, w, obs, increasing_only=False)
        b_inc = _best_two_phase(XT, yvec, w, obs, increasing_only=True)
        adopted = (r.model_selected == "two_phase")
        knot_dep = int(r.breakpoint_transfer) if adopted else (
            b_inc["knot"] if b_inc is not None else (b_unc["knot"] if b_unc is not None else None))
        rA_free, rB_free, nf_free = _loocv(cvec, yvec, knot=None)
        rA_fix, rB_fix, nf_fix = (_loocv(cvec, yvec, knot=knot_dep)
                                  if knot_dep is not None else (np.nan, np.nan, 0))
        rows.append(dict(
            Sample=r.Sample, Candidate=r.Candidate, verA=r.verA, verB=r.verB,
            n_obs_timepoints=int(r.n_obs_timepoints),
            s_constant_per_cycle=round(float(c["slope"]), 4),
            logA_init=float(r.logA_init),
            r_init_per_cycle=float(r.r_init_per_cycle), r_final_per_cycle=float(r.r_final_per_cycle),
            breakpoint_transfer=(int(r.breakpoint_transfer) if adopted else np.nan),
            deployed_model=r.model_selected,
            # in-sample fit
            r2_1var=round(float(c["r2"]), 4),
            r2_4var_capability=round(float(b_unc["r2"]), 4) if b_unc is not None else np.nan,
            r2_4var_deployed=round(float(b_inc["r2"]), 4) if adopted and b_inc is not None else round(float(c["r2"]), 4),
            wrss_1var=round(float(c["wrss"]), 5),
            wrss_4var_capability=round(float(b_unc["wrss"]), 5) if b_unc is not None else np.nan,
            # parsimony
            bic_1var=round(bic_c, 3),
            bic_4var_capability=round(bic(b_unc["wrss"], N, 3), 3) if b_unc is not None else np.nan,
            bic_4var_increasing=round(bic(b_inc["wrss"], N, 3), 3) if b_inc is not None else np.nan,
            bic_prefers_4var_capability=bool(b_unc is not None and bic(b_unc["wrss"], N, 3) < bic_c),
            bic_prefers_4var_increasing=bool(b_inc is not None and bic(b_inc["wrss"], N, 3) < bic_c),
            # prediction (LOOCV)
            loocv_rmse_1var_free=round(rA_free, 4) if np.isfinite(rA_free) else np.nan,
            loocv_rmse_4var_free=round(rB_free, 4) if np.isfinite(rB_free) else np.nan,
            loocv_folds_free=nf_free,
            loocv_rmse_1var_fixedelbow=round(rA_fix, 4) if np.isfinite(rA_fix) else np.nan,
            loocv_rmse_4var_fixedelbow=round(rB_fix, 4) if np.isfinite(rB_fix) else np.nan,
            loocv_folds_fixedelbow=nf_fix,
            oos_4var_better_free=bool(np.isfinite(rB_free) and rB_free < rA_free),
        ))
    bt = pd.DataFrame(rows)
    bt["abs_phase_range"] = (bt.r_final_per_cycle - bt.r_init_per_cycle).abs().round(4)
    return bt


# --------------------------------------------------------------------------- tables
def summary_table(seg, bt):
    n_est = len(seg); n_test = int(seg.breakpoint_testable.sum())
    n_nontest = n_est - n_test
    ad = bt[bt.deployed_model == "two_phase"]
    kept = bt[bt.deployed_model == "constant"]
    n_identical = n_nontest + len(kept)

    def med(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return round(float(s.median()), 4) if len(s) else np.nan

    def rows():
        yield ("free_parameters_per_trajectory", "2 (log-abundance + growth rate)",
               "4 (log-abundance, initial rate, final rate, breakpoint)", "the models compared")
        yield ("trajectories_fittable", n_est, n_test, "1-var needs >=2 obs; 4-var needs >=4 (>=2/segment)")
        yield ("trajectories_identical_output", n_identical, n_identical,
               "4-var reduces to 1-var where not testable or gates do not fire")
        yield ("trajectories_where_deployed_differs", 0, len(ad), "the 4-var only refines these")
        yield ("median_wR2_on_testable", med(bt.r2_1var), med(bt.r2_4var_capability),
               "4-var favoured BY CONSTRUCTION (nested, +params) — not a fair basis alone")
        yield ("median_wRSS_on_testable", med(bt.wrss_1var), med(bt.wrss_4var_capability),
               "lower is better; same caveat")
        yield ("BIC_prefers_4var_of_testable", "-", int(bt.bic_prefers_4var_capability.sum()),
               f"unconstrained broken-stick; of {n_test} (n=4 flexibility)")
        yield ("BIC_prefers_increasing_4var_of_testable", "-", int(bt.bic_prefers_4var_increasing.sum()),
               f"of {n_test}")
        yield ("deployed_decision_of_testable", f"{len(kept)} keep 1-var", f"{len(ad)} upgrade to 4-var",
               "increasing-only AND deltaBIC>6 AND +Δrate>10%")
        def ratio(num, den):
            num = pd.to_numeric(num, errors="coerce"); den = pd.to_numeric(den, errors="coerce")
            m = num.notna() & den.notna() & (den > 0)
            return round(float((num[m] / den[m]).median()), 3) if m.any() else np.nan
        yield ("median_LOOCV_RMSE_adopted_elbow_learned", med(ad.loocv_rmse_1var_free),
               med(ad.loocv_rmse_4var_free), "FAIR out-of-sample test (elbow re-learned per fold)")
        yield ("LOOCV_RMSE_ratio_adopted_elbow_learned", "1.00 (parity)",
               ratio(ad.loocv_rmse_4var_free, ad.loocv_rmse_1var_free),
               "median per-trajectory 4-var/1-var; <1 favours 4-var (the fair test ≈ parity)")
        yield ("adopted_4var_predicts_better_elbow_learned", "-",
               f"{int(ad.oos_4var_better_free.sum())}/{len(ad)}", "the honest predictive score")
        yield ("median_LOOCV_RMSE_adopted_elbow_supplied", med(ad.loocv_rmse_1var_fixedelbow),
               med(ad.loocv_rmse_4var_fixedelbow), "elbow FIXED -> isolates the two-phase shape")
        yield ("LOOCV_RMSE_ratio_adopted_elbow_supplied", "1.00 (parity)",
               ratio(ad.loocv_rmse_4var_fixedelbow, ad.loocv_rmse_1var_fixedelbow),
               "with the elbow supplied the 4-var shape predicts markedly better")
        yield ("median_abs_single_slope_adopted", med(ad.s_constant_per_cycle.abs()), "-",
               "what the 1-var reports for the adopted variants")
        yield ("median_abs_phase_range_adopted", "-", med(ad.abs_phase_range),
               "the |r_final - r_init| swing the single slope averages away")
    return pd.DataFrame(list(rows()), columns=["metric", "one_variable", "four_variable", "interpretation"])


# --------------------------------------------------------------------------- main
def main():
    seg = pd.read_csv(C.OUT / "segmented_growth_rates.csv")
    d4 = pd.read_csv(C.INTER / "barcode_4transfer_long.csv")
    n_test = int(seg.breakpoint_testable.sum())

    if n_test == 0:
        summ = pd.DataFrame([
            ("trajectories_fittable", len(seg), 0, "3-transfer dataset: 4-var not identifiable"),
            ("trajectories_identical_output", len(seg), len(seg), "4-var reduces to the 1-var fit everywhere"),
        ], columns=["metric", "one_variable", "four_variable", "interpretation"])
        summ.to_csv(C.OUT / "model_comparison_summary.csv", index=False)
        pd.DataFrame(columns=["Sample", "Candidate"]).to_csv(C.OUT / "model_comparison_by_trajectory.csv", index=False)
        print(f"[s14d] {C.DATASET}: 0 testable -> models identical everywhere; wrote summary, no figures.")
        return

    bt = compute(seg, d4)
    summ = summary_table(seg, bt)
    bt.to_csv(C.OUT / "model_comparison_by_trajectory.csv", index=False)
    summ.to_csv(C.OUT / "model_comparison_summary.csv", index=False)

    ad = bt[bt.deployed_model == "two_phase"]
    print(f"[s14d] {len(bt)} testable compared; 1-var==4-var on {len(bt)-len(ad)}/{len(bt)} "
          f"(+{len(seg)-len(bt)} non-testable), 4-var refines {len(ad)}.")
    print(f"[s14d] in-sample (unfair): median wR² 1-var={summ.loc[summ.metric=='median_wR2_on_testable','one_variable'].iat[0]} "
          f"vs 4-var={summ.loc[summ.metric=='median_wR2_on_testable','four_variable'].iat[0]} (nested).")
    print(f"[s14d] LOOCV adopted — elbow LEARNED (fair): 1-var={ad.loocv_rmse_1var_free.median():.3f} "
          f"4-var={ad.loocv_rmse_4var_free.median():.3f} ({int(ad.oos_4var_better_free.sum())}/{len(ad)} better); "
          f"elbow SUPPLIED: 1-var={ad.loocv_rmse_1var_fixedelbow.median():.3f} "
          f"4-var={ad.loocv_rmse_4var_fixedelbow.median():.3f}.")
    print("[s14d] wrote model_comparison_summary.csv, model_comparison_by_trajectory.csv")

    try:
        import s14d_comparison_figures as figs
        figs.make(bt, seg, d4)
    except Exception as e:
        print(f"[s14d] figures skipped ({type(e).__name__}: {e})")
    return bt, summ


if __name__ == "__main__":
    main()
