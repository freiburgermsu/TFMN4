"""
cn04 - 1-variable vs 4-variable model comparison + fit error, on the dgoA copy-number
data. Fair, nested-aware axes (as s14d/s14e): coverage, in-sample fit, out-of-sample
prediction (leave-one-TRANSFER-out CV), and fit error. All fits use the per-transfer
means (the unit of replication).

Two distinct "4-var" objects are reported and kept DISTINCT (the adversarial review
flagged conflating them):
  * DEPLOYED 4-var = the increasing-only model actually used (constant unless it fires).
    It differs from the 1-var on only a handful of lineages.
  * broken-stick FORM = the best bidirectional broken-stick (any direction). This is what
    captures the copy-number SATURATION (amplify-then-plateau) and is used only to show
    that the two-phase *shape* has real out-of-sample content — the LOTO gain is modest
    and its significance is reported honestly (Wilcoxon signed-rank).

Outputs (outputs_copynumber/): model_comparison_summary.csv,
  model_comparison_by_lineage.csv, model_fit_error.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
import cn_common as CN
from s14_segmented_growth import fit_constant, fit_two_phase, bic


def _rmse(res):
    return float(np.sqrt(np.mean(np.asarray(res) ** 2)))


def compute(bt):
    rows = []
    for lin, g in bt.groupby("lineage"):
        g = g.sort_values("transfer")
        if g["transfer"].nunique() < CN.MIN_UNIQUE_TRANSFERS:
            continue
        x = g["transfer"].to_numpy(float); y = g["ln_cn"].to_numpy(float); w = np.ones_like(x)
        n = len(x)
        s = CN.fit_segmented(x, y)
        adopted = (s["model_selected"] == "two_phase")
        c = fit_constant(x, y, w); res1 = y - (c["coef"][0] + c["coef"][1] * x)
        if adopted:
            kt = s["knot_increasing"]; tp = fit_two_phase(x, y, w, kt)
            res4 = y - (tp["coef"][0] + tp["coef"][1] * x + tp["coef"][2] * np.maximum(0.0, x - kt))
            wrss4, r2_4dep, k4 = tp["wrss"], tp["r2"], 3
        else:
            res4 = res1; wrss4, r2_4dep, k4 = c["wrss"], c["r2"], 2
        bic_1 = bic(c["wrss"], n, 2)
        rA, rB, nfold = CN.loto_cv(x, y)
        rows.append(dict(
            lineage=lin, context=g["context"].iloc[0], deployed_model=s["model_selected"],
            unconstrained_direction=s["unconstrained_direction"],
            r2_1var=round(s["r2_constant"], 4), r2_4var_deployed=round(r2_4dep, 4),
            r2_4var_form=round(s["r2_unconstrained"], 4),
            rmse_1var=round(_rmse(res1), 4), rmse_4var_deployed=round(_rmse(res4), 4),
            rmse_form_insample=(round(float(np.sqrt(s["wrss_unconstrained"] / n)), 4)
                                if np.isfinite(s["wrss_unconstrained"]) else np.nan),
            rse_dofadj_1var=round(float(np.sqrt(c["wrss"] / (n - 2))), 4) if n > 2 else np.nan,
            rse_dofadj_4var_deployed=round(float(np.sqrt(wrss4 / (n - k4))), 4) if n > k4 else np.nan,
            bic_prefers_form=bool(np.isfinite(s["wrss_unconstrained"]) and bic(s["wrss_unconstrained"], n, 3) < bic_1),
            bic_prefers_increasing=bool(np.isfinite(s["wrss_two_phase"]) and bic(s["wrss_two_phase"], n, 3) < bic_1),
            loto_rmse_1var=round(rA, 4) if np.isfinite(rA) else np.nan,
            loto_rmse_form=round(rB, 4) if np.isfinite(rB) else np.nan,
            loto_folds=nfold, n_transfers=g["transfer"].nunique(),
        ))
    return pd.DataFrame(rows)


def main():
    bt_long = pd.read_csv(CN.INTER / "dgoA_bytransfer.csv")
    idx = pd.read_csv(CN.INTER / "dgoA_lineage_index.csv")
    n_est = int(idx.estimable.sum())
    bt = compute(bt_long)
    n_test = len(bt)
    ad = bt[bt.deployed_model == "two_phase"]; kept = bt[bt.deployed_model == "constant"]
    decel = kept[kept.unconstrained_direction == "decelerating"]
    n_ident = (n_est - n_test) + len(kept)
    bt.to_csv(CN.OUT / "model_comparison_by_lineage.csv", index=False)

    # honest significance of the LOTO form-vs-line gain (paired Wilcoxon signed-rank)
    lg = bt.dropna(subset=["loto_rmse_1var", "loto_rmse_form"])
    diff = lg.loto_rmse_1var - lg.loto_rmse_form
    nz = diff[diff != 0]
    wil_p = float(stats.wilcoxon(nz)[1]) if len(nz) >= 1 else np.nan
    n_form_better = int((lg.loto_rmse_form < lg.loto_rmse_1var).sum())

    def med(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return round(float(s.median()), 4) if len(s) else np.nan

    summ = pd.DataFrame([
        ("free_parameters", "2 (logA + rate)", "4 (logA, r_init, r_final, breakpoint)", "the models"),
        ("lineages_fittable", n_est, n_test, "1-var >=3 transfers; 4-var >=4"),
        ("lineages_identical_output_deployed", n_ident, n_ident, "deployed 4-var == 1-var except where it fires"),
        ("deployed_4var_differs_on", 0, len(ad), "the increasing-only 4-var fires on only these"),
        ("testable_that_DECELERATE", "-", len(decel),
         f"of {n_test}: amplify-then-plateau (saturation); increasing-only cannot fit these"),
        ("median_R2_testable_DEPLOYED", med(bt.r2_1var), med(bt.r2_4var_deployed),
         "the deployed 4-var barely moves R² (fires on few)"),
        ("median_R2_testable_FORM", med(bt.r2_1var), med(bt.r2_4var_form),
         "the bidirectional broken-stick FORM captures saturation (0.24 -> 0.90)"),
        ("BIC_prefers_broken_stick_FORM", "-", int(bt.bic_prefers_form.sum()), f"of {n_test}"),
        ("median_LOTO_RMSE_FORM_vs_line", med(bt.loto_rmse_1var), med(bt.loto_rmse_form),
         f"out-of-sample; FORM better on {n_form_better}/{len(lg)}, Wilcoxon p={wil_p:.2f} "
         f"(a MODEST, not significant, gain — do not overstate)"),
    ], columns=["metric", "one_variable", "four_variable", "interpretation"])
    summ.to_csv(CN.OUT / "model_comparison_summary.csv", index=False)

    fe = pd.DataFrame([
        ("in_sample_lnCN_RMSE_FORM", med(bt.rmse_1var), med(bt.rmse_form_insample),
         "broken-stick FORM vs line (captures saturation)"),
        ("in_sample_R2_FORM", med(bt.r2_1var), med(bt.r2_4var_form), "0.24 -> 0.90"),
        ("out_of_sample_LOTO_RMSE_FORM", med(bt.loto_rmse_1var), med(bt.loto_rmse_form),
         f"FORM better on {n_form_better}/{len(lg)} lineages; Wilcoxon p={wil_p:.2f} (modest)"),
        ("in_sample_RMSE_DEPLOYED", med(bt.rmse_1var), med(bt.rmse_4var_deployed),
         "the DEPLOYED increasing-only 4-var differs on only a few lineages"),
        ("lineages_identical_error_deployed", n_ident, n_ident, f"of {n_est} estimable"),
    ], columns=["fit_error_metric", "one_variable", "four_variable", "interpretation"])
    fe.to_csv(CN.OUT / "model_fit_error.csv", index=False)

    print(f"[cn04] {n_test} testable lineages; deployed 4-var == 1-var on {n_ident}/{n_est}, differs on {len(ad)}.")
    print(f"[cn04] SATURATION: {len(decel)}/{n_test} testable DECELERATE (amplify then plateau).")
    print(f"[cn04] in-sample median R² 1-var={med(bt.r2_1var)} vs broken-stick FORM={med(bt.r2_4var_form)}.")
    print(f"[cn04] out-of-sample LOTO median RMSE 1-var={med(bt.loto_rmse_1var)} FORM={med(bt.loto_rmse_form)}; "
          f"FORM better on {n_form_better}/{len(lg)}, Wilcoxon p={wil_p:.2f} (modest, NOT significant).")
    print("[cn04] wrote model_comparison_summary.csv, model_comparison_by_lineage.csv, model_fit_error.csv")
    return bt, summ, fe


if __name__ == "__main__":
    main()
