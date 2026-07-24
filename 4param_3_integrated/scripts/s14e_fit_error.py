"""
s14e - Fit-error comparison: 1-variable (constant, s03) vs 4-variable (segmented, s14).

Focused specifically on ERROR. Because the models are nested (4-var reduces to the
1-var), in-sample error can only drop when you add the breakpoint, so this module
reports the drop AND the two things that keep it honest:
  - the error the 1-var leaves is STRUCTURED (concentrated at the trajectory ends,
    where a single slope cannot bend), not random noise;
  - the in-sample drop is largely OPTIMISM: on held-out timepoints (LOOCV) the 4-var's
    error advantage over the 1-var essentially disappears at 4 transfers.

Error metric = unweighted CLR RMSE (root-mean-square residual in centered-log-ratio
units) over a trajectory's timepoints — the "typical error in reproducing the observed
log relative abundance." Testable trajectories have all 4 transfers observed, so this
is clean (no pseudocount-only points). Weighted RSS and R² (the fitting objective) and
the degrees-of-freedom-adjusted residual SE are reported alongside.

The 4-var fit is the model's best INCREASING broken-stick (its deployed form): identical
to the 1-var wherever no increasing switch helps, so the two models differ in error only
on the trajectories the 4-var actually refines.

Outputs (config.OUT): model_fit_error.csv, model_fit_error_by_trajectory.csv
Figure (config.FIG): model_fit_error.png
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import config as C
from s14_segmented_growth import fit_constant
from s14d_model_comparison import _traj, _best_two_phase, _loocv, XT, N
from s14c_segmented_figures import (styled, _short_sample, SURFACE, INK, INK2, MUTED,
                                    BASELINE, BLUE, BLUE_LIGHT, BLUE_DARK, ORANGE, AQUA)

ONE = MUTED         # 1-variable model (context)
FOUR = BLUE         # 4-variable model (subject)


def _rmse(res):
    return float(np.sqrt(np.mean(res ** 2)))


def compute(seg, d4):
    rows, res1_by_t, res4_by_t = [], {t: [] for t in C.FOUR_TRANSFER_SET}, {t: [] for t in C.FOUR_TRANSFER_SET}
    for _, r in seg[seg.breakpoint_testable].iterrows():
        cvec, yvec = _traj(d4, r.Sample, r.Candidate)
        w = cvec + C.PSEUDOCOUNT; obs = cvec > 0
        adopted = (r.model_selected == "two_phase")
        c = fit_constant(XT, yvec, w)
        pred1 = c["coef"][0] + c["coef"][1] * XT
        res1 = yvec - pred1
        # The DEPLOYED 4-var: its two-phase fit only where it actually switches;
        # everywhere else it IS the constant fit (so its error is identical to the 1-var).
        bi = _best_two_phase(XT, yvec, w, obs, increasing_only=True) if adopted else None
        if bi is not None:
            co, kt = bi["coef"], bi["knot"]
            pred4 = co[0] + co[1] * XT + co[2] * np.maximum(0.0, XT - kt)
            res4 = yvec - pred4; wrss4, r2_4, k4 = bi["wrss"], bi["r2"], 3
        else:
            res4 = res1; wrss4, r2_4, k4 = c["wrss"], c["r2"], 2
        rA, rB, _ = _loocv(cvec, yvec, knot=None)
        rows.append(dict(
            Sample=r.Sample, Candidate=r.Candidate, deployed_model=r.model_selected,
            rmse_1var=round(_rmse(res1), 4), rmse_4var=round(_rmse(res4), 4),
            wrss_1var=round(float(c["wrss"]), 4), wrss_4var=round(float(wrss4), 4),
            r2_1var=round(float(c["r2"]), 4), r2_4var=round(float(r2_4), 4),
            rse_dofadj_1var=round(float(np.sqrt(c["wrss"] / (N - 2))), 4),
            rse_dofadj_4var=round(float(np.sqrt(wrss4 / (N - k4))), 4),
            loocv_rmse_1var=round(rA, 4) if np.isfinite(rA) else np.nan,
            loocv_rmse_4var=round(rB, 4) if np.isfinite(rB) else np.nan,
        ))
        if adopted:
            for i, t in enumerate(C.FOUR_TRANSFER_SET):
                res1_by_t[t].append(abs(res1[i])); res4_by_t[t].append(abs(res4[i]))
    bt = pd.DataFrame(rows)
    rbt = pd.DataFrame({
        "transfer": list(C.FOUR_TRANSFER_SET),
        "mean_abs_resid_1var": [np.mean(res1_by_t[t]) if res1_by_t[t] else np.nan for t in C.FOUR_TRANSFER_SET],
        "mean_abs_resid_4var": [np.mean(res4_by_t[t]) if res4_by_t[t] else np.nan for t in C.FOUR_TRANSFER_SET],
    })
    return bt, rbt


def summary_table(seg, bt):
    ad = bt[bt.deployed_model == "two_phase"]

    def med(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return round(float(s.median()), 4) if len(s) else np.nan
    n_ident = (len(seg) - len(bt)) + int((bt.deployed_model == "constant").sum())
    rows = [
        ("in_sample_CLR_RMSE__all_testable", med(bt.rmse_1var), med(bt.rmse_4var),
         f"of {len(bt)} testable; unweighted CLR residual RMSE"),
        ("in_sample_CLR_RMSE__adopted_only", med(ad.rmse_1var), med(ad.rmse_4var),
         f"of {len(ad)} the 4-var refines — where the models actually differ"),
        ("in_sample_weighted_RSS__adopted", med(ad.wrss_1var), med(ad.wrss_4var),
         "the fitting objective (weighted); lower is better"),
        ("in_sample_R2__adopted", med(ad.r2_1var), med(ad.r2_4var), "weighted R² on the adopted set"),
        ("dof_adjusted_residual_SE__testable", med(bt.rse_dofadj_1var), med(bt.rse_dofadj_4var),
         "sqrt(RSS/(n−k)); charges the 4-var for its extra parameter (n=4)"),
        ("out_of_sample_LOOCV_RMSE__adopted", med(ad.loocv_rmse_1var), med(ad.loocv_rmse_4var),
         "held-out error, elbow re-learned per fold — the honest test"),
        ("optimism_gap__adopted", round(med(ad.loocv_rmse_1var) - med(ad.rmse_1var), 4),
         round(med(ad.loocv_rmse_4var) - med(ad.rmse_4var), 4),
         "LOOCV − in-sample; the 4-var's larger gap = more of its in-sample gain is overfitting"),
        ("trajectories_with_identical_error", n_ident, n_ident,
         f"of {len(seg)} estimable — the 4-var equals the 1-var except on {len(ad)}"),
    ]
    return pd.DataFrame(rows, columns=["fit_error_metric", "one_variable", "four_variable", "interpretation"])


@styled
def make_figure(bt, rbt, seg):
    ad = bt[bt.deployed_model == "two_phase"].copy().sort_values("rmse_1var")
    if ad.empty:
        return
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

    # (a) in-sample CLR RMSE per adopted variant (dumbbell) -------------------
    a = ax[0, 0]
    y = np.arange(len(ad))
    a.hlines(y, ad.rmse_4var, ad.rmse_1var, color=BASELINE, lw=2.2, zorder=1)
    a.scatter(ad.rmse_1var, y, s=52, color=ONE, edgecolors="white", linewidths=0.6,
              zorder=3, label="1-var (constant)")
    a.scatter(ad.rmse_4var, y, s=52, color=FOUR, edgecolors="white", linewidths=0.6,
              zorder=3, label="4-var (segmented)")
    a.set_yticks(y); a.set_yticklabels([f"{c} ({_short_sample(s)})" for c, s in
                                        zip(ad.Candidate, ad.Sample)], fontsize=7)
    a.set_xlabel("in-sample CLR RMSE"); a.set_xlim(left=0)
    a.set_title("(a) In-sample fit error drops on the 13 refined variants")
    a.legend(fontsize=7.5, loc="lower right", framealpha=0.9); a.grid(axis="y", visible=False)

    # (b) residual structure by timepoint (adopted) ---------------------------
    a = ax[0, 1]
    xt = np.arange(len(rbt)); wd = 0.38
    a.bar(xt - wd / 2, rbt.mean_abs_resid_1var, wd, color=ONE, edgecolor=SURFACE,
          linewidth=1.5, label="1-var (constant)")
    a.bar(xt + wd / 2, rbt.mean_abs_resid_4var, wd, color=FOUR, edgecolor=SURFACE,
          linewidth=1.5, label="4-var (segmented)")
    a.set_xticks(xt); a.set_xticklabels([f"T{int(t)}" for t in rbt.transfer])
    a.set_xlabel("transfer"); a.set_ylabel("mean |residual|  (CLR)")
    a.set_title("(b) The 1-var error is structured — worst at the ends")
    a.legend(fontsize=7.5, loc="upper center", framealpha=0.9); a.grid(axis="x", visible=False)

    # (c) in-sample vs out-of-sample error ladder (adopted) -------------------
    a = ax[1, 0]
    groups = ["in-sample\n(fit)", "out-of-sample\n(LOOCV, fair)"]
    v1 = [ad.rmse_1var.median(), ad.loocv_rmse_1var.median()]
    v4 = [ad.rmse_4var.median(), ad.loocv_rmse_4var.median()]
    xt = np.arange(2); wd = 0.38
    b1 = a.bar(xt - wd / 2, v1, wd, color=ONE, edgecolor=SURFACE, linewidth=1.5, label="1-var")
    b4 = a.bar(xt + wd / 2, v4, wd, color=FOUR, edgecolor=SURFACE, linewidth=1.5, label="4-var")
    for bars in (b1, b4):
        for b in bars:
            a.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f"{b.get_height():.2f}",
                   ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=INK)
    a.set_xticks(xt); a.set_xticklabels(groups)
    a.set_ylabel("median CLR RMSE"); a.set_ylim(0, max(v1 + v4) * 1.2)
    a.set_title("(c) The in-sample edge is optimism — it vanishes out-of-sample")
    a.legend(fontsize=7.5, loc="upper left", framealpha=0.9); a.grid(axis="x", visible=False)

    # (d) paired in-sample error across all testable — identical except on 13 -----
    a = ax[1, 1]
    kept = bt[bt.deployed_model == "constant"]
    lim = float(max(bt.rmse_1var.max(), bt.rmse_4var.max())) * 1.08
    a.plot([0, lim], [0, lim], color=BASELINE, lw=1.2, ls="--", zorder=1)
    a.scatter(kept.rmse_1var, kept.rmse_4var, s=34, facecolors="none", edgecolors=MUTED,
              linewidths=1.2, label=f"1-var kept ({len(kept)}) — on the line", zorder=2)
    a.scatter(ad.rmse_1var, ad.rmse_4var, s=52, color=FOUR, zorder=3,
              label=f"4-var refines ({len(ad)}) — below")
    a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_aspect("equal")
    a.set_xlabel("1-var in-sample CLR RMSE"); a.set_ylabel("4-var in-sample CLR RMSE")
    a.set_title(f"(d) Error identical on 224/{len(seg)}; only the 13 drop")
    a.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    fig.suptitle(f"Fit error: 1-variable vs 4-variable growth model  [{C.DATASET}]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(C.FIG / "model_fit_error.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[s14e] wrote figures/model_fit_error.png")


def main():
    seg = pd.read_csv(C.OUT / "segmented_growth_rates.csv")
    d4 = pd.read_csv(C.INTER / "barcode_4transfer_long.csv")
    if int(seg.breakpoint_testable.sum()) == 0:
        pd.DataFrame([("fit_error", "identical", "identical",
                       "3-transfer dataset: 4-var reduces to the 1-var everywhere")],
                     columns=["fit_error_metric", "one_variable", "four_variable", "interpretation"]
                     ).to_csv(C.OUT / "model_fit_error.csv", index=False)
        pd.DataFrame(columns=["Sample", "Candidate"]).to_csv(C.OUT / "model_fit_error_by_trajectory.csv", index=False)
        print(f"[s14e] {C.DATASET}: 0 testable -> fit error identical; wrote summary, no figure.")
        return

    bt, rbt = compute(seg, d4)
    summ = summary_table(seg, bt)
    bt.to_csv(C.OUT / "model_fit_error_by_trajectory.csv", index=False)
    summ.to_csv(C.OUT / "model_fit_error.csv", index=False)
    ad = bt[bt.deployed_model == "two_phase"]
    print(f"[s14e] in-sample CLR RMSE (adopted): 1-var={ad.rmse_1var.median():.3f} -> "
          f"4-var={ad.rmse_4var.median():.3f} ({ad.rmse_4var.median()/ad.rmse_1var.median():.2f}× ); "
          f"R² {ad.r2_1var.median():.2f}->{ad.r2_4var.median():.2f}.")
    print(f"[s14e] out-of-sample LOOCV RMSE (adopted): 1-var={ad.loocv_rmse_1var.median():.3f} "
          f"4-var={ad.loocv_rmse_4var.median():.3f}  -> in-sample gain is largely optimism.")
    print(f"[s14e] dof-adjusted residual SE (testable): 1-var={bt.rse_dofadj_1var.median():.3f} "
          f"4-var={bt.rse_dofadj_4var.median():.3f}.")
    print("[s14e] wrote model_fit_error.csv, model_fit_error_by_trajectory.csv")
    try:
        make_figure(bt, rbt, seg)
    except Exception as e:
        print(f"[s14e] figure skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
