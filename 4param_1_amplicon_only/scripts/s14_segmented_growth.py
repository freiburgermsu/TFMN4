"""
s14 - Segmented (two-phase) per-(Sample, Candidate) growth-rate regression.

COMPLEMENTARY to s03 (constant single-slope CLR fit) and s12 (global static rate).
This module does NOT overwrite either; it writes its own output and is offered as
an alternative view for trajectories rich enough to support it.

MODEL (four parameters per variant trajectory)
  A variant's centered-log-ratio (CLR) of read counts is modelled as a CONTINUOUS
  broken-stick (hinge) in time:

      CLR(t) = logA + r_init * (t - t0) + (r_final - r_init) * max(0, t - tau)

    logA      : initial abundance    -> the fitted CLR (log relative abundance) at
                                        the first sampled transfer t0             [1]
    r_init    : initial growth rate   -> slope of the first segment                [2]
    r_final   : final growth rate     -> slope of the second segment               [3]
    tau       : breakpoint            -> the INTER-SAMPLE point (an interior sampled
                                        transfer) where r_init switches to r_final  [4]

  The two segments join continuously at tau (no jump), so a variant that never
  changes rate reduces exactly to the s03 constant-slope line (r_init == r_final).

WHY A BREAKPOINT AT A SAMPLED TRANSFER.  With only a handful of transfers per
culture, a free continuous breakpoint is not identifiable; the biologically natural
and estimable choice is a knot AT one of the interior sampled transfers (the points
"between samples"). tau is therefore searched over TRANSFERS[1:-1] (e.g. {4, 6} for
the WGS design) and the best-fitting knot is selected.

WHEN THE RATE IS ALLOWED TO CHANGE (all must hold; else the constant fit stands)
  0. DIRECTION.  ONLY an INCREASE is permitted: r_final > r_init. A variant whose data
            prefer a growth-rate DECREASE is not allowed a two-phase model and keeps
            the constant fit. (Its would-be decrease is still recorded, for audit, in
            the `unconstrained_direction` column.)
  1. FIT.   The (increasing) two-phase fit must beat the constant fit on a small-sample
            weighted BIC by a strong-evidence margin (Delta BIC > 6 on the Kass-Raftery
            scale: 2-6 positive, 6-10 strong, >10 very strong). BIC already charges for
            the extra parameter; the margin guards against the fact that with only 4
            transfers the 3-parameter model has just 1 residual d.o.f. and is flexible.
            This is the "determination depends on the fit before/after the change" and
            the "doesn't have to change if constant fits better" requirement.
  2. SIZE.  The rate must INCREASE by MORE THAN 10 %:
                (r_final - r_init) / max(|r_init|, R_SCALE_FLOOR) > 0.10
            (the floor only keeps the ratio finite when r_init ~ 0; the BIC gate does
            the real work of preventing trivial switches).

Identifiability: the continuous two-phase model has 3 linear parameters (logA plus
two slopes; tau is selected, not free-fit), so a residual degree of freedom needs
>= 4 observed timepoints with >= 2 on each side of the knot. Only such trajectories
are BREAKPOINT-TESTABLE; all others are reported with the constant fit and flagged.
(The 3-sampled-transfer amplicon designs therefore never switch -- correctly so.)

Output (config.OUT): segmented_growth_rates.csv  (one row per estimable trajectory)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C

# --- decision constants (documented above) --------------------------------
REL_RATE_CHANGE_MIN = 0.10   # rate must change by >10% to justify the switch
R_SCALE_FLOOR = 0.05         # per-cycle floor for the relative-change denominator
BIC_MARGIN = 6.0             # weighted-BIC improvement to switch (Kass-Raftery "strong")
MIN_OBS_FOR_BREAKPOINT = 4   # observed timepoints needed for a residual d.o.f.
MIN_SEG_OBS = 2              # observed timepoints needed on each side of the knot
_EPS = 1e-12


def wls(X, y, w):
    """Weighted least squares y ~ X. Returns (coef, wrss, r2_weighted).

    wrss is the weighted residual sum of squares; r2 is the weighted coefficient
    of determination. Both use the SAME weights/points for every model on a given
    trajectory, so BIC differences between models are comparable.
    """
    X = np.asarray(X, float); y = np.asarray(y, float); w = np.asarray(w, float)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    resid = y - X @ coef
    wrss = float(np.sum(w * resid ** 2))
    W = float(np.sum(w))
    ybar = float(np.sum(w * y) / W) if W > 0 else np.nan
    wtss = float(np.sum(w * (y - ybar) ** 2))
    r2 = (1.0 - wrss / wtss) if wtss > 0 else np.nan
    return coef, wrss, r2


def bic(wrss, n, k):
    """Small-sample weighted BIC (Gaussian quasi-likelihood on weighted residuals).
    Used only for RELATIVE comparison of the constant vs two-phase model on one
    trajectory (identical n and weights); lower is better."""
    return n * np.log(max(wrss, _EPS) / n) + k * np.log(n)


def _hinge(x, knot):
    return np.maximum(0.0, np.asarray(x, float) - knot)


def fit_two_phase(x, y, w, knot):
    """Continuous hinge fit at a fixed knot. Returns dict with intercept, r_init,
    r_final, wrss, r2."""
    X = np.column_stack([np.ones_like(x, float), np.asarray(x, float), _hinge(x, knot)])
    coef, wrss, r2 = wls(X, y, w)
    b0, b1, b2 = coef
    return dict(b0=b0, r_init=b1, r_final=b1 + b2, wrss=wrss, r2=r2, coef=coef)


def fit_constant(x, y, w):
    X = np.column_stack([np.ones_like(x, float), np.asarray(x, float)])
    coef, wrss, r2 = wls(X, y, w)
    return dict(b0=coef[0], slope=coef[1], wrss=wrss, r2=r2, coef=coef)


def _pred_init_clr(model_kind, cyc_fit, x0):
    """Fitted CLR at the first sampled transfer t0 = 'initial abundance' (logA)."""
    if model_kind == "two_phase":
        b0, b1 = cyc_fit["coef"][0], cyc_fit["coef"][1]  # x0 <= knot -> hinge term 0
        return b0 + b1 * x0
    return cyc_fit["coef"][0] + cyc_fit["coef"][1] * x0


def fit_one_sample(sub, axis):
    transfers = list(C.FOUR_TRANSFER_SET)
    interior = transfers[1:-1]                      # candidate inter-sample breakpoints
    counts = (sub.pivot_table(index="Candidate", columns="Transfer",
                              values="Count", aggfunc="sum", fill_value=0)
              .reindex(columns=transfers, fill_value=0))
    clr = C.clr_matrix(counts)

    x_cycle = np.array(transfers, float)
    cumgen = {t: float(axis.loc[t, "cumgen"]) for t in transfers}
    x_gen = np.array([cumgen[t] for t in transfers], float)
    depth = counts.sum(axis=0).to_numpy(float)
    x0 = x_cycle[0]
    n = len(transfers)
    meta = sub.drop_duplicates("Candidate").set_index("Candidate")

    rows = []
    for cand in counts.index:
        cvec = counts.loc[cand].to_numpy(float)
        yvec = clr.loc[cand].to_numpy(float)
        w = cvec + C.PSEUDOCOUNT
        obs = cvec > 0
        n_obs = int(obs.sum())
        fvec = np.divide(cvec, depth, out=np.zeros_like(cvec), where=depth > 0)

        # ---- constant (single-slope) reference on the per-cycle axis ----
        c_cyc = fit_constant(x_cycle, yvec, w)
        bic_const = bic(c_cyc["wrss"], n, 2)

        # ---- two-phase candidates: one continuous hinge per feasible knot.
        #      ONLY INCREASING-rate changes are permitted (r_final > r_init): a
        #      growth-rate DECREASE is not an allowed model, so a variant whose data
        #      want a decline falls back to the constant fit. `best_raw` tracks the
        #      unconstrained best fit only to populate `unconstrained_direction`, so
        #      the effect of the increasing-only rule stays auditable. ----
        feasible = []
        for kt in interior:
            left = int(np.sum(obs & (x_cycle <= kt)))
            right = int(np.sum(obs & (x_cycle >= kt)))
            if left >= MIN_SEG_OBS and right >= MIN_SEG_OBS:
                feasible.append(kt)
        testable = (n_obs >= MIN_OBS_FOR_BREAKPOINT) and (len(feasible) > 0)

        best = None          # best-fitting PERMITTED (increasing) two-phase model
        best_raw = None      # best-fitting two-phase model, any direction (diagnostic)
        if testable:
            for kt in feasible:
                tp = fit_two_phase(x_cycle, yvec, w, kt); tp["knot"] = kt
                if (best_raw is None) or (tp["wrss"] < best_raw["wrss"]):
                    best_raw = tp
                if (tp["r_final"] > tp["r_init"]) and ((best is None) or (tp["wrss"] < best["wrss"])):
                    best = tp
        bic_two = bic(best["wrss"], n, 3) if best is not None else np.nan
        delta_bic = (bic_const - bic_two) if best is not None else np.nan
        unconstrained_direction = ("none" if best_raw is None else
                                   "accelerating" if best_raw["r_final"] > best_raw["r_init"] else
                                   "decelerating")

        # ---- decision: switch only if an increasing fit improves AND grows >10% ----
        if best is not None:
            rel_change = (best["r_final"] - best["r_init"]) / max(abs(best["r_init"]), R_SCALE_FLOOR)
            abs_change = best["r_final"] - best["r_init"]        # > 0 by construction
        else:
            rel_change = np.nan; abs_change = np.nan
        switch = bool(testable and best is not None
                      and (delta_bic > BIC_MARGIN)
                      and (rel_change > REL_RATE_CHANGE_MIN))

        if switch:
            model_kind = "two_phase"; knot = best["knot"]
            r_init_c, r_fin_c = best["r_init"], best["r_final"]
            r2_sel = best["r2"]; cyc_fit = best
            # per-generation slopes: same knot, refit on the cumgen axis
            g = fit_two_phase(x_gen, yvec, w, cumgen[knot])
            r_init_g, r_fin_g = g["r_init"], g["r_final"]
        else:
            model_kind = "constant"; knot = np.nan
            r_init_c = r_fin_c = c_cyc["slope"]
            r2_sel = c_cyc["r2"]; cyc_fit = c_cyc
            g = fit_constant(x_gen, yvec, w)
            r_init_g = r_fin_g = g["slope"]

        logA_init = _pred_init_clr(model_kind, cyc_fit, x0)
        direction = "accelerating" if model_kind == "two_phase" else "constant"

        rows.append(dict(
            Sample=sub["Sample"].iloc[0],
            DNA_construct=meta.loc[cand, "DNA_construct"],
            Replicate=meta.loc[cand, "Replicate"],
            Candidate=cand,
            verA=meta.loc[cand, "verA"], verB=meta.loc[cand, "verB"],
            n_obs_timepoints=n_obs,
            transfers_observed=",".join(str(t) for t, c in zip(transfers, cvec) if c > 0),
            total_reads=int(cvec.sum()),
            # --- the four fitted parameters ---
            logA_init=round(float(logA_init), 4),
            r_init_per_cycle=round(float(r_init_c), 4),
            r_final_per_cycle=round(float(r_fin_c), 4),
            breakpoint_transfer=(int(knot) if model_kind == "two_phase" else np.nan),
            # --- other time axes (of the SELECTED model) ---
            r_init_per_generation=round(float(r_init_g), 4),
            r_final_per_generation=round(float(r_fin_g), 4),
            r_init_per_hour_measured=round(float(r_init_c / C.HOURS_PER_TRANSFER_MEASURED), 5),
            r_final_per_hour_measured=round(float(r_fin_c / C.HOURS_PER_TRANSFER_MEASURED), 5),
            # --- decision + diagnostics ---
            model_selected=model_kind,
            direction=direction,
            unconstrained_direction=unconstrained_direction,
            breakpoint_testable=bool(testable),
            rel_rate_change=(round(float(rel_change), 4) if np.isfinite(rel_change) else np.nan),
            abs_rate_change=(round(float(abs_change), 4) if np.isfinite(abs_change) else np.nan),
            delta_bic=(round(float(delta_bic), 3) if np.isfinite(delta_bic) else np.nan),
            r2_selected=(round(float(r2_sel), 4) if np.isfinite(r2_sel) else np.nan),
            r2_constant=(round(float(c_cyc["r2"]), 4) if np.isfinite(c_cyc["r2"]) else np.nan),
            r2_two_phase=(round(float(best["r2"]), 4) if best is not None and np.isfinite(best["r2"]) else np.nan),
            wrss_constant=round(float(c_cyc["wrss"]), 5),
            wrss_two_phase=(round(float(best["wrss"]), 5) if best is not None else np.nan),
            s_constant_per_cycle=round(float(c_cyc["slope"]), 4),
            freq_init=round(float(fvec[0]), 5),
            freq_final=round(float(fvec[-1]), 5),
            estimable=(n_obs >= C.MIN_OBS_TIMEPOINTS),
        ))
    return rows


def make_figure(res=None, d4=None):
    """Broken-stick fits for the strongest rate switches: observed CLR points, the
    constant single-slope fit, and the selected two-phase fit with its breakpoint."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if res is None:
        res = pd.read_csv(C.OUT / "segmented_growth_rates.csv")
    if d4 is None:
        d4 = pd.read_csv(C.INTER / "barcode_4transfer_long.csv")
    transfers = list(C.FOUR_TRANSFER_SET)
    x0, xN = transfers[0], transfers[-1]
    xx = np.linspace(x0, xN, 200)
    two = res[res.model_selected == "two_phase"].sort_values("delta_bic", ascending=False).head(6)
    if two.empty:
        print("[s14] no two-phase switches to plot; figure skipped."); return
    ncol = 3; nrow = int(np.ceil(len(two) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (_, row) in enumerate(two.iterrows()):
        ax = axes.flat[i]; ax.axis("on")
        sub = d4[d4.Sample == row.Sample]
        counts = (sub.pivot_table(index="Candidate", columns="Transfer",
                                  values="Count", aggfunc="sum", fill_value=0)
                  .reindex(columns=transfers, fill_value=0))
        clr = C.clr_matrix(counts)
        y = clr.loc[row.Candidate].to_numpy(float)
        cvec = counts.loc[row.Candidate].to_numpy(float)
        obs = cvec > 0
        ax.scatter(np.array(transfers)[obs], y[obs], s=45, color="#222", zorder=5, label="observed CLR")
        ax.scatter(np.array(transfers)[~obs], y[~obs], s=30, facecolors="none",
                   edgecolors="#bbb", zorder=4, label="below detection")
        # constant fit
        c = fit_constant(np.array(transfers, float), y, cvec + C.PSEUDOCOUNT)
        ax.plot(xx, c["coef"][0] + c["coef"][1] * xx, color="#1f77b4", lw=1.4, ls="--",
                label=f"constant s={c['slope']:+.2f}")
        # two-phase fit
        kt = float(row.breakpoint_transfer)
        yy = row.logA_init + row.r_init_per_cycle * (xx - x0) \
            + (row.r_final_per_cycle - row.r_init_per_cycle) * np.maximum(0.0, xx - kt)
        ax.plot(xx, yy, color="#d62728", lw=2.0,
                label=f"two-phase {row.r_init_per_cycle:+.2f}→{row.r_final_per_cycle:+.2f}")
        ax.axvline(kt, color="#d62728", ls=":", lw=1.2)
        ax.set_title(f"{row.Candidate}  ({row.direction})\n"
                     f"τ=T{int(kt)}  ΔBIC={row.delta_bic:.0f}  Δ={row.rel_rate_change*100:.0f}%",
                     fontsize=8)
        ax.set_xlabel("Transfer"); ax.set_ylabel("CLR (log rel. abundance)")
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("s14 segmented growth fits — strongest INCREASING-rate switches "
                 "(decreases not permitted)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(C.FIG / "segmented_growth_examples.png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"[s14] wrote figures/segmented_growth_examples.png ({len(two)} panels)")


def main():
    d4 = pd.read_csv(C.INTER / "barcode_4transfer_long.csv")
    axis = pd.read_csv(C.INTER / "od_time_axis.csv").set_index("transfer")

    rows = []
    for _, sub in d4.groupby("Sample"):
        rows.extend(fit_one_sample(sub, axis))
    res = pd.DataFrame(rows)
    res = res[res.estimable].copy()
    res = res.sort_values(["model_selected", "delta_bic"], ascending=[True, False])
    res.to_csv(C.OUT / "segmented_growth_rates.csv", index=False)

    n_all = len(res)
    n_test = int(res.breakpoint_testable.sum())
    two = res[res.model_selected == "two_phase"]
    tst = res[res.breakpoint_testable]
    held_decel = int(((tst.model_selected == "constant") &
                      (tst.unconstrained_direction == "decelerating")).sum())
    print(f"[s14] {n_all} estimable trajectories; {n_test} breakpoint-testable "
          f"(>= {MIN_OBS_FOR_BREAKPOINT} obs, >= {MIN_SEG_OBS}/segment).")
    print(f"[s14] INCREASING-only: two-phase (accelerating) selected for {len(two)}/{n_test} testable "
          f"(ΔBIC>{BIC_MARGIN:g} strong evidence AND +Δrate>{int(REL_RATE_CHANGE_MIN*100)}%); "
          f"{held_decel} held constant because their best fit was a DECREASE (not permitted).")
    if len(two):
        bp = two.breakpoint_transfer.value_counts().sort_index().to_dict()
        print(f"[s14] selected breakpoints (transfer -> #variants): "
              f"{ {int(k): int(v) for k, v in bp.items()} }")
        top = two.sort_values("delta_bic", ascending=False).head(4)
        print("[s14] strongest rate switches (by ΔBIC):")
        for _, x in top.iterrows():
            print(f"        {x.Sample.split('.')[-2]:>18s} {x.Candidate:10s} "
                  f"r {x.r_init_per_cycle:+.3f} -> {x.r_final_per_cycle:+.3f}/cyc "
                  f"@T{int(x.breakpoint_transfer)} (ΔBIC={x.delta_bic:.1f}, "
                  f"Δ={x.rel_rate_change*100:.0f}%)")
    print("[s14] wrote segmented_growth_rates.csv")
    return res


if __name__ == "__main__":
    res = main()
    try:
        make_figure(res)
    except Exception as e:
        print(f"[s14] figure skipped ({type(e).__name__}: {e})")
