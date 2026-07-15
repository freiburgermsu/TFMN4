"""
s14d figures - visual head-to-head of the 1-variable (constant) vs 4-variable
(segmented) model. Built by s14d_model_comparison.make(); split out for readability.

  model_comparison.png (2x2)  - the four fair axes: coverage/agreement, in-sample fit
                                (flagged as construction-favoured), BIC parsimony funnel,
                                and out-of-sample LOOCV under learned vs supplied elbow.
  model_comparison_blindspot.png (1x3) - what the single slope averages away: the
                                |single slope| vs |phase swing| scatter plus two
                                illustrative trajectories where the 1-var looks flat but
                                the 4-var reveals a strong dip-then-rise.

Palette: the data-viz skill's validated CVD-safe defaults (see s14c), by role.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import config as C
from s14c_segmented_figures import (styled, _short_sample, _clr_for_sample,
                                     SURFACE, INK, INK2, MUTED, GRID, BASELINE,
                                     BLUE, BLUE_LIGHT, BLUE_DARK, ORANGE, AQUA, VIOLET, GOOD)

ONE = MUTED        # the 1-variable model (context / de-emphasis)
FOUR = BLUE        # the 4-variable model (the subject under test)


@styled
def _fig_headtohead(bt, seg):
    ad = bt[bt.deployed_model == "two_phase"]
    kept = bt[bt.deployed_model == "constant"]
    n_test = len(bt); n_nontest = len(seg) - n_test
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

    # (a) coverage / agreement -------------------------------------------------
    a = ax[0, 0]; a.set_axisbelow(True)
    segs = [("identical — 4-var not testable", n_nontest, "#d7d6cf"),
            ("identical — testable, 1-var kept", len(kept), BLUE_LIGHT),
            ("4-var refines (deployed)", len(ad), FOUR)]
    left = 0; total = len(seg)
    for name, val, col in segs:
        a.barh(0, val, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.5)
        if val / total > 0.03:
            a.text(left + val / 2, 0, f"{val}", ha="center", va="center", color="white"
                   if col != BLUE_LIGHT else INK, fontsize=11, fontweight="bold")
        left += val
    a.set_ylim(-0.6, 0.6); a.set_yticks([]); a.set_xlabel("# estimable trajectories")
    a.set_title(f"(a) The 4-var equals the 1-var on {n_nontest + len(kept)}/{total} — it refines {len(ad)}")
    a.grid(axis="y", visible=False)
    a.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in segs],
             [s for s, _, _ in segs], fontsize=7.5, loc="lower center",
             bbox_to_anchor=(0.5, -0.5), ncol=1, framealpha=0.9)

    # (b) in-sample fit (deployed) --------------------------------------------
    a = ax[0, 1]
    a.plot([0, 1], [0, 1], color=BASELINE, lw=1.2, ls="--", zorder=1)
    a.scatter(kept.r2_1var, kept.r2_4var_deployed, s=34, facecolors="none",
              edgecolors=MUTED, linewidths=1.2, label=f"testable, 1-var kept ({len(kept)})", zorder=2)
    a.scatter(ad.r2_1var, ad.r2_4var_deployed, s=52, color=FOUR,
              label=f"4-var deployed ({len(ad)})", zorder=3)
    a.set_xlim(-0.02, 1.03); a.set_ylim(-0.02, 1.03); a.set_aspect("equal")
    a.set_xlabel("1-var weighted R²"); a.set_ylabel("4-var weighted R²")
    a.set_title("(b) In-sample fit — kept points sit on the identity line")
    a.text(0.03, 0.97, "raw fit favours the 4-var\nby construction (nested)",
           transform=a.transAxes, fontsize=7.5, color=INK2, va="top")
    a.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    # (c) parsimony (BIC) funnel ----------------------------------------------
    a = ax[1, 0]; a.set_axisbelow(True)
    stages = [("testable trajectories", n_test, BLUE_DARK),
              ("BIC prefers a broken-stick\n(any direction)", int(bt.bic_prefers_4var_capability.sum()), BLUE),
              ("…and increasing", int(bt.bic_prefers_4var_increasing.sum()), AQUA),
              ("…and ΔBIC>6 & +Δ>10%\n(deployed 4-var)", len(ad), FOUR)]
    y = np.arange(len(stages))[::-1]
    for yi, (name, val, col) in zip(y, stages):
        a.barh(yi, val, color=col, edgecolor=SURFACE, linewidth=2, height=0.66)
        a.text(val + 0.6, yi, f"{val}", va="center", fontsize=10, fontweight="bold", color=INK)
    a.set_yticks(y); a.set_yticklabels([s for s, _, _ in stages], fontsize=7.5)
    a.set_xlabel("# trajectories"); a.set_xlim(0, n_test * 1.12)
    a.set_title(f"(c) Parsimony funnel — the policy keeps the 1-var for {len(kept)}/{n_test}")
    a.grid(axis="y", visible=False)

    # (d) out-of-sample LOOCV: learned vs supplied elbow ----------------------
    a = ax[1, 1]
    def ratio(num, den):
        m = np.isfinite(num) & np.isfinite(den) & (den > 0)
        return np.median((num[m] / den[m]))
    r_free = ratio(ad.loocv_rmse_4var_free.to_numpy(float), ad.loocv_rmse_1var_free.to_numpy(float))
    r_fix = ratio(ad.loocv_rmse_4var_fixedelbow.to_numpy(float), ad.loocv_rmse_1var_fixedelbow.to_numpy(float))
    better_free = int(ad.oos_4var_better_free.sum())
    better_fix = int((pd.to_numeric(ad.loocv_rmse_4var_fixedelbow) <
                      pd.to_numeric(ad.loocv_rmse_1var_fixedelbow)).sum())
    labels = ["elbow LEARNED\nper fold (fair test)", "elbow SUPPLIED\n(shape only)"]
    vals = [r_free, r_fix]; cols = [ORANGE if v >= 1 else FOUR for v in vals]
    bars = a.bar(labels, vals, color=cols, edgecolor=SURFACE, linewidth=2, width=0.6)
    a.axhline(1.0, color=BASELINE, lw=1.4, ls="--")
    a.text(1.02, 1.0, "1-var parity", transform=a.get_yaxis_transform(), fontsize=7.5,
           color=INK2, va="center")
    for b, v, nb in zip(bars, vals, [better_free, better_fix]):
        a.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}\n({nb}/{len(ad)} 4-var better)",
               ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=INK)
    a.set_ylabel("LOOCV RMSE ratio  (4-var / 1-var)")
    a.set_ylim(0, max(1.15, max(vals) * 1.25))
    a.set_title("(d) Out-of-sample: the 4-var edge needs a known elbow")
    a.grid(axis="x", visible=False)

    fig.suptitle(f"1-variable vs 4-variable growth model — fair comparison  [{C.DATASET}]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(C.FIG / "model_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[s14d] wrote figures/model_comparison.png")


@styled
def _fig_blindspot(bt, seg, d4):
    ad = bt[bt.deployed_model == "two_phase"].copy()
    if ad.empty:
        return
    ad["hide_score"] = ad.abs_phase_range / (ad.s_constant_per_cycle.abs() + 0.05)
    picks = ad.sort_values("hide_score", ascending=False).head(2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.7))

    # (a) |single slope| vs |phase swing| -------------------------------------
    a = ax[0]
    lim = float(max(ad.abs_phase_range.max(), ad.s_constant_per_cycle.abs().max())) * 1.1
    a.plot([0, lim], [0, lim], color=BASELINE, lw=1.1, ls="--", label="equal (y=x)")
    a.plot([0, lim / 2], [0, lim], color=BASELINE, lw=1.0, ls=":", label="2× (y=2x)")
    a.scatter(ad.s_constant_per_cycle.abs(), ad.abs_phase_range, s=52, color=FOUR, zorder=3)
    a.set_xlim(0, lim); a.set_ylim(0, lim)
    a.set_xlabel("|single slope|  (what the 1-var reports)")
    a.set_ylabel("|r_final − r_init|  (the 4-var phase swing)")
    a.set_title("(a) The single slope understates the dynamics")
    ms, mp = ad.s_constant_per_cycle.abs().median(), ad.abs_phase_range.median()
    a.text(0.96, 0.06, f"median: 1-var |s|={ms:.2f}\nvs phase swing {mp:.2f}",
           transform=a.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
    a.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    # (b,c) illustrative overlays ---------------------------------------------
    x0, xN = C.FOUR_TRANSFER_SET[0], C.FOUR_TRANSFER_SET[-1]
    xx = np.linspace(x0, xN, 200); transfers = list(C.FOUR_TRANSFER_SET)
    for j, (_, row) in enumerate(picks.iterrows()):
        a = ax[1 + j]
        clr, counts = _clr_for_sample(d4, row.Sample, transfers)
        y = clr.loc[row.Candidate].to_numpy(float); cvec = counts.loc[row.Candidate].to_numpy(float)
        obs = cvec > 0; w = cvec + C.PSEUDOCOUNT
        ybar = np.sum(w * y) / np.sum(w); xbar = np.sum(w * np.array(transfers, float)) / np.sum(w)
        a.plot(xx, ybar + row.s_constant_per_cycle * (xx - xbar), color=ONE, lw=2.0, ls="--",
               label=f"1-var: s={row.s_constant_per_cycle:+.2f}", zorder=2)
        kt = float(row.breakpoint_transfer)
        yy = row.logA_init + row.r_init_per_cycle * (xx - x0) + \
            (row.r_final_per_cycle - row.r_init_per_cycle) * np.maximum(0.0, xx - kt)
        a.plot(xx, yy, color=FOUR, lw=2.4,
               label=f"4-var: {row.r_init_per_cycle:+.2f}→{row.r_final_per_cycle:+.2f} @T{int(kt)}", zorder=3)
        a.axvline(kt, color=FOUR, ls=":", lw=1.1)
        a.scatter(np.array(transfers)[obs], y[obs], s=42, color=INK, zorder=5)
        a.scatter(np.array(transfers)[~obs], y[~obs], s=28, facecolors="none",
                  edgecolors=BASELINE, zorder=4)
        a.set_xlabel("Transfer"); a.set_ylabel("CLR (log rel. abundance)")
        a.set_title(f"({'bc'[j]}) {row.Candidate} · {_short_sample(row.Sample)}", fontsize=9.5)
        a.legend(fontsize=7.5, loc="best", framealpha=0.9)

    fig.suptitle(f"What the single growth rate averages away  [{C.DATASET}]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(C.FIG / "model_comparison_blindspot.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[s14d] wrote figures/model_comparison_blindspot.png")


def make(bt, seg, d4):
    _fig_headtohead(bt, seg)
    _fig_blindspot(bt, seg, d4)


if __name__ == "__main__":
    import s14d_model_comparison as m
    m.main()
