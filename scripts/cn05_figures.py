"""
cn05 - Figures for the dgoA copy-number analysis.

  dgoA_growth_rates_forest.png  THE HEADLINE — every variant's amplification rate ± 95% CI,
                                grouped/coloured by condition context, with reliability
                                tiers (FDR-significant emphasised; low-leverage 3-transfer
                                lineages greyed because their CIs are not calibrated).
  dgoA_growth_examples.png      raw ln(copy-number) trajectories (replicate points shown)
                                with the per-transfer-mean 1-var line and the broken-stick.
  model_comparison.png          1-var vs 4-var: coverage, in-sample fit (broken-stick FORM
                                captures saturation), direction breakdown, out-of-sample LOTO.
  model_fit_error.png           in-sample vs out-of-sample fit error (FORM vs line).

The DEPLOYED increasing-only 4-var (fires on few) is kept distinct from the bidirectional
broken-stick FORM (captures saturation); the LOTO gain is labelled as modest/non-significant.
Palette: the data-viz skill's validated CVD-safe defaults (via s14c), by role.
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
import cn_common as CN
from s14_segmented_growth import fit_two_phase
from s14c_segmented_figures import (styled, SURFACE, INK, INK2, MUTED, BASELINE,
                                    BLUE, BLUE_LIGHT, BLUE_DARK, ORANGE, AQUA, VIOLET, GOOD)

CONTEXT_ORDER = ["TFMN1 single-KO", "TFMN1 double-KO", "TFMN4 exp1", "control (noDNA)"]
CTX_COLOR = {"TFMN1 single-KO": BLUE, "TFMN1 double-KO": VIOLET,
             "TFMN4 exp1": AQUA, "control (noDNA)": MUTED}
ONE = MUTED; FOUR = BLUE
UNREL = "#b9b8b2"


def _short(lineage):
    return str(lineage).replace("TFMN4.exp1.", "").replace("TFMN1.", "")


@styled
def fig_forest(rates):
    ctxs = [c for c in CONTEXT_ORDER if c in set(rates.context)]
    blocks, yticks, ylabels = [], [], []
    y = 0; ctx_spans = []
    for c in ctxs:
        g = rates[rates.context == c].sort_values("rate_per_transfer")
        start = y
        for _, r in g.iterrows():
            blocks.append((y, r)); yticks.append(y); ylabels.append(_short(r.lineage)); y += 1
        ctx_spans.append((c, start, y - 1)); y += 1

    h = max(6.0, 0.19 * len(blocks) + 1.5)
    fig, ax = plt.subplots(figsize=(10, h))
    ax.axvline(0, color=BASELINE, lw=1.2, ls="--", zorder=1)
    for yy, r in blocks:
        reliable = (r.tier == "reliable")
        sig = r.call in ("amplifying", "losing")
        col = CTX_COLOR.get(r.context, BLUE) if reliable else UNREL
        has_ci = np.isfinite(r.ci95_lo)
        xerr = [[r.rate_per_transfer - r.ci95_lo], [r.ci95_hi - r.rate_per_transfer]] if has_ci else None
        ax.errorbar(r.rate_per_transfer, yy, xerr=xerr, fmt="o",
                    ms=6 if sig else (4.5 if reliable else 3.3),
                    markerfacecolor=(col if reliable else "white"), markeredgecolor=col,
                    markeredgewidth=1.3 if sig else 1.0,
                    ecolor=col, elinewidth=1.6 if sig else (1.1 if reliable else 0.7),
                    capsize=2 if reliable else 0, alpha=1.0 if reliable else 0.7, zorder=4 if sig else 3)
        if sig:
            ax.annotate("★", (r.ci95_hi, yy), textcoords="offset points", xytext=(4, -3),
                        fontsize=8, color=col)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=5.5)
    ax.set_ylim(-1, y - 1)
    ax.set_xlabel("dgoA copy-number amplification rate  r = d ln(copy_number)/d transfer  (± 95% CI)")
    ax.set_title(f"Per-variant amplification rate ± error  ·  {len(blocks)} lineages "
                 f"(★ = FDR-significant)", fontsize=12)
    for c, s, e in ctx_spans:
        ax.text(1.005, (s + e) / 2, c, transform=ax.get_yaxis_transform(), rotation=270,
                va="center", ha="left", fontsize=7.5, color=CTX_COLOR.get(c, INK), fontweight="bold")
    handles = [Line2D([0], [0], marker="o", lw=0, color=CTX_COLOR[c], label=c) for c in ctxs]
    handles += [Line2D([0], [0], marker="*", lw=0, color=INK, label="FDR-significant (q<0.05)"),
                Line2D([0], [0], marker="o", lw=0, markerfacecolor="white", markeredgecolor=UNREL,
                       label="low-leverage (3 transfers) — CI not calibrated")]
    ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(CN.FIG / "dgoA_growth_rates_forest.png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"[cn05] wrote figures/dgoA_growth_rates_forest.png ({len(blocks)} variants)")


@styled
def fig_examples(long, rates, seg, bt):
    amp = rates[rates.tier == "reliable"].sort_values("rate_per_transfer", ascending=False).lineage.tolist()[:2]
    sat = seg[(seg.breakpoint_testable) & (seg.unconstrained_direction == "decelerating")] \
        .sort_values("r2_constant").lineage.tolist()[:2]
    acc = seg[seg.model_selected == "two_phase"].lineage.tolist()[:2]
    picks = [(l, "amplifier") for l in amp] + [(l, "saturating") for l in sat] + \
            [(l, "accelerating") for l in acc]
    picks = picks[:6]
    ncol = 3; nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.3 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (lin, kind) in enumerate(picks):
        a = axes.flat[i]; a.axis("on")
        raw = long[long.lineage == lin].sort_values("transfer")
        mean = bt[bt.lineage == lin].sort_values("transfer")
        xm = mean.transfer.to_numpy(float); ym = mean.ln_cn.to_numpy(float)
        xx = np.linspace(xm.min(), xm.max(), 100)
        f1 = CN.fit_1var(xm, ym)
        a.scatter(raw.transfer, raw.ln_cn, s=22, color=MUTED, alpha=0.6, zorder=4, label="measurements")
        a.scatter(xm, ym, s=40, color=INK, zorder=5, label="transfer mean")
        a.plot(xx, f1["intercept"] + f1["rate"] * xx, color=ONE, lw=2.0, ls="--",
               label=f"1-var: r={f1['rate']:+.3f}" + (f"±{f1['se']:.3f}" if np.isfinite(f1['se']) else ""),
               zorder=3)
        if kind in ("saturating", "accelerating"):
            s = CN.fit_segmented(xm, ym); kt = s["knot_unconstrained"]
            if np.isfinite(kt):
                tp = fit_two_phase(xm, ym, np.ones_like(xm), kt)
                yy = tp["coef"][0] + tp["coef"][1] * xx + tp["coef"][2] * np.maximum(0.0, xx - kt)
                a.plot(xx, yy, color=FOUR, lw=2.2,
                       label=f"broken-stick: {tp['r_init']:+.3f}→{tp['r_final']:+.3f}", zorder=6)
                a.axvline(kt, color=FOUR, ls=":", lw=1.0)
        a.set_title(f"{_short(lin)}  ({kind})", fontsize=8.5)
        a.set_xlabel("transfer"); a.set_ylabel("ln(copy number)")
        a.legend(fontsize=6, loc="best", framealpha=0.9)
    fig.suptitle("dgoA copy-number trajectories — data behind the rates", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(CN.FIG / "dgoA_growth_examples.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[cn05] wrote figures/dgoA_growth_examples.png ({len(picks)} panels)")


@styled
def fig_comparison(bt, idx):
    n_est = int(idx.estimable.sum()); n_test = len(bt)
    ad = bt[bt.deployed_model == "two_phase"]; kept = bt[bt.deployed_model == "constant"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

    a = ax[0, 0]; a.set_axisbelow(True)
    segs = [("identical — not testable (<4 transfers)", n_est - n_test, "#d7d6cf"),
            ("identical — testable, 1-var kept", len(kept), BLUE_LIGHT),
            ("deployed 4-var refines", len(ad), FOUR)]
    left = 0
    for name, val, col in segs:
        a.barh(0, val, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.5)
        if val > 1:
            a.text(left + val / 2, 0, f"{val}", ha="center", va="center",
                   color="white" if col != BLUE_LIGHT else INK, fontsize=11, fontweight="bold")
        left += val
    a.set_ylim(-0.6, 0.6); a.set_yticks([]); a.set_xlabel("# estimable lineages")
    a.set_title(f"(a) Deployed 4-var equals 1-var on {n_est - len(ad)}/{n_est} — refines {len(ad)}")
    a.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in segs], [s for s, _, _ in segs],
             fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.5), framealpha=0.9)
    a.grid(axis="y", visible=False)

    a = ax[0, 1]
    a.plot([0, 1], [0, 1], color=BASELINE, lw=1.2, ls="--")
    a.scatter(bt.r2_1var, bt.r2_4var_form, s=44, color=FOUR, zorder=3)
    a.set_xlim(-0.02, 1.03); a.set_ylim(-0.02, 1.03); a.set_aspect("equal")
    a.set_xlabel("1-var (straight line) R²"); a.set_ylabel("broken-stick FORM R²")
    a.set_title("(b) The straight line misses the saturation")
    a.text(0.03, 0.97, f"median R² {bt.r2_1var.median():.2f} → {bt.r2_4var_form.median():.2f}",
           transform=a.transAxes, fontsize=8, va="top", color=INK2)

    a = ax[1, 0]; a.set_axisbelow(True)
    dc = bt.unconstrained_direction.value_counts()
    vals = [int(dc.get(k, 0)) for k in ["decelerating", "accelerating", "none"]]
    bars = a.bar(["decelerating\n(saturation)", "accelerating", "flat"], vals,
                 color=[ORANGE, FOUR, MUTED], edgecolor=SURFACE, linewidth=2, width=0.6)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v + 0.2, str(v), ha="center", va="bottom",
               fontsize=11, fontweight="bold", color=INK)
    a.set_ylabel("# testable lineages"); a.set_ylim(0, max(vals) * 1.25)
    a.set_title("(c) Copy number mostly saturates")
    a.grid(axis="x", visible=False)

    a = ax[1, 1]
    g = bt.dropna(subset=["loto_rmse_1var", "loto_rmse_form"])
    lim = float(max(g.loto_rmse_1var.max(), g.loto_rmse_form.max())) * 1.08
    a.plot([0, lim], [0, lim], color=BASELINE, lw=1.2, ls="--")
    a.scatter(g.loto_rmse_1var, g.loto_rmse_form, s=44, color=FOUR, zorder=3)
    nb = int((g.loto_rmse_form < g.loto_rmse_1var).sum())
    a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_aspect("equal")
    a.set_xlabel("1-var LOTO RMSE"); a.set_ylabel("broken-stick FORM LOTO RMSE")
    a.set_title("(d) Out-of-sample: a modest broken-stick edge")
    a.text(0.96, 0.06, f"FORM better on {nb}/{len(g)}\nmedian {g.loto_rmse_1var.median():.2f} → "
           f"{g.loto_rmse_form.median():.2f} (not significant)", transform=a.transAxes,
           ha="right", va="bottom", fontsize=7.5, color=INK2)

    fig.suptitle("1-variable vs 4-variable growth model — dgoA copy number", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(CN.FIG / "model_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[cn05] wrote figures/model_comparison.png")


@styled
def fig_fit_error(bt):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.7))
    a = ax[0]
    g = bt.dropna(subset=["rmse_form_insample"])
    lim = float(max(g.rmse_1var.max(), g.rmse_form_insample.max())) * 1.1
    a.plot([0, lim], [0, lim], color=BASELINE, lw=1.2, ls="--")
    a.scatter(g.rmse_1var, g.rmse_form_insample, s=44, color=FOUR, zorder=3)
    a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_aspect("equal")
    a.set_xlabel("1-var ln(CN) RMSE"); a.set_ylabel("broken-stick ln(CN) RMSE")
    a.set_title("(a) In-sample error (broken-stick form)")
    a.text(0.96, 0.06, f"median {g.rmse_1var.median():.2f} → {g.rmse_form_insample.median():.2f}",
           transform=a.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
    a = ax[1]
    g = bt.dropna(subset=["loto_rmse_1var", "loto_rmse_form"])
    lim = float(max(g.loto_rmse_1var.max(), g.loto_rmse_form.max())) * 1.1
    a.plot([0, lim], [0, lim], color=BASELINE, lw=1.2, ls="--")
    a.scatter(g.loto_rmse_1var, g.loto_rmse_form, s=44, color=FOUR, zorder=3)
    nb = int((g.loto_rmse_form < g.loto_rmse_1var).sum())
    a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_aspect("equal")
    a.set_xlabel("1-var LOTO RMSE"); a.set_ylabel("broken-stick LOTO RMSE")
    a.set_title(f"(b) Out-of-sample ({nb}/{len(g)} below line, modest)")
    a.text(0.96, 0.06, f"median {g.loto_rmse_1var.median():.2f} → {g.loto_rmse_form.median():.2f}",
           transform=a.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
    a = ax[2]
    gi = bt.dropna(subset=["rmse_form_insample"]); gl = bt.dropna(subset=["loto_rmse_1var", "loto_rmse_form"])
    v1 = [gi.rmse_1var.median(), gl.loto_rmse_1var.median()]
    v4 = [gi.rmse_form_insample.median(), gl.loto_rmse_form.median()]
    xt = np.arange(2); wd = 0.38
    a.bar(xt - wd / 2, v1, wd, color=ONE, edgecolor=SURFACE, linewidth=1.5, label="1-var line")
    a.bar(xt + wd / 2, v4, wd, color=FOUR, edgecolor=SURFACE, linewidth=1.5, label="broken-stick")
    for xi, (b1, b4) in enumerate(zip(v1, v4)):
        a.text(xi - wd / 2, b1 + 0.004, f"{b1:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        a.text(xi + wd / 2, b4 + 0.004, f"{b4:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    a.set_xticks(xt); a.set_xticklabels(["in-sample", "out-of-sample\n(LOTO)"])
    a.set_ylabel("median ln(CN) RMSE"); a.set_ylim(0, max(v1 + v4) * 1.25)
    a.set_title("(c) In-sample gain is large; out-of-sample modest")
    a.legend(fontsize=8, loc="upper left", framealpha=0.9); a.grid(axis="x", visible=False)
    fig.suptitle("Fit error: 1-variable line vs 4-variable broken-stick — dgoA copy number", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(CN.FIG / "model_fit_error.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[cn05] wrote figures/model_fit_error.png")


def main():
    long = pd.read_csv(CN.INTER / "dgoA_copynumber_long.csv")
    bt_means = pd.read_csv(CN.INTER / "dgoA_bytransfer.csv")
    idx = pd.read_csv(CN.INTER / "dgoA_lineage_index.csv")
    rates = pd.read_csv(CN.OUT / "dgoA_growth_rates.csv")
    seg = pd.read_csv(CN.OUT / "dgoA_segmented.csv")
    bt = pd.read_csv(CN.OUT / "model_comparison_by_lineage.csv")
    fig_forest(rates)
    fig_examples(long, rates, seg, bt_means)
    fig_comparison(bt, idx)
    fig_fit_error(bt)
    print("[cn05] done.")


if __name__ == "__main__":
    main()
