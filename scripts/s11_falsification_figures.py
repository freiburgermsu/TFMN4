"""
s11 - Falsification figures for the cross-sample / OD-anchor analysis.

A 2x2 panel (outputs/figures/od_anchor_falsification.png):
  (a) GO/NO-GO: per-sample gauge gamma_c vs OD bulk rate mu_bulk. Theory says that
      to anchor per-variant absolute rates these must be strongly NEGATIVELY
      correlated; we show they are not -> OD cannot supply per-sample absolute info.
  (b) mu_bulk vs transfer per sample + the across-sample spread at T6/T13 -> mu_bulk
      is one shared community constant plus noise, not a per-sample gauge.
  (c) Bridge graph: samples linked by shared variants (edge width ~ #shared);
      the lone degree-1 (weakly anchored) sample is highlighted.
  (d) Forest plot of the drift-confirmed allele effects (relative, per cycle) with a
      twin top axis showing the SAME effects on the OD clock -> they all compress
      onto the ~community mu_bulk band, i.e. absolute differences are tiny.
"""
from __future__ import annotations
import sys
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

SHORT = lambda s: s.replace("TFMN4.exp2.ACN3788.", "")


def main():
    gauge = pd.read_csv(C.OUT / "sample_gauge.csv")
    bulk = pd.read_csv(C.INTER / "od_bulk_rate.csv")
    res = pd.read_csv(C.OUT / "selection_coefficients_long.csv")
    allele = pd.read_csv(C.OUT / "allele_growth_advantage.csv")
    cond = pd.read_csv(C.INTER / "od_bulk_condition_summary.csv")
    tau = float(bulk.loc[bulk.reliable & bulk.transfer.isin([6, 13]), "tau_exp_h"].median())
    mu_lo, mu_hi = float(cond["median"].min()), float(cond["median"].max())
    mu_mid = float(np.median([mu_lo, mu_hi]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # ---- (a) go/no-go ------------------------------------------------------
    mub = (bulk[bulk.reliable & bulk.transfer.isin([6, 13])]
           .groupby("Sample")["mu_bulk_per_h"].mean())
    g = gauge.set_index("Sample")["gamma_c"]
    j = pd.concat([g, mub], axis=1).dropna()
    j.columns = ["gamma_c", "mu_bulk"]
    ax = axes[0, 0]
    ax.scatter(j["mu_bulk"], j["gamma_c"], s=55, color="#1f77b4", zorder=3)
    if len(j) >= 3:
        r, p = stats.pearsonr(j["mu_bulk"], j["gamma_c"])
        b = np.polyfit(j["mu_bulk"], j["gamma_c"], 1)
        xs = np.array([j["mu_bulk"].min(), j["mu_bulk"].max()])
        ax.plot(xs, b[0] * xs + b[1], "--", color="#888")
        ax.set_title(f"(a) GO/NO-GO test  —  FAILS\nPearson r = {r:+.2f} (p={p:.2f}); "
                     f"need strongly NEGATIVE to anchor per-variant rates", fontsize=10)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("OD bulk rate  μ_bulk  (per h, T6/T13)")
    ax.set_ylabel("per-sample gauge  γ_c  (per cycle)")

    # ---- (b) mu_bulk vs transfer ------------------------------------------
    ax = axes[0, 1]
    for s, gg in bulk[bulk.reliable].groupby("Sample"):
        ax.plot(gg["transfer"], gg["mu_bulk_per_h"], marker="o", alpha=0.6, lw=1)
    rel = bulk[bulk.reliable & bulk.transfer.isin([6, 13])]
    ax.axhspan(rel["mu_bulk_per_h"].mean() - rel["mu_bulk_per_h"].std(),
               rel["mu_bulk_per_h"].mean() + rel["mu_bulk_per_h"].std(),
               color="#cccccc", alpha=0.5, zorder=0,
               label=f"T6/T13 mean±SD = {rel['mu_bulk_per_h'].mean():.2f}±{rel['mu_bulk_per_h'].std():.2f}")
    ax.set_title("(b) μ_bulk vs transfer  —  one shared constant + noise", fontsize=10)
    ax.set_xlabel("transfer"); ax.set_ylabel("μ_bulk (per h)")
    ax.set_xticks([3, 4, 6, 13]); ax.legend(fontsize=8)

    # ---- (c) bridge graph --------------------------------------------------
    ax = axes[1, 0]
    est = res[res.estimable]
    vs = est.groupby("Sample")["Candidate"].apply(set).to_dict()
    samples = sorted(vs)
    ang = {s: 2 * np.pi * k / len(samples) for k, s in enumerate(samples)}
    pos = {s: (np.cos(a), np.sin(a)) for s, a in ang.items()}
    deg = dict(zip(gauge["Sample"], gauge["bridge_degree"]))
    for a, b in itertools.combinations(samples, 2):
        n = len(vs[a] & vs[b])
        if n:
            (x0, y0), (x1, y1) = pos[a], pos[b]
            ax.plot([x0, x1], [y0, y1], color="#bbb", lw=0.4 + 0.35 * n, zorder=1)
    for s in samples:
        weak = deg.get(s, 9) <= 1
        ax.scatter(*pos[s], s=260, color="#d62728" if weak else "#2ca02c", zorder=3)
        ax.annotate(SHORT(s), pos[s], fontsize=6, ha="center", va="center", color="white")
    ax.set_title("(c) Bridge graph: 1 connected component\n(red = weakly anchored, degree 1)", fontsize=10)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.axis("off")

    # ---- (d) forest of robust allele effects + OD-clock twin axis ----------
    ax = axes[1, 1]
    rob = allele[allele.robust_selection].sort_values("marginal_s_per_cycle").reset_index(drop=True)
    y = np.arange(len(rob))
    ax.errorbar(rob["marginal_s_per_cycle"], y, xerr=rob["se"], fmt="o",
                color="#6a3d9a", capsize=2, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k} {a}" for k, a in zip(rob["allele_kind"], rob["allele"])], fontsize=7)
    ax.set_xlabel("relative selection coefficient  s  (per cycle)")
    ax.set_title("(d) Drift-confirmed allele effects (relative)\n"
                 "top axis: same effects on the OD clock → all near μ_bulk", fontsize=10)
    # twin top axis: OD/h = mu_mid + s/tau  (a linear remap of the bottom axis)
    def s2od(s): return mu_mid + s / tau
    def od2s(o): return (o - mu_mid) * tau
    secax = ax.secondary_xaxis("top", functions=(s2od, od2s))
    secax.set_xlabel("OD-clock rate R = μ_bulk + s/τ_exp  (per h)")
    ax.axvspan(od2s(mu_lo), od2s(mu_hi), color="#ffe08a", alpha=0.5, zorder=0)

    fig.suptitle("OD-anchor falsification: cross-sample integration is RELATIVE; "
                 "absolute OD/h is community-level only", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = C.FIG / "od_anchor_falsification.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"[s11] wrote {out}")


if __name__ == "__main__":
    main()
