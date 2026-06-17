"""
s13c - Figures for the joint WGS+amplicon 4MXB fusion and its error reduction.

Panels (outputs_joint_4MXB/figures/joint_error_reduction.png):
  (a) se_wgs_only vs se_joint scatter (below the diagonal = tightened), colored by tier
  (b) ESS-fold histogram for the both-deep tier (WGS-equivalent gain)
  (c) forest of the top shared variants: WGS-only CI vs joint CI side by side
  (d) B4 same-culture concordance: r(WGS-B4) vs r(amplicon-B4) with the gate correlation
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
import config as C
from s13_joint_4MXB import OUT

TIERCOLOR = {"both-deep": "#2ca02c", "amplicon-B4-only": "#1f77b4", "WGS-only": "#bbbbbb"}


def main():
    res = pd.read_csv(OUT / "joint_variant_growth_rates_4MXB.csv")
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))

    # (a) se_wgs_only vs se_joint
    a = ax[0, 0]
    for tier, g in res.groupby("evidence_tier"):
        a.scatter(g["se_wgs_only"], g["se_joint"], s=16, alpha=0.6,
                  color=TIERCOLOR.get(tier, "#888"), label=tier)
    lim = np.nanmax([res["se_wgs_only"].max(), res["se_joint"].max()])
    a.plot([0, lim], [0, lim], "k--", lw=0.8)
    a.set(xlabel="se (WGS-only)", ylabel="se (joint)",
          title="(a) per-variant SE: joint vs WGS-only\n(below diagonal = tightened by fusion)")
    a.legend(fontsize=7)

    # (b) ESS-fold for both-deep
    a = ax[0, 1]
    bd = res[res.evidence_tier == "both-deep"]["ESS_fold_vs_wgs"].dropna()
    a.hist(bd, bins=20, color="#2ca02c", alpha=0.8)
    a.axvline(1, color="k", ls="--", label="no gain")
    a.axvline(bd.median(), color="r", ls="-", label=f"median {bd.median():.1f}x")
    a.set(xlabel="ESS fold-gain vs WGS (= (se_wgs/se_joint)^2)", ylabel="# variants",
          title="(b) WGS-equivalent information gain (both-deep tier)")
    a.legend(fontsize=7)

    # (c) forest: top shared variants WGS-only vs joint
    a = ax[1, 0]
    sh = res[res.evidence_tier == "both-deep"].copy()
    sh = sh.reindex(sh["se_reduction_ratio"].sort_values(ascending=False).index).head(15)
    y = np.arange(len(sh))
    a.errorbar(sh["r_wgs_only"], y + 0.15, xerr=1.96 * sh["se_wgs_only"], fmt="o", ms=3,
               color="#bbbbbb", capsize=2, label="WGS-only ±95%")
    a.errorbar(sh["r_joint_per_h"], y - 0.15, xerr=1.96 * sh["se_joint"], fmt="o", ms=3,
               color="#2ca02c", capsize=2, label="joint ±95%")
    a.set_yticks(y); a.set_yticklabels(sh["Candidate"], fontsize=6)
    a.set(xlabel="r (per h)", title="(c) top-15 most-tightened shared variants")
    a.legend(fontsize=7)

    # (d) B4 same-culture concordance gate
    a = ax[1, 1]
    cpath = OUT / "b4_concordance.csv"
    if cpath.exists():
        c = pd.read_csv(cpath)
        a.scatter(c["r_WGS_B4"], c["r_ampl_B4"], s=18, alpha=0.6, color="#6a3d9a")
        lo = float(min(c["r_WGS_B4"].min(), c["r_ampl_B4"].min()))
        hi = float(max(c["r_WGS_B4"].max(), c["r_ampl_B4"].max()))
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        cc = np.corrcoef(c["r_WGS_B4"], c["r_ampl_B4"])[0, 1]
        a.set(xlabel="r (WGS-B4 only, shallow)", ylabel="r (amplicon-B4 only, deep)",
              title=f"(d) same-culture concordance GATE\n{len(c)} variants, corr={cc:.2f} "
                    f"(limited by shallow-WGS noise)")
    fig.suptitle("Joint WGS+amplicon 4MXB fusion: error reduction is modest & honest "
                 "(tightens shared/deep variants; adds B4-only coverage)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT / "figures" / "joint_error_reduction.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"[s13c] wrote {out}")


if __name__ == "__main__":
    main()
