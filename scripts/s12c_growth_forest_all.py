"""
s12c - Full forest plot of EVERY variant's global growth rate +/- bootstrap 95% CI.

This is the all-variants version of panel (b) of global_growth_fit.png (which only
showed the 8 slowest + 8 fastest has_slope_info variants). Here all variants in
global_variant_growth_rates.csv are plotted, sorted by rate, colored by the
`identifiability` flag, with prior-driven (no-slope-info) variants drawn as hollow
markers. Reads the s12/s12b output CSV (no refit needed).

Output: figures/global_growth_forest_all.png
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

COLOR = {"data-pinned": "#2ca02c", "mixed": "#ff7f0e", "anchor/prior-driven": "#d62728"}


def main():
    df = pd.read_csv(C.OUT / "global_variant_growth_rates.csv").copy()
    # community mu_bar (reliable T6/T13 median), for the reference line
    bulk = pd.read_csv(C.INTER / "od_bulk_rate.csv")
    mu_bar = float(bulk.loc[bulk.reliable & bulk.transfer.isin(list(C.RELIABLE_OD_TRANSFERS)), "mu_bulk_per_h"].median())

    df = df.sort_values("r_global_per_h").reset_index(drop=True)
    n = len(df)
    y = np.arange(n)
    lo = (df["r_global_per_h"] - df["ci_boot_lo"]).clip(lower=0)
    hi = (df["ci_boot_hi"] - df["r_global_per_h"]).clip(lower=0)
    colors = df["identifiability"].map(COLOR).fillna("#888888").values

    fig, ax = plt.subplots(figsize=(8.5, max(12, 0.085 * n)))
    # error bars colored individually; hollow markers for prior-driven (no slope info)
    for i in range(n):
        slope = bool(df["has_slope_info"].iloc[i])
        ax.errorbar(df["r_global_per_h"].iloc[i], y[i],
                    xerr=[[lo.iloc[i]], [hi.iloc[i]]],
                    fmt="o" if slope else "o", ms=3.2,
                    mfc=colors[i] if slope else "white",
                    mec=colors[i], ecolor=colors[i], elinewidth=0.6, capsize=1.2, zorder=2)
    ax.axvline(mu_bar, color="k", ls="--", lw=1.0, zorder=1,
               label=f"community μ_bar = {mu_bar:.2f}/h")
    ax.set_yticks(y)
    ax.set_yticklabels(df["Candidate"], fontsize=3)
    ax.set_ylim(-1, n)
    ax.set_xlabel("global static growth rate  r  (per hour)  ±  bootstrap 95% CI")
    ax.set_title(f"All {n} variants: global static growth rate ± error\n"
                 f"(sorted by rate; color = identifiability; hollow = no slope info / prior-driven)",
                 fontsize=11)
    # secondary top axis: doubling time
    def r2dt(r): return np.where(r > 0, np.log(2) / np.maximum(r, 1e-6), np.nan)
    secax = ax.secondary_xaxis("top", functions=(lambda r: r, lambda r: r))
    secax.set_xlabel("(x-axis is per-hour rate;  doubling time = ln2 / r,  e.g. 0.35/h ≈ 2.0 h)", fontsize=8)

    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor=c,
                     label=k, ms=6) for k, c in COLOR.items()]
    legend.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                         markeredgecolor="#555", label="no slope info (prior-driven)", ms=6))
    legend.append(Line2D([0], [0], color="k", ls="--", label=f"μ_bar = {mu_bar:.2f}/h"))
    ax.legend(handles=legend, fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    out = C.FIG / "global_growth_forest_all.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[s12c] wrote {out}  ({n} variants; "
          f"{int(df.has_slope_info.sum())} with slope info, {int((~df.has_slope_info).sum())} prior-driven)")
    print(f"[s12c] rate range [{df.r_global_per_h.min():.3f}, {df.r_global_per_h.max():.3f}]/h; "
          f"identifiability: {df.identifiability.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
