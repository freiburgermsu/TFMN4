"""
s13d - Counts-only (WGS SeqCenter, NO amplicon) comparison for the joint 4MXB per-variant
growth rates.

The standard WGS pipeline (outputs/) already uses only counts, and the amplicon runs
(outputs_amplicon_*) use only amplicon; the ONE place the modeling *fuses* WGS counts with
the Plasmidsaurus amplicon is the joint 4MXB fit (s13). This script re-runs that exact
model with the amplicon cells DROPPED (WGS counts only), so the per-variant 4MXB rates
from counts alone can be compared to the joint (counts+amplicon) fusion — i.e. it shows
precisely what the amplicon data adds on top of the counts.

It writes standalone `_counts_only` files and does NOT overwrite the joint outputs.

Outputs (outputs_joint_4MXB/):
  joint_variant_growth_rates_4MXB_counts_only.csv  per-variant 4MXB rate + Laplace SE, WGS counts only
  counts_vs_amplicon_4MXB.csv                       counts-only vs joint vs amplicon-only, per variant
  figures/counts_vs_amplicon_4MXB.png               rates + SE comparison (what the amplicon adds)
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
import s13_joint_4MXB as s13
from s14c_segmented_figures import (styled, SURFACE, INK, INK2, MUTED, BASELINE,
                                    BLUE, ORANGE, AQUA)

OUT = s13.OUT


@styled
def make_figure(cmp):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.7))
    b4 = cmp[cmp.in_B4 == True]; nb4 = cmp[cmp.in_B4 != True]

    # (a) rates: counts-only vs joint (does adding amplicon shift the rate?)
    a = ax[0]
    lim = [min(cmp.r_counts_only_per_h.min(), cmp.r_joint_per_h.min()) - 0.02,
           max(cmp.r_counts_only_per_h.max(), cmp.r_joint_per_h.max()) + 0.02]
    a.plot(lim, lim, color=BASELINE, lw=1.2, ls="--")
    a.scatter(nb4.r_counts_only_per_h, nb4.r_joint_per_h, s=26, facecolors="none",
              edgecolors=MUTED, linewidths=1.1, label="not in amplicon B4", zorder=2)
    a.scatter(b4.r_counts_only_per_h, b4.r_joint_per_h, s=34, color=BLUE,
              label="in amplicon B4", zorder=3)
    a.set_xlim(lim); a.set_ylim(lim); a.set_aspect("equal")
    a.set_xlabel("rate: WGS counts only (per h)"); a.set_ylabel("rate: joint (counts+amplicon)")
    a.set_title("(a) Rates barely move")
    a.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    # (b) SE: counts-only vs joint (amplicon tightens the B4-present variants)
    a = ax[1]
    lo = max(1e-3, min(cmp.se_counts_only.min(), cmp.se_joint.min()) * 0.8)
    hi = max(cmp.se_counts_only.max(), cmp.se_joint.max()) * 1.15
    a.plot([lo, hi], [lo, hi], color=BASELINE, lw=1.2, ls="--")
    a.scatter(nb4.se_counts_only, nb4.se_joint, s=26, facecolors="none",
              edgecolors=MUTED, linewidths=1.1, label="not in B4 (unchanged)", zorder=2)
    a.scatter(b4.se_counts_only, b4.se_joint, s=34, color=BLUE, label="in B4 (tightened)", zorder=3)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
    a.set_xlabel("SE: WGS counts only (per h)"); a.set_ylabel("SE: joint (counts+amplicon)")
    a.set_title("(b) Below line: amplicon tightened SE")
    a.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    # (c) SE-reduction summary (only B4-present variants can gain from the amplicon depth)
    a = ax[2]
    red = (cmp.se_counts_only / cmp.se_joint)  # >1 = joint tighter than counts-only
    med_b4 = float((b4.se_counts_only / b4.se_joint).median()) if len(b4) else np.nan
    med_nb4 = float((nb4.se_counts_only / nb4.se_joint).median()) if len(nb4) else np.nan
    a.bar(["in B4\n(gains depth)", "not in B4\n(no amplicon data)"], [med_b4, med_nb4],
          color=[BLUE, MUTED], edgecolor=SURFACE, linewidth=2, width=0.6)
    for i, v in enumerate([med_b4, med_nb4]):
        a.text(i, v + 0.02, f"{v:.2f}×", ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
    a.axhline(1.0, color=BASELINE, lw=1.2, ls="--")
    a.set_ylabel("median SE ratio  (counts-only / joint)")
    a.set_ylim(0, max(1.2, (med_b4 or 1) * 1.25))
    a.set_title("(c) Amplicon only helps B4 variants")
    a.grid(axis="x", visible=False)

    fig.suptitle("Barcode 4MXB growth rates: WGS counts only vs joint (counts+amplicon)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "figures" / "counts_vs_amplicon_4MXB.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[s13d] wrote figures/counts_vs_amplicon_4MXB.png")


def main():
    df = s13.load_combined()
    print("[s13d] re-fitting the joint 4MXB model with amplicon cells DROPPED (WGS counts only) ...")
    Dw = s13.build(df, ["WGS"], use_b=False)
    Fw = s13.fit(Dw)
    meta = df.drop_duplicates("Candidate").set_index("Candidate")
    co = pd.DataFrame({
        "Candidate": Dw["variants"],
        "verA": [meta.loc[v, "verA"] for v in Dw["variants"]],
        "verB": [meta.loc[v, "verB"] for v in Dw["variants"]],
        "r_counts_only_per_h": np.round(Fw["r"], 4),
        "se_counts_only": np.round(Fw["se"], 4),
    }).sort_values("r_counts_only_per_h", ascending=False)
    co.to_csv(OUT / "joint_variant_growth_rates_4MXB_counts_only.csv", index=False)

    # compare to the joint (counts+amplicon) fit already saved by s13
    joint = pd.read_csv(OUT / "joint_variant_growth_rates_4MXB.csv")
    cmp = co.merge(joint[["Candidate", "r_joint_per_h", "se_joint", "r_ampl_only",
                          "se_ampl_only", "evidence_tier", "in_B4"]], on="Candidate", how="left")
    cmp["dr_joint_minus_counts"] = (cmp.r_joint_per_h - cmp.r_counts_only_per_h).round(4)
    cmp["se_ratio_counts_over_joint"] = (cmp.se_counts_only / cmp.se_joint).round(3)  # >1 = amplicon tightened
    cmp = cmp.sort_values("se_ratio_counts_over_joint", ascending=False)
    cmp.to_csv(OUT / "counts_vs_amplicon_4MXB.csv", index=False)

    b4 = cmp[cmp.in_B4 == True]
    print(f"[s13d] counts-only fit: {len(co)} variants; freq_R2={Fw['freq_r2']:.3f} "
          f"OD_rmse={Fw['od_rmse']:.3f}; median SE={co.se_counts_only.median():.4f}/h.")
    print(f"[s13d] vs joint: rates essentially unchanged (median |Δr|="
          f"{cmp.dr_joint_minus_counts.abs().median():.4f}/h); "
          f"amplicon tightens the {len(b4)} B4-present variants "
          f"(median SE ratio counts/joint = {(b4.se_counts_only/b4.se_joint).median():.2f}×), "
          f"and leaves the {len(cmp)-len(b4)} non-B4 variants unchanged.")
    print("[s13d] wrote joint_variant_growth_rates_4MXB_counts_only.csv, counts_vs_amplicon_4MXB.csv")
    try:
        make_figure(cmp)
    except Exception as e:
        print(f"[s13d] figure skipped ({type(e).__name__}: {e})")
    return co, cmp


if __name__ == "__main__":
    main()
