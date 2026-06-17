"""
s06 - Diagnostic figures (also serve as sanity checks).

Outputs (outputs/figures/):
  richness_collapse.png      - distinct variants per transfer per Sample (sweep)
  sweep_trajectories.png     - frequency trajectories for one well-sampled Sample
  growth_matrix_heatmap.png  - selection coefficient heatmap (top variants)
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

SHORT = lambda s: s.replace("TFMN4.exp2.ACN3788.", "")


def main():
    d4 = pd.read_csv(C.INTER / "barcode_4transfer_long.csv")
    res = pd.read_csv(C.OUT / "selection_coefficients_long.csv")

    # 1) richness collapse
    rich = (d4[d4.Count > 0].groupby(["Sample", "Transfer"])["Candidate"].nunique()
            .unstack("Transfer"))
    fig, ax = plt.subplots(figsize=(7, 5))
    for s, row in rich.iterrows():
        ax.plot(row.index, row.values, marker="o", alpha=0.7, label=SHORT(s))
    ax.set(xlabel="Transfer", ylabel="# distinct variants (count>0)",
           title="Richness collapse (selective sweep)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(C.FIG / "richness_collapse.png", dpi=130); plt.close(fig)

    # 2) sweep trajectories for the richest Sample
    rich_sample = d4.groupby("Sample")["Candidate"].nunique().idxmax()
    sub = d4[d4.Sample == rich_sample]
    piv = sub.pivot_table(index="Candidate", columns="Transfer", values="freq", fill_value=0)
    top = piv.iloc[(piv.max(axis=1)).argsort()[::-1][:8].values]
    fig, ax = plt.subplots(figsize=(7, 5))
    for c, row in top.iterrows():
        ax.plot(row.index, row.values, marker="o", label=c)
    ax.set(xlabel="Transfer", ylabel="frequency",
           title=f"Sweep trajectories: {SHORT(rich_sample)}")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(C.FIG / "sweep_trajectories.png", dpi=130); plt.close(fig)

    # 3) growth-matrix heatmap (top variants by breadth)
    m = pd.read_csv(C.OUT / "growth_matrix_selection_per_cycle.csv", index_col=0)
    top = m.loc[m.notna().sum(axis=1).sort_values(ascending=False).index[:30]]
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(top.to_numpy(float), aspect="auto", cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(top.to_numpy(float))),
                   vmax=np.nanmax(np.abs(top.to_numpy(float))))
    ax.set_xticks(range(top.shape[1])); ax.set_xticklabels(top.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(top.shape[0])); ax.set_yticklabels(top.index, fontsize=6)
    ax.set_title("Selection coefficient (per cycle): top 30 variants")
    fig.colorbar(im, ax=ax, shrink=0.5, label="s per transfer-cycle")
    fig.tight_layout(); fig.savefig(C.FIG / "growth_matrix_heatmap.png", dpi=130); plt.close(fig)

    print(f"[s06] wrote 3 figures to {C.FIG}")


if __name__ == "__main__":
    main()
