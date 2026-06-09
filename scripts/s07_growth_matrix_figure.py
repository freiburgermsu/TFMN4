"""
s07 - A single figure that captures growth_matrix_selection_per_cycle.csv.

Top panel: heatmap of the matrix (variants x Samples), cells = selection
coefficient s per transfer-cycle, diverging colormap centered at 0; grey =
variant absent / not estimable; gold stars mark each Sample's sweep winner
(most abundant variant at T13). Bottom panel: distribution of all estimated
cells with the neutral (s=0) line.

Output: outputs/figures/growth_matrix_overview.png
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

PREFIX = "TFMN4.exp2.ACN3788."


def short_label(c):
    """Compact Sample label for the x-axis, e.g. concX_largeLib_SpeI.1 -> X·SpeI.1."""
    c = c.replace("concX_", "X·").replace("concY_", "Y·")
    c = c.replace("largeLib_", "").replace("smallLib_", "sm·")
    c = c.replace("PCR_DpnI_cleanup", "Dpn").replace("SpeI_cleanup", "SpeIcl")
    return c


def main():
    m = pd.read_csv(C.OUT / "growth_matrix_selection_per_cycle.csv", index_col=0)
    res = pd.read_csv(C.OUT / "selection_coefficients_long.csv")

    # rows sorted by mean selection coefficient (winners toward the top)
    order = m.mean(axis=1, numeric_only=True).sort_values(ascending=False).index
    M = m.loc[order]
    vals = M.to_numpy(float)
    cols = list(M.columns)
    rows = list(M.index)
    vmax = float(np.nanmax(np.abs(vals)))

    # sweep winners -> (col, row) marker positions + y labels
    win = res[res.sweep_winner].copy()
    win["short"] = win["Sample"].str.replace(PREFIX, "", regex=False)
    win_pts, win_rows = [], {}
    for _, r in win.iterrows():
        if r["short"] in cols and r["Candidate"] in rows:
            ci, ri = cols.index(r["short"]), rows.index(r["Candidate"])
            win_pts.append((ci, ri))
            win_rows[ri] = r["Candidate"]

    fig = plt.figure(figsize=(10.5, 13.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[6.5, 1.0], hspace=0.34)

    # ---- heatmap -----------------------------------------------------------
    ax = fig.add_subplot(gs[0])
    cmap = plt.cm.coolwarm_r.copy()
    cmap.set_bad("#e8e8e8")  # NaN / absent
    im = ax.imshow(np.ma.masked_invalid(vals), aspect="auto", cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([short_label(c) for c in cols], rotation=45, ha="right", fontsize=8)
    # label only the sweep-winner rows to avoid 168-label clutter
    ax.set_yticks(sorted(win_rows))
    ax.set_yticklabels([win_rows[i] for i in sorted(win_rows)], fontsize=7)
    ax.set_ylabel(f"{len(rows)} variants (verA–verB), sorted by mean s  →  winners near top")
    for ci, ri in win_pts:
        ax.scatter(ci, ri, marker="*", s=90, facecolor="#FFD400",
                   edgecolor="k", linewidth=0.6, zorder=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("selection coefficient  s  (per transfer-cycle)\n>0 sweeps in   ·   <0 sweeps out")
    ax.set_title(
        "Per-variant relative growth  —  growth_matrix_selection_per_cycle\n"
        "grey = variant absent / not estimable   ·   ★ = sweep winner   ·   "
        "s is gauged within each Sample (compare signs/ranks, not absolute values across columns)",
        fontsize=10)

    # ---- distribution ------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    finite = vals[np.isfinite(vals)]
    ax2.hist(finite, bins=40, color="#8a8a8a", edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="k", lw=1.2)
    ax2.set_xlabel("s (per transfer-cycle)")
    ax2.set_ylabel("# cells")
    ax2.set_title(
        f"distribution of the {finite.size} estimated cells   "
        f"(range {finite.min():+.2f} to {finite.max():+.2f}; "
        f"{(finite > 0).mean():.0%} positive)", fontsize=9)

    fig.text(0.5, 0.004,
             "Sample labels:  X/Y = concX/concY  ·  SpeI / SpeIcl = SpeI(_cleanup)  ·  "
             "Dpn = PCR_DpnI_cleanup  ·  sm = smallLib  ·  trailing .n = replicate",
             ha="center", fontsize=7, color="#555555")

    out = C.FIG / "growth_matrix_overview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[s07] wrote {out}  ({len(rows)} variants x {len(cols)} samples, "
          f"{finite.size} cells, {len(win_pts)} winners marked)")


if __name__ == "__main__":
    main()
