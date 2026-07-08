"""
s14c - Comprehensive figure suite communicating the s14 segmented (two-phase,
four-parameter, increasing-only) growth-rate results. Parallels the single-slope
assessment's figure set (s06/s07/s11/s12b/s12c) but for the breakpoint model.

Figures (config.FIG), for datasets with adopted switches (WGS):
  segmented_overview.png    model selection + the increasing-only constraint (2x2):
                            (a) trajectory classification, (b) r_init vs r_final with
                            the y=x permit line, (c) the two decision gates (ΔBIC &
                            +Δrate), (d) constant vs two-phase fit quality.
  segmented_parameters.png  the four fitted parameters (2x2): (a) r_init->r_final
                            dumbbell per adopted variant, (b) breakpoint counts,
                            (c) initial abundance vs acceleration, (d) normalized
                            fitted acceleration shapes.
  segmented_gallery.png     every adopted two-phase fit as small multiples.
  segmented_allele.png      per-verA / per-verB allele acceleration tallies.

For 3-transfer datasets (amplicon) nothing is breakpoint-testable, so only a reduced
segmented_overview.png is drawn: the classification plus the distribution of the
constant per-cycle slopes (there the four-parameter model reduces to the one-slope fit).

Colors are the data-viz skill's validated default palette (CVD-safe; see the skill's
references/palette.md) used in canonical roles: blue = the adopted/result series,
orange = the excluded (rate-decrease) series, gray = de-emphasis/context.
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
from s14b_segmented_summary import _classify, allele_effects_table

# --- validated palette (light surface), used by role ------------------------
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASELINE = "#c3c2b7"
BLUE = "#2a78d6"; BLUE_LIGHT = "#9ec5f4"; BLUE_DARK = "#256abf"
ORANGE = "#eb6834"; AQUA = "#1baf7a"; VIOLET = "#4a3aa7"; GOOD = "#0ca30c"
ADOPTED = BLUE; HELD = ORANGE; WEAK = MUTED
BP_COLOR = {4: AQUA, 6: VIOLET}


def _set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Segoe UI", "sans-serif"],
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.titleweight": "bold",
    })


def _short_sample(s):
    s = s.replace("TFMN4.exp2.ACN3788.", "")
    reps = [("concX", "cX"), ("concY", "cY"), ("largeLib", "lg"), ("smallLib", "sm"),
            ("PCR_DpnI", "PD"), ("_cleanup", "·cl"), ("SpeI", "Sp")]
    for a, b in reps:
        s = s.replace(a, b)
    return s


def _clr_for_sample(d4, sample, transfers):
    sub = d4[d4.Sample == sample]
    counts = (sub.pivot_table(index="Candidate", columns="Transfer", values="Count",
                              aggfunc="sum", fill_value=0).reindex(columns=transfers, fill_value=0))
    return C.clr_matrix(counts), counts


# ---------------------------------------------------------------------------
def fig_overview(r, cls):
    _set_style()
    transfers = list(C.FOUR_TRANSFER_SET)
    testable = cls["testable"]; adopted = cls["adopted"]; held = cls["held"]; weak = cls["weak"]
    has = len(testable) > 0

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

    # (a) classification stacked bars ---------------------------------------
    a = ax[0, 0]
    a.set_axisbelow(True)
    segs_est = [("breakpoint-testable", len(testable), BLUE_DARK),
                ("not testable (<4 obs / 3-transfer)", len(cls["nontest"]), "#d7d6cf")]
    segs_te = [("increasing two-phase (adopted)", len(adopted), ADOPTED),
               ("held constant — decrease not permitted", len(held), HELD),
               ("weak — kept constant", len(weak), WEAK)]
    for row, (label, segs, total) in enumerate([("Estimable", segs_est, len(r)),
                                                ("Testable", segs_te, len(testable))]):
        left = 0
        for name, val, col in segs:
            if val <= 0:
                continue
            a.barh(row, val, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.62)
            if val / max(total, 1) > 0.06:
                a.text(left + val / 2, row, f"{val}", ha="center", va="center",
                       color="white", fontsize=10, fontweight="bold")
            left += val
    a.set_yticks([0, 1]); a.set_yticklabels([f"Estimable\n(n={len(r)})", f"Testable\n(n={len(testable)})"])
    a.set_xlabel("# trajectories"); a.set_title("(a) How the trajectories were classified")
    a.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               [BLUE_DARK, "#d7d6cf", ADOPTED, HELD, WEAK]]
    labels = ["testable", "not testable", "adopted (increasing)", "held (decrease)", "weak-constant"]
    a.legend(handles, labels, fontsize=7, loc="upper right", framealpha=0.95, ncol=1,
             bbox_to_anchor=(1.0, 0.78))

    # (b) r_init vs r_final with the permit line ----------------------------
    a = ax[0, 1]
    if has:
        lim = np.nanmax(np.abs(np.r_[testable.r_init_per_cycle, testable.r_final_per_cycle])) * 1.1
        lim = float(lim) if np.isfinite(lim) and lim > 0 else 1.0
        a.fill_between([-lim, lim], [-lim, lim], lim, color=BLUE, alpha=0.05, zorder=0)
        a.plot([-lim, lim], [-lim, lim], color=BASELINE, lw=1.2, ls="--", zorder=1)
        a.text(0.97, 0.55, "shaded: r_final > r_init\n(increase — the only\nzone that can be adopted)",
               transform=a.transAxes, fontsize=7.5, color=INK2, va="top", ha="right")
        for sub, col, mk, fc, lab in [
            (held, HELD, "o", "none", "held (decrease)"),
            (weak, WEAK, "o", "none", "weak-constant"),
            (adopted, ADOPTED, "o", ADOPTED, "adopted (increasing)")]:
            if len(sub):
                a.scatter(sub.r_init_per_cycle, sub.r_final_per_cycle, s=42, marker=mk,
                          facecolors=fc, edgecolors=col, linewidths=1.4, label=lab, zorder=3)
        a.axhline(0, color=BASELINE, lw=0.8); a.axvline(0, color=BASELINE, lw=0.8)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
        a.legend(fontsize=7, loc="lower right", framealpha=0.9)
    a.set_xlabel("initial rate r_init (per cycle)"); a.set_ylabel("final rate r_final (per cycle)")
    a.set_title("(b) Only increasing-rate switches are adopted")

    # (c) the two decision gates -------------------------------------------
    a = ax[1, 0]
    g = testable[np.isfinite(testable.delta_bic)]
    if len(g):
        adop = g.model_selected == "two_phase"
        a.scatter(g.loc[~adop, "delta_bic"], g.loc[~adop, "rel_rate_change"] * 100,
                  s=36, facecolors="none", edgecolors=WEAK, linewidths=1.3,
                  label="not adopted", zorder=2)
        a.scatter(g.loc[adop, "delta_bic"], g.loc[adop, "rel_rate_change"] * 100,
                  s=48, color=ADOPTED, label="adopted", zorder=3)
        a.set_yscale("log")
        a.axvline(6, color=GOOD, lw=1.3, ls="--"); a.axhline(10, color=GOOD, lw=1.3, ls="--")
        a.text(6.4, a.get_ylim()[1] * 0.8, "ΔBIC>6", color=GOOD, fontsize=7.5, va="top")
        a.text(a.get_xlim()[1] * 0.98, 11, "+Δrate>10%", color=GOOD, fontsize=7.5, ha="right", va="bottom")
        n_noinc = len(testable) - len(g)
        a.set_title(f"(c) Both gates must pass  ({n_noinc} had no increasing candidate)")
        a.legend(fontsize=7, loc="upper left", framealpha=0.9)
    else:
        a.set_title("(c) Both gates must pass")
    a.set_xlabel("fit gate:  ΔBIC (constant − two-phase)")
    a.set_ylabel("size gate:  rate increase (%)")

    # (d) fit quality: constant vs two-phase --------------------------------
    a = ax[1, 1]
    if has:
        adop = testable.model_selected == "two_phase"
        a.plot([0, 1], [0, 1], color=BASELINE, lw=1.2, ls="--", zorder=1)
        a.scatter(testable.loc[~adop, "r2_constant"], testable.loc[~adop, "r2_two_phase"],
                  s=36, facecolors="none", edgecolors=WEAK, linewidths=1.3, label="not adopted", zorder=2)
        a.scatter(testable.loc[adop, "r2_constant"], testable.loc[adop, "r2_two_phase"],
                  s=48, color=ADOPTED, label="adopted", zorder=3)
        a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02); a.set_aspect("equal")
        a.legend(fontsize=7, loc="lower right", framealpha=0.9)
    a.set_xlabel("constant-fit weighted R²"); a.set_ylabel("two-phase weighted R²")
    a.set_title("(d) The second phase lifts the fit")

    fig.suptitle(f"Segmented growth model — selection & increasing-only constraint  "
                 f"[{C.DATASET}]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(C.FIG / "segmented_overview.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[s14c] wrote figures/segmented_overview.png")


def fig_overview_constant_only(r):
    """Reduced overview for datasets where nothing is breakpoint-testable."""
    _set_style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a = ax[0]
    a.barh([0], [len(r)], color=WEAK, edgecolor=SURFACE, linewidth=2, height=0.5)
    a.text(len(r) / 2, 0, f"{len(r)} constant", ha="center", va="center",
           color="white", fontsize=11, fontweight="bold")
    a.set_yticks([]); a.set_xlabel("# trajectories")
    a.set_title(f"(a) {C.DATASET}: {len(C.FOUR_TRANSFER_SET)} transfers → no breakpoint testable")
    a.grid(axis="y", visible=False)
    a.text(0.5, -0.35, "A two-phase model needs ≥4 timepoints (≥2 per segment);\n"
           "here the 4-parameter fit reduces to the single-slope fit.",
           transform=a.transAxes, ha="center", fontsize=8.5, color=INK2)
    a = ax[1]
    s = r.s_constant_per_cycle.dropna()
    a.hist(s, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    a.axvline(0, color=BASELINE, lw=1)
    a.set_xlabel("constant selection coefficient s (per cycle)")
    a.set_ylabel("# variants"); a.set_title("(b) Single-slope rate distribution")
    fig.suptitle(f"Segmented growth model — constant-only dataset  [{C.DATASET}]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(C.FIG / "segmented_overview.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[s14c] wrote figures/segmented_overview.png (constant-only)")


def fig_parameters(r, cls):
    _set_style()
    adopted = cls["adopted"].copy()
    if adopted.empty:
        return
    adopted = adopted.sort_values("r_final_per_cycle")
    transfers = list(C.FOUR_TRANSFER_SET); t0 = transfers[0]
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) dumbbell r_init -> r_final ---------------------------------------
    a = ax[0, 0]
    y = np.arange(len(adopted))
    a.hlines(y, adopted.r_init_per_cycle, adopted.r_final_per_cycle,
             color=BASELINE, lw=2.2, zorder=1)
    a.scatter(adopted.r_init_per_cycle, y, s=55, color=BLUE_LIGHT, edgecolors=BLUE_DARK,
              linewidths=1, zorder=3, label="r_init (initial rate)")
    a.scatter(adopted.r_final_per_cycle, y, s=55, color=BLUE_DARK, zorder=3,
              label="r_final (final rate)")
    a.axvline(0, color=BASELINE, lw=1, ls="--")
    a.set_yticks(y)
    a.set_yticklabels([f"{c}  ({_short_sample(s)})" for c, s in
                       zip(adopted.Candidate, adopted.Sample)], fontsize=7)
    a.set_xlabel("growth rate (per cycle)")
    a.set_title("(a) Initial → final rate per adopted variant")
    a.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    a.grid(axis="y", visible=False)

    # (b) breakpoint counts -------------------------------------------------
    a = ax[0, 1]
    bpc = adopted.breakpoint_transfer.value_counts().sort_index()
    xs = list(bpc.index)
    bars = a.bar([f"T{int(t)}" for t in xs], bpc.values,
                 color=[BP_COLOR.get(int(t), BLUE) for t in xs], edgecolor=SURFACE, linewidth=2, width=0.6)
    for b, v in zip(bars, bpc.values):
        a.text(b.get_x() + b.get_width() / 2, v, f"{int(v)}", ha="center", va="bottom",
               fontsize=11, fontweight="bold", color=INK)
    a.set_ylabel("# adopted variants"); a.set_xlabel("breakpoint transfer τ")
    a.set_title("(b) Where the rate switches (inter-sample point)")
    a.grid(axis="x", visible=False)
    a.set_ylim(0, max(bpc.values) * 1.2)

    # (c) initial abundance vs acceleration --------------------------------
    a = ax[1, 0]
    for t in sorted(adopted.breakpoint_transfer.unique()):
        s = adopted[adopted.breakpoint_transfer == t]
        a.scatter(s.logA_init, s.abs_rate_change, s=55, color=BP_COLOR.get(int(t), BLUE),
                  edgecolors="white", linewidths=0.6, label=f"τ = T{int(t)}", zorder=3)
    a.set_xlabel("initial abundance  logA_init  (CLR at first transfer)")
    a.set_ylabel("acceleration  r_final − r_init  (per cycle)")
    a.set_title("(c) Initial abundance vs acceleration")
    a.legend(fontsize=7.5, loc="best", framealpha=0.9)

    # (d) normalized fitted acceleration shapes ----------------------------
    a = ax[1, 1]
    xx = np.linspace(transfers[0], transfers[-1], 200)
    for _, row in adopted.iterrows():
        kt = float(row.breakpoint_transfer)
        yy = row.r_init_per_cycle * (xx - t0) + \
            (row.r_final_per_cycle - row.r_init_per_cycle) * np.maximum(0.0, xx - kt)
        a.plot(xx, yy, color=BP_COLOR.get(int(kt), BLUE), lw=1.3, alpha=0.7)
    a.axhline(0, color=BASELINE, lw=0.8)
    for t in sorted(adopted.breakpoint_transfer.unique()):
        a.axvline(t, color=BP_COLOR.get(int(t), BLUE), lw=1, ls=":")
    a.set_xlabel("Transfer"); a.set_ylabel("fitted CLR change from t₀")
    a.set_title("(d) Acceleration shapes (normalized to 0 at t₀)")
    handles = [Line2D([0], [0], color=BP_COLOR.get(int(t), BLUE), lw=2, label=f"τ = T{int(t)}")
               for t in sorted(adopted.breakpoint_transfer.unique())]
    a.legend(handles=handles, fontsize=7.5, loc="upper left", framealpha=0.9)

    fig.suptitle(f"Segmented growth model — the four fitted parameters  [{C.DATASET}]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(C.FIG / "segmented_parameters.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[s14c] wrote figures/segmented_parameters.png")


def fig_gallery(r, cls, d4):
    _set_style()
    adopted = cls["adopted"].sort_values("delta_bic", ascending=False)
    if adopted.empty:
        return
    transfers = list(C.FOUR_TRANSFER_SET); x0, xN = transfers[0], transfers[-1]
    xx = np.linspace(x0, xN, 200)
    n = len(adopted); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.9 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    # cache CLR per sample
    clr_cache = {}
    for i, (_, row) in enumerate(adopted.iterrows()):
        a = axes.flat[i]; a.axis("on")
        if row.Sample not in clr_cache:
            clr_cache[row.Sample] = _clr_for_sample(d4, row.Sample, transfers)
        clr, counts = clr_cache[row.Sample]
        y = clr.loc[row.Candidate].to_numpy(float)
        cvec = counts.loc[row.Candidate].to_numpy(float)
        obs = cvec > 0
        # constant single-slope reference line: slope s through the weighted centroid
        s = row.s_constant_per_cycle
        w = cvec + C.PSEUDOCOUNT
        ybar = np.sum(w * y) / np.sum(w); xbar = np.sum(w * np.array(transfers, float)) / np.sum(w)
        a.plot(xx, ybar + s * (xx - xbar), color=WEAK, lw=1.4, ls="--", zorder=2, label="constant")
        # two-phase fit
        kt = float(row.breakpoint_transfer)
        yy = row.logA_init + row.r_init_per_cycle * (xx - x0) + \
            (row.r_final_per_cycle - row.r_init_per_cycle) * np.maximum(0.0, xx - kt)
        a.plot(xx, yy, color=ADOPTED, lw=2.2, zorder=3, label="two-phase")
        a.axvline(kt, color=ADOPTED, ls=":", lw=1.1, zorder=1)
        a.scatter(np.array(transfers)[obs], y[obs], s=38, color=INK, zorder=5)
        a.scatter(np.array(transfers)[~obs], y[~obs], s=26, facecolors="none",
                  edgecolors=BASELINE, zorder=4)
        a.set_title(f"{row.Candidate} · {_short_sample(row.Sample)}\n"
                    f"τ=T{int(kt)}  {row.r_init_per_cycle:+.2f}→{row.r_final_per_cycle:+.2f}  "
                    f"ΔBIC={row.delta_bic:.0f}", fontsize=7.5)
        a.tick_params(labelsize=7)
        if i % ncol == 0:
            a.set_ylabel("CLR", fontsize=8)
        if i >= n - ncol:
            a.set_xlabel("Transfer", fontsize=8)
    handles = [Line2D([0], [0], color=INK, marker="o", lw=0, label="observed CLR"),
               Line2D([0], [0], color=WEAK, lw=1.6, ls="--", label="constant fit"),
               Line2D([0], [0], color=ADOPTED, lw=2.2, label="two-phase fit")]
    fig.legend(handles=handles, fontsize=8.5, loc="lower center", ncol=3, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Every adopted increasing two-phase fit  ({n})  [{C.DATASET}]", fontsize=13)
    fig.tight_layout(rect=[0.01, 0.03, 1, 0.96])
    fig.savefig(C.FIG / "segmented_gallery.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[s14c] wrote figures/segmented_gallery.png ({n} panels)")


def fig_allele(alle, cls):
    _set_style()
    if cls["adopted"].empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for j, level in enumerate(("verA", "verB")):
        a = ax[j]
        sub = alle[(alle.level == level) & (alle.n_accelerating > 0)].sort_values(
            "n_accelerating", ascending=True)
        if sub.empty:
            a.set_title(f"({'ab'[j]}) {level}: no adopted accelerations"); a.axis("off"); continue
        y = np.arange(len(sub))
        # emphasis: top allele(s) in blue, rest in de-emphasis gray
        mx = sub.n_accelerating.max()
        cols = [ADOPTED if v == mx else BLUE_LIGHT for v in sub.n_accelerating]
        a.barh(y, sub.n_accelerating, color=cols, edgecolor=SURFACE, linewidth=1.5, height=0.7)
        for yi, (_, x) in zip(y, sub.iterrows()):
            a.text(x.n_accelerating + 0.03, yi,
                   f"{int(x.n_accelerating)}/{int(x.n_testable)}  (μΔ={x.mean_acceleration:+.2f})",
                   va="center", fontsize=7.5, color=INK2)
        a.set_yticks(y); a.set_yticklabels(sub.allele, fontsize=8)
        a.set_xlabel("# adopted accelerating variants")
        a.set_title(f"({'ab'[j]}) {level} allele — acceleration tally")
        a.grid(axis="y", visible=False)
        a.set_xlim(0, mx * 1.35)
    fig.suptitle(f"Segmented growth model — acceleration by enzyme allele  [{C.DATASET}]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(C.FIG / "segmented_allele.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[s14c] wrote figures/segmented_allele.png")


def main():
    r = pd.read_csv(C.OUT / "segmented_growth_rates.csv")
    d4 = pd.read_csv(C.INTER / "barcode_4transfer_long.csv")
    cls = _classify(r)
    alle = allele_effects_table(r, cls)

    if cls["adopted"].empty:
        fig_overview_constant_only(r)
        print(f"[s14c] {C.DATASET}: 0 adopted switches → reduced (constant-only) overview.")
        return
    fig_overview(r, cls)
    fig_parameters(r, cls)
    fig_gallery(r, cls, d4)
    fig_allele(alle, cls)
    print(f"[s14c] {C.DATASET}: wrote 4 figures for {len(cls['adopted'])} adopted switches.")


if __name__ == "__main__":
    main()
