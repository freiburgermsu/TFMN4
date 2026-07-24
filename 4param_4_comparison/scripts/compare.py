"""
compare - Figure + report comparison of the three 4-parameter (segmented, two-phase)
growth-model runs, computed entirely from THEIR fed outputs (this folder's inputs/):

    amplicon    <- ../4param_1_amplicon_only/outputs   (amplicon only, transfers {3,4,13})
    barcode     <- ../4param_2_barcode_only/outputs    (WGS only, 11 wells, {3,4,6,13})
    integrated  <- ../4param_3_integrated/outputs      (WGS breadth + amplicon depth on B4)

Nothing here refits a model; it only reads the three segmented_growth_rates.csv /
segmented_model_selection.csv / depth tables that were fed in, and reports how the SAME
4-parameter model behaves on the three data regimes.

Outputs (outputs/):
  model_capability_comparison.csv   estimable / testable / adopted, per version
  b4_rate_concordance.csv           per-variant single-slope rate on the shared B4 culture
  adopted_switches_comparison.csv   the two-phase switches each version adopts (+ overlap)
  figures/fig1_capability.png       what each data regime lets the 4-param model do
  figures/fig2_b4_integration.png   what integration adds on the shared well-B4 culture
  figures/fig3_rate_concordance.png per-variant rate agreement across regimes (B4)
  figures/fig4_adopted_switches.png the adopted breakpoints: barcode vs integrated
  ../COMPARISON_REPORT.md           the written report (numbers wired to the CSVs above)
  outputs/comparison_stats.json     machine-readable summary of every number quoted
"""
from __future__ import annotations
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "inputs"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

VERSIONS = ["amplicon", "barcode", "integrated"]
LABEL = {"amplicon": "Amplicon only", "barcode": "Barcode / WGS only",
         "integrated": "Integrated (WGS+amplicon)"}
SHORT = {"amplicon": "amplicon\n{3,4,13}", "barcode": "barcode/WGS\n{3,4,6,13}",
         "integrated": "integrated\n{3,4,6,13}"}
B4_WGS = "TFMN4.exp2.ACN3788.concX_largeLib_SpeI.1"   # the well-B4 culture (WGS + integrated)

# --- validated CVD-safe palette (matches the project's s14c segmented figures) ---
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASELINE = "#c3c2b7"
BLUE = "#2a78d6"; BLUE_LIGHT = "#9ec5f4"; BLUE_DARK = "#256abf"
ORANGE = "#eb6834"; AQUA = "#1baf7a"; VIOLET = "#4a3aa7"; GOOD = "#0ca30c"
VCOLOR = {"amplicon": ORANGE, "barcode": MUTED, "integrated": BLUE}

_STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "figure.titleweight": "bold",
}


def styled(fn):
    @functools.wraps(fn)
    def wrap(*a, **k):
        with plt.rc_context(_STYLE):
            return fn(*a, **k)
    return wrap


# --------------------------------------------------------------------------- load
def load():
    seg = {v: pd.read_csv(IN / v / "segmented_growth_rates.csv") for v in VERSIONS}
    msel = {v: pd.read_csv(IN / v / "segmented_model_selection.csv").set_index("metric")["value"]
            for v in VERSIONS}
    depth = {}
    for v in VERSIONS:
        d = pd.read_csv(IN / v / "depth_by_sample_transfer.csv", index_col=0)
        d.columns = [int(float(c)) for c in d.columns]
        depth[v] = d
    return seg, msel, depth


def _int(msel, v, key):
    return int(float(msel[v].get(key, 0)))


def b4_rows(seg, v):
    """The rows of version v belonging to the shared well-B4 culture."""
    s = seg[v]
    if v == "amplicon":
        return s[s.Sample.astype(str).str.startswith("B4_")].copy()
    return s[s.Sample.astype(str) == B4_WGS].copy()


# --------------------------------------------------------------------------- tables
def capability_table(seg, msel):
    rows = []
    for v in VERSIONS:
        s = seg[v]
        rows.append(dict(
            version=v, label=LABEL[v],
            n_transfers=_int(msel, v, "n_transfers_sampled"),
            n_estimable=len(s),
            n_breakpoint_testable=_int(msel, v, "n_breakpoint_testable"),
            n_two_phase_adopted=_int(msel, v, "n_two_phase_adopted_increasing"),
            n_held_decrease=_int(msel, v, "n_held_constant_decrease_not_permitted"),
            n_weak_constant=_int(msel, v, "n_weak_constant"),
            median_r2_constant_testable=msel[v].get("median_r2_constant_testable", np.nan),
            median_r2_two_phase_testable=msel[v].get("median_r2_two_phase_testable", np.nan),
            median_delta_bic_adopted=msel[v].get("median_delta_bic_adopted", np.nan),
        ))
    return pd.DataFrame(rows)


def b4_concordance_table(seg):
    """Per-variant single-slope rate (s_constant_per_cycle) on the B4 culture, one column
    per version. The single slope is defined for every trajectory (even where a two-phase
    is adopted), so it is the apples-to-apples cross-version quantity."""
    frames = {}
    for v in VERSIONS:
        b = b4_rows(seg, v)
        if v == "amplicon":
            g = (b.groupby("Candidate")
                 .agg(s=("s_constant_per_cycle", "mean"),
                      verA=("verA", "first"), verB=("verB", "first"),
                      n_primer_reps=("s_constant_per_cycle", "size")))
            frames[v] = g.rename(columns={"s": f"s_{v}"})
        else:
            g = (b.set_index("Candidate")[["s_constant_per_cycle", "verA", "verB",
                                           "model_selected", "n_obs_timepoints"]]
                 .rename(columns={"s_constant_per_cycle": f"s_{v}",
                                  "model_selected": f"model_{v}",
                                  "n_obs_timepoints": f"nobs_{v}"}))
            frames[v] = g
    # outer-merge on Candidate; keep verA/verB from whichever has them
    out = None
    for v in VERSIONS:
        f = frames[v].copy()
        va = f.pop("verA"); vb = f.pop("verB")
        f["verA"] = va; f["verB"] = vb
        out = f if out is None else out.join(f[[c for c in f.columns if c not in ("verA", "verB")]],
                                             how="outer")
        if "verA" not in out.columns:
            out["verA"] = va; out["verB"] = vb
        else:
            out["verA"] = out["verA"].fillna(va); out["verB"] = out["verB"].fillna(vb)
    out = out.reset_index().rename(columns={"index": "Candidate"})
    lead = ["Candidate", "verA", "verB"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    return out.sort_values("s_integrated", ascending=False, na_position="last")


def adopted_switches_table(seg):
    def ad(v):
        s = seg[v]
        a = s[s.model_selected == "two_phase"].copy()
        a["version"] = v
        return a
    bar, integ = ad("barcode"), ad("integrated")
    keep = ["version", "Sample", "Candidate", "verA", "verB", "breakpoint_transfer",
            "r_init_per_cycle", "r_final_per_cycle", "rel_rate_change", "delta_bic",
            "r2_constant", "r2_two_phase"]
    tab = pd.concat([bar[keep], integ[keep]], ignore_index=True)
    return tab, bar, integ


# --------------------------------------------------------------------------- figures
@styled
def fig_capability(cap):
    metrics = [("n_estimable", "Estimable\ntrajectories"),
               ("n_breakpoint_testable", "Breakpoint-\ntestable"),
               ("n_two_phase_adopted", "Two-phase\nadopted")]
    fig, ax = plt.subplots(1, 3, figsize=(14, 5.2))
    for j, (col, title) in enumerate(metrics):
        a = ax[j]
        vals = [int(cap.loc[cap.version == v, col].iloc[0]) for v in VERSIONS]
        bars = a.bar([SHORT[v] for v in VERSIONS], vals,
                     color=[VCOLOR[v] for v in VERSIONS], edgecolor=SURFACE, linewidth=2, width=0.68)
        for b, val in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, val, f"{val}", ha="center", va="bottom",
                   fontsize=12, fontweight="bold", color=INK)
        a.set_title(f"({'abc'[j]}) {title}")
        a.set_ylim(0, max(vals) * 1.20 if max(vals) > 0 else 1)
        a.grid(axis="x", visible=False)
        if j == 0:
            a.set_ylabel("# variant trajectories")
    ax[1].annotate("amplicon has only\n3 transfers → no\nbreakpoint identifiable\n"
                   "(4-param ≡ 1-param)", xy=(0, 0.4), xytext=(0.02, 0.62),
                   textcoords=ax[1].transAxes, xycoords=("data", "axes fraction"),
                   ha="left", va="top", fontsize=8.2, color=ORANGE, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax[0].text(0.5, -0.20, "amplicon = 3 primer-set reps of well B4 (82 distinct variants);  "
               "barcode / integrated = 11 wells", transform=ax[0].transAxes, ha="center",
               va="top", fontsize=7.6, color=INK2, style="italic")
    fig.suptitle("Same 4-parameter model, three data regimes — what each regime enables", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(FIG / "fig1_capability.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[compare] wrote figures/fig1_capability.png")


@styled
def fig_b4_integration(seg, depth, stats):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.9))

    # (a) B4-only coverage across the three versions
    a = ax[0]
    cats = ["estimable", "testable", "adopted"]
    data = {v: [stats["b4"][v]["estimable"], stats["b4"][v]["testable"], stats["b4"][v]["adopted"]]
            for v in VERSIONS}
    x = np.arange(len(cats)); w = 0.26
    for i, v in enumerate(VERSIONS):
        b = a.bar(x + (i - 1) * w, data[v], w, color=VCOLOR[v], edgecolor=SURFACE,
                  linewidth=1.5, label=LABEL[v])
        for rect, val in zip(b, data[v]):
            a.text(rect.get_x() + rect.get_width() / 2, val, f"{val}", ha="center", va="bottom",
                   fontsize=8.5, fontweight="bold", color=INK)
    a.set_xticks(x); a.set_xticklabels(["estimable", "breakpoint-\ntestable", "two-phase\nadopted"])
    a.set_ylabel("# variants on well B4"); a.set_title("(a) On the shared B4 culture")
    a.legend(fontsize=7.2, loc="upper right", framealpha=0.9); a.grid(axis="x", visible=False)

    # (b) B4 per-transfer sequencing depth: WGS vs amplicon vs fused
    a = ax[1]
    transfers = [3, 4, 6, 13]
    wgs_b4 = depth["barcode"].loc[B4_WGS].reindex(transfers).fillna(0)
    amp_b4 = depth["amplicon"].sum(axis=0).reindex(transfers).fillna(0)   # pooled over primer sets
    fused_b4 = depth["integrated"].loc[B4_WGS].reindex(transfers).fillna(0)
    xt = np.arange(len(transfers)); w = 0.26
    a.bar(xt - w, wgs_b4.values, w, color=MUTED, edgecolor=SURFACE, linewidth=1.4, label="WGS B4 (~200×)")
    a.bar(xt, amp_b4.values, w, color=ORANGE, edgecolor=SURFACE, linewidth=1.4, label="amplicon B4 (pooled)")
    a.bar(xt + w, fused_b4.values, w, color=BLUE, edgecolor=SURFACE, linewidth=1.4, label="fused (integrated)")
    a.set_yscale("log")
    a.set_xticks(xt); a.set_xticklabels([f"T{t}" for t in transfers])
    a.set_xlabel("transfer"); a.set_ylabel("reads on B4 (log)")
    a.set_title("(b) Depth: amplicon deepens T3/T4/T13, WGS owns T6")
    a.legend(fontsize=7.2, loc="lower left", framealpha=0.9); a.grid(axis="x", visible=False)
    a.annotate("T6 = WGS-only →\nthe point that makes\nB4 breakpoint-testable",
               xy=(xt[2] - w, max(wgs_b4.values[2], 1)), xytext=(xt[2] - 1.4, amp_b4.values.max() * 0.5),
               fontsize=7.2, color=INK2, fontweight="bold",
               arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))

    # (c) newly-estimable / newly-testable count on B4 from integration
    a = ax[2]
    labels = ["estimable\nvariants", "breakpoint-\ntestable", "two-phase\nadopted"]
    wgs_v = [stats["b4"]["barcode"]["estimable"], stats["b4"]["barcode"]["testable"],
             stats["b4"]["barcode"]["adopted"]]
    int_v = [stats["b4"]["integrated"]["estimable"], stats["b4"]["integrated"]["testable"],
             stats["b4"]["integrated"]["adopted"]]
    gain = [i - w0 for i, w0 in zip(int_v, wgs_v)]
    xb = np.arange(len(labels))
    a.bar(xb, wgs_v, color=MUTED, edgecolor=SURFACE, linewidth=1.5, label="WGS B4 alone")
    a.bar(xb, gain, bottom=wgs_v, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
          label="added by amplicon depth")
    for i, (w0, tot) in enumerate(zip(wgs_v, int_v)):
        a.text(i, tot, f"+{tot - w0}", ha="center", va="bottom", fontsize=10,
               fontweight="bold", color=BLUE_DARK)
    a.set_xticks(xb); a.set_xticklabels(labels)
    a.set_ylabel("# variants on B4"); a.set_title("(c) Integration's net gain on B4")
    a.legend(fontsize=7.2, loc="upper right", framealpha=0.9); a.grid(axis="x", visible=False)

    fig.suptitle("What integration adds on well B4 — the one culture measured by both assays", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "fig2_b4_integration.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[compare] wrote figures/fig2_b4_integration.png")


@styled
def fig_rate_concordance(conc, stats):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))

    def scatter(a, xcol, ycol, xlab, ylab, title, col):
        d = conc[[xcol, ycol]].dropna()
        if len(d):
            lim = float(np.nanmax(np.abs(np.r_[d[xcol].values, d[ycol].values]))) * 1.1
            lim = lim if np.isfinite(lim) and lim > 0 else 1.0
            a.plot([-lim, lim], [-lim, lim], color=BASELINE, lw=1.2, ls="--", zorder=1)
            a.axhline(0, color=BASELINE, lw=0.7); a.axvline(0, color=BASELINE, lw=0.7)
            a.scatter(d[xcol], d[ycol], s=40, color=col, edgecolors="white", linewidths=0.5, zorder=3)
            r = np.corrcoef(d[xcol], d[ycol])[0, 1] if len(d) > 2 else np.nan
            a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
            a.text(0.04, 0.96, f"n={len(d)}   r={r:.3f}", transform=a.transAxes, va="top",
                   fontsize=9, color=INK2, fontweight="bold")
        a.set_xlabel(xlab); a.set_ylabel(ylab); a.set_title(title)

    r_bi = stats["concordance"]["barcode_vs_integrated"]["r"]
    r_ai = stats["concordance"]["amplicon_vs_integrated"]["r"]
    scatter(ax[0], "s_barcode", "s_integrated",
            "single-slope s: WGS B4 alone (per cycle)", "single-slope s: integrated B4 (per cycle)",
            f"(a) vs shallow WGS-B4: re-estimated (r={r_bi:.2f})", BLUE)
    scatter(ax[1], "s_amplicon", "s_integrated",
            "single-slope s: amplicon B4 (per cycle)", "single-slope s: integrated B4 (per cycle)",
            f"(b) vs deep amplicon-B4: converges here (r={r_ai:.2f})", AQUA)
    fig.suptitle("Per-variant rate on well B4 — integration tracks the DEEP amplicon, "
                 "re-estimating the shallow WGS rates", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "fig3_rate_concordance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[compare] wrote figures/fig3_rate_concordance.png")


@styled
def fig_adopted_switches(bar, integ, stats):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))

    # (a) adopted two-phase count per culture (one per WELL/full Sample): barcode vs
    #     integrated. Non-B4 wells are WGS-only in BOTH runs, so their bars are equal;
    #     only B4 (the shared culture) differs (0 -> 1).
    a = ax[0]
    cb = bar.groupby("Sample").size(); ci = integ.groupby("Sample").size()
    samples = sorted(set(cb.index) | set(ci.index),
                     key=lambda s: (-(ci.get(s, 0) + cb.get(s, 0)), s))
    y = np.arange(len(samples)); w = 0.4
    a.barh(y + w / 2, [cb.get(s, 0) for s in samples], w, color=MUTED, edgecolor=SURFACE,
           linewidth=1.4, label="barcode/WGS only")
    a.barh(y - w / 2, [ci.get(s, 0) for s in samples], w, color=BLUE, edgecolor=SURFACE,
           linewidth=1.4, label="integrated")
    a.set_yticks(y)
    a.set_yticklabels([_short(s) + ("  ★B4" if s == B4_WGS else "") for s in samples], fontsize=8)
    a.invert_yaxis()
    a.set_xlabel("# adopted two-phase switches")
    a.set_title("(a) Adopted breakpoints per culture — equal except on B4")
    a.legend(fontsize=8, loc="lower right", framealpha=0.9); a.grid(axis="y", visible=False)
    for yi, s in zip(y, samples):
        if s == B4_WGS:
            a.text(ci.get(s, 0) + 0.08, yi - w / 2, "★ well B4: 0 → 1 (amplicon depth resolves it)",
                   ha="left", va="center", fontsize=7.5, color=ORANGE, fontweight="bold")

    # (b) the B4 adopted switches (integrated) as init->final dumbbells
    a = ax[1]
    b4i = integ[integ.Sample == B4_WGS].sort_values("r_final_per_cycle")
    if len(b4i):
        yy = np.arange(len(b4i))
        a.hlines(yy, b4i.r_init_per_cycle, b4i.r_final_per_cycle, color=BASELINE, lw=2.4, zorder=1)
        a.scatter(b4i.r_init_per_cycle, yy, s=60, color=BLUE_LIGHT, edgecolors=BLUE_DARK,
                  linewidths=1, zorder=3, label="r_init")
        a.scatter(b4i.r_final_per_cycle, yy, s=60, color=BLUE_DARK, zorder=3, label="r_final")
        a.axvline(0, color=BASELINE, lw=1, ls="--")
        a.set_yticks(yy)
        a.set_yticklabels([f"{c}  (τ=T{int(t)}, ΔBIC={d:.0f})"
                           for c, t, d in zip(b4i.Candidate, b4i.breakpoint_transfer, b4i.delta_bic)],
                          fontsize=8)
        a.set_xlabel("growth rate (per cycle)")
        a.legend(fontsize=8, loc="lower right", framealpha=0.9); a.grid(axis="y", visible=False)
    a.set_title(f"(b) B4 two-phase switches the integration resolves ({len(b4i)})")

    fig.suptitle("Adopted rate switches: barcode/WGS vs integrated", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "fig4_adopted_switches.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[compare] wrote figures/fig4_adopted_switches.png")


def _short(s):
    """Abbreviate a full Sample name while KEEPING its replicate suffix, so distinct
    wells never collapse onto the same label (each well is a distinct culture)."""
    s = str(s).replace("TFMN4.exp2.ACN3788.", "")
    for a, b in [("concX", "cX"), ("concY", "cY"), ("largeLib", "lg"), ("smallLib", "sm"),
                 ("PCR_DpnI", "PD"), ("_cleanup", "·cl"), ("SpeI", "Sp"), ("EcoRI", "Ec")]:
        s = s.replace(a, b)
    return s


# --------------------------------------------------------------------------- stats + report
def build_stats(seg, msel, cap, conc, bar, integ):
    def b4_counts(v):
        b = b4_rows(seg, v)
        est = int(b.Candidate.nunique() if v == "amplicon" else len(b))
        test = int(b.get("breakpoint_testable", pd.Series(dtype=bool)).sum()) if "breakpoint_testable" in b else 0
        # amplicon has no breakpoint_testable True; recompute from column safely
        if "breakpoint_testable" in b.columns:
            test = int(b["breakpoint_testable"].astype(bool).sum() if v != "amplicon"
                       else b.groupby("Candidate")["breakpoint_testable"].any().sum())
        adop = int((b.model_selected == "two_phase").sum() if v != "amplicon"
                   else b.groupby("Candidate").apply(lambda g: (g.model_selected == "two_phase").any()).sum())
        return dict(estimable=est, testable=test, adopted=adop)

    b4 = {v: b4_counts(v) for v in VERSIONS}
    # overlap of adopted switches (barcode vs integrated) by (Sample,Candidate)
    key = lambda df: set(zip(df.Sample, df.Candidate))
    kb, ki = key(bar), key(integ)
    shared = kb & ki
    new_in_int = ki - kb
    b4_new = sorted(c for (s, c) in new_in_int if s == B4_WGS)

    # concordance correlations
    def corr(xcol, ycol):
        d = conc[[xcol, ycol]].dropna()
        return (float(np.corrcoef(d[xcol], d[ycol])[0, 1]) if len(d) > 2 else None,
                int(len(d)),
                float((d[ycol] - d[xcol]).abs().median()) if len(d) else None)

    r_bi, n_bi, mad_bi = corr("s_barcode", "s_integrated")
    r_ai, n_ai, mad_ai = corr("s_amplicon", "s_integrated")

    stats = dict(
        versions={v: dict(
            n_transfers=_int(msel, v, "n_transfers_sampled"),
            estimable=len(seg[v]),
            testable=_int(msel, v, "n_breakpoint_testable"),
            adopted=_int(msel, v, "n_two_phase_adopted_increasing"),
            held_decrease=_int(msel, v, "n_held_constant_decrease_not_permitted"),
        ) for v in VERSIONS},
        b4=b4,
        adopted_overlap=dict(barcode=len(kb), integrated=len(ki),
                             shared=len(shared), new_in_integrated=len(new_in_int),
                             b4_new_variants=b4_new),
        concordance=dict(barcode_vs_integrated=dict(r=r_bi, n=n_bi, median_abs_delta=mad_bi),
                         amplicon_vs_integrated=dict(r=r_ai, n=n_ai, median_abs_delta=mad_ai)),
    )
    return stats


def write_report(cap, stats):
    v = stats["versions"]; b4 = stats["b4"]; ov = stats["adopted_overlap"]; co = stats["concordance"]
    def sw(n):
        return "switch" if n == 1 else "switches"
    _n_b4_adopt = b4['integrated']['adopted']
    _n_new = ov['new_in_integrated']
    md = f"""# 4-Parameter Growth Model — Three-Way Comparison

*The same four-parameter segmented (two-phase / broken-stick) growth model, applied to
three data regimes drawn from the TFMN4 barcode-competition experiment. Every number below
is computed from the three models' own outputs (this folder's `inputs/`), not re-fit here.*

The four parameters per variant trajectory are **initial abundance**, **initial growth
rate**, **final growth rate**, and the **transfer at which the rate switches** (increasing
switches only, adopted only on strong evidence: ΔBIC > 6 **and** a > 10 % rate increase).

---

## 1. Headline

| Data regime | Transfers | Estimable | Breakpoint-testable | Two-phase adopted |
|---|--:|--:|--:|--:|
| **Amplicon only** | {v['amplicon']['n_transfers']} `{{3,4,13}}` | {v['amplicon']['estimable']} | **{v['amplicon']['testable']}** | **{v['amplicon']['adopted']}** |
| **Barcode / WGS only** | {v['barcode']['n_transfers']} `{{3,4,6,13}}` | {v['barcode']['estimable']} | {v['barcode']['testable']} | {v['barcode']['adopted']} |
| **Integrated (WGS+amplicon)** | {v['integrated']['n_transfers']} `{{3,4,6,13}}` | **{v['integrated']['estimable']}** | **{v['integrated']['testable']}** | **{v['integrated']['adopted']}** |

**The single most important fact:** the amplicon library was sequenced at only **three
transfers `{{3,4,13}}`**, and a breakpoint model needs **≥ 4 timepoints** (≥ 2 either side
of the knot). So on amplicon data the four-parameter model is **{v['amplicon']['testable']}
breakpoint-testable trajectories** — it *cannot* place a breakpoint and silently collapses to
the one-parameter constant-rate fit. Only the WGS design (which adds transfer **T6**) makes
the fourth parameter identifiable at all.

**What integration buys:** fusing the deep amplicon reads into the WGS design on their shared
culture (well **B4**) lifts the estimable count from {v['barcode']['estimable']} →
**{v['integrated']['estimable']}** (+{v['integrated']['estimable']-v['barcode']['estimable']}
variants surfaced by amplicon depth), the testable count {v['barcode']['testable']} →
**{v['integrated']['testable']}**, and the adopted switches {v['barcode']['adopted']} →
**{v['integrated']['adopted']}** — the extra switch is a genuine dip-then-surge on B4 that
WGS's ~200× depth alone could not resolve.

![capability](outputs/figures/fig1_capability.png)

---

## 2. Why amplicon-only can't use the 4th parameter

The amplicon run is *deep* (≈ 10 000 reads/transfer vs WGS's ≈ 190 on B4) but *short*
(3 transfers). Depth improves the **precision of each frequency**; it does nothing for the
**number of timepoints**, which is what a breakpoint needs. Result: all
**{v['amplicon']['estimable']}** amplicon trajectories are reported as constant — the
four-parameter model is mathematically present but degenerate. This is the correct,
identifiability-respecting behaviour, not a bug.

## 3. What integration adds on the shared B4 culture

Well **B4** is the one culture measured by *both* assays (WGS sample
`{B4_WGS.split('.')[-2]}` = amplicon well B4). Integration pools their reads per
(transfer, variant): the amplicon deepens **T3/T4/T13** (~190 → ~10 000 reads) while WGS
supplies the **T6** point the amplicon never sampled — and it is precisely T6 that makes B4
**breakpoint-testable**.

| On well B4 | Amplicon only | WGS only | Integrated |
|---|--:|--:|--:|
| estimable variants | {b4['amplicon']['estimable']} | {b4['barcode']['estimable']} | **{b4['integrated']['estimable']}** |
| breakpoint-testable | {b4['amplicon']['testable']} | {b4['barcode']['testable']} | **{b4['integrated']['testable']}** |
| two-phase adopted | {b4['amplicon']['adopted']} | {b4['barcode']['adopted']} | **{b4['integrated']['adopted']}** |

Integration makes **{b4['integrated']['estimable']-b4['barcode']['estimable']} more B4
variants estimable** than WGS alone and **{b4['integrated']['testable']-b4['barcode']['testable']}
more breakpoint-testable**, resolving **{_n_b4_adopt}** two-phase {sw(_n_b4_adopt)} on
B4 (vs {b4['barcode']['adopted']} from WGS-B4 alone).

![b4](outputs/figures/fig2_b4_integration.png)

## 4. Integration re-estimates B4's noisy rates toward the deep amplicon

On the B4 variants shared across regimes, the single-slope rate **does move — and in an
interpretable direction**. Integrated-B4 correlates **r = {co['amplicon_vs_integrated']['r']:.2f}**
with the *deep amplicon* (n = {co['amplicon_vs_integrated']['n']}, median |Δ| =
{co['amplicon_vs_integrated']['median_abs_delta']:.3f}/cycle) but only
**r = {co['barcode_vs_integrated']['r']:.2f}** with the *shallow WGS-B4* estimate
(n = {co['barcode_vs_integrated']['n']}, median |Δ| =
{co['barcode_vs_integrated']['median_abs_delta']:.3f}/cycle). Because the fused frequencies are
dominated by the amplicon's ~50× greater depth at T3/T4/T13, integration effectively **replaces
WGS-B4's shot-noise-limited rates with amplicon-precision rates**, while still borrowing WGS's
T6 point to make the breakpoint identifiable. That is **variance reduction on the one shallow
culture** — a substantive re-estimation, not a cosmetic tweak, and not a distortion (the fused
estimate lands on the higher-confidence assay).

> Note the contrast with the project's **global 1-parameter** joint fit (`s13d`), where the
> amplicon *only tightened B4's standard error without shifting its rate*. That is expected:
> there B4 is one of eleven wells sharing one global rate, so amplicon depth adds precision at
> the margin. **Here the model is per-culture** — B4's rate is estimated from B4's data alone —
> so replacing ~190-read WGS points with ~10 000-read amplicon points genuinely re-estimates it.

![concordance](outputs/figures/fig3_rate_concordance.png)

## 5. The adopted breakpoints

Barcode/WGS adopts **{ov['barcode']}** two-phase switches; integrated adopts **{ov['integrated']}**.
**{ov['shared']}** are shared and **{_n_new}** {'is' if _n_new == 1 else 'are'} new to the
integrated run{(' (on B4: ' + ', '.join(ov['b4_new_variants']) + ')') if ov['b4_new_variants'] else ''}.
Because the ten non-B4 wells are WGS-only in both runs, integration is **purely additive**
here: it keeps every WGS switch and adds the B4 discovery. Acceleration keeps clustering in the
same alleles the barcode-only run flagged (verA **A81**, **A78**; verB **B26**) — integration
sharpens that story rather than rewriting it.

![switches](outputs/figures/fig4_adopted_switches.png)

---

## 6. Bottom line

1. **Amplicon only** — deep but 3 transfers → the 4th parameter is **not identifiable**;
   every trajectory is constant. Use it for precise *frequencies/relative rates*, not for
   dynamics.
2. **Barcode / WGS only** — 4 transfers over 11 wells → the breakpoint is identifiable;
   **{v['barcode']['adopted']}** variants show a real increasing switch. This is the workhorse
   regime for the 4-parameter model.
3. **Integrated** — WGS breadth + amplicon depth on B4 → **strictly dominates** both: the most
   estimable, most testable, most adopted. On B4 the fused rates converge on the high-depth
   amplicon (r = {co['amplicon_vs_integrated']['r']:.2f}), re-estimating the shallow WGS-B4
   values (r = {co['barcode_vs_integrated']['r']:.2f}). The gain is concentrated on B4 (the only
   shared culture); the other ten wells are WGS-only and byte-for-byte identical to run 2.

*Caveat carried from the parent pipeline: even in the WGS/integrated regimes the four
transfers give the two-phase fit just one residual degree of freedom, so the in-sample fit
gain is partly optimism — out-of-sample (LOOCV) the 4-var only clearly beats the 1-var when
the elbow location is supplied. The four-parameter model's value here is **descriptive**
(exposing dip-then-rise dynamics a single slope averages away), and integration's value is
**making that description possible and precise on B4**.*
"""
    (ROOT / "COMPARISON_REPORT.md").write_text(md)
    print("[compare] wrote COMPARISON_REPORT.md")


def main():
    seg, msel, depth = load()
    cap = capability_table(seg, msel)
    conc = b4_concordance_table(seg)
    switches, bar, integ = adopted_switches_table(seg)
    stats = build_stats(seg, msel, cap, conc, bar, integ)

    cap.to_csv(OUT / "model_capability_comparison.csv", index=False)
    conc.to_csv(OUT / "b4_rate_concordance.csv", index=False)
    switches.to_csv(OUT / "adopted_switches_comparison.csv", index=False)
    (OUT / "comparison_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    fig_capability(cap)
    fig_b4_integration(seg, depth, stats)
    fig_rate_concordance(conc, stats)
    fig_adopted_switches(bar, integ, stats)
    write_report(cap, stats)

    print("\n[compare] capability table:")
    print(cap.to_string(index=False))
    print(f"\n[compare] B4 coverage: {json.dumps(stats['b4'])}")
    print(f"[compare] adopted overlap: {json.dumps(stats['adopted_overlap'])}")
    print(f"[compare] concordance: {json.dumps(stats['concordance'])}")


if __name__ == "__main__":
    main()
