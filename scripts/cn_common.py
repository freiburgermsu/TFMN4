"""
cn_common - shared paths, constants, and fitting primitives for the dgoA copy-number
amplification analysis (copy-number-dgoA_.csv).

Copy number is a DIRECT measurement, so each lineage's absolute amplification rate and
its standard error are directly identifiable:

    ln(copy_number) = a + r * transfer          (1-variable model)
    r  = amplification rate per transfer;  SE_r from the ordinary-least-squares fit.

IMPORTANT — unit of replication.  Several timepoints were measured MORE THAN ONCE (the
sample suffixes .P/.S1/.S2/.L1/.L2 are technical re-measurements of the same population,
not independent evolutionary observations). Treating them as independent is
pseudoreplication and makes the slope SE/CI optimistic. We therefore fit on the
PER-TRANSFER MEAN of ln(copy_number) (one observation per transfer = the unit of
replication), which is the honest error model. `by_transfer()` performs this collapse and
every model (1-var, segmented, LOTO) consumes its output.

The 4-variable model is the same increasing-only broken-stick used for the barcode data
(s14), fit per lineage on the per-transfer means.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats

from s14_segmented_growth import (fit_constant, fit_two_phase, bic,
                                   BIC_MARGIN, REL_RATE_CHANGE_MIN, R_SCALE_FLOOR)

ROOT = Path(__file__).resolve().parent.parent
CN_CSV = ROOT / "copy-number-dgoA_.csv"
OUT = ROOT / "outputs_copynumber"
INTER = OUT / "intermediate"
FIG = OUT / "figures"
for _d in (OUT, INTER, FIG):
    _d.mkdir(parents=True, exist_ok=True)

MIN_UNIQUE_TRANSFERS = 4     # unique transfers to identify a breakpoint / a calibrated CI
MIN_SEG_UNIQUE = 2           # unique transfers on each side of the knot
MIN_EST_TRANSFERS = 3        # unique transfers for a 1-var rate WITH a standard error
CI_RELIABLE_TRANSFERS = 4    # unique transfers for a trustworthy CI (dof >= 2)


# --------------------------------------------------------------------------- labels
def genotype_of(lineage):
    """Precise variant genotype: lineage with the trailing replicate index and the
    experiment prefix stripped (e.g. TFMN1.sohB.tpiA.2 -> 'sohB.tpiA';
    TFMN4.exp1.combo5aWT.1 -> 'combo5aWT'; TFMN1.fba.1 -> 'fba')."""
    stem = re.sub(r"\.\d+$", "", str(lineage))
    for pre in ("TFMN1.", "TFMN4.exp1.", "TFMN4."):
        if stem.startswith(pre):
            return stem[len(pre):]
    return stem


def context_of(lineage):
    """Coarse condition context (for grouping/colour): control, TFMN4 exp1, or a TFMN1
    single- vs double-knockout. Correctly separates double-KOs and no-DNA controls
    (which the naive first-two-tokens grouping merged into single-KO / exp1)."""
    lin = str(lineage); stem = genotype_of(lin)
    if "noDNA" in stem:
        return "control (noDNA)"
    if lin.startswith("TFMN4.exp1"):
        return "TFMN4 exp1"
    return "TFMN1 double-KO" if len(stem.split(".")) >= 2 else "TFMN1 single-KO"


def by_transfer(long_df):
    """Collapse technical replicates to one observation per (lineage, transfer): the
    mean of ln(copy_number). This is the unit of replication used for every fit."""
    return (long_df.groupby(["lineage", "transfer"], as_index=False)
            .agg(ln_cn=("ln_cn", "mean"), copy_number=("copy_number", "mean"),
                 n_reps=("ln_cn", "size")))


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values for a 1-D array of p-values (NaNs pass through)."""
    p = np.asarray(pvals, float); q = np.full(p.shape, np.nan)
    ok = np.isfinite(p); idx = np.where(ok)[0]
    if idx.size == 0:
        return q
    pv = p[idx]; order = np.argsort(pv); m = pv.size
    ranked = pv[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qv = np.empty(m); qv[order] = np.clip(ranked, 0, 1)
    q[idx] = qv
    return q


# --------------------------------------------------------------------------- 1-var
def fit_1var(x, y):
    """OLS ln(CN) ~ a + r*transfer on PER-TRANSFER MEANS (pass by_transfer output).
    Returns the amplification rate, its regression SE, t-based 95% CI, two-sided p-value
    (H0: r=0), R², dof, and the number of transfers. SE/CI/p require dof>=1 (>=3
    transfers); a trustworthy CI needs dof>=2 (>=4 transfers, see ci_reliable)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x); nu = len(np.unique(x))
    out = dict(rate=np.nan, se=np.nan, ci_lo=np.nan, ci_hi=np.nan, intercept=np.nan,
               r2=np.nan, p_value=np.nan, n_transfers=nu, dof=max(n - 2, 0))
    if nu < 2:
        return out
    xbar = x.mean(); Sxx = float(np.sum((x - xbar) ** 2))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x); ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out.update(rate=float(slope), intercept=float(intercept),
               r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan)
    dof = n - 2
    if dof >= 1 and Sxx > 0:
        sigma2 = ss_res / dof; se = float(np.sqrt(sigma2 / Sxx))
        out["se"] = se; out["dof"] = dof
        if se > 0:
            tstat = slope / se
            out["p_value"] = float(2 * stats.t.sf(abs(tstat), dof))
            tcrit = float(stats.t.ppf(0.975, dof))
            out["ci_lo"] = float(slope - tcrit * se); out["ci_hi"] = float(slope + tcrit * se)
        else:  # perfect fit to a horizontal line -> no evidence of a slope
            out["p_value"] = 1.0 if slope == 0 else 0.0
            out["ci_lo"] = out["ci_hi"] = float(slope)
    return out


# --------------------------------------------------------------------------- 4-var
def _feasible_knots(x):
    uniq = np.unique(x); ks = []
    for kt in uniq[1:-1]:
        if (len(np.unique(x[x <= kt])) >= MIN_SEG_UNIQUE
                and len(np.unique(x[x >= kt])) >= MIN_SEG_UNIQUE):
            ks.append(float(kt))
    return ks


def fit_segmented(x, y):
    """Per-lineage increasing-only broken-stick on per-transfer-mean ln(CN), same gates
    as s14 (strong-evidence BIC AND >10% rate increase). Falls back to the constant fit.
    x, y are the per-transfer means (one point per transfer)."""
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.ones_like(x)
    n = len(x); uniq = np.unique(x); n_uniq = len(uniq); t0 = float(uniq[0])
    c = fit_constant(x, y, w); bic_c = bic(c["wrss"], n, 2)
    feasible = _feasible_knots(x)
    testable = (n_uniq >= MIN_UNIQUE_TRANSFERS) and (len(feasible) > 0)

    best = best_raw = None
    if testable:
        for kt in feasible:
            tp = fit_two_phase(x, y, w, kt); tp["knot"] = kt
            if best_raw is None or tp["wrss"] < best_raw["wrss"]:
                best_raw = tp
            if (tp["r_final"] > tp["r_init"]) and (best is None or tp["wrss"] < best["wrss"]):
                best = tp
    bic_two = bic(best["wrss"], n, 3) if best is not None else np.nan
    delta_bic = (bic_c - bic_two) if best is not None else np.nan
    uncon = ("none" if best_raw is None else
             "accelerating" if best_raw["r_final"] > best_raw["r_init"] else "decelerating")
    rel = ((best["r_final"] - best["r_init"]) / max(abs(best["r_init"]), R_SCALE_FLOOR)
           if best is not None else np.nan)
    switch = bool(testable and best is not None and delta_bic > BIC_MARGIN
                  and rel > REL_RATE_CHANGE_MIN)

    if switch:
        model = "two_phase"; knot = best["knot"]; r_init, r_final = best["r_init"], best["r_final"]
        r2_sel, cyc = best["r2"], best
    else:
        model = "constant"; knot = np.nan; r_init = r_final = c["slope"]; r2_sel, cyc = c["r2"], c
    logA_init = cyc["coef"][0] + cyc["coef"][1] * t0
    return dict(
        model_selected=model, breakpoint_testable=bool(testable),
        logA_init=float(logA_init), r_init=float(r_init), r_final=float(r_final),
        breakpoint_transfer=(float(knot) if switch else np.nan), unconstrained_direction=uncon,
        delta_bic=(float(delta_bic) if np.isfinite(delta_bic) else np.nan),
        rel_rate_change=(float(rel) if np.isfinite(rel) else np.nan),
        r2_constant=float(c["r2"]), r2_two_phase=(float(best["r2"]) if best is not None else np.nan),
        r2_selected=float(r2_sel),
        wrss_constant=float(c["wrss"]), wrss_two_phase=(float(best["wrss"]) if best is not None else np.nan),
        wrss_unconstrained=(float(best_raw["wrss"]) if best_raw is not None else np.nan),
        r2_unconstrained=(float(best_raw["r2"]) if best_raw is not None else np.nan),
        knot_unconstrained=(float(best_raw["knot"]) if best_raw is not None else np.nan),
        knot_increasing=(float(best["knot"]) if best is not None else np.nan),
        s_constant=float(c["slope"]), n_transfers=n_uniq)


def loto_cv(x, y, seg_knot=None):
    """Leave-one-transfer-out CV on the per-transfer means: hold out one transfer, fit on
    the rest, predict it. Compares the constant fit vs the broken-stick FORM (per-fold
    best feasible knot, or a supplied knot). Paired over folds usable for BOTH models.
    Returns (rmse_1var, rmse_4var, n_folds_used)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    uniq = np.unique(x); eA, eB, used = [], [], 0
    for t in uniq:
        m = x != t; xt, yt = x[m], y[m]
        if len(np.unique(xt)) < 2:
            continue
        cA = fit_constant(xt, yt, np.ones_like(xt)); predA = cA["coef"][0] + cA["coef"][1] * t
        knots = [seg_knot] if seg_knot is not None else _feasible_knots(xt)
        bestk = None
        for kt in [k for k in knots if k is not None]:
            h = np.maximum(0.0, xt - kt); X = np.column_stack([np.ones_like(xt), xt, h])
            if np.linalg.matrix_rank(X) < 3:
                continue
            tp = fit_two_phase(xt, yt, np.ones_like(xt), kt)
            if bestk is None or tp["wrss"] < bestk["wrss"]:
                bestk = tp; bestk["knot"] = kt
        if bestk is None:
            continue
        predB = bestk["coef"][0] + bestk["coef"][1] * t + bestk["coef"][2] * max(0.0, t - bestk["knot"])
        yh = y[x == t]
        eA.extend((yh - predA) ** 2); eB.extend((yh - predB) ** 2); used += 1
    if used < 2:
        return np.nan, np.nan, used
    return float(np.sqrt(np.mean(eA))), float(np.sqrt(np.mean(eB))), used
