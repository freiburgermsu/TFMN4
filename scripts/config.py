"""
Shared configuration and small numerical helpers for the TFMN4 barcode
relative-fitness (growth-rate) pipeline.

The model and its justification are documented in docs/METHODS.md. In brief:
sequencing read counts are *compositional* (fixed depth per timepoint), so only
*relative* growth rates are identifiable. We estimate, for each variant
(Candidate = verA-verB) in each Sample, a selection coefficient s = slope of the
centered-log-ratio (CLR) of its read count versus time. s is the variant's
growth rate relative to the community mean (replicator-equation gauge).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Paths (everything is relative to the repository root = parent of this file's dir)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
BARCODE_CSV = ROOT / "SeqCenter_2026-05-26_QUO1022807_exp2_extracted_barcodes.csv"
OD_CSV = ROOT / "TFMN4_robotic_OD_processed_final.csv"

OUT = ROOT / "outputs"
INTER = OUT / "intermediate"
FIG = OUT / "figures"
for _d in (OUT, INTER, FIG):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
# "First assessment": only Samples observed at all four transfers.
FOUR_TRANSFER_SET = (3, 4, 6, 13)

# Pseudocount for log/CLR (Haldane-Anscombe style). Applied to every cell of the
# per-Sample variant x transfer union grid so that a true zero read becomes a
# small "below detection" frequency rather than -inf.
PSEUDOCOUNT = 0.5

# A (Sample, Candidate) selection coefficient is only reported when the variant
# is actually observed (count > 0) at this many timepoints or more.
MIN_OBS_TIMEPOINTS = 2

# Time-axis conversions. The barcode "Transfer" index is the gauge-robust clock
# (slope per transfer-cycle is invariant to how long a cycle is). Hours require a
# cycle length; the two candidates below are reconciled in docs/METHODS.md:
#   - MEASURED: 26.4 h/transfer, derived from File-B instrument timestamps.
#   - NOMINAL : 10 h/transfer, as stated in the original task description.
HOURS_PER_TRANSFER_MEASURED = 26.4
HOURS_PER_TRANSFER_NOMINAL = 10.0

# Robustness bounds for the OD-derived generations-per-cycle estimate (Delta g).
OD_INOCULUM_FLOOR = 0.01      # net OD floor for the post-dilution inoculum
DELTA_G_BOUNDS = (0.5, 12.0)  # plausible generations per serial-transfer cycle


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def weighted_linfit(x, y, w):
    """Weighted least-squares straight-line fit y ~ a + b*x.

    Returns dict with slope, intercept, slope_se, r2 (weighted), n_points.
    Weights w act as precisions (larger = more trusted). SE uses the empirical
    weighted residual variance, so it is meaningful even though our weights are
    a Poisson precision proxy (count + pseudocount) rather than calibrated
    inverse variances.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    n = int(np.sum(w > 0))
    W = w.sum()
    out = dict(slope=np.nan, intercept=np.nan, slope_se=np.nan, r2=np.nan, n_points=n)
    if n < 2 or W <= 0:
        return out
    xbar = np.sum(w * x) / W
    ybar = np.sum(w * y) / W
    Sxx = np.sum(w * (x - xbar) ** 2)
    if Sxx <= 0:
        return out
    Sxy = np.sum(w * (x - xbar) * (y - ybar))
    slope = Sxy / Sxx
    intercept = ybar - slope * xbar
    resid = y - (intercept + slope * x)
    ss_res = np.sum(w * resid ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    dof = max(n - 2, 1)
    sigma2 = ss_res / dof
    out.update(
        slope=slope,
        intercept=intercept,
        slope_se=np.sqrt(sigma2 / Sxx),
        r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        n_points=n,
    )
    return out


def clr_matrix(count_df, pseudocount=PSEUDOCOUNT):
    """Centered-log-ratio of a (variant x timepoint) count frame.

    Each *column* (timepoint) is centered by subtracting the column mean of
    log(count + pseudocount), i.e. the geometric-mean reference. This is the
    gauge-free log-frequency used for the slope fit.
    """
    L = np.log(count_df.to_numpy(float) + pseudocount)
    L = L - L.mean(axis=0, keepdims=True)
    import pandas as pd
    return pd.DataFrame(L, index=count_df.index, columns=count_df.columns)
