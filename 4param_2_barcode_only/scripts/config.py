"""
Folder-local configuration + numerical helpers for the SELF-CONTAINED 4-parameter
(segmented / two-phase breakpoint) growth-rate analysis in this folder.

This is a trimmed, self-contained copy of the TFMN4 pipeline's shared config, holding
ONLY what the 4-parameter segmented sub-pipeline needs
(s01 -> s02 -> s14 -> s14b -> s14c -> s14d -> s14e). The model is derived in the
top-level docs/METHODS.md (section 11). In brief: sequencing read counts are
compositional (fixed depth per timepoint), so only RELATIVE growth is identifiable;
the 4-parameter segmented model fits, per variant trajectory, an initial abundance,
an initial growth rate, a final growth rate, and the transfer at which the rate
switches (a continuous broken-stick, increasing-rate switches only).

Paths are LOCAL to this folder (inputs/, outputs/) so the analysis runs standalone.
The active dataset is fixed by DATASET below (no env var required).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Local paths: this folder is self-contained (config.py lives in <folder>/scripts/).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # the analysis folder
INPUTS = ROOT / "inputs"
BARCODE_CSV = INPUTS / "SeqCenter_2026-05-26_QUO1022807_exp2_extracted_barcodes.csv"
OD_CSV = INPUTS / "OD_4MXB.csv"
AMPLICON_READS = INPUTS / "amplicons_reads.csv"

# ---------------------------------------------------------------------------
# Dataset this folder analyses. The SAME segmented code runs on every dataset;
# only s01 (ingestion) and s02 (OD linkage) branch on it.
#   amplicon_4MXB   - Plasmidsaurus amplicon, well B4 on 4-methoxybenzoate, transfers {3,4,13}
#   wgs             - SeqCenter WGS extracted barcodes, 11 wells, transfers {3,4,6,13}
#   integrated_4MXB - WGS breadth (11 wells) + amplicon depth fused on well B4, {3,4,6,13}
# ---------------------------------------------------------------------------
DATASET = os.environ.get("TFMN4_DATASET", "wgs")
_CFG = {
    "wgs": dict(transfers=(3, 4, 6, 13), reliable_od=(6, 13),
                od_series="exp2", od_condition=None, amplicon_carbon=None),
    "amplicon_4MXB": dict(transfers=(3, 4, 13), reliable_od=(4, 13),
                          od_series="exp2", od_condition="Methoxybenzoate 4 mM + Kan", amplicon_carbon="M"),
    "amplicon_pyruvate": dict(transfers=(3, 4, 13), reliable_od=(4, 13),
                              od_series="exp1", od_condition="Pyruvate 20 mM + Kan", amplicon_carbon="p"),
    "integrated_4MXB": dict(transfers=(3, 4, 6, 13), reliable_od=(6, 13),
                            od_series="exp2", od_condition=None, amplicon_carbon="M"),
}
if DATASET not in _CFG:
    raise SystemExit(f"Unknown TFMN4_DATASET={DATASET!r}; choose from {list(_CFG)}")
_c = _CFG[DATASET]

TRANSFERS = _c["transfers"]                 # the sampled transfers for this dataset
FOUR_TRANSFER_SET = TRANSFERS               # backward-compatible alias
RELIABLE_OD_TRANSFERS = _c["reliable_od"]   # transfers with a usable within-cycle mu_bulk
OD_SERIES = _c["od_series"]                 # OD-file series to read
OD_CONDITION = _c["od_condition"]           # OD-file Condition filter (None = no filter)
AMPLICON_CARBON = _c["amplicon_carbon"]     # 'M' (4MXB) or 'p' (pyruvate), or None for wgs

OUT = ROOT / "outputs"                       # local outputs/ (self-contained)
INTER = OUT / "intermediate"
FIG = OUT / "figures"
for _d in (OUT, INTER, FIG):
    _d.mkdir(parents=True, exist_ok=True)


def read_focus_od(usecols, focus_wells):
    """Read the OD file filtered to this dataset's series/condition and the given
    wells. Centralizes the per-dataset OD linkage used by s02."""
    import pandas as pd
    need = sorted(set(list(usecols) + ["series", "Condition", "Microtiter_plate_well"]))
    od = pd.read_csv(OD_CSV, usecols=need)
    od = od[od["series"].astype(str) == OD_SERIES]
    if OD_CONDITION is not None:
        od = od[od["Condition"].astype(str) == OD_CONDITION]
    od = od[od["Microtiter_plate_well"].astype(str).isin(set(map(str, focus_wells)))]
    return od.copy()

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
# Pseudocount for log/CLR (Haldane-Anscombe style). Applied to every cell of the
# per-Sample variant x transfer union grid so that a true zero read becomes a small
# "below detection" frequency rather than -inf.
PSEUDOCOUNT = 0.5

# A (Sample, Candidate) trajectory is only estimable when the variant is observed
# (count > 0) at this many timepoints or more.
MIN_OBS_TIMEPOINTS = 2

# Time-axis conversions. The barcode "Transfer" index is the gauge-robust clock
# (slope per transfer-cycle is invariant to how long a cycle is). Hours require a
# cycle length; MEASURED = 26.4 h/transfer (instrument timestamps), NOMINAL = 10 h.
HOURS_PER_TRANSFER_MEASURED = 26.4
HOURS_PER_TRANSFER_NOMINAL = 10.0

# Robustness bounds for the OD-derived generations-per-cycle estimate (Delta g).
OD_INOCULUM_FLOOR = 0.01      # net OD floor for the post-dilution inoculum
DELTA_G_BOUNDS = (0.5, 12.0)  # plausible generations per serial-transfer cycle


# ---------------------------------------------------------------------------
# Numerical helpers (identical to the parent pipeline)
# ---------------------------------------------------------------------------
def weighted_linfit(x, y, w):
    """Weighted least-squares straight-line fit y ~ a + b*x.

    Returns dict with slope, intercept, slope_se, r2 (weighted), n_points.
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

    Each column (timepoint) is centered by subtracting the column mean of
    log(count + pseudocount), i.e. the geometric-mean reference. This is the
    gauge-free log-frequency used for the slope fit.
    """
    L = np.log(count_df.to_numpy(float) + pseudocount)
    L = L - L.mean(axis=0, keepdims=True)
    import pandas as pd
    return pd.DataFrame(L, index=count_df.index, columns=count_df.columns)
