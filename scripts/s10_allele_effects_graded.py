"""
s10 - Graded allele-level growth table: relative effects (trustworthy) plus a
clearly-flagged DERIVED OD-clock rate.

For each verA / verB allele we report the drift-gated marginal selection
coefficient (per cycle AND per generation) from s05, then append a derived
absolute-ish quantity on the OD clock:

    R_allele = mu_bulk_community + (marginal_s_per_cycle / tau_exp)   [OD-equiv 1/h]

This is reported as an INTERVAL over the community mu_bulk (concX vs concY) and is
explicitly NOT a per-variant measurement: the correction term s/tau_exp is only a
few percent of mu_bulk, and algebraically equals marginal_s_per_generation*(mu/ln2)
-- so it carries no information beyond the relative effect. Assumption-flag
booleans make that explicit in the file.

Output (outputs/): allele_growth_advantage.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C


def main():
    cond = pd.read_csv(C.INTER / "od_bulk_condition_summary.csv").set_index("construct_family")
    bulk = pd.read_csv(C.INTER / "od_bulk_rate.csv")
    mu_lo = float(cond["median"].min())          # concY (lower)
    mu_hi = float(cond["median"].max())          # concX (higher)
    mu_mid = float(np.median([mu_lo, mu_hi]))
    tau = float(bulk.loc[bulk.reliable & bulk.transfer.isin([6, 13]), "tau_exp_h"].median())

    ax = pd.read_csv(C.INTER / "od_time_axis.csv").set_index("transfer")["cumgen"]
    gpc = (ax[13] - ax[3]) / (13 - 3)             # generations per cycle

    frames = []
    for part in ("verA", "verB"):
        a = pd.read_csv(C.OUT / f"allele_effects_{part}.csv")
        a.insert(0, "allele_kind", part)
        a["marginal_s_per_generation"] = (a["marginal_s_per_cycle"] / gpc).round(4)
        # derived OD-clock rate (per hour) as an interval over the community mu_bulk
        corr = a["marginal_s_per_cycle"] / tau          # relative advantage, per hour
        a["R_allele_OD_clock_per_h"] = (mu_mid + corr).round(4)
        a["R_allele_lo"] = (mu_lo + corr).round(4)
        a["R_allele_hi"] = (mu_hi + corr).round(4)
        a["correction_frac_of_mubulk"] = (corr.abs() / mu_mid).round(3)
        # assumption flags
        a["flag_anchor_is_community_level"] = True
        a["flag_correction_is_few_pct_of_mubulk"] = a["correction_frac_of_mubulk"] < 0.10
        a["flag_OD_not_biomass"] = True
        a["flag_derived_not_measured"] = True
        frames.append(a)

    out = pd.concat(frames, ignore_index=True)
    cols = ["allele_kind", "allele", "marginal_s_per_cycle", "marginal_s_per_generation",
            "se", "n_pairs", "n_samples", "drift_null_p", "exceeds_drift_null",
            "robust_selection", "R_allele_OD_clock_per_h", "R_allele_lo", "R_allele_hi",
            "correction_frac_of_mubulk", "flag_anchor_is_community_level",
            "flag_correction_is_few_pct_of_mubulk", "flag_OD_not_biomass",
            "flag_derived_not_measured"]
    out = out[[c for c in cols if c in out.columns]].sort_values(
        ["allele_kind", "marginal_s_per_cycle"], ascending=[True, False])
    out.to_csv(C.OUT / "allele_growth_advantage.csv", index=False)

    rob = out[out.robust_selection]
    print(f"[s10] community mu_bulk interval {mu_lo:.3f}-{mu_hi:.3f}/h, tau_exp={tau:.1f}h, "
          f"gen/cycle={gpc:.2f}")
    print(f"[s10] robust (drift-confirmed) alleles: {len(rob)}; their OD-clock correction is "
          f"{rob['correction_frac_of_mubulk'].min():.1%}-{rob['correction_frac_of_mubulk'].max():.1%} of mu_bulk")
    for _, r in rob.iterrows():
        print(f"        {r.allele_kind} {r.allele:6s} s={r.marginal_s_per_cycle:+.3f}/cyc  "
              f"R_OD≈{r.R_allele_OD_clock_per_h:.3f}/h (corr {r.correction_frac_of_mubulk:.1%} of bulk)")
    print("[s10] wrote allele_growth_advantage.csv")
    return out


if __name__ == "__main__":
    main()
