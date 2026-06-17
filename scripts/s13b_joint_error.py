"""
s13b - Error decomposition for the joint WGS+amplicon 4MXB fit.

s13 gives the SAMPLING error (Laplace se_joint, and se_wgs_only/se_ampl_only for the
reduction ratio). This script adds the components the headline must not hide:

  se_heterogeneity  - leave-one-WGS-well-out refits of the JOINT model -> SD of r_i
                      across folds. This is CROSS-CULTURE (biological) variation; it
                      must NOT shrink just because the single B4 culture is deep.
  b_amp_sensitivity - refit at w_b in {strong, weak, off}: how much the assay-drift
                      regularization moves r_i (confirms b_amp ~ 0 is robust).

Output: augments outputs_joint_4MXB/joint_variant_growth_rates_4MXB.csv in place.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C
from s13_joint_4MXB import load_combined, build, fit_point, OUT, W_B


def main():
    df = load_combined()
    wells = sorted(df[df.assay == "WGS"]["Microtiter_plate_well"].astype(str).unique())

    # ---- LOSO-by-WGS-well on the joint model (point estimates) ----
    loso = {}
    for w in wells:
        sub = df[~((df.assay == "WGS") & (df["Microtiter_plate_well"].astype(str) == w))]
        D = build(sub, ["WGS", "amplicon"], use_b=True)
        rmap, _ = fit_point(D)
        for v, r in rmap.items():
            loso.setdefault(v, []).append(r)
    het = {v: (np.std(rs, ddof=1) if len(rs) >= 2 else np.nan) for v, rs in loso.items()}

    # ---- b_amp / assay-drift sensitivity ----
    Dj = build(df, ["WGS", "amplicon"], use_b=True)
    r_strong, b_s = fit_point(Dj, w_b=W_B)
    r_weak, b_w = fit_point(Dj, w_b=W_B / 50.0)
    r_off, _ = fit_point(Dj, w_b=1e12)        # b_amp pinned ~0
    sens = {v: abs(r_weak.get(v, np.nan) - r_off.get(v, np.nan)) for v in r_strong}

    res = pd.read_csv(OUT / "joint_variant_growth_rates_4MXB.csv")
    res["se_heterogeneity_loso"] = res["Candidate"].map(het).round(4)
    res["r_sensitivity_to_b_amp"] = res["Candidate"].map(sens).round(4)
    # honest headline CI half-width = combine sampling (Laplace) and heterogeneity in quadrature
    res["se_combined"] = np.sqrt(res["se_joint"].fillna(0) ** 2
                                 + res["se_heterogeneity_loso"].fillna(0) ** 2).round(4)
    res.to_csv(OUT / "joint_variant_growth_rates_4MXB.csv", index=False)

    print(f"[s13b] b_amp under weak ridge = {b_w:+.4f} (strong={b_s:+.4f}); "
          f"median r sensitivity to assay-drift = {np.nanmedian(list(sens.values())):.4f}/h")
    print(f"[s13b] LOSO-by-well heterogeneity: median={np.nanmedian(list(het.values())):.4f}/h, "
          f"max={np.nanmax([v for v in het.values() if np.isfinite(v)]):.3f}/h")
    both = res[res.evidence_tier == "both-deep"]
    print(f"[s13b] both-deep: median se_joint={both.se_joint.median():.4f} vs "
          f"median se_heterogeneity={both.se_heterogeneity_loso.median():.4f} "
          f"(heterogeneity often >= sampling -> cross-culture variation dominates)")
    print("[s13b] augmented joint_variant_growth_rates_4MXB.csv")


if __name__ == "__main__":
    main()
