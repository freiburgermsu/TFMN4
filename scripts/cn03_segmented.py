"""
cn03 - The 4-variable (segmented / breakpoint) model on the dgoA copy-number data.

Per lineage, fit the increasing-only continuous broken-stick on ln(copy_number):

    ln(CN) = logA + r_init*(t - t0) + (r_final - r_init)*max(0, t - tau)

the four parameters being the initial abundance (logA), the initial amplification rate,
the final amplification rate, and the breakpoint transfer tau. Identical decision policy
to the barcode model (s14): a switch is adopted only if it is an INCREASE (r_final >
r_init), the weighted BIC improves by the strong-evidence margin, and the rate changes
by >10%; otherwise the constant fit stands. Needs >=4 unique transfers (>=2 per segment),
which — unlike the 3-transfer barcode amplicon — many of these lineages have.

Output (outputs_copynumber/): dgoA_segmented.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import cn_common as CN


def main():
    bt = pd.read_csv(CN.INTER / "dgoA_bytransfer.csv")
    rows = []
    for lin, g in bt.groupby("lineage"):
        g = g.sort_values("transfer")
        if g["transfer"].nunique() < CN.MIN_EST_TRANSFERS:
            continue
        s = CN.fit_segmented(g["transfer"].to_numpy(float), g["ln_cn"].to_numpy(float))
        direction = ("accelerating" if s["model_selected"] == "two_phase" else "constant")
        rows.append(dict(
            lineage=lin, genotype=g["genotype"].iloc[0], context=g["context"].iloc[0],
            model_selected=s["model_selected"], direction=direction,
            unconstrained_direction=s["unconstrained_direction"],
            breakpoint_testable=s["breakpoint_testable"],
            logA_init=round(s["logA_init"], 4),
            r_init_per_transfer=round(s["r_init"], 5), r_final_per_transfer=round(s["r_final"], 5),
            breakpoint_transfer=(s["breakpoint_transfer"] if np.isfinite(s["breakpoint_transfer"]) else np.nan),
            rel_rate_change=(round(s["rel_rate_change"], 4) if np.isfinite(s["rel_rate_change"]) else np.nan),
            delta_bic=(round(s["delta_bic"], 3) if np.isfinite(s["delta_bic"]) else np.nan),
            r2_constant=round(s["r2_constant"], 4),
            r2_two_phase=(round(s["r2_two_phase"], 4) if np.isfinite(s["r2_two_phase"]) else np.nan),
            s_constant_per_transfer=round(s["s_constant"], 5),
            n_transfers=s["n_transfers"],
        ))
    res = pd.DataFrame(rows).sort_values(["model_selected", "delta_bic"], ascending=[True, False])
    res.to_csv(CN.OUT / "dgoA_segmented.csv", index=False)

    test = res[res.breakpoint_testable]
    two = res[res.model_selected == "two_phase"]
    held = test[(test.model_selected == "constant") & (test.unconstrained_direction == "decelerating")]
    print(f"[cn03] segmented model fit for {len(res)} lineages; "
          f"{len(test)} breakpoint-testable (>=4 transfers).")
    print(f"[cn03] increasing two-phase adopted for {len(two)}/{len(test)} testable "
          f"(strong-evidence BIC AND +Δrate>10%); "
          f"{len(held)} held constant because their best fit was a DECREASE (plateau/saturation).")
    if len(two):
        for _, x in two.sort_values("delta_bic", ascending=False).head(4).iterrows():
            print(f"        {x.lineage:20s} r {x.r_init_per_transfer:+.3f}->{x.r_final_per_transfer:+.3f}/transfer "
                  f"@T{x.breakpoint_transfer:g} (ΔBIC={x.delta_bic:.1f})")
    print("[cn03] wrote dgoA_segmented.csv")
    return res


if __name__ == "__main__":
    main()
