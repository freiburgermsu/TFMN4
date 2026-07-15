"""
cn02 - THE HEADLINE: predicted growth (amplification) rate + honest error per variant.

For every lineage, fit ln(copy_number) = a + r*transfer by OLS on the PER-TRANSFER MEANS
(technical replicates collapsed to the unit of replication — this is the honest error
model; treating replicates as independent would make SEs/CIs ~24-42% too small). Report
the amplification rate r with its regression SE, t-based 95% CI, two-sided p-value, R²,
and the per-transfer fold change e^r.

Two honesty layers requested by the adversarial review:
  * LOW LEVERAGE — a trustworthy CI needs >=4 distinct transfers (dof>=2). Lineages with
    3 transfers (dof=1, t_crit~12.7; nearly all `exp1`) are rate-estimable but CI is not
    calibrated; they are tiered `low-leverage` and NOT given significance calls.
  * MULTIPLICITY — with dozens of lineages tested, a per-lineage CI-excludes-0 rule
    yields many false positives. Significance (`amplifying`/`losing`) is therefore based
    on a Benjamini-Hochberg FDR q-value < 0.05 computed among the CI-reliable lineages.
The raw CI-excludes-0 flag is kept (`raw_ci_excludes_0`) for transparency.

Output (outputs_copynumber/): dgoA_growth_rates.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import cn_common as CN

FDR_Q = 0.05


def main():
    bt = pd.read_csv(CN.INTER / "dgoA_bytransfer.csv")
    idx = pd.read_csv(CN.INTER / "dgoA_lineage_index.csv").set_index("lineage")
    rows = []
    for lin, g in bt.groupby("lineage"):
        g = g.sort_values("transfer")
        f = CN.fit_1var(g["transfer"].to_numpy(float), g["ln_cn"].to_numpy(float))
        if not np.isfinite(f["rate"]):
            continue
        nt = f["n_transfers"]
        tier = ("reliable" if nt >= CN.CI_RELIABLE_TRANSFERS else
                "low-leverage (3 transfers)" if nt == 3 else "rate-only (2 transfers)")
        rows.append(dict(
            lineage=lin, genotype=g["genotype"].iloc[0], context=g["context"].iloc[0],
            rate_per_transfer=round(f["rate"], 5), se=round(f["se"], 5) if np.isfinite(f["se"]) else np.nan,
            ci95_lo=round(f["ci_lo"], 5) if np.isfinite(f["ci_lo"]) else np.nan,
            ci95_hi=round(f["ci_hi"], 5) if np.isfinite(f["ci_hi"]) else np.nan,
            p_value=f["p_value"], fold_change_per_transfer=round(float(np.exp(f["rate"])), 4),
            r2=round(f["r2"], 4) if np.isfinite(f["r2"]) else np.nan,
            n_points=int(idx.loc[lin, "n_points"]), n_transfers=nt, dof=f["dof"], tier=tier,
            raw_ci_excludes_0=bool(np.isfinite(f["ci_lo"]) and (f["ci_lo"] > 0 or f["ci_hi"] < 0)),
        ))
    res = pd.DataFrame(rows)

    # FDR among the CI-reliable lineages only (the tests we actually interpret)
    res["q_value"] = np.nan
    rel = res["tier"] == "reliable"
    res.loc[rel, "q_value"] = CN.bh_fdr(res.loc[rel, "p_value"].to_numpy(float))

    def call(r):
        if r.tier != "reliable" or not np.isfinite(r.q_value):
            return "uncertain (low leverage)"
        if r.q_value < FDR_Q and r.rate_per_transfer > 0:
            return "amplifying"
        if r.q_value < FDR_Q and r.rate_per_transfer < 0:
            return "losing"
        return "flat"
    res["call"] = res.apply(call, axis=1)
    res["p_value"] = res["p_value"].round(5)
    res["q_value"] = res["q_value"].round(4)
    res = res.sort_values(["context", "rate_per_transfer"], ascending=[True, False])
    res.to_csv(CN.OUT / "dgoA_growth_rates.csv", index=False)

    n = len(res); nrel = int(rel.sum())
    amp = res[res.call == "amplifying"]; los = res[res.call == "losing"]
    raw = int(res.raw_ci_excludes_0.sum())
    print(f"[cn02] fitted 1-var amplification rate ± SE for {n} lineages "
          f"(OLS of ln(copy_number) on transfer, per-transfer means).")
    print(f"[cn02] rate range [{res.rate_per_transfer.min():+.3f}, {res.rate_per_transfer.max():+.3f}]/transfer; "
          f"median SE={res.se.median():.3f}.")
    print(f"[cn02] tiers: {int((res.tier=='reliable').sum())} CI-reliable (>=4 transfers), "
          f"{int((res.tier.str.startswith('low')).sum())} low-leverage (3 transfers), "
          f"{int((res.tier.str.startswith('rate')).sum())} rate-only (2 transfers).")
    print(f"[cn02] significance among reliable (BH-FDR q<{FDR_Q}): {len(amp)} amplifying, {len(los)} losing "
          f"(vs {raw} lineages whose RAW CI excludes 0 — most are multiplicity/low-leverage artifacts).")
    if len(amp):
        print("[cn02] robust amplifiers (FDR-significant):")
        for _, x in amp.sort_values("rate_per_transfer", ascending=False).iterrows():
            print(f"        {x.lineage:22s} r={x.rate_per_transfer:+.3f}±{x.se:.3f}/transfer "
                  f"(×{x.fold_change_per_transfer:.2f}, q={x.q_value:.3f}, R²={x.r2:.2f}, {x.n_transfers} transfers)")
    print("[cn02] per-context median rate:")
    for c, g in res.groupby("context"):
        print(f"        {c:18s} n={len(g):2d}  median r={g.rate_per_transfer.median():+.3f}/transfer  "
              f"({int((g.call=='amplifying').sum())} FDR-amplifying)")
    print("[cn02] wrote dgoA_growth_rates.csv")
    return res


if __name__ == "__main__":
    main()
