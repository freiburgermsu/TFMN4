"""
s03 - Estimate the per-(Sample, Candidate) selection coefficient.

Model (see docs/METHODS.md): within a Sample, latent abundances grow as
dN_i/dt = r_i N_i. Sequencing sees only frequencies f_i = N_i / sum_j N_j, whose
dynamics are the replicator equation df_i/dt = f_i (r_i - rbar). Hence only the
*relative* rate s_i = r_i - rbar is identifiable, and it equals the slope of the
centered-log-ratio (CLR) of read counts versus time:

    CLR_i(t) = ln(count_i + d) - mean_j ln(count_j + d)
    s_i      = d/dt CLR_i(t)        (estimated by weighted least squares)

For each Sample we build the variant x transfer union grid (absent variants get
count 0 -> CLR via pseudocount = "below detection"), then fit s_i on three time
axes:
  - per transfer-cycle  (x = Transfer index; gauge-robust, no assumptions)
  - per generation      (x = cumulative generations from OD; canonical fitness)
  - per hour            (per-cycle / cycle-length; reported for both 26.4 h and 10 h)

Outputs (outputs/):
  selection_coefficients_long.csv  - full per-(Sample,Candidate) table + flags
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C


def fit_one_sample(sub, axis):
    """Fit all Candidates within one Sample. `sub` is the long rows for the
    Sample; `axis` maps transfer -> (cumgen). Returns a list of result dicts."""
    transfers = list(C.FOUR_TRANSFER_SET)
    counts = (sub.pivot_table(index="Candidate", columns="Transfer",
                              values="Count", aggfunc="sum", fill_value=0)
              .reindex(columns=transfers, fill_value=0))
    clr = C.clr_matrix(counts)

    x_cycle = np.array(transfers, float)
    x_gen = np.array([axis.loc[t, "cumgen"] for t in transfers], float)
    depth = counts.sum(axis=0).to_numpy(float)         # within-sample depth per transfer

    meta = sub.drop_duplicates("Candidate").set_index("Candidate")
    results = []
    for cand in counts.index:
        cvec = counts.loc[cand].to_numpy(float)        # raw counts per transfer
        yvec = clr.loc[cand].to_numpy(float)           # CLR per transfer
        w = cvec + C.PSEUDOCOUNT                        # Poisson precision proxy
        n_obs = int((cvec > 0).sum())
        # within-sample frequency at each transfer = count / depth at that transfer
        fvec = np.divide(cvec, depth, out=np.zeros_like(cvec), where=depth > 0)

        fit_cycle = C.weighted_linfit(x_cycle, yvec, w)
        fit_gen = C.weighted_linfit(x_gen, yvec, w)

        s_cycle = fit_cycle["slope"]
        r2 = fit_cycle["r2"]
        # Monotonicity over the OBSERVED timepoints (a single global slope is
        # only a good summary for monotone trajectories; ~80% are not).
        obs_mask = cvec > 0
        if obs_mask.sum() >= 2:
            d = np.diff(np.log(fvec[obs_mask]))
            monotonic = bool(np.all(d >= -1e-12) or np.all(d <= 1e-12))
        else:
            monotonic = False
        # Confidence tier for reading the matrix.
        if n_obs >= 4 and (r2 or 0) >= 0.8:
            quality = "high"
        elif n_obs >= 3 and (r2 or 0) >= 0.6:
            quality = "good"
        elif n_obs >= C.MIN_OBS_TIMEPOINTS:
            quality = "fair"
        else:
            quality = "none"
        results.append(dict(
            Sample=sub["Sample"].iloc[0],
            DNA_construct=meta.loc[cand, "DNA_construct"],
            Replicate=meta.loc[cand, "Replicate"],
            Candidate=cand,
            verA=meta.loc[cand, "verA"],
            verB=meta.loc[cand, "verB"],
            n_obs_timepoints=n_obs,
            total_reads=int(cvec.sum()),
            max_count=int(cvec.max()),
            transfers_observed=",".join(str(t) for t, c in zip(transfers, cvec) if c > 0),
            freq_T3=fvec[0],
            freq_T13=fvec[-1],
            s_per_cycle=s_cycle,
            se_per_cycle=fit_cycle["slope_se"],
            r2_per_cycle=fit_cycle["r2"],
            s_per_generation=fit_gen["slope"],
            se_per_generation=fit_gen["slope_se"],
            s_per_hour_measured=s_cycle / C.HOURS_PER_TRANSFER_MEASURED,
            s_per_hour_nominal=s_cycle / C.HOURS_PER_TRANSFER_NOMINAL,
            monotonic=monotonic,
            quality=quality,
            estimable=(n_obs >= C.MIN_OBS_TIMEPOINTS),
        ))
    return results


def main():
    d4 = pd.read_csv(C.INTER / "barcode_4transfer_long.csv")
    axis = pd.read_csv(C.INTER / "od_time_axis.csv").set_index("transfer")

    rows = []
    for sample, sub in d4.groupby("Sample"):
        rows.extend(fit_one_sample(sub, axis))
    res = pd.DataFrame(rows)

    # Flag the dominant variant per Sample at the final transfer (sweep winner).
    res["sweep_winner"] = False
    for sample, sub in res.groupby("Sample"):
        win = sub.loc[sub["freq_T13"].idxmax(), "Candidate"]
        res.loc[(res.Sample == sample) & (res.Candidate == win), "sweep_winner"] = True

    res = res.sort_values(["Sample", "s_per_cycle"], ascending=[True, False])
    res.to_csv(C.OUT / "selection_coefficients_long.csv", index=False)

    est = res[res.estimable]
    print(f"[s03] {len(res)} (Sample,Candidate) pairs; "
          f"{est.shape[0]} estimable (>={C.MIN_OBS_TIMEPOINTS} timepoints), "
          f"{(~res.estimable).sum()} single-timepoint (non-estimable).")
    print(f"[s03] s_per_cycle range (estimable): "
          f"[{est.s_per_cycle.min():.3f}, {est.s_per_cycle.max():.3f}]")
    print("[s03] wrote selection_coefficients_long.csv")
    return res


if __name__ == "__main__":
    main()
