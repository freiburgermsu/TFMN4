"""
s08 - Community bulk exponential-phase growth rate (mu_bulk) from the OD file.

For each (focus Sample, transfer in {3,4,6,13}) we fit the within-cycle net-OD
growth curve and report the maximum specific growth rate
    mu_bulk = max_t  d ln(OD_net)/dt        [per hour]
estimated as the steepest slope of ln(net_od) vs elapsed hours over a sliding
window of readings. By the replicator identity mu_bulk = sum_i f_i r_i = rbar,
the COMMUNITY-mean absolute rate -- the one quantity OD legitimately supplies.

We also report the exponential-phase clock
    tau_exp = delta_g * ln2 / mu_bulk       [hours]
(delta_g = generations/cycle from od_time_axis.csv), which is the CORRECT
per-hour denominator for selection coefficients (NOT the full ~26.4 h cycle).

Outputs (outputs/intermediate/):
  od_bulk_rate.csv              per (Sample, transfer): mu_bulk, se, K, tau_exp, flags
  od_bulk_condition_summary.csv community mu_bulk by construct family (concX/concY)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
import config as C

WINDOW = 3          # readings per sliding log-slope window
RISE_LO, RISE_HI = 0.05, 0.85   # net-OD band (fraction of plateau) counted as "rising"


def mu_max_window(t, net):
    """Max specific growth rate via steepest sliding-window slope of ln(net)."""
    t = np.asarray(t, float)
    y = np.log(np.clip(net, 1e-4, None))
    best = dict(mu=np.nan, se=np.nan, r2=np.nan, t_mid=np.nan)
    for i in range(len(t) - WINDOW + 1):
        sl = slice(i, i + WINDOW)
        if np.ptp(t[sl]) <= 0:
            continue
        lr = stats.linregress(t[sl], y[sl])
        if not np.isfinite(lr.slope):
            continue
        if np.isnan(best["mu"]) or lr.slope > best["mu"]:
            best = dict(mu=lr.slope, se=lr.stderr, r2=lr.rvalue ** 2,
                        t_mid=float(np.mean(t[sl])))
    return best


def main():
    wm = pd.read_csv(C.INTER / "sample_well_map.csv")
    sample_rows = wm[["Sample", "DNA_construct", "Microtiter_plate_well"]].drop_duplicates()
    focus = set(wm["Microtiter_plate_well"].astype(str))

    axis = pd.read_csv(C.INTER / "od_time_axis.csv").set_index("transfer")
    dg = axis["delta_g"].to_dict()
    dg_med = float(np.nanmedian(list(dg.values())))

    od = C.read_focus_od(["transfer", "reading", "datetime", "od", "background"], focus)
    od["well"] = od["Microtiter_plate_well"].astype(str)
    od["rn"] = pd.to_numeric(od["reading"], errors="coerce")   # 'contam' -> NaN
    od = od.dropna(subset=["rn"])
    od["dt"] = pd.to_datetime(od["datetime"], errors="coerce")
    od["net"] = (od["od"] - od["background"]).clip(lower=1e-4)

    # per (well, transfer) within-cycle exponential rate, computed once
    well_rate = {}
    for (w, tr), g in od.groupby(["well", "transfer"]):
        if tr not in C.TRANSFERS:
            continue
        g = g.sort_values("rn")
        t = (g["dt"] - g["dt"].min()).dt.total_seconds().to_numpy() / 3600.0
        net = g["net"].to_numpy(float)
        K = float(np.nanpercentile(net, 95))
        n_band = int(((net > RISE_LO) & (net < RISE_HI * K)).sum())
        fit = mu_max_window(t, net)
        mu = fit["mu"]
        delta_g = float(dg.get(tr, dg_med))
        tau = delta_g * np.log(2) / mu if (mu and mu > 0) else np.nan
        reliable = bool(n_band >= WINDOW and np.isfinite(mu) and mu > 0 and K > 0.1
                        and np.isfinite(tau) and tau < C.HOURS_PER_TRANSFER_MEASURED)
        well_rate[(w, int(tr))] = dict(
            mu_bulk_per_h=round(mu, 4) if np.isfinite(mu) else np.nan,
            mu_se=round(fit["se"], 4) if np.isfinite(fit["se"]) else np.nan,
            fit_r2=round(fit["r2"], 3) if np.isfinite(fit["r2"]) else np.nan,
            plateau_K=round(K, 3), inoculum_net=round(float(net[0]), 4),
            n_band_readings=n_band,
            doubling_h=round(np.log(2) / mu, 2) if (mu and mu > 0) else np.nan,
            tau_exp_h=round(tau, 2) if np.isfinite(tau) else np.nan,
            reliable=reliable)

    # expand to every Sample sharing that well (e.g. the 3 amplicon primerset reps)
    def family(dc):
        return "concX" if "concX" in str(dc) else ("concY" if "concY" in str(dc) else str(dc))
    rows = []
    for sr in sample_rows.itertuples():
        w = str(sr.Microtiter_plate_well)
        for tr in C.TRANSFERS:
            wr = well_rate.get((w, tr))
            if wr is None:
                continue
            rows.append(dict(Sample=sr.Sample, construct_family=family(sr.DNA_construct),
                             transfer=tr, **wr))
    bulk = pd.DataFrame(rows).sort_values(["Sample", "transfer"])
    bulk.to_csv(C.INTER / "od_bulk_rate.csv", index=False)

    # condition-level community rate at the reliable transfers (6, 13)
    rel = bulk[bulk.reliable & bulk.transfer.isin(list(C.RELIABLE_OD_TRANSFERS))]
    summ = (rel.groupby("construct_family")["mu_bulk_per_h"]
            .agg(n="size", median="median", mean="mean", sd="std",
                 min="min", max="max").reset_index())
    summ["doubling_h_at_median"] = (np.log(2) / summ["median"]).round(2)
    summ = summ.round(4)
    summ.to_csv(C.INTER / "od_bulk_condition_summary.csv", index=False)

    print(f"[s08] mu_bulk per (Sample,transfer): {bulk.reliable.sum()}/{len(bulk)} reliable "
          f"(T3/T4 mostly unreliable as expected).")
    print(f"[s08] reliable mu_bulk median={rel['mu_bulk_per_h'].median():.3f}/h "
          f"(doubling {np.log(2)/rel['mu_bulk_per_h'].median():.2f} h); "
          f"tau_exp median={rel['tau_exp_h'].median():.1f} h")
    print("[s08] community mu_bulk by construct (the only honestly-absolute OD/h number):")
    for _, r in summ.iterrows():
        print(f"        {r.construct_family}: median {r['median']:.3f}/h (n={int(r['n'])})")
    print("[s08] wrote od_bulk_rate.csv, od_bulk_condition_summary.csv")
    return bulk, summ


if __name__ == "__main__":
    main()
