"""
s05 - Allele-level (verA / verB) marginal selection coefficients + neutral-drift null.

Per-Candidate coefficients are noisy and largely irreproducible across replicate
cultures (different full Candidates win each replicate). The reproducible signal
lives at the *allele* level: e.g. verA A81 dominates many cultures. This script:

  1. Collapses the estimable per-(Sample,Candidate) selection coefficients into a
     marginal coefficient per verA and per verB (inverse-variance weighted mean
     across all cultures in which the allele appears). CLR slopes are already
     gauge-centered within each Sample, so these marginals are centered.

  2. Builds a NEUTRAL-DRIFT NULL. For each Sample we resample read counts at the
     observed depths from a *constant* (time-averaged) frequency vector -- i.e.
     no fitness differences -- and recompute the allele marginals. This guards
     against the survivorship artifact whereby a variant's frequency rises at
     fixed depth merely because competitors drop below detection. An allele
     effect is "real" only if it exceeds the neutral envelope.

Outputs (outputs/):
  allele_effects_verA.csv, allele_effects_verB.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C

RNG = np.random.default_rng(0)
N_NULL = 300


def vectorized_clr_slopes(counts, x, pc=C.PSEUDOCOUNT):
    """Weighted CLR slope for every row of a (K variants x T) count matrix.
    Weights = count + pseudocount (Poisson precision proxy). Returns length-K
    slope vector. Vectorized over variants."""
    counts = np.asarray(counts, float)
    L = np.log(counts + pc)
    clr = L - L.mean(axis=0, keepdims=True)          # center each timepoint
    w = counts + pc
    x = np.asarray(x, float)[None, :]
    W = w.sum(axis=1, keepdims=True)
    xbar = (w * x).sum(axis=1, keepdims=True) / W
    ybar = (w * clr).sum(axis=1, keepdims=True) / W
    Sxx = (w * (x - xbar) ** 2).sum(axis=1)
    Sxy = (w * (x - xbar) * (clr - ybar)).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(Sxx > 0, Sxy / Sxx, np.nan)


def weighted_mean_se(values, weights):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    W = weights.sum()
    if W <= 0:
        return np.nan, np.nan
    m = np.sum(weights * values) / W
    # variance of a weighted mean (treats weights as inverse-variances)
    se = np.sqrt(1.0 / W)
    return m, se


def build_sample_grids(d4):
    """Return {Sample: (counts[KxT], depths[T], candidate_index, verA, verB)}."""
    transfers = list(C.FOUR_TRANSFER_SET)
    grids = {}
    for sample, sub in d4.groupby("Sample"):
        counts = (sub.pivot_table(index="Candidate", columns="Transfer",
                                  values="Count", aggfunc="sum", fill_value=0)
                  .reindex(columns=transfers, fill_value=0))
        meta = sub.drop_duplicates("Candidate").set_index("Candidate")
        depths = counts.sum(axis=0).to_numpy(float)
        grids[sample] = dict(
            counts=counts.to_numpy(float),
            depths=depths,
            cands=list(counts.index),
            verA=meta.loc[counts.index, "verA"].to_numpy(),
            verB=meta.loc[counts.index, "verB"].to_numpy(),
        )
    return grids


def allele_marginals(slopes_by_sample, grids, allele_key, weights_by_sample):
    """Collapse per-(sample,candidate) slopes into a weighted mean per allele.
    Returns dict allele -> (weighted_mean, sumW, n_pairs)."""
    acc = {}
    for sample, slopes in slopes_by_sample.items():
        alleles = grids[sample][allele_key]
        w = weights_by_sample[sample]
        for a, s, wi in zip(alleles, slopes, w):
            if not np.isfinite(s) or wi <= 0:
                continue
            m = acc.setdefault(a, [0.0, 0.0, 0])  # sum(w*s), sum(w), n
            m[0] += wi * s
            m[1] += wi
            m[2] += 1
    return {a: (v[0] / v[1], v[1], v[2]) for a, v in acc.items() if v[1] > 0}


def main():
    d4 = pd.read_csv(C.INTER / "barcode_4transfer_long.csv")
    res = pd.read_csv(C.OUT / "selection_coefficients_long.csv")
    grids = build_sample_grids(d4)
    x_cycle = np.array(C.FOUR_TRANSFER_SET, float)

    # Observed slopes + weights per (sample, candidate), keeping only estimable.
    est = res[res.estimable].copy()
    est["w"] = 1.0 / np.clip(est["se_per_cycle"], 1e-3, None) ** 2
    obs_slopes, obs_w = {}, {}
    for sample, g in grids.items():
        sl = res[res.Sample == sample].set_index("Candidate")
        slopes = np.array([sl.loc[c, "s_per_cycle"] for c in g["cands"]], float)
        # weight estimable pairs by inverse-variance; non-estimable -> 0
        wv = np.array([
            (1.0 / max(sl.loc[c, "se_per_cycle"], 1e-3) ** 2)
            if bool(sl.loc[c, "estimable"]) else 0.0
            for c in g["cands"]], float)
        obs_slopes[sample] = slopes
        obs_w[sample] = wv

    obs_A = allele_marginals(obs_slopes, grids, "verA", obs_w)
    obs_B = allele_marginals(obs_slopes, grids, "verB", obs_w)

    # ---- neutral-drift null --------------------------------------------------
    null_A = {a: [] for a in obs_A}
    null_B = {a: [] for a in obs_B}
    for _ in range(N_NULL):
        sim_slopes = {}
        for sample, g in grids.items():
            counts = g["counts"]
            p = counts.mean(axis=1)               # time-averaged freq (neutral)
            p = p / p.sum()
            sim = np.empty_like(counts)
            for j, D in enumerate(g["depths"]):    # resample each timepoint
                sim[:, j] = RNG.multinomial(int(round(D)), p)
            sim_slopes[sample] = vectorized_clr_slopes(sim, x_cycle)
        nA = allele_marginals(sim_slopes, grids, "verA", obs_w)
        nB = allele_marginals(sim_slopes, grids, "verB", obs_w)
        for a in null_A:
            if a in nA:
                null_A[a].append(nA[a][0])
        for a in null_B:
            if a in nB:
                null_B[a].append(nB[a][0])

    # ---- win rates (sweep winner per Sample) --------------------------------
    winners = res[res.sweep_winner]
    winA = winners["verA"].value_counts().to_dict()
    n_samples = res["Sample"].nunique()

    def assemble(obs, null, allele_key, win_counts):
        rows = []
        # how many distinct samples each allele appears in
        appears = (d4.groupby(allele_key)["Sample"].nunique()
                   if allele_key in d4 else {})
        for a, (m, sumW, npair) in obs.items():
            nd = np.array(null.get(a, []), float)
            if nd.size >= 10:
                p = float((np.abs(nd) >= abs(m)).mean())
                lo, hi = np.percentile(nd, [2.5, 97.5])
            else:
                p, lo, hi = np.nan, np.nan, np.nan
            rows.append(dict(
                allele=a,
                marginal_s_per_cycle=m,
                se=np.sqrt(1.0 / sumW),
                n_pairs=npair,
                n_samples=int(appears.get(a, np.nan)) if len(appears) else np.nan,
                sweep_wins=int(win_counts.get(a, 0)),
                drift_null_p=p,
                null_lo=lo, null_hi=hi,
                exceeds_drift_null=(np.isfinite(p) and p < 0.05),
                # robust = beats drift null AND backed by enough data to trust
                robust_selection=(np.isfinite(p) and p < 0.05
                                  and npair >= 5
                                  and int(appears.get(a, 0) or 0) >= 3),
            ))
        return (pd.DataFrame(rows)
                .sort_values("marginal_s_per_cycle", ascending=False))

    dfA = assemble(obs_A, null_A, "verA", winA)
    dfB = assemble(obs_B, null_B, "verB", {})
    dfA.to_csv(C.OUT / "allele_effects_verA.csv", index=False)
    dfB.to_csv(C.OUT / "allele_effects_verB.csv", index=False)

    sigA = dfA[dfA.exceeds_drift_null]
    print(f"[s05] verA: {len(dfA)} alleles, {len(sigA)} exceed the drift null (p<0.05).")
    top = dfA.head(3)[["allele", "marginal_s_per_cycle", "sweep_wins", "drift_null_p"]]
    print(f"[s05] top verA by marginal selection (of {n_samples} samples):")
    for _, r in top.iterrows():
        print(f"        {r.allele:6s} s={r.marginal_s_per_cycle:+.3f}/cycle  "
              f"wins={int(r.sweep_wins)}  drift_p={r.drift_null_p:.3f}")
    print("[s05] wrote allele_effects_verA.csv, allele_effects_verB.csv")
    return dfA, dfB


if __name__ == "__main__":
    main()
