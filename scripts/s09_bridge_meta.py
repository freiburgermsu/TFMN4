"""
s09 - Cross-sample integration: put all 11 samples on ONE common relative scale.

Each sample's selection coefficients s_{c,i} = r_{c,i} - rbar_c are gauged to that
sample's own community mean. We tie the gauges together with a two-way model

    s_{c,i} = beta_i + gamma_c + e_{c,i},   e ~ (0, phi * se_{c,i}^2)

  beta_i  = variant effect on a common cross-sample RELATIVE scale (the deliverable)
  gamma_c = per-sample gauge nuisance (pinned by variants SHARED across samples)

Identifiable up to one global constant because the variant-overlap (bridge) graph
is connected. Fit by weighted alternating least squares (robust to the sparse,
mostly-single-sample design); gamma_c is constrained to sum to zero. SEs are
inflated by the observed overdispersion phi = weighted RSS / dof.

Outputs (outputs/):
  variant_bridged_relative.csv  per-variant beta_i (+CI, I^2, bridge flag)
  sample_gauge.csv              per-sample gamma_c + bridge degree + anchor flag
"""
from __future__ import annotations
import sys
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C


def connected_components(samples, shared):
    parent = {s: s for s in samples}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in shared:
        parent[find(a)] = find(b)
    return len({find(s) for s in samples})


def main():
    res = pd.read_csv(C.OUT / "selection_coefficients_long.csv")
    est = res[res.estimable].copy()
    est["w"] = 1.0 / np.clip(est["se_per_cycle"], 1e-3, None) ** 2

    samples = sorted(est["Sample"].unique())
    cands = sorted(est["Candidate"].unique())
    nspc = est.groupby("Candidate")["Sample"].nunique()
    bridge = set(nspc[nspc >= 2].index)

    # variant-overlap graph: degree + connectivity
    vs = est.groupby("Sample")["Candidate"].apply(set).to_dict()
    shared_edges = [(a, b) for a, b in itertools.combinations(samples, 2) if vs[a] & vs[b]]
    n_comp = connected_components(samples, shared_edges)
    degree = {s: sum(1 for o in samples if o != s and (vs[s] & vs[o])) for s in samples}
    assert n_comp == 1, f"bridge graph not connected ({n_comp} components) -- gauges not tied"

    # records grouped for the alternating fit
    by_cand = {i: g[["Sample", "s_per_cycle", "w"]].values for i, g in est.groupby("Candidate")}
    by_samp_bridge = {c: est[(est.Sample == c) & (est.Candidate.isin(bridge))][["Candidate", "s_per_cycle", "w"]].values
                      for c in samples}

    gamma = {c: 0.0 for c in samples}
    beta = {i: 0.0 for i in cands}
    for _ in range(2000):
        for i, arr in by_cand.items():            # beta_i = wmean(s - gamma)  (all variants)
            sc, s, w = arr[:, 0], arr[:, 1].astype(float), arr[:, 2].astype(float)
            beta[i] = np.sum(w * (s - np.array([gamma[c] for c in sc]))) / np.sum(w)
        new_gamma = {}
        for c, arr in by_samp_bridge.items():     # gamma_c pinned by bridge variants only
            if len(arr) == 0:
                new_gamma[c] = gamma[c]
                continue
            ci, s, w = arr[:, 0], arr[:, 1].astype(float), arr[:, 2].astype(float)
            new_gamma[c] = np.sum(w * (s - np.array([beta[i] for i in ci]))) / np.sum(w)
        mean_g = np.mean(list(new_gamma.values()))  # sum-to-zero gauge
        new_gamma = {c: v - mean_g for c, v in new_gamma.items()}
        delta = max(abs(new_gamma[c] - gamma[c]) for c in samples)
        gamma = new_gamma
        if delta < 1e-9:
            break

    # overdispersion phi from bridge-variant residuals
    rss = n_obs = 0.0
    for c, arr in by_samp_bridge.items():
        for ci, s, w in arr:
            rss += w * (s - beta[ci] - gamma[c]) ** 2
            n_obs += 1
    dof = max(n_obs - (len(bridge) + len(samples) - 1), 1)
    phi = rss / dof

    # per-cycle -> per-generation conversion factor (generations per transfer-cycle)
    ax = pd.read_csv(C.INTER / "od_time_axis.csv").set_index("transfer")["cumgen"]
    gpc = (ax[13] - ax[3]) / (13 - 3)

    rows = []
    for i in cands:
        arr = by_cand[i]
        sc, s, w = arr[:, 0], arr[:, 1].astype(float), arr[:, 2].astype(float)
        W = np.sum(w)
        b = beta[i]
        se_raw = np.sqrt(1.0 / W)        # sampling SE (phi=1)
        se = np.sqrt(phi / W)            # overdispersion-inflated SE for CIs
        # heterogeneity across samples (multi-sample variants only)
        if len(s) >= 2:
            Q = np.sum(w * (s - np.array([gamma[c] for c in sc]) - b) ** 2)
            dfh = len(s) - 1
            I2 = max(0.0, (Q - dfh) / Q) if Q > 0 else 0.0
        else:
            I2 = np.nan
        meta = est[est.Candidate == i].iloc[0]
        rows.append(dict(
            Candidate=i, verA=meta.verA, verB=meta.verB,
            n_samples=int(len(s)), bridged=(i in bridge),
            beta_per_cycle=round(b, 4), beta_se=round(se, 4), beta_se_raw=round(se_raw, 4),
            beta_per_generation=round(b / gpc, 4),
            I2_heterogeneity=round(I2, 3) if np.isfinite(I2) else np.nan,
            samples=";".join(sorted(c.replace("TFMN4.exp2.ACN3788.", "") for c in sc)),
        ))
    var = pd.DataFrame(rows).sort_values("beta_per_cycle", ascending=False)
    var.to_csv(C.OUT / "variant_bridged_relative.csv", index=False)

    nbridge_c = {c: int(sum(1 for i in bridge if c in {x for x in est[est.Candidate == i].Sample}))
                 for c in samples}
    gdf = pd.DataFrame([dict(
        Sample=c, gamma_c=round(gamma[c], 4), bridge_degree=degree[c],
        n_bridge_variants=nbridge_c[c], weakly_anchored=(degree[c] <= 1),
    ) for c in samples]).sort_values("gamma_c")
    gdf.to_csv(C.OUT / "sample_gauge.csv", index=False)

    print(f"[s09] bridge graph: 1 connected component, {len(shared_edges)} edges, "
          f"{len(bridge)} bridge variants; gen/cycle={gpc:.2f}")
    weak = [c.replace('TFMN4.exp2.ACN3788.', '') for c in samples if degree[c] <= 1]
    print(f"[s09] overdispersion phi = {phi:.2f}  (SEs inflated by {np.sqrt(phi):.2f}x); "
          f"weakly-anchored: {weak}")
    print(f"[s09] beta_per_cycle range [{var.beta_per_cycle.min():+.3f},{var.beta_per_cycle.max():+.3f}]; "
          f"{var.bridged.sum()} bridged / {len(var)} variants")
    print("[s09] wrote variant_bridged_relative.csv, sample_gauge.csv")
    return var, gdf


if __name__ == "__main__":
    main()
