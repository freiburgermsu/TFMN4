"""
s12 - Global static per-variant growth-rate optimization.

Goal (user request): assign each variant (Candidate) ONE mostly-static growth rate
r_i shared across ALL samples, and solve for the combination {r_i} that best
reproduces, in every sample, both (a) the observed relative read counts at each
transfer and (b) the observed community OD growth at each transfer. Per the stated
assumption, a variant's biomass is its read fraction times the community OD.

FORWARD MODEL (softmax replicator with an OD-mean anchor)
  Time axis x_t = cumulative generations (cumgen) converted to exponential-phase
  hours via the global community doubling time:  x_t = cumgen_t * ln2 / mu_bar.
  Latent log-abundance:  eta_{c,i,t} = logA_{c,i} + r_i * x_t
     r_i        : GLOBAL per-variant rate  (per hour)         <- the deliverable
     logA_{c,i} : per-(sample,variant) log initial abundance  (nuisance)
  Predicted frequency:  f_{c,i,t} = softmax over variants in sample c of eta.
  Predicted community growth rate:  rbar_{c,t} = sum_i f_{c,i,t} * r_i.

LOSS
  L = multinomial NLL(observed counts | f)                      # recreate counts
    + w_OD * sum_{reliable (c,t)} (rbar_{c,t} - mu_bulk_{c,t})^2 / sig_OD^2   # recreate OD growth
    + w_r  * sum_i (r_i - mu_bar)^2                              # "mostly static / minimally flexible"
    + w_A  * sum logA^2                                          # conditioning

Only the OD term carries absolute information (the count term is gauge-free); the
ridge keeps rates near the community rate unless the data demand otherwise. So the
absolute level is community-dominated by construction -- see the identifiability
notes printed at the end and in docs.

ERROR in the rates is quantified four ways (see s12b): Laplace SEs (this file),
bootstrap CIs, leave-one-sample-out heterogeneity, and prior/OD-weight sensitivity.

Output (outputs/): global_variant_growth_rates.csv  (+ Laplace SE here; CIs in s12b)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.ops import segment_sum
from scipy.optimize import minimize
import config as C

LN2 = float(np.log(2.0))
W_OD = 5.0       # weight on the OD-anchor term
W_R = 50.0       # ridge toward community mean (minimal flexibility)
W_A = 0.05       # conditioning: pull logA toward its data value (fixes flat dirs)
SIG_OD = 0.08    # per-hour SD of a mu_bulk measurement


def build_design():
    """Flatten the data into ragged-but-vectorizable arrays for jax."""
    d4 = pd.read_csv(C.INTER / "barcode_4transfer_long.csv")
    ax = pd.read_csv(C.INTER / "od_time_axis.csv").set_index("transfer")["cumgen"].to_dict()
    bulk = pd.read_csv(C.INTER / "od_bulk_rate.csv")
    mu_bar = float(bulk.loc[bulk.reliable & bulk.transfer.isin([6, 13]), "mu_bulk_per_h"].median())
    mub = {(r.Sample, int(r.transfer)): r.mu_bulk_per_h
           for r in bulk[bulk.reliable].itertuples()}

    transfers = list(C.FOUR_TRANSFER_SET)
    variants = sorted(d4["Candidate"].unique())
    vidx = {v: k for k, v in enumerate(variants)}
    samples = sorted(d4["Sample"].unique())

    # per-sample variant union (variants ever seen in that sample)
    union = {s: sorted(g["Candidate"].unique()) for s, g in d4.groupby("Sample")}
    # (sample,variant) -> logA index
    saidx, sa_pairs = {}, []
    for s in samples:
        for v in union[s]:
            saidx[(s, v)] = len(sa_pairs); sa_pairs.append((s, v))

    counts = (d4.pivot_table(index=["Sample", "Candidate"], columns="Transfer",
                             values="Count", aggfunc="sum", fill_value=0))
    # rows: one per (sample, transfer, variant-in-union); cells grouped for softmax
    rows = dict(var=[], logA=[], cell=[], cnt=[], x=[])
    cells = dict(sample=[], transfer=[], mu=[], reliable=[], x=[])
    cell_id = {}
    hours_per_gen = LN2 / mu_bar
    for s in samples:
        for t in transfers:
            cid = len(cells["sample"])
            cell_id[(s, t)] = cid
            cells["sample"].append(s); cells["transfer"].append(t)
            cells["mu"].append(mub.get((s, t), np.nan))
            cells["reliable"].append((s, t) in mub)
            cells["x"].append(ax[t] * hours_per_gen)
            for v in union[s]:
                cnt = counts.loc[(s, v), t] if (s, v) in counts.index else 0
                rows["var"].append(vidx[v]); rows["logA"].append(saidx[(s, v)])
                rows["cell"].append(cid); rows["cnt"].append(float(cnt))
                rows["x"].append(ax[t] * hours_per_gen)
    D = {k: jnp.asarray(np.array(v, dtype=float)) for k, v in rows.items()}
    D["var"] = D["var"].astype(int); D["logA"] = D["logA"].astype(int); D["cell"] = D["cell"].astype(int)
    cell_mu = jnp.asarray(np.array(cells["mu"], float))
    cell_rel = jnp.asarray(np.array(cells["reliable"], bool))
    n_cells = len(cells["sample"])
    mean_freq = d4.groupby(["Sample", "Candidate"])["freq"].mean()
    logA0 = np.array([np.log(max(mean_freq.get((s, v), 1e-3), 1e-3)) for (s, v) in sa_pairs])
    meta = dict(variants=variants, vidx=vidx, samples=samples, union=union,
                sa_pairs=sa_pairs, mu_bar=mu_bar, n_cells=n_cells,
                cells=cells, transfers=transfers, logA0=logA0)
    return D, cell_mu, cell_rel, meta


def make_loss(D, cell_mu, cell_rel, mu_bar, n_var, n_cells, logA0):
    cell = D["cell"]; var = D["var"]; logA_idx = D["logA"]; x = D["x"]; cnt = D["cnt"]
    rel_mu = jnp.where(cell_rel, cell_mu, 0.0)
    logA0 = jnp.asarray(logA0)

    def loss(params):
        r = params[:n_var]
        logA = params[n_var:]
        eta = logA[logA_idx] + r[var] * x
        # softmax per cell (numerically stable)
        m = segment_max(eta, cell, n_cells)
        ex = jnp.exp(eta - m[cell])
        Z = segment_sum(ex, cell, num_segments=n_cells)
        logf = eta - m[cell] - jnp.log(Z[cell])
        f = ex / Z[cell]
        L_count = -jnp.sum(cnt * logf)
        rbar = segment_sum(f * r[var], cell, num_segments=n_cells)
        L_od = W_OD * jnp.sum(jnp.where(cell_rel, (rbar - rel_mu) ** 2, 0.0)) / SIG_OD ** 2
        L_r = W_R * jnp.sum((r - mu_bar) ** 2)
        L_a = W_A * jnp.sum((logA - logA0) ** 2)   # conditions flat single-obs dirs
        return L_count + L_od + L_r + L_a
    return loss


def segment_max(data, seg, n):
    # jax has no segment_max in older versions; emulate via scatter-max
    return jax.ops.segment_max(data, seg, num_segments=n)


def main():
    D, cell_mu, cell_rel, meta = build_design()
    n_var = len(meta["variants"]); n_A = len(meta["sa_pairs"])
    mu_bar = meta["mu_bar"]

    logA0 = meta["logA0"]   # data-driven logA anchor (mean observed log-frequency)
    loss = make_loss(D, cell_mu, cell_rel, mu_bar, n_var, meta["n_cells"], logA0)
    loss_jit = jax.jit(loss)
    grad_jit = jax.jit(jax.grad(loss))

    p0 = np.concatenate([np.full(n_var, mu_bar), logA0])

    def f(p): return float(loss_jit(jnp.asarray(p)))
    def g(p): return np.asarray(grad_jit(jnp.asarray(p)), float)
    # two L-BFGS passes (warm restart) for clean convergence
    p = p0
    for _ in range(2):
        res = minimize(f, p, jac=g, method="L-BFGS-B",
                       options=dict(maxiter=5000, maxfun=50000, ftol=1e-12, gtol=1e-8))
        p = res.x
    gnorm = float(np.linalg.norm(g(p)))
    r = p[:n_var]

    # ---- Laplace SE on r (full Hessian, marginalize logA via Schur complement) ----
    H = np.asarray(jax.hessian(loss)(jnp.asarray(p)), float)
    Hrr, HrA, HAA = H[:n_var, :n_var], H[:n_var, n_var:], H[n_var:, n_var:]
    # Schur complement: cov(r) = (Hrr - HrA HAA^-1 HAr)^-1
    try:
        HAA_inv_HAr = np.linalg.solve(HAA + 1e-8 * np.eye(n_A), HrA.T)
        S = Hrr - HrA @ HAA_inv_HAr
        cov_r = np.linalg.inv(S + 1e-8 * np.eye(n_var))
        se = np.sqrt(np.clip(np.diag(cov_r), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(n_var, np.nan)

    # ---- fit diagnostics: predicted vs observed frequency + community rate ----
    diag = fit_diagnostics(p, D, cell_mu, cell_rel, meta)

    d4 = pd.read_csv(C.INTER / "barcode_4transfer_long.csv")
    nspc = d4.groupby("Candidate")["Sample"].nunique()
    cov = d4[d4.Count > 0].groupby(["Candidate"])["Transfer"].apply(lambda s: 0)  # placeholder
    est = (d4[d4.Count > 0].groupby(["Sample", "Candidate"])["Transfer"].nunique() >= 2)
    est_var = set(est[est].reset_index()["Candidate"])
    meta_v = d4.drop_duplicates("Candidate").set_index("Candidate")

    out = pd.DataFrame({
        "Candidate": meta["variants"],
        "r_global_per_h": np.round(r, 4),
        "se_laplace": np.round(se, 4),
        "doubling_h": np.round(LN2 / np.where(r > 0, r, np.nan), 2),
    })
    out["verA"] = out.Candidate.map(meta_v["verA"]); out["verB"] = out.Candidate.map(meta_v["verB"])
    out["n_samples"] = out.Candidate.map(nspc)
    out["has_slope_info"] = out.Candidate.isin(est_var)
    out["rel_advantage_per_h"] = np.round(r - mu_bar, 4)
    out = out[["Candidate", "verA", "verB", "n_samples", "has_slope_info",
               "r_global_per_h", "se_laplace", "rel_advantage_per_h", "doubling_h"]]
    out = out.sort_values("r_global_per_h", ascending=False)
    out.to_csv(C.OUT / "global_variant_growth_rates.csv", index=False)

    print(f"[s12] optimize: success={res.success} final_loss={res.fun:.1f} "
          f"grad_norm={gnorm:.2e} ({n_var} global rates + {n_A} intercepts)")
    print(f"[s12] community mu_bar={mu_bar:.3f}/h; fitted r range "
          f"[{r.min():.3f},{r.max():.3f}]/h; median SE={np.nanmedian(se):.4f}")
    print(f"[s12] FIT QUALITY: freq pseudo-R2={diag['freq_r2']:.3f}; "
          f"community-rate (rbar vs mu_bulk) RMSE={diag['od_rmse']:.3f}/h on {diag['n_od']} cells")
    top = out[out.has_slope_info].head(4)
    print("[s12] fastest variants (with slope info):")
    for _, x in top.iterrows():
        print(f"        {x.Candidate:10s} r={x.r_global_per_h:.3f}±{x.se_laplace:.3f}/h "
              f"(rel {x.rel_advantage_per_h:+.3f})")
    print("[s12] wrote global_variant_growth_rates.csv")
    return out, diag


def fit_diagnostics(p, D, cell_mu, cell_rel, meta):
    n_var = len(meta["variants"])
    r = p[:n_var]; logA = p[n_var:]
    var = np.asarray(D["var"]); cell = np.asarray(D["cell"]); x = np.asarray(D["x"])
    cnt = np.asarray(D["cnt"]); logA_idx = np.asarray(D["logA"])
    eta = logA[logA_idx] + r[var] * x
    nc = meta["n_cells"]
    fmod = np.zeros_like(eta); rbar = np.zeros(nc)
    for cid in range(nc):
        msk = cell == cid
        e = eta[msk]; e = e - e.max()
        fm = np.exp(e) / np.exp(e).sum()
        fmod[msk] = fm
        rbar[cid] = float((fm * r[var[msk]]).sum())
    tot = np.zeros(nc); np.add.at(tot, cell, cnt)
    fobs = cnt / np.where(tot[cell] > 0, tot[cell], 1)
    mask = tot[cell] > 0
    ss_res = np.sum((fobs[mask] - fmod[mask]) ** 2)
    ss_tot = np.sum((fobs[mask] - fobs[mask].mean()) ** 2)
    freq_r2 = 1 - ss_res / ss_tot
    cm = np.asarray(cell_mu); rel = np.asarray(cell_rel)
    od_rmse = float(np.sqrt(np.mean((rbar[rel] - cm[rel]) ** 2)))
    return dict(freq_r2=float(freq_r2), od_rmse=od_rmse, n_od=int(rel.sum()))


if __name__ == "__main__":
    main()
