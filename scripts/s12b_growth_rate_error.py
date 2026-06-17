"""
s12b - Error / uncertainty in the global per-variant growth rates from s12.

Four complementary error layers:
  1. Laplace SE             (computed in s12; carried through)
  2. Bootstrap CI           resample multinomial counts at observed depths AND
                            jitter each sample's mu_bulk within its noise -> refit
                            -> percentile CI per r_i (sampling + OD-anchor error)
  3. LOSO heterogeneity     leave-one-sample-out refits -> SD of r_i across folds
                            = "is the rate really static across samples?" error
  4. Weight sensitivity     vary the OD-anchor weight and the ridge -> how much each
                            r_i (and the absolute level) moves = data-pinned vs prior

Output (outputs/): global_variant_growth_rates.csv  (augmented in place)
                   figures/global_growth_fit.png
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
from jax.ops import segment_sum, segment_max
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from s12_global_growth_fit import build_design, LN2, W_OD, W_R, W_A, SIG_OD

RNG = np.random.default_rng(0)
N_BOOT = 60


def main():
    D, cell_mu, cell_rel, meta = build_design()
    n_var = len(meta["variants"]); mu_bar = meta["mu_bar"]; nc = meta["n_cells"]
    logA0 = jnp.asarray(meta["logA0"])
    var = np.asarray(D["var"]); cell = np.asarray(D["cell"]); xrow = np.asarray(D["x"])
    logA_idx = np.asarray(D["logA"]); cnt0 = np.asarray(D["cnt"])
    cmu = np.asarray(cell_mu); crel = np.asarray(cell_rel)
    sample_of_cell = np.array(meta["cells"]["sample"])
    sample_of_row = sample_of_cell[cell]

    varj = jnp.asarray(var); cellj = jnp.asarray(cell); xj = jnp.asarray(xrow)
    logAj = jnp.asarray(logA_idx); crelj = jnp.asarray(crel); cmuj = jnp.asarray(np.nan_to_num(cmu))

    def loss(params, cnt, row_keep, cell_keep, w_od, w_r):
        r = params[:n_var]; logA = params[n_var:]
        eta = logA[logAj] + r[varj] * xj
        m = segment_max(eta, cellj, num_segments=nc)
        ex = jnp.exp(eta - m[cellj])
        Z = segment_sum(ex, cellj, num_segments=nc)
        logf = eta - m[cellj] - jnp.log(Z[cellj])
        f = ex / Z[cellj]
        L_count = -jnp.sum(row_keep * cnt * logf)
        rbar = segment_sum(f * r[varj], cellj, num_segments=nc)
        L_od = w_od * jnp.sum(cell_keep * crelj * (rbar - cmuj) ** 2) / SIG_OD ** 2
        L_r = w_r * jnp.sum((r - mu_bar) ** 2)
        L_a = W_A * jnp.sum((logA - logA0) ** 2)
        return L_count + L_od + L_r + L_a

    vg = jax.jit(jax.value_and_grad(loss))
    n_par = n_var + len(meta["sa_pairs"])

    def solve(cnt, row_keep, cell_keep, w_od, w_r, p_init, maxiter=400):
        cnt = jnp.asarray(cnt); rk = jnp.asarray(row_keep); ck = jnp.asarray(cell_keep)
        wo = jnp.asarray(float(w_od)); wr = jnp.asarray(float(w_r))
        def fn(p):
            v, g = vg(jnp.asarray(p), cnt, rk, ck, wo, wr)
            return float(v), np.asarray(g, float)
        res = minimize(fn, p_init, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, ftol=1e-11, gtol=1e-7))
        return res.x

    ones_row = np.ones_like(cnt0); ones_cell = np.ones(nc)
    p_map = solve(cnt0, ones_row, ones_cell, W_OD, W_R,
                  np.concatenate([np.full(n_var, mu_bar), np.asarray(logA0)]), maxiter=3000)
    r_map = p_map[:n_var]

    # ---- (2) bootstrap: resample counts per cell + jitter mu_bulk ----
    cell_depth = np.zeros(nc); np.add.at(cell_depth, cell, cnt0)
    fobs = cnt0 / np.where(cell_depth[cell] > 0, cell_depth[cell], 1)
    rows_by_cell = [np.where(cell == c)[0] for c in range(nc)]
    mu_se = 0.05
    boot = np.full((N_BOOT, n_var), np.nan)
    for b in range(N_BOOT):
        cntb = cnt0.copy()
        for c in range(nc):
            idx = rows_by_cell[c]; D_c = int(round(cell_depth[c]))
            if D_c > 0 and len(idx) > 0:
                p = fobs[idx]; p = p / p.sum()
                cntb[idx] = RNG.multinomial(D_c, p)
        cmu_b = cmu.copy()
        cmu_b[crel] = cmu[crel] + RNG.normal(0, mu_se, size=int(crel.sum()))
        # rebuild loss closure with jittered mu via a temp (cmuj is captured) -> use w_od path
        cmuj_local = jnp.asarray(np.nan_to_num(cmu_b))
        def lossb(params, cnt, wo, wr):
            r = params[:n_var]; logA = params[n_var:]
            eta = logA[logAj] + r[varj] * xj
            m = segment_max(eta, cellj, num_segments=nc); ex = jnp.exp(eta - m[cellj])
            Z = segment_sum(ex, cellj, num_segments=nc); logf = eta - m[cellj] - jnp.log(Z[cellj])
            f = ex / Z[cellj]
            Lc = -jnp.sum(cnt * logf)
            rbar = segment_sum(f * r[varj], cellj, num_segments=nc)
            Lod = wo * jnp.sum(crelj * (rbar - cmuj_local) ** 2) / SIG_OD ** 2
            return Lc + Lod + wr * jnp.sum((r - mu_bar) ** 2) + W_A * jnp.sum((logA - logA0) ** 2)
        vgb = jax.jit(jax.value_and_grad(lossb))
        def fnb(p):
            v, g = vgb(jnp.asarray(p), jnp.asarray(cntb), jnp.asarray(float(W_OD)), jnp.asarray(float(W_R)))
            return float(v), np.asarray(g, float)
        rb = minimize(fnb, p_map, jac=True, method="L-BFGS-B",
                      options=dict(maxiter=300, ftol=1e-10, gtol=1e-6)).x
        boot[b] = rb[:n_var]
    ci_lo = np.nanpercentile(boot, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot, 97.5, axis=0)
    boot_sd = np.nanstd(boot, axis=0)

    # ---- (3) leave-one-sample-out heterogeneity ----
    samples = meta["samples"]
    loso = np.full((len(samples), n_var), np.nan)
    for k, s in enumerate(samples):
        rk = (sample_of_row != s).astype(float)
        ck = (sample_of_cell != s).astype(float)
        rl = solve(cnt0, rk, ck, W_OD, W_R, p_map, maxiter=300)
        loso[k] = rl[:n_var]
    loso_sd = np.nanstd(loso, axis=0)

    # ---- (4) weight sensitivity (data-pinned vs prior/OD-driven) ----
    r_od0 = solve(cnt0, ones_row, ones_cell, 0.0, W_R, p_map, maxiter=800)[:n_var]   # no OD
    r_odH = solve(cnt0, ones_row, ones_cell, 20.0, W_R, p_map, maxiter=800)[:n_var]  # strong OD
    sens_od = np.abs(r_odH - r_od0)        # movement of r_i with the OD anchor
    level_shift = float(np.mean(r_odH) - np.mean(r_od0))

    # ---- merge into the s12 table ----
    g = pd.read_csv(C.OUT / "global_variant_growth_rates.csv").set_index("Candidate")
    vorder = meta["variants"]
    add = pd.DataFrame({
        "Candidate": vorder,
        "ci_boot_lo": np.round(ci_lo, 4), "ci_boot_hi": np.round(ci_hi, 4),
        "se_boot": np.round(boot_sd, 4),
        "loso_heterogeneity_sd": np.round(loso_sd, 4),
        "sens_to_OD_weight": np.round(sens_od, 4),
    }).set_index("Candidate")
    out = g.join(add).reset_index()
    # identifiability: small movement with OD weight => data-pinned relative rate
    out["identifiability"] = np.where(out["sens_to_OD_weight"] < 0.02, "data-pinned",
                              np.where(out["sens_to_OD_weight"] < 0.05, "mixed", "anchor/prior-driven"))
    out = out.sort_values("r_global_per_h", ascending=False)
    out.to_csv(C.OUT / "global_variant_growth_rates.csv", index=False)

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    # (a) Laplace vs bootstrap SE
    a = ax[0, 0]
    a.scatter(out["se_laplace"], out["se_boot"], s=12, alpha=0.5)
    lim = np.nanmax([out["se_laplace"].max(), out["se_boot"].max()])
    a.plot([0, lim], [0, lim], "k--", lw=0.8)
    a.set(xlabel="Laplace SE (per h)", ylabel="bootstrap SE (per h)",
          title="(a) Laplace vs bootstrap SE agreement")
    # (b) forest of top/bottom variants with bootstrap CI
    sl = out[out.has_slope_info].sort_values("r_global_per_h")
    pick = pd.concat([sl.head(8), sl.tail(8)])
    y = np.arange(len(pick))
    a = ax[0, 1]
    xerr_lo = (pick["r_global_per_h"] - pick["ci_boot_lo"]).clip(lower=0)
    xerr_hi = (pick["ci_boot_hi"] - pick["r_global_per_h"]).clip(lower=0)
    a.errorbar(pick["r_global_per_h"], y, xerr=[xerr_lo, xerr_hi],
               fmt="o", capsize=2, color="#1f77b4")
    a.axvline(mu_bar, color="r", ls="--", lw=1, label=f"community μ_bar={mu_bar:.2f}")
    a.set_yticks(y); a.set_yticklabels(pick["Candidate"], fontsize=6)
    a.set(xlabel="r_global (per h)", title="(b) rates ± bootstrap 95% CI"); a.legend(fontsize=7)
    # (c) LOSO heterogeneity vs rate
    a = ax[1, 0]
    sc = a.scatter(out["r_global_per_h"], out["loso_heterogeneity_sd"],
                   c=out["n_samples"], cmap="viridis", s=14)
    a.set(xlabel="r_global (per h)", ylabel="LOSO heterogeneity SD (per h)",
          title="(c) cross-sample stability ('is it static?')"); fig.colorbar(sc, ax=a, label="n_samples")
    # (d) sensitivity to OD weight
    a = ax[1, 1]
    a.hist(out["sens_to_OD_weight"], bins=30, color="#888")
    a.axvline(0.02, color="g", ls="--", label="data-pinned <0.02")
    a.set(xlabel="|Δr| from OD weight 0→20 (per h)", ylabel="# variants",
          title=f"(d) identifiability; mean abs-level shift={level_shift:+.3f}/h"); a.legend(fontsize=7)
    fig.suptitle("Global static growth rates: error & identifiability", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(C.FIG / "global_growth_fit.png", dpi=135, bbox_inches="tight")
    plt.close(fig)

    print(f"[s12b] bootstrap B={N_BOOT}; median boot SE={np.nanmedian(boot_sd):.4f}/h, "
          f"median Laplace SE={out['se_laplace'].median():.4f}/h")
    print(f"[s12b] LOSO heterogeneity median={np.nanmedian(loso_sd):.4f}/h "
          f"(max={np.nanmax(loso_sd):.3f})")
    print(f"[s12b] absolute-level shift when OD weight 0->20: {level_shift:+.4f}/h; "
          f"identifiability: {out['identifiability'].value_counts().to_dict()}")
    print("[s12b] wrote augmented global_variant_growth_rates.csv + figures/global_growth_fit.png")


if __name__ == "__main__":
    main()
