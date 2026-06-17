"""
s13 - Joint WGS + amplicon optimization for per-variant 4MXB growth rates.

Both datasets are the SAME selective environment (WGS exp2 = "Methoxybenzoate
4 mM + Kan" = 4MXB; amplicon 4MXB). We fuse them into ONE fit so that WGS supplies
cross-well breadth (11 wells) and the deep amplicon supplies depth on well B4
(which is the SAME physical culture as WGS sample concX_largeLib_SpeI.1).

Model (extends s12's softmax-replicator):
  eta_{c,i} = logA_{well(c),i} + r_i * x_c + b_amp * x_c * 1[assay(c)=amplicon]
    r_i      : ONE global 4MXB growth rate per variant, shared across all cells   <- deliverable
    logA     : per (PHYSICAL well, variant) nuisance; WGS-B4 and amplicon-B4 share
               one logA_B4 block (this is what identifies the assay term)
    b_amp    : a single assay-DIFFERENTIAL rate drift (a constant offset would
               cancel in the per-cell softmax; only the x-drift is identifiable),
               strongly ridged so biology (shared slope) is preferred over batch
  Per-cell multinomial NLL at NATIVE depth -> the deep amplicon cell (~10k reads)
  tightens B4-present rates; OD anchor entered ONCE per physical well.
  Amplicon's 3 primer sets are collapsed to one cell/transfer (freq corr 0.995-0.998).

We fit the SAME model three ways (joint / WGS-only / amplicon-B4-only) so the
per-variant SEs are apples-to-apples, and report the error reduction.

Output: outputs_joint_4MXB/joint_variant_growth_rates_4MXB.csv
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
import config as C

LN2 = float(np.log(2.0))
ROOT = C.ROOT
OUT = ROOT / "outputs_joint_4MXB"
(OUT / "intermediate").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
W_OD, W_R, W_A, SIG_OD, W_B = 5.0, 50.0, 0.05, 0.08, 200.0   # W_B: strong ridge on assay drift


def load_combined():
    comb = pd.read_csv(OUT / "intermediate" / "combined_4MXB_long.csv")
    # collapse the 3 amplicon primer sets -> one amplicon-B4 cell per (transfer, variant)
    amp = comb[comb.assay == "amplicon"]
    amp = (amp.groupby(["Transfer", "verA", "verB", "Candidate"], as_index=False)["Count"].sum())
    amp["Sample"] = "AMP_B4"; amp["assay"] = "amplicon"; amp["Microtiter_plate_well"] = "B4"
    wgs = comb[comb.assay == "WGS"].copy()
    df = pd.concat([wgs[["Sample", "assay", "Microtiter_plate_well", "Transfer", "verA", "verB", "Candidate", "Count"]],
                    amp[["Sample", "assay", "Microtiter_plate_well", "Transfer", "verA", "verB", "Candidate", "Count"]]],
                   ignore_index=True)
    return df


def build(df, include_assays, use_b):
    """Build flat jax arrays for the cells in `include_assays`. Returns a dict."""
    d = df[df["assay"].isin(include_assays)].copy()
    # time axis: WGS-derived cumgen (covers 3,4,6,13), reconciled for B4 across assays
    ax = pd.read_csv(ROOT / "outputs/intermediate/od_time_axis.csv").set_index("transfer")["cumgen"]
    bulk = pd.read_csv(ROOT / "outputs/intermediate/od_bulk_rate.csv")
    mu_bar = float(bulk.loc[bulk.reliable & bulk.transfer.isin(list(C.RELIABLE_OD_TRANSFERS)),
                            "mu_bulk_per_h"].median())
    hrs_per_gen = LN2 / mu_bar
    xof = {int(t): float(ax[t]) * hrs_per_gen for t in ax.index}

    variants = sorted(d["Candidate"].unique()); vidx = {v: k for k, v in enumerate(variants)}
    wells = sorted(d["Microtiter_plate_well"].astype(str).unique())
    # per-well variant union -> logA blocks
    union = {w: sorted(g["Candidate"].unique()) for w, g in d.groupby(d["Microtiter_plate_well"].astype(str))}
    logA_index, logA_keys = {}, []
    for w in wells:
        for v in union[w]:
            logA_index[(w, v)] = len(logA_keys); logA_keys.append((w, v))

    # cells = (Sample, Transfer); represented over the well's variant union
    counts = d.pivot_table(index=["Microtiter_plate_well", "Sample", "assay", "Transfer", "Candidate"],
                           values="Count", aggfunc="sum")
    cells, rows = [], dict(var=[], logA=[], cell=[], cnt=[], x=[], amp=[])
    cell_meta = []
    for (w, samp, assay), g in d.groupby([d["Microtiter_plate_well"].astype(str), "Sample", "assay"]):
        cnt_by = g.groupby(["Transfer", "Candidate"])["Count"].sum()
        for t in sorted(g["Transfer"].unique()):
            cid = len(cell_meta)
            cell_meta.append(dict(well=w, sample=samp, assay=assay, transfer=int(t)))
            for v in union[w]:
                rows["var"].append(vidx[v]); rows["logA"].append(logA_index[(w, v)])
                rows["cell"].append(cid); rows["cnt"].append(float(cnt_by.get((t, v), 0.0)))
                rows["x"].append(xof[int(t)]); rows["amp"].append(1.0 if assay == "amplicon" else 0.0)
    cm = pd.DataFrame(cell_meta)
    # OD anchor: one representative cell per (physical well, reliable transfer); prefer WGS
    mub = {(r.Sample, int(r.transfer)): r.mu_bulk_per_h for r in bulk[bulk.reliable].itertuples()}
    # map physical well -> its WGS sample name (for mu_bulk lookup), B4 once
    well2wgs = (d[d.assay == "WGS"].drop_duplicates("Microtiter_plate_well")
                .set_index(d[d.assay == "WGS"].drop_duplicates("Microtiter_plate_well")["Microtiter_plate_well"].astype(str))["Sample"].to_dict())
    od_cell, od_mu = [], []
    seen = set()
    for cid, m in cm.iterrows():
        key = (m.well, m.transfer)
        if key in seen or m.transfer not in C.RELIABLE_OD_TRANSFERS:
            continue
        # prefer the WGS cell of this well/transfer for the anchor
        wgs_cells = cm[(cm.well == m.well) & (cm.transfer == m.transfer) & (cm.assay == "WGS")]
        use_cid = wgs_cells.index[0] if len(wgs_cells) else cid
        wgs_samp = well2wgs.get(m.well)
        mu = mub.get((wgs_samp, m.transfer)) if wgs_samp else mub.get((m.sample, m.transfer))
        if mu is None or not np.isfinite(mu):
            continue
        od_cell.append(int(use_cid)); od_mu.append(float(mu)); seen.add(key)

    # data-driven logA0
    freq = d.assign(well=d["Microtiter_plate_well"].astype(str))
    depth = freq.groupby(["Sample", "Transfer"])["Count"].transform("sum")
    freq["freq"] = freq["Count"] / depth
    mlf = freq.groupby(["well", "Candidate"])["freq"].mean()
    logA0 = np.array([np.log(max(mlf.get((w, v), 1e-3), 1e-3)) for (w, v) in logA_keys])

    return dict(
        variants=variants, n_var=len(variants), n_A=len(logA_keys), mu_bar=mu_bar,
        var=np.array(rows["var"]), logA=np.array(rows["logA"]), cell=np.array(rows["cell"]),
        cnt=np.array(rows["cnt"]), x=np.array(rows["x"]), amp=np.array(rows["amp"]),
        n_cells=len(cell_meta), logA0=logA0, use_b=use_b,
        od_cell=np.array(od_cell, int), od_mu=np.array(od_mu), cm=cm)


def fit(D):
    n_var, n_A, nc = D["n_var"], D["n_A"], D["n_cells"]
    var = jnp.asarray(D["var"]); logA_idx = jnp.asarray(D["logA"]); cell = jnp.asarray(D["cell"])
    cnt = jnp.asarray(D["cnt"]); x = jnp.asarray(D["x"]); amp = jnp.asarray(D["amp"])
    logA0 = jnp.asarray(D["logA0"]); mu_bar = D["mu_bar"]; use_b = D["use_b"]
    od_cell = jnp.asarray(D["od_cell"]); od_mu = jnp.asarray(D["od_mu"])
    npar = n_var + n_A + 1   # last param = b_amp

    def loss(p):
        r = p[:n_var]; logA = p[n_var:n_var + n_A]; b = p[-1] * (1.0 if use_b else 0.0)
        eta = logA[logA_idx] + r[var] * x + b * x * amp
        m = segment_max(eta, cell, num_segments=nc)
        ex = jnp.exp(eta - m[cell]); Z = segment_sum(ex, cell, num_segments=nc)
        logf = eta - m[cell] - jnp.log(Z[cell]); f = ex / Z[cell]
        L_count = -jnp.sum(cnt * logf)
        rbar = segment_sum(f * r[var], cell, num_segments=nc)
        L_od = W_OD * jnp.sum((rbar[od_cell] - od_mu) ** 2) / SIG_OD ** 2 if od_cell.size else 0.0
        L_r = W_R * jnp.sum((r - mu_bar) ** 2)
        L_a = W_A * jnp.sum((logA - logA0) ** 2)
        L_b = W_B * p[-1] ** 2
        return L_count + L_od + L_r + L_a + L_b

    vg = jax.jit(jax.value_and_grad(loss))
    def fn(p):
        v, g = vg(jnp.asarray(p)); return float(v), np.asarray(g, float)
    p = np.concatenate([np.full(n_var, mu_bar), D["logA0"], [0.0]])
    for _ in range(4):
        res = minimize(fn, p, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=8000, maxfun=80000, ftol=1e-13, gtol=1e-7, maxls=60))
        p = res.x
    # Laplace cov(r): Schur complement marginalizing logA (+ b if used)
    H = np.asarray(jax.hessian(loss)(jnp.asarray(p)), float)
    nu = slice(n_var, npar if use_b else npar - 1)
    Hrr = H[:n_var, :n_var]; Hrn = H[:n_var, nu]; Hnn = H[nu, nu]
    S = Hrr - Hrn @ np.linalg.solve(Hnn + 1e-8 * np.eye(Hnn.shape[0]), Hrn.T)
    cov = np.linalg.inv(S + 1e-8 * np.eye(n_var))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    r = p[:n_var]
    # fit quality
    fr2, od_rmse = fit_quality(D, p)
    return dict(r=r, se=se, b_amp=float(p[-1]) if use_b else 0.0, success=res.success,
                gnorm=float(np.linalg.norm(fn(p)[1])), freq_r2=fr2, od_rmse=od_rmse)


def fit_point(D, w_b=W_B):
    """Point-estimate-only fit (no Hessian) for fast LOSO / sensitivity refits."""
    n_var, n_A, nc = D["n_var"], D["n_A"], D["n_cells"]
    var = jnp.asarray(D["var"]); logA_idx = jnp.asarray(D["logA"]); cell = jnp.asarray(D["cell"])
    cnt = jnp.asarray(D["cnt"]); x = jnp.asarray(D["x"]); amp = jnp.asarray(D["amp"])
    logA0 = jnp.asarray(D["logA0"]); mu_bar = D["mu_bar"]; use_b = D["use_b"]
    od_cell = jnp.asarray(D["od_cell"]); od_mu = jnp.asarray(D["od_mu"])

    def loss(p):
        r = p[:n_var]; logA = p[n_var:n_var + n_A]; b = p[-1] * (1.0 if use_b else 0.0)
        eta = logA[logA_idx] + r[var] * x + b * x * amp
        m = segment_max(eta, cell, num_segments=nc)
        ex = jnp.exp(eta - m[cell]); Z = segment_sum(ex, cell, num_segments=nc)
        f = ex / Z[cell]; logf = eta - m[cell] - jnp.log(Z[cell])
        L = -jnp.sum(cnt * logf)
        rbar = segment_sum(f * r[var], cell, num_segments=nc)
        if od_cell.size:
            L = L + W_OD * jnp.sum((rbar[od_cell] - od_mu) ** 2) / SIG_OD ** 2
        return L + W_R * jnp.sum((r - mu_bar) ** 2) + W_A * jnp.sum((logA - logA0) ** 2) + w_b * p[-1] ** 2
    vg = jax.jit(jax.value_and_grad(loss))
    def fn(p):
        v, g = vg(jnp.asarray(p)); return float(v), np.asarray(g, float)
    p = np.concatenate([np.full(n_var, mu_bar), D["logA0"], [0.0]])
    for _ in range(3):
        p = minimize(fn, p, jac=True, method="L-BFGS-B",
                     options=dict(maxiter=6000, ftol=1e-12, gtol=1e-7, maxls=60)).x
    return dict(zip(D["variants"], p[:n_var])), float(p[-1])


def fit_quality(D, p):
    n_var, nc = D["n_var"], D["n_cells"]
    r = p[:n_var]; logA = p[n_var:n_var + D["n_A"]]; b = p[-1] if D["use_b"] else 0.0
    eta = logA[D["logA"]] + r[D["var"]] * D["x"] + b * D["x"] * D["amp"]
    cell = D["cell"]; cnt = D["cnt"]
    fmod = np.zeros_like(eta); rbar = np.zeros(nc)
    for cid in range(nc):
        msk = cell == cid; e = eta[msk] - eta[msk].max(); fm = np.exp(e) / np.exp(e).sum()
        fmod[msk] = fm; rbar[cid] = float((fm * r[D["var"][msk]]).sum())
    tot = np.zeros(nc); np.add.at(tot, cell, cnt)
    fobs = cnt / np.where(tot[cell] > 0, tot[cell], 1); mk = tot[cell] > 0
    fr2 = 1 - np.sum((fobs[mk] - fmod[mk]) ** 2) / np.sum((fobs[mk] - fobs[mk].mean()) ** 2)
    od_rmse = float(np.sqrt(np.mean((rbar[D["od_cell"]] - D["od_mu"]) ** 2))) if D["od_cell"].size else np.nan
    return float(fr2), od_rmse


def main():
    df = load_combined()
    Dj = build(df, ["WGS", "amplicon"], use_b=True)
    Dw = build(df, ["WGS"], use_b=False)
    Da = build(df, ["amplicon"], use_b=False)
    Db = build(df[df["Microtiter_plate_well"].astype(str) == "B4"], ["WGS"], use_b=False)  # WGS-B4 only
    print("[s13] fitting joint / WGS-only / amplicon-only / WGS-B4-only ...")
    Fj, Fw, Fa, Fb = fit(Dj), fit(Dw), fit(Da), fit(Db)
    print(f"[s13] joint: success={Fj['success']} gnorm={Fj['gnorm']:.2e} freq_R2={Fj['freq_r2']:.3f} "
          f"OD_rmse={Fj['od_rmse']:.3f} b_amp={Fj['b_amp']:+.4f}")
    print(f"[s13] WGS-only: freq_R2={Fw['freq_r2']:.3f}; amplicon-only: freq_R2={Fa['freq_r2']:.3f}")

    # assemble per-variant comparison (align on the joint variant union)
    def se_map(F, D): return dict(zip(D["variants"], F["se"]))
    def r_map(F, D): return dict(zip(D["variants"], F["r"]))
    se_w, se_a = se_map(Fw, Dw), se_map(Fa, Da)
    r_w, r_a = r_map(Fw, Dw), r_map(Fa, Da)
    meta = df.drop_duplicates("Candidate").set_index("Candidate")
    b4_amp = set(df[(df.assay == "amplicon")].Candidate)
    rows = []
    for i, v in enumerate(Dj["variants"]):
        sw, sa, sj = se_w.get(v, np.nan), se_a.get(v, np.nan), Fj["se"][i]
        best_single = np.nanmin([sw, sa]) if np.isfinite(np.nanmin([x for x in [sw, sa] if np.isfinite(x)] or [np.nan])) else np.nan
        sw_f = sw if np.isfinite(sw) else np.inf
        sa_f = sa if np.isfinite(sa) else np.inf
        best = min(sw_f, sa_f)
        red = best / sj if (np.isfinite(best) and sj > 0) else np.nan
        ess = (sw / sj) ** 2 if (np.isfinite(sw) and sj > 0) else np.nan
        in_w, in_a = np.isfinite(sw), np.isfinite(sa)
        tier = ("both-deep" if (in_w and in_a) else "amplicon-B4-only" if in_a else "WGS-only")
        rows.append(dict(
            Candidate=v, verA=meta.loc[v, "verA"], verB=meta.loc[v, "verB"],
            r_joint_per_h=round(Fj["r"][i], 4), se_joint=round(sj, 4),
            se_wgs_only=round(sw, 4) if np.isfinite(sw) else np.nan,
            se_ampl_only=round(sa, 4) if np.isfinite(sa) else np.nan,
            se_reduction_ratio=round(red, 3) if np.isfinite(red) else np.nan,
            ESS_fold_vs_wgs=round(ess, 2) if np.isfinite(ess) else np.nan,
            r_wgs_only=round(r_w.get(v, np.nan), 4) if v in r_w else np.nan,
            r_ampl_only=round(r_a.get(v, np.nan), 4) if v in r_a else np.nan,
            evidence_tier=tier, in_B4=v in b4_amp))
    res = pd.DataFrame(rows).sort_values("se_reduction_ratio", ascending=False, na_position="last")
    res.to_csv(OUT / "joint_variant_growth_rates_4MXB.csv", index=False)

    # B4 concordance GATE: same physical culture, WGS-B4-only vs amplicon-B4-only.
    rb4w = dict(zip(Db["variants"], Fb["r"])); seb4w = dict(zip(Db["variants"], Fb["se"]))
    rb4a = dict(zip(Da["variants"], Fa["r"]))
    dual = [v for v in Db["variants"] if v in rb4a and np.isfinite(seb4w.get(v, np.nan))]
    if len(dual) >= 3:
        xw = np.array([rb4w[v] for v in dual]); xa = np.array([rb4a[v] for v in dual])
        cc = np.corrcoef(xw, xa)[0, 1]; bias = float((xa - xw).mean())
        pd.DataFrame({"Candidate": dual, "r_WGS_B4": xw.round(4), "r_ampl_B4": xa.round(4)}) \
            .to_csv(OUT / "b4_concordance.csv", index=False)
        print(f"[s13] B4 concordance GATE (same culture): {len(dual)} dual-estimable-in-B4; "
              f"corr(r_WGS-B4, r_ampl-B4)={cc:.3f}, mean bias(ampl-WGS)={bias:+.3f}/h")
    tiers = res.evidence_tier.value_counts().to_dict()
    both = res[res.evidence_tier == "both-deep"]
    print(f"[s13] tiers: {tiers}")
    print(f"[s13] both-deep variants: median se_reduction_ratio={both.se_reduction_ratio.median():.2f}x, "
          f"median ESS_fold={both.ESS_fold_vs_wgs.median():.1f}x")
    print(f"[s13] wrote {OUT/'joint_variant_growth_rates_4MXB.csv'}")
    return res


if __name__ == "__main__":
    main()
