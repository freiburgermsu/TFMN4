"""
s09b - Partial pooling of the cross-sample variant effects toward allele means.

The 117/168 variants seen in a single sample have noisy beta_i. We shrink each
variant's bridged beta_i toward an additive verA+verB allele prediction by
empirical-Bayes, so single-sample variants borrow strength from their alleles
while well-bridged variants keep their own estimate.

  allele model (ridge-regularized WLS):  beta_i ~ mu + a_{verA} + b_{verB}
  shrinkage:  beta_pooled = lambda*beta_raw + (1-lambda)*allele_pred,
              lambda = (1/se^2) / (1/se^2 + 1/tau^2)

(The full crossed-random-effects Bayesian version is the numpyro/pymc upgrade in
docs/METHODS.md; this EB approximation needs only installed libraries.)

Output (outputs/): variant_pooled.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
import config as C


def main():
    v = pd.read_csv(C.OUT / "variant_bridged_relative.csv")
    # use the RAW sampling SE (phi=1) for the hierarchical variance/shrinkage, so we
    # don't double-count the overdispersion already folded into beta_se.
    se_raw = np.clip(v["beta_se_raw"], 1e-3, None).values
    w = 1.0 / se_raw ** 2

    # additive allele design (one-hot verA + verB), ridge-regularized weighted fit
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = enc.fit_transform(v[["verA", "verB"]])
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(X, v["beta_per_cycle"].values, sample_weight=w)
    allele_pred = ridge.predict(X)

    # empirical-Bayes between-variant variance (method of moments), estimated on
    # the BRIDGED variants only -- the single-sample variants are too noisy and
    # would collapse tau^2 to ~0 and over-shrink everything.
    resid = v["beta_per_cycle"].values - allele_pred
    br = v["bridged"].values
    wb = w[br]
    tau2 = max(0.0, np.average(resid[br] ** 2, weights=wb)
               - np.average(se_raw[br] ** 2, weights=wb))
    tau2 = max(tau2, 1e-4)
    lam = (1.0 / se_raw ** 2) / (1.0 / se_raw ** 2 + 1.0 / tau2)
    pooled = lam * v["beta_per_cycle"].values + (1 - lam) * allele_pred

    out = v[["Candidate", "verA", "verB", "n_samples", "bridged",
             "beta_per_cycle", "beta_se", "beta_per_generation"]].copy()
    out["allele_pred_per_cycle"] = np.round(allele_pred, 4)
    out["beta_pooled_per_cycle"] = np.round(pooled, 4)
    out["shrinkage_lambda"] = np.round(lam, 3)
    out = out.sort_values("beta_pooled_per_cycle", ascending=False)
    out.to_csv(C.OUT / "variant_pooled.csv", index=False)

    print(f"[s09b] tau^2(between-variant)={tau2:.4f}; mean shrinkage lambda={lam.mean():.2f} "
          f"(single-sample variants shrink hardest)")
    print(f"[s09b] single-sample mean lambda={lam[v.n_samples==1].mean():.2f}  "
          f"vs multi-sample mean lambda={lam[v.n_samples>=2].mean():.2f}")
    print("[s09b] wrote variant_pooled.csv")
    return out


if __name__ == "__main__":
    main()
