"""
cn01 - Ingest the dgoA copy-number data (copy-number-dgoA_.csv) into tidy per-lineage
trajectories for the growth-rate models.

Each row is the copy number of the dgoA* region for one lineage at one transfer; the
sample suffixes .P/.S1/.S2/.L1/.L2 are technical re-measurements of the SAME population,
so we also emit a per-transfer-mean table (the unit of replication used for every fit).
Lineage labels are parsed into a precise `genotype` and a coarse `context` (single-KO,
double-KO, exp1, control).

Outputs (outputs_copynumber/intermediate/):
  dgoA_copynumber_long.csv   - lineage, genotype, context, transfer, copy_number, ln_cn (all measurements)
  dgoA_bytransfer.csv        - lineage, transfer, ln_cn (mean), copy_number (mean), n_reps
  dgoA_lineage_index.csv     - per lineage: genotype, context, n_points, n_transfers,
                               has_replicates, estimable (>=3 transfers), breakpoint_testable/ci_reliable (>=4)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import cn_common as CN


def main():
    d = pd.read_csv(CN.CN_CSV)
    n_raw = len(d)
    stock = d[d.transfer.isna()].copy()
    d = d.dropna(subset=["transfer"]).copy()
    d["transfer"] = d["transfer"].astype(int)
    d = d[d["copy_number"] > 0].copy()
    d["genotype"] = d["lineage"].map(CN.genotype_of)
    d["context"] = d["lineage"].map(CN.context_of)
    d["ln_cn"] = np.log(d["copy_number"].to_numpy(float))

    long = d[["lineage", "genotype", "context", "transfer", "copy_number", "ln_cn"]] \
        .sort_values(["context", "lineage", "transfer"])
    long.to_csv(CN.INTER / "dgoA_copynumber_long.csv", index=False)

    bt = CN.by_transfer(long).merge(
        d[["lineage", "genotype", "context"]].drop_duplicates(), on="lineage", how="left")
    bt.to_csv(CN.INTER / "dgoA_bytransfer.csv", index=False)

    idx_rows = []
    for lin, g in d.groupby("lineage"):
        nt = g["transfer"].nunique()
        idx_rows.append(dict(
            lineage=lin, genotype=g["genotype"].iloc[0], context=g["context"].iloc[0],
            n_points=len(g), n_transfers=nt,
            transfers=",".join(str(t) for t in sorted(g["transfer"].unique())),
            has_replicates=bool((g.groupby("transfer").size() > 1).any()),
            estimable=bool(nt >= CN.MIN_EST_TRANSFERS),
            ci_reliable=bool(nt >= CN.CI_RELIABLE_TRANSFERS),
            breakpoint_testable=bool(nt >= CN.MIN_UNIQUE_TRANSFERS),
        ))
    idx = pd.DataFrame(idx_rows).sort_values(["context", "lineage"])
    idx.to_csv(CN.INTER / "dgoA_lineage_index.csv", index=False)

    stock_cn = float(stock["copy_number"].iloc[0]) if len(stock) else np.nan
    print(f"[cn01] {n_raw} rows -> {len(d)} measurements ({len(bt)} per-transfer means) across "
          f"{d['lineage'].nunique()} lineages (ancestral stock dgoA copy_number={stock_cn:.2f}).")
    print("[cn01] contexts: " + ", ".join(
        f"{c}={g['lineage'].nunique()}" for c, g in d.groupby("context")))
    print(f"[cn01] estimable (>=3 transfers): {int(idx.estimable.sum())}; "
          f"CI-reliable (>=4 transfers): {int(idx.ci_reliable.sum())}; "
          f"lineages with technical replicates: {int(idx.has_replicates.sum())}.")
    print("[cn01] wrote dgoA_copynumber_long.csv, dgoA_bytransfer.csv, dgoA_lineage_index.csv")
    return long, bt, idx


if __name__ == "__main__":
    main()
