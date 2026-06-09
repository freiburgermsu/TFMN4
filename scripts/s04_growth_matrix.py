"""
s04 - Pivot the long selection-coefficient table into the headline matrices:
variant (Candidate) rows x Sample columns, growth (selection coefficient) cells.

A cell is the variant's relative growth rate in that Sample. Cells are blank
where the variant is absent from that Sample, or NaN-flagged where present at
only one timepoint (not estimable). Per-transfer-cycle is the primary unit
(gauge-robust); a per-generation matrix is also written.

Outputs (outputs/):
  growth_matrix_selection_per_cycle.csv        <- PRIMARY DELIVERABLE
  growth_matrix_selection_per_generation.csv
  sample_summary.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C


def _matrix(res, value_col, only_estimable=True):
    df = res.copy()
    if only_estimable:
        df.loc[~df["estimable"], value_col] = np.nan
    m = df.pivot_table(index="Candidate", columns="Sample", values=value_col, aggfunc="first")
    # Short, readable Sample column names (drop the common prefix).
    m.columns = [c.replace("TFMN4.exp2.ACN3788.", "") for c in m.columns]
    # Order variants by how broadly informative they are, then by mean growth.
    m = m.assign(_n=m.notna().sum(axis=1), _mean=m.mean(axis=1, numeric_only=True))
    m = m.sort_values(["_n", "_mean"], ascending=[False, False]).drop(columns=["_n", "_mean"])
    m.index.name = "Candidate"
    return m


def main():
    res = pd.read_csv(C.OUT / "selection_coefficients_long.csv")

    m_cycle = _matrix(res, "s_per_cycle")
    m_cycle.to_csv(C.OUT / "growth_matrix_selection_per_cycle.csv")

    m_gen = _matrix(res, "s_per_generation")
    m_gen.to_csv(C.OUT / "growth_matrix_selection_per_generation.csv")

    # ---- per-Sample summary -------------------------------------------------
    d4 = pd.read_csv(C.INTER / "barcode_4transfer_long.csv")
    rich = (d4[d4.Count > 0].groupby(["Sample", "Transfer"])["Candidate"].nunique()
            .unstack("Transfer"))
    rich.columns = [f"n_variants_T{c}" for c in rich.columns]

    rows = []
    for sample, sub in res.groupby("Sample"):
        est = sub[sub.estimable]
        win = sub.loc[sub.sweep_winner]
        win_cand = win["Candidate"].iloc[0] if len(win) else None
        win_verA = win["verA"].iloc[0] if len(win) else None
        rows.append(dict(
            Sample=sample,
            DNA_construct=sub["DNA_construct"].iloc[0],
            Replicate=sub["Replicate"].iloc[0],
            n_candidates=sub["Candidate"].nunique(),
            n_estimable=int(est.shape[0]),
            sweep_winner=win_cand,
            winner_verA=win_verA,
            winner_freq_T13=float(win["freq_T13"].iloc[0]) if len(win) else np.nan,
            top_s_per_cycle=float(est["s_per_cycle"].max()) if len(est) else np.nan,
            bottom_s_per_cycle=float(est["s_per_cycle"].min()) if len(est) else np.nan,
        ))
    summ = pd.DataFrame(rows).merge(rich, on="Sample", how="left").sort_values("Sample")
    summ.to_csv(C.OUT / "sample_summary.csv", index=False)

    print(f"[s04] growth matrix: {m_cycle.shape[0]} variants x {m_cycle.shape[1]} samples "
          f"({int(m_cycle.notna().sum().sum())} filled cells).")
    print(f"[s04] verA winners across samples: "
          f"{summ['winner_verA'].value_counts().to_dict()}")
    print("[s04] wrote growth_matrix_selection_per_cycle.csv, "
          "growth_matrix_selection_per_generation.csv, sample_summary.csv")
    return m_cycle, summ


if __name__ == "__main__":
    main()
