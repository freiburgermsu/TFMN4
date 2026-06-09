"""
s01 - Load the barcode count table, restrict to the "first assessment" set
(Samples observed at all four transfers {3,4,6,13}), and emit tidy intermediate
files.

Outputs (outputs/intermediate/):
  barcode_4transfer_long.csv   - one row per (Sample, Transfer, Candidate) with
                                 Count, verA, verB, depth, freq.
  depth_by_sample_transfer.csv - sequencing depth (total reads) per cell; this is
                                 the multinomial exposure, NOT a population size.
  sample_well_map.csv          - Sample -> Microtiter_plate_well (+ construct,
                                 replicate); used to join the OD file in s02.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import config as C


def main():
    df = pd.read_csv(C.BARCODE_CSV)

    # Samples observed at all four transfers.
    n_transfers = df.groupby("Sample")["Transfer"].nunique()
    samples_4t = sorted(n_transfers[n_transfers == 4].index)
    d4 = df[df["Sample"].isin(samples_4t)].copy()
    # Guard: every kept Sample really has exactly {3,4,6,13}.
    bad = {s for s in samples_4t
           if set(d4.loc[d4.Sample == s, "Transfer"]) != set(C.FOUR_TRANSFER_SET)}
    if bad:
        raise SystemExit(f"Samples with 4 transfers but not == {C.FOUR_TRANSFER_SET}: {bad}")

    # Sequencing depth per (Sample, Transfer) = multinomial exposure.
    depth = (d4.groupby(["Sample", "Transfer"])["Count"].sum()
             .rename("depth").reset_index())
    d4 = d4.merge(depth, on=["Sample", "Transfer"], how="left")
    d4["freq"] = d4["Count"] / d4["depth"]

    long_cols = ["Sample", "DNA_construct", "Replicate", "Microtiter_plate_well",
                 "Transfer", "verA", "verB", "Candidate", "Count", "depth", "freq"]
    d4_long = d4[long_cols].sort_values(["Sample", "Transfer", "Candidate"])
    d4_long.to_csv(C.INTER / "barcode_4transfer_long.csv", index=False)

    depth_wide = depth.pivot(index="Sample", columns="Transfer", values="depth")
    depth_wide.to_csv(C.INTER / "depth_by_sample_transfer.csv")

    well_map = (d4[["Sample", "DNA_construct", "Replicate", "Microtiter_plate_well"]]
                .drop_duplicates().sort_values("Sample"))
    well_map.to_csv(C.INTER / "sample_well_map.csv", index=False)

    print(f"[s01] {len(samples_4t)} four-transfer Samples, {len(d4_long)} rows, "
          f"{d4_long['Candidate'].nunique()} distinct Candidates.")
    print(f"[s01] depth per cell: min={depth['depth'].min()} "
          f"median={int(depth['depth'].median())} max={depth['depth'].max()}")
    print(f"[s01] wrote barcode_4transfer_long.csv, depth_by_sample_transfer.csv, "
          f"sample_well_map.csv")
    return samples_4t


if __name__ == "__main__":
    main()
