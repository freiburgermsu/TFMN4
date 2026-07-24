"""
s01 - Load the count data and emit the canonical tidy intermediates the rest of
the pipeline consumes. Branches by dataset (config.DATASET):

  wgs       - SeqCenter WGS extracted_barcodes.csv; keep Samples observed at all
              of config.TRANSFERS {3,4,6,13}.
  amplicon* - Plasmidsaurus amplicons_reads.csv (per-read verA/verB); restrict to
              one carbon source (config.AMPLICON_CARBON); the three primer-set
              technical replicates become the cross-sample units; transfers {3,4,13}.

Outputs (into config.INTER, which is dataset-specific so nothing is overwritten):
  barcode_4transfer_long.csv   - (Sample, Transfer, Candidate) + verA/verB/Count/depth/freq
  depth_by_sample_transfer.csv - sequencing depth per cell (multinomial exposure)
  sample_well_map.csv          - Sample -> well (+ construct/condition, replicate)
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import config as C

LONG_COLS = ["Sample", "DNA_construct", "Replicate", "Microtiter_plate_well",
             "Transfer", "verA", "verB", "Candidate", "Count", "depth", "freq"]
CARBON_LABEL = {"M": "4MXB", "p": "pyruvate"}


def _finalize(d, label):
    """Shared: compute depth+freq, write the three intermediate files."""
    depth = (d.groupby(["Sample", "Transfer"])["Count"].sum().rename("depth").reset_index())
    d = d.merge(depth, on=["Sample", "Transfer"], how="left")
    d["freq"] = d["Count"] / d["depth"]
    d_long = d[LONG_COLS].sort_values(["Sample", "Transfer", "Candidate"])
    d_long.to_csv(C.INTER / "barcode_4transfer_long.csv", index=False)
    depth.pivot(index="Sample", columns="Transfer", values="depth").to_csv(
        C.INTER / "depth_by_sample_transfer.csv")
    (d[["Sample", "DNA_construct", "Replicate", "Microtiter_plate_well"]]
     .drop_duplicates().sort_values("Sample").to_csv(C.INTER / "sample_well_map.csv", index=False))
    print(f"[s01:{label}] {d['Sample'].nunique()} samples, {len(d_long)} rows, "
          f"{d_long['Candidate'].nunique()} distinct Candidates; "
          f"depth/cell min={int(depth.depth.min())} median={int(depth.depth.median())} "
          f"max={int(depth.depth.max())}")
    return sorted(d["Sample"].unique())


def main_wgs():
    df = pd.read_csv(C.BARCODE_CSV)
    nt = df.groupby("Sample")["Transfer"].nunique()
    keep = sorted(nt[nt == len(C.TRANSFERS)].index)
    d = df[df["Sample"].isin(keep)].copy()
    bad = {s for s in keep if set(d.loc[d.Sample == s, "Transfer"]) != set(C.TRANSFERS)}
    if bad:
        raise SystemExit(f"Samples with {len(C.TRANSFERS)} transfers but not == {C.TRANSFERS}: {bad}")
    return _finalize(d, "wgs")


def main_amplicon():
    ar = pd.read_csv(C.AMPLICON_READS)
    pat = re.compile(r"^([A-H]\d+)_(\d+)([A-Za-z]+)_([a-z]+)$")
    meta = ar["sample_name"].drop_duplicates().to_frame()
    parsed = meta["sample_name"].str.extract(pat)
    meta[["well", "transfer", "carbon", "primer"]] = parsed
    meta = meta.dropna(subset=["well"])
    ar = ar.merge(meta, on="sample_name")
    ar = ar[ar["carbon"] == C.AMPLICON_CARBON].copy()
    if ar.empty:
        raise SystemExit(f"No amplicon reads for carbon={C.AMPLICON_CARBON!r}")
    ar["Transfer"] = ar["transfer"].astype(int)
    ar["Candidate"] = ar["verA"].astype(str) + "-" + ar["verB"].astype(str)
    ar["Sample"] = ar["well"] + "_" + ar["carbon"] + "_" + ar["primer"]   # 3 primerset reps
    # aggregate reads -> counts per (Sample, Transfer, variant)
    d = (ar.groupby(["Sample", "well", "carbon", "primer", "Transfer", "verA", "verB", "Candidate"])
         .size().rename("Count").reset_index())
    # keep only the primerset-samples observed at every sampled transfer
    full = d.groupby("Sample")["Transfer"].apply(lambda s: set(s.unique()) >= set(C.TRANSFERS))
    d = d[d["Sample"].isin(full[full].index)].copy()
    d = d[d["Transfer"].isin(C.TRANSFERS)]
    d["DNA_construct"] = d["carbon"].map(CARBON_LABEL).fillna(d["carbon"])
    d["Replicate"] = d["primer"]
    d["Microtiter_plate_well"] = d["well"]
    print(f"[s01:{C.DATASET}] carbon={C.AMPLICON_CARBON} ({CARBON_LABEL.get(C.AMPLICON_CARBON)}); "
          f"primerset samples: {sorted(d['Sample'].unique())}")
    return _finalize(d, C.DATASET)


def main():
    return main_wgs() if C.DATASET == "wgs" else main_amplicon()


if __name__ == "__main__":
    main()
