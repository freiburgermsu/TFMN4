"""
s01 (INTEGRATED) - Build the tidy count intermediate that FUSES the two 4MXB assays:

  * WGS breadth  - SeqCenter extracted_barcodes.csv: the 11 samples observed at all of
                   transfers {3,4,6,13} (one per well: B4,B5,C4,C9,D10,D4,D5,D9,E4,E9,F4).
  * Amplicon depth - Plasmidsaurus amplicons_reads.csv: deep re-sequencing of the SAME
                   physical culture in well B4 (carbon = 4MXB), transfers {3,4,13},
                   3 primer-set technical replicates.

FUSION.  Well B4 is the one culture measured by BOTH assays (WGS sample
concX_largeLib_SpeI.1 == amplicon well B4). We POOL reads there: for each
(Transfer, variant) on B4 we sum the WGS count and the (primer-set-collapsed) amplicon
count. This is the maximum-likelihood combination of two multinomial samples of the same
underlying composition. Consequences, both intended:
  - T3/T4/T13 on B4 gain the amplicon's ~10k-read depth on top of WGS's ~200x, so B4
    frequencies (and therefore the segmented fit's weights) are far more precise, and
    many rare B4 variants become estimable.
  - T6 is present ONLY in WGS (the amplicon never sampled transfer 6). That WGS-only
    point is exactly what lets B4 be BREAKPOINT-TESTABLE (>=4 observed transfers,
    >=2 per segment) — which the 3-transfer amplicon alone can never be.
The other 10 wells are WGS-only (no amplicon data), unchanged from the barcode-only run.

Outputs (config.INTER): barcode_4transfer_long.csv, depth_by_sample_transfer.csv,
sample_well_map.csv — identical schema to the barcode-only / amplicon-only s01, so the
downstream segmented code (s02/s14/...) consumes it without modification.
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
GROUP_KEYS = ["Sample", "DNA_construct", "Replicate", "Microtiter_plate_well",
              "Transfer", "verA", "verB", "Candidate"]


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


def _load_wgs():
    """The 11 WGS samples observed at exactly the four transfers {3,4,6,13}."""
    df = pd.read_csv(C.BARCODE_CSV)
    nt = df.groupby("Sample")["Transfer"].nunique()
    keep = sorted(nt[nt == len(C.TRANSFERS)].index)
    d = df[df["Sample"].isin(keep)].copy()
    bad = {s for s in keep if set(d.loc[d.Sample == s, "Transfer"]) != set(C.TRANSFERS)}
    if bad:
        raise SystemExit(f"WGS samples with {len(C.TRANSFERS)} transfers but not == {C.TRANSFERS}: {bad}")
    return d[["Sample", "DNA_construct", "Replicate", "Microtiter_plate_well",
              "Transfer", "verA", "verB", "Candidate", "Count"]].copy()


def _load_amplicon_b4(b4_sample, b4_construct, b4_replicate):
    """Amplicon reads for well B4 on this carbon (4MXB), primer sets collapsed to one
    count per (Transfer, variant), tagged onto the B4 physical culture."""
    ar = pd.read_csv(C.AMPLICON_READS)
    pat = re.compile(r"^([A-H]\d+)_(\d+)([A-Za-z]+)_([a-z]+)$")
    meta = ar["sample_name"].drop_duplicates().to_frame()
    parsed = meta["sample_name"].str.extract(pat)
    meta[["well", "transfer", "carbon", "primer"]] = parsed
    meta = meta.dropna(subset=["well"])
    ar = ar.merge(meta, on="sample_name")
    ar = ar[(ar["carbon"] == C.AMPLICON_CARBON) & (ar["well"] == "B4")].copy()
    if ar.empty:
        raise SystemExit(f"No amplicon reads for carbon={C.AMPLICON_CARBON!r} well B4")
    ar["Transfer"] = ar["transfer"].astype(int)
    ar["Candidate"] = ar["verA"].astype(str) + "-" + ar["verB"].astype(str)
    amp = (ar.groupby(["Transfer", "verA", "verB", "Candidate"]).size()
           .rename("Count").reset_index())            # collapse the 3 primer sets
    amp["Sample"] = b4_sample
    amp["DNA_construct"] = b4_construct
    amp["Replicate"] = b4_replicate
    amp["Microtiter_plate_well"] = "B4"
    print(f"[s01:integrated_4MXB] amplicon B4 (carbon={C.AMPLICON_CARBON}): "
          f"{len(amp)} (transfer,variant) cells over transfers {sorted(amp.Transfer.unique())}, "
          f"{amp.Candidate.nunique()} distinct variants, {int(amp.Count.sum())} reads.")
    return amp[["Sample", "DNA_construct", "Replicate", "Microtiter_plate_well",
                "Transfer", "verA", "verB", "Candidate", "Count"]]


def main_integrated():
    wgs = _load_wgs()
    # identify the single WGS sample physically in well B4 (the amplicon culture)
    b4 = wgs[wgs["Microtiter_plate_well"].astype(str) == "B4"].drop_duplicates("Sample")
    if b4["Sample"].nunique() != 1:
        raise SystemExit(f"Expected exactly one four-transfer WGS sample at well B4; "
                         f"found {sorted(b4['Sample'].unique())}")
    b4_sample = b4["Sample"].iloc[0]
    b4_construct = b4["DNA_construct"].iloc[0]
    b4_replicate = b4["Replicate"].iloc[0]
    print(f"[s01:integrated_4MXB] WGS wells: {sorted(wgs['Microtiter_plate_well'].astype(str).unique())}; "
          f"B4 culture = {b4_sample}")

    amp = _load_amplicon_b4(b4_sample, b4_construct, b4_replicate)

    # FUSE: pool WGS + amplicon reads per (B4 culture, Transfer, variant). For the other
    # 10 wells only WGS rows exist, so the groupby passes them through unchanged.
    combined = pd.concat([wgs, amp], ignore_index=True)
    combined = combined.groupby(GROUP_KEYS, as_index=False)["Count"].sum()

    # audit: report the B4 per-transfer depth gain from the fusion
    b4c = combined[combined.Microtiter_plate_well.astype(str) == "B4"]
    dep = b4c.groupby("Transfer")["Count"].sum().to_dict()
    print(f"[s01:integrated_4MXB] fused B4 depth per transfer (T6 is WGS-only): "
          f"{{{', '.join(f'T{int(t)}:{int(v)}' for t, v in sorted(dep.items()))}}}")
    return _finalize(combined, "integrated_4MXB")


def main():
    if C.DATASET != "integrated_4MXB":
        raise SystemExit(f"This folder's s01 builds the integrated dataset; "
                         f"config.DATASET={C.DATASET!r}. Set DATASET='integrated_4MXB'.")
    return main_integrated()


if __name__ == "__main__":
    main()
