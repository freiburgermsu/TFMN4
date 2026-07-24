"""
run - Execute the self-contained 4-parameter (segmented / two-phase breakpoint)
growth-rate analysis in THIS folder, end to end, writing everything to ./outputs/.

Everything is dataset-aware via scripts/config.py (this folder's DATASET is fixed there).
No environment variables and no files outside this folder are required.

Stages:
  s01  ingest counts             -> outputs/intermediate/barcode_4transfer_long.csv
  s02  OD -> time/generations axis-> outputs/intermediate/od_time_axis.csv
  s14  4-parameter segmented fit  -> outputs/segmented_growth_rates.csv
       (per variant: initial abundance, initial rate, final rate, breakpoint transfer;
        increasing-rate switches only, adopted only on strong evidence)
  s14b rollup tables             -> segmented_{model_selection,sample_summary,allele_effects}.csv
  s14d 1-var vs 4-var comparison -> model_comparison_{summary,by_trajectory}.csv (+figs)
  s14e 1-var vs 4-var fit error  -> model_fit_error{,_by_trajectory}.csv (+fig)
  figures                        -> outputs/figures/segmented_*.png

Run:  ~/Documents/py_venv/bin/python run.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import config as C
import s01_prepare_data
import s02_time_axis
import s14_segmented_growth
import s14b_segmented_summary
import s14c_segmented_figures
import s14d_model_comparison
import s14e_fit_error


def main():
    line = "-" * 72
    print("=" * 72)
    print(f"  4-PARAMETER SEGMENTED GROWTH MODEL   [dataset = {C.DATASET}]")
    print(f"  transfers sampled : {tuple(C.FOUR_TRANSFER_SET)}")
    print(f"  outputs -> {C.OUT}")
    print("=" * 72)
    # --- essential: tables ---
    s01_prepare_data.main(); print(line)
    s02_time_axis.main(); print(line)
    s14_segmented_growth.main(); print(line)
    s14b_segmented_summary.main(); print(line)
    s14d_model_comparison.main(); print(line)
    s14e_fit_error.main(); print(line)
    # --- non-essential: figures (never fail the run) ---
    try:
        s14_segmented_growth.make_figure()
        s14c_segmented_figures.main()
    except Exception as e:
        print(f"[figures] skipped ({type(e).__name__}: {e})")
    print("=" * 72)
    print("DONE.")


if __name__ == "__main__":
    main()
