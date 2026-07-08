"""
run_all - Execute the full TFMN4 relative-fitness pipeline in order.

Usage:
    ~/Documents/py_venv/bin/python scripts/run_all.py

Stages:
  s01 prepare data            s05 allele drift-gated effects   s09  cross-sample bridge
  s02 OD time axis            s06 per-sample figures           s09b partial pooling
  s03 per-sample growth       s07 growth-matrix figure         s10  graded allele table
  s04 growth matrix           s08 OD bulk rate (mu_bulk)       s11  falsification figures
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s01_prepare_data
import s02_time_axis
import s03_estimate_growth
import s04_growth_matrix
import s05_allele_effects
import s08_od_bulk_rate
import s09_bridge_meta
import s09b_partial_pooling
import s10_allele_effects_graded
import s12_global_growth_fit
import s12b_growth_rate_error
import s12c_growth_forest_all
import s14_segmented_growth
import s14b_segmented_summary
import s14c_segmented_figures
import s06_figures
import s07_growth_matrix_figure
import s11_falsification_figures


def main():
    line = "-" * 70
    print("=" * 70)
    # --- per-sample relative fitness ---
    s01_prepare_data.main(); print(line)
    s02_time_axis.main(); print(line)
    s03_estimate_growth.main(); print(line)
    s04_growth_matrix.main(); print(line)
    s05_allele_effects.main(); print(line)
    # --- cross-sample variant-level integration + OD anchor ---
    s08_od_bulk_rate.main(); print(line)
    s09_bridge_meta.main(); print(line)
    s09b_partial_pooling.main(); print(line)
    s10_allele_effects_graded.main(); print(line)
    # --- global static-rate optimization (assumption-laden absolute rates) + error ---
    s12_global_growth_fit.main(); print(line)
    s12b_growth_rate_error.main(); print(line)
    # --- complementary two-phase (breakpoint) per-variant regression (does NOT
    #     replace s03/s12; offered alongside them) + its summary rollups ---
    s14_segmented_growth.main(); print(line)
    s14b_segmented_summary.main(); print(line)
    # --- figures (non-essential) ---
    try:
        s06_figures.main()
        s07_growth_matrix_figure.main()
        s11_falsification_figures.main()
        s12c_growth_forest_all.main()
        s14_segmented_growth.make_figure()
        s14c_segmented_figures.main()
    except Exception as e:
        print(f"[figures] skipped ({type(e).__name__}: {e})")
    print("=" * 70)
    print("DONE. Headlines:")
    print("  outputs/growth_matrix_selection_per_cycle.csv  (per-sample, per-variant)")
    print("  outputs/variant_bridged_relative.csv           (cross-sample variant effects)")
    print("  outputs/allele_growth_advantage.csv            (graded allele table + OD-clock)")
    print("  outputs/global_variant_growth_rates.csv        (global static per-variant rate + error)")
    print("  outputs/segmented_growth_rates.csv             (complementary two-phase breakpoint fit)")
    print("  outputs/segmented_{model_selection,sample_summary,allele_effects}.csv  (segmented rollups)")
    print("  outputs/figures/segmented_{overview,parameters,gallery,allele}.png     (segmented figures)")
    print("  outputs/intermediate/od_bulk_condition_summary.csv  (the one absolute OD/h rate)")


if __name__ == "__main__":
    main()
