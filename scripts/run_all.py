"""
run_all - Execute the full TFMN4 relative-fitness pipeline in order.

Usage:
    ~/Documents/py_venv/bin/python scripts/run_all.py
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
import s06_figures


def main():
    print("=" * 70)
    s01_prepare_data.main()
    print("-" * 70)
    s02_time_axis.main()
    print("-" * 70)
    s03_estimate_growth.main()
    print("-" * 70)
    s04_growth_matrix.main()
    print("-" * 70)
    s05_allele_effects.main()
    print("-" * 70)
    try:
        s06_figures.main()
    except Exception as e:  # figures are non-essential
        print(f"[s06] skipped figures ({type(e).__name__}: {e})")
    print("=" * 70)
    print("DONE. See outputs/ (growth_matrix_selection_per_cycle.csv is the headline).")


if __name__ == "__main__":
    main()
